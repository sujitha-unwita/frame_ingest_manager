"""Behaviour tests using the real SAHI NMM implementation; no model required."""

import unittest
from unittest.mock import patch
from uuid import UUID

from config.config_loader import ConfigLoader
from config.models import Deduplication, FourPointGrading
from inference_pipeline.engines.deduplication_engine import DeduplicationEngine
from inference_pipeline.engines.four_point_grading_engine import (
    FourPointGradingEngine,
)
from inference_pipeline.models.contracts import BoundingBoxPx, MappedDetection


class DeduplicationEngineTests(unittest.TestCase):
    """Check grouping, provenance, settlement, and invalid-row atomicity."""

    def setUp(self) -> None:
        """Use small pixel distances so boundary expectations are explicit."""
        self.config = Deduplication(
            ios_threshold=0.5,
            match_metric="IOS",
            horizontal_gap_px=2,
            vertical_gap_px=2,
            min_axis_overlap_ratio=0.5,
        )
        raw = ConfigLoader().raw_config["inference"]
        self.grader = FourPointGradingEngine(
            FourPointGrading.model_validate(raw["grading"]), active_width_mm=900
        )
        self.engine = DeduplicationEngine(
            self.config, ["left", "right"], self.grader
        )

    @staticmethod
    def detection(number, trigger, camera, box, defect_class="hole"):
        """Build a persisted mapped observation; tests use one mm per pixel."""
        x1, y1, x2, y2 = box
        return MappedDetection(
            detection_id=UUID(int=number),
            trigger_id=trigger,
            camera_id=camera,
            frame_reference=f"{number}.jpg",
            defect_class=defect_class,
            confidence=0.9,
            bounding_box_px=BoundingBoxPx(x1=0, y1=0, x2=x2-x1, y2=y2-y1),
            frame_width_px=100,
            frame_height_px=100,
            global_bounding_box_px=BoundingBoxPx(x1=x1, y1=y1, x2=x2, y2=y2),
            world_x1_mm=x1, world_y1_mm=y1,
            world_x2_mm=x2, world_y2_mm=y2,
        )

    def test_same_trigger_overlap_merges_with_both_member_ids(self):
        """Overlapping cameras yield one physical group with both raw UUIDs."""
        left = self.detection(1, 1, "left", (0, 0, 10, 10))
        right = self.detection(2, 1, "right", (5, 0, 15, 10))
        self.assertEqual(self.engine.process_trigger(
            1, 0, {"left": [left], "right": [right]}), [])
        settled = self.engine.process_trigger(2, 20, {"left": [], "right": []})
        self.assertEqual(len(settled), 1)
        self.assertEqual(set(settled[0].member_detection_ids),
                         {UUID(int=1), UUID(int=2)})
        self.assertEqual(settled[0].global_bounding_box_px.x2, 15)
        self.assertEqual(settled[0].four_point_score, 4)
        self.assertNotIn("defect_no", settled[0].model_dump())

    def test_cross_trigger_group_grows_until_frontier_passes(self):
        """Later trigger observations extend an open physical defect."""
        first = self.detection(1, 1, "left", (0, 5, 10, 15))
        later = self.detection(2, 2, "left", (0, 10, 10, 20))
        self.engine.process_trigger(1, 0, {"left": [first], "right": []})
        self.assertEqual(self.engine.process_trigger(
            2, 10, {"left": [later], "right": []}), [])
        self.assertEqual(self.engine.open_count, 1)
        # Touching the frontier is still open (20 == 22 - 2).
        self.assertEqual(self.engine.process_trigger(
            3, 22, {"left": [], "right": []}), [])
        settled = self.engine.process_trigger(4, 23, {"left": [], "right": []})
        self.assertEqual(len(settled), 1)
        self.assertEqual(len(settled[0].member_detection_ids), 2)
        self.assertEqual(self.engine.finish(), [])

    def test_classes_and_distant_boxes_stay_separate(self):
        """Neither a different class nor an unrelated region is merged."""
        boxes = [self.detection(1, 1, "left", (0, 0, 10, 10)),
                 self.detection(2, 1, "left", (0, 0, 10, 10), "streak"),
                 self.detection(3, 1, "left", (50, 0, 60, 10))]
        self.engine.process_trigger(1, 0, {"left": boxes, "right": []})
        self.assertEqual(len(self.engine.finish()), 3)

    def test_horizontal_gap_uses_configured_budget(self):
        """A two-pixel gap merges only with a sufficient YAML gap budget."""
        row = {"left": [self.detection(1, 1, "left", (0, 0, 10, 10))],
               "right": [self.detection(2, 1, "right", (12, 0, 22, 10))]}
        self.engine.process_trigger(1, 0, row)
        self.assertEqual(len(self.engine.finish()), 1)
        self.config.horizontal_gap_px = 1
        other = DeduplicationEngine(self.config, ["left", "right"], self.grader)
        other.process_trigger(1, 0, row)
        self.assertEqual(len(other.finish()), 2)

    def test_missing_camera_does_not_consume_trigger(self):
        """A corrected complete row can retry after validation fails."""
        with self.assertRaises(ValueError):
            self.engine.process_trigger(1, 0, {"left": []})
        self.assertEqual(self.engine.process_trigger(
            1, 0, {"left": [], "right": []}), [])

    def test_duplicate_trigger_and_backward_position_are_rejected(self):
        """Invalid order must not settle or overwrite open groups."""
        box = self.detection(1, 1, "left", (0, 10, 10, 30))
        self.engine.process_trigger(1, 10, {"left": [box], "right": []})
        for trigger, top in ((1, 10), (2, 9)):
            with self.assertRaises(ValueError):
                self.engine.process_trigger(trigger, top, {"left": [], "right": []})
        self.assertEqual(self.engine.open_count, 1)

    def test_unpersisted_detection_is_rejected(self):
        """Audit must assign raw UUIDs before they enter the merger."""
        box = self.detection(1, 1, "left", (0, 0, 10, 10))
        box.detection_id = None
        with self.assertRaises(ValueError):
            self.engine.process_trigger(1, 0, {"left": [box], "right": []})
        self.assertEqual(self.engine.open_count, 0)

    def test_finish_releases_once_and_closes_the_roll(self):
        """End of roll flushes groups without assigning final numbers."""
        box = self.detection(1, 1, "left", (0, 0, 10, 10))
        self.engine.process_trigger(1, 0, {"left": [box], "right": []})
        self.assertEqual(len(self.engine.finish()), 1)
        self.assertEqual(self.engine.finish(), [])
        with self.assertRaises(ValueError):
            self.engine.process_trigger(2, 20, {"left": [], "right": []})

    def test_grading_failure_preserves_open_groups_and_trigger(self):
        """A failed grade must not consume settlement or prevent its retry."""
        box = self.detection(1, 1, "left", (0, 0, 10, 10))
        self.engine.process_trigger(1, 0, {"left": [box], "right": []})
        with patch.object(self.grader, "grade", side_effect=ValueError("grade")):
            with self.assertRaisesRegex(ValueError, "grade"):
                self.engine.process_trigger(2, 20, {"left": [], "right": []})
            with self.assertRaisesRegex(ValueError, "grade"):
                self.engine.finish()
        self.assertEqual(self.engine.open_count, 1)
        settled = self.engine.process_trigger(2, 20, {"left": [], "right": []})
        self.assertEqual(settled[0].four_point_score, 4)

    def test_grade_uses_merged_size_not_individual_observations(self):
        """Two one-point observations can form one longer two-point defect."""
        first = self.detection(1, 1, "left", (10, 0, 20, 60), "stain")
        later = self.detection(2, 2, "left", (10, 50, 20, 110), "stain")
        self.engine.process_trigger(1, 0, {"left": [first], "right": []})
        self.engine.process_trigger(2, 50, {"left": [later], "right": []})
        settled = self.engine.process_trigger(3, 120, {"left": [], "right": []})
        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0].four_point_score, 2)
        self.assertEqual(len(settled[0].member_detection_ids), 2)
