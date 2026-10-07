"""Poster-policy boundaries, special classes, and settled-score ownership."""

import unittest
from uuid import UUID

from pydantic import ValidationError

from config.config_loader import ConfigLoader
from config.models import FourPointGrading
from inference_pipeline.engines.four_point_grading_engine import (
    FourPointGradingEngine,
)
from inference_pipeline.models.defect_group import DefectGroup


class FourPointGradingTests(unittest.TestCase):
    """Use the actual YAML policy without loading model weights or a database."""

    def setUp(self):
        raw = ConfigLoader().raw_config["inference"]
        self.config = FourPointGrading.model_validate(raw["grading"])
        self.engine = FourPointGradingEngine(self.config, active_width_mm=900)

    @staticmethod
    def group(width_mm, height_mm=1, defect_class="stain", x1=10):
        """Create a settled rectangle; pixel geometry is irrelevant to grading."""
        return DefectGroup(
            defect_class=defect_class, confidence=0.9,
            global_bounding_box_px={"x1": 0, "y1": 0, "x2": 1, "y2": 1},
            member_detection_ids=(UUID(int=1),),
            world_x1_mm=x1, world_x2_mm=x1 + width_mm,
            world_y1_mm=0, world_y2_mm=height_mm,
        )

    def test_inclusive_length_boundaries_and_oversize(self):
        for inches, expected in (
            (0.1, 1), (3, 1), (3.01, 2), (6, 2),
            (6.01, 4), (9, 4), (9.01, 4),
        ):
            with self.subTest(inches=inches):
                group = self.group(inches * self.config.mm_per_inch, x1=0)
                self.assertEqual(self.engine.grade(group).four_point_score, expected)
                self.assertIsNone(group.four_point_score)

    def test_longest_dimension_not_width_alone(self):
        result = self.engine.grade(self.group(1, 7 * self.config.mm_per_inch))
        self.assertEqual(result.four_point_score, 4)

    def test_scores_are_read_from_config_not_fixed_in_engine(self):
        config = FourPointGrading.model_validate({
            **self.config.model_dump(), "length_points": [1, 2, 3],
            "hole_points": 2,
        })
        engine = FourPointGradingEngine(config, active_width_mm=900)
        self.assertEqual(engine.grade(self.group(7 * 25.4)).four_point_score, 3)
        self.assertEqual(
            engine.grade(self.group(1, defect_class="hole")).four_point_score, 2
        )

    def test_all_configured_holes_score_four_regardless_of_size(self):
        for label in self.config.hole_classes:
            for size in (1, 500):
                with self.subTest(label=label, size=size):
                    result = self.engine.grade(self.group(size, defect_class=label.upper()))
                    self.assertEqual(result.four_point_score, 4)

    def test_full_active_width_rule_uses_fabric_not_camera_edges(self):
        # A narrow fabric makes the special rule distinguishable from length scoring.
        engine = FourPointGradingEngine(self.config, active_width_mm=50)
        self.assertEqual(engine.grade(self.group(50, x1=0)).four_point_score, 4)
        self.assertEqual(engine.grade(self.group(49, x1=0)).four_point_score, 1)

    def test_invalid_geometry_is_rejected(self):
        for width in (0, -1, float("nan"), float("inf")):
            with self.subTest(width=width), self.assertLogs(
                "inference_pipeline.engines.four_point_grading_engine", level="ERROR"
            ), self.assertRaises(ValueError):
                self.engine.grade(self.group(width))

    def test_invalid_yaml_bands_are_rejected(self):
        for updates in (
            {"length_limits_inches": [6, 3, 9]},
            {"length_limits_inches": [3, 3, 9]},
            {"length_limits_inches": [3, 6, float("inf")]},
            {"length_points": [1, 2]},
            {"length_points": [1, 2, 5]},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                FourPointGrading.model_validate({**self.config.model_dump(), **updates})
