"""Tests for configuration-driven camera and encoder coordinate mapping."""

from __future__ import annotations

import unittest

from config.models import InferenceLayout
from frame_ingest_manager.models import TriggerTelemetry
from inference_pipeline.engines.coordinate_mapper import CoordinateMapper
from inference_pipeline.models import BoundingBoxPx, Detection


class CoordinateMapperTests(unittest.TestCase):
    """Verify camera and trigger metadata determine the mapped result."""

    def test_maps_second_camera_with_configured_overlap(self) -> None:
        """The camera ordering and geometry come only from supplied settings."""
        mapper = CoordinateMapper(
            camera_ids=["camera_1", "camera_2"],
            layout_config=InferenceLayout(
                horizontal_overlap_ratio=0.2,
                vertical_overlap_ratio=0.1,
                mm_per_pixel_x=0.5,
                mm_per_pixel_y=0.5,
                origin="top_left",
                camera_position_indices={"camera_1": 0, "camera_2": 1},
                frame_width_px=100,
                left_stencil_mm=0,
                right_stencil_mm=0,
            ),
            encoder_mm_per_count=0.25,
        )
        detection = Detection(
            trigger_id=4,
            camera_id="camera_2",
            frame_reference="image.jpg",
            defect_class="small_hole",
            confidence=0.9,
            bounding_box_px=BoundingBoxPx(x1=10, y1=2, x2=20, y2=6),
            frame_width_px=100,
            frame_height_px=80,
        )

        mapped = mapper.map_detection(
            detection,
            TriggerTelemetry(trigger_id=4, encoder_count=100, timestamp_us=1),
        )

        self.assertEqual(mapped.global_bounding_box_px.x1, 90)
        self.assertEqual(mapped.global_bounding_box_px.y1, 52)
        self.assertEqual(mapped.world_x1_mm, 45)
        self.assertEqual(mapped.world_x2_mm, 50)
        self.assertEqual(mapped.world_y1_mm, 26)
        self.assertEqual(mapped.world_y2_mm, 28)
        self.assertEqual(mapped.global_bounding_box_px.y2, 56)

    def test_map_y_uses_current_count_and_configured_calibration(self) -> None:
        """Add travel once, including stationary and later trigger readings."""
        mapper = CoordinateMapper(
            ["camera_3"], self.make_layout(mm_per_pixel_y=0.1101), 0.25
        )
        for count in (0, 100, 100, 200):
            with self.subTest(encoder_count=count):
                y1, y2 = mapper._map_y(2, 6, count)
                self.assertAlmostEqual(y1, count * 0.25 + 2 * 0.1101)
                self.assertAlmostEqual(y2, count * 0.25 + 6 * 0.1101)

    def test_negative_encoder_count_is_rejected_by_trigger(self) -> None:
        """Telemetry validation rejects invalid counts before mapping."""
        with self.assertRaises(ValueError):
            TriggerTelemetry(trigger_id=1, encoder_count=-1, timestamp_us=1)

    def make_layout(self, **changes: object) -> InferenceLayout:
        """Use small dimensions so physical positions are easy to verify."""
        values = dict(
            horizontal_overlap_ratio=0.2,
            vertical_overlap_ratio=0.1,
            mm_per_pixel_x=0.5,
            mm_per_pixel_y=0.5,
            origin="top_left",
            camera_position_indices={"camera_3": 2, "camera_4": 3},
            frame_width_px=100,
            left_stencil_mm=10,
            right_stencil_mm=20,
        )
        values.update(changes)
        return InferenceLayout(**values)

    def map_box(self, mapper: CoordinateMapper, x1: float, x2: float):
        """Map one camera-3 observation with stationary encoder travel."""
        return mapper.map_detection(
            Detection(
                trigger_id=1,
                camera_id="camera_3",
                frame_reference="image.jpg",
                defect_class="small_hole",
                confidence=0.9,
                bounding_box_px=BoundingBoxPx(x1=x1, y1=2, x2=x2, y2=6),
                frame_width_px=100,
                frame_height_px=80,
            ),
            TriggerTelemetry(trigger_id=1, encoder_count=0, timestamp_us=1),
        )

    def test_physical_slots_are_not_rebased(self) -> None:
        """Selecting cameras 3/4 keeps their original bar positions."""
        mapper = CoordinateMapper(["camera_4", "camera_3"], self.make_layout(), 1)
        mapped = self.map_box(mapper, 30, 40)
        self.assertEqual(mapped.global_bounding_box_px.x1, 190)
        self.assertEqual((mapped.world_x1_mm, mapped.world_x2_mm), (5, 10))
        self.assertFalse(mapped.clipped_to_active_area)
        self.assertEqual(
            mapper.inspection_coverage(100),
            {"camera_4": (30, 60), "camera_3": (0, 40)},
        )

    def test_masked_and_touching_boxes_are_excluded(self) -> None:
        """A box ending at the active left edge contains no inspected area."""
        mapper = CoordinateMapper(["camera_3"], self.make_layout(), 1)
        self.assertIsNone(self.map_box(mapper, 0, 10))
        self.assertIsNone(self.map_box(mapper, 0, 20))

    def test_boundary_box_is_clipped_without_losing_raw_box(self) -> None:
        """Only the mapped box is clipped; raw image evidence is preserved."""
        mapper = CoordinateMapper(["camera_3"], self.make_layout(), 1)
        mapped = self.map_box(mapper, 10, 30)
        self.assertEqual((mapped.world_x1_mm, mapped.world_x2_mm), (0, 5))
        self.assertEqual(mapped.bounding_box_px.x1, 10)
        self.assertTrue(mapped.clipped_to_active_area)

    def test_right_boundary_and_uncovered_camera(self) -> None:
        """Fabric narrower than camera coverage is clipped at its right mask."""
        layout = self.make_layout()
        mapper = CoordinateMapper(["camera_3"], layout, 1)
        mapped = self.map_box(mapper, 50, 80)
        self.assertEqual((mapped.world_x1_mm, mapped.world_x2_mm), (15, 20))
        self.assertTrue(mapped.clipped_to_active_area)
        self.assertIsNone(self.map_box(mapper, 60, 80))
        self.assertEqual(mapper.inspection_coverage(100), {"camera_3": (0, 20)})

    def test_right_origin_uses_active_edge(self) -> None:
        """Right-origin mapping refers to the right stencil, not camera count."""
        mapper = CoordinateMapper(
            ["camera_3"], self.make_layout(origin="top_right"), 1
        )
        mapped = self.map_box(mapper, 30, 40)
        self.assertEqual((mapped.world_x1_mm, mapped.world_x2_mm), (10, 15))

    def test_invalid_inspection_layout_is_rejected(self) -> None:
        """Slots must be physical and unique; masks must leave active fabric."""
        for changes in (
            {"camera_position_indices": {"camera_3": -1}},
            {"camera_position_indices": {"camera_3": 2, "camera_4": 2}},
            {"frame_width_px": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.make_layout(**changes)

    def test_mv1_camera_3_and_4_outer_stencils(self):
        layout = self.make_layout(
            frame_width_px=4096, mm_per_pixel_x=0.113,
            horizontal_overlap_ratio=0.10, left_stencil_mm=50,
            right_stencil_mm=50,
        )
        mapper = CoordinateMapper(["camera_3", "camera_4"], layout, 1)
        self.assertAlmostEqual(mapper.active_width_mm, 779.4112)
        coverage = mapper.inspection_coverage(4096)
        self.assertAlmostEqual(coverage["camera_3"][0], 0)
        self.assertAlmostEqual(coverage["camera_3"][1], 412.848)
        self.assertAlmostEqual(coverage["camera_4"][0], 366.5632)
        self.assertAlmostEqual(coverage["camera_4"][1], 779.4112)

    def test_invalid_active_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stencils"):
            CoordinateMapper(["camera_3"], self.make_layout(left_stencil_mm=30), 1)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            CoordinateMapper(
                ["camera_3", "camera_4"],
                self.make_layout(camera_position_indices={"camera_3": 2, "camera_4": 4}),
                1,
            )

    def test_mismatched_frame_width_is_rejected(self):
        mapper = CoordinateMapper(
            ["camera_3"], self.make_layout(frame_width_px=200), 1
        )
        with self.assertRaisesRegex(ValueError, "frame width"):
            self.map_box(mapper, 30, 40)


if __name__ == "__main__":
    unittest.main()
