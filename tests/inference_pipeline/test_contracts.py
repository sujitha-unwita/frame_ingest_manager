"""Tests for Pydantic contracts at the inference boundary."""

from __future__ import annotations

import unittest

from inference_pipeline.models import BoundingBoxPx, Detection
from pydantic import ValidationError


class InferenceContractTests(unittest.TestCase):
    """Prove invalid model-engine output stops at the boundary."""

    def test_detection_accepts_a_valid_pixel_box(self) -> None:
        """A raw engine result becomes one typed detection."""
        detection = Detection(
            trigger_id=1,
            camera_id="camera_1",
            frame_reference="image.jpg",
            defect_class="small_hole",
            confidence=0.9,
            bounding_box_px=BoundingBoxPx(x1=1, y1=2, x2=3, y2=4),
            frame_width_px=10,
            frame_height_px=10,
        )

        self.assertEqual(detection.bounding_box_px.x2, 3)

    def test_reversed_box_is_rejected(self) -> None:
        """A detector cannot send a zero-size or reversed rectangle onward."""
        with self.assertRaises(ValidationError):
            BoundingBoxPx(x1=5, y1=2, x2=5, y2=4)


if __name__ == "__main__":
    unittest.main()
