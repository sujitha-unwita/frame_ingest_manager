"""Live event contract tests without GPU, database, or storage dependencies."""

import unittest

from dashboard_service.services.event_factory import defect_event
from audit_write.utils.identifiers import Identifiers


class LiveDefectEventTests(unittest.TestCase):
    """Whole-frame references and rectangles are sent without internal UUIDs."""

    def test_live_event_contains_both_rectangles_and_optional_metadata(self):
        event = defect_event(
            trigger_id=42, camera_id="camera_3", tile_ids=None,
            frame_variant=None, defect_class="hole", confidence=0.9,
            world_x1_mm=1, world_y1_mm=2, world_x2_mm=3, world_y2_mm=4,
            bbox_x1_px=10, bbox_y1_px=20, bbox_x2_px=30, bbox_y2_px=40,
            crop_thumbnail_ref="frames/whole-frame.jpg",
        )
        self.assertEqual(event["trigger_id"], 42)
        self.assertEqual(event["camera_id"], "camera_3")
        self.assertIsNone(event["tile_ids"])
        self.assertIsNone(event["frame_variant"])
        self.assertEqual(event["world_x2_mm"], 3)
        self.assertEqual(event["bbox_y2_px"], 40)
        self.assertEqual(event["crop_thumbnail_ref"], "frames/whole-frame.jpg")
        self.assertNotIn("detection_id", event)
        self.assertNotIn("defect_no", event)

    def test_uuid_utility_generates_distinct_version_four_ids(self):
        first, second = Identifiers.new_id(), Identifiers.new_id()
        self.assertEqual(first.version, 4)
        self.assertNotEqual(first, second)
