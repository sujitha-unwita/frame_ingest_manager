"""Worker business-flow tests; run with the application's inference/DB packages."""

import unittest
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

from inference_pipeline.models import MappedDetection
from runtime.audit_worker import AuditWorker
from utils.timing import Timing


class AuditWorkerTests(unittest.IsolatedAsyncioTestCase):
    """Verify persisted IDs, complete camera rows and final-write failure behavior."""

    def setUp(self):
        # Exercise processing without constructing real clients.
        self.worker = AuditWorker.__new__(AuditWorker)
        self.worker.timing = Timing(False)
        self.worker.config = SimpleNamespace(
            encoder=SimpleNamespace(mm_per_count=0.5),
            layout=SimpleNamespace(mm_per_pixel_y=0.1),
        )
        self.worker.channels = Mock(report=AsyncMock())
        self.worker.writer = Mock(
            roll_id="roll", write_frame=AsyncMock(),
            write_final_defects=AsyncMock(),
        )
        self.worker.merger = Mock(process_trigger=Mock(return_value=[]))
        self.worker.completed = 0
        self.worker.read_frame = Mock(return_value=b"mock-file-content")
        self.job = {
            "sequence": 1, "roll_id": "roll", "admitted_at": time.monotonic(),
            "vision_finished_at": time.monotonic(),
            "trigger": {"trigger_id": 1, "encoder_count": 10},
            "frames": {
                "left": {"detections": []}, "right": {"detections": []},
            },
        }

    async def test_empty_rows_still_advance_merger(self):
        await self.worker.process(self.job)
        self.worker.merger.process_trigger.assert_called_once_with(
            1, 50.0, {"left": [], "right": []},
        )
        self.worker.writer.write_frame.assert_not_awaited()
        self.assertEqual(self.worker.completed, 1)

    async def test_audit_assigns_ids_before_merge_and_publishing(self):
        detection = MappedDetection(
            trigger_id=1, camera_id="left", frame_reference="frame.jpg",
            defect_class="hole", confidence=0.9,
            bounding_box_px={"x1": 1, "y1": 1, "x2": 5, "y2": 5},
            global_bounding_box_px={"x1": 1, "y1": 1, "x2": 5, "y2": 5},
            frame_width_px=100, frame_height_px=100,
            world_x1_mm=1, world_y1_mm=1, world_x2_mm=5, world_y2_mm=5,
        )
        self.job["frames"]["left"]["detections"] = [detection.model_dump()]
        persisted = detection.model_copy(update={"detection_id": UUID(int=1)})
        order = []

        async def write(**kwargs):
            order.append("commit")
            return SimpleNamespace(detections=[persisted], media_ref="s3://bucket/key")

        async def report(*args, **kwargs):
            order.append("publish")

        self.worker.writer.write_frame.side_effect = write
        self.worker.channels.report.side_effect = report
        await self.worker.process(self.job)
        self.assertEqual(order, ["commit", "publish"])
        row = self.worker.merger.process_trigger.call_args.args[2]
        self.assertEqual(row["left"][0].detection_id, UUID(int=1))
        self.assertEqual(row["right"], [])

    async def test_final_write_failure_prevents_completion(self):
        self.worker.writer.write_final_defects.side_effect = RuntimeError("DB failed")
        self.worker.merger.process_trigger.return_value = ["settled-group"]
        with self.assertRaisesRegex(RuntimeError, "DB failed"):
            await self.worker.process(self.job)
        self.assertEqual(self.worker.completed, 0)

    async def test_wrong_roll_and_order_rejected_before_writing(self):
        self.job["roll_id"] = "other"
        with self.assertRaisesRegex(RuntimeError, "different roll"):
            await self.worker.process(self.job)
        self.job["roll_id"] = "roll"
        self.job["sequence"] = 2
        with self.assertRaisesRegex(RuntimeError, "out-of-order"):
            await self.worker.process(self.job)
        self.worker.writer.write_frame.assert_not_awaited()
