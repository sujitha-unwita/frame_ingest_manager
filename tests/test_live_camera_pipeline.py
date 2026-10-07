"""Real ingestion/tiling/mapping/encoding flow with hardware and model I/O replaced."""
import asyncio
import pickle
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import cv2
import numpy as np
import torch
from config.config_loader import load_config
from inference_pipeline.engines.coordinate_mapper import CoordinateMapper
from inference_pipeline.engines.sahi_detection_engine import SahiDetectionEngine
from inference_pipeline.engines.tile_generator import TileGenerator
from runtime.vision_worker import VisionWorker
from runtime.audit_worker import AuditWorker
from utils.timing import Timing
from tests.frame_ingest_manager.test_mvs_ingest import make_source, telemetry


class LiveCameraPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        model = Path(folder.name) / "model.pt"
        model.touch()
        self.config = load_config(overrides={
            "audit_write": {"enabled": False},
            "inference": {"model_path": str(model), "device_preference": "cpu",
                          "tile_size_px": 4, "tile_batch_size": 1, "input_channels": 1,
                          "layout": {"frame_width_px": 4, "left_stencil_mm": 0,
                                     "right_stencil_mm": 0, "camera_position_indices": {
                                         "left": 6, "right": 7}}}})
        self.source, self.manager, self.sdk = make_source()
        self.addCleanup(self.manager.stop)
        self.cameras = self.source.cameras[:]
        detector = SahiDetectionEngine(self.config.inference, TileGenerator(self.config.inference))
        detector._model = Mock()
        def predict(model, tensor):
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=torch.tensor([[0., 0., 2., 2.]]), conf=torch.tensor([0.9]),
                cls=torch.tensor([0.])), names={0: "hole"})]
        detector._predict = Mock(side_effect=predict)
        self.worker = VisionWorker.__new__(VisionWorker)
        self.worker.config = SimpleNamespace(frame_encoding="png")
        self.worker.timing = Timing(False)
        self.worker.frames = self.manager
        self.worker.detector = detector
        self.worker.mapper = CoordinateMapper(["left", "right"], self.config.inference.layout,
                                               self.config.encoder.mm_per_count)
        self.worker.gpu = Mock(end_inference=Mock(return_value=25))
        self.worker.channels = Mock(report=AsyncMock())
        self.job = {"trigger": telemetry(100).model_dump(), "sequence": 1,
                    "roll_id": "roll", "admitted_at": time.monotonic()}

    def emit(self):
        self.cameras[0].emit(2, 17)
        self.cameras[1].emit(5, 29)

    async def test_images_reach_real_inference_mapping_and_audit_handoff(self):
        self.emit()
        with patch("cv2.imread", side_effect=AssertionError("live capture must not read files")):
            result = await self.worker.process(self.job)
        # IPC serialization owns immutable bytes independent of returned camera slots.
        result = pickle.loads(pickle.dumps(result))
        for camera, value in zip(self.cameras, [17, 29]):
            self.assertEqual(camera.pool.free_count(), 3)
            for slot in range(3):
                camera.pool.raw(slot)[:] = 0
            frame = result["frames"][camera.camera_id]
            content = AuditWorker.read_frame(frame)
            restored = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(np.all(restored == value))
            self.assertEqual(frame["detections"][0]["trigger_id"], 100)
            self.assertGreaterEqual(frame["detections"][0]["world_y1_mm"], 0)
            self.assertEqual(frame["camera_metadata"]["capture_timestamp_us"], 100000)
        self.assertNotIn("reference", result["frames"]["left"])
        self.assertEqual(self.worker.detector._predict.call_count, 2)

    async def test_inference_failure_releases_entire_bundle(self):
        self.emit()
        self.worker.detector.detect = AsyncMock(side_effect=RuntimeError("model failure"))
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            await self.worker.process(self.job)
        self.assertTrue(all(camera.pool.free_count() == 3 for camera in self.cameras))

    async def test_inference_cancellation_releases_entire_bundle(self):
        self.emit()
        entered = asyncio.Event()
        async def wait_forever(frame):
            entered.set()
            await asyncio.Event().wait()
        self.worker.detector.detect = wait_forever
        task = asyncio.create_task(self.worker.process(self.job))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(all(camera.pool.free_count() == 3 for camera in self.cameras))

    async def test_cancelled_match_wait_releases_frames_arriving_after_cancel(self):
        task = asyncio.create_task(self.worker.process(self.job))
        await asyncio.sleep(0.01)
        task.cancel()
        self.emit()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(all(camera.pool.free_count() == 3 for camera in self.cameras))

    async def test_empty_detection_rows_still_reach_audit_without_image_payload(self):
        self.emit()
        self.worker.detector.detect = AsyncMock(return_value=[])
        result = await self.worker.process(self.job)
        self.assertEqual(set(result["frames"]), {"left", "right"})
        for frame in result["frames"].values():
            self.assertEqual(frame["detections"], [])
            self.assertNotIn("encoded_bytes", frame)
        self.assertTrue(all(camera.pool.free_count() == 3 for camera in self.cameras))

    def test_audit_rejects_empty_or_mutable_encoded_buffers(self):
        for value in (b"", bytearray(b"test"), "test"):
            with self.assertRaises(ValueError):
                AuditWorker.read_frame({"encoded_bytes": value})

    async def test_live_rows_continue_through_raw_audit_merging_and_final_write(self):
        from uuid import UUID
        from audit_write.services.audit_writer import FrameAuditResult
        from inference_pipeline.models import MappedDetection
        from inference_pipeline.engines.deduplication_engine import DeduplicationEngine
        from inference_pipeline.engines.four_point_grading_engine import FourPointGradingEngine
        audit = AuditWorker.__new__(AuditWorker)
        audit.timing = Timing(False)
        audit.config = SimpleNamespace(encoder=self.config.encoder, layout=self.config.inference.layout)
        audit.channels = Mock(report=AsyncMock())
        audit.writer = Mock(roll_id="roll", write_frame=AsyncMock(), write_final_defects=AsyncMock())
        audit.merger = DeduplicationEngine(self.config.inference.deduplication, ["left", "right"],
            FourPointGradingEngine(self.config.inference.grading, self.worker.mapper.active_width_mm))
        audit.completed = 0
        next_id = iter(range(1, 100))
        async def commit_frame(**kwargs):
            self.assertTrue(kwargs["frame_bytes"].startswith(b"\x89PNG"))
            return FrameAuditResult(media_ref="test.png", detections=tuple(
                item.model_copy(update={"detection_id": UUID(int=next(next_id))})
                for item in kwargs["detections"]))
        audit.writer.write_frame.side_effect = commit_frame
        self.emit()
        await audit.process(await self.worker.process(self.job))
        self.assertEqual(audit.writer.write_frame.await_count, 2)
        self.assertEqual(audit.completed, 1)
        self.assertGreater(audit.merger.open_count, 0)
        # An empty later row still advances the real merger and settles old groups.
        self.cameras[0].emit(3)
        self.cameras[1].emit(6)
        self.worker.detector.detect = AsyncMock(return_value=[])
        job = {**self.job, "sequence": 2,
               "trigger": {"trigger_id": 101, "encoder_count": 100000, "timestamp_us": 101000}}
        await audit.process(await self.worker.process(job))
        self.assertEqual(audit.completed, 2)
        audit.writer.write_final_defects.assert_awaited_once()
        groups = audit.writer.write_final_defects.call_args.args[0]
        self.assertTrue(all(group.four_point_score == 4 for group in groups))
        self.assertEqual(audit.writer.write_frame.await_count, 2)
