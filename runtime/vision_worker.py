"""Vision process: live/folder frame ingestion, CUDA model, mapping and GPU sampling."""

import asyncio
import logging
import time
from pathlib import Path

from frame_ingest_manager.frame_sources.frame_source_factory import FrameSourceFactory
from frame_ingest_manager.models.trigger_telemetry import TriggerTelemetry
from frame_ingest_manager.services.frame_ingest_manager import FrameIngestManager
from inference_pipeline.engines.coordinate_mapper import CoordinateMapper
from inference_pipeline.engines.sahi_detection_engine import SahiDetectionEngine
from inference_pipeline.engines.tile_generator import TileGenerator
from utils.gpu_monitor import GpuMonitor
from utils.timing import Timing

LOGGER = logging.getLogger(__name__)


class VisionWorker:
    """Process one complete camera row at a time using one shared model."""

    def __init__(self, config, channels) -> None:
        """Construct vision dependencies only after the child has been spawned."""
        self.timing = Timing.from_config(config.logging, Path(config.log_root), "vision")
        self.config = config
        self.channels = channels
        self.source = FrameSourceFactory.create(config.frame_ingest)
        self.frames = FrameIngestManager(
            camera_ids=config.frame_ingest.camera_ids,
            expected_camera_count=config.frame_ingest.expected_camera_count,
            frame_source=self.source,
        )
        self.detector = SahiDetectionEngine(
            config.inference, TileGenerator(config.inference),
            timing=self.timing,
        )
        self.mapper = CoordinateMapper(
            config.frame_ingest.camera_ids, config.inference.layout,
            config.encoder.mm_per_count,
        )
        self.gpu = GpuMonitor(config.gpu_monitoring)

    async def process(self, job: dict) -> dict:
        """Match a full row; consume live images before returning their leases."""
        self.timing.record("vision_queue_wait", job["admitted_at"],
                           trigger_id=job["trigger"]["trigger_id"], sequence=job["sequence"])
        trigger = TriggerTelemetry.model_validate(job["trigger"])
        if trigger.encoder_count < 0:
            raise ValueError("downstream mapping/merging requires nonnegative forward encoder positions")
        with self.timing.measure("frame_bundle", trigger_id=trigger.trigger_id):
            acquisition = asyncio.create_task(asyncio.to_thread(self.frames.create_bundle, trigger))
            try:
                bundle = await asyncio.shield(acquisition)
            except asyncio.CancelledError:
                acquired = await acquisition
                acquired.release()
                raise
        with bundle:
            frames = {}
            for camera_id, frame in bundle.camera_frames.items():
                live = frame.image is not None
                if not live:
                    path = Path(frame.frame_reference).resolve()
                    before = path.stat()
                self.gpu.begin_inference()
                try:
                    with self.timing.measure("inference", trigger_id=trigger.trigger_id, camera_id=camera_id):
                        detections = await self.detector.detect(frame)
                finally:
                    average = self.gpu.end_inference()
                mapped = []
                with self.timing.measure("coordinate_mapping", trigger_id=trigger.trigger_id, camera_id=camera_id):
                    for detection in detections:
                        result = self.mapper.map_detection(detection, trigger)
                        if result is not None:
                            mapped.append(result.model_dump())
                if live:
                    record = {"detections": mapped, "camera_metadata": frame.model_dump()}
                    if mapped:
                        encoding = asyncio.create_task(asyncio.to_thread(self._encode_frame, frame.image))
                        try:
                            record["encoded_bytes"] = await asyncio.shield(encoding)
                        except asyncio.CancelledError:
                            await encoding
                            raise
                    frames[camera_id] = record
                else:
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError(f"mock frame changed during inference: {path}")
                    frames[camera_id] = {"reference": str(path), "size": after.st_size,
                                        "mtime_ns": after.st_mtime_ns, "detections": mapped}
                await self.channels.report("gpu", average=average)
        return {**job, "frames": frames, "vision_finished_at": time.monotonic()}

    def _encode_frame(self, image):
        """Immutable bytes use the audit storage key's configured image format."""
        import cv2
        ok, encoded = cv2.imencode("." + self.config.frame_encoding, image)
        if not ok:
            raise RuntimeError("could not encode live frame for audit")
        return encoded.tobytes()

    async def run(self) -> None:
        """Arm sources, consume FIFO jobs, then pass the drain marker to audit."""
        pulse = asyncio.create_task(self.channels.pulse())
        try:
            await self.gpu.start()
            with self.timing.measure("model_startup"):
                await self.detector.start()
            # Warm the model before arming; warmup must not consume camera buffers.
            starting = asyncio.create_task(asyncio.to_thread(self.frames.start))
            try:
                await asyncio.shield(starting)
            except asyncio.CancelledError:
                await starting
                raise
            if self.config.frame_ingest.source_kind == "mvs":
                for camera in self.source.cameras:
                    if camera.geometry.width != self.config.inference.layout.frame_width_px:
                        raise RuntimeError(f"{camera.camera_id}: camera width differs from inference.layout.frame_width_px")
            await self.channels.report("ready")
            while True:
                received, job = await self.channels.receive(self.channels.triggers)
                self.frames.check_health()
                if not received:
                    continue
                if job is None:
                    await self.channels.send(self.channels.audit, None)
                    break
                await self.channels.report("started", sequence=job["sequence"])
                try:
                    with self.timing.measure(
                        "vision_total", trigger_id=job["trigger"]["trigger_id"],
                        sequence=job["sequence"],
                    ):
                        result = await self.process(job)
                except Exception:
                    LOGGER.exception(
                        "Vision failed: trigger=%s sequence=%s",
                        job["trigger"]["trigger_id"], job["sequence"],
                    )
                    raise
                with self.timing.measure(
                    "audit_queue_put", trigger_id=job["trigger"]["trigger_id"],
                    sequence=job["sequence"],
                ):
                    await self.channels.send(self.channels.audit, result)
                await self.channels.report("completed", sequence=job["sequence"])
            await self.channels.report("stopped")
        finally:
            try:
                await asyncio.to_thread(self.frames.stop)
            finally:
                try:
                    await self.detector.stop()
                finally:
                    try:
                        await self.gpu.stop()
                    finally:
                        pulse.cancel()
                        await asyncio.gather(pulse, return_exceptions=True)


async def run(config, channels) -> None:
    """Child entry point; no GPU object is constructed by the parent."""
    await VisionWorker(config, channels).run()
