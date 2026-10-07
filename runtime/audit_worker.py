"""Audit process: roll preparation, S3, raw persistence, merging and finals."""

import asyncio
import logging
from pathlib import Path
from queue import Empty

from audit_write.database import Database
from audit_write.repositories import Repositories
from audit_write.schema_checker import SchemaChecker
from audit_write.services.audit_writer import AuditWriter
from audit_write.storage.client import create_s3_client
from audit_write.storage.objects import ObjectStore
from dashboard_service.services import event_factory as events
from inference_pipeline.engines.coordinate_mapper import CoordinateMapper
from inference_pipeline.engines.deduplication_engine import DeduplicationEngine
from inference_pipeline.engines.four_point_grading_engine import FourPointGradingEngine
from inference_pipeline.models import MappedDetection
from utils.timing import Timing
from config.config_loader import initialize_audit_settings
LOGGER = logging.getLogger(__name__)


class AuditWorker:
    """Persist complete rows in FIFO order; audit assigns raw IDs before merging."""

    def __init__(self, config, channels) -> None:
        """Create child-owned clients and the roll's deduplication state."""
        initialize_audit_settings(config.audit_write)
        self.timing = Timing.from_config(config.logging, Path(config.log_root), "audit")
        self.config = config
        self.channels = channels
        self.database = Database(config.audit_write.database)
        self.storage = ObjectStore(
            create_s3_client(config.audit_write.storage), config.audit_write.storage,
        )
        self.writer = AuditWriter(
            self.database, Repositories.create(self.database, config.audit_write),
            self.storage, config.audit_write, timing=self.timing,
        )
        mapper = CoordinateMapper(
            config.frame_ingest.camera_ids, config.layout,
            config.encoder.mm_per_count,
        )
        self.merger = DeduplicationEngine(
            config.deduplication, config.frame_ingest.camera_ids,
            FourPointGradingEngine(config.grading, mapper.active_width_mm),
        )
        self.completed = 0

    @staticmethod
    def read_frame(frame: dict) -> bytes:
        """Read owned live bytes or verify unchanged mock files before upload."""
        if "encoded_bytes" in frame:
            content = frame["encoded_bytes"]
            if not isinstance(content, bytes) or not content:
                raise ValueError("live frame must carry nonempty owned encoded bytes")
            return content
        path = Path(frame["reference"])
        expected = (frame["size"], frame["mtime_ns"])
        before = path.stat()
        if (before.st_size, before.st_mtime_ns) != expected:
            raise RuntimeError(f"mock frame changed before audit: {path}")
        content = path.read_bytes()
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != expected:
            raise RuntimeError(f"mock frame changed while reading: {path}")
        return content

    async def prepare_roll(self, command: dict) -> None:
        """Acknowledge the roll only after its database transaction commits."""
        await self.channels.report("started", request_id=command["request_id"])
        with self.timing.measure("roll_preparation"):
            roll_id = await self.writer.start_roll(self.config.frame_ingest.camera_ids)
        await self.channels.report(
            "roll_ready", request_id=command["request_id"], roll_id=str(roll_id),
        )

    async def process(self, job: dict) -> None:
        """Upload one frame at a time, then merge the complete persisted row."""
        self.timing.record(
            "audit_handoff_wait", job["vision_finished_at"],
            trigger_id=job["trigger"]["trigger_id"], sequence=job["sequence"],
        )
        if job["sequence"] != self.completed + 1:
            raise RuntimeError("audit received an out-of-order trigger")
        if job["roll_id"] != str(self.writer.roll_id):
            raise RuntimeError("audit job belongs to a different roll")
        trigger = job["trigger"]
        mapped = {}
        for camera_id, frame in job["frames"].items():
            mapped[camera_id] = []
            detections = [
                MappedDetection.model_validate(item) for item in frame["detections"]
            ]
            if not detections:
                continue
            with self.timing.measure(
                "frame_read", trigger_id=trigger["trigger_id"], camera_id=camera_id,
            ):
                frame_bytes = await asyncio.to_thread(self.read_frame, frame)
            with self.timing.measure(
                "frame_audit", trigger_id=trigger["trigger_id"], camera_id=camera_id,
            ):
                result = await self.writer.write_frame(
                    trigger_id=trigger["trigger_id"],
                    encoder_count=trigger["encoder_count"],
                    camera_id=camera_id, frame_bytes=frame_bytes,
                    detections=detections,
                )
            del frame_bytes
            mapped[camera_id] = list(result.detections)
            with self.timing.measure(
                "live_event_handoff", trigger_id=trigger["trigger_id"], camera_id=camera_id,
            ):
                for detection in result.detections:
                    box = detection.bounding_box_px
                    event = events.defect_event(
                        trigger_id=detection.trigger_id, camera_id=detection.camera_id,
                        tile_ids=detection.tile_ids, frame_variant=None,
                        defect_class=detection.defect_class,
                        confidence=detection.confidence,
                        world_x1_mm=detection.world_x1_mm,
                        world_y1_mm=detection.world_y1_mm,
                        world_x2_mm=detection.world_x2_mm,
                        world_y2_mm=detection.world_y2_mm,
                        bbox_x1_px=box.x1, bbox_y1_px=box.y1,
                        bbox_x2_px=box.x2, bbox_y2_px=box.y2,
                        crop_thumbnail_ref=result.media_ref,
                    )
                    await self.channels.report("dashboard", event=event)
        row_top = (
            trigger["encoder_count"] * self.config.encoder.mm_per_count
            / self.config.layout.mm_per_pixel_y
        )
        with self.timing.measure(
            "deduplication_grading", trigger_id=trigger["trigger_id"],
        ):
            settled = self.merger.process_trigger(trigger["trigger_id"], row_top, mapped)
        if settled:
            with self.timing.measure(
                "final_defect_write", trigger_id=trigger["trigger_id"],
            ):
                await self.writer.write_final_defects(settled)
        self.completed = job["sequence"]
        LOGGER.info(
            "Committed trigger=%s sequence=%s settled=%s",
            trigger["trigger_id"], self.completed, len(settled),
        )

    async def run(self) -> None:
        """Serve roll requests and ordered audit work using child-owned clients."""
        pulse = asyncio.create_task(self.channels.pulse())
        try:
            with self.timing.measure("database_connect"):
                await self.database.connect()
            with self.timing.measure("schema_check"):
                async with self.database.connection() as connection:
                    await SchemaChecker(self.config.audit_write).check(connection)
            await self.channels.report("ready")
            while True:
                try:
                    command = self.channels.commands.get_nowait()
                except Empty:
                    command = None
                if command is not None:
                    await self.prepare_roll(command)
                received, job = await self.channels.receive(self.channels.audit)
                if not received:
                    continue
                if job is None:
                    break
                await self.channels.report("started", sequence=job["sequence"])
                try:
                    with self.timing.measure(
                        "trigger_to_audit", started_at=job["admitted_at"],
                        trigger_id=job["trigger"]["trigger_id"], sequence=job["sequence"],
                    ), self.timing.measure(
                        "audit_total", trigger_id=job["trigger"]["trigger_id"],
                        sequence=job["sequence"],
                    ):
                        await self.process(job)
                except Exception:
                    LOGGER.exception(
                        "Audit failed: trigger=%s sequence=%s; inspect persisted "
                        "rows and S3 before retrying",
                        job["trigger"]["trigger_id"], job["sequence"],
                    )
                    raise
                await self.channels.report("completed", sequence=self.completed)
            # Application shutdown is not a confirmed physical end-of-roll.
            if self.merger.open_count:
                LOGGER.warning(
                    "Shutdown leaves %s open groups; explicit roll completion "
                    "is still required. Raw detections remain in the database.",
                    self.merger.open_count,
                )
            await self.channels.report("stopped")
        finally:
            try:
                await self.database.close()
            finally:
                await self.storage.close()
                pulse.cancel()
                await asyncio.gather(pulse, return_exceptions=True)


async def run(config, channels) -> None:
    """Construct all database and storage resources inside the audit process."""
    await AuditWorker(config, channels).run()
