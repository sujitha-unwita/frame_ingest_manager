"""Audit business flow: roll ownership, whole-frame storage, and raw rows."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from asyncpg import Connection

from audit_write.database import Database
from audit_write.models.defect_live import DefectLiveCreate
from audit_write.models.defect_final import DefectFinal, DefectFinalCreate
from audit_write.models.media import MediaCreate
from audit_write.models.roll import RollCreate
from audit_write.models.trigger import TriggerCreate
from audit_write.repositories import Repositories
from audit_write.storage.keys import build_key, content_type_for
from audit_write.storage.objects import ObjectStore
from audit_write.utils.identifiers import Identifiers
from config.audit_models import AuditWrite
from inference_pipeline.models import MappedDetection
from inference_pipeline.models.defect_group import DefectGroup
from utils.timing import Timing

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameAuditResult:
    """Saved observations retain their UUIDs and share one whole-frame key."""

    media_ref: str
    detections: tuple[MappedDetection, ...]


class AuditWriter:
    """Own the current roll and coordinate storage; never publish or infer."""

    def __init__(
        self,
        database: Database,
        repositories: Repositories,
        storage: ObjectStore,
        audit_write_config: AuditWrite,
        *, timing: Timing | None = None,
    ) -> None:
        """Keep injected resources and settings; perform no I/O."""
        # Existing standalone callers can omit instrumentation.
        self._timing = timing if timing is not None else Timing(enabled=False)
        self._database = database
        self._repositories = repositories
        self._storage = storage
        self._audit_write_config = audit_write_config
        self._roll_id: UUID | None = None
        self._prepared_roll: RollCreate | None = None
        self._roll_lock = asyncio.Lock()

    @property
    def roll_id(self) -> UUID | None:
        """Return the committed current roll, or None before Start Capture."""
        return self._roll_id

    async def start_roll(self, camera_ids: Sequence[str]) -> UUID:
        """Create one roll before capture; reuse it for repeat Start or resume.

        Keep the prepared UUID on failure. A retry checks the database for that
        UUID first, covering a commit that succeeded but lost its reply.
        Roll completion and recovery after a process restart are not implemented.
        """
        LOGGER.info("AuditWriter.start_roll: preparing capture")
        async with self._roll_lock:
            if self._roll_id is not None:
                return self._roll_id
            try:
                if self._prepared_roll is None:
                    roll_id = Identifiers.new_id()
                    self._prepared_roll = RollCreate(
                        roll_id=roll_id,
                        roll_label=f"{self._audit_write_config.roll_label_prefix}{roll_id}",
                    )
                roll = self._prepared_roll
                async with self._database.transaction() as connection:
                    existing = await self._repositories.rolls.get(
                        roll.roll_id, conn=connection
                    )
                    if existing is None:
                        await self._repositories.rolls.insert(roll, conn=connection)
                    for camera_id in camera_ids:
                        await self._repositories.cameras.register(
                            camera_id, conn=connection
                        )
                # Publish the in-memory identity only after commit succeeds.
                self._roll_id = roll.roll_id
                LOGGER.info("AuditWriter.start_roll: ready roll=%s", self._roll_id)
                return self._roll_id
            except (Exception, asyncio.CancelledError):
                LOGGER.exception("AuditWriter.start_roll: failed")
                raise

    async def write_frame(
        self,
        *,
        trigger_id: int,
        encoder_count: int,
        camera_id: str,
        frame_bytes: bytes,
        detections: Sequence[MappedDetection],
        frame_variant: str | None = None,
    ) -> FrameAuditResult:
        """Save one camera frame and all its detections, returning committed IDs.

        trigger_id is the MCU sequence, not a database UUID. Input detections
        remain unchanged on failure. Keys use a unique media UUID, so a failed
        attempt cannot overwrite another frame. No automatic frame retry occurs.
        """
        LOGGER.info(
            "AuditWriter.write_frame: trigger=%s camera=%s", trigger_id, camera_id
        )
        storage_key = None
        try:
            if self._roll_id is None:
                raise RuntimeError("Start Capture must create a roll before auditing")
            if not frame_bytes:
                raise ValueError("whole-frame bytes must not be empty")
            for detection in detections:
                if (
                    detection.trigger_id != trigger_id
                    or detection.camera_id != camera_id
                ):
                    raise ValueError("detection does not belong to this camera trigger")
                if detection.detection_id is not None:
                    raise ValueError("detection is already persisted")

            media_id = Identifiers.new_id()
            # This is a storage date, not an invented camera capture timestamp.
            storage_key = build_key(
                self._audit_write_config.frame_key_kind,
                settings=self._audit_write_config.storage,
                roll_id=self._roll_id,
                media_id=media_id,
                camera_id=camera_id,
                trigger_sequence=trigger_id,
                captured_at=datetime.now(timezone.utc),
            )
            with self._timing.measure(
                "s3_upload", trigger_id=trigger_id, camera_id=camera_id,
            ):
                await self._upload_frame(storage_key, frame_bytes)
            with self._timing.measure(
                "raw_database_transaction", trigger_id=trigger_id, camera_id=camera_id,
            ):
                async with self._database.transaction() as connection:
                    trigger = await self._insert_trigger(
                        trigger_id, encoder_count, camera_id, connection
                    )
                    await self._insert_media(
                        trigger, media_id, storage_key, frame_variant, connection
                    )
                    saved = await self._insert_detections(
                        trigger, media_id, detections, frame_variant, connection
                    )
            LOGGER.info(
                "AuditWriter.write_frame: committed media=%s detections=%s",
                media_id, len(saved),
            )
            return FrameAuditResult(media_ref=storage_key, detections=tuple(saved))
        except (Exception, asyncio.CancelledError):
            LOGGER.exception("AuditWriter.write_frame: failed")
            if storage_key is not None:
                # Do not delete: commit may have succeeded despite a lost reply.
                LOGGER.error(
                    "Reconcile S3 key %s with the database before retrying",
                    storage_key,
                )
            raise

    async def write_final_defects(
        self, groups: Sequence[DefectGroup]
    ) -> list[DefectFinal]:
        """Persist settled groups and link their raw members atomically.

        The caller supplies only settled merger output, never open groups.
        Numbers follow finalization order within the roll. The database roll
        lock protects max-number allocation across writers, not just tasks.
        A retry of the exact same members returns the existing final record,
        covering a successful commit whose reply was lost.
        """
        LOGGER.info("AuditWriter.write_final_defects: groups=%s", len(groups))
        if not groups:
            return []
        try:
            if self._roll_id is None:
                raise RuntimeError("Start Capture must create a roll first")
            seen: set[UUID] = set()
            for group in groups:
                if group.four_point_score is None:
                    raise ValueError("final group must be graded before persistence")
                members = group.member_detection_ids
                if len(set(members)) != len(members) or seen.intersection(members):
                    raise ValueError("raw members must appear in only one final group")
                if (
                    group.world_x2_mm <= group.world_x1_mm
                    or group.world_y2_mm <= group.world_y1_mm
                ):
                    raise ValueError("final group must have a non-empty rectangle")
                seen.update(members)

            saved = []
            async with self._database.transaction() as connection:
                roll = await self._repositories.rolls.lock_for_finalization(
                    self._roll_id, conn=connection
                )
                if roll is None:
                    raise RuntimeError("current audit roll no longer exists")
                for group in groups:
                    saved.append(await self._write_final_group(group, connection))
            LOGGER.info("AuditWriter.write_final_defects: committed=%s", len(saved))
            return saved
        except (Exception, asyncio.CancelledError):
            LOGGER.exception("AuditWriter.write_final_defects: failed")
            raise

    async def _write_final_group(
        self, group: DefectGroup, connection: Connection
    ) -> DefectFinal:
        """Check membership, insert one final row, and link its raw observations."""
        LOGGER.info("AuditWriter._write_final_group: checking raw members")
        members = group.member_detection_ids
        rows = await self._repositories.defects_live.lock_final_members(
            self._roll_id, members, conn=connection
        )
        if {row["detection_id"] for row in rows} != set(members):
            raise ValueError("final group contains missing or different-roll detections")
        links = {row["defect_id"] for row in rows}
        if links != {None}:
            # A partial link or two different final IDs is a conflict, not a retry.
            if None in links or len(links) != 1:
                raise ValueError("final group members already have conflicting links")
            defect_id = next(iter(links))
            existing = await self._repositories.defects_final.get(
                defect_id, conn=connection
            )
            count = await self._repositories.defects_live.count(
                {"roll_id": self._roll_id, "defect_id": defect_id}, conn=connection
            )
            if (
                existing is None or existing["roll_id"] != self._roll_id
                or count != len(members)
            ):
                raise ValueError("retry must match an entire existing final group")
            return DefectFinal.model_validate(existing)

        highest = await self._repositories.defects_final.max_number(
            self._roll_id, conn=connection
        )
        final = DefectFinalCreate(
            defect_id=Identifiers.new_id(), roll_id=self._roll_id,
            defect_no=highest + 1,
            defect_class=group.defect_class, confidence=group.confidence,
            world_x1_mm=group.world_x1_mm, world_y1_mm=group.world_y1_mm,
            world_x2_mm=group.world_x2_mm, world_y2_mm=group.world_y2_mm,
            four_point_score=group.four_point_score,
            # Bounding-box area is not necessarily actual defect area.
            # Actual defect area measurement remains separate work.
        )
        row = await self._repositories.defects_final.insert(final, conn=connection)
        linked = await self._repositories.defects_live.merge_into(
            members, final.defect_id, conn=connection
        )
        if {item["detection_id"] for item in linked} != set(members):
            raise RuntimeError("not all raw detections were linked; rolling back")
        return DefectFinal.model_validate(row)

    async def _upload_frame(self, key: str, frame_bytes: bytes) -> None:
        """Upload the original encoded frame, with no crop or conversion."""
        LOGGER.info("AuditWriter._upload_frame: %s", key)
        await self._storage.upload(
            key, frame_bytes,
            content_type=content_type_for(
                self._audit_write_config.frame_key_kind, self._audit_write_config.storage
            ),
        )

    async def _insert_trigger(
        self, trigger_id: int, encoder_count: int, camera_id: str,
        connection: Connection,
    ) -> TriggerCreate:
        """Create and insert the UUID trigger with its original MCU sequence."""
        trigger = TriggerCreate(
            trigger_primary_id=Identifiers.new_id(), roll_id=self._roll_id,
            camera_id=camera_id, trigger_sequence=trigger_id,
            encoder_count=encoder_count,
        )
        LOGGER.info("AuditWriter._insert_trigger: %s", trigger.trigger_primary_id)
        await self._repositories.triggers.insert(trigger, conn=connection)
        return trigger

    async def _insert_media(
        self, trigger: TriggerCreate, media_id: UUID, key: str,
        frame_variant: str | None, connection: Connection,
    ) -> None:
        """Create and insert one whole-frame media record for this trigger."""
        LOGGER.info("AuditWriter._insert_media: %s", media_id)
        media = MediaCreate(
            media_id=media_id, roll_id=trigger.roll_id,
            trigger_primary_id=trigger.trigger_primary_id,
            camera_id=trigger.camera_id, frame_variant=frame_variant,
            media_ref=key, uploaded_at=datetime.now(timezone.utc),
        )
        await self._repositories.media.insert(media, conn=connection)

    async def _insert_detections(
        self, trigger: TriggerCreate, media_id: UUID,
        detections: Sequence[MappedDetection], frame_variant: str | None,
        connection: Connection,
    ) -> list[MappedDetection]:
        """Create raw rows sharing the media/trigger IDs; never number defects."""
        LOGGER.info("AuditWriter._insert_detections: count=%s", len(detections))
        saved = []
        for detection in detections:
            detection_id = Identifiers.new_id()
            box = detection.bounding_box_px
            row = DefectLiveCreate(
                detection_id=detection_id, roll_id=trigger.roll_id,
                trigger_primary_id=trigger.trigger_primary_id,
                encoder_count=trigger.encoder_count, camera_id=trigger.camera_id,
                frame_variant=frame_variant, media_id=media_id,
                tile_ids=detection.tile_ids,
                defect_class=detection.defect_class, confidence=detection.confidence,
                bbox_x1_px=box.x1, bbox_y1_px=box.y1,
                bbox_x2_px=box.x2, bbox_y2_px=box.y2,
                world_x1_mm=detection.world_x1_mm,
                world_y1_mm=detection.world_y1_mm,
                world_x2_mm=detection.world_x2_mm,
                world_y2_mm=detection.world_y2_mm,
                model_version=self._audit_write_config.model_version,
            )
            await self._repositories.defects_live.insert(row, conn=connection)
            saved.append(detection.model_copy(update={"detection_id": detection_id}))
        return saved
