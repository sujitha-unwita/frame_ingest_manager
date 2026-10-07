"""Audit business flow with fake repositories and storage; no external writes."""

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from audit_write.services.audit_writer import AuditWriter
from config.audit_models import AuditWrite
from config.config_loader import ConfigLoader
from inference_pipeline.models import MappedDetection
from inference_pipeline.models.defect_group import DefectGroup


class AuditWriterTests(unittest.IsolatedAsyncioTestCase):
    """Only committed results are returned; failures reach the caller."""

    def setUp(self) -> None:
        settings = AuditWrite.model_validate(ConfigLoader().raw_config["audit_write"])
        settings_patch = patch(
            "audit_write.models._common.get_settings", return_value=settings
        )
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.connection = object()
        self.transaction = AsyncMock()
        self.transaction.__aenter__.return_value = self.connection
        self.transaction.__aexit__.return_value = False
        self.database = Mock(transaction=Mock(return_value=self.transaction))
        self.trigger_uuid = UUID(int=2)
        self.media_uuid = UUID(int=3)
        self.repositories = Mock(
            rolls=Mock(get=AsyncMock(return_value=None), insert=AsyncMock()),
            cameras=Mock(register=AsyncMock()),
            triggers=Mock(insert=AsyncMock(return_value={
                "trigger_primary_id": self.trigger_uuid,
            })),
            media=Mock(insert=AsyncMock(return_value={
                "media_id": self.media_uuid, "media_ref": "frames/frame.jpg",
            })),
            defects_live=Mock(insert=AsyncMock(return_value={
                "detection_id": UUID(int=4),
            })),
        )
        self.storage = Mock(upload=AsyncMock(), delete=AsyncMock())
        self.writer = AuditWriter(
            self.database, self.repositories, self.storage, settings
        )
        self.detection = MappedDetection(
            trigger_id=42, camera_id="camera_3", frame_reference="frame.jpg",
            defect_class="hole", confidence=0.9,
            bounding_box_px={"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            global_bounding_box_px={"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            frame_width_px=4096, frame_height_px=3072,
            world_x1_mm=1, world_y1_mm=2, world_x2_mm=3, world_y2_mm=4,
            four_point_score=4,
        )

    async def write(self, detections=None):
        """Supply a whole encoded frame without opening a file or bucket."""
        return await self.writer.write_frame(
            trigger_id=42, encoder_count=100, camera_id="camera_3",
            detections=[self.detection] if detections is None else detections,
            frame_bytes=b"whole-frame",
        )

    async def asyncSetUp(self):
        await self.writer.start_roll(["camera_3"])
        self.database.reset_mock()
        self.transaction.reset_mock()

    async def test_repeat_start_reuses_committed_roll(self):
        roll_id = self.writer.roll_id
        self.assertEqual(await self.writer.start_roll(["camera_3"]), roll_id)
        self.repositories.rolls.insert.assert_awaited_once()
        self.database.transaction.assert_not_called()

    async def test_failed_roll_commit_is_not_exposed_and_retry_reuses_uuid(self):
        self.writer._roll_id = None
        roll = self.writer._prepared_roll
        self.transaction.__aexit__.side_effect = RuntimeError("commit uncertain")
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaises(RuntimeError):
                await self.writer.start_roll(["camera_3"])
        self.assertIsNone(self.writer.roll_id)
        self.transaction.__aexit__.side_effect = None
        self.repositories.rolls.get.return_value = {"roll_id": roll.roll_id}
        inserts = self.repositories.rolls.insert.await_count
        self.assertEqual(await self.writer.start_roll(["camera_3"]), roll.roll_id)
        self.assertEqual(self.repositories.rolls.insert.await_count, inserts)

    async def test_multiple_detections_share_one_frame_and_transaction(self):
        result = await self.write([self.detection, self.detection])
        self.storage.upload.assert_awaited_once_with(
            result.media_ref, b"whole-frame", content_type="image/jpeg"
        )
        self.repositories.triggers.insert.assert_awaited_once()
        self.repositories.media.insert.assert_awaited_once()
        self.assertEqual(len(result.detections), 2)
        trigger = self.repositories.triggers.insert.await_args.args[0]
        media = self.repositories.media.insert.await_args.args[0]
        self.assertEqual(trigger.roll_id, self.writer.roll_id)
        self.assertEqual(trigger.trigger_sequence, 42)
        self.assertEqual(media.trigger_primary_id, trigger.trigger_primary_id)
        self.assertEqual(media.media_ref, result.media_ref)
        for call in self.repositories.defects_live.insert.await_args_list:
            row = call.args[0]
            self.assertEqual(row.trigger_primary_id, trigger.trigger_primary_id)
            self.assertEqual(row.media_id, media.media_id)
            self.assertIsNone(row.frame_variant)
            self.assertIsNone(row.defect_id)
            self.assertIsNone(row.tile_ids)
            self.assertIsInstance(row.detection_id, UUID)
            self.assertIs(call.kwargs["conn"], self.connection)
        self.transaction.__aexit__.assert_awaited_once_with(None, None, None)
        self.assertIsNone(self.detection.detection_id)
        self.assertNotEqual(
            result.detections[0].detection_id, result.detections[1].detection_id
        )

    async def test_storage_failure_prevents_database_writes(self):
        failure = RuntimeError("upload failed")
        self.storage.upload.side_effect = failure
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaises(RuntimeError) as raised:
                await self.write()
        self.assertIs(raised.exception, failure)
        self.database.transaction.assert_not_called()

    async def test_database_failure_exits_transaction_with_error(self):
        failure = RuntimeError("insert failed")
        self.repositories.defects_live.insert.side_effect = failure
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaises(RuntimeError) as raised:
                await self.write()
        self.assertIs(raised.exception, failure)
        self.assertIs(self.transaction.__aexit__.await_args.args[1], failure)
        self.storage.delete.assert_not_awaited()

    async def test_commit_failure_returns_no_success(self):
        self.transaction.__aexit__.side_effect = RuntimeError("commit uncertain")
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "commit uncertain"):
                await self.write()
        self.storage.delete.assert_not_awaited()

    async def test_wrong_camera_is_rejected_before_upload(self):
        self.detection.camera_id = "camera_4"
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "does not belong"):
                await self.write()
        self.storage.upload.assert_not_awaited()

    def prepare_final_group(self):
        """Provide persisted raw members and repository replies for finalization."""
        members = (UUID(int=10), UUID(int=11))
        group = DefectGroup(
            defect_class="hole", confidence=0.9,
            global_bounding_box_px={"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            member_detection_ids=members,
            world_x1_mm=1, world_y1_mm=2, world_x2_mm=3, world_y2_mm=4,
        )
        self.repositories.rolls.lock_for_finalization = AsyncMock(
            return_value={"roll_id": self.writer.roll_id}
        )
        self.repositories.defects_live.lock_final_members = AsyncMock(return_value=[
            {"detection_id": member, "defect_id": None} for member in members
        ])
        self.repositories.defects_live.merge_into = AsyncMock(return_value=[
            {"detection_id": member} for member in members
        ])
        self.repositories.defects_live.count = AsyncMock(return_value=len(members))

        async def insert_final(model, **kwargs):
            return {**model.to_row(), "inserted_at": datetime.now(timezone.utc)}

        self.repositories.defects_final = Mock(
            max_number=AsyncMock(return_value=7),
            insert=AsyncMock(side_effect=insert_final), get=AsyncMock(),
        )
        return group

    async def test_final_number_resumes_from_saved_maximum_and_links_members(self):
        group = self.prepare_final_group()
        saved = await self.writer.write_final_defects([group])
        self.assertEqual(saved[0].defect_no, 8)
        self.repositories.rolls.lock_for_finalization.assert_awaited_once_with(
            self.writer.roll_id, conn=self.connection
        )
        self.repositories.defects_live.merge_into.assert_awaited_once_with(
            group.member_detection_ids, saved[0].defect_id, conn=self.connection
        )
        self.transaction.__aexit__.assert_awaited_once_with(None, None, None)
        self.assertEqual(saved[0].four_point_score, 4)
        self.assertIsNone(saved[0].defect_area_mm2)

    async def test_final_retry_reuses_identity_without_allocating_number(self):
        group = self.prepare_final_group()
        first = (await self.writer.write_final_defects([group]))[0]
        self.repositories.defects_live.lock_final_members.return_value = [
            {"detection_id": member, "defect_id": first.defect_id}
            for member in group.member_detection_ids
        ]
        self.repositories.defects_final.get.return_value = first.model_dump()
        second = (await self.writer.write_final_defects([group]))[0]
        self.assertEqual(second.defect_id, first.defect_id)
        self.repositories.defects_final.max_number.assert_awaited_once()
        self.repositories.defects_final.insert.assert_awaited_once()

    async def test_missing_members_block_number_allocation(self):
        group = self.prepare_final_group()
        self.repositories.defects_live.lock_final_members.return_value = []
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "missing"):
                await self.writer.write_final_defects([group])
        self.repositories.defects_final.max_number.assert_not_awaited()

    async def test_ungraded_group_is_rejected_before_numbering(self):
        group = self.prepare_final_group()
        group.four_point_score = None
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "must be graded"):
                await self.writer.write_final_defects([group])
        self.repositories.defects_final.max_number.assert_not_awaited()

    async def test_failed_final_link_rolls_back(self):
        group = self.prepare_final_group()
        self.repositories.defects_live.merge_into.return_value = []
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "not all raw"):
                await self.writer.write_final_defects([group])
        self.assertIs(self.transaction.__aexit__.await_args.args[0], RuntimeError)

    async def test_conflicting_members_are_not_reassigned(self):
        group = self.prepare_final_group()
        self.repositories.defects_live.lock_final_members.return_value[0][
            "defect_id"
        ] = UUID(int=99)
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "conflicting"):
                await self.writer.write_final_defects([group])
        self.repositories.defects_final.insert.assert_not_awaited()

    async def test_final_commit_failure_does_not_return_success(self):
        group = self.prepare_final_group()
        self.transaction.__aexit__.side_effect = RuntimeError("commit uncertain")
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "commit uncertain"):
                await self.writer.write_final_defects([group])
        # There is no in-memory number counter to advance on a failed commit.
        self.assertNotIn("_next_defect_no", vars(self.writer))

    async def test_duplicate_members_across_groups_are_rejected(self):
        group = self.prepare_final_group()
        with self.assertLogs("audit_write.services.audit_writer", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "only one final group"):
                await self.writer.write_final_defects([group, group])
        self.repositories.defects_final.insert.assert_not_awaited()
