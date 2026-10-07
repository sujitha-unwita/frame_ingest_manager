"""Offline process tests using actual spawn and bounded multiprocessing queues."""

import asyncio
import ast
import pickle
import unittest
from pathlib import Path
from queue import Full
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from runtime.process_runtime import ProcessRuntime
from tests.process_worker_fixture import fake_worker


def make_config():
    """Small explicit test limits; production limits live in config.yaml."""
    processing = SimpleNamespace(
        trigger_queue_capacity=4, audit_queue_capacity=4,
        command_queue_capacity=2, event_queue_capacity=32,
        poll_interval_s=0.02, queue_put_timeout_s=0.5,
        startup_timeout_s=5.0, roll_timeout_s=2.0, health_timeout_s=2.0,
        vision_job_timeout_s=3.0, audit_job_timeout_s=3.0,
        shutdown_timeout_s=3.0, terminate_timeout_s=1.0,
    )
    return SimpleNamespace(
        processing=processing,
        frame_ingest=SimpleNamespace(source_kind="folder", folder_path="test"),
        audit_write=SimpleNamespace(enabled=True),
        logging=None, source_paths=(), encoder=None, gpu_monitoring=None,
        inference=SimpleNamespace(layout=None, deduplication=None, grading=None),
    )


class ProcessRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Prove slow audit does not monopolize the parent event loop."""

    def setUp(self):
        self.events = []
        self.runtime = ProcessRuntime(
            make_config(), self.publish, fake_worker,
        )

    async def publish(self, event):
        self.events.append(event)

    async def asyncTearDown(self):
        await self.runtime.stop()

    async def test_spawn_order_drain_and_control_responsiveness(self):
        await self.runtime.start()
        await self.runtime.prepare_roll()
        for trigger_id in range(3):
            self.runtime.submit({"trigger_id": trigger_id})
        ticks = 0
        while self.runtime.completed < 3:
            await asyncio.sleep(0.01)
            ticks += 1
            self.assertLess(ticks, 500)
        self.assertGreater(ticks, 10)
        self.assertEqual(self.events, [{"sequence": n} for n in (1, 2, 3)])
        self.assertEqual(self.runtime.queue_depth, 0)
        self.assertTrue(self.runtime.accepting)

    async def test_shutdown_drains_accepted_work(self):
        await self.runtime.start()
        await self.runtime.prepare_roll()
        self.runtime.submit({"trigger_id": 99})
        await self.runtime.stop()
        self.assertEqual(self.runtime.completed, 1)
        self.assertFalse(self.runtime.processes)

    async def test_shutdown_without_roll_does_not_hang(self):
        await self.runtime.start()
        await self.runtime.stop()
        self.assertFalse(self.runtime.processes)

    async def test_child_failure_blocks_start(self):
        self.runtime.app_config.frame_ingest.folder_path = "crash"
        with self.assertRaises(RuntimeError):
            await self.runtime.start()
        self.assertIn("failed", self.runtime.states.values())
        self.assertFalse(self.runtime.accepting)

    async def test_queue_full_does_not_disable_later_submissions(self):
        self.runtime.accepting = True
        self.runtime.require_workers = Mock()
        self.runtime.roll_id = "roll"
        self.runtime.queues["triggers"] = Mock()
        self.runtime.queues["triggers"].put_nowait.side_effect = Full
        with self.assertRaises(RuntimeError):
            self.runtime.submit({"trigger_id": 7})
        self.assertEqual(self.runtime.accepted, 0)
        self.assertTrue(self.runtime.accepting)
        self.runtime.queues["triggers"].put_nowait.side_effect = None
        self.runtime.submit({"trigger_id": 8})
        self.assertEqual(self.runtime.accepted, 1)

    async def test_start_requires_real_telemetry_for_mvs_and_audit(self):
        self.runtime.app_config.frame_ingest.source_kind = "mvs"
        self.runtime.app_config.transport = SimpleNamespace(is_mock=True)
        with self.assertRaisesRegex(RuntimeError, "real MCU"):
            await self.runtime.start()
        self.runtime.app_config.frame_ingest.source_kind = "folder"
        self.runtime.app_config.audit_write.enabled = False
        with self.assertRaisesRegex(RuntimeError, "audit_write"):
            await self.runtime.start()

    async def test_mvs_uses_existing_process_runtime(self):
        app = self.runtime.app_config
        app.frame_ingest.source_kind = "mvs"
        from frame_ingest_manager.models.mvs_config import MvsSettings
        app.frame_ingest.mvs = MvsSettings(trigger_id_base=1, cameras=[
            {"id": "camera_1", "serial_number": "test", "frame_number_base": 1}])
        app.transport = SimpleNamespace(is_mock=False)
        app.audit_write.frame_key_kind = "whole_frame"
        app.audit_write.storage = SimpleNamespace(keys={
            "whole_frame": SimpleNamespace(ext="png", content_type="image/png")})
        await self.runtime.start()
        await self.runtime.prepare_roll()
        self.runtime.submit({"trigger_id": 42})
        await self.runtime.stop()
        self.assertEqual(self.runtime.completed, 1)
        self.assertEqual(self.runtime._worker_config("vision").frame_encoding, "png")

    async def test_duplicate_roll_start_reuses_prepared_roll(self):
        await self.runtime.start()
        first, second = await asyncio.gather(
            self.runtime.prepare_roll(), self.runtime.prepare_roll(),
        )
        self.assertEqual(first, second)

    async def test_job_deadline_only_logs_warning(self):
        self.runtime.app_config.frame_ingest.folder_path = "hang"
        self.runtime.config.vision_job_timeout_s = 0.1
        self.runtime.config.shutdown_timeout_s = 0.2
        await self.runtime.start()
        await self.runtime.prepare_roll()
        self.runtime.submit({"trigger_id": 1})
        for _ in range(100):
            if ("vision", "job") in self.runtime._reported_delays:
                break
            await asyncio.sleep(0.02)
        self.assertIn(("vision", "job"), self.runtime._reported_delays)
        self.assertTrue(self.runtime.accepting)

    async def test_running_child_crash_is_reported_as_unavailable(self):
        await self.runtime.start()
        self.runtime.processes["vision"].terminate()
        for _ in range(100):
            if self.runtime.states["vision"] == "failed":
                break
            await asyncio.sleep(0.02)
        with self.assertRaisesRegex(RuntimeError, "vision"):
            self.runtime.require_workers()
        self.runtime.config.shutdown_timeout_s = 0.2

    async def test_runtime_cannot_restart_with_stale_queues(self):
        await self.runtime.start()
        await self.runtime.stop()
        with self.assertRaisesRegex(RuntimeError, "new runtime"):
            await self.runtime.start()


class OwnershipTests(unittest.TestCase):
    def test_parent_does_not_import_gpu_or_database_modules(self):
        root = Path(__file__).resolve().parents[1]
        for relative in ("app_orchestrator.py", "runtime/process_runtime.py"):
            tree = ast.parse((root / relative).read_text())
            imports = [
                node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            ]
            self.assertFalse(any(
                name.startswith(("audit_write", "inference_pipeline", "torch", "sahi"))
                for name in imports
            ))

    def test_vision_configuration_does_not_contain_database_credentials(self):
        runtime = ProcessRuntime(make_config(), AsyncMock())
        payload = runtime._worker_config("vision")
        self.assertFalse(hasattr(payload, "audit_write"))
        self.assertIsNotNone(pickle.loads(pickle.dumps(payload)))
