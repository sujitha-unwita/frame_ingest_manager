"""Control orchestration tests; hardware methods are mocked, not dependencies."""

import asyncio
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from app_orchestrator import AppOrchestrator
from config.config_loader import load_config
from utils.timing import Timing


class AppOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    """Resource processing belongs to workers, not the serial callback."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        model = Path(folder.name) / "model.pt"
        model.touch()
        config = load_config(overrides={
            "audit_write": {"enabled": False},
            "inference": {"model_path": str(model)},
            "frame_ingest": {"folder_path": folder.name},
        })
        self.owner = AppOrchestrator(config)
        self.runtime = Mock(
            start=AsyncMock(), stop=AsyncMock(), prepare_roll=AsyncMock(),
            roll_id="test-roll",
        )
        self.owner._runtime = self.runtime
        self.owner._mcu_link.send_command = AsyncMock(
            return_value=Mock(acknowledged=True),
        )

    async def asyncTearDown(self):
        await self.owner.stop()

    async def test_constructor_does_not_create_parent_model_or_database(self):
        self.assertFalse(hasattr(self.owner, "_detector"))
        self.assertFalse(hasattr(self.owner, "_database"))
        self.assertIs(self.owner._dashboard.link, self.owner.mcu_link)

    async def test_panel_timings_identify_the_message_without_mutating_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "panel.csv"
            self.owner._timing = Timing(True, path)
            self.owner._hub.broadcast = AsyncMock()
            messages = [
                {"event": "system_status", "trigger_id": 10},
                {"event": "defect_event", "trigger_id": 10,
                 "camera_id": "camera_3", "event_id": 42, "stream_id": "stream-a"},
                {"event": "command_failed", "command": "pause",
                 "event_id": 43, "stream_id": "stream-a"},
            ]
            task = asyncio.create_task(self.owner._send_panels())
            try:
                for message in messages:
                    await self.owner._enqueue_panel(message)
                await asyncio.wait_for(self.owner._panel_queue.join(), timeout=2)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            for index, message in enumerate(messages):
                pair = rows[index * 2:index * 2 + 2]
                self.assertEqual(
                    [row["stage"] for row in pair],
                    ["panel_queue_wait", "websocket_broadcast"],
                )
                for row in pair:
                    self.assertEqual(row["event_type"], message["event"])
                    self.assertEqual(row["event_id"], str(message.get("event_id", "")))
                    self.assertEqual(row["stream_id"], message.get("stream_id", ""))
                    self.assertEqual(row["camera_id"], message.get("camera_id", ""))
                    self.assertEqual(row["command_name"], message.get("command", ""))
            self.assertNotIn("event_id", messages[0])
            self.assertEqual(
                [call.args[0] for call in self.owner._hub.broadcast.await_args_list],
                messages,
            )
            self.owner._timing = Timing(False)

    async def test_failed_workers_do_not_start_mcu(self):
        self.runtime.start.side_effect = RuntimeError("worker failed")
        self.owner._mcu_link.run = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            await self.owner.start()
        self.owner._mcu_link.run.assert_not_called()
        self.runtime.stop.assert_awaited()

    async def test_trigger_only_submits_metadata(self):
        self.owner._running = True
        await self.owner.on_trigger({
            "trigger_id": 5, "encoder_count": 10, "timestamp_us": 50,
        })
        self.runtime.submit.assert_called_once_with({
            "trigger_id": 5, "encoder_count": 10, "timestamp_us": 50,
        })

    async def test_trigger_before_start_is_rejected(self):
        with self.assertRaises(RuntimeError):
            await self.owner.on_trigger({"trigger_id": 1})
        self.runtime.submit.assert_not_called()

    async def test_roll_commits_before_resume(self):
        self.owner._running = True
        order = []

        async def prepare():
            order.append("roll")

        async def send(*args, **kwargs):
            order.append("command")
            return Mock(acknowledged=True)

        self.runtime.prepare_roll.side_effect = prepare
        self.owner._mcu_link.send_command.side_effect = send
        await self.owner.handle_operator_command("operator_start_capture")
        self.assertEqual(order, ["roll", "command"])

    async def test_stop_bypasses_slow_roll_and_prevents_delayed_resume(self):
        self.owner._running = True
        entered, release = asyncio.Event(), asyncio.Event()

        async def prepare():
            entered.set()
            await release.wait()

        self.runtime.prepare_roll.side_effect = prepare
        start = asyncio.create_task(
            self.owner.handle_operator_command("operator_start_capture"),
        )
        await entered.wait()
        await self.owner.handle_operator_command("operator_stop_request")
        self.assertEqual(self.owner._mcu_link.send_command.call_args.args[0], "pause")
        release.set()
        with self.assertRaises(RuntimeError):
            await start
        self.assertEqual(self.owner._mcu_link.send_command.await_count, 1)

    async def test_roll_failure_blocks_capture(self):
        self.owner._running = True
        self.runtime.prepare_roll.side_effect = RuntimeError("database failed")
        with self.assertRaises(RuntimeError):
            await self.owner.handle_operator_command("operator_start_capture")
        self.owner._mcu_link.send_command.assert_not_awaited()

    async def test_slow_websocket_is_not_awaited_by_publish(self):
        self.owner._hub.broadcast = AsyncMock()
        await self.owner._enqueue_panel({"type": "test"})
        self.owner._hub.broadcast.assert_not_awaited()
        self.assertEqual(self.owner._panel_queue.qsize(), 1)

    async def test_unavailable_worker_blocks_resume_but_allows_stop(self):
        self.owner._running = True
        self.runtime.require_workers.side_effect = RuntimeError("vision unavailable")
        with self.assertRaises(RuntimeError):
            await self.owner.handle_operator_command("operator_resume_request")
        await self.owner.handle_operator_command("operator_stop_request")
        self.assertEqual(self.owner._mcu_link.send_command.call_args.args[0], "pause")

    async def test_run_serves_existing_app(self):
        with patch("app_orchestrator.uvicorn.run") as serve:
            self.owner.run()
        self.assertIs(serve.call_args.args[0], self.owner._app)

    async def test_enqueue_error_does_not_send_any_mcu_command(self):
        self.owner._running = True
        self.runtime.submit.side_effect = RuntimeError("queue full")
        with self.assertRaises(RuntimeError):
            await self.owner.on_trigger({
                "trigger_id": 1, "encoder_count": 0, "timestamp_us": 0,
            })
        self.owner._mcu_link.send_command.assert_not_awaited()

    async def test_shutdown_does_not_send_any_mcu_command(self):
        self.owner._running = True
        await self.owner.stop()
        self.owner._mcu_link.send_command.assert_not_awaited()
