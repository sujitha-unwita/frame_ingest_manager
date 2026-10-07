"""CSV timing tests without hardware or optional application dependencies."""

import asyncio
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils.timing import Timing


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "timings" / "test.csv"
        self.timer = Timing(True, self.path)

    def rows(self):
        with self.path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_elapsed_milliseconds_and_context(self):
        with patch("utils.timing.time.monotonic", side_effect=[10.0, 10.125]):
            with self.timer.measure("inference", trigger_id=7, camera_id="camera_3"):
                pass
        row = self.rows()[0]
        self.assertEqual(row["stage"], "inference")
        self.assertEqual(row["trigger_id"], "7")
        self.assertEqual(row["camera_id"], "camera_3")
        self.assertEqual(row["elapsed_ms"], "125.000")
        self.assertEqual(row["status"], "ok")
        self.assertTrue(row["timestamp_utc"].endswith("+00:00"))
        self.assertTrue(row["process"])
        self.assertTrue(row["pid"])
        self.assertEqual(row["stage_label"], "Complete detection for one camera")
        self.assertEqual(row["measurement_kind"], "total")
        self.assertEqual(row["included_in"], "vision_total")

    def test_appends_rows_with_one_header_and_csv_quoting(self):
        self.timer.record('stage,with "quotes"', 0)
        self.timer.record("second", 0)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["stage"], 'stage,with "quotes"')
        self.assertEqual(rows[1]["stage"], "second")
        self.assertEqual(rows[1]["trigger_id"], "")

    def test_original_error_propagates_with_failed_timing(self):
        failure = ValueError("original failure")
        with self.assertRaises(ValueError) as caught:
            with self.timer.measure("upload"):
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.rows()[0]["status"], "error")

    def test_disabled_timer_creates_no_file_and_reads_no_clock(self):
        with patch("utils.timing.time.monotonic") as clock:
            timer = Timing(False, self.path)
            with timer.measure("disabled"):
                pass
            self.assertIsNone(timer.record("disabled", 0))
            clock.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_disabled_timer_still_propagates_error(self):
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with Timing(False).measure("disabled"):
                raise RuntimeError("failed")

    def test_existing_timestamp_measures_queue_wait(self):
        with patch("utils.timing.time.monotonic", return_value=20.5):
            elapsed = self.timer.record("queue_wait", 20, sequence=3)
        self.assertEqual(elapsed, 500)
        self.assertEqual(self.rows()[0]["sequence"], "3")

    def test_end_to_end_uses_admission_timestamp(self):
        with patch("utils.timing.time.monotonic", return_value=22):
            with self.timer.measure("trigger_to_audit", started_at=20):
                pass
        self.assertEqual(self.rows()[0]["elapsed_ms"], "2000.000")

    def test_existing_file_is_not_overwritten(self):
        self.timer.record("original", 0)
        second = Timing(True, self.path)
        with self.assertLogs("utils.timing", "ERROR"):
            second.record("replacement", 0)
        self.assertEqual([row["stage"] for row in self.rows()], ["original"])

    def test_write_failure_does_not_mask_operation_error(self):
        failure = ValueError("inference failed")
        with patch("pathlib.Path.open", side_effect=PermissionError("denied")):
            with self.assertLogs("utils.timing", "ERROR"):
                with self.assertRaises(ValueError) as caught:
                    with self.timer.measure("inference"):
                        raise failure
        self.assertIs(caught.exception, failure)
        with self.assertNoLogs("utils.timing"):
            self.timer.record("later", 0)

    def test_csv_records_do_not_depend_on_info_log_level(self):
        with patch("utils.timing.LOGGER.info") as log:
            self.timer.record("first", 0)
            self.timer.record("second", 0)
        # Only announce the filename, not each timing row.
        self.assertEqual(log.call_count, 1)
        self.assertEqual(len(self.rows()), 2)

    def test_config_paths_are_separate_for_each_process(self):
        config = SimpleNamespace(
            timing_enabled=True,
            timing_file="timings/{timestamp}-{process}-{pid}.csv",
            filename_datefmt="%Y%m%d-%H%M%S-%f", file_encoding="utf-8",
        )
        root = Path(self.directory.name)
        timers = [Timing.from_config(config, root, role)
                  for role in ("control", "vision", "audit")]
        self.assertEqual(len({timer.output_path for timer in timers}), 3)
        for timer in timers:
            self.assertTrue(timer.output_path.is_relative_to(root))

    def test_enabled_timer_requires_destination(self):
        with self.assertRaises(ValueError):
            Timing(True)

    def test_event_identity_survives_record_and_measure(self):
        context = {
            "trigger_id": 10, "camera_id": "camera_3",
            "event_type": "defect_event", "event_id": 42,
            "stream_id": "stream-a", "command_name": None,
        }
        self.timer.record("panel_queue_wait", 0, **context)
        with self.timer.measure("websocket_broadcast", **context):
            pass
        for row in self.rows():
            self.assertEqual(row["event_type"], "defect_event")
            self.assertEqual(row["event_id"], "42")
            self.assertEqual(row["stream_id"], "stream-a")
            self.assertEqual(row["camera_id"], "camera_3")
            self.assertEqual(row["command_name"], "")

    def test_status_snapshot_does_not_invent_event_id(self):
        with self.timer.measure(
            "websocket_broadcast", event_type="system_status", trigger_id=10,
        ):
            pass
        row = self.rows()[0]
        self.assertEqual(row["event_type"], "system_status")
        self.assertEqual(row["event_id"], "")
        self.assertEqual(row["stream_id"], "")

    def test_failed_command_keeps_request_and_command_name(self):
        with self.assertRaises(ValueError):
            with self.timer.measure(
                "operator_command", event_type="operator_start_capture",
                command_name="resume",
            ):
                raise ValueError("send failed")
        row = self.rows()[0]
        self.assertEqual(row["event_type"], "operator_start_capture")
        self.assertEqual(row["command_name"], "resume")
        self.assertEqual(row["status"], "error")


class AsyncTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_await_and_cancellation_are_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "async.csv"
            timer = Timing(True, path)
            with timer.measure("await"):
                await asyncio.sleep(0)
            with self.assertRaises(asyncio.CancelledError):
                with timer.measure("cancelled"):
                    raise asyncio.CancelledError()
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["status"] for row in rows], ["ok", "cancelled"])
