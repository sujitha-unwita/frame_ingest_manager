"""Offline coverage for legacy/new CSV reporting and repetition semantics."""

import csv
import tempfile
import unittest
from pathlib import Path

from utils.timing_report import TimingReport


class TimingReportTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / "timing-example.csv"
        self.header = [
            "timestamp_utc", "process", "pid", "stage", "trigger_id",
            "camera_id", "sequence", "status", "elapsed_ms",
        ]

    def write_rows(self, rows):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.header)
            writer.writerows(rows)

    def row(self, stage="inference", camera="camera_3", status="ok", elapsed="100"):
        return ["2026-10-01T12:00:00+00:00", "vision", "12", stage,
                "1", camera, "1", status, elapsed]

    def test_legacy_rows_keep_cameras_statuses_and_repetitions_separate(self):
        self.write_rows([
            self.row(elapsed="100"), self.row(elapsed="300"),
            self.row(camera="camera_4", elapsed="900"),
            self.row(status="error", elapsed="10"),
        ])
        report = TimingReport(self.root)
        rows = report.read_rows()
        html = report.render(rows)
        self.assertEqual(len(rows), 4)
        self.assertIn("200.000", html)
        self.assertIn("900.000", html)
        self.assertIn("10.000", html)
        self.assertIn("unavailable", html)
        self.assertIn("camera_4", html)

    def test_new_read_measurement_removes_missing_measurement_warning(self):
        self.header += ["stage_label"]
        self.write_rows([self.row("image_read_decode") + ["Read pixels"]])
        report = TimingReport(self.path)
        html = report.render(report.read_rows())
        self.assertNotIn("unavailable in these older logs", html)
        self.assertIn("cv2.imread", html)

    def test_report_escapes_data_and_preserves_source(self):
        self.write_rows([self.row(camera="<script>alert(1)</script>")])
        before = self.path.read_bytes()
        output = self.root / "report.html"
        report = TimingReport(self.path)
        report.write(output)
        html = output.read_text(encoding="utf-8")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertEqual(before, self.path.read_bytes())
        with self.assertRaises(FileExistsError):
            report.write(output)

    def test_invalid_durations_are_not_silently_averaged(self):
        for value in ("nan", "inf", "-1", "broken"):
            with self.subTest(value=value):
                self.write_rows([self.row(elapsed=value)])
                with self.assertRaises(ValueError):
                    TimingReport(self.path).read_rows()

    def test_empty_input_or_missing_trigger_is_reported(self):
        with self.assertRaises(ValueError):
            TimingReport(self.root).read_rows()
        self.write_rows([self.row()])
        with self.assertRaises(ValueError):
            TimingReport(self.path, "999").read_rows()

    def test_missing_columns_and_partial_rows_are_rejected(self):
        self.header = ["stage"]
        self.write_rows([["inference"]])
        with self.assertRaises(ValueError):
            TimingReport(self.path).read_rows()

    def test_unknown_stage_remains_visible(self):
        self.write_rows([self.row(stage="future_stage")])
        report = TimingReport(self.path)
        html = report.render(report.read_rows())
        self.assertIn("future_stage", html)
        self.assertIn("unclassified", html)

    def test_event_types_commands_and_ids_are_visible_and_separate(self):
        self.header += ["event_type", "event_id", "stream_id", "command_name"]
        self.write_rows([
            self.row("websocket_broadcast", elapsed="100")
            + ["system_status", "", "", ""],
            self.row("websocket_broadcast", elapsed="900")
            + ["defect_event", "42", "stream-example", ""],
            self.row("operator_command", elapsed="20")
            + ["operator_start_capture", "", "", "resume"],
            self.row("operator_command", elapsed="40")
            + ["operator_resume_request", "", "", "resume"],
        ])
        report = TimingReport(self.path)
        html = report.render(report.read_rows())
        self.assertIn("system_status", html)
        self.assertIn("defect_event", html)
        self.assertIn("<td>42</td>", html)
        self.assertIn("stream-example", html)
        self.assertIn("operator_start_capture", html)
        self.assertIn("<td>resume</td>", html)
        self.assertNotIn("500.000", html)
        self.assertIn("100.000", html)
        self.assertIn("900.000", html)
        self.assertNotIn("did not record event identity", html)

    def test_old_panel_rows_explicitly_report_missing_identity(self):
        self.write_rows([self.row("panel_queue_wait")])
        report = TimingReport(self.path)
        html = report.render(report.read_rows())
        self.assertIn("did not record event identity", html)

    def test_event_identity_is_html_escaped(self):
        self.header += ["event_type", "event_id", "stream_id", "command_name"]
        self.write_rows([
            self.row("websocket_broadcast")
            + ["<script>event</script>", "42", "<b>stream</b>", "<i>command</i>"],
        ])
        report = TimingReport(self.path)
        html = report.render(report.read_rows())
        self.assertNotIn("<script>", html)
        self.assertNotIn("<b>stream</b>", html)
        self.assertIn("&lt;script&gt;", html)
