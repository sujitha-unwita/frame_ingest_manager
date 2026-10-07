"""Build a readable offline HTML report from old or enriched timing CSV files.

Run from integrated_code:
    python -m utils.timing_report logs/log_recent/timings --output report.html
"""

import argparse
import csv
import logging
import math
from collections import defaultdict
from html import escape
from pathlib import Path
from statistics import mean

from utils.timing_stages import describe_stage

LOGGER = logging.getLogger(__name__)


class TimingReport:
    """Explain repeated measurements without summing nested stage totals."""

    def __init__(self, input_path: Path, trigger_id: str | None = None) -> None:
        """Keep a source file/folder and an optional single-trigger filter."""
        self.input_path = input_path
        self.trigger_id = trigger_id

    def read_rows(self) -> list[dict]:
        """Load timing files; reject malformed durations rather than hide them."""
        paths = (
            [self.input_path] if self.input_path.is_file()
            else sorted(self.input_path.rglob("timing-*.csv"))
        )
        if not paths:
            raise ValueError(f"No timing CSV files found in {self.input_path}")
        rows = []
        required = {
            "timestamp_utc", "process", "pid", "stage", "trigger_id",
            "camera_id", "sequence", "status", "elapsed_ms",
        }
        for path in paths:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Missing timing columns: {path}")
                for line, row in enumerate(reader, start=2):
                    if None in row or any(row[key] is None for key in required):
                        raise ValueError(f"Malformed timing row: {path}:{line}")
                    try:
                        elapsed = float(row["elapsed_ms"])
                    except ValueError as exc:
                        raise ValueError(f"Invalid duration: {path}:{line}") from exc
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError(f"Invalid duration: {path}:{line}")
                    if self.trigger_id is not None and row["trigger_id"] != self.trigger_id:
                        continue
                    rows.append({
                        **row, "elapsed_ms": elapsed,
                        "source": str(path.resolve()), "source_line": line,
                    })
        if not rows:
            raise ValueError("No measurements match the requested trigger")
        return rows

    @staticmethod
    def table(headers: list[str], rows: list[list]) -> str:
        """Escape log values so a report cannot execute content from a CSV."""
        heading = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(str(item))}</td>" for item in row) + "</tr>"
            for row in rows
        )
        return f"<div class='scroll'><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"

    def render(self, rows: list[dict]) -> str:
        """Show per-process summaries and expandable trigger details."""
        sections = [
            "<!doctype html><html lang='en'><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>VisionBar timing report</title>",
            "<style>body{font:16px system-ui,sans-serif;margin:32px;color:#172b43;"
            "background:#f6f8fb}h1,h2{color:#103c62}section,details,.note{"
            "background:white;padding:20px;margin:16px 0;border:1px solid #d7e1eb;"
            "border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}"
            "td,th{padding:10px;border:1px solid #d7e1eb;text-align:left;vertical-align:top}"
            "th{background:#e7f1f9}tr:nth-child(even){background:#f7fafc}"
            ".scroll{overflow-x:auto}.warning{border-left:5px solid #b45c00}"
            "summary{cursor:pointer;font-weight:600}small{overflow-wrap:anywhere}"
            "</style><h1>VisionBar — timing explained</h1>",
            "<div class='note'><strong>How to read this report</strong><ul>"
            "<li>All durations are milliseconds (1,000 ms = 1 second).</li>"
            "<li>The same stage repeats for each trigger, camera or dashboard message; "
            "it is not automatically a duplicate.</li>"
            "<li>Totals include child measurements. Never add inference to image_read_decode, "
            "frame_audit to S3/database timings, or trigger_to_audit to stage totals.</li>"
            "<li>Waits are elapsed queue delays, not CPU/GPU processing time. "
            "audit_handoff_wait overlaps audit_queue_put.</li>"
            "<li>Error/cancelled rows are separate from successful rows. "
            "An ok row means that measured block returned, not that every job was saved.</li>"
            "<li>UTC timestamps mark completion of each measurement; nested children "
            "can appear before their parent total.</li>"
            "<li>Files are kept separate so different runs are not accidentally combined. "
            "A missing measurement means unknown, not zero.</li>"
            "<li>Event/request identifies what is being sent or requested. "
            "A system_status trigger ID is the latest observed trigger, not a new capture.</li>"
            "<li>The same stream ID + event ID on wait and broadcast rows identifies "
            "one discrete message measured twice. Event IDs are not defect database IDs. "
            "Status snapshots have no event ID; blank identity is not an error.</li>"
            "<li>Command shows the MCU action; the operator request distinguishes "
            "Start Capture from Resume even when both send resume.</li></ul></div>",
        ]
        if not any(row["stage"] == "image_read_decode" for row in rows):
            sections.append(
                "<div class='note warning'><strong>Inference image-read time is unavailable "
                "in these older logs.</strong> frame_bundle only selects references. "
                "frame_read measures the audit process reading bytes for S3. "
                "Run the updated application to capture image_read_decode separately.</div>"
            )
        sections.append("<details><summary>Stage guide — what each measurement means</summary>")
        sections.append(self.table(
            ["CSV stage", "Operation / code owner", "Kind", "Included in", "Meaning"],
            [[stage, f"{describe_stage(stage).label} / {describe_stage(stage).component}",
              describe_stage(stage).kind, describe_stage(stage).parent or "—",
              describe_stage(stage).description]
             for stage in sorted({row["stage"] for row in rows})],
        ))
        sections.append("</details>")
        sources = defaultdict(list)
        for row in rows:
            sources[row["source"]].append(row)
        for source, source_rows in sorted(sources.items()):
            sections.append(
                f"<section><h2>{escape(Path(source).name)}</h2>"
                f"<small>Source: {escape(source)}</small>"
                f"<p>{len(source_rows)} measurements. "
                "Summary split by stage, event/request, command, camera and status.</p>"
            )
            panel_rows = [
                row for row in source_rows
                if row["stage"] in {"panel_queue_wait", "websocket_broadcast"}
            ]
            if any("event_type" not in row for row in panel_rows):
                sections.append(
                    "<p class='warning'>These older dashboard timing rows did not "
                    "record event identity. Their event types cannot be recovered "
                    "from this CSV; no IDs or event types have been invented.</p>"
                )
            groups = defaultdict(list)
            for row in source_rows:
                key = (
                    row["process"], row["pid"], row["stage"], row["camera_id"],
                    row["status"], row.get("event_type", ""),
                    row.get("command_name", ""),
                )
                groups[key].append(row["elapsed_ms"])
            summary = []
            for (process, pid, stage, camera, status, event, command), values in groups.items():
                info = describe_stage(stage)
                summary.append([
                    info.label, event or "—", command or "—", camera or "—", status,
                    len(values), f"{mean(values):.3f}", f"{min(values):.3f}",
                    f"{max(values):.3f}", info.kind,
                ])
            sections.append(self.table(
                ["Operation", "Event / request", "Command", "Camera", "Status", "Calls", "Average ms",
                 "Minimum ms", "Maximum ms", "Kind"],
                summary,
            ))
            triggers = defaultdict(list)
            for row in source_rows:
                triggers[row["trigger_id"] or "No trigger / lifecycle or dashboard"].append(row)
            for trigger, records in triggers.items():
                sections.append(
                    f"<details><summary>Trigger {escape(trigger)} — "
                    f"{len(records)} measurements</summary>"
                )
                sections.append(self.table(
                    ["Completed at (UTC)", "Operation", "Event / request",
                     "Event ID", "Stream ID", "Command", "Camera", "Sequence",
                     "Status", "Duration ms", "CSV line"],
                    [[row["timestamp_utc"], describe_stage(row["stage"]).label,
                      row.get("event_type") or "—", row.get("event_id") or "—",
                      row.get("stream_id") or "—", row.get("command_name") or "—",
                      row["camera_id"] or "—", row["sequence"] or "—",
                      row["status"], f'{row["elapsed_ms"]:.3f}', row["source_line"]]
                     for row in sorted(records, key=lambda item: item["timestamp_utc"])],
                ))
                sections.append("</details>")
            sections.append("</section>")
        return "".join(sections) + "</html>"

    def write(self, output_path: Path) -> None:
        """Create a new report, preserving existing reports and all raw CSVs."""
        rows = self.read_rows()
        report = self.render(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(report)
        LOGGER.info("Timing report: %s (%s measurements)", output_path, len(rows))


def main() -> None:
    """Run without loading application settings, GPU, database or hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Timing CSV file or folder")
    parser.add_argument("--output", type=Path, required=True, help="New HTML report")
    parser.add_argument("--trigger-id", help="Show only this trigger")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        TimingReport(args.input, args.trigger_id).write(args.output)
    except (OSError, ValueError, csv.Error):
        LOGGER.exception("Could not create timing report")
        raise


if __name__ == "__main__":
    main()
