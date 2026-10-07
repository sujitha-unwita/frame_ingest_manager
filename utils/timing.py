"""Per-process CSV timings using the host monotonic clock for durations."""

import asyncio
import csv
import logging
import multiprocessing
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from utils.timing_stages import describe_stage

LOGGER = logging.getLogger(__name__)


class Timing:
    """Write stage durations to one CSV without retaining an in-memory history."""

    COLUMNS = (
        "timestamp_utc", "process", "pid", "stage", "trigger_id", "camera_id",
        "sequence", "status", "elapsed_ms", "stage_label", "component",
        "measurement_kind", "included_in", "description",
        "event_type", "event_id", "stream_id", "command_name",
    )

    def __init__(
        self, enabled: bool, output_path: Path | None = None,
        encoding: str = "utf-8",
    ) -> None:
        """Keep destination settings; create the file only on the first record."""
        if enabled and output_path is None:
            raise ValueError("enabled timing requires a CSV output path")
        self.enabled = enabled
        self.output_path = output_path
        self.encoding = encoding
        self._created = False
        self._write_failed = False
        self._lock = Lock()

    @classmethod
    def from_config(cls, logging_config, root: Path, process: str) -> "Timing":
        """Resolve the YAML filename separately inside each owning process."""
        timestamp = datetime.now(timezone.utc).strftime(logging_config.filename_datefmt)
        path = Path(logging_config.timing_file.format(
            timestamp=timestamp, process=process, pid=os.getpid(),
        ))
        if not path.is_absolute():
            path = root / path
        return cls(
            logging_config.timing_enabled, path, logging_config.file_encoding,
        )

    def record(
        self, stage: str, started_at: float, *, status: str = "ok",
        trigger_id: int | None = None, camera_id: str | None = None,
        sequence: int | None = None,
        event_type: str | None = None, event_id: int | None = None,
        stream_id: str | None = None, command_name: str | None = None,
    ) -> float | None:
        """Append one CSV row; durations exclude this row's own file-write time."""
        if not self.enabled:
            return None
        elapsed_ms = (time.monotonic() - started_at) * 1000
        stage_info = describe_stage(stage)
        row = (
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            multiprocessing.current_process().name, os.getpid(), stage,
            trigger_id, camera_id, sequence, status, f"{elapsed_ms:.3f}",
            stage_info.label, stage_info.component, stage_info.kind,
            stage_info.parent, stage_info.description,
            event_type, event_id, stream_id, command_name,
        )
        with self._lock:
            if self._write_failed:
                return elapsed_ms
            try:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation protects earlier runs; later rows append.
                mode = "a" if self._created else "x"
                with self.output_path.open(
                    mode, newline="", encoding=self.encoding,
                ) as stream:
                    writer = csv.writer(stream)
                    if not self._created:
                        writer.writerow(self.COLUMNS)
                    writer.writerow(row)
                if not self._created:
                    LOGGER.info("Timing CSV: %s", self.output_path)
                self._created = True
            except (OSError, csv.Error):
                # Diagnostics must not replace an inference/DB exception or
                # cancellation. Report once rather than flooding logs on full disk.
                self._write_failed = True
                LOGGER.exception(
                    "Timing CSV writes disabled after failure: %s", self.output_path,
                )
        return elapsed_ms

    @contextmanager
    def measure(
        self, stage: str, *, started_at: float | None = None,
        trigger_id: int | None = None, camera_id: str | None = None,
        sequence: int | None = None,
        event_type: str | None = None, event_id: int | None = None,
        stream_id: str | None = None, command_name: str | None = None,
    ) -> Generator[None, None, None]:
        """Record success, failure or cancellation without suppressing exceptions."""
        if not self.enabled:
            yield
            return
        started = time.monotonic() if started_at is None else started_at
        status = "ok"
        try:
            yield
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except BaseException:
            status = "error"
            raise
        finally:
            self.record(
                stage, started, status=status, trigger_id=trigger_id,
                camera_id=camera_id, sequence=sequence,
                event_type=event_type, event_id=event_id,
                stream_id=stream_id, command_name=command_name,
            )
