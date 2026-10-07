"""Strict Pydantic models for every top-level ``config.yaml`` section."""

from __future__ import annotations

from control_service.protocol import CrcConfig


import struct

from pathlib import Path
from string import Formatter
from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, FilePath, PrivateAttr, StrictBool,
    model_validator,
)

from constants import ByteSize, LogLevel, Parity, TransportKind
from config.audit_models import AuditWrite
from frame_ingest_manager.models.mvs_config import MvsSettings


class ConfigModel(BaseModel):
    """Base model: reject unknown YAML keys and retain read-only mapping access."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def __getitem__(self, key: str) -> object:
        """Support existing ``section[\"key\"]`` reads during migration."""
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        """Return a field or a caller-supplied default."""
        return getattr(self, key, default)

    def keys(self) -> tuple[str, ...]:
        """Expose field names for legacy ``{**model}`` test fixtures."""
        return tuple(type(self).model_fields)


class Transport(ConfigModel):
    """Choose the sole source of MCU bytes."""
    kind: TransportKind

    @property
    def is_mock(self) -> bool:
        """Whether the configured transport is the simulator."""
        return self.kind is TransportKind.MOCK


class Reconnect(ConfigModel):
    """Serial reconnection timing."""
    enabled: StrictBool
    initial_delay_s: float = Field(gt=0)
    max_delay_s: float = Field(gt=0)
    backoff_factor: float = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "Reconnect":
        """Ensure the retry range can grow rather than shrink."""
        if self.max_delay_s < self.initial_delay_s:
            raise ValueError("max_delay_s must be >= initial_delay_s")
        return self


class Serial(ConfigModel):
    """Physical RS-422 settings."""
    port: str = Field(min_length=1)
    baud: int = Field(gt=0)
    bytesize: ByteSize
    parity: Parity
    stopbits: Literal[1, 1.5, 2]
    read_timeout_s: float = Field(gt=0)
    write_timeout_s: float = Field(gt=0)
    exclusive: StrictBool
    reconnect: Reconnect
    expected_latency_timer_ms: int = Field(gt=0)


class Crc(ConfigModel):
    """CRC-8 parameters from the MCU wire contract."""
    poly: int = Field(ge=0, le=255)
    init: int = Field(ge=0, le=255)
    xorout: int = Field(ge=0, le=255)
    include_stx: StrictBool


class Protocol(ConfigModel):
    """MCU framing and parser limits."""
    stx: int = Field(ge=0, le=255)
    etx: int = Field(ge=0, le=255)
    message_types: dict[str, int]
    payload_lengths: dict[str, int]
    status_flags: dict[str, int]
    crc: Crc
    max_payload: int = Field(gt=0, le=255)

    @property
    def crc_config(self) -> CrcConfig:
        """Build the existing frame-parser CRC value object."""
        return CrcConfig(self.crc.poly, self.crc.init, self.crc.xorout, self.crc.include_stx)

    @property
    def payload_lengths_by_code(self) -> dict[int, int]:
        """Map configured message codes to configured payload lengths."""
        return {
            self.message_types[name]: length
            for name, length in self.payload_lengths.items()
        }

    @property
    def status_flag_names(self) -> dict[int, str]:
        """Map configured bit positions to their readable flag names."""
        return {bit: name for name, bit in self.status_flags.items()}

    @model_validator(mode="after")
    def validate_payload_limit(self) -> "Protocol":
        """Reject ambiguous codes and lengths incompatible with our decoder."""
        formats = {
            "telemetry": "<IiiIbBB",
            "heartbeat": "<IBBH",
            "command": "<BB",
            "ack_nack": "<BBBB",
        }
        if set(self.message_types) != set(formats):
            raise ValueError("message_types must name the four supported messages")
        if self.payload_lengths.keys() != self.message_types.keys():
            raise ValueError("payload_lengths must match message_types names")

        codes = list(self.message_types.values())
        if any(not 0 <= code <= 255 for code in codes):
            raise ValueError("Message type codes must fit in one byte")
        if len(codes) != len(set(codes)):
            raise ValueError("Message type codes must be unique")

        for name, size in self.payload_lengths.items():
            if not 0 < size <= self.max_payload:
                raise ValueError("Payload lengths must be positive and fit max_payload")
            if size != struct.calcsize(formats[name]):
                raise ValueError(
                    f"payload_lengths.{name} does not match the decoder layout"
                )

        bits = list(self.status_flags.values())
        required_flags = {
            "strobe_fault", "encoder_fault", "overcurrent_trip", "panel_estop",
        }
        if set(self.status_flags) != required_flags:
            raise ValueError("status_flags must name the four supported flags")
        if any(not 0 <= bit <= 7 for bit in bits):
            raise ValueError("Status flag positions must be between 0 and 7")
        if len(bits) != len(set(bits)):
            raise ValueError("Status flag positions must be unique")
        return self


class Encoder(ConfigModel):
    """Encoder calibration."""
    mm_per_count: float = Field(gt=0)
    counts_per_trigger: int = Field(gt=0)


class McuStates(ConfigModel):
    """Firmware state-code mapping."""
    idle: int = Field(ge=0, le=255)
    running: int = Field(ge=0, le=255)
    paused: int = Field(ge=0, le=255)
    fault: int = Field(ge=0, le=255)
    armed: int = Field(ge=0, le=255)

    @property
    def names(self) -> dict[int, str]:
        """Return display names keyed by configured firmware code."""
        return {self.idle: "IDLE", self.running: "RUNNING", self.paused: "PAUSED", self.fault: "FAULT", self.armed: "ARMED"}

    @model_validator(mode="after")
    def validate_unique_codes(self) -> "McuStates":
        """Reject ambiguous reverse state lookups."""
        codes = (self.idle, self.running, self.paused, self.fault, self.armed)
        if len(set(codes)) != len(codes):
            raise ValueError("MCU state codes must be distinct")
        return self


class Heartbeat(ConfigModel):
    """Heartbeat and link-health settings."""
    interval_s: float = Field(gt=0)
    bands: list[tuple[float, Literal["LINK_UP", "LINK_DEGRADED", "LINK_DOWN"]]]
    check_interval_s: float = Field(gt=0)
    debounce_samples: int = Field(gt=0)
    debounce_exempt_down: StrictBool
    drift_min_samples: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bands(self) -> "Heartbeat":
        """Protect first-match band selection and polling cadence."""
        bounds = [bound for bound, _ in self.bands]
        if not bounds or bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
            raise ValueError("bands bounds must strictly increase")
        if self.check_interval_s > self.interval_s:
            raise ValueError("check_interval_s must not exceed interval_s")
        return self


class Commands(ConfigModel):
    """MCU operator-command configuration."""
    ack_timeout_s: float = Field(gt=0)
    max_attempts: int = Field(gt=0)
    ack_result_ok: int = Field(ge=0, le=255)
    codes: dict[str, int] = Field(min_length=1)

    @property
    def names(self) -> dict[int, str]:
        """Return command display names derived from configured codes."""
        return {code: name.upper() for name, code in self.codes.items()}

    @model_validator(mode="after")
    def validate_wire_codes(self) -> "Commands":
        """Require supported command names and distinct one-byte codes."""
        if set(self.codes) != {"pause", "resume", "stop"}:
            raise ValueError("codes must contain pause, resume and stop")
        codes = list(self.codes.values())
        if any(not 0 <= code <= 255 for code in codes):
            raise ValueError("Command codes must fit in one byte")
        if len(codes) != len(set(codes)):
            raise ValueError("Command codes must be unique")
        return self


class Fallback(ConfigModel):
    """HTTP dashboard-fallback setting."""
    available: StrictBool


class EventsStore(ConfigModel):
    """Single-panel event replay capacity."""
    max_pending_events: int = Field(gt=0)


class Dashboard(ConfigModel):
    """Dashboard server and delivery settings."""
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    ws_path: str = Field(min_length=1)
    status_push_interval_s: float = Field(gt=0)
    client_status_timeout_s: float = Field(gt=0)
    client_status_check_interval_s: float = Field(gt=0)
    client_reconnect_delay_s: float = Field(gt=0)
    fallback: Fallback
    events_store: EventsStore
    enable_health_endpoint: StrictBool

    @property
    def ws_url(self) -> str:
        """Return the local panel WebSocket URL."""
        return f"ws://127.0.0.1:{self.port}{self.ws_path}"

    @model_validator(mode="after")
    def validate_status_timeout(self) -> "Dashboard":
        """Prevent a panel becoming stale between healthy status pushes."""
        if self.client_status_timeout_s <= self.status_push_interval_s:
            raise ValueError("client_status_timeout_s must exceed status_push_interval_s")
        return self


class Simulator(ConfigModel):
    """Mock MCU configuration."""
    trigger_interval_s: float = Field(gt=0)
    trigger_spacing_mm: float = Field(gt=0)
    encoder_counts_per_trigger: int = Field(gt=0)
    initial_uptime_s: int = Field(ge=0)
    initial_state: Literal["armed", "running", "idle", "paused", "fault"]
    drop_acks: int = Field(ge=0)
    drop_triggers: int = Field(ge=0)


class GpuMonitoring(ConfigModel):
    """Deployment settings for Jetson GPU sampling."""

    enabled: StrictBool
    executable: str = Field(min_length=1)
    sample_interval_ms: int = Field(gt=0)
    stop_timeout_s: float = Field(gt=0, allow_inf_nan=False)


class InferenceLayout(ConfigModel):
    """Physical camera arrangement used for coordinate mapping."""

    horizontal_overlap_ratio: float = Field(ge=0, lt=1)
    vertical_overlap_ratio: float = Field(ge=0, lt=1)
    mm_per_pixel_x: float = Field(gt=0, allow_inf_nan=False)
    mm_per_pixel_y: float = Field(gt=0)
    origin: Literal["top_left", "top_right"]
    camera_position_indices: dict[str, int] = Field(min_length=1)
    frame_width_px: int = Field(gt=0)
    left_stencil_mm: float = Field(ge=0, allow_inf_nan=False)
    right_stencil_mm: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_inspection_area(self) -> "InferenceLayout":
        """Require physical camera slots and a nonempty unmasked fabric area."""
        positions = list(self.camera_position_indices.values())
        if any(index < 0 for index in positions):
            raise ValueError("camera position indices must be nonnegative")
        if len(set(positions)) != len(positions):
            raise ValueError("camera position indices must be unique")
        return self


class Deduplication(ConfigModel):
    """Thresholds used while merging raw detections into physical defects."""

    ios_threshold: float = Field(gt=0, le=1)
    match_metric: Literal["IOS", "IOU"]
    horizontal_gap_px: float = Field(ge=0)
    vertical_gap_px: float = Field(ge=0)
    min_axis_overlap_ratio: float = Field(gt=0, le=1)


class FourPointGrading(ConfigModel):
    """Project poster policy, applied to a settled defect's longest dimension."""

    mm_per_inch: float = Field(gt=0, allow_inf_nan=False)
    length_limits_inches: list[float] = Field(min_length=1)
    length_points: list[int] = Field(min_length=1)
    oversize_points: int = Field(ge=1, le=4)
    hole_classes: list[str]
    hole_points: int = Field(ge=1, le=4)
    full_width_points: int = Field(ge=1, le=4)
    full_width_tolerance_mm: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bands(self) -> "FourPointGrading":
        """Require ordered finite bounds and one valid score per bound."""
        limits = self.length_limits_inches
        if any(not 0 < value < float("inf") for value in limits):
            raise ValueError("length limits must be positive and finite")
        if limits != sorted(set(limits)):
            raise ValueError("length limits must strictly increase")
        if len(limits) != len(self.length_points):
            raise ValueError("each length limit needs one point value")
        if any(point < 1 or point > 4 for point in self.length_points):
            raise ValueError("length points must be between 1 and 4")
        if any(not label.strip() for label in self.hole_classes):
            raise ValueError("hole class labels must not be empty")
        return self


class Inference(ConfigModel):
    """Model runtime, mapping, merging, and roll settings."""

    engine_kind: Literal["sahi"]
    model_path: FilePath
    model_type: Literal["ultralytics"]
    device_preference: Literal["auto", "cuda:0", "cpu"]
    confidence_threshold: float = Field(ge=0, le=1)
    tile_size_px: int = Field(gt=0)
    tile_overlap_ratio: float = Field(ge=0, lt=1)
    tile_batch_size: int = Field(gt=0)
    sahi_postprocess_type: Literal["NMM", "NMS"]
    sahi_match_metric: Literal["IOS", "IOU"]
    postprocess_class_agnostic: StrictBool
    sahi_match_threshold: float = Field(gt=0, le=1)
    input_channels: Literal[1, 3]
    normalization_divisor: float = Field(ge=255, allow_inf_nan=False)
    padding_value: int = Field(ge=0, le=255)
    prediction_iou_threshold: float = Field(gt=0, le=1)
    prediction_max_detections: int = Field(gt=0)
    prediction_class_agnostic: StrictBool
    prediction_verbose: StrictBool
    layout: InferenceLayout
    deduplication: Deduplication
    grading: FourPointGrading

class Logging(ConfigModel):
    """Application logging configuration."""
    timing_enabled: StrictBool
    timing_file: str = Field(min_length=1)
    level: LogLevel
    quiet_loggers: dict[str, LogLevel]
    file: str | None
    file_mode: Literal["a", "w", "x"]
    file_encoding: str = Field(min_length=1)
    filename_datefmt: str = Field(min_length=1)
    format: str = Field(min_length=1)
    datefmt: str = Field(min_length=1)
    counter_report_interval_s: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_timing_file(self) -> "Logging":
        """Require separate filenames for each process and application run."""
        fields = {
            name for _, name, _, _ in Formatter().parse(self.timing_file)
            if name is not None
        }
        if fields != {"timestamp", "process", "pid"}:
            raise ValueError(
                "timing_file must contain {timestamp}, {process}, {pid} only"
            )
        # Reject invalid formatting before starting any worker.
        self.timing_file.format(timestamp="run", process="vision", pid=1)
        return self


class FrameIngest(ConfigModel):
    """Frame-source selection and one-frame-per-camera bundle contract."""
    source_kind: Literal["folder", "mvs"]
    folder_path: str = ""
    mvs: MvsSettings | None = None
    camera_ids: list[str] = Field(min_length=1)
    expected_camera_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_camera_set(self) -> "FrameIngest":
        """Require unique IDs and a matching expected count."""
        if self.source_kind == "folder" and not self.folder_path:
            raise ValueError("folder source requires folder_path")
        if self.source_kind == "mvs":
            if self.mvs is None:
                raise ValueError("mvs source requires frame_ingest.mvs settings")
            if set(self.camera_ids) != {camera.id for camera in self.mvs.cameras}:
                raise ValueError("mvs camera IDs must equal frame_ingest.camera_ids")
        if any(not camera_id.strip() for camera_id in self.camera_ids):
            raise ValueError("camera IDs cannot be blank")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must not contain duplicates")
        if self.expected_camera_count != len(self.camera_ids):
            raise ValueError("expected_camera_count must equal the number of camera_ids")
        return self


class Processing(ConfigModel):
    """Bounded process queues and operational deadlines, all configured in YAML."""

    model_config = ConfigDict(allow_inf_nan=False)

    trigger_queue_capacity: int = Field(gt=0)
    audit_queue_capacity: int = Field(gt=0)
    command_queue_capacity: int = Field(gt=0)
    event_queue_capacity: int = Field(gt=0)
    panel_queue_capacity: int = Field(gt=0)
    poll_interval_s: float = Field(gt=0)
    queue_put_timeout_s: float = Field(gt=0)
    startup_timeout_s: float = Field(gt=0)
    roll_timeout_s: float = Field(gt=0)
    health_timeout_s: float = Field(gt=0)
    vision_job_timeout_s: float = Field(gt=0)
    audit_job_timeout_s: float = Field(gt=0)
    shutdown_timeout_s: float = Field(gt=0)
    terminate_timeout_s: float = Field(gt=0)
    panel_send_timeout_s: float = Field(gt=0)


class AppConfig(ConfigModel):
    """Complete application configuration validated from one YAML document."""
    transport: Transport
    serial: Serial
    protocol: Protocol
    encoder: Encoder
    mcu_states: McuStates
    heartbeat: Heartbeat
    commands: Commands
    dashboard: Dashboard
    simulator: Simulator
    gpu_monitoring: GpuMonitoring
    inference: Inference
    audit_write: AuditWrite
    logging: Logging
    frame_ingest: FrameIngest
    processing: Processing

    _source_paths: tuple[Path, ...] = PrivateAttr(default=())

    @property
    def source_paths(self) -> tuple[Path, ...]:
        """Return YAML source files used by the loader."""
        return self._source_paths

    def set_source_paths(self, paths: tuple[Path, ...]) -> None:
        """Record provenance without accepting it from YAML."""
        self._source_paths = paths


__all__ = ["AppConfig"]
