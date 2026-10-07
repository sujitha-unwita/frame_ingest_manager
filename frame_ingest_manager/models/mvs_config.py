"""MVS configuration without importing SDK or camera resources."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class MvsSdkSettings(Settings):
    mvimport_dir: Path = Path("frame_ingest_manager/frame_sources/mvs/MvImport")
    transport_layers: list[Literal["gige", "usb", "gentl_gige"]] = Field(default_factory=lambda: ["gige"], min_length=1)
    access_mode: Literal["exclusive", "exclusive_with_switch", "control"] = "exclusive"
    finalize_on_stop: bool = True

class MvsCameraSettings(Settings):
    id: str = Field(min_length=1)
    serial_number: str | None = None
    ip: str | None = None
    user_defined_name: str | None = None
    device_index: int | None = Field(default=None, ge=0)
    frame_number_base: int | None = Field(default=None, ge=0, lt=2**32)
    load_user_set: str | None = None
    grab_timeout_ms: int = Field(default=1000, gt=0)
    sdk_buffer_nodes: int = Field(default=3, gt=0)
    packet_size: Literal["auto"] | int = "auto"
    timestamp_tick_hz: float = Field(default=100000000, gt=0)
    convert_to_gray: bool = True
    feature_errors: Literal["fail"] = "fail"
    features: dict[str, str | int | float | bool] = Field(default_factory=dict)
    optional_features: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self):
        identities = (self.serial_number, self.ip, self.user_defined_name, self.device_index)
        if sum(value is not None for value in identities) != 1:
            raise ValueError("each camera requires exactly one device identity")
        if any(isinstance(value, str) and not value.strip() for value in identities):
            raise ValueError("camera identity cannot be blank")
        if isinstance(self.packet_size, int) and self.packet_size <= 0:
            raise ValueError("packet_size must be positive")
        if any(name.startswith("Trigger") for name in self.features):
            raise ValueError("configure Trigger features in mvs.trigger_features")
        if any(name in {"TriggerMode", "TriggerSource", "TriggerSelector", "TriggerActivation"}
               for name in self.optional_features):
            raise ValueError("required trigger controls cannot be optional")
        return self

    def describe_identity(self):
        return str(self.serial_number or self.ip or self.user_defined_name or self.device_index)

class MvsSettings(Settings):
    sdk: MvsSdkSettings = Field(default_factory=MvsSdkSettings)
    cameras: list[MvsCameraSettings] = Field(min_length=1)
    trigger_id_base: int | None = Field(default=None, ge=0, lt=2**32)
    sequence_source: Literal["frame_number", "trigger_index"] = "frame_number"
    slots_per_camera: int = Field(default=16, ge=2)
    max_total_memory_mb: float = Field(default=512, gt=0)
    bundle_timeout_s: float = Field(default=1.0, gt=0)
    trigger_features: dict[str, str | int | float | bool] = Field(default_factory=lambda: {
        "TriggerSelector": "FrameBurstStart", "TriggerMode": "On",
        "TriggerSource": "Line0", "TriggerActivation": "RisingEdge", "TriggerDelay": 0.0,
    })

    @model_validator(mode="after")
    def validate_cameras(self):
        ids = [camera.id for camera in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate camera IDs")
        if self.trigger_features.get("TriggerMode") != "On":
            raise ValueError("hardware capture requires TriggerMode: 'On'")
        source = self.trigger_features.get("TriggerSource", "")
        if not isinstance(source, str) or not source.startswith("Line"):
            raise ValueError("hardware capture requires a physical Line trigger source")
        return self

    def require_alignment(self):
        if self.trigger_id_base is None or any(camera.frame_number_base is None for camera in self.cameras):
            raise ValueError("MVS requires verified trigger_id_base and frame_number_base for every camera; arrival order is not an alignment")
