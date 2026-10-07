"""Common lifecycle and ownership contract for camera and folder sources."""
from typing import Protocol
from frame_ingest_manager.models.camera_frame import CameraFrame
from frame_ingest_manager.models.trigger_telemetry import TriggerTelemetry

class FrameSource(Protocol):
    def start(self) -> None:
        """Open and arm the source before worker readiness."""

    def stop(self) -> None:
        """Stop acquisition and release source-owned frames."""

    def check_health(self) -> None:
        """Raise if acquisition failed, even while awaiting telemetry."""

    def get_frames(self, trigger: TriggerTelemetry, camera_ids: list[str]) -> dict[str, CameraFrame]:
        """Transfer a complete set; the caller must release its frames."""
