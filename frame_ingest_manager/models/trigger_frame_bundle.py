"""Validated result of assigning source frames to one MCU trigger."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from frame_ingest_manager.models.camera_frame import CameraFrame
from frame_ingest_manager.models.trigger_telemetry import TriggerTelemetry


class TriggerFrameBundle(BaseModel):
    """One trigger and its frame references, keyed by configured camera ID.

    Folder references are paths. MVS references identify leased in-memory
    images; consumers must release the complete bundle after processing.
    """

    model_config = ConfigDict(extra="forbid")

    trigger: TriggerTelemetry
    camera_frames: dict[str, CameraFrame] = Field(min_length=1)

    @property
    def trigger_id(self) -> int:
        """Expose the trigger ID without duplicating it in this model."""
        return self.trigger.trigger_id

    @model_validator(mode="after")
    def camera_frames_match_keys(self) -> "TriggerFrameBundle":
        """Require each mapping key to equal its frame's camera ID."""
        if any(not camera_id for camera_id in self.camera_frames):
            raise ValueError("camera_frames cannot contain an empty camera ID")
        if any(
            camera_id != frame.camera_id
            for camera_id, frame in self.camera_frames.items()
        ):
            raise ValueError("each camera-frame key must equal frame.camera_id")
        if any(frame.trigger_id != self.trigger.trigger_id for frame in self.camera_frames.values()):
            raise ValueError("every camera frame must belong to the bundle trigger")
        return self

    def release(self) -> None:
        for frame in self.camera_frames.values():
            frame.release()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()
        return False

    @property
    def frame_references(self) -> dict[str, str]:
        """Return references for consumers that do not need frame metadata."""
        return {
            camera_id: frame.frame_reference
            for camera_id, frame in self.camera_frames.items()
        }


__all__ = ["TriggerFrameBundle"]
