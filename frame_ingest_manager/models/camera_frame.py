"""Validated description of one camera frame."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class CameraFrame(BaseModel):
    """One frame assigned to a camera for a particular trigger.

    For the temporary folder source, ``frame_reference`` is a file path and
    the manager assigns trigger metadata. Live MVS frames carry a leased image
    and separate camera-counter/device-clock metadata until release().
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    frame_reference: str = Field(min_length=1)
    trigger_id: int = Field(ge=0)
    capture_timestamp_us: int = Field(ge=0)
    hardware_trigger_id: int | None = Field(default=None, ge=0)
    camera_frame_number: int | None = Field(default=None, ge=0)
    camera_trigger_index: int | None = Field(default=None, ge=0)
    device_timestamp_ticks: int | None = Field(default=None, ge=0)
    host_timestamp_ms: int | None = None
    pixel_format: str | None = None
    image: Any = Field(default=None, exclude=True, repr=False)
    _lease: Any = PrivateAttr(default=None)

    def release(self) -> None:
        """Return a live buffer exactly once after every consumer is done."""
        self.image = None
        if self._lease is not None:
            self._lease.release()
            self._lease = None


__all__ = ["CameraFrame"]
