"""Validated trigger metadata received from the MCU-control boundary."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TriggerTelemetry(BaseModel):
    """The telemetry fields needed to associate frames with one trigger."""

    model_config = ConfigDict(extra="forbid")

    trigger_id: int = Field(ge=0)
    encoder_count: int = Field(ge=0)
    timestamp_us: int = Field(ge=0)


__all__ = ["TriggerTelemetry"]
