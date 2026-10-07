"""Shapes for the ``trigger`` table.

One physical trigger fires every camera; each camera capture is one row.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator

from audit_write.models._common import RowModel, WriteModel, check_camera_id


class TriggerCreate(WriteModel):
    trigger_primary_id: UUID | None = None
    roll_id: UUID
    camera_id: str
    trigger_sequence: int = Field(ge=0)
    encoder_count: int
    captured_at: datetime | None = None

    v_camera = field_validator("camera_id")(check_camera_id)


class TriggerUpdate(WriteModel):
    encoder_count: int | None = None
    captured_at: datetime | None = None


class Trigger(RowModel):
    trigger_primary_id: UUID
    roll_id: UUID
    camera_id: str
    trigger_sequence: int
    encoder_count: int
    captured_at: datetime
