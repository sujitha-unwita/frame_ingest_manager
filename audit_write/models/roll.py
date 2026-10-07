"""Shapes for the ``roll`` table."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator

from audit_write.models._common import RowModel, WriteModel, check_label, check_not_blank


class RollCreate(WriteModel):
    roll_id: UUID | None = None
    roll_label: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    length_mm: float | None = Field(default=None, ge=0)
    suspect: bool | None = None
    suspect_reason: str | None = None

    v_label = field_validator("roll_label")(check_label)
    v_blank = field_validator("roll_label", "suspect_reason")(check_not_blank)


class RollUpdate(WriteModel):
    roll_label: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    length_mm: float | None = Field(default=None, ge=0)
    suspect: bool | None = None
    suspect_reason: str | None = None

    v_label = field_validator("roll_label")(check_label)
    v_blank = field_validator("roll_label", "suspect_reason")(check_not_blank)


class Roll(RowModel):
    roll_id: UUID
    roll_label: str
    started_at: datetime
    ended_at: datetime | None = None
    length_mm: float | None = None
    suspect: bool
    suspect_reason: str | None = None
