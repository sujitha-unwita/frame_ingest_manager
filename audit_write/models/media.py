"""Shapes for the ``media`` table.

``media_ref`` is an S3 object key: never a URL, never an empty string.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import field_validator

from audit_write.models._common import (
    RowModel,
    WriteModel,
    check_camera_id,
    check_frame_variant,
    check_not_blank,
)


class MediaCreate(WriteModel):
    media_id: UUID | None = None
    roll_id: UUID
    trigger_primary_id: UUID
    camera_id: str
    frame_variant: str | None = None
    media_ref: str | None = None
    uploaded_at: datetime | None = None
    captured_at: datetime | None = None
    defect_id: UUID | None = None

    v_camera = field_validator("camera_id")(check_camera_id)
    v_variant = field_validator("frame_variant")(check_frame_variant)
    v_blank = field_validator("frame_variant", "media_ref")(check_not_blank)


class MediaUpdate(WriteModel):
    media_ref: str | None = None
    uploaded_at: datetime | None = None
    defect_id: UUID | None = None

    v_blank = field_validator("media_ref")(check_not_blank)


class Media(RowModel):
    media_id: UUID
    roll_id: UUID
    trigger_primary_id: UUID
    camera_id: str
    frame_variant: str | None = None
    media_ref: str | None = None
    uploaded_at: datetime | None = None
    captured_at: datetime
    defect_id: UUID | None = None
