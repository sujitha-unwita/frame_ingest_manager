"""Shapes for the ``camera`` table."""

from __future__ import annotations

from pydantic import field_validator

from audit_write.models._common import RowModel, WriteModel, check_camera_id, check_not_blank


class CameraCreate(WriteModel):
    camera_id: str
    active: bool | None = None

    v_camera = field_validator("camera_id")(check_camera_id)
    v_blank = field_validator("camera_id")(check_not_blank)


class CameraUpdate(WriteModel):
    active: bool | None = None


class Camera(RowModel):
    camera_id: str
    active: bool
