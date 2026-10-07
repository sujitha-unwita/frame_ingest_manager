"""Shapes for the ``defect_live`` table: raw model output before deduplication."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import field_validator

from audit_write.models._common import (
    RowModel,
    WriteModel,
    check_camera_id,
    check_confidence,
    check_defect_class,
    check_frame_variant,
)


class DefectLiveCreate(WriteModel):
    detection_id: UUID | None = None
    roll_id: UUID
    trigger_primary_id: UUID
    encoder_count: int | None = None
    camera_id: str
    frame_variant: str | None = None
    media_id: UUID | None = None
    tile_ids: list[int] | None = None
    defect_class: str
    confidence: float
    bbox_x1_px: float | None = None
    bbox_y1_px: float | None = None
    bbox_x2_px: float | None = None
    bbox_y2_px: float | None = None
    world_x1_mm: float | None = None
    world_y1_mm: float | None = None
    world_x2_mm: float | None = None
    world_y2_mm: float | None = None
    model_version: str | None = None
    detector_meta: dict[str, Any] | None = None
    defect_id: UUID | None = None
    captured_at: datetime | None = None

    v_camera = field_validator("camera_id")(check_camera_id)
    v_variant = field_validator("frame_variant")(check_frame_variant)
    v_class = field_validator("defect_class")(check_defect_class)
    v_confidence = field_validator("confidence")(check_confidence)


class DefectLiveUpdate(WriteModel):
    media_id: UUID | None = None
    defect_id: UUID | None = None
    detector_meta: dict[str, Any] | None = None


class DefectLive(RowModel):
    detection_id: UUID
    roll_id: UUID
    trigger_primary_id: UUID
    encoder_count: int | None = None
    camera_id: str
    frame_variant: str | None = None
    media_id: UUID | None = None
    tile_ids: list[int] | None = None
    defect_class: str
    confidence: float
    bbox_x1_px: float | None = None
    bbox_y1_px: float | None = None
    bbox_x2_px: float | None = None
    bbox_y2_px: float | None = None
    world_x1_mm: float | None = None
    world_y1_mm: float | None = None
    world_x2_mm: float | None = None
    world_y2_mm: float | None = None
    model_version: str | None = None
    detector_meta: dict[str, Any] | None = None
    defect_id: UUID | None = None
    captured_at: datetime
