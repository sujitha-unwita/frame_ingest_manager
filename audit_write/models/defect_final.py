"""Shapes for the ``defect_final`` table: the operator-facing defect after deduplication."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator

from audit_write.models._common import RowModel, WriteModel, check_confidence, check_defect_class


class DefectFinalCreate(WriteModel):
    defect_id: UUID | None = None
    roll_id: UUID
    defect_no: int = Field(ge=1)
    defect_class: str
    confidence: float | None = None
    world_x1_mm: float | None = None
    world_y1_mm: float | None = None
    world_x2_mm: float | None = None
    world_y2_mm: float | None = None
    defect_area_mm2: float | None = Field(default=None, ge=0)
    four_point_score: float | None = None

    v_class = field_validator("defect_class")(check_defect_class)
    v_confidence = field_validator("confidence")(check_confidence)


class DefectFinalUpdate(WriteModel):
    defect_class: str | None = None
    confidence: float | None = None
    world_x1_mm: float | None = None
    world_y1_mm: float | None = None
    world_x2_mm: float | None = None
    world_y2_mm: float | None = None
    defect_area_mm2: float | None = Field(default=None, ge=0)
    four_point_score: float | None = None

    v_class = field_validator("defect_class")(check_defect_class)
    v_confidence = field_validator("confidence")(check_confidence)


class DefectFinal(RowModel):
    defect_id: UUID
    roll_id: UUID
    defect_no: int
    defect_class: str
    confidence: float | None = None
    world_x1_mm: float | None = None
    world_y1_mm: float | None = None
    world_x2_mm: float | None = None
    world_y2_mm: float | None = None
    defect_area_mm2: float | None = None
    four_point_score: float | None = None
    inserted_at: datetime
