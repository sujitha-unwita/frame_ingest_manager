"""Unnumbered merger output, before final-defect persistence."""

from uuid import UUID

from pydantic import Field

from inference_pipeline.models.contracts import BoundingBoxPx, InferenceModel


class DefectGroup(InferenceModel):
    """One merged region and the persisted raw detections contributing to it.

    The same shape represents an open group or a newly settled group. Only
    AuditWriter assigns a defect number after the merger releases it.
    """

    defect_class: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    global_bounding_box_px: BoundingBoxPx
    member_detection_ids: tuple[UUID, ...] = Field(min_length=1)
    world_x1_mm: float
    world_y1_mm: float
    world_x2_mm: float
    world_y2_mm: float
    # Open groups are ungraded. Deduplication fills this only on settlement.
    four_point_score: int | None = Field(default=None, ge=1, le=4)
