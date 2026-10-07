"""Validated objects exchanged by the inference pipeline."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InferenceModel(BaseModel):
    """Base contract that rejects fields no pipeline stage owns."""

    model_config = ConfigDict(extra="forbid")


class BoundingBoxPx(InferenceModel):
    """A non-empty rectangle in source-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_extent(self) -> "BoundingBoxPx":
        """Reject a reversed or zero-size rectangle."""
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box x2/y2 must be greater than x1/y1")
        return self


class Detection(InferenceModel):
    """One raw model detection before cross-frame merging."""

    trigger_id: int = Field(ge=0)
    camera_id: str = Field(min_length=1)
    frame_reference: str = Field(min_length=1)
    defect_class: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bounding_box_px: BoundingBoxPx
    frame_width_px: int = Field(gt=0)
    frame_height_px: int = Field(gt=0)
    # SAHI's merged result currently does not preserve contributing tile IDs.
    tile_ids: list[int] | None = None


class MappedDetection(Detection):
    """Raw observation plus the inspected region in calibrated coordinates.

    bounding_box_px remains the original camera-local box. Global pixels and
    world coordinates describe its intersection with the active stencil area.
    X millimetres are relative to the selected active-area edge.
    """

    world_x1_mm: float
    world_y1_mm: float
    world_x2_mm: float
    world_y2_mm: float
    detection_id: UUID | None = None
    global_bounding_box_px: BoundingBoxPx
    clipped_to_active_area: bool = False

    @model_validator(mode="after")
    def validate_world_extent(self) -> "MappedDetection":
        """Reject a reversed or zero-size calibrated rectangle."""
        if (
            self.world_x2_mm <= self.world_x1_mm
            or self.world_y2_mm <= self.world_y1_mm
        ):
            raise ValueError("world x2/y2 must be greater than x1/y1")
        return self


class FinalDefect(InferenceModel):
    """One settled physical defect ready for transactional persistence."""

    defect_no: int = Field(ge=1)
    defect_class: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    world_x1_mm: float
    world_y1_mm: float
    world_x2_mm: float
    world_y2_mm: float
    member_detection_ids: list[UUID] = Field(min_length=1)


class TriggerInferenceResult(InferenceModel):
    """Raw and newly settled results produced for one trigger bundle."""

    trigger_id: int = Field(ge=0)
    raw_detections: list[MappedDetection]
    settled_defects: list[FinalDefect]


__all__ = [
    "BoundingBoxPx",
    "Detection",
    "FinalDefect",
    "MappedDetection",
    "TriggerInferenceResult",
]
