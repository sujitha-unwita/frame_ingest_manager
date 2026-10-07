"""Validated inference exchange models."""

from inference_pipeline.models.contracts import (
    BoundingBoxPx,
    Detection,
    FinalDefect,
    MappedDetection,
    TriggerInferenceResult,
)

__all__ = [
    "BoundingBoxPx",
    "Detection",
    "FinalDefect",
    "MappedDetection",
    "TriggerInferenceResult",
]
