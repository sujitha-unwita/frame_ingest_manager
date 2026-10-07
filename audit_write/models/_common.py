"""Pieces every model shares: base classes and validators driven by the YAML.

Models describe shapes only.  They never import ``src.database`` or
``src.repositories`` — a model does not know where the data came from.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from config.config_loader import get_audit_settings as get_settings


class RowModel(BaseModel):
    """A row as read from the database (response shape).  Unknown keys are ignored."""

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class WriteModel(BaseModel):
    """A payload for insert/update (request shape).  Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    def to_row(self) -> dict[str, Any]:
        """Only the fields the caller actually set, so database defaults still apply."""
        return self.model_dump(exclude_unset=True)


def check_confidence(value: float | None) -> float | None:
    if value is None:
        return value
    rules = get_settings().validation
    if not rules.confidence_min <= value <= rules.confidence_max:
        raise ValueError(f"confidence must be between {rules.confidence_min} and {rules.confidence_max}")
    return value


def check_frame_variant(value: str | None) -> str | None:
    return _check_allowed(value, get_settings().validation.frame_variants, "frame_variant")


def check_defect_class(value: str | None) -> str | None:
    return _check_allowed(value, get_settings().validation.defect_classes, "defect_class")


def check_camera_id(value: str | None) -> str | None:
    return _check_length(value, get_settings().validation.camera_id_max_length, "camera_id")


def check_label(value: str | None) -> str | None:
    return _check_length(value, get_settings().validation.label_max_length, "label")


def check_not_blank(value: str | None) -> str | None:
    """Text columns use NULL for "nothing", never an empty string."""
    if value is not None and value.strip() == "":
        raise ValueError("must be null or non-empty, not an empty string")
    return value


def _check_allowed(value: str | None, allowed: tuple[str, ...], field: str) -> str | None:
    if value is not None and allowed and value not in allowed:
        raise ValueError(f"{field} must be one of {list(allowed)}")
    return value


def _check_length(value: str | None, max_length: int, field: str) -> str | None:
    if value is not None and len(value) > max_length:
        raise ValueError(f"{field} must be at most {max_length} characters")
    return value
