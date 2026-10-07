"""Database result contract owned by the audit-write boundary."""

from __future__ import annotations

from uuid import UUID

from inference_pipeline.models.contracts import FinalDefect


class PersistedFinalDefect(FinalDefect):
    """A final defect after the database has assigned a stable UUID."""

    defect_id: UUID


__all__ = ["PersistedFinalDefect"]
