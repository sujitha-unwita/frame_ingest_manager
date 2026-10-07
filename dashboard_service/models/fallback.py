"""Strict Pydantic models for HTTP fallback requests."""

from pydantic import Field

from dashboard_service.models.commands import DashboardRequestModel


class EventReplayQuery(DashboardRequestModel):
    """Validated cursor supplied to ``GET /api/v1/events``."""

    after: int = Field(default=0, ge=0)


class EventAcknowledgementRequest(DashboardRequestModel):
    """Single-panel acknowledgement for all events through one cursor."""

    stream_id: str = Field(min_length=1)
    last_processed_event_id: int = Field(ge=0)
