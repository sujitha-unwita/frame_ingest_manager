"""Strict inbound command model shared by WebSocket and HTTP transports."""

from typing import Literal

from pydantic import BaseModel


class DashboardRequestModel(BaseModel):
    """Base model for every dashboard-originated JSON request.

    Dashboard requests are a closed contract. Rejecting unknown keys prevents
    a spelling error or an unapproved future command field from being silently
    ignored before the request reaches control logic.
    """

    # ``Config`` works with both Pydantic 1 and 2. FastAPI 0.110 supports
    # either line, so the deployed Jetson environment need not match this
    # development environment exactly.
    class Config:
        """Forbid fields outside the public dashboard request contract."""

        extra = "forbid"


class OperatorCommandRequest(DashboardRequestModel):
    """One fully validated Start, Resume, or Stop dashboard command."""

    command: Literal[
        "operator_start_capture",
        "operator_resume_request",
        "operator_stop_request",
    ]
