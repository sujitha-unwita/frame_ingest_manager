"""Optional operational health endpoint for the Dashboard Service."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from fastapi import APIRouter
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise SystemExit("fastapi is required for the dashboard service") from _exc

if TYPE_CHECKING:
    from dashboard_service.routers.websocket import WebSocketHub
    from dashboard_service.services.dashboard_coordinator import DashboardCoordinator


def create_health_router(
    coordinator: "DashboardCoordinator", hub: "WebSocketHub"
) -> APIRouter:
    """Create the optional HTTP healthcheck router."""
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, object]:
        """Return diagnostics; this is not a dashboard fallback endpoint."""
        if coordinator.link is None:
            return {"status": "starting"}
        return {
            "status": "ok",
            "wall_clock": time.time(),
            **coordinator.snapshot(),
            **hub.snapshot(),
            **coordinator.link.snapshot(),
        }

    return router


__all__ = ["create_health_router"]
