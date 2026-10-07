"""Build dashboard routes around components supplied by the orchestrator.

The routers translate transport requests; lifecycle ownership stays with
AppOrchestrator. This module creates no runtime service objects.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI
from config.config_loader import AttrDict
from dashboard_service.routers.fallback import create_fallback_router
from dashboard_service.routers.health import create_health_router
from dashboard_service.routers.websocket import WebSocketHub, create_websocket_router
from dashboard_service.services.dashboard_coordinator import DashboardCoordinator

LOGGER = logging.getLogger(__name__)


def create_app(
    config: AttrDict,
    coordinator: DashboardCoordinator,
    hub: WebSocketHub,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]],
) -> FastAPI:
    """Build the dashboard app around shared validated runtime components."""
    app = FastAPI(title="VisionBar Dashboard Service", lifespan=lifespan)
    app.state.dashboard_coordinator = coordinator
    app.state.websocket_hub = hub
    app.include_router(
        create_websocket_router(coordinator, hub, config.dashboard.ws_path)
    )

    if config.dashboard.fallback.available:
        app.include_router(create_fallback_router(coordinator))
        LOGGER.info("HTTP dashboard fallback API is available at /api/v1")

    if config.dashboard.enable_health_endpoint:
        app.include_router(create_health_router(coordinator, hub))

    return app


__all__ = ["create_app"]
