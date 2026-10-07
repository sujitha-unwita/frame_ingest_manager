"""FastAPI HTTP fallback routes for the single operator dashboard panel.

WebSocket remains the preferred real-time transport. These routes serve the
same DashboardCoordinator when the panel cannot establish or retain its WebSocket
connection:

    GET  /api/v1/status                current system_status snapshot
    GET  /api/v1/events?after=<id>     replay discrete events
    POST /api/v1/events/ack            acknowledge events through an ID
    POST /api/v1/commands              submit an existing operator command

This module owns HTTP request/response handling only. It neither opens the
serial port nor sends WebSocket messages; DashboardCoordinator is the injected
orchestrator for both paths.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from control_service.mcu_command import CommandResult
from dashboard_service.services.event_store import EventReplay, EventStoreError

if TYPE_CHECKING:
    from dashboard_service.services.dashboard_coordinator import DashboardCoordinator

try:
    from fastapi import APIRouter, Depends, HTTPException
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "fastapi is required for the dashboard service. Install it with:\n"
        "    python3 -m pip install fastapi uvicorn websockets pyyaml"
    ) from _exc

LOGGER = logging.getLogger(__name__)

from dashboard_service.models.commands import OperatorCommandRequest
from dashboard_service.models.fallback import (
    EventAcknowledgementRequest,
    EventReplayQuery,
)


def _replay_payload(replay: EventReplay) -> dict[str, object]:
    """Convert an EventReplay into the documented HTTP JSON shape."""
    return {
        "stream_id": replay.stream_id,
        "events": list(replay.events),
        "latest_event_id": replay.latest_event_id,
        "oldest_available_event_id": replay.oldest_available_event_id,
        "resync_required": replay.resync_required,
    }


def _command_payload(result: CommandResult) -> dict[str, object]:
    """Convert an existing McuLink command result into HTTP JSON."""
    return {
        "command": result.command,
        "cmd_seq": result.command_sequence,
        "acknowledged": result.acknowledged,
        "attempts": result.attempts,
        "nacked": result.nacked,
        "reason": result.reason,
        "mcu_state": result.mcu_state,
    }


def create_fallback_router(service: "DashboardCoordinator") -> APIRouter:
    """Build HTTP fallback routes around one DashboardCoordinator instance.

    Args:
        service: Running dashboard orchestrator. It owns the EventStore,
            DashboardMirror and McuLink shared with WebSocket delivery.

    Returns:
        An API router rooted at `/api/v1`.
    """
    router = APIRouter(prefix="/api/v1", tags=["dashboard fallback"])

    @router.get("/status")
    async def status() -> dict[str, object]:
        """Return current dashboard status, never retained event history."""
        try:
            return service.current_status()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.get("/events")
    async def replay_events(
        query: EventReplayQuery = Depends(),
    ) -> dict[str, object]:
        """Return full retained events after the panel's event cursor."""
        try:
            return _replay_payload(service.replay_events(query.after))
        except EventStoreError as exc:
            LOGGER.warning("invalid event replay cursor %r: %s", query.after, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/events/ack")
    async def acknowledge_events(
        body: EventAcknowledgementRequest,
    ) -> dict[str, object]:
        """Acknowledge processed events and permit their queue removal."""
        try:
            acknowledged = service.acknowledge_events(
                body.stream_id, body.last_processed_event_id
            )
        except EventStoreError as exc:
            LOGGER.warning("event acknowledgement rejected: %s", exc)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "stream_id": body.stream_id,
            "acknowledged_through": acknowledged,
        }

    @router.post("/commands")
    async def submit_command(body: OperatorCommandRequest) -> dict[str, object]:
        """Submit a panel command through the existing McuLink sender."""
        try:
            result = await service.submit_operator_command(body.command)
        except ValueError as exc:
            LOGGER.warning("HTTP panel sent unknown command %r", body.command)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _command_payload(result)

    return router


__all__ = ["create_fallback_router"]
