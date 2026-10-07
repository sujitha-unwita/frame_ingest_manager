"""WebSocket transport for the dashboard operator panel.

The router owns connected sockets and transport failures. It delegates command
execution and event retention to ``DashboardCoordinator``; it never opens the
MCU serial port or maintains a second command path.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import constants
from dashboard_service.models.commands import OperatorCommandRequest
from dashboard_service.services import event_factory as events

if TYPE_CHECKING:
    from dashboard_service.services.dashboard_coordinator import DashboardCoordinator

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    from pydantic import ValidationError
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "fastapi is required for the dashboard service. Install it with:\n"
        "    python3 -m pip install fastapi uvicorn websockets pyyaml"
    ) from _exc

LOGGER = logging.getLogger(__name__)


def _validate_command(message: Mapping[str, Any]) -> OperatorCommandRequest:
    """Validate one WebSocket command with the shared HTTP/WS model."""
    # Support the Pydantic version provided by either supported FastAPI line.
    if hasattr(OperatorCommandRequest, "model_validate"):
        return OperatorCommandRequest.model_validate(message)
    return OperatorCommandRequest.parse_obj(message)


class WebSocketHub:
    """Own connected panels and safely fan one event out to all of them."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self.broadcasts_sent = 0
        self.broadcast_failures = 0

    def add(self, websocket: WebSocket) -> None:
        """Register an accepted panel connection."""
        self._clients.add(websocket)

    def discard(self, websocket: WebSocket) -> None:
        """Forget a disconnected panel connection."""
        self._clients.discard(websocket)

    @property
    def client_count(self) -> int:
        """Return the current number of connected panel sockets."""
        return len(self._clients)

    async def broadcast(self, message: Mapping[str, Any]) -> None:
        """Deliver an already-recorded event to every connected panel."""
        for websocket in list(self._clients):
            try:
                await websocket.send_json(dict(message))
                self.broadcasts_sent += 1
            except (WebSocketDisconnect, RuntimeError, OSError) as exc:
                self.broadcast_failures += 1
                LOGGER.info("dropping disconnected panel (%s)", type(exc).__name__)
                self.discard(websocket)
            except Exception:
                self.broadcast_failures += 1
                LOGGER.exception("unexpected error sending to a panel")
                self.discard(websocket)

    def snapshot(self) -> dict[str, object]:
        """Return WebSocket transport counters for diagnostics."""
        return {
            "panels_connected": self.client_count,
            "broadcasts_sent": self.broadcasts_sent,
            "broadcast_failures": self.broadcast_failures,
        }


def create_websocket_router(
    coordinator: "DashboardCoordinator", hub: WebSocketHub, ws_path: str
) -> APIRouter:
    """Create the dashboard WebSocket endpoint around shared service state."""
    router = APIRouter()

    @router.websocket(ws_path)
    async def dashboard_endpoint(websocket: WebSocket) -> None:
        """Receive panel commands while the hub independently sends events."""
        await websocket.accept()
        hub.add(websocket)
        LOGGER.info("panel connected (%d total)", hub.client_count)

        try:
            # A panel connecting mid-run starts with a current capture state.
            await websocket.send_json(
                events.capture_paused(
                    coordinator.capture_paused, constants.SOURCE_SNAPSHOT
                )
            )
            while True:
                message = await websocket.receive_json()
                if not isinstance(message, Mapping):
                    raise ValueError("command message must be a JSON object")
                try:
                    request = _validate_command(message)
                except ValidationError as exc:
                    command = message.get("command")
                    await coordinator.publish(
                        events.command_failed(str(command), -1, str(exc))
                    )
                    continue
                try:
                    await coordinator.submit_operator_command(request.command)
                except (ValueError, RuntimeError) as exc:
                    LOGGER.warning("panel command %r rejected: %s", request.command, exc)
                    await coordinator.publish(
                        events.command_failed(request.command, -1, str(exc))
                    )
        except WebSocketDisconnect:
            LOGGER.info("panel disconnected")
        except ValueError as exc:
            LOGGER.warning("invalid panel message: %s", exc)
            with contextlib.suppress(Exception):
                await coordinator.publish(events.command_failed("unknown", -1, str(exc)))
        except Exception:
            LOGGER.exception("panel connection failed")
        finally:
            hub.discard(websocket)
            LOGGER.info("panel removed (%d remain)", hub.client_count)

    return router


__all__ = ["WebSocketHub", "create_websocket_router"]
