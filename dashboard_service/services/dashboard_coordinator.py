"""Business coordinator for the AGX Dashboard Service.

This module deliberately knows about the MCU link and dashboard state, but not
about FastAPI, HTTP requests, or WebSocket objects. The orchestrator composes it
with the transport routers. Therefore the same event is recorded once and can
be delivered over WebSocket now or replayed through HTTP later.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from config.config_loader import AttrDict
from control_service.mcu_command import CommandResult
from control_service.mcu_link import McuLink
from dashboard_service.services.event_store import EventReplay, EventStore, EventStoreError
from dashboard_service.services.status_mirror import DashboardMirror

LOGGER = logging.getLogger(__name__)

# Panel vocabulary -> existing MCU command vocabulary. These names form the
# dashboard wire contract, so they are code constants rather than YAML knobs.
OPERATOR_COMMANDS: dict[str, str] = {
    "operator_stop_request": "pause",
    "operator_resume_request": "resume",
    "operator_start_capture": "resume",
}

PanelPublisher = Callable[[Mapping[str, Any]], Awaitable[None]]


class DashboardCoordinator:
    """Own dashboard state shared by the WebSocket and HTTP transports.

    The coordinator records each discrete event before handing it to the
    injected panel publisher. The publisher is the WebSocket hub in normal
    operation; the EventStore makes the same event available to HTTP replay.
    """

    def __init__(
        self,
        config: AttrDict,
        publish_to_panels: PanelPublisher,
        link: McuLink,
        mirror: DashboardMirror,
        on_operator_command: Callable[[str], Awaitable[CommandResult]],
    ) -> None:
        """Construct the service without opening a serial port.

        Args:
            config: Fully validated application configuration.
            publish_to_panels: Transport callback supplied by the WebSocket
                hub. It receives already-enriched event dictionaries.
            link: Shared MCU link created and started by the orchestrator.
            mirror: Shared status builder, also owned by the orchestrator.
        """
        self.config = config
        self._publish_to_panels = publish_to_panels
        self.link = link
        self.mirror = mirror
        self._on_operator_command = on_operator_command
        self.event_store = EventStore(
            max_pending_events=config.dashboard.events_store.max_pending_events
        )

    @property
    def capture_paused(self) -> bool:
        """Return the current capture state for a newly connected panel."""
        return self.link.capture_paused

    async def publish(self, message: Mapping[str, Any]) -> None:
        """Record a discrete event, then deliver it to connected WebSockets.

        ``system_status`` is a current-state snapshot and intentionally is not
        retained. It is read from ``GET /api/v1/status`` during fallback. All
        other events receive ``stream_id`` and ``event_id`` so both transports
        share one ordered event contract.
        """
        outbound: Mapping[str, Any] = message
        if message.get("event") != "system_status":
            try:
                outbound = self.event_store.record_event(message)
            except EventStoreError:
                # Event storage must not stop the safety/control serial loop.
                LOGGER.exception(
                    "could not retain %r event; sending live only",
                    message.get("event"),
                )
        await self._publish_to_panels(outbound)

    def current_status(self) -> dict[str, object]:
        """Return the current ``system_status`` payload for HTTP polling."""
        return self.mirror.build_status()

    def replay_events(self, after_event_id: int) -> EventReplay:
        """Return retained discrete events after the panel's cursor."""
        return self.event_store.events_after(after_event_id)

    def acknowledge_events(
        self, stream_id: str, last_processed_event_id: int
    ) -> int:
        """Advance the one panel's acknowledgement cursor and prune events."""
        if stream_id != self.event_store.stream_id:
            raise EventStoreError(
                "event stream changed; refresh status before acknowledging"
            )
        return self.event_store.acknowledge(last_processed_event_id)

    async def submit_operator_command(self, command: object) -> CommandResult:
        """Send an operator command through the one existing MCU link."""
        if command not in OPERATOR_COMMANDS:
            known = ", ".join(sorted(OPERATOR_COMMANDS))
            raise ValueError(f"unknown command; expected one of: {known}")
        return await self._on_operator_command(command)

    def snapshot(self) -> dict[str, object]:
        """Return service diagnostics, excluding transport-specific counters."""
        data: dict[str, object] = {**self.event_store.snapshot()}
        data.update(self.mirror.snapshot())
        return data


__all__ = ["DashboardCoordinator", "OPERATOR_COMMANDS", "PanelPublisher"]
