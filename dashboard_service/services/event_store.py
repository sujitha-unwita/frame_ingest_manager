"""Single-panel replay queue for WebSocket-to-HTTP dashboard fallback.

The dashboard normally receives events immediately over WebSocket.  If that
connection breaks, the one operator panel asks this store for every discrete
event after its last processed ``event_id``.  The panel explicitly
acknowledges processed IDs, allowing those events to be removed.

``system_status`` is intentionally not stored here: it is a current-state
snapshot and is obtained from the status endpoint.  This store is for events
whose occurrence must remain visible across a short connection outage.

DATA IN   : complete event dictionaries created by services/event_factory.py
DATA OUT  : copies enriched with ``stream_id`` / ``event_id`` and ordered
            replay batches
CREATED BY: services/dashboard_coordinator.DashboardCoordinator
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from typing import Any, Mapping
from uuid import uuid4


LOGGER = logging.getLogger(__name__)


class EventStoreError(ValueError):
    """Raised when a caller supplies an invalid event cursor or acknowledgement."""


@dataclass(frozen=True)
class EventReplay:
    """One ordered answer to a panel's request for missed events.

    Attributes:
        stream_id: Unique identifier for this service process's event stream.
            A changed value means the service restarted and in-memory events
            from the old process cannot be replayed.
        events: Complete event dictionaries, ordered by increasing event_id.
        latest_event_id: Largest ID allocated by this store, including events
            created after the returned batch was prepared on a later request.
        resync_required: True when the requested cursor is older than the
            retained/acknowledged range, so replay cannot prove completeness.
        oldest_available_event_id: First ID still available for replay, or the
            next ID to allocate when no events are pending.
    """

    stream_id: str
    events: tuple[dict[str, object], ...]
    latest_event_id: int
    resync_required: bool
    oldest_available_event_id: int


class EventStore:
    """Bounded, in-memory event replay queue for exactly one dashboard panel.

    The queue provides at-least-once delivery.  A panel may see an event again
    after a failed response or acknowledgement; it ignores the duplicate using
    its saved last_processed_event_id.  Events are pruned only after that one
    panel explicitly acknowledges them.

    If the panel stays disconnected long enough to exceed
    ``max_pending_events``, the
    oldest unacknowledged events are discarded and a later replay reports
    ``resync_required`` rather than silently hiding the gap.
    """

    def __init__(self, max_pending_events: int) -> None:
        """Create an empty store.

        Args:
            max_pending_events: Maximum number of unacknowledged events
                retained.  This comes from
                ``dashboard.events_store.max_pending_events`` when the
                DashboardCoordinator creates the store.

        Raises:
            EventStoreError: If max_pending_events is not a positive integer.
        """
        if (
            not isinstance(max_pending_events, int)
            or isinstance(max_pending_events, bool)
        ):
            raise EventStoreError(
                "max_pending_events must be a positive integer"
            )
        if max_pending_events <= 0:
            raise EventStoreError("max_pending_events must be a positive integer")

        self.max_pending_events = max_pending_events
        self.stream_id = str(uuid4())
        self._next_event_id = 1
        self._acknowledged_through = 0
        self._unavailable_through = 0
        self._overflow_drops = 0
        self._pending: deque[dict[str, object]] = deque()

    @property
    def latest_event_id(self) -> int:
        """Return the most recently allocated event ID, or zero initially."""
        return self._next_event_id - 1

    @property
    def acknowledged_through(self) -> int:
        """Return the highest event ID the sole panel has acknowledged."""
        return self._acknowledged_through

    def record_event(self, message: Mapping[str, Any]) -> dict[str, object]:
        """Store one complete discrete event and return its enriched copy.

        The input mapping is never mutated.  The returned dict is suitable for
        immediate WebSocket delivery and later HTTP replay.

        Args:
            message: Existing dashboard event payload, containing a non-empty
                string ``event`` field.

        Returns:
            A copy of the event with backend-assigned ``stream_id`` and
            ``event_id`` delivery metadata.

        Raises:
            EventStoreError: If the payload does not look like a dashboard
                event or tries to supply its own delivery metadata.
        """
        event_name = message.get("event")
        if not isinstance(event_name, str) or not event_name:
            raise EventStoreError("event payload must contain a non-empty 'event'")
        reserved = {"event_id", "stream_id"} & set(message)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise EventStoreError(
                f"event payload must not supply reserved field(s): {names}"
            )

        enriched: dict[str, object] = dict(message)
        enriched["stream_id"] = self.stream_id
        enriched["event_id"] = self._next_event_id
        self._next_event_id += 1
        self._pending.append(enriched)

        while len(self._pending) > self.max_pending_events:
            removed = self._pending.popleft()
            self._unavailable_through = max(
                self._unavailable_through, int(removed["event_id"])
            )
            self._overflow_drops += 1
            LOGGER.warning(
                "event queue capacity %d exceeded; dropped unacknowledged "
                "event_id=%s (panel resync required)",
                self.max_pending_events,
                removed["event_id"],
            )

        return dict(enriched)

    def events_after(self, event_id: int) -> EventReplay:
        """Return complete retained events with IDs greater than ``event_id``.

        Args:
            event_id: Last event successfully processed by the panel.  Zero is
                valid for a newly connected panel in this service process.

        Returns:
            An ordered replay answer.  When ``resync_required`` is true, the
            caller must refresh current status before trusting the stream.

        Raises:
            EventStoreError: If event_id is negative or not an integer.
        """
        if not isinstance(event_id, int) or isinstance(event_id, bool):
            raise EventStoreError("event_id must be a non-negative integer")
        if event_id < 0:
            raise EventStoreError("event_id must be a non-negative integer")

        oldest = (
            int(self._pending[0]["event_id"])
            if self._pending else self._next_event_id
        )
        resync_required = event_id < self._unavailable_through
        events = tuple(
            dict(item) for item in self._pending
            if int(item["event_id"]) > event_id
        )
        return EventReplay(
            stream_id=self.stream_id,
            events=events,
            latest_event_id=self.latest_event_id,
            resync_required=resync_required,
            oldest_available_event_id=oldest,
        )

    def acknowledge(self, event_id: int) -> int:
        """Acknowledge all events through event_id and prune them.

        Repeating an acknowledgement, or acknowledging an older cursor, is
        harmless and returns the current acknowledgement cursor.

        Args:
            event_id: Highest event the single panel has completely processed.

        Returns:
            The resulting acknowledged-through cursor.

        Raises:
            EventStoreError: If the ID is invalid or was never allocated.
        """
        if not isinstance(event_id, int) or isinstance(event_id, bool):
            raise EventStoreError("event_id must be a non-negative integer")
        if event_id < 0:
            raise EventStoreError("event_id must be a non-negative integer")
        if event_id > self.latest_event_id:
            raise EventStoreError(
                f"cannot acknowledge event_id {event_id}; latest is "
                f"{self.latest_event_id}"
            )
        if event_id <= self._acknowledged_through:
            return self._acknowledged_through

        self._acknowledged_through = event_id
        self._unavailable_through = max(self._unavailable_through, event_id)
        while self._pending and int(self._pending[0]["event_id"]) <= event_id:
            self._pending.popleft()
        return self._acknowledged_through

    def snapshot(self) -> dict[str, object]:
        """Return queue diagnostics for `/health` and tests."""
        return {
            "event_stream_id": self.stream_id,
            "latest_event_id": self.latest_event_id,
            "events_acknowledged_through": self._acknowledged_through,
            "events_pending": len(self._pending),
            "event_queue_capacity": self.max_pending_events,
            "event_queue_overflow_drops": self._overflow_drops,
        }


__all__ = ["EventReplay", "EventStore", "EventStoreError"]
