"""Dashboard heartbeat mirroring -- LLD SW S3.8a.

The name is the document's, not ours. S3.8a's mechanism table has a row
titled exactly "Dashboard heartbeat mirroring":

    "system_status pushed to the Pi on its own 500ms-1s timer, not only when
     a new event exists"
    Rationale: "Same fix one layer up -- see S3.9's updated event table"

WHERE THIS SITS
===============

    control_service/McuLink            (owns the port, one reader)
         |
         |  properties read every 500 ms -- this module never mutates it
         v
    dashboard_service/services/status_mirror.py  (this file)
         |
         |  events.system_status(...)
         v
    dashboard_service/services/dashboard_coordinator.py  publish()
         |
         v
    every connected panel

WHY ITS OWN TIMER
=================

An opportunistic push -- one tied to defect events -- goes silent exactly
when the pipeline stalls. That is the moment the operator MOST needs to see
that something is wrong. A health strip that stops updating when the system
gets sick is worse than none, because the last frame it painted looks fine.

So this loop runs unconditionally, whether or not anything is happening.

WHAT system_status IS MADE OF -- three sources, one payload
===========================================================

    link_state, last_heartbeat_age_ms, missed_intervals
        <- HeartbeatLinkMonitor   (host arrival clock)

    trigger_id, last_trigger_age_ms
        <- TelemetryReceiver      (trigger continuity)

    capture_paused
        <- McuState               (OBSERVED from the MCU, never inferred)

    duty_cycle
        <- GpuMonitor (last inference's mean GPU activity, 0..100 or null)
    queue_depth
        <- accepted triggers not yet audit-completed (includes in-flight work)

There is no separate "telemetry mirror". The receivers already hold what is
needed; this module reads their properties and assembles one payload.

ONE CLOCK READING PER PUSH
==========================

Every age in a payload comes from a SINGLE perf_counter() call, passed into
each getter. Two separate clock reads would differ by the duration of the
work between them, and the payload would describe two slightly different
moments while claiming to describe one.

WHY published_link_state RATHER THAN THE RAW STATE
==================================================

The mirror reports the DEBOUNCED state that link_state events also carry.
Reading the raw instantaneous band here would let system_status say
LINK_DEGRADED at the same moment no link_state event had been published --
two channels contradicting each other about the same link, which is exactly
the sort of thing that teaches an operator to distrust the panel.

DATA IN   : control_service.mcu_link.McuLink (read-only)
            config_loader: AttrDict, AttrDict
DATA OUT  : events.system_status payloads, handed to the injected publisher
CREATED BY: AppOrchestrator.__init__()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Mapping, Protocol

from config.config_loader import AttrDict
from control_service.mcu_link import McuLink
from dashboard_service.services import event_factory as events

LOGGER = logging.getLogger(__name__)

#: Signature of the fan-out callable this module pushes through.
Publish = Callable[[Mapping[str, Any]], Awaitable[None]]


class GpuMetrics(Protocol):
    """Read-only metric supplied by a local sampler or the process supervisor."""

    def get_inference_average_percent(self) -> float | None:
        """Return the last inference average, or None when no sample exists."""
        ...


class DashboardMirror:
    """Pushes system_status to every panel on its own timer.

    Attributes:
        link: The MCU link this mirrors. Read-only from here.
        dashboard_config: Push cadence and related settings.
        gpu_monitor: Read-only GPU metrics provider; never runs inference here.
        publish: Injected fan-out.
        pushes: Running total of status payloads emitted.
        failures: Running total of pushes that raised.
    """

    def __init__(
        self,
        link: McuLink,
        dashboard_config: AttrDict,
        gpu_monitor: GpuMetrics,
        publish: Publish,
    ) -> None:
        """Wire the mirror to a link and a publisher.

        Args:
            link: The McuLink to read state from. Never mutated here.
            dashboard_config: Validated dashboard settings.
            gpu_monitor: Latest device-wide GPU activity, sampled independently.
            publish: Awaitable called with each system_status payload.
        """
        self.link = link
        self.dashboard_config = dashboard_config
        self.gpu_monitor = gpu_monitor
        self.publish = publish

        self.pushes = 0
        self.failures = 0

    def build_status(self, now: float | None = None) -> dict[str, object]:
        """Assemble one system_status payload from the three sources.

        Args:
            now: perf_counter() reading. Defaults to now. Taken ONCE and
                passed into every age calculation -- see the module docstring.

        Returns:
            The system_status payload, ready to serialise.
        """
        now = time.perf_counter() if now is None else now

        _, heartbeat_age = self.link.link_state(now)
        gpu_percent = self.gpu_monitor.get_inference_average_percent()

        # None until a real trigger arrives. Kept as None on the wire rather
        # than defaulted, so the panel can say "never" instead of asserting
        # a trigger id and an age that never happened.
        trigger_age_s = self.link.telemetry.trigger_age_s(now)

        status = events.system_status(
            trigger_id=self.link.telemetry.last_trigger_id,
            # Last completed inference's sampled mean, already a percentage.
            duty_cycle=gpu_percent,
            # Outstanding pipeline work, not a UART byte backlog.
            queue_depth=getattr(self.gpu_monitor, "queue_depth", 0),
            # OBSERVED from the MCU's heartbeat, never inferred from whether
            # a PAUSE command was acknowledged. See mcu_state.py.
            capture_paused=self.link.capture_paused,
            # Grows without bound while the MCU is paused, because a paused
            # line genuinely has not inspected anything recently. That is
            # correct, and it is what the panel should show.
            last_trigger_age_ms=(
                None if trigger_age_s is None
                else round(trigger_age_s * 1000, 1)
            ),
            # The DEBOUNCED state, matching what link_state events reported.
            link_state=self.link.link.published_state,
            last_heartbeat_age_ms=round(heartbeat_age * 1000, 1),
            # Separates the two bands sharing LINK_DEGRADED: 1 is a hiccup,
            # 2 is one interval away from LINK_DOWN.
            missed_intervals=self.link.missed_intervals(now),
            # [OURS -- Pooja] Both OBSERVED from the MCU, never inferred.
            # is_armed drives the START CAPTURE button; faults drives the
            # alarm strip. Repeated on every push so a panel that connected
            # late, or dropped an event, is correct within 500 ms.
            is_armed=self.link.state.is_armed,
            faults=self.link.state.active_faults,
        )

        status["workers"] = dict(getattr(self.gpu_monitor, "states", {}))
        return status

    async def run(self) -> None:
        """Push system_status forever, on the configured cadence.

        Never exits on error. This loop must be the LAST thing to die: if it
        stopped, every panel would freeze on its last reading and go on
        looking healthy indefinitely.

        Raises:
            asyncio.CancelledError: Propagated on shutdown.
        """
        interval = self.dashboard_config.status_push_interval_s
        LOGGER.info(
            "dashboard mirror pushing system_status every %.0f ms",
            interval * 1000,
        )

        while True:
            try:
                await asyncio.sleep(interval)
                await self.publish(self.build_status())
                self.pushes += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.failures += 1
                LOGGER.exception("system_status push failed; continuing")

    def snapshot(self) -> dict[str, object]:
        """Return counters, for logging and diagnostics.

        Returns:
            A plain dict of mirror counters. Not a wire payload.
        """
        return {
            "status_pushes": self.pushes,
            "status_push_failures": self.failures,
        }
