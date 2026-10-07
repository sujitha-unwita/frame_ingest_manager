"""OBSERVES the MCU's state machine. Does NOT implement one.

WHERE THIS SITS
===============

    STM32 FIRMWARE                      AGX -- this file
    --------------                      ----------------
    RUNS the state machine       ──►    STORES the reported value
    decides every transition            decides nothing
    (HW LLD S3.5.3)                     answers: capture_paused? state_name?

    MCU --(heartbeat / telemetry / ack)--> mcu_link.McuLink._dispatch
                                                    |
                                                    v
                                        McuState.observe_state / _flags
                                                    |
                                        returns event dicts, or []
                                                    |
                                                    v
                                        McuLink publishes them onward

READ THIS FIRST: WE DO NOT RUN A STATE MACHINE HERE
===================================================

HW LLD S3.5.3 defines the machine, and it lives in firmware:

    IDLE           -> RUNNING   on RESUME, ACK'd
    RUNNING        -> PAUSED    on PAUSE, graceful -- finishes the in-flight
                                trigger, "never halts mid-strobe-sequence"
    RUNNING/PAUSED -> FAULT     autonomously on any status_flags fault bit;
                                TIM1 halts immediately, "an e-stop can't
                                wait for an in-flight trigger"
    FAULT          -> RUNNING   only on RESUME **AND** fault bits clear --
                                "a command alone can't clear a real hardware
                                fault"

This class NEVER computes a transition. It reads the `state` byte off the
wire and remembers it. If we ran our own copy of the machine it could
disagree with the MCU's, and ours would be the wrong one.

S3.5.3 also explains why `state` and `status_flags` are separate fields:
"state is what the MCU is doing, status_flags is which fault(s) are set, and
a FAULT state can co-occur with any combination of flag bits."

THE RULE THIS MODULE ENFORCES
=============================

    capture_paused is OBSERVED from the MCU's `state` byte.
    It is NEVER INFERRED from whether a command succeeded.

A command is a REQUEST. The heartbeat is the TRUTH.

The lost-ACK problem (LLD SW S3.6, S3.8a):

    Host sends PAUSE -> MCU receives it, pauses, sends ACK -> ACK is lost

    If the host inferred state from the ACK, it now believes capture is
    RUNNING while the MCU has actually stopped. The panel shows a green
    running indicator over a stopped line, and the operator -- seeing no
    defects scrolling past -- concludes the fabric is clean.

    Inverted, it is just as bad: an ACK arrives but the MCU faults before
    acting, and the panel shows PAUSED over a line that is still moving.

Observing state from the heartbeat closes both. Within 100 ms of any real
state change, the MCU tells us what it is actually doing.

WHY THREE MESSAGE TYPES FEED THIS
=================================

heartbeat, telemetry and ack_nack payloads ALL carry a `state` byte (HW LLD
S3.5.2 -- telemetry gained it "for consistency with HEARTBEAT"). The
heartbeat is the authoritative 100 ms cadence, but taking state from all
three means an ACK's state field can confirm a command took effect up to
100 ms before the next beat would have. They cannot disagree for long; if
they do, the most recent frame wins, which is correct either way.

DATA IN   : decoded message dicts from mcu_link.McuLink._dispatch()
            config_loader.AttrDict, for the state code points
DATA OUT  : event dicts from dashboard/events.py, returned to McuLink
CREATED BY: control_service.mcu_link.McuLink.__init__
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import constants
from config.config_loader import AttrDict
from dashboard_service.services import event_factory as events

LOGGER = logging.getLogger(__name__)


class McuState:
    """Tracks the MCU's reported state, fault flags and uptime.

    Every observe_* method is PURE with respect to I/O: it mutates this
    object's fields and RETURNS event dicts, but never publishes them. McuLink
    does the publishing.

    That split is deliberate. It keeps this class synchronous and trivially
    testable -- self_test.py drives it with plain dicts and no event loop --
    and it keeps every outbound event flowing through one place in McuLink,
    so there is a single point to look at when asking "what does the panel
    get told, and when?"

    Attributes:
        states: The configured state code points and their names.
        state: Last observed state code, or None before the first frame.
        status_flags: Last observed raw flags byte.
        uptime_s: Last observed uptime, or None before the first heartbeat.
    """

    def __init__(self, states: AttrDict) -> None:
        """Initialise with nothing observed yet.

        Args:
            states: Validated state code points from config.yaml.
        """
        self.states = states

        # None, not a default state. "We have not heard from the MCU yet" is
        # genuinely different from "the MCU is IDLE", and conflating them
        # would have the panel assert a state nothing has confirmed.
        self.state: int | None = None
        self.status_flags: int = 0
        self.uptime_s: int | None = None

        # Last flag list published, so a latched fault does not spam the
        # panel at 10 Hz. Compared by value, not identity.
        self._reported_flags: list[str] = []

    # ---- derived views ---------------------------------------------------

    @property
    def capture_paused(self) -> bool:
        """Whether the MCU is currently NOT inspecting fabric.

        RUNNING is the ONLY state in which fabric is being inspected, so this
        is defined as "not RUNNING" rather than as a list of paused states.

        WHY THAT DIRECTION MATTERS: written the other way round -- returning
        True for PAUSED and IDLE -- FAULT falls through the list and returns
        False, so an e-stop would report the line as running. Defining it
        negatively means any state added to the firmware later is treated as
        not-inspecting until someone deliberately says otherwise, which is
        the safe default.

        Before the first heartbeat the state is unknown; report not-paused so
        a freshly connected panel does not flash a false alarm at startup.

        Returns:
            True if the MCU is in any state other than RUNNING.
        """
        if self.state is None:
            return False
        return self.state != self.states.running

    @property
    def state_name(self) -> str:
        """Human-readable state name for the dashboard.

        Returns:
            The configured name; "UNKNOWN" before the first frame; or
            "UNKNOWN_<code>" for a code the firmware sent but config.yaml
            does not define -- surfaced rather than hidden, because it means
            firmware and config have diverged.
        """
        if self.state is None:
            return "UNKNOWN"
        return self.states.names.get(self.state, f"UNKNOWN_{self.state}")

    @property
    def active_faults(self) -> list[str]:
        """The fault flag names currently set, newest observation wins.

        [OURS -- Pooja] Decoded from the status_flags byte, which arrives on
        BOTH heartbeat and telemetry frames (HW LLD S3.5.2), so this is
        refreshed at 10 Hz whether or not the line is running.

        Empty list means no faults -- NOT unknown. Before the first frame
        status_flags is 0 and this is empty, which is the safe reading: the
        panel shows no alarm until the MCU says otherwise, and link_state
        already covers "we have not heard from it".

        Returns:
            A copy of the active flag names, e.g. ["encoder_fault"], or [].
        """
        return list(self._reported_flags)

    @property
    def is_armed(self) -> bool:
        """Whether the MCU has self-checked and is waiting to start.

        [OURS -- Pooja] ARMED is not in HW LLD S3.5.3. Firmware must send
        code 4; until it does this is never true and the MCU reports IDLE.

        Returns:
            True if the last observed state was ARMED.
        """
        return self.state == self.states.armed

    @property
    def is_faulted(self) -> bool:
        """Whether the MCU is in the FAULT state.

        Returns:
            True if the last observed state was FAULT.
        """
        return self.state == self.states.fault

    # ---- observation -----------------------------------------------------

    def observe_state(
        self, message: Mapping[str, Any]
    ) -> list[dict[str, object]]:
        """Record the `state` byte from any frame that carries one.

        Args:
            message: A decoded heartbeat, telemetry or ack_nack dict. Frames
                without a "state" key are ignored, so callers need not
                pre-filter.

        Returns:
            A capture_paused event if the paused/running condition changed,
            otherwise an empty list.

        Raises:
            Nothing. A malformed `state` value is logged and dropped rather
            than raised: one bad frame must not kill the read loop, and the
            next heartbeat is only 100 ms away.
        """
        raw = message.get("state")
        if raw is None:
            return []

        try:
            new_state = int(raw)
        except (TypeError, ValueError):
            LOGGER.warning(
                "ignoring frame with non-integer state %r (message_type=%s)",
                raw, message.get("message_type"),
            )
            return []

        previous = self.state
        was_paused = self.capture_paused
        self.state = new_state

        # First frame ever: adopt the state silently. Publishing a transition
        # here would announce a "change" from nothing to something, which the
        # panel would render as the line having just stopped or started.
        # dashboard_ws sends a snapshot on connect for this purpose instead.
        if previous is None:
            LOGGER.info(
                "MCU state observed for the first time: %s", self.state_name
            )
            return []

        if self.capture_paused == was_paused:
            return []

        # A pause caused by a fault must be rendered differently from an
        # operator pause (S3.9), so the source is derived from the state we
        # landed in rather than from who asked.
        source = (
            constants.SOURCE_FAULT if self.is_faulted else constants.SOURCE_MCU
        )
        LOGGER.info(
            "MCU state change: %s -> %s (capture_paused=%s, source=%s)",
            self.states.names.get(previous, previous),
            self.state_name, self.capture_paused, source,
        )
        return [events.capture_paused(self.capture_paused, source)]

    def observe_flags(
        self, message: Mapping[str, Any]
    ) -> list[dict[str, object]]:
        """Record the status_flags byte and its decoded flag names.

        Publishes on EVERY change, including a change back to no flags at
        all. A panel told only when a fault raises can never clear the alarm.

        Args:
            message: A decoded heartbeat or telemetry dict, carrying
                "status_flags" and "active_flags".

        Returns:
            Always an empty list. The flag change is logged and stored;
            system_status.faults is what the panel reads.

        Raises:
            Nothing. Malformed values are logged and dropped.
        """
        if "status_flags" not in message:
            return []

        try:
            self.status_flags = int(message["status_flags"])
        except (TypeError, ValueError):
            LOGGER.warning(
                "ignoring frame with non-integer status_flags %r",
                message.get("status_flags"),
            )
            return []

        raw_flags = message.get("active_flags") or []
        try:
            flags = [str(flag) for flag in raw_flags]
        except TypeError:
            LOGGER.warning("ignoring non-iterable active_flags %r", raw_flags)
            return []

        if flags == self._reported_flags:
            return []

        # WARNING when faults are SET, INFO when they clear. A panel_estop
        # must not sit at the same level as "command requested".
        log = LOGGER.warning if flags else LOGGER.info
        log(
            "MCU fault flags changed: %s -> %s (state %s)",
            self._reported_flags or "(none)", flags or "(none)",
            self.state_name,
        )
        self._reported_flags = flags

        # NO EVENT. mcu_fault appears NOWHERE in the SW LLD. The flags reach
        # the panel as system_status.faults, on every 500 ms push. One
        # channel, no chance of the two disagreeing.
        return []

    def observe_uptime(
        self, message: Mapping[str, Any]
    ) -> list[dict[str, object]]:
        """Record uptime_s and detect an unexpected MCU reboot.

        Uptime going BACKWARDS is the only evidence a crash-and-restart
        leaves. Heartbeats resume within 100 ms of the restart, so link_state
        stays LINK_UP across the whole event and nothing else notices.

        Args:
            message: A decoded heartbeat dict carrying "uptime_s".

        Returns:
            An mcu_reboot event if uptime went backwards, otherwise empty.

        Raises:
            Nothing. Malformed values are logged and dropped.
        """
        if "uptime_s" not in message:
            return []

        try:
            uptime = int(message["uptime_s"])
        except (TypeError, ValueError):
            LOGGER.warning(
                "ignoring frame with non-integer uptime_s %r",
                message.get("uptime_s"),
            )
            return []

        previous, self.uptime_s = self.uptime_s, uptime
        if previous is None or uptime >= previous:
            return []

        # NOTE ON THE 16-BIT WRAP: uptime_s is a uint16 (HW LLD S3.5.2), so
        # it wraps to 0 after 65535 s (18h12m). That wrap is indistinguishable
        # from a reboot using this field alone, and is reported as one.
        # Accepted deliberately: a false reboot notice once every 18 hours of
        # continuous running is far cheaper than missing a real crash. If it
        # becomes a nuisance, the fix belongs in the firmware -- widen the
        # field -- not in a heuristic here.
        LOGGER.warning(
            "MCU reboot detected: uptime %ds -> %ds", previous, uptime
        )
        return [events.mcu_reboot(previous, uptime)]

    def snapshot(self) -> dict[str, object]:
        """Return the current observed state, for logging and diagnostics.

        Returns:
            A plain dict of every observed field. Not a wire payload -- use
            dashboard/events.py for anything the dashboard consumes.
        """
        return {
            "state": self.state,
            "state_name": self.state_name,
            "capture_paused": self.capture_paused,
            "status_flags": self.status_flags,
            "active_flags": list(self._reported_flags),
            "uptime_s": self.uptime_s,
        }
