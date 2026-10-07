"""Heartbeat handling: two classes, two clocks, two different questions.

WHERE THIS SITS
===============

    MCU --(MSG_HEARTBEAT every 100ms)--> mcu_link.McuLink._on_heartbeat
                                                    |
                    +-------------------------------+------------------+
                    |                                                  |
                    v                                                  v
        HeartbeatReceiver.update(msg)              HeartbeatLinkMonitor.record(t)
        MCU'S OWN TIMESTAMP                        HOST ARRIVAL TIME
                    |                                                  |
                    v                                                  v
        "was a beat actually LOST?"                "how STALE is my info?"
                    |                                                  |
                    v                                                  v
        events.heartbeat_missed()                  .poll() on a 50ms timer
        clock_drift_ppm (bench diag)               published_state -> system_status

WHY TWO CLASSES -- read this before changing either
===================================================

They look similar. They are not. Conflating them is the most common way a
link monitor of this kind goes wrong.

    HeartbeatLinkMonitor    HOST ARRIVAL TIME
                            age = NOW - (arrival of the last beat)
                            Polluted by USB latency batching, host
                            scheduling and GC pauses.
                            CANNOT distinguish a LATE beat from a LOST one.

    HeartbeatReceiver       THE MCU'S OWN TIMESTAMP
                            delta = this_stamp - previous_stamp
                            Written before the frame reaches USB, so immune
                            to all of the above.
                            CAN distinguish -- but only in RETROSPECT, once
                            the next beat arrives.

Worked example. An FTDI adapter on its factory 16 ms latency timer aliases a
perfect 100 ms cadence into 96 ms and 112 ms arrival buckets:

          MCU stamp      host arrival    MCU delta   arrival gap
  beat 1  1,000,000 us   t = 0.000 s
  beat 2  1,100,000 us   t = 0.112 s     100,000 us     112 ms
  beat 3  1,200,000 us   t = 0.208 s     100,000 us      96 ms

  arrival clock says: 112 ms gap -> LINK_DEGRADED       WRONG
  MCU clock says:     exactly 100,000 us -> nothing lost  RIGHT

Both are correct about their own question. Reacting to them identically is
how a healthy link ends up looking broken -- and how a team learns to ignore
the DEGRADED indicator.

WHY perf_counter() AND NOT monotonic()
======================================

On LINUX -- the AGX target -- there is NO difference. time.get_clock_info()
reports both as clock_gettime(CLOCK_MONOTONIC), same clock, same resolution.
Either would work. perf_counter() is simply the one that was picked.

NEVER move these ages onto time.time(). That is CLOCK_REALTIME and NTP STEPS
it. One correction would make an age jump or go negative and flap LINK_DOWN
on a healthy link. This one DOES matter on the AGX.

perf_counter() and monotonic() have DIFFERENT EPOCHS on some platforms.
Mixing them yields nonsense, so every elapsed-time call in this package moves
together or none of them do.

DATA IN   : decoded heartbeat dicts and a `received_at` sampled ONCE at the
            read boundary in McuLink.run()
            config_loader.AttrDict
DATA OUT  : event dicts returned to McuLink to publish; link state read by
            dashboard/dashboard_mirror.py
CREATED BY: control_service.mcu_link.McuLink.__init__
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

import constants
from config.config_loader import AttrDict
from dashboard_service.services import event_factory as events

LOGGER = logging.getLogger(__name__)

#: The MCU's timestamp_us field is a uint32, so it wraps every ~71.6 minutes.
_TIMESTAMP_MODULUS = 0x1_0000_0000


class HeartbeatReceiver:
    """Decodes heartbeats and counts GENUINELY lost beats, on the MCU clock.

    Answers "was a beat actually lost?" using timestamps the MCU wrote before
    the frame ever reached USB.

    Heartbeats carry NO SEQUENCE NUMBER (HW LLD S3.5.2 payload is
    timestamp_us / state / status_flags / uptime_s), so comparing timestamps
    is the only way to detect a missing beat. Contrast telemetry, where
    trigger_id makes a gap directly visible.

    Attributes:
        config: The validated heartbeat section.
        last_timestamp_us: Previous beat's MCU timestamp, or None.
        missed: Running total of beats detected as lost.
        received: Running total of beats folded in.
        wraps: How many times the MCU's uint32 microsecond counter wrapped.
    """

    def __init__(self, config: AttrDict) -> None:
        """Initialise with no beat yet seen.

        Args:
            config: Validated heartbeat settings, for the interval and the
                drift sample floor.
        """
        self.config = config
        self.last_timestamp_us: int | None = None
        self.missed = 0
        self.received = 0
        self.wraps = 0

        # Clock-drift baseline. Both are captured on the first beat and never
        # moved again, so the drift ratio is measured over the longest
        # available span rather than between two adjacent frames.
        self._first_timestamp_us: int | None = None
        self._first_received_at: float | None = None
        self._last_received_at: float | None = None
        self._drift_samples = 0

    @property
    def expected_step_us(self) -> float:
        """Nominal spacing between two consecutive MCU timestamps.

        Returns:
            The heartbeat interval expressed in microseconds.
        """
        return self.config.interval_s * 1_000_000

    def update(
        self, message: Mapping[str, Any], received_at: float
    ) -> list[dict[str, object]]:
        """Fold in one heartbeat and report beats lost just before it.

        Args:
            message: A decoded heartbeat dict carrying "timestamp_us".
            received_at: perf_counter() sampled ONCE at the read boundary in
                McuLink.run(). Passed in rather than sampled here so that
                every frame in one chunk shares the arrival time it actually
                had -- they came out of one USB transfer -- and so parse time
                never accumulates into the measurement.

        Returns:
            A list containing a heartbeat_missed event when beats were lost,
            otherwise empty.

        Raises:
            Nothing. A malformed timestamp is logged and treated as no loss.
            One bad frame must not kill the read loop or inflate the alarm
            count, and the next beat is only 100 ms away.
        """
        raw = message.get("timestamp_us")
        if raw is None:
            LOGGER.warning("heartbeat has no timestamp_us, ignoring")
            return []

        try:
            timestamp_us = int(raw)
        except (TypeError, ValueError):
            LOGGER.warning(
                "ignoring heartbeat with non-integer timestamp_us %r", raw
            )
            return []

        self.received += 1
        self._last_received_at = received_at

        if self._first_timestamp_us is None:
            self._first_timestamp_us = timestamp_us
            self._first_received_at = received_at

        previous, self.last_timestamp_us = self.last_timestamp_us, timestamp_us
        if previous is None:
            return []  # first beat, nothing to compare against

        self._drift_samples += 1
        delta = timestamp_us - previous

        if delta <= 0:
            # Either the MCU rebooted and restarted its counter, or the
            # uint32 microsecond counter wrapped (every ~71.6 minutes).
            # Neither is a lost beat. McuState.observe_uptime() reports the
            # reboot case properly; counting it here too would report one
            # event as two different alarms.
            #
            # Either way the drift baseline is now meaningless, so it is
            # restarted from this beat.
            self.wraps += 1
            self._first_timestamp_us = timestamp_us
            self._first_received_at = received_at
            self._drift_samples = 0
            LOGGER.debug(
                "MCU timestamp went backwards (%d -> %d): reboot or counter "
                "wrap, not a dropped beat. Drift baseline restarted.",
                previous, timestamp_us,
            )
            return []

        # How many intervals fit in this gap, to the nearest whole one, minus
        # the one that is meant to be there. What is left went missing.
        #
        # round() rather than floor: a 290,000 us gap is far more likely 3
        # intervals with jitter than 2 intervals plus a 90 ms delay.
        #
        # Rounding also IS the on-time cut-off. A gap under half an interval
        # past the nominal -- under 150,000 us -- rounds to 1 and falls out
        # below, so a separate tolerance setting would only restate this
        # boundary, or move it outwards and hide real losses.
        dropped = round(delta / self.expected_step_us) - 1
        if dropped <= 0:
            return []  # on time, or a duplicate/early frame

        self.missed += dropped
        LOGGER.warning(
            "heartbeat gap of %.1f ms on the MCU clock: %d beat(s) genuinely "
            "lost (total %d)", delta / 1000, dropped, self.missed,
        )
        return [events.heartbeat_missed(dropped, self.missed)]

    def clock_drift_ppm(self) -> float | None:
        """Measure how fast the AGX clock runs relative to the MCU clock.

        A BENCH DIAGNOSTIC, not a dashboard field. LLD SW S3.9's event table
        has no clock_drift event; nothing pushes this to the operator.

        It exists to answer LLD SW S5.3's MV1 validation item: "confirm the
        MCU/AGX Orin clock drift is small enough over a multi-hour bench run
        to not require active correction."

        ppm = parts per million. 1 ppm is 1 microsecond of error per second,
        so a 50 ppm crystal drifts 180 ms per hour. Two crystals each +/-50
        ppm can differ by 100 ppm.

        Returns:
            Drift in ppm -- positive means the host clock runs fast relative
            to the MCU -- or None when there are too few samples for the
            ratio to mean anything, or after a counter wrap reset the
            baseline.
        """
        if (
            self._drift_samples < self.config.drift_min_samples
            or self._first_timestamp_us is None
            or self._first_received_at is None
            or self._last_received_at is None
            or self.last_timestamp_us is None
        ):
            return None

        mcu_elapsed_s = (
            self.last_timestamp_us - self._first_timestamp_us
        ) / 1_000_000
        host_elapsed_s = self._last_received_at - self._first_received_at

        if mcu_elapsed_s <= 0:
            return None
        return ((host_elapsed_s / mcu_elapsed_s) - 1.0) * 1_000_000

    def snapshot(self) -> dict[str, object]:
        """Return counters, for logging and diagnostics.

        Returns:
            A plain dict of receive/loss/drift figures. Not a wire payload.
        """
        drift = self.clock_drift_ppm()
        return {
            "heartbeats_received": self.received,
            "heartbeats_missed": self.missed,
            "mcu_counter_wraps": self.wraps,
            "last_timestamp_us": self.last_timestamp_us,
            "clock_drift_ppm": None if drift is None else round(drift, 2),
            "drift_samples": self._drift_samples,
        }


class HeartbeatLinkMonitor:
    """Link health from heartbeat ARRIVAL times, with debouncing.

    Answers "how stale is my information about the MCU?" -- deliberately NOT
    "was a beat lost", which is HeartbeatReceiver's job.

    THE KEY PROPERTY: age = NOW - last_arrival. `now` advances on its own, so
    the age grows whether or not anything arrives. That is precisely how a
    disconnected cable is detected -- the stored arrival time stops moving
    while the clock keeps going. Nothing needs to be received.

    But something must LOOK. poll() is driven by McuLink's own timer, not by
    frame arrival, because a dead link produces no frames and the one failure
    this class exists to catch would otherwise never be reported.

    Attributes:
        config: The validated heartbeat section.
        started_at: perf_counter() at construction, the age reference before
            any beat has arrived.
        last_heartbeat_at: perf_counter() of the most recent beat, or None.
    """

    def __init__(self, config: AttrDict) -> None:
        """Initialise with no beat yet received.

        Args:
            config: Validated heartbeat settings, including the band table
                and the debounce policy.
        """
        self.config = config

        # None, not "now". Seeding this with the current time reports a
        # healthy link to a panel that connects before the MCU has said
        # anything -- with the RS-422 cable unplugged it would show LINK_UP
        # for the first 100 ms and only then decay.
        self.last_heartbeat_at: float | None = None

        self.started_at = time.perf_counter()

        # What the rest of the system has been told. Distinct from the
        # instantaneous state, which may be mid-debounce.
        self._published_state: str | None = None

        # Debounce bookkeeping: a state must hold for debounce_samples
        # consecutive polls before it is published.
        self._candidate_state: str | None = None
        self._candidate_count = 0

        self.transitions_published = 0
        self.transitions_debounced = 0

    def record(self, received_at: float) -> None:
        """Note that a heartbeat arrived.

        Args:
            received_at: perf_counter() sampled ONCE at the read boundary in
                McuLink.run(). See HeartbeatReceiver.update for why this is
                passed in rather than sampled here.
        """
        self.last_heartbeat_at = received_at

    def link_state(self, now: float | None = None) -> tuple[str, float]:
        """Look up the instantaneous link state and heartbeat age.

        LLD SW S3.8a's rationale for the 300 ms LINK_DOWN threshold: it is
        "faster than one trigger interval at every committed speed
        (492.8-574.9ms) -- link death is detected before a second missed
        trigger would even be noticed".

        Args:
            now: perf_counter() reading to evaluate against. Defaults to now.
                MUST come from perf_counter() -- see the module docstring on
                mixing clock epochs.

        Returns:
            (state name, age in seconds). State is one of LINK_UP,
            LINK_DEGRADED or LINK_DOWN. This is the RAW state,
            before debouncing.
        """
        now = time.perf_counter() if now is None else now

        # Before the first beat, age runs from service start -- so a link
        # that never comes up still ages into DEGRADED and then DOWN, rather
        # than sitting at zero forever looking fine.
        reference = (
            self.last_heartbeat_at
            if self.last_heartbeat_at is not None
            else self.started_at
        )

        # max(0.0, ...) guards a caller passing a `now` from slightly before
        # the recorded arrival, which happens when two tasks sample the clock
        # either side of a beat.
        age = max(0.0, now - reference)

        state = constants.LINK_DOWN  # fallthrough: past every band bound
        for limit, name in self.config.bands:
            if age < limit * self.config.interval_s:
                state = name
                break

        return state, age

    def missed_intervals(self, now: float | None = None) -> int:
        """Report how many intervals' worth of silence have elapsed.

        The two DEGRADED bands share a state name, so this is what tells them
        apart on the wire: DEGRADED with 2 is one beat from LINK_DOWN, with 1
        it is a hiccup.

        Returns zero whenever the link is UP, even if a widened UP band could
        hold an age above one interval. Left raw, a 1.5x UP band would report
        LINK_UP alongside missed_intervals=1, which says two contradictory
        things about the same link in the same payload.

        Args:
            now: perf_counter() reading. Defaults to now.

        Returns:
            Whole intervals elapsed since the last beat, or 0 when UP.
        """
        state, age = self.link_state(now)
        if state == constants.LINK_UP:
            return 0
        return int(age / self.config.interval_s)

    def poll(self, now: float | None = None) -> dict[str, object] | None:
        """Evaluate the link and return an event if a change should publish.

        Called on McuLink's own timer at heartbeat.check_interval_s, NOT on
        frame arrival. Two reasons, both load-bearing:

          1. A totally dead link produces no frames. If this were driven by
             arriving data, the one failure it exists to detect could never
             be reported.
          2. Checking after each frame biases the sample. Measured over 6 s
             of mock traffic: checks after a heartbeat saw age 0.0 ms every
             time (trivially UP); checks after a telemetry frame saw
             95-113 ms. A fixed cadence samples evenly.

        DEBOUNCING. A new state must hold for debounce_samples consecutive
        polls before it is published. A beat arriving 112 ms late pushes the
        age past 100 ms for about 12 ms before the beat lands and resets it
        -- a real crossing, but not a real fault. Requiring persistence drops
        that blip while still catching a link whose age keeps climbing.

        LINK_DOWN bypasses the debounce when debounce_exempt_down is set: a
        dead link must never wait for confirmation.

        Args:
            now: perf_counter() reading. Defaults to now.

        Returns:
            None. Always. Kept as a return value so the caller's shape does
            not change if a link event is ever wanted again.
        """
        now = time.perf_counter() if now is None else now
        state, age = self.link_state(now)

        if state == self._published_state:
            # Already reported. Any part-formed candidate is stale.
            self._candidate_state = None
            self._candidate_count = 0
            return None

        if state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = state
            self._candidate_count = 1

        required = self.config.debounce_samples
        if state == constants.LINK_DOWN and self.config.debounce_exempt_down:
            required = 1

        if self._candidate_count < required:
            self.transitions_debounced += 1
            LOGGER.debug(
                "link state %s held %d/%d polls (age %.1f ms), not published "
                "yet", state, self._candidate_count, required, age * 1000,
            )
            return None

        previous = self._published_state
        self._published_state = state
        self._candidate_state = None
        self._candidate_count = 0
        self.transitions_published += 1

        # DOWN and the recovery out of it are what an operator cares about;
        # an intermediate DEGRADED is noise in a log at INFO.
        log = (
            LOGGER.warning
            if state in (constants.LINK_DOWN, constants.LINK_DEGRADED)
            else LOGGER.info
        )
        log(
            "link state %s -> %s (heartbeat age %.1f ms)",
            previous or "(none)", state, age * 1000,
        )

        # NO EVENT. S3.9 lists link_state as a FIELD of system_status, not
        # as an event of its own -- the 500 ms mirror carries it. This method
        # exists to advance published_state (the debounced value the mirror
        # reads) and to log the transition, nothing more.
        return None

    @property
    def published_state(self) -> str:
        """The link state the rest of the system has been told about.

        Returns:
            The last published state. Falls back to LINK_DOWN before the
            first poll -- assume the link is not confirmed until it is.
            dashboard_mirror uses this so system_status and link_state events
            can never disagree with each other.
        """
        return self._published_state or constants.LINK_DOWN

    def snapshot(self) -> dict[str, object]:
        """Return counters, for logging and diagnostics.

        Returns:
            A plain dict of link-monitor figures. Not a wire payload.
        """
        state, age = self.link_state()
        return {
            "link_state": state,
            "published_link_state": self.published_state,
            "heartbeat_age_ms": round(age * 1000, 1),
            "missed_intervals": self.missed_intervals(),
            "transitions_published": self.transitions_published,
            "transitions_debounced": self.transitions_debounced,
        }
