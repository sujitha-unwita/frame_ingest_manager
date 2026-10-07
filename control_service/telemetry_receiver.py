"""Telemetry: decoding trigger frames and proving no fabric went uninspected.

WHERE THIS SITS
===============

    MCU --(MSG_TELEMETRY, one per trigger)--> mcu_link.McuLink._on_telemetry
                                                        |
                                                        v
                                            TelemetryReceiver.update()
                                                        |
                    +-----------------------------------+------------------+
                    |                                                      |
                    v                                                      v
        events.missed_trigger_alarm()                    last_trigger_id / age
        (a gap means uninspected fabric)                 -> dashboard_mirror.py
                                                         -> system_status

WHAT A TRIGGER IS
=================

Per HW LLD S3.5, the MCU fires the strobe and cameras on a fixed
fabric-travel interval driven by the encoder, then sends one TELEMETRY frame
per trigger carrying trigger_id, encoder_count, its own timestamp_us, state
and status_flags. Each frame is a promise that one slice of fabric was
illuminated and captured.

WHY A GAP IS AN ALARM, NOT A LOG LINE
=====================================

LLD SW S3.6, verbatim:

    "trigger_id is a monotonic counter specifically so a gap is visible to
     software as a missed trigger (a system fault), not a silent skip. The
     inference service must ALARM on a gap, not just log it."

A gap means one of:
  - the frame was lost in transit (a second process reading the port, a bad
    cable, a CRC failure), or
  - the MCU fired a trigger the host never learned about.

Either way a slice of fabric moved past the cameras and nothing inspected it.
Nobody reads a backend log while the line is running, and the fabric is on
the roll by then.

CONTRAST WITH HEARTBEATS -- why this file is simpler
====================================================

Telemetry carries trigger_id, a real counter. A gap is DIRECTLY visible:
101 -> 105 means three are missing, known immediately and exactly.

Heartbeats carry NO sequence number (HW LLD S3.5.2), which is why
heartbeat_receiver.py has to infer losses by comparing timestamps and can
only do so retrospectively. Here there is nothing to infer.

THE THREE CASES, AND WHY DUPLICATE IS NOT AN ALARM
==================================================

    ok         id == previous + 1     the normal case
    duplicate  id == previous         the MCU re-sent, or our parser resynced
                                      onto a frame it had already decoded.
                                      NO fabric was missed, so this must NOT
                                      alarm -- crying wolf on a duplicate is
                                      how an operator learns to ignore the
                                      real alarm.
    gap        anything else          fabric passed uninspected. ALARM.

DATA IN   : decoded telemetry dicts and a `received_at` sampled ONCE at the
            read boundary in McuLink.run()
DATA OUT  : event dicts returned to McuLink; last_trigger_id and trigger age
            read by dashboard/dashboard_mirror.py
CREATED BY: control_service.mcu_link.McuLink.__init__
"""

from __future__ import annotations

import logging
import time
from typing import Any, MutableMapping

from dashboard_service.services import event_factory as events

LOGGER = logging.getLogger(__name__)

#: trigger_id is a uint32 on the wire (HW LLD S3.5.2).
TRIGGER_ID_MODULUS = 0x1_0000_0000
TRIGGER_ID_MASK = 0xFFFF_FFFF

#: Sequence verdicts, attached to the decoded message for diagnostics.
SEQUENCE_FIRST = "first"
SEQUENCE_OK = "ok"
SEQUENCE_DUPLICATE = "duplicate"
SEQUENCE_GAP = "gap"


class TelemetryReceiver:
    """Decodes telemetry frames and tracks trigger continuity.

    Like McuState and the heartbeat classes, update() is PURE with respect to
    I/O: it mutates state and RETURNS event dicts for McuLink to publish.
    That keeps this class synchronous and testable with plain dicts, and
    keeps every outbound event flowing through one place.

    Attributes:
        last_trigger_id: Most recent trigger_id seen, or None.
        last_encoder_count: Most recent encoder_count seen, or None.
        last_trigger_at: perf_counter() when the last frame arrived.
        received: Running total of telemetry frames folded in.
        gaps: Running count of gap events raised.
        duplicates: Running count of duplicate frames seen.
        total_missed_triggers: Sum of every gap_count raised.
        counter_resets: How many times trigger_id jumped backwards.
    """

    def __init__(self) -> None:
        """Initialise with no telemetry yet received.

        Takes no config: everything here is derived from the wire format
        rather than tuned. The one value that could have been a knob -- how
        big a gap is worth alarming on -- deliberately is not one, because
        the answer is always "one trigger".
        """
        self.last_trigger_id: int | None = None
        self.last_encoder_count: int | None = None

        # The MCU's own clock at the last trigger. Kept so the trigger
        # interval can be reported on the MCU clock as well as on ours --
        # the two answer different questions, exactly as for heartbeats.
        self.last_timestamp_us: int | None = None

        # None, not "now". Seeding this with the current time would report
        # process uptime as if it were a trigger age -- a panel that has never
        # seen a trigger would show "last trigger 113s ago", which is a fact
        # about the service, not about the fabric.
        #
        # perf_counter once set, matching heartbeat_receiver.py. See its
        # module docstring on why mixing clock epochs yields nonsense.
        self.last_trigger_at: float | None = None

        self.received = 0
        self.gaps = 0
        self.duplicates = 0
        self.total_missed_triggers = 0
        self.counter_resets = 0

    def update(
        self, message: MutableMapping[str, Any], received_at: float
    ) -> list[dict[str, object]]:
        """Fold in one telemetry frame and check trigger continuity.

        Args:
            message: A decoded telemetry dict carrying "trigger_id" and
                optionally "encoder_count". MUTATED to add a "sequence"
                verdict, so anything downstream -- the inference pipeline
                hook, a log line -- can see how this frame related to the
                previous one without recomputing it.
            received_at: perf_counter() sampled ONCE at the read boundary in
                McuLink.run().

        Returns:
            A list containing a missed_trigger_alarm event when a gap is
            detected, otherwise empty.

        Raises:
            Nothing. A malformed trigger_id is logged and dropped. One bad
            frame must not kill the read loop and -- more importantly -- must
            not be mistaken for a gap and raise a false uninspected-fabric
            alarm.
        """
        raw = message.get("trigger_id")
        if raw is None:
            LOGGER.warning("telemetry frame has no trigger_id, ignoring")
            return []

        try:
            trigger_id = int(raw) & TRIGGER_ID_MASK
        except (TypeError, ValueError):
            LOGGER.warning(
                "ignoring telemetry with non-integer trigger_id %r", raw
            )
            return []

        self.received += 1
        self.last_trigger_at = received_at
        raw_stamp = message.get("timestamp_us")
        if raw_stamp is not None:
            try:
                self.last_timestamp_us = int(raw_stamp)
            except (TypeError, ValueError):
                LOGGER.warning("telemetry timestamp_us not an int: %r", raw_stamp)

        raw_encoder = message.get("encoder_count")
        if raw_encoder is not None:
            try:
                self.last_encoder_count = int(raw_encoder)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "ignoring non-integer encoder_count %r", raw_encoder
                )

        previous = self.last_trigger_id
        self.last_trigger_id = trigger_id

        if previous is None:
            message["sequence"] = SEQUENCE_FIRST
            LOGGER.info("first trigger observed: id=%d", trigger_id)
            return []

        expected = (previous + 1) & TRIGGER_ID_MASK

        if trigger_id == expected:
            message["sequence"] = SEQUENCE_OK
            return []

        if trigger_id == previous:
            # The MCU re-sent, or our parser resynced onto a frame it had
            # already decoded. No fabric was missed. Counted, logged at
            # debug, and deliberately NOT alarmed.
            self.duplicates += 1
            message["sequence"] = SEQUENCE_DUPLICATE
            LOGGER.debug("duplicate telemetry for trigger_id=%d", trigger_id)
            return []

        # Modular arithmetic, so a wrap past 0xFFFFFFFF yields a small
        # positive gap rather than a ~4.3 billion one. At the 0.5 s committed
        # cadence a real wrap takes about 68 years, so this will not happen in
        # service -- it is handled anyway, because the alternative is a single
        # false alarm nobody alive could explain from the logs.
        gap_count = (trigger_id - expected) & TRIGGER_ID_MASK

        # A gap_count in the top half of the uint32 range is not a forward
        # jump of two billion triggers -- it is a BACKWARDS jump, meaning the
        # MCU restarted and its counter reset. Reporting that as billions of
        # uninspected triggers would bury the real signal.
        if gap_count > TRIGGER_ID_MODULUS // 2:
            self.counter_resets += 1
            LOGGER.warning(
                "trigger_id went backwards (%d -> %d): MCU counter reset, not "
                "a gap. Continuity tracking restarts from here.",
                previous, trigger_id,
            )
            message["sequence"] = SEQUENCE_FIRST
            return []

        self.gaps += 1
        self.total_missed_triggers += gap_count
        message["sequence"] = SEQUENCE_GAP
        message["expected_trigger_id"] = expected
        message["gap_count"] = gap_count

        LOGGER.error(
            "TRIGGER GAP: expected %d, got %d -- %d trigger(s) of fabric "
            "passed UNINSPECTED (total %d)",
            expected, trigger_id, gap_count, self.total_missed_triggers,
        )
        return [
            events.missed_trigger_alarm(
                expected_trigger_id=expected,
                last_seen_trigger_id=previous,
                gap_count=gap_count,
            )
        ]

    def trigger_age_s(self, now: float | None = None) -> float | None:
        """Report how long it has been since the last trigger arrived.

        While the MCU is paused it fires no triggers, so this grows without
        bound -- which is correct, and is what the panel should show. A paused
        line genuinely has not inspected anything recently.

        Args:
            now: perf_counter() reading. Defaults to now. MUST come from
                perf_counter() -- see heartbeat_receiver.py on clock epochs.

        Returns:
            Age in seconds, never negative; or None if no trigger has EVER
            arrived. None is not zero and it is not a large number -- callers
            must render it as "unknown", never as an age.
        """
        if self.last_trigger_at is None:
            return None
        now = time.perf_counter() if now is None else now
        return max(0.0, now - self.last_trigger_at)

    def snapshot(self) -> dict[str, object]:
        """Return counters, for logging and diagnostics.

        Returns:
            A plain dict of continuity counters. Not a wire payload.
        """
        return {
            "telemetry_received": self.received,
            "last_trigger_id": self.last_trigger_id,
            "last_encoder_count": self.last_encoder_count,
            "trigger_gaps": self.gaps,
            "duplicate_triggers": self.duplicates,
            "total_missed_triggers": self.total_missed_triggers,
            "trigger_counter_resets": self.counter_resets,
        }
