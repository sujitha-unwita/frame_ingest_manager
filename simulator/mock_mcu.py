"""The EXPECTED MCU: a simulator that behaves as the real firmware should.

WHERE THIS SITS
===============

    simulator/mock_mcu.py  MockMcu     <-- this file, stands in for the STM32
            |
            |  subclasses the SAME Transport ABC as SerialTransport:
            |      async open() / read() / write(frame) / close()
            v
    control_service/mcu_link.py  McuLink   <-- cannot tell which one it holds

Selected by `transport.kind: mock` in config.yaml, or `main.py --mock`.

WHY THIS EMITS REAL BYTES AND NOT FAKE DICTS
============================================

The tempting shortcut is to hand McuLink ready-made message dictionaries.
That would be worthless -- it would bypass, and therefore leave untested,
every part of the system most likely to be wrong:

    framing            STX / MSG_TYPE / LEN / PAYLOAD / CRC8 / ETX
    CRC-8              polynomial, init, xorout, whether STX is covered
    struct packing     field order, endianness, widths
    stream reassembly  frames split across chunk boundaries
    resynchronisation  recovery after a corrupt byte

So MockMcu builds genuine frames with build_unified_frame() and returns raw
bytes. The parser, the state logic and the command retry exercised against
this mock are the SAME CODE that will run against the real MCU.

WHAT THE REAL MCU IS EXPECTED TO DO -- and what this models
==========================================================

    HW LLD S3.5.2   heartbeat every 100 ms, from a HARDWARE timer
    HW LLD S3.5.2   telemetry, one frame per trigger
    HW LLD S3.5.2   "MCU must remember the last-processed seq and re-ACK a
                     retried command rather than re-executing it"
    HW LLD S3.5.3   IDLE / RUNNING / PAUSED / FAULT, and
                     "FAULT->RUNNING only on a RESUME command AND fault bits
                      clear (both required -- a command alone can't clear a
                      real hardware fault)"

ON THE HEARTBEAT CADENCE -- a bug worth remembering
===================================================

An earlier revision slept for the interval AFTER building each frame, so the
period was 100 ms PLUS processing time and drifted a few ms every beat. That
produced heartbeat ages of 103-111 ms and flapped LINK_DEGRADED on a link
that had missed nothing -- and it nearly led to the band table being widened
to hide it.

The real STM32 fires from a hardware timer interrupt: the period is fixed and
errors do NOT accumulate. This mock advances an ABSOLUTE DEADLINE to model
that. The lesson generalises: when the simulator and the spec disagree, fix
the simulator, never the threshold.

EXPECTED RESIDUAL JITTER ON WINDOWS -- measured, not assumed
============================================================

asyncio.sleep() here is bounded by the Windows scheduler quantum of
15.625 ms, so measured gap distribution over 120 beats was BIMODAL:

    90- 95 ms  #############                    (13)
    95-100 ms  ########################...####  (72)
   100-105 ms  ####                             (4)
   105-110 ms  #####                            (5)
   110-115 ms  ##########################       (26)

Mean 100.0 ms with no drift, but nothing in the 100-110 valley. That is the
SENDER's scheduler. The real AGX never generates the heartbeat -- it only
receives one whose timing comes from the STM32's crystal -- so this has no
equivalent on the real link.

DO NOT widen heartbeat.bands to hide it. heartbeat.debounce_samples exists
precisely so a transient crossing of any size is ignored without moving a
threshold the LLD specifies.

DATA IN   : command frames from mcu_command.McuCommandSender, via McuLink
            config_loader: AttrDict, AttrDict, AttrDict
DATA OUT  : raw protocol bytes, fed to McuLink's UnifiedStreamParser
CREATED BY: control_service.mcu_link.build_transport()
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time

from config.config_loader import AttrDict
from control_service.protocol import build_unified_frame, status_flag_bit
from control_service.transport import Transport

LOGGER = logging.getLogger(__name__)

#: Byte offsets into a COMMAND frame: STX, MSG_TYPE, LEN, then the payload.
_COMMAND_SEQ_OFFSET = 3
_COMMAND_CODE_OFFSET = 4
_COMMAND_FRAME_MIN_LEN = 7

#: uint32 microsecond counter, as the firmware declares it.
_TIMESTAMP_MASK = 0xFFFF_FFFF

#: uint16 uptime, as the firmware declares it.
_UPTIME_MASK = 0xFFFF

#: uint32 trigger_id.
_TRIGGER_MASK = 0xFFFF_FFFF

#: ACK result byte meaning success.
_ACK_OK = 0
# Real MCU rejects PAUSE while ARMED with result=3.
_ACK_PAUSE_REJECTED = 3


class MockMcu(Transport):
    """A simulated VisionBar MCU speaking the real wire protocol.

    Implements the same Transport ABC as SerialTransport, so McuLink is
    constructed identically either way.

    Attributes:
        state: Current MCU state code.
        status_flags: Current raw flags byte.
        trigger_id: Most recent trigger id emitted.
        encoder_count: Simulated encoder position.
        drop_acks: Remaining ACKs to swallow, for retry testing.
        drop_triggers: Trigger ids to skip on the next telemetry frame.
        frames_sent: Running total of frames produced.
        commands_received: Running total of command frames accepted.
    """

    def __init__(
        self,
        simulator_config: AttrDict,
        heartbeat_config: AttrDict,
        protocol_config: AttrDict,
        states: AttrDict,
        encoder_config: AttrDict,
        command_codes: dict[str, int],
    ) -> None:
        """Build a simulated MCU in the RUNNING state.

        Args:
            simulator_config: Trigger cadence, encoder step, fault injection.
            heartbeat_config: Beat interval, so the mock and the monitor
                cannot disagree about what "on time" means.
            protocol_config: Validated framing, message codes and flag settings.
            command_codes: Configured codes for pause, resume and stop.
            states: The MCU state code points.
        """
        self.simulator_config = simulator_config
        self.heartbeat_config = heartbeat_config
        self.protocol_config = protocol_config
        self.command_codes = command_codes
        self.states = states
        self.encoder_config = encoder_config
        # Boots into whichever state config.yaml names. "armed" is the
        # default, so `main.py serve --mock` exercises the START CAPTURE flow
        # rather than starting triggers on its own. [OURS -- Pooja]
        self._initial_state = getattr(
            states, simulator_config.initial_state
        )
        self.state = self._initial_state
        self.status_flags = 0

        self.trigger_id = 0
        self.encoder_count = 0

        self.drop_acks = simulator_config.drop_acks
        self.drop_triggers = simulator_config.drop_triggers

        self._booted_at = time.perf_counter()
        self._uptime_offset_s = simulator_config.initial_uptime_s

        # Absolute deadlines, not sleep(interval). See the module docstring
        # on why this distinction produced a real bug.
        now = time.perf_counter()
        self._next_beat_at = now
        self._next_trigger_at = now + simulator_config.trigger_interval_s

        # An ACK waiting to be returned by the next read(). One slot is
        # enough: McuCommandSender serialises commands, so a second cannot
        # arrive before the first is collected.
        self._pending_ack: bytes = b""

        # Last sequence acted on, for the S3.5.2 idempotency rule.
        self._last_command_sequence: int | None = None
        self._last_command_result = _ACK_OK

        self._closed = False
        self.frames_sent = 0
        self.commands_received = 0

    # ---- lifecycle -------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """Whether the simulated link is usable.

        Returns:
            True until close() is called.
        """
        return not self._closed

    async def open(self) -> None:
        """Start the simulated MCU. Resets both cadence deadlines.

        Resetting matters: without it, a mock constructed well before open()
        would fire a burst of backdated beats to catch up.
        """
        self._closed = False
        now = time.perf_counter()
        self._booted_at = now
        self._next_beat_at = now
        self._next_trigger_at = now + self.simulator_config.trigger_interval_s
        LOGGER.info(
            "MockMcu started: state=%s, heartbeat every %.0f ms, trigger "
            "every %.0f ms",
            self.states.names.get(self.state, self.state),
            self.heartbeat_config.interval_s * 1000,
            self.simulator_config.trigger_interval_s * 1000,
        )

    async def close(self) -> None:
        """Stop the simulated MCU. Safe to call more than once."""
        self._closed = True
        LOGGER.info(
            "MockMcu stopped after %d frames, %d commands",
            self.frames_sent, self.commands_received,
        )

    # ---- simulated firmware state ---------------------------------------

    def _uptime_s(self) -> int:
        """Compute simulated uptime, wrapping like the firmware's uint16.

        Returns:
            Seconds since boot, masked to 16 bits.
        """
        elapsed = int(time.perf_counter() - self._booted_at)
        return (elapsed + self._uptime_offset_s) & _UPTIME_MASK

    def _timestamp_us(self) -> int:
        """Produce an MCU-style microsecond timestamp.

        perf_counter is used so the value has genuine microsecond resolution.
        On monotonic()'s 15.625 ms Windows grid this counter would step in
        15625 us jumps, making it useless as the precise relative reference
        HeartbeatReceiver needs it to be.

        Returns:
            Microseconds, masked to 32 bits like the firmware's counter.
        """
        return int(time.perf_counter() * 1e6) & _TIMESTAMP_MASK

    def reboot(self) -> None:
        """Simulate a crash and restart: uptime resets, beats never stop.

        Exercises the one path that detects an MCU crash. Because heartbeats
        resume within 100 ms, link_state stays LINK_UP throughout and the
        uptime going backwards is the ONLY evidence the host ever gets
        (HW LLD S3.5.2).
        """
        self._booted_at = time.perf_counter()
        self._uptime_offset_s = 0
        self.state = self._initial_state
        self._last_command_sequence = None
        self._last_command_result = _ACK_OK
        LOGGER.info("MockMcu simulated reboot: uptime reset to 0")

    def raise_flag(self, name: str) -> None:
        """Set one status_flags bit by name.

        Args:
            name: A flag name from the configured status_flags, e.g.
                "panel_estop" or "strobe_fault".

        Raises:
            ValueError: If the flag name is unknown. Deliberately fatal -- a
                mistyped name that silently set the wrong bit would report
                the wrong fault to the operator.
        """
        self.status_flags |= 1 << status_flag_bit(name, self.protocol_config.status_flags)
        if name == "panel_estop":
            # HW LLD S3.5.3: RUNNING/PAUSED -> FAULT autonomously on any
            # fault bit, immediate, "an e-stop can't wait for an in-flight
            # trigger".
            self.state = self.states.fault
        LOGGER.info(
            "MockMcu raised flag %s (flags=0x%02X, state=%s)",
            name, self.status_flags,
            self.states.names.get(self.state, self.state),
        )

    def clear_flag(self, name: str) -> None:
        """Clear one status_flags bit by name.

        Clearing the last flag leaves the MCU in IDLE, not RUNNING. Per
        S3.5.3 a fault that has been cleared does not restart the line by
        itself -- an operator must explicitly RESUME. That is the safe
        behaviour and what the real firmware does.

        Args:
            name: A flag name from configured status_flags.

        Raises:
            ValueError: If the flag name is unknown.
        """
        self.status_flags &= ~(1 << status_flag_bit(name, self.protocol_config.status_flags)) & 0xFF
        if self.status_flags == 0 and self.state == self.states.fault:
            self.state = self.states.idle
        LOGGER.info(
            "MockMcu cleared flag %s (flags=0x%02X, state=%s)",
            name, self.status_flags,
            self.states.names.get(self.state, self.state),
        )

    # ---- frame builders --------------------------------------------------

    def _build_heartbeat(self) -> bytes:
        """Build one MSG_HEARTBEAT frame.

        Payload is 8 bytes, HW LLD S3.5.2:
        <IBBH> = timestamp_us, state, status_flags, uptime_s.

        NOTE: no sequence number, exactly as the real firmware. That is why
        HeartbeatReceiver must infer losses from timestamps.

        Returns:
            The complete framed heartbeat.
        """
        payload = struct.pack(
            "<IBBH",
            self._timestamp_us(),
            self.state,
            self.status_flags,
            self._uptime_s(),
        )
        return build_unified_frame(
            self.protocol_config.message_types["heartbeat"],
            payload,
            self.protocol_config.crc_config,
            stx=self.protocol_config.stx,
            etx=self.protocol_config.etx
        )

    def _build_telemetry(self) -> bytes:
        """Build telemetry with the seven fields in the 19-byte wire format."""
        encoder_step = self.simulator_config.encoder_counts_per_trigger

        # Advance over skipped triggers to simulate lost telemetry.
        if self.drop_triggers > 0:
            skipped = self.drop_triggers
            self.trigger_id = (self.trigger_id + skipped) & _TRIGGER_MASK
            self.encoder_count += skipped * encoder_step
            self.drop_triggers = 0
            LOGGER.info("MockMcu skipped %d trigger IDs", skipped)

        # Advance to the current trigger.
        self.trigger_id = (self.trigger_id + 1) & _TRIGGER_MASK
        self.encoder_count += encoder_step

        # Convert encoder position to the wire unit: millimetres multiplied by 1000.
        distance_mm_x1000 = round(
            self.encoder_count * self.encoder_config.mm_per_count * 1000
        )

        if encoder_step > 0:
            direction = 1
        elif encoder_step < 0:
            direction = -1
        else:
            direction = 0

        payload = struct.pack(
            "<IiiIbBB",
            self.trigger_id,       # uint32
            self.encoder_count,    # int32
            distance_mm_x1000,     # int32
            self._timestamp_us(),  # uint32
            direction,             # int8
            self.state,            # uint8
            self.status_flags,     # uint8
        )

        return build_unified_frame(
            self.protocol_config.message_types["telemetry"],
            payload,
            self.protocol_config.crc_config,
            stx=self.protocol_config.stx,
            etx=self.protocol_config.etx
        )
    def _build_ack(self, sequence: int, code: int, result: int) -> bytes:
        """Build one MSG_ACK_NACK frame.

        Payload is 4 bytes, HW LLD S3.5.2:
        <BBBB> = cmd_seq, cmd_code, result, state (post-transition).

        Args:
            sequence: The command sequence being answered.
            code: The command code being answered.
            result: 0 for ACK, non-zero for NACK.

        Returns:
            The complete framed ACK.
        """
        payload = struct.pack("<BBBB", sequence, code, result, self.state)
        return build_unified_frame(
            self.protocol_config.message_types["ack_nack"],
            payload,
            self.protocol_config.crc_config,
            stx=self.protocol_config.stx,
            etx=self.protocol_config.etx
        )

    # ---- I/O -------------------------------------------------------------

    async def read(self) -> bytes:
        """Return the next frame the simulated MCU would send.

        Sleeps until whichever is due first -- the next heartbeat or the next
        trigger -- so the two cadences interleave the way they do on the real
        link rather than one starving the other.

        Returns:
            One complete frame. A pending ACK jumps the queue, matching real
            firmware, which answers a command immediately rather than waiting
            for its next scheduled transmission.
        """
        if self._closed:
            # Close enough to a closed real port for McuLink's loop: nothing
            # arrives, and the caller keeps waiting. The link monitor's own
            # timer then correctly ages into LINK_DOWN.
            await asyncio.sleep(self.heartbeat_config.interval_s)
            return b""

        if self._pending_ack:
            frame, self._pending_ack = self._pending_ack, b""
            self.frames_sent += 1
            return frame

        # While not RUNNING the MCU fires no triggers, so only the heartbeat
        # is scheduled. Pushing the trigger deadline forward stops it firing a
        # catch-up burst the moment the line resumes.
        running = self.state == self.states.running
        now = time.perf_counter()
        if not running:
            self._next_trigger_at = (
                now + self.simulator_config.trigger_interval_s
            )

        due_at = min(self._next_beat_at, self._next_trigger_at)
        delay = due_at - now
        if delay > 0:
            await asyncio.sleep(delay)

        # A command may have arrived while we slept; answer it first.
        if self._pending_ack:
            frame, self._pending_ack = self._pending_ack, b""
            self.frames_sent += 1
            return frame

        now = time.perf_counter()
        if self._next_beat_at <= self._next_trigger_at:
            self._next_beat_at = self._advance(
                self._next_beat_at, self.heartbeat_config.interval_s, now
            )
            self.frames_sent += 1
            return self._build_heartbeat()

        self._next_trigger_at = self._advance(
            self._next_trigger_at,
            self.simulator_config.trigger_interval_s,
            now,
        )
        self.frames_sent += 1
        return self._build_telemetry()

    @staticmethod
    def _advance(deadline: float, interval: float, now: float) -> float:
        """Advance an absolute deadline by one interval, resynchronising.

        Advancing the deadline -- rather than sleeping for the interval --
        is what stops the period drifting by the processing time of each
        frame. See the module docstring.

        Args:
            deadline: The deadline that just fired.
            interval: Nominal period.
            now: Current perf_counter() reading.

        Returns:
            The next deadline. If the caller fell far behind -- a debugger
            pause, a long GC -- the deadline resynchronises to now + interval
            instead of firing a burst of backdated frames to catch up.
        """
        nxt = deadline + interval
        if nxt < now:
            return now + interval
        return nxt

    async def write(self, frame: bytes) -> None:
        """Accept a COMMAND frame and queue the ACK the real MCU would send.

        Implements the HW LLD S3.5.2 idempotency rule: a REPEATED cmd_seq is
        re-ACKed but NOT re-executed. That rule is what makes the host's retry
        safe when an ACK was merely lost rather than the command missed --
        without it, a retried PAUSE could toggle the line twice.

        Args:
            frame: A complete command frame from McuCommandSender.
        """
        self.commands_received += 1

        if len(frame) < _COMMAND_FRAME_MIN_LEN:
            LOGGER.warning(
                "MockMcu received a runt frame (%d bytes), ignoring: %s",
                len(frame), frame.hex(" "),
            )
            return

        try:
            sequence = frame[_COMMAND_SEQ_OFFSET]
            code = frame[_COMMAND_CODE_OFFSET]
        except IndexError:
            LOGGER.warning(
                "MockMcu could not read seq/code from %s", frame.hex(" ")
            )
            return

        # Idempotency: act only on a sequence not seen before.
        if sequence != self._last_command_sequence:
            self._last_command_result = self._apply_command(code)
            self._last_command_sequence = sequence
        else:
            LOGGER.info(
                "MockMcu re-ACKing duplicate sequence %d without re-executing",
                sequence,
            )

        if self.drop_acks > 0:
            self.drop_acks -= 1
            LOGGER.info(
                "MockMcu dropping ACK for seq=%d (%d drop(s) remaining)",
                sequence, self.drop_acks,
            )
            return

        self._pending_ack = self._build_ack(
            sequence, code, result=self._last_command_result,
        )

    def _apply_command(self, code: int) -> int:
        """Apply one command code to the simulated state machine.

        Args:
            code: The wire command code point.
        """
        if code == self.command_codes["pause"]:
            if self.state == self.states.armed:
                LOGGER.info("MockMcu rejecting PAUSE while ARMED")
                return _ACK_PAUSE_REJECTED
            # HW LLD S3.5.3: graceful. Real firmware finishes the in-flight
            # trigger first; the mock has no in-flight concept, so the state
            # changes immediately.
            self.state = self.states.paused
        elif code == self.command_codes["resume"]:
            # S3.5.3: "FAULT->RUNNING only on a RESUME command AND fault bits
            # clear (both required -- a command alone can't clear a real
            # hardware fault)".
            if self.state == self.states.fault:
                LOGGER.info("MockMcu ignoring RESUME while in FAULT")
                return _ACK_OK
            self.state = self.states.running
        elif code == self.command_codes["stop"]:
            self.state = self.states.idle
        else:
            LOGGER.warning("MockMcu received unknown command code 0x%02X", code)
            return _ACK_OK

        LOGGER.info(
            "MockMcu state is now %s",
            self.states.names.get(self.state, self.state),
        )
        return _ACK_OK

    def snapshot(self) -> dict[str, object]:
        """Return counters, for logging and diagnostics.

        Returns:
            A plain dict of mock counters. Not a wire payload.
        """
        return {
            "transport": "mock",
            "state": self.states.names.get(self.state, self.state),
            "status_flags": self.status_flags,
            "trigger_id": self.trigger_id,
            "encoder_count": self.encoder_count,
            "frames_sent": self.frames_sent,
            "commands_received": self.commands_received,
            "uptime_s": self._uptime_s(),
        }
