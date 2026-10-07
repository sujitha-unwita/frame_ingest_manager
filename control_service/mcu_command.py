"""Outbound commands, and the ACK correlation that answers them.

Two classes, one file:

    AckReceiver        the CORRELATOR. Holds NO transport. Cannot read.
    McuCommandSender   validates, frames, writes, retries.

WHERE THIS SITS
===============

    panel --(operator_stop_request)--> dashboard_ws.handle_command
                                              |   validates the operator_* name
                                              v
                                    McuCommandSender.send("pause")
                                              |
                        +---------------------+---------------------+
                        |                                           |
                        v                                           v
              transport.write(frame)                await acks.wait_for(seq)
                        |                                           ^
                        v                                           |
                      MCU                                           |
                        |                                           |
                   (ACK frame)                                      |
                        v                                           |
              transport.read() -> McuLink._dispatch ----------------+
                                          acks.offer(msg)

THE POINT OF THIS FILE: THE ACK WAIT IS A SUBSCRIPTION, NOT A READ
==================================================================

The obvious implementation -- write the frame, then read the port until an
ACK appears -- is wrong, and wrong in a way that only shows up in production.

The ACK does not arrive alone. It arrives interleaved into a stream already
carrying a heartbeat every 100 ms and a telemetry frame per trigger. Code
that reads the port hunting for an ACK CONSUMES all of that and throws it
away. One attempt at a 200 ms timeout silently eats two heartbeats. Three
attempts eat six -- 600 ms of silence -- which drives link_state to
LINK_DOWN.

So the operator presses PAUSE and the panel reports the link as dead. Every
single time. That is exactly what the original mcu_command_client.py did.

The fix is structural: exactly ONE piece of code reads the port
(McuLink.run), and it routes ACK frames here through AckReceiver.offer().

AckReceiver HOLDS NO TRANSPORT REFERENCE. It physically cannot read the port.
The rule is enforced by the object graph, not by a comment.

IDEMPOTENCY: WHY THE SEQUENCE IS ALLOCATED OUTSIDE THE RETRY LOOP
=================================================================

HW LLD S3.5.2: "MCU must remember the last-processed seq and re-ACK a retried
command rather than re-executing it (idempotency)". LLD SW S3.8a: "A retried
RESUME after a lost ACK must not double-fire anything."

So a retry must carry the SAME cmd_seq as the original -- that is what makes
retrying safe when the first ACK was merely lost rather than the command
missed.

The mirror-image bug matters just as much: reusing an already-burned sequence
for a NEW command has the MCU discard it as a duplicate, and the new command
is silently ignored. So a sequence is allocated ONCE per logical command,
outside the loop, and never reused.

WHY A LOCK
==========

_lock serialises whole COMMANDS, not port access. Two operators pressing
PAUSE and RESUME at the same instant would otherwise interleave their sends
and their ACK waits, and each could match the other's ACK.

Note what the lock does NOT wrap: the read loop. Holding a lock across the
600 ms worst-case retry window while the reader waited on it would
reintroduce exactly the starvation this file exists to avoid.

KNOWN LIMITATION -- STOP LATENCY
================================

Commands are strictly serialised, so a STOP arriving while a RESUME is
mid-retry waits up to max_attempts x ack_timeout_s (600 ms by default). A
priority queue or preemption would fix it. Deliberately NOT built yet: it is
a separate concern from the link ownership this package exists to get right,
and mixing the two would make both harder to review.

DATA IN   : command names from dashboard_ws.handle_command(), via McuLink
            ACK frames from McuLink._dispatch(), via AckReceiver.offer()
DATA OUT  : CommandResult, returned up to the dashboard
            command frames, written to the transport
CREATED BY: control_service.mcu_link.McuLink.__init__
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from config.config_loader import AttrDict
from control_service.protocol import build_command_frame

LOGGER = logging.getLogger(__name__)

#: Signature of the transport write used to put a frame on the wire.
FrameWriter = Callable[[bytes], Awaitable[None]]

#: cmd_seq reported when a command failed before a sequence was allocated --
#: an unknown command name, for instance.
NO_SEQUENCE = -1

#: Bound on the ACK queue. A burst of unmatched ACKs -- a firmware fault, or
#: line noise decoding as valid frames -- must not grow memory without limit.
ACK_QUEUE_MAXSIZE = 32


class CommandError(Exception):
    """Raised when a command cannot even be attempted.

    Distinct from a command that WAS attempted and not acknowledged, which is
    reported as CommandResult(acknowledged=False) rather than raised. The
    difference matters to the caller: this exception means the request was
    malformed or misconfigured, while an unacknowledged result means the MCU
    was asked and did not answer.
    """


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one logical command, retries included.

    Attributes:
        command: The command name as requested, e.g. "pause".
        command_sequence: The sequence used for every attempt, or NO_SEQUENCE
            if the command failed before allocation.
        acknowledged: True only if the MCU returned a matching ACK whose
            result byte equals commands.ack_result_ok.
        attempts: How many times the frame actually reached the wire.
        reason: Empty on success; a human-readable cause on failure.
        mcu_state: The MCU state carried on the ACK, if one arrived.
        nacked: True when the MCU ANSWERED but refused. Distinct from
            silence -- a NACK means the link is fine and the MCU declined,
            which no retry will fix.
    """

    command: str
    command_sequence: int
    acknowledged: bool
    attempts: int
    reason: str = ""
    mcu_state: int | None = None
    nacked: bool = False


class AckReceiver:
    """Rendezvous point where a dispatched ACK meets the coroutine waiting.

    NOT A READER. This class holds no transport and has no way to reach the
    serial port. McuLink's single read loop hands ACK frames in through
    offer(); McuCommandSender waits for one through wait_for(). That is the
    whole of it.

    Attributes:
        offered: Running total of ACK frames handed in.
        matched: Running total returned to a waiting sender.
        unmatched: ACKs discarded because they matched no in-flight command.
        dropped: ACKs lost to a full queue.
    """

    def __init__(self, maxsize: int = ACK_QUEUE_MAXSIZE) -> None:
        """Create an empty correlator.

        Args:
            maxsize: Queue bound. On overflow the OLDEST entry is discarded.
        """
        self._queue: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue(
            maxsize=maxsize
        )
        self.offered = 0
        self.matched = 0
        self.unmatched = 0
        self.dropped = 0

    def offer(self, message: Mapping[str, Any]) -> None:
        """Hand an ACK frame to whichever command is waiting for one.

        Called by McuLink._dispatch from inside the read loop. NEVER BLOCKS:
        the read loop must not be able to stall behind a full queue, because
        stalling it stops heartbeat processing too.

        Args:
            message: A decoded ack_nack dict.
        """
        self.offered += 1
        try:
            self._queue.put_nowait(message)
            return
        except asyncio.QueueFull:
            pass

        # Drop the OLDEST to make room. A stale ACK cannot match any
        # in-flight command -- send() drains the queue before each attempt --
        # so losing it costs nothing, while dropping the NEWEST could lose the
        # one being waited on right now.
        try:
            discarded = self._queue.get_nowait()
            self._queue.put_nowait(message)
            self.dropped += 1
            LOGGER.warning(
                "ACK queue full, discarded stale ACK for seq=%s",
                discarded.get("command_sequence"),
            )
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            # Raced with a consumer. This ACK is lost, the command will
            # retry, and retry is idempotent. Nothing further to do.
            self.dropped += 1
            LOGGER.warning("ACK queue contention, dropped one ACK")

    async def wait_for(
        self, sequence: int, code: int, timeout: float
    ) -> Mapping[str, Any] | None:
        """Wait for an ACK matching BOTH the sequence and the command code.

        BOTH must match. Sequence alone is not enough: cmd_seq is one byte
        (HW LLD S3.5.2) so it wraps every 256 commands, and an ancient
        delayed ACK could collide. Requiring the code as well makes a false
        match need two independent coincidences.

        Non-matching frames are discarded and the wait CONTINUES on the
        remaining budget rather than restarting it. Restarting would let a
        stream of unmatched ACKs extend one attempt indefinitely.

        Args:
            sequence: The command sequence being awaited.
            code: The command code being awaited.
            timeout: Total seconds to wait.

        Returns:
            The matching ACK message, or None if the timeout expired first.

        Raises:
            asyncio.CancelledError: Propagated on shutdown. Swallowing it
                would make shutdown hang.
        """
        deadline = time.perf_counter() + timeout

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None

            try:
                ack = await asyncio.wait_for(
                    self._queue.get(), timeout=remaining
                )
            except asyncio.TimeoutError:
                return None

            if (
                ack.get("command_sequence") == sequence
                and ack.get("command_code") == code
            ):
                self.matched += 1
                return ack

            self.unmatched += 1
            LOGGER.debug(
                "discarding non-matching ACK (seq=%s, code=%s); awaiting "
                "seq=%d, code=%d",
                ack.get("command_sequence"), ack.get("command_code"),
                sequence, code,
            )

    def drain(self) -> int:
        """Empty the queue, discarding anything left from a prior command.

        Called before each attempt. Anything present predates the frame about
        to be written, so it cannot be that attempt's answer.

        Returns:
            How many stale entries were discarded.
        """
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            drained += 1
        if drained:
            LOGGER.debug("drained %d stale ACK(s) before sending", drained)
        return drained

    def snapshot(self) -> dict[str, object]:
        """Return counters, for logging and diagnostics.

        Returns:
            A plain dict of correlator counters. Not a wire payload.
        """
        return {
            "acks_offered": self.offered,
            "acks_matched": self.matched,
            "acks_unmatched": self.unmatched,
            "acks_dropped": self.dropped,
        }


class McuCommandSender:
    """Validates, frames, sends and retries operator commands.

    Owns the sequence allocator and the retry policy. Delegates the ACK wait
    to an injected AckReceiver, and never touches the port for reading.

    Attributes:
        config: Validated commands section -- timeout, attempts, codes.
        protocol: Validated protocol section, for the CRC used when framing.
        acks: The correlator this sender waits on.
        sent: Running total of frames written, retries counted separately.
        acknowledged: Running total of commands that succeeded.
        failed: Running total of commands that failed or were refused.
    """

    def __init__(
        self,
        config: AttrDict,
        protocol: AttrDict,
        write_frame: FrameWriter,
        acks: AckReceiver,
    ) -> None:
        """Wire the sender to a transport's write method and a correlator.

        Args:
            config: Validated commands settings from config.yaml.
            protocol: Validated protocol settings, for the CRC parameters.
            write_frame: Awaitable that puts raw bytes on the wire. Injected
                rather than taking the transport itself, so this class cannot
                accidentally grow a call to transport.read() -- the one thing
                it must never do.
            acks: The AckReceiver McuLink also routes ACK frames into.
        """
        self.config = config
        self.protocol = protocol
        self.acks = acks
        self._write_frame = write_frame

        # Serialises whole commands. See the module docstring.
        self._lock = asyncio.Lock()

        # cmd_seq is a uint8 on the wire, so it wraps at 256. Starts at 0 and
        # is pre-incremented, so the first command uses 1.
        self._sequence = 0

        self.sent = 0
        self.acknowledged = 0
        self.failed = 0

    async def send(self, command: str) -> CommandResult:
        """Send one command and wait for its ACK, retrying on silence.

        Args:
            command: A key of config.commands.codes, e.g. "pause".

        Returns:
            A CommandResult describing the outcome. Failure to be
            acknowledged is a RESULT, not an exception -- the caller needs
            the attempt count and reason to tell the operator.

        Raises:
            CommandError: If `command` is not a configured command name.
                Raised rather than returned because it is a programming or
                configuration fault, not a link condition, and no retry could
                ever help.
        """
        # The dashboard validates its public ``operator_*`` JSON command and
        # DashboardCoordinator maps it to ``pause`` or ``resume``. Validate
        # again at this boundary because direct tools/future callers can use
        # McuCommandSender without going through the dashboard router.
        if not isinstance(command, str):
            raise CommandError("command must be a string")
        try:
            code = self.config.codes[command]
        except KeyError:
            known = ", ".join(sorted(self.config.codes))
            raise CommandError(
                f"unknown command {command!r}; configured commands: {known}"
            ) from None

        async with self._lock:
            return await self._send_locked(command, code)

    async def _send_locked(self, command: str, code: int) -> CommandResult:
        """Run the send-and-retry loop. Caller must hold self._lock.

        Args:
            command: Command name, for logging and the result.
            code: The wire code point for that command.

        Returns:
            A CommandResult describing the outcome.
        """
        # Allocated ONCE, outside the retry loop. See the module docstring on
        # idempotency -- this single line is why retry is safe.
        self._sequence = (self._sequence + 1) & 0xFF
        sequence = self._sequence

        try:
            frame = build_command_frame(
                sequence,
                code,
                self.protocol.crc_config,
                message_type=self.protocol.message_types["command"],
                stx=self.protocol.stx,
                etx=self.protocol.etx,
            )
        except (ValueError, TypeError) as exc:
            # Cannot happen with validated config, but a framing failure must
            # never surface as a bare traceback in the WebSocket handler.
            LOGGER.exception("failed to build %s command frame", command)
            self.failed += 1
            return CommandResult(
                command=command,
                command_sequence=sequence,
                acknowledged=False,
                attempts=0,
                reason=f"could not build command frame: {exc}",
            )

        last_reason = "no matching ACK"
        for attempt in range(1, self.config.max_attempts + 1):
            # Discard anything left from a previous command before writing.
            # A late ACK from the PREVIOUS command would not match, but
            # examining it would waste part of this attempt's budget.
            self.acks.drain()

            try:
                await self._write_frame(frame)
            except (OSError, asyncio.TimeoutError) as exc:
                # The port is gone or wedged. Retrying immediately would fail
                # again, but the transport may be reconnecting underneath us,
                # so the remaining attempts are still worth spending.
                last_reason = f"write failed: {type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "%s attempt %d/%d: %s", command.upper(), attempt,
                    self.config.max_attempts, last_reason,
                )
                continue

            self.sent += 1
            LOGGER.info(
                "%s sent (seq=%d, attempt %d/%d): %s",
                command.upper(), sequence, attempt,
                self.config.max_attempts, frame.hex(" "),
            )

            ack = await self.acks.wait_for(
                sequence, code, self.config.ack_timeout_s
            )
            if ack is None:
                last_reason = (
                    f"no ACK within {self.config.ack_timeout_s * 1000:.0f} ms"
                )
                LOGGER.warning(
                    "%s attempt %d/%d timed out waiting for ACK (seq=%d)",
                    command.upper(), attempt, self.config.max_attempts,
                    sequence,
                )
                continue

            result_byte = ack.get("result")
            mcu_state = ack.get("state")

            if result_byte == self.config.ack_result_ok:
                self.acknowledged += 1
                LOGGER.info(
                    "%s ACKed (seq=%d) on attempt %d; MCU state=%s",
                    command.upper(), sequence, attempt,
                    ack.get("state_name", mcu_state),
                )
                return CommandResult(
                    command=command,
                    command_sequence=sequence,
                    acknowledged=True,
                    attempts=attempt,
                    mcu_state=mcu_state,
                )

            # A NACK is an ANSWER, not silence. The MCU received the command
            # and refused it -- because it is in FAULT, or the command is
            # invalid in its current state (HW LLD S3.5.3: FAULT->RUNNING
            # needs RESUME **and** the fault bits clear). Retrying an
            # identical frame produces an identical refusal, so stop here and
            # tell the operator why, rather than burning two more attempts and
            # reporting the vaguer "no ACK".
            self.failed += 1
            LOGGER.warning(
                "%s NACKed (seq=%d, result=%s, MCU state=%s) -- not retrying",
                command.upper(), sequence, result_byte,
                ack.get("state_name", mcu_state),
            )
            return CommandResult(
                command=command,
                command_sequence=sequence,
                acknowledged=False,
                attempts=attempt,
                reason=f"MCU refused the command (result={result_byte})",
                mcu_state=mcu_state,
                nacked=True,
            )

        self.failed += 1
        LOGGER.error(
            "%s FAILED after %d attempts (seq=%d): %s",
            command.upper(), self.config.max_attempts, sequence, last_reason,
        )
        return CommandResult(
            command=command,
            command_sequence=sequence,
            acknowledged=False,
            attempts=self.config.max_attempts,
            reason=f"{last_reason} after {self.config.max_attempts} attempts",
        )

    def snapshot(self) -> dict[str, object]:
        """Return counters, for logging and diagnostics.

        Returns:
            A plain dict merging sender and correlator counters.
        """
        return {
            "commands_sent": self.sent,
            "commands_acknowledged": self.acknowledged,
            "commands_failed": self.failed,
            "next_sequence": (self._sequence + 1) & 0xFF,
            **self.acks.snapshot(),
        }
