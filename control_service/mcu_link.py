"""THE SINGLE OWNER of the MCU port. Creates every helper object.

LLD SW S3.8 step 3 calls this the "AGX Orin control service"; S3.8a calls it
"Control Service, block 7 in the diagram".

WHERE THIS SITS -- the whole picture
====================================

                          config.yaml
                               |
                    config_loader.ConfigLoader
                               |
    +--------------------------+---------------------------+
    |                                                      |
    v                                                      v
  build_transport()                                   McuLink.__init__
    |                                        creates ALL helpers, once:
    |  transport.kind ?                                    |
    |    mock   -> simulator.MockMcu          UnifiedStreamParser <- protocol
    |    serial -> transport.SerialTransport  McuState            <- mcu_state
    |                    |                    HeartbeatReceiver   <- heartbeat_
    +--------------------+                    HeartbeatLinkMonitor   receiver
                         |                    TelemetryReceiver   <- telemetry_
                         v                    AckReceiver            receiver
                  McuLink.run()               McuCommandSender    <- mcu_command
                         |
       +-----------------+-----------------+
       |                                   |
  READ LOOP                          LINK-CHECK LOOP
  read() -> received_at              every 50 ms, ALWAYS
  parser.feed()                      HeartbeatLinkMonitor.poll()
  _dispatch()                        (runs even in total silence --
       |                              that is how a dead cable is caught)
       |
  +----+--------+----------------+
  |             |                |
heartbeat   telemetry        ack_nack
  |             |                |
  v             v                v
Heartbeat   Telemetry        AckReceiver
Receiver    Receiver          .offer()
LinkMonitor     |                |
McuState        |                v
  |             |          wakes the coroutine
  +------+------+           parked in send()
         |
         v
   self.publish(event) -> dashboard_ws.broadcast() -> panels

WHY EVERYTHING IS CREATED HERE
==============================

Flows 1, 3, 4, 5 and 6 of the service all read or write the same RS-422 port.
They CANNOT each open it independently -- see transport.py for what happens
when two readers share a tty. So one class owns the port, and every object
that needs to see the traffic is constructed here and fed from the one loop.

The helpers are deliberately SYNCHRONOUS and I/O-free. Each observe/update
method mutates its own state and RETURNS event dicts; McuLink does all the
publishing. Two payoffs:

  1. Every outbound event passes through one place, so "what does the panel
     get told, and when?" has a single answer to read.
  2. The helpers test with plain dicts and a fake clock, no event loop --
     which is why self_test.py runs in seconds instead of waiting on timers.

THE RACE THIS DESIGN EXISTS TO PREVENT
======================================

Two levels, same rule.

BETWEEN PROCESSES: a tty has one kernel receive buffer, and each byte goes to
exactly ONE reader. Two processes reading /dev/ttyUSB0 get disjoint halves of
every frame; neither can decode anything, and the parser's resync can
manufacture a CRC-valid frame from fragments about 1 time in 256. On Linux
both processes open the port happily and this happens SILENTLY. On Windows
the second open fails outright -- so the bug is INVISIBLE on a Windows dev
box and appears only on the Jetson.

WITHIN THIS PROCESS: the ACK to a command arrives interleaved with heartbeats
and telemetry. Code that reads the port hunting for an ACK swallows all of
it. One 200 ms attempt eats two beats; three attempts eat six, and the panel
reports LINK_DOWN every time an operator presses pause. So the ACK wait is a
QUEUE SUBSCRIPTION fed by this loop -- see mcu_command.py.

DATA IN   : bytes from the transport; commands from dashboard_ws.py
DATA OUT  : event dicts, handed to the injected `publish` callable
CREATED BY: main.py (monitor / send) or AppOrchestrator.__init__() (serve)
"""

from __future__ import annotations

from simulator.mock_mcu import MockMcu


import asyncio
import contextlib
import logging
import time
from typing import Any, Awaitable, Callable, Mapping, MutableMapping

import constants
from config.config_loader import AttrDict
from control_service.heartbeat_receiver import (
    HeartbeatLinkMonitor,
    HeartbeatReceiver,
)
from control_service.mcu_command import (
    AckReceiver,
    CommandError,
    CommandResult,
    McuCommandSender,
    NO_SEQUENCE,
)
from control_service.mcu_state import McuState
from control_service.protocol import UnifiedStreamParser, parser_counters
from control_service.telemetry_receiver import TelemetryReceiver
from control_service.transport import SerialTransport, Transport, TransportError
from dashboard_service.services import event_factory as events

LOGGER = logging.getLogger(__name__)

#: Signature of the fan-out callable McuLink publishes events through.
Publish = Callable[[Mapping[str, Any]], Awaitable[None]]

#: Signature of the optional per-trigger hook the inference pipeline uses.
TriggerHook = Callable[[Mapping[str, Any]], Awaitable[None]]


def build_transport(config: AttrDict) -> Transport:
    """Construct the transport named by `transport.kind` in config.yaml.

    The ONE place that decides mock versus real. Everything downstream is
    written against the Transport ABC and cannot tell the difference, which
    is what lets the same parsing and retry code be exercised without
    hardware.

    Args:
        config: The whole validated configuration.

    Returns:
        A MockMcu when transport.kind is "mock", otherwise a SerialTransport.

    Raises:
        ValueError: If the kind is unknown. config_loader validates this at
            startup, so reaching here means the two got out of step.
    """
    if config.transport.kind == constants.TRANSPORT_MOCK:
        # Configuration selects the transport; both classes are imported above.

        LOGGER.info("transport: MockMcu (simulated, no hardware)")
        return MockMcu(
            simulator_config=config.simulator,
            heartbeat_config=config.heartbeat,
            protocol_config=config.protocol,
            command_codes=config.commands.codes,
            states=config.mcu_states,
            encoder_config=config.encoder
        )

    if config.transport.kind == constants.TRANSPORT_SERIAL:
        LOGGER.info(
            "transport: SerialTransport on %s at %d baud",
            config.serial.port, config.serial.baud,
        )
        return SerialTransport(config.serial)

    raise ValueError(
        f"unknown transport.kind {config.transport.kind!r}; "
        f"config_loader should have rejected this at startup"
    )


async def _publish_nowhere(message: Mapping[str, Any]) -> None:
    """Default publisher: log the event and drop it.

    Used when McuLink runs without a dashboard attached -- `main.py monitor`
    replaces it, and self_test.py uses it. Having a default means `publish`
    is never None and no call site has to guard against it.

    Args:
        message: The event that would have been broadcast.
    """
    LOGGER.debug("event (no subscriber): %s", message)


class McuLink:
    """Owns the MCU port: reads frames, tracks state, sends commands.

    Attributes:
        config: The whole validated configuration.
        transport: The link, real or simulated.
        publish: Injected fan-out for outbound events.
        parser: The single stream parser fed by the single read loop.
        state: Observed MCU state, flags and uptime.
        heartbeat: Dropped-beat counting on the MCU clock.
        link: Link health on the host arrival clock.
        telemetry: Trigger continuity.
        acks: The ACK correlator.
        commands: Outbound command sender.
    """

    def __init__(
        self,
        config: AttrDict,
        transport: Transport,
        publish: Publish | None = None,
        on_trigger: TriggerHook | None = None,
    ) -> None:
        """Construct the link and every helper it drives.

        `publish` is INJECTED rather than imported so this module never
        depends on the WebSocket layer. Dependencies flow one way --
        dashboard imports control_service, never the reverse -- which keeps
        this runnable headless and free of an import cycle.

        Args:
            config: The whole validated configuration.
            transport: A MockMcu or SerialTransport, from build_transport().
            publish: Awaitable called with each outbound event. Defaults to
                logging them and dropping them.
            on_trigger: Optional awaitable called with each decoded telemetry
                message. This is flow 2's entry point -- the real inference
                pipeline hangs off it, driven by the MCU rather than a timer.
        """
        self.config = config
        self.transport = transport
        self.publish = publish or _publish_nowhere
        self.on_trigger = on_trigger

        # ---- every helper object in the service, created here, once ----
        self.parser = UnifiedStreamParser(
            crc_config=config.protocol.crc_config,
            state_names=config.mcu_states.names,
            max_payload=config.protocol.max_payload,
            payload_lengths=config.protocol.payload_lengths_by_code,
            stx=config.protocol.stx,
            etx=config.protocol.etx,
            message_types=config.protocol.message_types,
            command_names=config.commands.names,
            status_flag_names=config.protocol.status_flag_names,
        )
        self.state = McuState(config.mcu_states)
        self.heartbeat = HeartbeatReceiver(config.heartbeat)
        self.link = HeartbeatLinkMonitor(config.heartbeat)
        self.telemetry = TelemetryReceiver()
        self.acks = AckReceiver()
        self.commands = McuCommandSender(
            config=config.commands,
            protocol=config.protocol,
            write_frame=self.transport.write,
            acks=self.acks,
        )

        self._running = False
        self._last_counter_report_at = time.perf_counter()
        self.ignored_messages = 0

    # ---- observed state, read by the dashboard mirror --------------------

    @property
    def capture_paused(self) -> bool:
        """Whether the MCU is currently NOT inspecting fabric.

        Returns:
            True in any state other than RUNNING. See mcu_state.py for why
            this is observed from the heartbeat and never inferred from an
            ACK.
        """
        return self.state.capture_paused

    def link_state(self, now: float | None = None) -> tuple[str, float]:
        """Report the instantaneous link state and heartbeat age.

        Args:
            now: perf_counter() reading. Defaults to now. Callers that also
                compute other ages MUST pass the same reading to all of them
                -- see heartbeat_receiver.py on clock epochs.

        Returns:
            (state name, age in seconds).
        """
        return self.link.link_state(now)

    def missed_intervals(self, now: float | None = None) -> int:
        """Report intervals of silence, disambiguating the DEGRADED bands.

        Args:
            now: perf_counter() reading. Defaults to now.

        Returns:
            Whole intervals since the last beat, or 0 while UP.
        """
        return self.link.missed_intervals(now)

    # ---- the two loops ---------------------------------------------------

    async def run(self) -> None:
        """Open the link and run the read loop and link-check loop together.

        TWO loops, not one, and they are independent on purpose:

          read loop  -- blocks on transport.read(), dispatches frames
          link check -- fires every heartbeat.check_interval_s regardless

        The link check MUST NOT be driven by frame arrival. A totally dead
        link produces no frames, and that is precisely the failure it exists
        to detect; tying it to inbound data would mean the panel sat on
        LINK_UP forever with the cable unplugged.

        Raises:
            TransportError: If the port cannot be opened, or is lost and
                reconnection is disabled in config.yaml.
        """
        self._running = True
        try:
            await self.transport.open()
        except TransportError:
            LOGGER.exception("could not open the MCU link")
            raise

        LOGGER.info("MCU link running")
        watcher = asyncio.create_task(self._watch_link(), name="link-watch")
        try:
            await self._read_loop()
        except asyncio.CancelledError:
            LOGGER.info("MCU link cancelled, shutting down")
            raise
        finally:
            self._running = False
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            await self._close_quietly()
            LOGGER.info("MCU link stopped. %s", self.counters_line())

    async def _read_loop(self) -> None:
        """Read the port forever, dispatching every frame that parses.

        THE ONLY place in the service that reads the transport. Everything
        else subscribes to what this loop dispatches.
        """
        while self._running:
            try:
                chunk = await self.transport.read()
            except TransportError as exc:
                LOGGER.error("link lost: %s", exc)
                await self._handle_link_loss()
                continue

            # ONE clock reading, taken at the boundary, immediately after the
            # bytes reach us. Passed down to every consumer rather than each
            # sampling its own.
            #
            # WHY: a single read() can return SEVERAL frames when the USB
            # latency timer batches them -- they arrived at the host at the
            # same instant, in one transfer. Sampling the clock separately per
            # frame would stamp them at different times, which is simply
            # false, and would let parse time accumulate into the measurement.
            received_at = time.perf_counter()

            if chunk:
                await self._consume(chunk, received_at)

            self._maybe_report_counters()

    async def _watch_link(self) -> None:
        """Evaluate link health on a fixed timer, forever.

        Deliberately independent of traffic. See run() and
        heartbeat_receiver.HeartbeatLinkMonitor.poll() for why.
        """
        interval = self.config.heartbeat.check_interval_s
        while True:
            try:
                await asyncio.sleep(interval)
                event = self.link.poll()
                if event is not None:
                    await self._publish(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The link watcher must be the last thing to die. If it
                # crashed, the panel would freeze on its last link state and
                # look healthy forever.
                LOGGER.exception("link watcher failed; continuing")

    async def _consume(self, chunk: bytes, received_at: float) -> None:
        """Feed one chunk to the parser and dispatch what comes out.

        Args:
            chunk: Raw bytes from the transport.
            received_at: perf_counter() sampled at the read boundary.
        """
        try:
            messages = self.parser.feed(chunk)
        except Exception:
            # The parser is defensive and should not raise, but a crash here
            # would kill the read loop and take the whole link down with it.
            # Log, drop the chunk, keep reading -- the parser resynchronises
            # on the next STX.
            LOGGER.exception(
                "parser raised on %d bytes; dropping chunk: %s",
                len(chunk), chunk.hex(" "),
            )
            return

        for message in messages:
            try:
                await self._dispatch(message, received_at)
            except asyncio.CancelledError:
                raise
            except Exception:
                # One malformed message must not stop the others being
                # handled, and must not stop the loop. The next heartbeat is
                # 100 ms away and will re-establish state.
                LOGGER.exception(
                    "error dispatching %s message; continuing",
                    message.get("message_type"),
                )

    async def _dispatch(
        self, message: MutableMapping[str, Any], received_at: float
    ) -> None:
        """Route one decoded message to the helper that owns it.

        Args:
            message: A decoded message dict from the parser.
            received_at: perf_counter() sampled at the read boundary.
        """
        kind = message.get("message_type")

        if kind == constants.MSG_NAME_HEARTBEAT:
            await self._on_heartbeat(message, received_at)
        elif kind == constants.MSG_NAME_TELEMETRY:
            await self._on_telemetry(message, received_at)
        elif kind == constants.MSG_NAME_ACK_NACK:
            # Handed to the waiting command coroutine. McuCommandSender never
            # reads the port itself -- that is the whole point.
            # Field names are the DECODER's -- command_sequence /
            # command_code, not the wire's cmd_seq / cmd_code. Getting these
            # wrong prints None and hides the very mismatch this line exists
            # to show.
            LOGGER.debug(
                "ACK        seq=%s  cmd=%s (%s)  result=%s  state=%s  raw=%s",
                message.get("command_sequence"), message.get("command_code"),
                message.get("command_name"), message.get("result"),
                message.get("state_name"), message.get("raw_hex"),
            )
            self.acks.offer(message)
            # The ACK payload carries `state` too (HW LLD S3.5.2), so a
            # command taking effect is observed up to 100 ms sooner than the
            # next heartbeat would have reported it.
            for event in self.state.observe_state(message):
                await self._publish(event)
        else:
            # COMMAND frames are host->MCU; seeing one echoed back means the
            # RS-422 pair is looped or the adapter is in local-echo mode.
            self.ignored_messages += 1
            LOGGER.debug("ignoring %s message", kind)

    async def _on_heartbeat(
        self, message: Mapping[str, Any], received_at: float
    ) -> None:
        """Handle one heartbeat: arrival clock, MCU clock, then state.

        Args:
            message: A decoded heartbeat dict.
            received_at: perf_counter() sampled at the read boundary.
        """
        # One line per beat, for hardware bringup. DEBUG so it costs nothing
        # at the default INFO level -- set logging.level: "DEBUG" to see it.
        _, age = self.link.link_state(received_at)
        LOGGER.debug(
            "HEARTBEAT  gap %6.1f ms  mcu_us=%d  state=%s  flags=0x%02X  "
            "uptime=%ss",
            age * 1000, message.get("timestamp_us", 0),
            self.state.state_name, message.get("status_flags", 0),
            message.get("uptime_s", 0),
        )

        # (A) arrival clock -- link freshness
        self.link.record(received_at)

        # (B) MCU clock -- genuine losses. Independent of (A) by design.
        for event in self.heartbeat.update(message, received_at):
            await self._publish(event)

        # (C) observed MCU state. All folded in before anything is published,
        # so a single beat cannot emit a half-updated view.
        for event in self.state.observe_uptime(message):
            await self._publish(event)
        for event in self.state.observe_flags(message):
            await self._publish(event)
        for event in self.state.observe_state(message):
            await self._publish(event)

    def _log_telemetry(
        self,
        message: Mapping[str, Any],
        received_at: float,
        previous: tuple[int | None, int | None, int | None, float | None],
    ) -> None:
        """Log one telemetry frame with everything derived from it.

        Skipped entirely unless DEBUG is on -- the deltas and formatting are
        not free at the trigger rate, and this runs on the read path.

        Args:
            message: The decoded telemetry dict, AFTER TelemetryReceiver has
                added its "sequence" verdict.
            received_at: perf_counter() at the read boundary.
            previous: (trigger_id, encoder_count, timestamp_us, arrival) from
                before this frame was folded in. Any may be None on the first.
        """
        LOGGER.info(
            "TELEMETRY: trigger_id=%s | encoder_count=%s | "
            "distance_mm_x1000=%s | timestamp_us=%s | direction=%s | "
            "state=%s | status_flags=%s",
            message.get("trigger_id"),
            message.get("encoder_count"),
            message.get("distance_mm_x1000"),
            message.get("timestamp_us"),
            message.get("direction"),
            message.get("state"),
            message.get("status_flags"),
        )
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return

        prev_id, prev_enc, prev_us, prev_at = previous
        trigger_id = message.get("trigger_id")
        encoder = message.get("encoder_count")
        stamp = message.get("timestamp_us")

        def delta(now: object, before: object) -> str:
            """Render a +N delta, or blank when there is nothing to compare."""
            if now is None or before is None:
                return ""
            return f" ({int(now) - int(before):+d})"

        # Encoder counts -> millimetres of fabric, at the documented scale.
        # HW LLD S3.3: x4 quadrature, 2000 counts/rev, 200mm wheel.
        travelled = ""
        if encoder is not None and prev_enc is not None:
            mm = (int(encoder) - int(prev_enc)) * self.config.encoder.mm_per_count
            travelled = f" = {mm:.1f}mm"

        mcu_gap = ""
        if stamp is not None and prev_us is not None:
            mcu_gap = f" (+{(int(stamp) - int(prev_us)) / 1000:.1f}ms)"

        host_gap = ""
        if prev_at is not None:
            host_gap = f"  host_gap={(received_at - prev_at) * 1000:.1f}ms"

        LOGGER.debug(
            "TELEMETRY  trig=%s%s  enc=%s%s%s  mcu_us=%s%s%s  state=%s  [%s]",
            trigger_id, delta(trigger_id, prev_id),
            encoder, delta(encoder, prev_enc), travelled,
            stamp, mcu_gap, host_gap,
            message.get("state_name") or self.state.state_name,
            message.get("sequence", "?"),
        )

    async def _on_telemetry(
        self, message: MutableMapping[str, Any], received_at: float
    ) -> None:
        """Handle one telemetry frame: continuity, state, then the pipeline.

        DELIBERATE DIFFERENCE FROM THE EARLIER PROTOTYPE: telemetry does NOT
        refresh the heartbeat clock. Treating any inbound frame as proof of
        life conflates two different questions -- "is the link alive?" and
        "is the MCU's heartbeat timer alive?". If the firmware's beat timer
        died while triggers kept flowing, the earlier code would have shown
        LINK_UP throughout and nobody would ever have known. link_state is
        driven by heartbeats only.

        Args:
            message: A decoded telemetry dict. Mutated by TelemetryReceiver to
                add a "sequence" verdict before the pipeline hook sees it.
            received_at: perf_counter() sampled at the read boundary.
        """
        # Snapshot the previous trigger BEFORE update() overwrites it, so the
        # debug line can show what changed rather than only the raw values.
        previous = (
            self.telemetry.last_trigger_id,
            self.telemetry.last_encoder_count,
            self.telemetry.last_timestamp_us,
            self.telemetry.last_trigger_at,
        )

        for event in self.telemetry.update(message, received_at):
            await self._publish(event)

        # Logged AFTER update() so message["sequence"] carries the verdict --
        # ok / first / duplicate / gap.
        self._log_telemetry(message, received_at, previous)

        # Telemetry payloads carry state and flags too (HW LLD S3.5.2 added
        # `state` "for consistency with HEARTBEAT").
        for event in self.state.observe_flags(message):
            await self._publish(event)
        for event in self.state.observe_state(message):
            await self._publish(event)

        if self.on_trigger is None:
            return
        try:
            await self.on_trigger(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            # The inference pipeline failing must not stop the link. Losing
            # detections is bad; losing the heartbeat monitor as well would
            # also hide the fact that anything was wrong.
            LOGGER.exception(
                "on_trigger hook failed for trigger_id=%s; link continues",
                message.get("trigger_id"),
            )

    async def _publish(self, message: Mapping[str, Any]) -> None:
        """Hand one event to the injected publisher, never raising.

        A dashboard fan-out failure -- a panel that died mid-broadcast -- must
        not propagate into the read loop and stop the link.

        Args:
            message: The event to publish.
        """
        try:
            await self.publish(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception(
                "publish failed for %s event; link continues",
                message.get("event"),
            )

    async def _handle_link_loss(self) -> None:
        """Attempt to recover a lost port, or stop the loop if we cannot.

        Raises:
            TransportError: If the transport cannot or will not reconnect.
        """
        try:
            await self.transport.reopen()
        except asyncio.CancelledError:
            raise
        except TransportError:
            LOGGER.exception("reconnection failed; stopping the link")
            self._running = False
            raise

    async def _close_quietly(self) -> None:
        """Close the transport, swallowing errors.

        Called from `finally`, where raising would mask the original cause of
        the shutdown.
        """
        with contextlib.suppress(Exception):
            await self.transport.close()

    # ---- outbound --------------------------------------------------------

    async def send_command(
        self, command: str, source: str = constants.SOURCE_OPERATOR
    ) -> CommandResult:
        """Send a command to the MCU and report the outcome.

        Does NOT set capture_paused. A successful ACK means only that the
        command was RECEIVED; the state change is confirmed when the next
        heartbeat reports it. See mcu_state.py for why that distinction is a
        safety property and not a nicety.

        Args:
            command: A configured command name, e.g. "pause".
            source: Who asked, for the audit log.

        Returns:
            A CommandResult. A command_failed event is published on failure so
            the panel learns about it even if the caller ignores the return
            value.
        """
        LOGGER.info("command requested: %s (source=%s)", command, source)
        try:
            result = await self.commands.send(command)
        except CommandError as exc:
            LOGGER.error("cannot send %r: %s", command, exc)
            await self._publish(
                events.command_failed(command, NO_SEQUENCE, str(exc))
            )
            return CommandResult(
                command=command,
                command_sequence=NO_SEQUENCE,
                acknowledged=False,
                attempts=0,
                reason=str(exc),
            )

        if not result.acknowledged:
            await self._publish(
                events.command_failed(
                    result.command, result.command_sequence, result.reason
                )
            )
        return result

    # ---- diagnostics -----------------------------------------------------

    def _maybe_report_counters(self) -> None:
        """Log parser health on the configured cadence.

        WHAT TO LOOK FOR: on a healthy link with a SINGLE reader, every
        counter stays at zero. A rising discarded_bytes is the signature of a
        second process reading the same port -- the exact failure this whole
        design exists to prevent, made visible.
        """
        now = time.perf_counter()
        interval = self.config.logging.counter_report_interval_s
        if now - self._last_counter_report_at < interval:
            return
        self._last_counter_report_at = now

        counters = parser_counters(self.parser)
        unhealthy = any(
            value > 0
            for value in (
                self.parser.discarded_bytes, self.parser.crc_errors,
                self.parser.etx_errors, self.parser.length_errors,
                self.parser.decode_errors,
            )
        )
        if unhealthy:
            LOGGER.warning(
                "parser errors present -- is another process reading %s? %s",
                self.config.serial.port, counters,
            )
        else:
            LOGGER.debug("parser healthy: %s", counters)

    def counters_line(self) -> str:
        """Build a one-line summary of every counter in the link.

        Returns:
            A human-readable summary for the shutdown log.
        """
        parts: dict[str, object] = {
            **self.heartbeat.snapshot(),
            **self.telemetry.snapshot(),
            **self.commands.snapshot(),
            "ignored_messages": self.ignored_messages,
        }
        rendered = ", ".join(f"{key}={value}" for key, value in parts.items())
        return f"{rendered}; {parser_counters(self.parser)}"

    def snapshot(self) -> dict[str, object]:
        """Return the full observed state of the link.

        Returns:
            A plain dict merging every helper's snapshot. For logging, the
            /health endpoint and bench diagnostics -- not a wire payload.
        """
        return {
            **self.link.snapshot(),
            **self.state.snapshot(),
            **self.heartbeat.snapshot(),
            **self.telemetry.snapshot(),
            **self.commands.snapshot(),
            "ignored_messages": self.ignored_messages,
            "parser": {
                "discarded_bytes": self.parser.discarded_bytes,
                "crc_errors": self.parser.crc_errors,
                "etx_errors": self.parser.etx_errors,
                "length_errors": self.parser.length_errors,
                "decode_errors": self.parser.decode_errors,
            },
            "link": self.transport.snapshot(),
        }
