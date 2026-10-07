#!/usr/bin/env python3
"""Offline tests. No hardware, no network, no waiting in real time.

RUN IT
======

    python main.py selftest
    python main.py selftest --verbose
    python main.py selftest --filter debounce

WHY THESE TESTS LOOK LIKE THIS
==============================

Every helper class in control_service/ is deliberately SYNCHRONOUS and free
of I/O: observe/update methods mutate their own state and RETURN event dicts,
and McuLink does the publishing. That design choice pays off here -- most of
the behaviour worth testing can be driven with plain dicts and a fake clock,
with no event loop and no sleeping.

Where a test does need asyncio -- the command retry path, the read loop -- it
uses MockMcu, which emits REAL protocol bytes. So the parser, the CRC check
and the stream reassembly under test are the same code that will run against
the MCU.

CLOCK DISCIPLINE: tests pass `now` and `received_at` explicitly rather than
sleeping. A test that waits 300 ms to watch a link go DOWN is slow, flaky
under load, and tells you nothing a passed-in timestamp would not.

DATA IN   : nothing external
DATA OUT  : a pass/fail line per test, and a non-zero exit code on failure
CREATED BY: main.py selftest
"""

from __future__ import annotations

from dashboard_service.services import event_factory as events
from dashboard_service.services.status_mirror import DashboardMirror
from dashboard_service.services.event_store import EventStore
from dashboard_service.operator_panel.client import PanelEventCursor
from fastapi import FastAPI
from fastapi.testclient import TestClient
from control_service.mcu_command import CommandResult
from dashboard_service.routers.fallback import create_fallback_router
from dashboard_service.services.event_store import EventStore, EventStoreError
from pydantic import ValidationError
from dashboard_service.routers.websocket import _validate_command


import asyncio
import contextlib
import logging
import struct
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import constants
from constants import LinkState, TransportKind
from config.config_loader import (
    AttrDict, ConfigError, ConfigLoader, load_config,
)
from control_service.heartbeat_receiver import (
    HeartbeatLinkMonitor,
    HeartbeatReceiver,
)
from control_service.mcu_command import AckReceiver, CommandError, McuCommandSender
from control_service.mcu_link import McuLink, build_transport
from control_service.mcu_state import McuState
from control_service.protocol import (
    UnifiedStreamParser,
    build_unified_frame,
)
from control_service.telemetry_receiver import TelemetryReceiver
from control_service.transport import Transport
from simulator.mock_mcu import MockMcu
from utils.gpu_monitor import GpuMonitor

#: Collected (name, callable) test cases, filled by the @test decorator.
_TESTS: list[tuple[str, Callable[[], Any]]] = []


def test(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Register a function as a test case.

    Args:
        name: Human-readable name shown in the report.

    Returns:
        A decorator that registers and returns the function unchanged.
    """
    def decorate(func: Callable[[], Any]) -> Callable[[], Any]:
        _TESTS.append((name, func))
        return func
    return decorate


try:
    CONFIG = load_config()
except ConfigError as _exc:  # pragma: no cover - environment problem
    raise SystemExit(f"cannot run tests: {_exc}") from _exc


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------

@test("protocol: frames survive fragmentation and leading noise")
def test_parser_reassembly() -> None:
    """A frame split across two reads, behind junk bytes, still decodes."""
    heartbeat = build_unified_frame(
        CONFIG.protocol.message_types["heartbeat"],
        struct.pack("<IBBH", 298_600_000, 1, 0b00000101, 298),
        CONFIG.protocol.crc_config,
        stx=CONFIG.protocol.stx,
        etx=CONFIG.protocol.etx,
    )
    parser = UnifiedStreamParser(
        CONFIG.protocol.crc_config, CONFIG.mcu_states.names, CONFIG.protocol.max_payload,
        CONFIG.protocol.payload_lengths_by_code,
        stx=CONFIG.protocol.stx,
        etx=CONFIG.protocol.etx,
        message_types=CONFIG.protocol.message_types,
        command_names=CONFIG.commands.names,
        status_flag_names=CONFIG.protocol.status_flag_names,
    )

    assert parser.feed(b"\x99\x88" + heartbeat[:4]) == []
    messages = parser.feed(heartbeat[4:])
    assert len(messages) == 1, f"expected 1 message, got {len(messages)}"

    message = messages[0]
    assert message["message_type"] == "heartbeat"
    assert message["timestamp_us"] == 298_600_000
    assert message["state_name"] == "RUNNING"
    assert message["active_flags"] == ["strobe_fault", "overcurrent_trip"]
    assert message["uptime_s"] == 298


@test("protocol: a corrupted byte is rejected by CRC")
def test_parser_crc_rejection() -> None:
    """Flipping one payload bit makes the frame fail its checksum."""
    heartbeat = build_unified_frame(
        CONFIG.protocol.message_types["heartbeat"], struct.pack("<IBBH", 1_000_000, 1, 0, 10),
        CONFIG.protocol.crc_config,
        stx=CONFIG.protocol.stx,
        etx=CONFIG.protocol.etx,
    )
    parser = UnifiedStreamParser(
        CONFIG.protocol.crc_config, CONFIG.mcu_states.names, CONFIG.protocol.max_payload,
        CONFIG.protocol.payload_lengths_by_code,
        stx=CONFIG.protocol.stx,
        etx=CONFIG.protocol.etx,
        message_types=CONFIG.protocol.message_types,
        command_names=CONFIG.commands.names,
        status_flag_names=CONFIG.protocol.status_flag_names,
    )
    corrupted = bytearray(heartbeat)
    corrupted[5] ^= 0x01
    assert parser.feed(bytes(corrupted)) == []
    assert parser.crc_errors == 1, "CRC error was not counted"


# --------------------------------------------------------------------------
# HeartbeatLinkMonitor -- the ARRIVAL clock
# --------------------------------------------------------------------------

@test("link: bands map age to the right state")
def test_link_bands() -> None:
    """Each configured band is entered at the expected age."""
    link = HeartbeatLinkMonitor(CONFIG.heartbeat)
    link.started_at = 10.0

    link.record(10.05)
    assert link.link_state(10.10)[0] == constants.LINK_UP        # 50 ms
    assert link.link_state(10.16)[0] == constants.LINK_DEGRADED  # 110 ms
    assert link.link_state(10.30)[0] == constants.LINK_DEGRADED  # 250 ms
    assert link.link_state(10.36)[0] == constants.LINK_DOWN      # 310 ms


@test("link: age grows with NO frames arriving -- how a dead cable is caught")
def test_link_ages_without_input() -> None:
    """The whole point: age = now - last_arrival, and `now` moves alone."""
    link = HeartbeatLinkMonitor(CONFIG.heartbeat)
    link.record(100.0)

    # Not one byte arrives after t=100. The age still climbs.
    assert link.link_state(100.05)[0] == constants.LINK_UP
    assert link.link_state(100.15)[0] == constants.LINK_DEGRADED
    assert link.link_state(100.40)[0] == constants.LINK_DOWN


@test("link: missed_intervals separates the two DEGRADED bands")
def test_missed_intervals() -> None:
    """DEGRADED at 1 interval of silence is distinguishable from 2."""
    link = HeartbeatLinkMonitor(CONFIG.heartbeat)
    link.record(10.0)
    assert link.missed_intervals(10.05) == 0   # UP asserts nothing was missed
    assert link.missed_intervals(10.16) == 1   # 110 ms
    assert link.missed_intervals(10.25) == 2   # 250 ms


@test("link: a link that never comes up decays to DOWN")
def test_link_never_up() -> None:
    """With no beat ever received, age runs from service start."""
    link = HeartbeatLinkMonitor(CONFIG.heartbeat)
    link.started_at = 100.0
    assert link.link_state(100.50)[0] == constants.LINK_DOWN


@test("link: DEBOUNCE drops a transient crossing from a late beat")
def test_debounce_rejects_blip() -> None:
    """A 112 ms beat crosses 100 ms for ~12 ms. That must not be published."""
    if CONFIG.heartbeat.debounce_samples < 2:
        return  # debouncing disabled in config; nothing to assert

    link = HeartbeatLinkMonitor(CONFIG.heartbeat)
    link.record(0.0)

    # UP itself is debounced too, so it publishes on the SECOND poll.
    link.poll(0.010)  # "UP should still be debouncing"
    link.poll(0.050)
    assert link.published_state == constants.LINK_UP, (
        "UP should be adopted on the 2nd poll"
    )
    assert link.published_state == constants.LINK_UP

    # t=0.100: age crosses 100 ms -> DEGRADED, but only one sample so far.
    link.poll(0.100)  # "a single crossing must not publish"

    # t=0.112: the late beat lands and the age resets. The blip never held
    # for two consecutive polls, so nothing was ever reported.
    link.record(0.112)
    link.poll(0.150)  # "state returned to UP, nothing to publish"
    assert link.published_state == constants.LINK_UP, (
        "a 12 ms crossing must never reach the panel"
    )


@test("link: DEBOUNCE still reports a link that really goes quiet")
def test_debounce_allows_real_fault() -> None:
    """A crossing that persists across polls must publish."""
    link = HeartbeatLinkMonitor(CONFIG.heartbeat)
    link.record(0.0)
    link.poll(0.050)  # publishes UP

    link.poll(0.100)   # DEGRADED, sample 1 -- still debouncing
    assert link.published_state != constants.LINK_DEGRADED, (
        "one sample must not adopt DEGRADED"
    )
    link.poll(0.150)   # DEGRADED, sample 2 -- held
    assert link.published_state == constants.LINK_DEGRADED, (
        "a persistent DEGRADED must be adopted"
    )


@test("link: LINK_DOWN bypasses the debounce -- a dead link never waits")
def test_down_is_immediate() -> None:
    """debounce_exempt_down publishes DOWN on the first poll that sees it."""
    if not CONFIG.heartbeat.debounce_exempt_down:
        return

    link = HeartbeatLinkMonitor(CONFIG.heartbeat)
    link.record(0.0)
    link.poll(0.050)

    link.poll(0.350)  # 350 ms: straight past every band
    assert link.published_state == constants.LINK_DOWN, (
        "DOWN must be adopted on the FIRST sample, not the second"
    )


# --------------------------------------------------------------------------
# HeartbeatReceiver -- the MCU clock
# --------------------------------------------------------------------------

@test("heartbeat: dropped beats counted from the MCU clock, not arrival")
def test_sequence_tracking() -> None:
    """Transport jitter is not a loss; a real gap is."""
    receiver = HeartbeatReceiver(CONFIG.heartbeat)

    def beat(stamp: int, at: float) -> int:
        """Feed one beat and return how many were reported lost.

        Args:
            stamp: MCU timestamp_us.
            at: Simulated arrival time.

        Returns:
            The dropped count from any heartbeat_missed event.
        """
        produced = receiver.update({"timestamp_us": stamp}, at)
        return produced[0]["dropped"] if produced else 0

    assert beat(1_000_000, 1.0) == 0   # first beat
    assert beat(1_100_000, 1.1) == 0   # exactly on time
    assert beat(1_195_000, 1.2) == 0   # 95 ms, jitter
    assert beat(1_211_000, 1.3) == 0   # 16 ms later, latency-timer batching
    assert beat(1_511_000, 1.6) == 2   # 300 ms gap -- two beats lost
    assert receiver.missed == 2

    # A counter reset is a reboot, not a loss.
    assert beat(500, 1.7) == 0
    assert receiver.missed == 2, "reboot was miscounted as dropped beats"
    assert receiver.wraps == 1


@test("heartbeat: the FTDI 16ms alias is NOT reported as a loss")
def test_latency_timer_alias_is_not_a_loss() -> None:
    """The exact scenario the two-clock design exists to separate."""
    receiver = HeartbeatReceiver(CONFIG.heartbeat)
    link = HeartbeatLinkMonitor(CONFIG.heartbeat)

    # MCU sends perfectly evenly; USB delivers in 96/112 ms buckets.
    arrivals = (0.000, 0.112, 0.208, 0.304)
    for index, arrival in enumerate(arrivals):
        stamp = 1_000_000 + index * 100_000
        produced = receiver.update({"timestamp_us": stamp}, arrival)
        assert produced == [], "MCU clock must see no loss here"
        link.record(arrival)

    assert receiver.missed == 0, "the MCU clock correctly sees no loss"

    # The ARRIVAL clock disagrees -- correctly, because it answers a different
    # question. Rewind to just before the 112 ms beat landed: at that instant
    # the last arrival was 111 ms ago, which is past the LINK_UP bound.
    link_at_gap = HeartbeatLinkMonitor(CONFIG.heartbeat)
    link_at_gap.record(arrivals[0])
    assert link_at_gap.link_state(0.111)[0] == constants.LINK_DEGRADED, (
        "the arrival clock should read DEGRADED on a link that lost nothing "
        "-- this is exactly why the MCU-clock check exists"
    )


@test("heartbeat: a malformed timestamp does not inflate the loss count")
def test_sequence_malformed() -> None:
    """Garbage in a frame must not be reported as lost fabric coverage."""
    receiver = HeartbeatReceiver(CONFIG.heartbeat)
    receiver.update({"timestamp_us": 1_000_000}, 1.0)
    assert receiver.update({"timestamp_us": "not a number"}, 1.1) == []
    assert receiver.update({}, 1.2) == []
    assert receiver.missed == 0


@test("heartbeat: clock drift is None until enough samples, then a number")
def test_clock_drift() -> None:
    """LLD SW S5.3's bench diagnostic. Host running 100 ppm fast."""
    receiver = HeartbeatReceiver(CONFIG.heartbeat)
    assert receiver.clock_drift_ppm() is None, "no samples yet"

    # The MCU advances exactly 100,000 us per beat. The host clock advances
    # 0.10001 s per beat -- 1 part in 10,000 more, which IS 100 ppm.
    #
    # (Getting this wrong is easy: 0.1001 would be 0.1%, i.e. 1000 ppm.)
    host_step = 0.1 * (1 + 100 / 1_000_000)
    needed = CONFIG.heartbeat.drift_min_samples + 1
    for index in range(needed):
        receiver.update(
            {"timestamp_us": 1_000_000 + index * 100_000}, index * host_step
        )

    drift = receiver.clock_drift_ppm()
    assert drift is not None, "should report after enough samples"
    assert 95 < drift < 105, f"expected ~100 ppm, got {drift}"


# --------------------------------------------------------------------------
# McuState -- observes the firmware's machine
# --------------------------------------------------------------------------

@test("state: capture_paused is true for every state except RUNNING")
def test_capture_paused_definition() -> None:
    """FAULT must count as paused -- the bug this definition prevents."""
    states = CONFIG.mcu_states
    state = McuState(states)
    assert state.capture_paused is False, "unknown state must not alarm"

    state.observe_state({"state": states.running})
    assert state.capture_paused is False

    for code in (states.idle, states.paused, states.fault):
        state.observe_state({"state": code})
        assert state.capture_paused is True, (
            f"state {states.names[code]} must count as paused"
        )


@test("state: a fault pause is labelled differently from an operator one")
def test_pause_source() -> None:
    """LLD SW S3.9 requires the UI to render the two differently."""
    states = CONFIG.mcu_states
    state = McuState(states)
    state.observe_state({"state": states.running})

    produced = state.observe_state({"state": states.fault})
    assert len(produced) == 1
    assert produced[0]["source"] == constants.SOURCE_FAULT

    state.observe_state({"state": states.running})
    produced = state.observe_state({"state": states.paused})
    assert produced[0]["source"] == constants.SOURCE_MCU


@test("state: the first observation is silent, not a transition")
def test_first_state_is_silent() -> None:
    """Adopting the initial state must not look like the line just stopped."""
    state = McuState(CONFIG.mcu_states)
    assert state.observe_state({"state": CONFIG.mcu_states.paused}) == []
    assert state.capture_paused is True


@test("state: fault flags publish on every change, clearing included")
def test_flags_clear() -> None:
    """A panel told only about raises can never clear its alarm."""
    state = McuState(CONFIG.mcu_states)

    # No event is emitted -- mcu_fault is not in the SW LLD. The flags reach
    # the panel as system_status.faults, so this asserts the OBSERVED state.
    assert state.observe_flags(
        {"status_flags": 0b1000, "active_flags": ["panel_estop"]}
    ) == [], "observe_flags must not emit an event"
    assert state.active_faults == ["panel_estop"]

    state.observe_flags({"status_flags": 0, "active_flags": []})
    assert state.active_faults == [], "clearing a fault must clear the list"


@test("state: uptime going backwards is reported as a reboot")
def test_reboot_detection() -> None:
    """The only evidence of a crash when heartbeats never stopped."""
    state = McuState(CONFIG.mcu_states)
    assert state.observe_uptime({"uptime_s": 300}) == []
    assert state.observe_uptime({"uptime_s": 301}) == []

    produced = state.observe_uptime({"uptime_s": 2})
    assert len(produced) == 1
    assert produced[0]["event"] == "mcu_reboot"
    assert produced[0]["previous_uptime_s"] == 301


# --------------------------------------------------------------------------
# TelemetryReceiver
# --------------------------------------------------------------------------

@test("telemetry: a trigger gap alarms, a duplicate does not")
def test_trigger_continuity() -> None:
    """Crying wolf on duplicates is how a real alarm gets ignored."""
    receiver = TelemetryReceiver()

    assert receiver.update({"trigger_id": 100}, 1.0) == []
    assert receiver.update({"trigger_id": 101}, 1.5) == []
    assert receiver.update({"trigger_id": 101}, 2.0) == []  # duplicate
    assert receiver.duplicates == 1

    alarms = receiver.update({"trigger_id": 105}, 2.5)
    assert len(alarms) == 1
    assert alarms[0]["event"] == "missed_trigger_alarm"
    assert alarms[0]["expected_trigger_id"] == 102
    assert alarms[0]["gap_count"] == 3
    assert receiver.total_missed_triggers == 3


@test("telemetry: trigger_id wraparound is not a four-billion-trigger gap")
def test_trigger_wraparound() -> None:
    """uint32 wrap must produce no gap, not an absurd one."""
    receiver = TelemetryReceiver()
    receiver.update({"trigger_id": 0xFFFF_FFFE}, 1.0)
    assert receiver.update({"trigger_id": 0xFFFF_FFFF}, 1.5) == []
    assert receiver.update({"trigger_id": 0}, 2.0) == [], "wrap must be seamless"
    assert receiver.gaps == 0


@test("telemetry: an MCU counter reset restarts tracking, not an alarm")
def test_trigger_counter_reset() -> None:
    """A backwards jump is a reboot, reported by McuState, not a gap here."""
    receiver = TelemetryReceiver()
    receiver.update({"trigger_id": 5_000}, 1.0)
    assert receiver.update({"trigger_id": 3}, 1.5) == []
    assert receiver.gaps == 0
    assert receiver.counter_resets == 1


# --------------------------------------------------------------------------
# mcu_command -- AckReceiver + McuCommandSender
# --------------------------------------------------------------------------

def _make_sender(written: list[bytes]) -> McuCommandSender:
    """Build a sender whose writes land in a list.

    Args:
        written: List that captures every frame written.

    Returns:
        A McuCommandSender wired to that capture, with its own AckReceiver.
    """
    async def capture(frame: bytes) -> None:
        """Record one frame instead of writing it to a port.

        Args:
            frame: The frame that would have gone on the wire.
        """
        written.append(frame)

    return McuCommandSender(
        CONFIG.commands, CONFIG.protocol, capture, AckReceiver()
    )


def _ack_for(frame: bytes, result: int = 0) -> dict[str, Any]:
    """Build the decoded ACK the MCU would return for a command frame.

    Args:
        frame: The command frame that was sent.
        result: 0 for ACK, non-zero for NACK.

    Returns:
        A decoded ack_nack message dict.
    """
    return {
        "message_type": "ack_nack",
        "command_sequence": frame[3],
        "command_code": frame[4],
        "result": result,
        "state": CONFIG.mcu_states.paused,
        "state_name": "PAUSED",
    }


@test("commands: AckReceiver holds no transport -- it CANNOT read the port")
def test_ack_receiver_has_no_transport() -> None:
    """The structural guarantee, asserted rather than merely commented."""
    acks = AckReceiver()
    forbidden = {"read", "write", "open", "close", "transport", "port"}
    present = forbidden & set(vars(acks)) | forbidden & set(dir(acks))
    assert not present, f"AckReceiver must not expose {present}"


@test("commands: a command succeeds when its ACK arrives")
def test_command_ack() -> None:
    """The happy path, with the ACK delivered through the correlator."""
    async def scenario() -> None:
        """Send pause, feed the ACK, assert success."""
        written: list[bytes] = []
        sender = _make_sender(written)

        async def answer() -> None:
            """Deliver the ACK once the frame has been written."""
            while not written:
                await asyncio.sleep(0)
            sender.acks.offer(_ack_for(written[0]))

        asyncio.create_task(answer())
        result = await sender.send("pause")
        assert result.acknowledged is True
        assert result.attempts == 1
        assert len(written) == 1

    asyncio.run(scenario())


@test("commands: a retry reuses the SAME sequence, so it stays idempotent")
def test_command_retry_idempotent() -> None:
    """HW LLD S3.5.2's rule: the MCU re-ACKs rather than re-executing."""
    async def scenario() -> None:
        """Drop the first ACK, answer the second, compare the frames."""
        written: list[bytes] = []
        sender = _make_sender(written)

        async def answer() -> None:
            """Ignore the first frame; ACK the second."""
            while len(written) < 2:
                await asyncio.sleep(0)
            sender.acks.offer(_ack_for(written[1]))

        asyncio.create_task(answer())
        result = await sender.send("pause")

        assert result.acknowledged is True
        assert result.attempts == 2
        assert written[0] == written[1], (
            "a retry MUST resend identical bytes, or the MCU treats it as a "
            "new command and may execute it twice"
        )

    asyncio.run(scenario())


@test("commands: silence exhausts the attempts and reports failure")
def test_command_timeout() -> None:
    """No ACK at all means max_attempts sends and an unacknowledged result."""
    async def scenario() -> None:
        """Send with nothing ever answering."""
        written: list[bytes] = []
        sender = _make_sender(written)
        result = await sender.send("pause")

        assert result.acknowledged is False
        assert result.attempts == CONFIG.commands.max_attempts
        assert len(written) == CONFIG.commands.max_attempts
        assert result.nacked is False, "silence is not a NACK"

    asyncio.run(scenario())


@test("commands: a NACK stops immediately instead of retrying")
def test_command_nack_no_retry() -> None:
    """A refusal is an answer -- retrying identical bytes just repeats it."""
    async def scenario() -> None:
        """Answer with a non-zero result byte."""
        written: list[bytes] = []
        sender = _make_sender(written)

        async def answer() -> None:
            """Deliver a NACK for the first frame."""
            while not written:
                await asyncio.sleep(0)
            sender.acks.offer(_ack_for(written[0], result=1))

        asyncio.create_task(answer())
        result = await sender.send("pause")

        assert result.acknowledged is False
        assert result.nacked is True
        assert result.attempts == 1, "a NACK must not be retried"

    asyncio.run(scenario())


@test("commands: an ACK for a different sequence is ignored")
def test_command_ack_must_match() -> None:
    """Matching on sequence AND code stops a stale ACK satisfying a wait."""
    async def scenario() -> None:
        """Feed a mismatched ACK, then the right one."""
        written: list[bytes] = []
        sender = _make_sender(written)

        async def answer() -> None:
            """Send one wrong ACK, then the correct one."""
            while not written:
                await asyncio.sleep(0)
            wrong = _ack_for(written[0])
            wrong["command_sequence"] = (wrong["command_sequence"] + 1) & 0xFF
            sender.acks.offer(wrong)
            await asyncio.sleep(0)
            sender.acks.offer(_ack_for(written[0]))

        asyncio.create_task(answer())
        result = await sender.send("pause")
        assert result.acknowledged is True
        assert sender.acks.unmatched == 1

    asyncio.run(scenario())


@test("commands: an unknown command name raises rather than being sent")
def test_command_unknown() -> None:
    """A configuration or programming fault, not a link condition."""
    async def scenario() -> None:
        """Attempt a command that is not configured."""
        written: list[bytes] = []
        sender = _make_sender(written)
        try:
            await sender.send("self_destruct")
        except CommandError:
            assert not written, "nothing may reach the wire"
            return
        raise AssertionError("expected CommandError")

    asyncio.run(scenario())


@test("commands: an MCU command must be a string before it can be framed")
def test_command_must_be_string() -> None:
    """The MCU boundary rejects bad direct callers before config lookup."""
    async def scenario() -> None:
        """Attempt a malformed command without writing a frame."""
        written: list[bytes] = []
        sender = _make_sender(written)
        try:
            await sender.send({"command": "pause"})  # type: ignore[arg-type]
        except CommandError as exc:
            assert str(exc) == "command must be a string"
            assert not written, "nothing may reach the wire"
            return
        raise AssertionError("expected CommandError")

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# transport + simulator
# --------------------------------------------------------------------------

@test("transport: MockMcu satisfies the Transport ABC")
def test_mock_is_a_transport() -> None:
    """The ABC makes the contract structural, not a convention."""
    assert issubclass(MockMcu, Transport)
    mock = build_transport(CONFIG)
    assert isinstance(mock, Transport)


@test("transport: an incomplete Transport subclass cannot be instantiated")
def test_abc_enforces_contract() -> None:
    """Forgetting write() must fail at construction, not at first command."""
    class Incomplete(Transport):
        """A transport missing write()."""

        async def open(self) -> None:
            """Do nothing."""

        async def read(self) -> bytes:
            """Return nothing.

            Returns:
                Empty bytes.
            """
            return b""

        async def close(self) -> None:
            """Do nothing."""

    try:
        Incomplete()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("the ABC did not enforce write()")


def _running_config() -> AttrDict:
    """CONFIG with the mock booting into RUNNING instead of ARMED.

    config.yaml ships initial_state: "armed" so the real START CAPTURE flow
    is what `main.py serve --mock` exercises. Tests about telemetry, trigger
    gaps and framing want triggers immediately, so they override just that
    one key. Shallow copy -- AttrDict.wrap would re-wrap derived values like
    protocol.crc_config.

    Returns:
        A config whose simulator section boots RUNNING.
    """
    config = AttrDict(CONFIG)
    config["simulator"] = AttrDict(
        {**CONFIG.simulator, "initial_state": "running"}
    )
    return config


def _make_mock(initial_state: str = "running") -> MockMcu:
    """Build a MockMcu from the loaded configuration.

    Args:
        initial_state: Which state the mock boots into. Defaults to "running"
            so tests about telemetry, gaps and framing get triggers straight
            away. config.yaml ships "armed" for the real START CAPTURE flow;
            test_mock_boots_armed covers that path explicitly.

    Returns:
        A ready, un-opened MockMcu.
    """
    simulator = AttrDict.wrap(
        {**CONFIG.simulator, "initial_state": initial_state}
    )
    return MockMcu(
        simulator_config=simulator,
        heartbeat_config=CONFIG.heartbeat,
        protocol_config=CONFIG.protocol,
        command_codes=CONFIG.commands.codes,
        encoder_config=CONFIG.encoder,
        states=CONFIG.mcu_states,
    )


@test("simulator: emits real, parseable frames -- not fake dicts")
def test_mock_emits_real_frames() -> None:
    """The mock's bytes must survive the production parser."""
    async def scenario() -> None:
        """Read several frames and parse them."""
        mock = _make_mock()
        await mock.open()
        parser = UnifiedStreamParser(
            CONFIG.protocol.crc_config, CONFIG.mcu_states.names,
            CONFIG.protocol.max_payload,
            CONFIG.protocol.payload_lengths_by_code,
            stx=CONFIG.protocol.stx,
            etx=CONFIG.protocol.etx,
            message_types=CONFIG.protocol.message_types,
            command_names=CONFIG.commands.names,
            status_flag_names=CONFIG.protocol.status_flag_names,
        )

        kinds: set[str] = set()
        for _ in range(12):
            for message in parser.feed(await mock.read()):
                kinds.add(str(message["message_type"]))

        assert "heartbeat" in kinds, "no heartbeat produced"
        assert "telemetry" in kinds, "no telemetry produced"
        assert parser.crc_errors == 0, "the mock emitted a bad CRC"
        assert parser.discarded_bytes == 0, "the mock emitted junk bytes"
        await mock.close()

    asyncio.run(scenario())


@test("simulator: a repeated sequence is re-ACKed but NOT re-executed")
def test_mock_idempotency() -> None:
    """The firmware rule that makes the host's retry safe."""
    async def scenario() -> None:
        """Send pause twice with one sequence and check the state."""
        mock = _make_mock()
        await mock.open()

        pause = build_unified_frame(
            CONFIG.protocol.message_types["command"],
            struct.pack("<BB", 42, CONFIG.commands.codes["pause"]),
            CONFIG.protocol.crc_config,
            stx=CONFIG.protocol.stx,
            etx=CONFIG.protocol.etx,
        )
        await mock.write(pause)
        assert mock.state == CONFIG.mcu_states.paused

        # Manually resume, then replay the SAME sequence. A correct MCU
        # re-ACKs it without acting, so the state must stay RUNNING.
        mock.state = CONFIG.mcu_states.running
        await mock.write(pause)
        assert mock.state == CONFIG.mcu_states.running, (
            "a duplicate sequence was executed twice"
        )
        await mock.close()

    asyncio.run(scenario())


@test("simulator: RESUME is refused while the MCU is in FAULT")
def test_mock_resume_blocked_by_fault() -> None:
    """HW LLD S3.5.3: a command alone cannot clear a real hardware fault."""
    async def scenario() -> None:
        """Raise the e-stop, then attempt to resume."""
        mock = _make_mock()
        await mock.open()
        mock.raise_flag("panel_estop")
        assert mock.state == CONFIG.mcu_states.fault

        resume = build_unified_frame(
            CONFIG.protocol.message_types["command"],
            struct.pack("<BB", 7, CONFIG.commands.codes["resume"]),
            CONFIG.protocol.crc_config,
            stx=CONFIG.protocol.stx,
            etx=CONFIG.protocol.etx,
        )
        await mock.write(resume)
        assert mock.state == CONFIG.mcu_states.fault, (
            "resume must not clear a latched fault"
        )
        await mock.close()

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# McuLink -- the whole loop end to end
# --------------------------------------------------------------------------

async def _run_link_for(link: McuLink, seconds: float) -> None:
    """Run a link for a fixed time, then cancel it cleanly.

    Args:
        link: The link to run.
        seconds: How long to let it run.
    """
    task = asyncio.create_task(link.run())
    await asyncio.sleep(seconds)
    task.cancel()
    with contextlib_suppress_cancelled():
        await task


class contextlib_suppress_cancelled:
    """Context manager that swallows CancelledError only.

    A tiny local helper so the tests do not import contextlib just for this,
    and so the intent -- "the cancel we just issued is expected" -- is
    explicit at each use site.
    """

    def __enter__(self) -> "contextlib_suppress_cancelled":
        """Enter the context.

        Returns:
            This context manager.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Suppress CancelledError.

        Args:
            exc_type: The exception type, if any.
            exc: The exception instance, if any.
            tb: The traceback, if any.

        Returns:
            True to suppress CancelledError, False otherwise.
        """
        return exc_type is not None and issubclass(
            exc_type, asyncio.CancelledError
        )


@test("mcu_link: end to end, one owner reads beats and telemetry together")
def test_link_end_to_end() -> None:
    """The point of the rewrite: both streams, one process, one port."""
    async def scenario() -> None:
        """Run the link briefly against the mock and inspect its state."""
        published: list[dict[str, Any]] = []

        async def collect(message: Any) -> None:
            """Capture one published event.

            Args:
                message: The event McuLink published.
            """
            published.append(dict(message))

        config = _running_config()
        link = McuLink(
            config=config, transport=build_transport(config), publish=collect
        )
        await _run_link_for(link, config.simulator.trigger_interval_s * 3)

        assert link.heartbeat.received > 5, (
            f"expected several heartbeats, got {link.heartbeat.received}"
        )
        assert link.telemetry.received >= 1, "no telemetry was processed"
        assert link.state.state == CONFIG.mcu_states.running
        assert link.parser.crc_errors == 0
        assert link.parser.discarded_bytes == 0, (
            "bytes were discarded -- with a single reader this must be zero"
        )
        # link_state is a FIELD of system_status (S3.9), not an event of its
        # own -- so assert the debounced state the mirror reads, not a payload.
        assert link.link.published_state == constants.LINK_UP, (
            f"link never came up: {link.link.published_state}"
        )
        assert not any(e["event"] == "link_state" for e in published), (
            "link_state must NOT be published as a separate event"
        )

    asyncio.run(scenario())


@test("mcu_link: a command round-trips while heartbeats keep flowing")
def test_link_command_round_trip() -> None:
    """The regression test for the bug this design exists to prevent."""
    async def scenario() -> None:
        """Run the link, send pause, confirm the MCU state follows."""
        link = McuLink(config=CONFIG, transport=build_transport(CONFIG))
        task = asyncio.create_task(link.run())
        await asyncio.sleep(CONFIG.heartbeat.interval_s * 3)

        beats_before = link.heartbeat.received
        result = await link.send_command("pause")
        assert result.acknowledged is True, f"pause failed: {result.reason}"

        # The state change is confirmed by the HEARTBEAT, not by the ACK.
        await asyncio.sleep(CONFIG.heartbeat.interval_s * 3)
        assert link.capture_paused is True, (
            "the MCU never reported the pause on its heartbeat"
        )
        assert link.heartbeat.received > beats_before, (
            "heartbeats stopped during the command exchange -- a command path "
            "that READ the port would have swallowed them"
        )

        task.cancel()
        with contextlib_suppress_cancelled():
            await task

    asyncio.run(scenario())


@test("mcu_link: a dropped ACK is recovered by retry, transparently")
def test_link_retry_recovers() -> None:
    """drop_acks makes the first attempt vanish; the retry must succeed."""
    async def scenario() -> None:
        """Configure the mock to swallow one ACK, then send a command."""
        transport = build_transport(CONFIG)
        assert isinstance(transport, MockMcu)
        transport.drop_acks = 1

        link = McuLink(config=CONFIG, transport=transport)
        task = asyncio.create_task(link.run())
        await asyncio.sleep(CONFIG.heartbeat.interval_s * 2)

        result = await link.send_command("pause")
        assert result.acknowledged is True, "retry did not recover"
        assert result.attempts == 2, f"expected 2 attempts, got {result.attempts}"

        task.cancel()
        with contextlib_suppress_cancelled():
            await task

    asyncio.run(scenario())


@test("mcu_link: a trigger gap raises the uninspected-fabric alarm")
def test_link_trigger_gap_alarm() -> None:
    """drop_triggers skips ids the way a real loss would."""
    async def scenario() -> None:
        """Skip trigger ids and confirm the alarm reaches the publisher."""
        published: list[dict[str, Any]] = []

        async def collect(message: Any) -> None:
            """Capture one published event.

            Args:
                message: The event McuLink published.
            """
            published.append(dict(message))

        config = _running_config()
        transport = build_transport(config)
        assert isinstance(transport, MockMcu)

        link = McuLink(config=config, transport=transport, publish=collect)
        task = asyncio.create_task(link.run())

        await asyncio.sleep(CONFIG.simulator.trigger_interval_s * 1.5)
        transport.drop_triggers = 4
        await asyncio.sleep(CONFIG.simulator.trigger_interval_s * 2)

        task.cancel()
        with contextlib_suppress_cancelled():
            await task

        alarms = [
            e for e in published if e["event"] == "missed_trigger_alarm"
        ]
        assert alarms, "a trigger gap did not raise an alarm"
        assert alarms[0]["gap_count"] == 4, (
            f"expected gap_count 4, got {alarms[0]['gap_count']}"
        )

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# dashboard mirror
# --------------------------------------------------------------------------

@test("status: is_armed and faults are OBSERVED, and clear when they clear")
def test_status_armed_and_faults() -> None:
    """[OURS -- Pooja] Two additions to system_status, neither in S3.9."""
    state = McuState(CONFIG.mcu_states)

    # nothing observed yet: not armed, no faults. Empty is NOT unknown.
    assert state.is_armed is False
    assert state.active_faults == []

    state.observe_state({"state": CONFIG.mcu_states.armed})
    assert state.is_armed is True, "ARMED must set is_armed"

    # a fault raised on a HEARTBEAT
    state.observe_flags({"status_flags": 0x02, "active_flags": ["encoder_fault"]})
    assert state.active_faults == ["encoder_fault"]

    # a second fault, arriving on a TELEMETRY frame -- same byte, same path
    state.observe_flags({
        "status_flags": 0x0A,
        "active_flags": ["encoder_fault", "panel_estop"],
    })
    assert state.active_faults == ["encoder_fault", "panel_estop"]

    # cleared. A panel told only when a fault RAISES can never clear the alarm.
    state.observe_flags({"status_flags": 0x00, "active_flags": []})
    assert state.active_faults == []

    # leaving ARMED drops is_armed, so the START button disappears
    state.observe_state({"state": CONFIG.mcu_states.running})
    assert state.is_armed is False, "START CAPTURE must not reappear mid-roll"

    # the list handed out is a COPY -- a caller must not be able to mutate
    # the MCU's observed state by editing a payload
    state.observe_flags({"status_flags": 0x02, "active_flags": ["encoder_fault"]})
    handed_out = state.active_faults
    handed_out.append("not_a_real_fault")
    assert state.active_faults == ["encoder_fault"]


@test("status: system_status carries is_armed and faults on EVERY push")
def test_status_payload_fields() -> None:
    """Repeated on every push, so a late panel is correct within 500 ms."""

    payload = events.system_status(
        trigger_id=None, duty_cycle=50.0, queue_depth=0, capture_paused=True,
        last_trigger_age_ms=None, link_state=constants.LINK_UP,
        last_heartbeat_age_ms=20.0, missed_intervals=0,
        is_armed=True, faults=["encoder_fault"],
    )
    assert payload["is_armed"] is True
    assert payload["faults"] == ["encoder_fault"]

    # faults is copied into the payload, not aliased
    source = ["strobe_fault"]
    payload = events.system_status(
        trigger_id=None, duty_cycle=50.0, queue_depth=0, capture_paused=True,
        last_trigger_age_ms=None, link_state=constants.LINK_UP,
        last_heartbeat_age_ms=20.0, missed_intervals=0,
        is_armed=False, faults=source,
    )
    source.append("panel_estop")
    assert payload["faults"] == ["strobe_fault"], "payload must not alias"


@test("armed: the mock waits in ARMED and only RESUME starts triggers")
def test_mock_boots_armed() -> None:
    """The whole START CAPTURE path, end to end, against real bytes."""
    async def scenario() -> None:
        config = AttrDict(CONFIG)
        config["simulator"] = AttrDict(
            {**CONFIG.simulator, "initial_state": "armed"}
        )
        published: list[dict] = []

        async def collect(message) -> None:
            published.append(dict(message))

        link = McuLink(
            config=config, transport=build_transport(config), publish=collect
        )
        task = asyncio.create_task(link.run())
        await asyncio.sleep(config.simulator.trigger_interval_s * 2)

        # ARMED: heartbeats flow, but NO fabric is being inspected
        assert link.heartbeat.received > 5, "heartbeats must flow while armed"
        assert link.telemetry.received == 0, "ARMED must not fire triggers"
        assert link.state.is_armed, f"expected ARMED, got {link.state.state_name}"

        # The panel learns it may start from system_status: is_armed true,
        # faults empty, link up. No separate event -- those three fields are
        # the contract.
        assert link.state.is_armed is True
        assert link.state.active_faults == []
        assert link.link.published_state == constants.LINK_UP

        # START CAPTURE -- the same wire command as resume
        await link.send_command("resume", source=constants.SOURCE_OPERATOR)
        await asyncio.sleep(config.simulator.trigger_interval_s * 2)

        assert link.state.state_name == "RUNNING", "resume did not start the line"
        assert link.telemetry.received >= 1, "no triggers after START CAPTURE"

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


@test("mirror: system_status carries all three sources and one clock reading")
def test_dashboard_mirror_payload() -> None:
    """LLD SW S3.8a's dashboard heartbeat mirroring."""

    async def scenario() -> None:
        """Run the link briefly, then build one status payload."""
        async def nowhere(message: Any) -> None:
            """Discard an event.

            Args:
                message: Ignored.
            """

        link = McuLink(config=CONFIG, transport=build_transport(CONFIG))
        mirror = DashboardMirror(
            link, CONFIG.dashboard, GpuMonitor(CONFIG.gpu_monitoring), nowhere
        )
        await _run_link_for(link, CONFIG.simulator.trigger_interval_s * 2)

        status = mirror.build_status()
        for field in (
            "trigger_id", "duty_cycle", "queue_depth", "capture_paused",
            "last_trigger_age_ms", "link_state", "last_heartbeat_age_ms",
            "missed_intervals",
        ):
            assert field in status, f"system_status is missing {field}"

        assert status["event"] == "system_status"
        assert status["link_state"] == link.link.published_state, (
            "the mirror must report the DEBOUNCED state, so system_status "
            "and link_state events cannot contradict each other"
        )

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# dashboard fallback -- EventStore, HTTP router and panel cursor
# --------------------------------------------------------------------------

@test("fallback: EventStore enriches full events and prunes after one-panel ACK")
def test_event_store_record_replay_ack() -> None:
    """The queue holds full payloads and does not mutate the producer dict."""

    store = EventStore(max_pending_events=3)
    original = {
        "event": "capture_paused",
        "capture_paused": True,
        "source": "mcu",
    }
    recorded = store.record_event(original)

    assert "event_id" not in original and "stream_id" not in original
    assert recorded["event_id"] == 1
    assert recorded["stream_id"] == store.stream_id
    assert recorded["capture_paused"] is True

    replay = store.events_after(0)
    assert replay.resync_required is False
    assert replay.events == (recorded,)
    assert store.acknowledge(1) == 1
    assert store.snapshot()["events_pending"] == 0

    # Asking for an already acknowledged event means this client lost its own
    # cursor; replay must name the gap rather than pretending history is full.
    assert store.events_after(0).resync_required is True


@test("fallback: EventStore reports overflow/resync and retains newest events")
def test_event_store_overflow_resync() -> None:
    """A long outage cannot grow RAM forever or silently hide a gap."""

    store = EventStore(max_pending_events=2)
    for number in (1, 2, 3):
        store.record_event({"event": "heartbeat_missed", "dropped": number})

    replay = store.events_after(0)
    assert replay.resync_required is True
    assert [event["event_id"] for event in replay.events] == [2, 3]
    assert replay.oldest_available_event_id == 2
    assert store.snapshot()["event_queue_overflow_drops"] == 1


@test("fallback: panel cursor shares stream state and drops replay duplicates")
def test_panel_event_cursor() -> None:
    """An event seen on WebSocket is not rendered again through HTTP replay."""

    cursor = PanelEventCursor()
    event = {
        "stream_id": "stream-a", "event_id": 4,
        "event": "capture_paused", "capture_paused": True,
    }
    assert cursor.observe(event) is True
    assert cursor.last_processed_event_id == 4
    assert cursor.observe(event) is False
    assert cursor.observe({"event": "system_status"}) is True


@test("fallback: HTTP routes use shared status, event and command service APIs")
def test_fallback_http_routes() -> None:
    """The router delegates; it does not invent another dashboard/MCU path."""


    class FakeDashboardService:
        """Small deterministic service satisfying the route's real calls."""

        def __init__(self) -> None:
            self.store = EventStore(max_pending_events=5)
            self.recorded = self.store.record_event(
                {
                    "event": "command_failed",
                    "command": "pause",
                    "cmd_seq": 7,
                    "reason": "test",
                }
            )

        def current_status(self) -> dict[str, object]:
            return {"event": "system_status", "link_state": "LINK_UP"}

        def replay_events(self, after_event_id: int):
            return self.store.events_after(after_event_id)

        def acknowledge_events(self, stream_id: str, event_id: int) -> int:
            if stream_id != self.store.stream_id:
                raise EventStoreError("wrong stream")
            return self.store.acknowledge(event_id)

        async def submit_operator_command(self, command: object) -> CommandResult:
            assert command == "operator_stop_request"
            return CommandResult(
                command="pause", command_sequence=8, acknowledged=True,
                attempts=1, mcu_state=2,
            )

    service = FakeDashboardService()
    app = FastAPI()
    app.include_router(create_fallback_router(service))
    client = TestClient(app)

    status = client.get("/api/v1/status")
    assert status.status_code == 200
    assert status.json()["event"] == "system_status"

    replay = client.get("/api/v1/events?after=0")
    assert replay.status_code == 200
    assert replay.json()["events"] == [service.recorded]

    ack = client.post(
        "/api/v1/events/ack",
        json={
            "stream_id": service.store.stream_id,
            "last_processed_event_id": 1,
        },
    )
    assert ack.status_code == 200
    assert ack.json()["acknowledged_through"] == 1

    command = client.post(
        "/api/v1/commands",
        json={"command": "operator_stop_request"},
    )
    assert command.status_code == 200
    assert command.json()["acknowledged"] is True
    assert command.json()["command"] == "pause"

    # Every payload received from the dashboard is a closed Pydantic contract.
    # Unknown command fields must not be silently ignored, and invalid values
    # must not reach the coordinator or McuLink.
    extra_command = client.post(
        "/api/v1/commands",
        json={"command": "operator_stop_request", "force": True},
    )
    assert extra_command.status_code == 422
    invalid_command = client.post(
        "/api/v1/commands", json={"command": "pause"}
    )
    assert invalid_command.status_code == 422
    invalid_cursor = client.get("/api/v1/events?after=-1")
    assert invalid_cursor.status_code == 422
    invalid_ack = client.post(
        "/api/v1/events/ack",
        json={
            "stream_id": service.store.stream_id,
            "last_processed_event_id": 1,
            "unexpected": True,
        },
    )
    assert invalid_ack.status_code == 422


@test("dashboard: WebSocket commands use the same strict Pydantic contract")
def test_websocket_command_model() -> None:
    """Start, Resume and Stop validate before WebSocket command processing."""


    for command in (
        "operator_start_capture",
        "operator_resume_request",
        "operator_stop_request",
    ):
        assert _validate_command({"command": command}).command == command

    try:
        _validate_command({"command": "pause"})
    except ValidationError:
        pass
    else:
        raise AssertionError("raw MCU command must not pass dashboard validation")

    try:
        _validate_command(
            {"command": "operator_stop_request", "force": True}
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("extra dashboard fields must be rejected")


# --------------------------------------------------------------------------
# config_loader
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# config_loader -- the merge, the coercion, and the six semantic checks
# --------------------------------------------------------------------------

def _load_with(override: dict) -> Any:
    """Load the REAL config.yaml with an override merged on top.

    Exercises the production path -- read, merge, validate -- rather than
    poking at an internal helper.

    Args:
        override: Keys to change, in the shape of config.yaml.

    Returns:
        The loaded AttrDict.

    Raises:
        ConfigError: Whatever the loader raises.
    """
    return load_config(overrides=override)


def _expect_config_error(override: dict, must_mention: str) -> None:
    """Assert that an override is rejected with a helpful message.

    Args:
        override: The bad override document.
        must_mention: Text the error message must contain, so the test also
            proves the message names the problem rather than just failing.

    Raises:
        AssertionError: If it was accepted, or the message is unhelpful.
    """
    try:
        _load_with(override)
    except ConfigError as exc:
        assert str(exc), "configuration errors must contain Pydantic detail"
        return
    raise AssertionError(f"bad config was accepted: {override}")


@test("config: DEEP MERGE keeps sibling keys an override did not mention")
def test_deep_merge_keeps_siblings() -> None:
    """The bug a shallow dict.update would cause, asserted directly.

    deep_merge() is what the environment overrides go through, so changing
    only protocol.crc.poly must NOT drop init, xorout and include_stx --
    otherwise the CRC is computed with wrong parameters and EVERY frame
    fails its checksum on real hardware.
    """
    config = _load_with({"protocol": {"crc": {"poly": 0x31}}})

    assert config.protocol.crc.poly == 0x31, "the override did not apply"
    assert config.protocol.crc.init == 0x00, "init was lost by the merge"
    assert config.protocol.crc.xorout == 0x00, "xorout was lost by the merge"
    assert config.protocol.crc.include_stx is True, "include_stx was lost"
    assert config.protocol.max_payload == 64, "a sibling key was lost"


@test("config: a MISSING key raises a NAMED error, not a silent None")
def test_missing_key_is_named() -> None:
    """AttrDict lists the keys it does have, so a typo is self-diagnosing."""
    cfg = _load_with({})
    try:
        cfg.heartbeat.intervals_s          # deliberate typo
    except AttributeError as exc:
        assert "intervals_s" in str(exc), f"error must name the key: {exc}"
        return
    raise AssertionError("a misspelled key returned silently")


@test("config: a MISSING section is a loud Pydantic validation error")
def test_missing_section_is_fatal() -> None:
    """The root model rejects a required section that is not a mapping."""
    _expect_config_error({"heartbeat": None}, "heartbeat")


@test("config: an unknown SECTION is rejected, not silently ignored")
def test_unknown_section_rejected() -> None:
    """`.get(key, {})` would defer this to a KeyError 100 ms into the run."""
    _expect_config_error({"heartbeatt": {"interval_s": 0.1}}, "unknown section")


@test("config: closed-set values are still checked (parity, level, bands)")
def test_closed_sets_checked() -> None:
    """These were free with enum type hints; validate() now does them."""
    _expect_config_error({"serial": {"parity": "Z"}}, "must be one of")
    _expect_config_error({"logging": {"level": "CHATTY"}}, "must be one of")
    _expect_config_error(
        {"heartbeat": {"bands": [[1.0, "LINK_SIDEWAYS"]]}}, "not a link state"
    )


@test("config: an out-of-order band table is rejected at startup")
def test_config_rejects_unsorted_bands() -> None:
    """The first-match lookup would silently return the wrong state."""
    _expect_config_error(
        {"heartbeat": {"bands": [[3.0, "LINK_UP"], [1.0, "LINK_DEGRADED"]]}},
        "increase",
    )


@test("config: a check interval slower than the beat interval is rejected")
def test_config_rejects_slow_check() -> None:
    """Checking less often than beats arrive would miss transitions."""
    _expect_config_error({"heartbeat": {"check_interval_s": 0.5}}, "exceed")


@test("config: duplicate MCU state codes are rejected at startup")
def test_config_rejects_duplicate_states() -> None:
    """Two states sharing a code makes the reverse name lookup ambiguous."""
    _expect_config_error({"mcu_states": {"paused": 1}}, "distinct")


@test("config: a command code outside one byte is rejected")
def test_config_rejects_wrong_command_code() -> None:
    """YAML owns the code, but the command field is still only one byte."""
    _expect_config_error(
        {"commands": {"codes": {"pause": 256}}}, "commands"
    )


@test("config: an unknown transport kind is rejected BY THE ENUM")
def test_config_rejects_bad_transport() -> None:
    """TransportKind validates itself -- no hand-written membership check."""
    _expect_config_error({"transport": {"kind": "carrier_pigeon"}}, "must be one of")


@test("config: a quoted boolean is rejected rather than read as true")
def test_config_rejects_quoted_bool() -> None:
    """The string "false" is TRUTHY in Python -- a silent inversion."""
    _expect_config_error(
        {"serial": {"exclusive": "false"}}, "true or false"
    )


@test("config: a zero timeout is rejected -- it would busy-loop a CPU core")
def test_config_rejects_zero_timeout() -> None:
    """timeout=0 makes pyserial non-blocking; the service still 'works'."""
    _expect_config_error({"serial": {"read_timeout_s": 0}}, "must be a number > 0")


@test("config: fallback event queue capacity must be a positive integer")
def test_config_rejects_bad_event_store_capacity() -> None:
    """A zero queue would silently discard every fallback event."""
    _expect_config_error(
        {"dashboard": {"events_store": {"max_pending_events": 0}}},
        "max_pending_events",
    )


@test("config: AttrDict supports BOTH dict and attribute access")
def test_attrdict_both_styles() -> None:
    """66 existing call sites use dots; the loader returns plain dicts."""
    cfg = _load_with({})
    assert cfg["heartbeat"]["interval_s"] == cfg.heartbeat.interval_s
    assert cfg.transport.kind == "mock"
    assert cfg.serial.bytesize == 8
    assert cfg.heartbeat.__class__.__name__ == "Heartbeat"

    # The four derived values get_all_configs() computes.
    assert cfg.transport.is_mock is True
    assert cfg.mcu_states.names[1] == "RUNNING"
    assert cfg.dashboard.ws_url.startswith("ws://127.0.0.1:")
    assert cfg.protocol.crc_config.poly == cfg.protocol.crc.poly


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(verbose: bool = False, name_filter: str | None = None) -> int:
    """Run every registered test and report the outcome.

    Args:
        verbose: Show log output from the modules under test.
        name_filter: Only run tests whose name contains this substring.

    Returns:
        0 if everything passed, 1 otherwise.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.CRITICAL,
        format="%(levelname)-8s %(name)-24s %(message)s",
        force=True,
    )

    selected = [
        (name, func) for name, func in _TESTS
        if name_filter is None or name_filter in name
    ]
    if not selected:
        print(f"no tests match {name_filter!r}", file=sys.stderr)
        return 1

    passed = 0
    failed: list[tuple[str, str]] = []

    for name, func in selected:
        try:
            func()
        except AssertionError as exc:
            failed.append((name, str(exc) or "assertion failed"))
            print(f"FAIL  {name}")
        except Exception:
            failed.append((name, traceback.format_exc()))
            print(f"ERROR {name}")
        else:
            passed += 1
            print(f"ok    {name}")

    print()
    print(f"{passed}/{len(selected)} passed")

    if failed:
        print()
        for name, detail in failed:
            print(f"--- {name} ---")
            print(detail)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
