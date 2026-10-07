"""Enums and symbolic names fixed by the design documents. NOT tunables.

WHERE THIS SITS
===============

    config.yaml           every NUMBER you might change  -> config_loader.py
    constants.py          every NAME the LLD fixes        -> this file

The split matters. "300 ms is LINK_DOWN" is a tuning decision and belongs in
YAML. The string "LINK_DOWN" is part of the LLD SW S3.9 wire contract the
dashboard parses -- changing it breaks the panel, so it is not a knob and
must not look like one.

WHY ENUMS RATHER THAN LOOSE STRINGS
===================================

Seven settings in the YAML are closed sets: a transport kind, a parity
letter, a log level, a link state, and so on. Declaring each as an Enum means
config_loader gets its validation FOR FREE -- Enum("carrier_pigeon") raises,
and the coercion layer turns that into a named ConfigError. That replaces
seven hand-written membership checks with zero.

Every enum subclasses (str, Enum) so that:
  - members compare equal to plain strings, so nothing downstream changes
  - json.dumps serialises them as strings, so the wire format is unaffected

(StrEnum would be tidier but is Python 3.11+; this package targets the
Jetson's system Python, so the (str, Enum) mixin is used instead.)

BACK-COMPATIBLE ALIASES
=======================

Module-level names like LINK_UP are kept as aliases for the enum members, so
the eight other modules that already import them need no edits. New code
should prefer the enum: LinkState.LINK_UP.

DATA IN   : nothing -- this module imports only from the standard library
DATA OUT  : imported by config_loader.py and by control_service/*
CREATED BY: nothing, these are module-level definitions
"""

from __future__ import annotations

from enum import Enum, IntEnum


class LinkState(str, Enum):
    """Link health states. LLD SW S3.8a.

    Attributes:
        LINK_UP: A heartbeat arrived within one interval.
        LINK_DEGRADED: One or two intervals of silence.
        LINK_DOWN: Three or more intervals of silence.
    """

    LINK_UP = "LINK_UP"
    LINK_DEGRADED = "LINK_DEGRADED"
    LINK_DOWN = "LINK_DOWN"


#: The states a heartbeat.bands entry may name -- all of them.
CONFIGURABLE_LINK_STATES: frozenset[LinkState] = frozenset(
    {LinkState.LINK_UP, LinkState.LINK_DEGRADED, LinkState.LINK_DOWN}
)


class TransportKind(str, Enum):
    """Which byte source the service talks to. config.yaml `transport.kind`.

    Attributes:
        SERIAL: control_service/transport.py SerialTransport, the real port.
        MOCK: simulator/mock_mcu.py MockMcu, no hardware needed.
    """

    SERIAL = "serial"
    MOCK = "mock"


class Parity(str, Enum):
    """UART parity setting. HW LLD v1.3 specifies N (8-N-1).

    Attributes:
        NONE: No parity bit.
        EVEN: Even parity.
        ODD: Odd parity.
        MARK: Parity bit always 1.
        SPACE: Parity bit always 0.
    """

    NONE = "N"
    EVEN = "E"
    ODD = "O"
    MARK = "M"
    SPACE = "S"


class ByteSize(IntEnum):
    """UART data bits. HW LLD v1.3 specifies 8.

    An IntEnum so it compares and arithmetics as a plain int, which is what
    pyserial's constants expect.

    Attributes:
        FIVE: 5 data bits.
        SIX: 6 data bits.
        SEVEN: 7 data bits.
        EIGHT: 8 data bits -- the VisionBar setting.
    """

    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8


class LogLevel(str, Enum):
    """Root logger level. config.yaml `logging.level`.

    Attributes:
        DEBUG: Everything, including per-frame detail.
        INFO: Normal operation.
        WARNING: Degraded conditions.
        ERROR: Failures.
        CRITICAL: Only fatal problems.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class McuStateName(str, Enum):
    """MCU state machine states. HW LLD S3.5.3.

    The MACHINE runs in firmware; control_service/mcu_state.py only observes
    it. These are the display names for the numeric codes in
    `mcu_states:`.

    Attributes:
        IDLE: Stopped, not faulted.
        RUNNING: The ONLY state in which fabric is being inspected.
        PAUSED: Stopped by a PAUSE command, gracefully.
        FAULT: Stopped autonomously by a status_flags fault bit.
        ARMED: [OURS -- Pooja] Self-checked and healthy, waiting for the
            operator's START CAPTURE. Not in any design document; requires
            firmware to send code 4. Until it does, the MCU reports IDLE.
        UNKNOWN: No frame has been received yet. Not a firmware state.
    """

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FAULT = "FAULT"

    # [OURS -- Pooja] NOT in HW LLD S3.5.3. IDLE currently means three
    # different things -- just booted, never started, stopped and not
    # resumed. ARMED separates "checked and ready" from "powered but
    # unverified", so the panel can offer START CAPTURE only when starting
    # is genuinely safe.
    ARMED = "ARMED"

    # Not a firmware state. "We have not heard from the MCU yet" is genuinely
    # different from "the MCU is IDLE", and conflating them would have the
    # panel assert a state nothing has confirmed.
    UNKNOWN = "UNKNOWN"


class EventSource(str, Enum):
    """Who caused a capture_paused transition. LLD SW S3.9.

    S3.9 requires the UI to render an operator pause differently from a
    fault-triggered one.

    Attributes:
        OPERATOR: A human pressed the button on the panel.
        MCU: The MCU changed state on its own.
        FAULT: The MCU entered FAULT.
        SNAPSHOT: Current state, sent to a newly connected panel.
    """

    OPERATOR = "operator"
    MCU = "mcu"
    FAULT = "fault"
    SNAPSHOT = "snapshot"


class MessageName(str, Enum):
    """Message type names, as UnifiedStreamParser labels them.

    The decoder derives names from YAML message codes.
    McuLink dispatches on these stable names.

    Attributes:
        TELEMETRY: 0x01, one per trigger.
        HEARTBEAT: 0x02, every 100 ms.
        COMMAND: 0x03, host to MCU.
        ACK_NACK: 0x04, MCU to host.
    """

    TELEMETRY = "telemetry"
    HEARTBEAT = "heartbeat"
    COMMAND = "command"
    ACK_NACK = "ack_nack"


# --------------------------------------------------------------------------
# Back-compatible aliases
# --------------------------------------------------------------------------
#
# The eight modules written before the enums existed import these names
# directly. Because every enum above subclasses str, `state == LINK_UP` and
# `json.dumps({"link_state": LINK_UP})` behave exactly as they did with plain
# string constants -- so nothing downstream needed changing.
#
# New code should prefer the enum member: LinkState.LINK_UP.

LINK_UP = LinkState.LINK_UP
LINK_DEGRADED = LinkState.LINK_DEGRADED
LINK_DOWN = LinkState.LINK_DOWN

#: Kept for config_loader's band validation and for existing imports.
LINK_STATES: frozenset[LinkState] = CONFIGURABLE_LINK_STATES

TRANSPORT_SERIAL = TransportKind.SERIAL
TRANSPORT_MOCK = TransportKind.MOCK

MSG_NAME_TELEMETRY = MessageName.TELEMETRY
MSG_NAME_HEARTBEAT = MessageName.HEARTBEAT
MSG_NAME_COMMAND = MessageName.COMMAND
MSG_NAME_ACK_NACK = MessageName.ACK_NACK

SOURCE_OPERATOR = EventSource.OPERATOR
SOURCE_MCU = EventSource.MCU
SOURCE_FAULT = EventSource.FAULT
SOURCE_SNAPSHOT = EventSource.SNAPSHOT
