"""The VisionBar wire protocol. Self-contained -- no imports outside stdlib.

HW LLD v1.3 S3.5.2 unified frame:

    STX | MSG_TYPE | LEN | PAYLOAD | CRC8 | ETX

    TELEMETRY  trigger_id(4) encoder_count(4, signed)
               distance_mm_x1000(4, signed) timestamp_us(4)
               direction(1, signed) state(1) status_flags(1)
    HEARTBEAT  timestamp_us(4) state(1) status_flags(1) uptime_s(2)
                    <- NO SEQUENCE NUMBER. A missing beat can only be found
                       by comparing timestamps, unlike telemetry where
                       trigger_id makes a gap directly visible.
    COMMAND    cmd_seq(1) cmd_code(1)
    ACK/NACK   cmd_seq(1) cmd_code(1) result(1) state(1)

Markers, message codes, payload lengths, flag bits and CRC parameters come
from validated YAML settings passed by callers. Binary field layouts remain
part of the encoder/decoder implementation.

This file was merged in from ../../visionbar_protocol.py so that working_code
is a self-contained tree: no sys.path shims, no imports from outside it.

DATA IN   : raw bytes off the transport
DATA OUT  : decoded message dicts; built frames
USED BY   : mcu_link.py, mcu_command.py, transport.py, simulator/mock_mcu.py
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CrcConfig:
    """CRC settings supplied by the validated application configuration."""

    poly: int
    init: int
    xorout: int
    include_stx: bool


def crc8(data: Iterable[int], config: CrcConfig) -> int:
    """Calculate non-reflected, MSB-first CRC-8."""
    crc = config.init
    for value in data:
        crc ^= value
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ config.poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc ^ config.xorout


def _crc_input(frame_without_crc_etx: bytes, config: CrcConfig) -> bytes:
    return (
        frame_without_crc_etx
        if config.include_stx
        else frame_without_crc_etx[1:]
    )


def _active_flags(
    status_flags: int, flag_names: dict[int, str],
) -> list[str]:
    """Decode set bits using names derived from YAML."""
    return [
        name for bit, name in flag_names.items() if status_flags & (1 << bit)
    ]


def _state_name(state: int, state_names: dict[int, str]) -> str:
    return state_names.get(state, f"UNKNOWN_0x{state:02X}")


def build_unified_frame(
    message_type: int,
    payload: bytes,
    crc_config: CrcConfig,
    *,
    stx: int,
    etx: int,
) -> bytes:
    """Wrap a payload with configured markers and its calculated CRC."""
    if not 0 <= message_type <= 0xFF:
        raise ValueError("message_type must fit in one byte")
    if len(payload) > 0xFF:
        raise ValueError("payload cannot exceed 255 bytes")

    body = bytes((stx, message_type, len(payload))) + payload
    checksum = crc8(_crc_input(body, crc_config), crc_config)
    return body + bytes((checksum, etx))


def build_command_frame(
    command_sequence: int,
    command_code: int,
    crc_config: CrcConfig,
    *,
    message_type: int,
    stx: int,
    etx: int,

) -> bytes:
    """Build one COMMAND frame: cmd_seq(1) | cmd_code(1)."""

    return build_unified_frame(
        message_type=message_type,
        payload=struct.pack("<BB", command_sequence, command_code),
        crc_config=crc_config,
        stx=stx,
        etx=etx,
    )


def decode_unified_frame(
    frame: bytes,
    crc_config: CrcConfig,
    state_names: dict[int, str],
    payload_lengths: dict[int, int],
    *,
    stx: int,
    etx: int,
    message_types: dict[str, int],
    command_names: dict[int, str],
    status_flag_names: dict[int, str],
) -> dict[str, object]:
    """Decode a frame using only the protocol settings supplied by its caller."""
    message_names = {code: name for name, code in message_types.items()}
    if len(frame) < 5:
        raise ValueError("unified frame is shorter than five bytes")
    if frame[0] != stx:
        raise ValueError(f"bad STX: 0x{frame[0]:02X}")
    if frame[-1] != etx:
        raise ValueError(f"bad ETX: 0x{frame[-1]:02X}")

    message_type = frame[1]
    payload_length = frame[2]
    if len(frame) != payload_length + 5:
        raise ValueError(f"LEN declares {payload_length}, frame has {len(frame)} bytes")

    received_crc = frame[-2]
    calculated_crc = crc8(_crc_input(frame[:-2], crc_config), crc_config)
    if received_crc != calculated_crc:
        raise ValueError(
            f"bad CRC: received 0x{received_crc:02X}, "
            f"calculated 0x{calculated_crc:02X}"
        )

    # Unknown types remain skippable; known types use the configured length.
    expected = payload_lengths.get(message_type)
    if expected is not None and payload_length != expected:
        raise ValueError(
            f"{message_names.get(message_type, 'type 0x%02X' % message_type)} "
            f"payload must be {expected} bytes, got {payload_length}"
        )

    payload = frame[3:-2]
    decoded: dict[str, object] = {
        "message_type": message_names.get(message_type, "unknown"),
        "message_type_code": message_type,
    }

    if message_type == message_types["telemetry"]:
        (
            trigger_id, encoder_count, distance_mm_x1000, timestamp_us,
            direction, state, flags,
        ) = struct.unpack("<IiiIbBB", payload)
        decoded.update(
            trigger_id=trigger_id,
            encoder_count=encoder_count,
            distance_mm_x1000=distance_mm_x1000,
            timestamp_us=timestamp_us,
            direction=direction,
            state=state,
            state_name=_state_name(state, state_names),
            status_flags=flags,
            active_flags=_active_flags(flags, status_flag_names),
        )
    elif message_type == message_types["heartbeat"]:
        timestamp_us, state, flags, uptime_s = struct.unpack("<IBBH", payload)
        decoded.update(
            timestamp_us=timestamp_us,
            state=state,
            state_name=_state_name(state, state_names),
            status_flags=flags,
            active_flags=_active_flags(flags, status_flag_names),
            uptime_s=uptime_s,
        )
    elif message_type == message_types["command"]:
        command_sequence, command_code = struct.unpack("<BB", payload)
        decoded.update(
            command_sequence=command_sequence,
            command_code=command_code,
            command_name=command_names.get(command_code, "UNKNOWN"),
        )
    elif message_type == message_types["ack_nack"]:
        command_sequence, command_code, result, state = struct.unpack("<BBBB", payload)
        decoded.update(
            command_sequence=command_sequence,
            command_code=command_code,
            command_name=command_names.get(command_code, "UNKNOWN"),
            result=result,
            state=state,
            state_name=_state_name(state, state_names),
        )
    else:
        decoded["payload_hex"] = payload.hex(" ")

    decoded["raw_hex"] = frame.hex(" ")
    return decoded


class UnifiedStreamParser:
    """Recover variable-length unified frames from fragmented serial reads."""

    def __init__(
        self,
        crc_config: CrcConfig,
        state_names: dict[int, str],
        max_payload: int,
        payload_lengths: dict[int, int],
        *,
        stx: int,
        etx: int,
        message_types: dict[str, int],
        command_names: dict[int, str],
        status_flag_names: dict[int, str],
    ) -> None:
        self.crc_config = crc_config
        self.state_names = state_names
        self.max_payload = max_payload
        self.payload_lengths = payload_lengths
        self.stx = stx
        self.etx = etx
        self.message_types = message_types
        self.command_names = command_names
        self.status_flag_names = status_flag_names
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.crc_errors = 0
        self.etx_errors = 0
        self.length_errors = 0
        self.decode_errors = 0

    def feed(self, data: bytes) -> list[dict[str, object]]:
        self.buffer.extend(data)
        decoded: list[dict[str, object]] = []

        while True:
            header_index = self.buffer.find(bytes((self.stx,)))
            if header_index < 0:
                self.discarded_bytes += len(self.buffer)
                self.buffer.clear()
                break
            if header_index:
                self.discarded_bytes += header_index
                del self.buffer[:header_index]
            if len(self.buffer) < 3:
                break

            payload_length = self.buffer[2]
            if payload_length > self.max_payload:
                self.length_errors += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            frame_length = payload_length + 5
            if len(self.buffer) < frame_length:
                break

            candidate = bytes(self.buffer[:frame_length])
            if candidate[-1] != self.etx:
                self.etx_errors += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            try:
                message = decode_unified_frame(
                    candidate, self.crc_config, self.state_names,
                    self.payload_lengths,
                    stx=self.stx,
                    etx=self.etx,
                    message_types=self.message_types,
                    command_names=self.command_names,
                    status_flag_names=self.status_flag_names,
                )
            except ValueError as exc:
                if "CRC" in str(exc):
                    self.crc_errors += 1
                else:
                    self.decode_errors += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue

            del self.buffer[:frame_length]
            decoded.append(message)
        return decoded


def parser_counters(parser: UnifiedStreamParser) -> str:
    return (
        f"discarded_bytes={parser.discarded_bytes}, "
        f"crc_errors={parser.crc_errors}, etx_errors={parser.etx_errors}, "
        f"length_errors={parser.length_errors}, decode_errors={parser.decode_errors}"
    )


def status_flag_bit(name: str, flag_bits: dict[str, int]) -> int:
    """Look up a configured flag bit, rejecting unknown operator flag names."""
    try:
        return flag_bits[name]
    except KeyError as exc:
        raise ValueError(f"unknown status flag {name!r}") from exc


__all__ = [
    "CrcConfig", "UnifiedStreamParser", "build_command_frame",
    "build_unified_frame", "crc8", "decode_unified_frame", "parser_counters",
    "status_flag_bit",
]
