"""Verify framing uses supplied YAML settings, without hardware or services."""

import copy
import struct
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from config.models import Commands, Protocol
from control_service.protocol import (
    UnifiedStreamParser,
    build_command_frame,
    build_unified_frame,
    decode_unified_frame,
    status_flag_bit,
)


class ProtocolConfigurationTests(unittest.TestCase):
    """Exercise non-default codes to expose accidental hardcoded fallbacks."""

    def setUp(self) -> None:
        """Read only protocol settings, without loading credentials or models."""
        with (Path(__file__).resolve().parents[1] / "config.yaml").open(
            encoding="utf-8",
        ) as stream:
            raw = yaml.safe_load(stream)
        self.raw_protocol = raw["protocol"]
        self.raw_commands = raw["commands"]
        changed = copy.deepcopy(self.raw_protocol)
        changed["stx"] = 0xA5
        changed["etx"] = 0x5A
        changed["message_types"] = {
            "telemetry": 0x11, "heartbeat": 0x12,
            "command": 0x13, "ack_nack": 0x14,
        }
        changed["status_flags"] = {
            "strobe_fault": 4, "encoder_fault": 5,
            "overcurrent_trip": 6, "panel_estop": 7,
        }
        self.protocol = Protocol.model_validate(changed)
        commands = copy.deepcopy(self.raw_commands)
        commands["codes"] = {"pause": 0x21, "resume": 0x22, "stop": 0x23}
        self.commands = Commands.model_validate(commands)
        self.decode_settings = {
            "stx": self.protocol.stx,
            "etx": self.protocol.etx,
            "message_types": self.protocol.message_types,
            "command_names": self.commands.names,
            "status_flag_names": self.protocol.status_flag_names,
        }

    def frame(self, name: str, payload: bytes) -> bytes:
        """Build a message using the deliberately altered configuration."""
        return build_unified_frame(
            self.protocol.message_types[name], payload, self.protocol.crc_config,
            stx=self.protocol.stx, etx=self.protocol.etx,
        )

    def decode(self, frame: bytes) -> dict[str, object]:
        """Decode with the same injected configuration as the parser."""
        return decode_unified_frame(
            frame, self.protocol.crc_config, {},
            self.protocol.payload_lengths_by_code, **self.decode_settings,
        )

    def test_telemetry_and_heartbeat_use_configured_codes_and_flags(self):
        for name, payload in (
            ("telemetry", struct.pack("<IiiIbBB", 7, -5, -250, 123, -1, 1, 128)),
            ("heartbeat", struct.pack("<IBBH", 123, 1, 128, 10)),
        ):
            with self.subTest(name=name):
                frame = self.frame(name, payload)
                self.assertEqual(frame[0], self.protocol.stx)
                self.assertEqual(frame[-1], self.protocol.etx)
                decoded = self.decode(frame)
                self.assertEqual(decoded["message_type"], name)
                self.assertEqual(decoded["active_flags"], ["panel_estop"])

    def test_command_and_ack_use_configured_command_names(self):
        frame = build_command_frame(
            7, self.commands.codes["pause"], self.protocol.crc_config,
            message_type=self.protocol.message_types["command"],
            stx=self.protocol.stx, etx=self.protocol.etx,
        )
        self.assertEqual(self.decode(frame)["command_name"], "PAUSE")
        ack = self.frame(
            "ack_nack", struct.pack("<BBBB", 7, self.commands.codes["pause"], 0, 1),
        )
        self.assertEqual(self.decode(ack)["command_name"], "PAUSE")

    def test_stream_parser_recovers_noise_and_fragmented_frames(self):
        parser = UnifiedStreamParser(
            self.protocol.crc_config, {}, self.protocol.max_payload,
            self.protocol.payload_lengths_by_code, **self.decode_settings,
        )
        frame = self.frame("heartbeat", struct.pack("<IBBH", 123, 1, 0, 10))
        self.assertEqual(parser.feed(b"\x99\x88" + frame[:4]), [])
        messages = parser.feed(frame[4:])
        self.assertEqual(messages[0]["message_type"], "heartbeat")
        self.assertEqual(parser.discarded_bytes, 2)
        damaged = bytearray(frame)
        damaged[4] ^= 1
        self.assertEqual(parser.feed(bytes(damaged)), [])
        self.assertEqual(parser.crc_errors, 1)

    def test_wrong_packet_length_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "payload must be 19 bytes"):
            self.decode(self.frame("telemetry", bytes(14)))

    def test_unknown_message_remains_skippable(self):
        frame = build_unified_frame(
            0xFE, b"abc", self.protocol.crc_config,
            stx=self.protocol.stx, etx=self.protocol.etx,
        )
        self.assertEqual(self.decode(frame)["message_type"], "unknown")

    def test_invalid_yaml_is_rejected_before_parser_creation(self):
        cases = [
            ("payload_lengths", {"telemetry": 18}),
            ("payload_lengths", {"extra": 19}),
            ("message_types", {"heartbeat": self.raw_protocol["message_types"]["telemetry"]}),
            ("status_flags", {"panel_estop": 8}),
        ]
        for section, update in cases:
            with self.subTest(section=section, update=update):
                raw = copy.deepcopy(self.raw_protocol)
                raw[section].update(update)
                with self.assertRaises(ValidationError):
                    Protocol.model_validate(raw)
        raw = copy.deepcopy(self.raw_protocol)
        del raw["payload_lengths"]["telemetry"]
        with self.assertRaises(ValidationError):
            Protocol.model_validate(raw)

    def test_invalid_command_codes_are_rejected(self):
        for codes in (
            {"pause": 1, "resume": 1, "stop": 3},
            {"pause": 256, "resume": 2, "stop": 3},
            {"pause": 1},
        ):
            with self.subTest(codes=codes):
                raw = {**self.raw_commands, "codes": codes}
                with self.assertRaises(ValidationError):
                    Commands.model_validate(raw)

    def test_flag_lookup_uses_configured_bit(self):
        self.assertEqual(
            status_flag_bit("panel_estop", self.protocol.status_flags), 7,
        )
        with self.assertRaises(ValueError):
            status_flag_bit("unknown", self.protocol.status_flags)
