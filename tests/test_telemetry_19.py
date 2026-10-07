"""Regression tests for the signed 19-byte telemetry wire contract."""

import struct
import unittest
from pathlib import Path

import yaml

from config.models import Commands, Protocol

from control_service.protocol import (
    build_unified_frame, decode_unified_frame,
)


class Telemetry19Tests(unittest.TestCase):
    def test_forward_stopped_and_reverse_round_trip(self):
        with (Path(__file__).resolve().parents[1] / "config.yaml").open(
            encoding="utf-8",
        ) as stream:
            raw = yaml.safe_load(stream)
        protocol = Protocol.model_validate(raw["protocol"])
        commands = Commands.model_validate(raw["commands"])
        crc = protocol.crc_config
        for count, distance, direction in ((5, 1250, 1), (0, 0, 0), (-5, -1250, -1)):
            with self.subTest(direction=direction):
                payload = struct.pack("<IiiIbBB", 7, count, distance, 123, direction, 2, 3)
                self.assertEqual(len(payload), 19)
                frame = build_unified_frame(
                    protocol.message_types["telemetry"], payload, crc,
                    stx=protocol.stx, etx=protocol.etx,
                )
                self.assertEqual(len(frame), 24)
                decoded = decode_unified_frame(
                    frame, crc, {}, protocol.payload_lengths_by_code,
                    stx=protocol.stx, etx=protocol.etx,
                    message_types=protocol.message_types,
                    command_names=commands.names,
                    status_flag_names=protocol.status_flag_names,
                )
                self.assertEqual(decoded["trigger_id"], 7)
                self.assertEqual(decoded["encoder_count"], count)
                self.assertEqual(decoded["distance_mm_x1000"], distance)
                self.assertEqual(decoded["direction"], direction)
                self.assertEqual(decoded["timestamp_us"], 123)
                self.assertEqual(decoded["state"], 2)
                self.assertEqual(decoded["status_flags"], 3)
