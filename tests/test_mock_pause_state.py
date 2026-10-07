"""Mock state regression tests against the observed real-MCU Pause response."""

import unittest
from types import SimpleNamespace

from pathlib import Path

import yaml

from config.models import Commands
from simulator.mock_mcu import MockMcu


class MockPauseStateTests(unittest.TestCase):
    def setUp(self):
        with (Path(__file__).resolve().parents[1] / "config.yaml").open(
            encoding="utf-8",
        ) as stream:
            raw = yaml.safe_load(stream)
        self.mcu = MockMcu.__new__(MockMcu)
        self.mcu.command_codes = Commands.model_validate(raw["commands"]).codes
        self.mcu.states = SimpleNamespace(
            armed=1, running=2, paused=3, fault=4, idle=0, names={},
        )

    def test_pause_while_armed_is_rejected_without_state_change(self):
        self.mcu.state = self.mcu.states.armed
        result = self.mcu._apply_command(self.mcu.command_codes["pause"])
        self.assertEqual(result, 3)
        self.assertEqual(self.mcu.state, self.mcu.states.armed)

    def test_operator_pause_while_running_still_works(self):
        self.mcu.state = self.mcu.states.running
        result = self.mcu._apply_command(self.mcu.command_codes["pause"])
        self.assertEqual(result, 0)
        self.assertEqual(self.mcu.state, self.mcu.states.paused)

    def test_resume_from_armed_still_starts_capture(self):
        self.mcu.state = self.mcu.states.armed
        result = self.mcu._apply_command(self.mcu.command_codes["resume"])
        self.assertEqual(result, 0)
        self.assertEqual(self.mcu.state, self.mcu.states.running)
