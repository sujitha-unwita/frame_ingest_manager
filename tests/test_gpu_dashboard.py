"""Dashboard contract checks; require the application's serial dependency."""

import unittest
from unittest.mock import AsyncMock, Mock

from dashboard_service.services.status_mirror import DashboardMirror
from tests.test_gpu_monitor import settings
from utils.gpu_monitor import GpuMonitor


class GpuDashboardTests(unittest.TestCase):
    def test_inference_average_percent_or_null(self) -> None:
        link = Mock()
        link.link_state.return_value = ("LINK_UP", 0.01)
        link.telemetry.trigger_age_s.return_value = None
        link.telemetry.last_trigger_id = None
        link.capture_paused = True
        link.link.published_state = "LINK_UP"
        link.missed_intervals.return_value = 0
        link.state.is_armed = True
        link.state.active_faults = []
        monitor = GpuMonitor(settings())
        mirror = DashboardMirror(link, Mock(), monitor, AsyncMock())
        self.assertIsNone(mirror.build_status()["duty_cycle"])
        monitor.begin_inference()
        monitor.record_sample("GR3D_FREQ 60%")
        monitor.record_sample("GR3D_FREQ 90%")
        self.assertIsNone(mirror.build_status()["duty_cycle"])
        monitor.end_inference()
        monitor.record_sample("GR3D_FREQ 0%")
        status = mirror.build_status()
        self.assertEqual(status["duty_cycle"], 75)
        self.assertEqual(status["queue_depth"], 0)
        monitor.begin_inference()
        monitor.end_inference()
        self.assertIsNone(mirror.build_status()["duty_cycle"])
