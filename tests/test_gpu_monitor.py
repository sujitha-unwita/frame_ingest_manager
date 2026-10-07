"""GPU metrics tested offline: no Jetson, subprocess, or AWS access required."""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError

from config.models import GpuMonitoring
from utils.gpu_monitor import GpuMonitor


def settings(**changes: object) -> GpuMonitoring:
    """Build explicit test settings independent of deployment YAML."""
    values = dict(enabled=True, executable="tegrastats", sample_interval_ms=100,
                  stop_timeout_s=2.0)
    values.update(changes)
    return GpuMonitoring(**values)


class GpuMonitorTests(unittest.TestCase):
    def test_last_inference_average_not_latest_sample(self) -> None:
        monitor = GpuMonitor(settings())
        self.assertIsNone(monitor.get_inference_average_percent())
        monitor.begin_inference()
        monitor.record_sample("GR3D_FREQ 60%")
        monitor.record_sample("GR3D_FREQ 90%")
        self.assertIsNone(monitor.get_inference_average_percent())
        monitor.end_inference()
        monitor.record_sample("GR3D_FREQ 0%")
        self.assertEqual(monitor.get_inference_average_percent(), 75)
        monitor.begin_inference()
        self.assertEqual(monitor.get_inference_average_percent(), 75)
        monitor.end_inference()
        self.assertIsNone(monitor.get_inference_average_percent())

    def test_zero_inference_average_is_valid(self) -> None:
        monitor = GpuMonitor(settings())
        monitor.begin_inference()
        monitor.record_sample("GR3D_FREQ 0%")
        monitor.end_inference()
        self.assertEqual(monitor.get_inference_average_percent(), 0)

    def test_parses_jetson_output_formats(self) -> None:
        monitor = GpuMonitor(settings())
        monitor.begin_inference()
        monitor.record_sample("RAM 2962/7607MB GR3D_FREQ 0% cpu@54C")
        self.assertEqual(monitor.end_inference(), 0)
        monitor.begin_inference()
        monitor.record_sample("GR3D_FREQ 85%@[918,918]")
        self.assertEqual(monitor.end_inference(), 85)

    def test_invalid_samples_are_ignored(self) -> None:
        monitor = GpuMonitor(settings())
        monitor.begin_inference()
        monitor.record_sample("GR3D_FREQ 40%")
        monitor.record_sample("GR3D_FREQ 101%")
        monitor.record_sample("permission denied")
        self.assertEqual(monitor.end_inference(), 40)

    def test_inference_window_excludes_prior_samples(self) -> None:
        monitor = GpuMonitor(settings())
        monitor.record_sample("GR3D_FREQ 100%")
        monitor.begin_inference()
        monitor.record_sample("GR3D_FREQ 20%")
        monitor.record_sample("GR3D_FREQ 80%")
        self.assertEqual(monitor.end_inference(), 50)
        monitor.begin_inference()
        self.assertIsNone(monitor.end_inference())

    def test_settings_validation(self) -> None:
        for changes in ({"sample_interval_ms": 0},
                        {"stop_timeout_s": 0}, {"executable": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                settings(**changes)


class GpuLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_does_not_spawn(self) -> None:
        monitor = GpuMonitor(settings(enabled=False))
        with patch("utils.gpu_monitor.asyncio.create_subprocess_exec") as spawn:
            await monitor.start()
            spawn.assert_not_called()
        await monitor.stop()

    async def test_missing_command_is_unavailable(self) -> None:
        monitor = GpuMonitor(settings())
        with patch("utils.gpu_monitor.asyncio.create_subprocess_exec",
                   side_effect=FileNotFoundError), self.assertLogs(level="ERROR"):
            await monitor.start()
        self.assertIsNone(monitor.get_inference_average_percent())
        await monitor.stop()

    async def test_start_read_and_stop(self) -> None:
        stream = asyncio.StreamReader()
        process = Mock(stdout=stream, returncode=None, wait=AsyncMock(return_value=0))
        monitor = GpuMonitor(settings())
        with patch("utils.gpu_monitor.asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=process)) as spawn:
            await monitor.start()
            self.assertEqual(spawn.call_args.args, ("tegrastats", "--interval", "100"))
        monitor.begin_inference()
        stream.feed_data(b"GR3D_FREQ 60%\n")
        await asyncio.sleep(0)
        self.assertEqual(monitor.end_inference(), 60)
        await monitor.stop()
        process.terminate.assert_called_once()
        process.wait.assert_awaited_once()
        self.assertIsNone(monitor.get_inference_average_percent())

    async def test_process_exit_leaves_empty_window_unknown(self) -> None:
        stream = asyncio.StreamReader()
        process = Mock(stdout=stream, returncode=1, wait=AsyncMock(return_value=1))
        monitor = GpuMonitor(settings())
        with patch("utils.gpu_monitor.asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=process)):
            await monitor.start()
        monitor.begin_inference()
        stream.feed_eof()
        with self.assertLogs(level="WARNING"):
            await asyncio.sleep(0)
        self.assertIsNone(monitor.end_inference())
        await monitor.stop()

    async def test_shutdown_timeout_kills_process(self) -> None:
        process = Mock(returncode=None)
        process.wait = AsyncMock(side_effect=[asyncio.TimeoutError, 0])
        monitor = GpuMonitor(settings())
        monitor._process = process
        await monitor.stop()
        process.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
