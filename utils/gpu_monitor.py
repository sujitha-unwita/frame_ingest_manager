"""Read device-wide Jetson GPU activity without blocking inference."""

from __future__ import annotations

import asyncio
import logging
import re

from config.models import GpuMonitoring


LOGGER = logging.getLogger(__name__)
GPU_ACTIVITY = re.compile(r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%")


class GpuMonitor:
    """Own one tegrastats process; unknown readings are never reported as zero."""

    def __init__(self, gpu_monitoring_config: GpuMonitoring) -> None:
        self._config = gpu_monitoring_config
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._measuring = False
        self._count = 0
        self._total = 0.0
        self._last_inference_average: float | None = None

    async def start(self) -> None:
        """Start once; missing Jetson tooling leaves metrics unavailable."""
        if self._task is not None:
            raise RuntimeError("GPU monitor already started")
        if not self._config.enabled:
            LOGGER.info("GPU monitoring disabled")
            return
        try:
            self._process = await asyncio.create_subprocess_exec(
                self._config.executable,
                "--interval", str(self._config.sample_interval_ms),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError:
            LOGGER.exception("Cannot start GPU monitor; utilization unavailable")
            return
        self._task = asyncio.create_task(self._read_samples(), name="gpu-monitor")
        LOGGER.info("GPU monitor started")

    async def _read_samples(self) -> None:
        """Consume output continuously with bounded memory."""
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            while line := await process.stdout.readline():
                self.record_sample(line.decode(errors="replace"))
            LOGGER.warning("GPU monitor output ended; utilization unavailable")
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("GPU monitoring failed; utilization unavailable")

    def record_sample(self, line: str) -> None:
        """Parse the GR3D percentage, ignoring output without a valid sample."""
        if not self._measuring:
            return
        match = GPU_ACTIVITY.search(line)
        if match is None:
            return
        percent = float(match.group(1))
        if not 0 <= percent <= 100:
            return
        self._count += 1
        self._total += percent

    def begin_inference(self) -> None:
        """Start collecting sample statistics immediately before inference."""
        self._measuring = True
        self._count = 0
        self._total = 0.0

    def get_inference_average_percent(self) -> float | None:
        """Return the last completed inference's sample mean on a 0..100 scale.

        This historical result stays until the next inference ends or stop().
        Idle samples do not overwrite it. No samples means unknown, not idle.
        """
        return self._last_inference_average

    def end_inference(self) -> float | None:
        """Return sampled estimates, not exact per-process CUDA utilization.

        Very short inference may produce no samples; its mean stays unknown.
        Sample periods may straddle inference boundaries.
        """
        self._measuring = False
        self._last_inference_average = (
            self._total / self._count if self._count else None
        )
        return self._last_inference_average

    async def stop(self) -> None:
        """Stop our subprocess with a deadline and release the reader task."""
        process, self._process = self._process, None
        task, self._task = self._task, None
        try:
            if process is not None:
                if process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(process.wait(), self._config.stop_timeout_s)
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self._measuring = False
            self._last_inference_average = None
        LOGGER.info("GPU monitor stopped")
