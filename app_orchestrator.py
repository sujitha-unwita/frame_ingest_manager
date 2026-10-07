"""Composition root: responsive MCU/dashboard control and isolated workers."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI

import constants
from config.models import AppConfig
from control_service.mcu_command import CommandResult
from control_service.mcu_link import McuLink, build_transport
from dashboard_service.app import create_app
from dashboard_service.routers.websocket import WebSocketHub
from dashboard_service.services.dashboard_coordinator import (
    DashboardCoordinator, OPERATOR_COMMANDS,
)
from dashboard_service.services.status_mirror import DashboardMirror
from frame_ingest_manager.models.trigger_telemetry import TriggerTelemetry
from runtime.process_runtime import ProcessRuntime
from utils.timing import Timing

LOGGER = logging.getLogger(__name__)


class AppOrchestrator:
    """Own control in this process; vision and audit own their resources in children."""

    def __init__(self, app_config: AppConfig) -> None:
        """Create the dependency graph without starting devices or processes."""
        self._app_config = app_config
        root = app_config.source_paths[0].parent if app_config.source_paths else Path.cwd()
        self._timing = Timing.from_config(app_config.logging, root, "control")
        self._running = False
        self._pause_generation = 0
        self._mcu_task = None
        self._mirror_task = None
        self._panel_task = None
        self._hub = WebSocketHub()
        # Local delivery queue, not an IPC queue: slow panels cannot stall serial.
        self._panel_queue = asyncio.Queue(
            maxsize=app_config.processing.panel_queue_capacity,
        )
        self._runtime = ProcessRuntime(app_config, self._publish)
        self._mcu_link = McuLink(
            config=app_config, transport=build_transport(app_config),
            publish=self._publish, on_trigger=self.on_trigger,
        )
        self._mirror = DashboardMirror(
            link=self._mcu_link, dashboard_config=app_config.dashboard,
            gpu_monitor=self._runtime, publish=self._publish,
        )
        self._dashboard = DashboardCoordinator(
            config=app_config, publish_to_panels=self._enqueue_panel,
            link=self._mcu_link, mirror=self._mirror,
            on_operator_command=self.handle_operator_command,
        )
        self._app = create_app(
            app_config, self._dashboard, self._hub, self._lifespan,
        )

    def run(self) -> None:
        """Serve the app; Uvicorn owns the control process's asyncio loop."""
        uvicorn.run(
            self._app, host=self._app_config.dashboard.host,
            port=self._app_config.dashboard.port, log_config=None,
        )

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Pair application startup with bounded worker draining and shutdown."""
        app.state.app_orchestrator = self
        try:
            await self.start()
            yield
        finally:
            await self.stop()
            del app.state.app_orchestrator

    @property
    def mcu_link(self) -> McuLink:
        """The only link allowed to open the MCU serial port."""
        return self._mcu_link

    async def _publish(self, message: Mapping[str, Any]) -> None:
        """Record events and enqueue panel delivery without network waiting."""
        await self._dashboard.publish(message)

    async def _enqueue_panel(self, message: Mapping[str, Any]) -> None:
        """Keep websocket I/O outside heartbeat and ACK dispatch."""
        try:
            self._panel_queue.put_nowait((time.monotonic(), dict(message)))
        except asyncio.QueueFull:
            # Durable audit is unaffected; dashboard events remain replayable
            # within the existing event store's configured retention.
            LOGGER.error("Dashboard delivery queue full; use event replay")

    async def _send_panels(self) -> None:
        """Bound slow panel writes; never block the MCU reader on a socket."""
        while True:
            enqueued_at, message = await self._panel_queue.get()
            # Both timing rows describe the same outgoing message. Reuse its
            # real replay identity; status snapshots deliberately have no ID.
            timing_context = {
                "trigger_id": message.get("trigger_id"),
                "camera_id": message.get("camera_id"),
                "event_type": message.get("event"),
                "event_id": message.get("event_id"),
                "stream_id": message.get("stream_id"),
                "command_name": message.get("command"),
            }
            self._timing.record(
                "panel_queue_wait", enqueued_at, **timing_context,
            )
            try:
                with self._timing.measure(
                    "websocket_broadcast", **timing_context,
                ):
                    await asyncio.wait_for(
                        self._hub.broadcast(message),
                        self._app_config.processing.panel_send_timeout_s,
                    )
            except Exception:
                LOGGER.exception("Dashboard delivery failed; reconnect and replay")
            finally:
                self._panel_queue.task_done()

    async def handle_operator_command(self, command: str) -> CommandResult:
        """Prepare a roll asynchronously; Stop never waits on roll preparation."""
        try:
            if not self._running:
                raise RuntimeError("orchestrator is not running")
            if command == "operator_stop_request":
                self._pause_generation += 1
            else:
                generation = self._pause_generation
                if command == "operator_start_capture":
                    await self._runtime.prepare_roll()
                elif command == "operator_resume_request":
                    if not self._runtime.roll_id:
                        raise RuntimeError("use Start Capture to create a roll first")
                self._runtime.require_workers()
                if generation != self._pause_generation:
                    raise RuntimeError("capture cancelled by an operator pause")
            with self._timing.measure(
                "operator_command", event_type=command,
                command_name=OPERATOR_COMMANDS[command],
            ):
                return await self._mcu_link.send_command(
                    OPERATOR_COMMANDS[command], source=constants.SOURCE_OPERATOR,
                )
        except Exception as exc:
            LOGGER.exception("Capture command failed")
            raise RuntimeError("Capture command failed; check application logs") from exc

    async def start(self) -> None:
        """Require worker readiness before starting control's capture workflow."""
        if self._running:
            raise RuntimeError("orchestrator is already running")
        self._panel_task = asyncio.create_task(self._send_panels())
        try:
            with self._timing.measure("workers_startup"):
                await self._runtime.start()
            self._running = True
            self._mcu_task = asyncio.create_task(self._mcu_link.run(), name="mcu-link")
            self._mirror_task = asyncio.create_task(
                self._mirror.run(), name="dashboard-mirror",
            )
            self._mcu_task.add_done_callback(self._mcu_finished)
        except BaseException:
            LOGGER.exception("Application startup failed")
            await self.stop()
            raise

    def _mcu_finished(self, task) -> None:
        """An unexpectedly stopped reader must not leave capture marked healthy."""
        if self._running and not task.cancelled():
            LOGGER.error("MCU reader stopped; check serial connection")

    async def on_trigger(self, message: Mapping[str, Any]) -> None:
        """Validate metadata and enqueue immediately; no inference or audit await."""
        try:
            if not self._running:
                raise RuntimeError("orchestrator is not running")
            trigger = TriggerTelemetry.model_validate({
                "trigger_id": message.get("trigger_id"),
                "encoder_count": message.get("encoder_count"),
                "timestamp_us": message.get("timestamp_us"),
            })
            with self._timing.measure("trigger_enqueue", trigger_id=trigger.trigger_id):
                self._runtime.submit(trigger.model_dump())
        except Exception:
            LOGGER.exception("Trigger rejected: id=%s", message.get("trigger_id"))
            raise

    async def stop(self) -> None:
        """Drain admitted jobs and close resources without changing MCU state."""
        self._pause_generation += 1
        self._running = False
        self._runtime.accepting = False
        try:
            with self._timing.measure("workers_shutdown"):
                await self._runtime.stop()
        finally:
            tasks = [
                self._mirror_task, self._mcu_task, self._panel_task,
            ]
            for task in tasks:
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(task for task in tasks if task is not None), return_exceptions=True,
            )
            self._mirror_task = self._mcu_task = self._panel_task = None
            LOGGER.info("Final MCU counters: %s", self._mcu_link.counters_line())


__all__ = ["AppOrchestrator"]
