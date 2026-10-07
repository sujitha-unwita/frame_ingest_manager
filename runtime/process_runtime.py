"""Control-side process supervisor; no model, database or S3 imports."""

import asyncio
import logging
import multiprocessing
import time
from pathlib import Path
from queue import Empty, Full
from types import SimpleNamespace
from uuid import uuid4

from runtime.channels import WorkerChannels
from runtime.worker_entry import run_worker

LOGGER = logging.getLogger(__name__)


class ProcessRuntime:
    """Own four bounded IPC queues, worker lifecycle and admission counters."""

    def __init__(self, app_config, publish, worker_target=run_worker):
        """Retain configuration only; start() creates OS resources."""
        self.app_config = app_config
        self.config = app_config.processing
        self.publish = publish
        self.worker_target = worker_target
        self.context = multiprocessing.get_context("spawn")
        self.processes = {}
        self.channels = {}
        self.queues = {}
        self.tasks = []
        self.states = {"vision": "stopped", "audit": "stopped"}
        self.busy_since = {}
        self._reported_delays = set()
        self.ready = asyncio.Event()
        self.roll_lock = asyncio.Lock()
        self.requests = {}
        self.roll_id = None
        self.accepting = False
        self.closing = False
        self.accepted = 0
        self.completed = 0
        self.gpu_average = None
        self.started_once = False

    def get_inference_average_percent(self):
        """Return the latest measurement reported by the vision worker."""
        return self.gpu_average

    @property
    def queue_depth(self) -> int:
        """Accepted but not audit-completed triggers, including in-flight work."""
        return self.accepted - self.completed

    def _worker_config(self, role):
        """Give vision no database credentials; preserve audit's private password."""
        app = self.app_config
        common = dict(
            logging=app.logging,
            log_root=str(app.source_paths[0].parent if app.source_paths else Path.cwd()),
            frame_ingest=app.frame_ingest, encoder=app.encoder,
        )
        if role == "vision":
            return SimpleNamespace(
                **common, inference=app.inference, gpu_monitoring=app.gpu_monitoring,
                frame_encoding=(app.audit_write.storage.keys[app.audit_write.frame_key_kind].ext
                                if app.frame_ingest.source_kind == "mvs" else None),
            )
        return SimpleNamespace(
            **common, audit_write=app.audit_write, layout=app.inference.layout,
            deduplication=app.inference.deduplication, grading=app.inference.grading,
        )

    async def start(self) -> None:
        """Spawn children and require explicit resource readiness from both."""
        if self.started_once:
            raise RuntimeError("create a new runtime after shutdown or failure")
        if self.app_config.frame_ingest.source_kind == "mvs":
            if self.app_config.transport.is_mock:
                raise RuntimeError("MVS cameras require real MCU telemetry; run serve --real")
            self.app_config.frame_ingest.mvs.require_alignment()
            media = self.app_config.audit_write.storage.keys[self.app_config.audit_write.frame_key_kind]
            types = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
            if media.ext not in types or media.content_type != types[media.ext]:
                raise RuntimeError("MVS audit format must be jpg/jpeg or png with matching content_type")
        if not self.app_config.audit_write.enabled:
            raise RuntimeError("process runtime requires audit_write.enabled")
        self.started_once = True
        self.abort = self.context.Event()
        capacities = {
            "triggers": self.config.trigger_queue_capacity,
            "audit": self.config.audit_queue_capacity,
            "commands": self.config.command_queue_capacity,
            "events": self.config.event_queue_capacity,
        }
        self.queues = {
            name: self.context.Queue(maxsize=size) for name, size in capacities.items()
        }
        try:
            for role in self.states:
                channels = WorkerChannels(
                    **self.queues, abort=self.abort, failed=self.context.Event(),
                    heartbeat=self.context.Value("d", time.monotonic(), lock=False),
                    config=self.config, role=role,
                )
                self.channels[role] = channels
                process = self.context.Process(
                    name=f"visionbar-{role}", target=self.worker_target,
                    args=(role, self._worker_config(role), channels),
                )
                process.start()
                self.processes[role] = process
                self.states[role] = "starting"
            self.tasks = [
                asyncio.create_task(self._events(), name="worker-events"),
                asyncio.create_task(self._supervise(), name="worker-supervision"),
            ]
            await asyncio.wait_for(self.ready.wait(), self.config.startup_timeout_s)
            self.require_workers()
            self.accepting = True
            LOGGER.info("Vision and audit workers are ready")
        except BaseException:
            await self.stop()
            raise

    def require_workers(self) -> None:
        """Reject work only when its required worker or event reader is unavailable."""
        for role, process in self.processes.items():
            if process.exitcode is not None or self.channels[role].failed.is_set():
                raise RuntimeError(f"{role} worker is unavailable; check its log")
        if not self.processes or any(
            state != "running" for state in self.states.values()
        ):
            raise RuntimeError("workers are not ready")
        if self.tasks and self.tasks[0].done():
            raise RuntimeError("worker event reader is unavailable; check its log")

    def _fail_requests(self, reason: str) -> None:
        """Return an actual worker failure to callers waiting for a response."""
        self.ready.set()
        for future in self.requests.values():
            if not future.done():
                future.set_exception(RuntimeError(reason))

    def submit(self, trigger: dict) -> None:
        """Nonblocking admission; serial reception never waits for worker capacity."""
        self.require_workers()
        if not self.accepting or not self.roll_id:
            raise RuntimeError("Start Capture must prepare a roll first")
        sequence = self.accepted + 1
        job = {
            "sequence": sequence, "roll_id": self.roll_id, "trigger": trigger,
            "admitted_at": time.monotonic(),
        }
        try:
            self.queues["triggers"].put_nowait(job)
        except Full as exc:
            reason = f"trigger queue full; rejected trigger={trigger['trigger_id']}"
            LOGGER.error(reason)
            raise RuntimeError(reason) from exc
        self.accepted = sequence
        LOGGER.info("Enqueued trigger=%s sequence=%s", trigger["trigger_id"], sequence)

    async def prepare_roll(self) -> str:
        """Wait for audit's durable roll ID without locking MCU command handling."""
        async with self.roll_lock:
            self.require_workers()
            if not self.accepting:
                raise RuntimeError("runtime is not accepting requests")
            if self.roll_id:
                return self.roll_id
            request_id = str(uuid4())
            future = asyncio.get_running_loop().create_future()
            self.requests[request_id] = future
            try:
                self.queues["commands"].put_nowait({"request_id": request_id})
                result = await asyncio.wait_for(future, self.config.roll_timeout_s)
                self.roll_id = result
                return result
            except (Full, asyncio.TimeoutError) as exc:
                raise RuntimeError("roll preparation queue full or request timed out") from exc
            finally:
                self.requests.pop(request_id, None)

    async def _events(self) -> None:
        """Bridge blocking IPC reads outside the control event loop."""
        try:
            while True:
                try:
                    event = await asyncio.to_thread(
                        self.queues["events"].get, True, self.config.poll_interval_s,
                    )
                except Empty:
                    continue
                role, kind = event["worker"], event["kind"]
                if kind == "ready":
                    self.states[role] = "running"
                    if all(state == "running" for state in self.states.values()):
                        self.ready.set()
                elif kind == "started":
                    self.busy_since[role] = time.monotonic()
                elif kind == "completed":
                    self.busy_since.pop(role, None)
                    if role == "audit":
                        self.completed = event["sequence"]
                elif kind == "roll_ready":
                    self.busy_since.pop(role, None)
                    future = self.requests.get(event["request_id"])
                    if future is not None and not future.done():
                        future.set_result(event["roll_id"])
                elif kind == "gpu":
                    self.gpu_average = event["average"]
                elif kind == "dashboard":
                    await self.publish(event["event"])
                elif kind == "stopped":
                    self.states[role] = "stopped"
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Worker event bridge failed")
            self._fail_requests("worker event bridge failed")

    async def _supervise(self) -> None:
        """Detect crashes, blocked event loops and jobs exceeding their deadline."""
        while True:
            await asyncio.sleep(self.config.poll_interval_s)
            now = time.monotonic()
            for role, process in self.processes.items():
                channels = self.channels[role]
                if channels.failed.is_set() or (
                    process.exitcode is not None
                    and (process.exitcode != 0 or not self.closing)
                ):
                    if self.states[role] != "failed":
                        self.states[role] = "failed"
                        reason = f"{role} worker exited or reported failure"
                        LOGGER.error(reason)
                        self._fail_requests(reason)
                if self.states[role] != "running":
                    continue
                age = now - channels.heartbeat.value
                self._report_delay(
                    role, "heartbeat", age > self.config.health_timeout_s,
                )
                started = self.busy_since.get(role)
                deadline = getattr(self.config, f"{role}_job_timeout_s")
                self._report_delay(
                    role, "job", started is not None and now - started > deadline,
                )

    def _report_delay(self, role: str, kind: str, overdue: bool) -> None:
        """Log a delay once until it clears; do not change capture or worker state."""
        key = (role, kind)
        if overdue and key not in self._reported_delays:
            LOGGER.warning("%s worker %s exceeded its configured timing limit", role, kind)
            self._reported_delays.add(key)
        elif not overdue:
            self._reported_delays.discard(key)

    async def stop(self) -> None:
        """Drain accepted jobs, then bound joins; report any incomplete work."""
        if not self.processes:
            return
        self.accepting = False
        self.closing = True
        try:
            await asyncio.to_thread(
                self.queues["triggers"].put, None, True,
                self.config.queue_put_timeout_s,
            )
            deadline = time.monotonic() + self.config.shutdown_timeout_s
            while any(process.is_alive() for process in self.processes.values()):
                if any(channel.failed.is_set() for channel in self.channels.values()):
                    break
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(self.config.poll_interval_s)
            # The audit producer reports completion before stopped on one FIFO.
            while self.completed < self.accepted and time.monotonic() < deadline:
                if any(channel.failed.is_set() for channel in self.channels.values()):
                    break
                await asyncio.sleep(self.config.poll_interval_s)
        except Full:
            LOGGER.error("Shutdown could not enqueue drain marker")
        finally:
            self.abort.set()
            for role, process in self.processes.items():
                if process.is_alive():
                    LOGGER.error("Terminating %s; drain did not finish", role)
                    process.terminate()
                await asyncio.to_thread(process.join, self.config.terminate_timeout_s)
                if process.is_alive():
                    process.kill()
                    await asyncio.to_thread(process.join, self.config.terminate_timeout_s)
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            if self.queue_depth:
                LOGGER.error(
                    "Shutdown incomplete: accepted=%s audited=%s. Inspect worker "
                    "logs/database before retry; queues are not durable storage.",
                    self.accepted, self.completed,
                )
            for queue in self.queues.values():
                queue.cancel_join_thread()
                queue.close()
            self.processes.clear()
