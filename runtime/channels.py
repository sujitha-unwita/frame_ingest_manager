"""Finite queue operations used by the two child processes."""

import asyncio
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any


@dataclass
class WorkerChannels:
    """IPC handles only: connections, clients and models never cross processes."""

    triggers: Any
    audit: Any
    commands: Any
    events: Any
    abort: Any
    failed: Any
    heartbeat: Any
    config: Any
    role: str

    async def receive(self, queue) -> tuple[bool, Any]:
        """Distinguish a poll timeout from an explicit None shutdown sentinel."""
        if self.abort.is_set():
            raise RuntimeError("worker aborted")
        try:
            item = await asyncio.to_thread(
                queue.get, True, self.config.poll_interval_s,
            )
            return True, item
        except Empty:
            return False, None

    async def send(self, queue, item) -> None:
        """Bound waiting for capacity; a full downstream queue is a fault."""
        if self.abort.is_set():
            raise RuntimeError("worker aborted")
        await asyncio.to_thread(
            queue.put, item, True, self.config.queue_put_timeout_s,
        )

    async def report(self, kind: str, **values) -> None:
        """Report ordered progress to control."""
        await self.send(self.events, {"kind": kind, "worker": self.role, **values})

    async def pulse(self) -> None:
        """Prove the worker event loop is responsive while it waits on I/O."""
        while True:
            # One writer per heartbeat; no process-shared lock to strand on exit.
            self.heartbeat.value = time.monotonic()
            await asyncio.sleep(self.config.poll_interval_s)
