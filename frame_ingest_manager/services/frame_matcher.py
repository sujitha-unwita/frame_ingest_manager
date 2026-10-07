"""Camera leases matched by anchored counters, never arrival order."""
import logging
import threading
import time
LOGGER = logging.getLogger(__name__)

class FrameMatchError(RuntimeError):
    pass

class FrameMatcher:
    def __init__(self, camera_ids, trigger_id_base, timeout_s):
        self.camera_ids = tuple(camera_ids)
        self.trigger_id_base = trigger_id_base
        self.timeout_s = timeout_s
        self._condition = threading.Condition()
        self._pending = {}
        self._last_trigger = -1
        self._fault = None
        self._closed = False

    def fail(self, error):
        with self._condition:
            if self._fault is None:
                self._fault = error
            self._condition.notify_all()

    def check(self):
        with self._condition:
            if self._fault is not None:
                raise FrameMatchError(str(self._fault)) from self._fault
            if self._closed:
                raise FrameMatchError("camera source is closed")

    def add(self, frame):
        with self._condition:
            trigger_id = self.trigger_id_base + frame.pulse_index
            if self._closed or self._fault or trigger_id <= self._last_trigger:
                frame.release()
                return
            if frame.camera_id not in self.camera_ids or frame.info.lost_packets or trigger_id >= 2**32:
                frame.release()
                self.fail(FrameMatchError("invalid/incomplete camera frame or counter wrap; re-establish alignment"))
                return
            row = self._pending.setdefault(trigger_id, {})
            if frame.camera_id in row:
                frame.release()
                return
            row[frame.camera_id] = frame
            self._condition.notify_all()

    def take(self, trigger):
        deadline = time.monotonic() + self.timeout_s
        with self._condition:
            self.check()
            if trigger.trigger_id < self.trigger_id_base or trigger.trigger_id <= self._last_trigger:
                raise FrameMatchError("MCU trigger counter is before alignment, repeated, or reset; restart with verified anchors")
            for old in [key for key in self._pending if key < trigger.trigger_id]:
                for frame in self._pending.pop(old).values():
                    frame.release()
                LOGGER.warning("Released camera row with no admitted telemetry: trigger=%s", old)
            while True:
                self.check()
                row = self._pending.get(trigger.trigger_id, {})
                if all(camera_id in row for camera_id in self.camera_ids):
                    self._last_trigger = trigger.trigger_id
                    return self._pending.pop(trigger.trigger_id)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [camera_id for camera_id in self.camera_ids if camera_id not in row]
                    for frame in self._pending.pop(trigger.trigger_id, {}).values():
                        frame.release()
                    self._last_trigger = trigger.trigger_id
                    raise FrameMatchError(f"trigger {trigger.trigger_id}: timed out waiting for cameras {missing}")
                self._condition.wait(remaining)

    def close(self):
        with self._condition:
            self._closed = True
            for row in self._pending.values():
                for frame in row.values():
                    frame.release()
            self._pending.clear()
            self._condition.notify_all()
