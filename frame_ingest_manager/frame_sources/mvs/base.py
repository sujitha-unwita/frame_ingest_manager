"""Backend-independent camera grab loop: pulse numbering, buffer copy, stats."""

from collections import deque
from dataclasses import dataclass, asdict
import logging
import threading
import time

import numpy as np

from ...services.buffer_pool import CameraBufferPool
from ...models.captured_frame import Frame
from . import pixel_formats


LOGGER = logging.getLogger(__name__)

_COUNTER_MODULO = 1 << 32          # SDK frame/trigger counters are uint32
_RECENT_PULSES = 1024              # pulses remembered per camera for duplicate detection


@dataclass
class DeviceGeometry:
    """What the backend reports after opening/configuring the device."""

    payload_bytes: int
    width: int
    height: int
    pixel_type: int
    timestamp_tick_hz: float


@dataclass
class CameraStats:
    frames_received: int = 0
    frames_delivered: int = 0
    dropped_no_slot: int = 0
    sequence_gaps: int = 0          # pulses the camera exposed but we never got
    duplicate_or_reordered: int = 0
    late_in_camera: int = 0         # older frame arrived after a newer one (still paired correctly)
    frames_with_lost_packets: int = 0
    pulse_spacing_violations: int = 0
    grab_timeouts: int = 0

    def as_dict(self):
        return asdict(self)


class GrabbedFrame:
    """A frame still owned by the backend. Copy it, then call free()."""

    def __init__(self, info):
        self.info = info

    def copy_into(self, destination):
        raise NotImplementedError

    def free(self):
        raise NotImplementedError


class CameraSource:
    """One thread per camera: wait for a triggered frame, copy it into a pool
    slot, return the SDK buffer immediately and hand the Frame to `sink`."""

    def __init__(self, camera_config, trigger_config, slot_count, sink,
                 on_pool_exhausted, on_fault):
        self.config = camera_config
        self.trigger = trigger_config
        self.camera_id = camera_config.id
        self.pool = CameraBufferPool(camera_config.id, slot_count)
        self.stats = CameraStats()
        self.geometry = None
        self.error = None
        self._sink = sink
        self._on_pool_exhausted = on_pool_exhausted
        self._on_fault = on_fault
        self._stop = threading.Event()
        self._thread = None
        self._reset_sequence()

    # ------------------------------------------------------------------ #
    # Backend hooks
    # ------------------------------------------------------------------ #
    def _open_device(self):
        """Open + configure the device; return DeviceGeometry."""
        raise NotImplementedError

    def _start_device(self):
        raise NotImplementedError

    def _stop_device(self):
        raise NotImplementedError

    def _close_device(self):
        raise NotImplementedError

    def _grab(self, timeout_ms):
        """Return a GrabbedFrame, or None when no trigger arrived in time."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def open(self, *, memory_budget_bytes=None):
        self.geometry = self._open_device()
        g = self.geometry
        gray_shape = None
        if self.config.convert_to_gray and pixel_formats.needs_conversion(g.pixel_type):
            gray_shape = (g.height, g.width)
        # Validate the format and memory budget before allocating any pool slots.
        pixel_formats.raw_shape(g.pixel_type, g.height, g.width)
        gray_bytes = g.height * g.width if gray_shape is not None else 0
        needed = (g.payload_bytes + gray_bytes) * self.pool.slot_count
        if memory_budget_bytes is not None and needed > memory_budget_bytes:
            raise RuntimeError(f"[{self.camera_id}] camera buffers exceed mvs.max_total_memory_mb")
        self.pool.allocate(g.payload_bytes, gray_shape)
        LOGGER.info(
            "[%s] ready: %dx%d %s, payload %d bytes, tick %.0f Hz",
            self.camera_id, g.width, g.height, pixel_formats.name_of(g.pixel_type),
            g.payload_bytes, g.timestamp_tick_hz,
        )

    def start(self):
        self._reset_sequence()
        self._stop.clear()
        self._start_device()
        self._thread = threading.Thread(
            target=self._run, name=f"grab-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def _interrupt(self):
        """Optional backend hook to wake a blocked _grab() early."""

    def request_stop(self):
        """Signal the grab loop without waiting (lets N cameras stop in parallel)."""
        self._stop.set()
        self._interrupt()

    def stop(self):
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=self.config.grab_timeout_ms / 1000.0 + 2.0)
            self._thread = None
        try:
            self._stop_device()
        except Exception as error:  # noqa: BLE001 - shutdown must continue
            LOGGER.warning("[%s] stop failed: %s", self.camera_id, error)

    def close(self):
        try:
            self._close_device()
        except Exception as error:  # noqa: BLE001 - shutdown must continue
            LOGGER.warning("[%s] close failed: %s", self.camera_id, error)

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    # Grab loop
    # ------------------------------------------------------------------ #
    def _run(self):
        try:
            while not self._stop.is_set():
                grabbed = self._grab(self.config.grab_timeout_ms)
                if grabbed is None:
                    # Hardware trigger: no pulse within the timeout is normal.
                    self.stats.grab_timeouts += 1
                    continue
                frame = self._ingest(grabbed)
                if frame is not None:
                    self.stats.frames_delivered += 1
                    self._sink(frame)
        except Exception as error:  # noqa: BLE001 - reported to the manager
            if not self._stop.is_set():
                self.error = error
                LOGGER.exception("[%s] grab loop failed", self.camera_id)
                self._on_fault(self, error)

    def _ingest(self, grabbed):
        slot = None
        try:
            info = grabbed.info
            self.stats.frames_received += 1
            if info.lost_packets:
                self.stats.frames_with_lost_packets += 1

            pulse = self._pulse_index(info)
            if pulse is None:
                return None
            self._check_spacing(pulse, info)

            slot = self.pool.acquire()
            if slot is None:
                slot = self._on_pool_exhausted(self)
            if slot is None:
                self.stats.dropped_no_slot += 1
                LOGGER.warning("[%s] no free buffer slot; dropped pulse %d", self.camera_id, pulse)
                return None

            image = self._copy_to_slot(grabbed, slot)
            # trigger_id and label are assigned by the assembler.
            frame = Frame(
                camera_id=self.camera_id,
                pulse_index=pulse,
                image=image,
                info=info,
                arrival_time=time.monotonic(),
                pool=self.pool,
                slot_index=slot,
            )
            slot = None       # ownership moved into the Frame
            return frame
        finally:
            if slot is not None:
                self.pool.release(slot)
            grabbed.free()

    def _copy_to_slot(self, grabbed, slot):
        info = grabbed.info
        g = self.geometry
        if (info.width, info.height, info.pixel_type) != (g.width, g.height, g.pixel_type):
            raise RuntimeError(
                f"[{self.camera_id}] frame format changed to {info.width}x{info.height} "
                f"{pixel_formats.name_of(info.pixel_type)} while running; restart required"
            )
        if info.frame_bytes > self.pool.raw_capacity:
            raise RuntimeError(
                f"[{self.camera_id}] frame of {info.frame_bytes} bytes exceeds "
                f"PayloadSize {self.pool.raw_capacity}"
            )
        shape = pixel_formats.raw_shape(info.pixel_type, info.height, info.width)
        expected = int(np.prod(shape))
        if info.frame_bytes < expected:
            raise RuntimeError(
                f"[{self.camera_id}] frame has {info.frame_bytes} bytes, {shape} needs {expected}"
            )

        raw_buffer = self.pool.raw(slot)
        grabbed.copy_into(raw_buffer)
        raw_image = raw_buffer[:expected].reshape(shape)
        gray_buffer = self.pool.gray(slot)
        if gray_buffer is not None:
            return pixel_formats.to_gray(raw_image, info.pixel_type, gray_buffer)
        return raw_image

    # ------------------------------------------------------------------ #
    # Pulse numbering
    # ------------------------------------------------------------------ #
    def _reset_sequence(self):
        self._base = None
        self._newest_raw = None
        self._newest_pulse = None
        self._recent_pulses = set()
        self._recent_order = deque()
        self._last_timestamp = None
        self._last_timestamp_pulse = None

    def _pulse_index(self, info):
        """Pulse number from the camera's own counter - never from arrival order.

        The counter value identifies the pulse even if frames arrive late or
        out of order (e.g. a congested LACP link), so a late frame still lands
        in the correct trigger set instead of being discarded or mis-paired.
        """
        raw = info.frame_number if self.trigger.sequence_source == "frame_number" else info.trigger_index
        if self._newest_raw is None:
            base = raw if self.trigger.sequence_base is None else self.trigger.sequence_base
            pulse = (raw - base) % _COUNTER_MODULO
            if pulse >= _COUNTER_MODULO // 2:
                LOGGER.warning(
                    "[%s] first sequence value %d is below sequence_base %d; frame ignored",
                    self.camera_id, raw, base,
                )
                return None
            self._base = base
            if pulse:
                self.stats.sequence_gaps += pulse
                LOGGER.warning("[%s] first frame is pulse %d; %d frame(s) missed at start",
                               self.camera_id, pulse, pulse)
            self._newest_raw, self._newest_pulse = raw, pulse
        else:
            # Signed distance from the newest frame seen so far (uint32 counter).
            delta = (raw - self._newest_raw) % _COUNTER_MODULO
            if delta >= _COUNTER_MODULO // 2:
                delta -= _COUNTER_MODULO
            pulse = self._newest_pulse + delta
            if delta > 0:
                if delta > 1:
                    self.stats.sequence_gaps += delta - 1
                    LOGGER.warning("[%s] %d frame(s) not yet received before sequence %d",
                                   self.camera_id, delta - 1, raw)
                self._newest_raw, self._newest_pulse = raw, pulse
            elif pulse < 0 or pulse in self._recent_pulses:
                self.stats.duplicate_or_reordered += 1
                LOGGER.warning("[%s] duplicate sequence %d (pulse %d); frame ignored",
                               self.camera_id, raw, pulse)
                return None
            else:
                # An older frame arriving late: fill the gap it left.
                self.stats.late_in_camera += 1
                self.stats.sequence_gaps = max(0, self.stats.sequence_gaps - 1)
                LOGGER.info("[%s] late frame: sequence %d (pulse %d) arrived after pulse %d",
                            self.camera_id, raw, pulse, self._newest_pulse)
        self._remember(pulse)
        return pulse

    def _remember(self, pulse):
        self._recent_pulses.add(pulse)
        self._recent_order.append(pulse)
        while len(self._recent_order) > _RECENT_PULSES:
            self._recent_pulses.discard(self._recent_order.popleft())

    def _check_spacing(self, pulse, info):
        """Consecutive pulses (e.g. frame A -> B) must be >= min_pulse_interval_ms apart."""
        ticks = info.device_timestamp_ticks
        limit_ms = self.trigger.min_pulse_interval_ms
        if ticks and self._last_timestamp is not None and limit_ms > 0:
            pulses = pulse - self._last_timestamp_pulse
            elapsed_ms = (ticks - self._last_timestamp) * 1000.0 / self.geometry.timestamp_tick_hz
            if pulses > 0 and elapsed_ms / pulses < limit_ms:
                self.stats.pulse_spacing_violations += 1
                LOGGER.warning(
                    "[%s] pulse %d came %.2f ms after pulse %d (< %.2f ms minimum)",
                    self.camera_id, pulse, elapsed_ms, self._last_timestamp_pulse, limit_ms,
                )
        if ticks and (self._last_timestamp_pulse is None or pulse > self._last_timestamp_pulse):
            self._last_timestamp = ticks
            self._last_timestamp_pulse = pulse
