"""Preallocated, application-owned frame buffers (one pool per camera)."""

import logging
import threading

import numpy as np


LOGGER = logging.getLogger(__name__)


class CameraBufferPool:
    """Fixed number of reusable slots, allocated once when the camera opens.

    Sized from the camera's PayloadSize, so no allocation happens on the hot
    path. Each slot has a raw buffer (the SDK frame is memmoved into it) and,
    if colour conversion is enabled, a same-size-as-image mono buffer.
    Both are zero-initialized at allocation, before any frames are received.
    """

    def __init__(self, camera_id, slot_count):
        if slot_count < 1:
            raise ValueError("slot_count must be >= 1")
        self.camera_id = camera_id
        self.slot_count = slot_count
        self.raw_capacity = 0
        self._raw = []
        self._gray = []
        self._free = []
        self._in_use = set()
        self._lock = threading.Lock()

    @property
    def allocated(self):
        return bool(self._raw)

    def allocate(self, raw_capacity, gray_shape=None):
        with self._lock:
            if self._raw:
                raise RuntimeError(f"[{self.camera_id}] buffer pool already allocated")
            self.raw_capacity = int(raw_capacity)
            self._raw = [np.zeros(self.raw_capacity, dtype=np.uint8) for _ in range(self.slot_count)]
            if gray_shape is not None:
                self._gray = [np.zeros(gray_shape, dtype=np.uint8) for _ in range(self.slot_count)]
            self._free = list(range(self.slot_count))
            self._in_use.clear()
        LOGGER.info(
            "[%s] allocated and zero-initialized %d slots x %.2f MB (%.2f MB total)",
            self.camera_id, self.slot_count, self.slot_bytes / 1e6, self.total_bytes / 1e6,
        )

    @property
    def slot_bytes(self):
        gray_bytes = self._gray[0].nbytes if self._gray else 0
        return self.raw_capacity + gray_bytes

    @property
    def total_bytes(self):
        return self.slot_bytes * self.slot_count

    def free_count(self):
        with self._lock:
            return len(self._free)

    def acquire(self):
        """Return a free slot index, or None when every slot is held."""
        with self._lock:
            if not self._free:
                return None
            index = self._free.pop(0)
            self._in_use.add(index)
            return index

    def release(self, index):
        with self._lock:
            if index not in self._in_use:
                LOGGER.warning("[%s] release of slot %s that is not in use", self.camera_id, index)
                return
            self._in_use.discard(index)
            self._free.append(index)

    def raw(self, index):
        return self._raw[index]

    def gray(self, index):
        return self._gray[index] if self._gray else None
