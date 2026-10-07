"""Data objects handed from the ingest manager to downstream consumers."""

from dataclasses import dataclass, field
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RawFrameInfo:
    """Camera-side metadata for one exposure, copied out of the SDK struct."""

    width: int
    height: int
    pixel_type: int
    pixel_format_name: str
    frame_number: int
    trigger_index: int
    device_timestamp_ticks: int
    host_timestamp_ms: int
    frame_bytes: int
    lost_packets: int


@dataclass
class EncoderEvent:
    """One MCU telemetry packet: the encoder position latched at one trigger pulse."""

    trigger_id: int                            # MCU trigger counter
    encoder_count: int                         # signed encoder position at this pulse
    host_time: float                           # time.monotonic() when received
    distance_mm_x1000: Optional[int] = None    # signed fabric distance, millimetres x1000
    timestamp_us: Optional[int] = None         # MCU clock at the pulse (uint32, wraps)
    direction: Optional[int] = None            # +1 forward, 0 stopped/unknown, -1 reverse
    state: Optional[int] = None                # MCU/machine state byte
    status_flags: Optional[int] = None         # MCU status bits

    @property
    def distance_mm(self):
        """Fabric distance in millimetres (None if the MCU did not send it)."""
        return None if self.distance_mm_x1000 is None else self.distance_mm_x1000 / 1000.0


class Frame:
    """One camera exposure held in an application-owned buffer slot.

    `image` stays valid until `release()` is called (directly or through the
    owning TriggerSet). Each frame carries its own encoder_count; frames A and B
    of the same set are separate pulses and never share a position.

    `pulse_index` is set by the camera thread from the camera's own frame
    counter. `trigger_id` and `label` are assigned by the assembler, which
    knows how pulses are grouped into triggers (see trigger.grouping).
    """

    __slots__ = (
        "camera_id",
        "pulse_index",
        "trigger_id",
        "label",
        "image",
        "info",
        "encoder",
        "arrival_time",
        "_pool",
        "_slot_index",
        "_released",
        "_lock",
    )

    def __init__(self, camera_id, pulse_index, image, info,
                 arrival_time, pool, slot_index, trigger_id=None, label=None):
        self.camera_id = camera_id
        self.pulse_index = pulse_index
        self.trigger_id = trigger_id
        self.label = label
        self.image: np.ndarray = image
        self.info: RawFrameInfo = info
        self.encoder: Optional[EncoderEvent] = None
        self.arrival_time = arrival_time
        self._pool = pool
        self._slot_index = slot_index
        self._released = False
        self._lock = threading.Lock()

    @property
    def encoder_count(self):
        return None if self.encoder is None else self.encoder.encoder_count

    @property
    def mcu_trigger_id(self):
        """trigger_id of the MCU packet matched to this frame (None if missing)."""
        return None if self.encoder is None else self.encoder.trigger_id

    @property
    def distance_mm(self):
        """Fabric distance at this frame's pulse, in millimetres."""
        return None if self.encoder is None else self.encoder.distance_mm

    @property
    def direction(self):
        """+1 forward, 0 stopped/unknown, -1 reverse, None if not sent."""
        return None if self.encoder is None else self.encoder.direction

    @property
    def slot_index(self):
        return self._slot_index

    @property
    def released(self):
        return self._released

    def release(self):
        """Return the buffer slot to the pool. Safe to call more than once."""
        with self._lock:
            if self._released:
                return
            self._released = True
            self.image = None
        self._pool.release(self._slot_index)

    def __repr__(self):
        return (
            f"Frame(camera={self.camera_id}, trigger={self.trigger_id}, "
            f"label={self.label}, pulse={self.pulse_index}, "
            f"encoder={self.encoder_count}, frame_no={self.info.frame_number})"
        )
