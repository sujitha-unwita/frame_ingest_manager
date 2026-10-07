"""Discover and arm live cameras; lease complete rows matched to MCU telemetry."""
import logging
from types import SimpleNamespace
from frame_ingest_manager.models.camera_frame import CameraFrame
from frame_ingest_manager.services.frame_matcher import FrameMatcher, FrameMatchError
LOGGER = logging.getLogger(__name__)

class MvsFrameSource:
    def __init__(self, config, *, sdk=None, camera_factory=None):
        self.config = config
        self.settings = config.mvs
        self._sdk = sdk
        self._camera_factory = camera_factory
        self.cameras = []
        self._matcher = None
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("camera source already started")
        self.settings.require_alignment()
        from .mvs.mvs_sdk import MvsSdk
        from .mvs.mvs_camera import MvsCamera
        self._sdk = self._sdk or MvsSdk.get(self.settings.sdk)
        factory = self._camera_factory or MvsCamera
        self._matcher = FrameMatcher(self.config.camera_ids, self.settings.trigger_id_base,
                                     self.settings.bundle_timeout_s)
        try:
            self._sdk.initialize()
            descriptors = self._sdk.enumerate()
            LOGGER.info("Detected %s cameras: %s", len(descriptors), [str(item) for item in descriptors])
            assigned = set()
            total_bytes = 0
            for original in self.settings.cameras:
                descriptor = self._sdk.find(original, descriptors)
                if descriptor.index in assigned:
                    raise FrameMatchError("two configured camera IDs resolve to the same device")
                assigned.add(descriptor.index)
                features = {**self.settings.trigger_features, **original.features}
                config = original.model_copy(update={"features": features})
                sequence = SimpleNamespace(sequence_source=self.settings.sequence_source,
                                           sequence_base=original.frame_number_base,
                                           min_pulse_interval_ms=0)
                camera = factory(self._sdk, self.settings.sdk, descriptor,
                    camera_config=config, trigger_config=sequence,
                    slot_count=self.settings.slots_per_camera, sink=self._matcher.add,
                    on_pool_exhausted=self._overflow, on_fault=self._camera_fault)
                self.cameras.append(camera)
                camera.open(memory_budget_bytes=int(self.settings.max_total_memory_mb * 1e6) - total_bytes)
                total_bytes += camera.pool.total_bytes
                if total_bytes / 1e6 > self.settings.max_total_memory_mb:
                    raise FrameMatchError("camera buffers exceed mvs.max_total_memory_mb")
            for camera in self.cameras:
                camera.start()
            self._matcher.check()
            self._started = True
            LOGGER.info("All %s cameras armed; waiting for MCU hardware pulses", len(self.cameras))
        except BaseException:
            self.stop()
            raise

    def _overflow(self, camera):
        self._matcher.fail(FrameMatchError(f"{camera.camera_id}: camera buffer pool exhausted; capture cannot keep up"))
        return None

    def _camera_fault(self, camera, error):
        self._matcher.fail(FrameMatchError(f"{camera.camera_id}: {error}"))

    def check_health(self):
        if self._matcher is not None:
            self._matcher.check()

    def get_frames(self, trigger, camera_ids):
        if not self._started:
            raise FrameMatchError("camera source has not started")
        leases = self._matcher.take(trigger)
        result = {}
        try:
            for camera_id in camera_ids:
                captured = leases[camera_id]
                frame = CameraFrame(
                    camera_id=camera_id, frame_reference=f"mvs://{camera_id}/{trigger.trigger_id}",
                    trigger_id=trigger.trigger_id, capture_timestamp_us=trigger.timestamp_us,
                    camera_frame_number=captured.info.frame_number,
                    camera_trigger_index=captured.info.trigger_index,
                    device_timestamp_ticks=captured.info.device_timestamp_ticks,
                    host_timestamp_ms=captured.info.host_timestamp_ms,
                    pixel_format=captured.info.pixel_format_name,
                    image=self._inference_image(captured))
                frame._lease = captured
                result[camera_id] = frame
                LOGGER.info("Matched camera=%s frame=%s trigger=%s encoder=%s MCU_timestamp_us=%s device_ticks=%s",
                            camera_id, captured.info.frame_number, trigger.trigger_id,
                            trigger.encoder_count, trigger.timestamp_us, captured.info.device_timestamp_ticks)
            return result
        except BaseException:
            for frame in leases.values():
                frame.release()
            raise

    @staticmethod
    def _inference_image(frame):
        # Boundary contract: uint8 mono or BGR, with an application-owned lease.
        import cv2
        from .mvs import pixel_formats as pf
        image = frame.image
        kind = frame.info.pixel_type
        if frame._pool.gray(frame.slot_index) is not None or kind == pf.MONO8:
            return image
        if kind == pf.BGR8_PACKED:
            return image
        if kind == pf.RGB8_PACKED:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        conversions = {pf.BAYER_GR8: cv2.COLOR_BAYER_GR2BGR,
                       pf.BAYER_RG8: cv2.COLOR_BAYER_RG2BGR,
                       pf.BAYER_GB8: cv2.COLOR_BAYER_GB2BGR,
                       pf.BAYER_BG8: cv2.COLOR_BAYER_BG2BGR}
        return cv2.cvtColor(image, conversions[kind])

    def stop(self):
        if self._matcher is not None:
            self._matcher.close()
        for camera in self.cameras:
            camera.request_stop()
        for camera in self.cameras:
            camera.stop()
        for camera in self.cameras:
            camera.close()
        self.cameras.clear()
        if self._sdk is not None and self.settings.sdk.finalize_on_stop:
            self._sdk.finalize()
        self._started = False
