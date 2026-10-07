"""Lifecycle and complete camera/telemetry bundles for folder or live sources."""
from frame_ingest_manager.frame_sources.frame_source import FrameSource
from frame_ingest_manager.models.trigger_frame_bundle import TriggerFrameBundle
from frame_ingest_manager.models.trigger_telemetry import TriggerTelemetry

class FrameIngestManager:
    def __init__(self, camera_ids: list[str], expected_camera_count: int, frame_source: FrameSource):
        if len(set(camera_ids)) != len(camera_ids) or expected_camera_count != len(camera_ids):
            raise ValueError("a unique camera set matching expected_camera_count is required")
        self._camera_ids = camera_ids
        self._frame_source = frame_source

    def start(self):
        self._frame_source.start()

    def stop(self):
        self._frame_source.stop()

    def check_health(self):
        self._frame_source.check_health()

    def create_bundle(self, trigger: TriggerTelemetry) -> TriggerFrameBundle:
        frames = self._frame_source.get_frames(trigger, self._camera_ids)
        try:
            if set(frames) != set(self._camera_ids):
                raise ValueError("a complete frame set is required for every trigger")
            return TriggerFrameBundle(trigger=trigger, camera_frames=frames)
        except BaseException:
            for frame in frames.values():
                frame.release()
            raise
