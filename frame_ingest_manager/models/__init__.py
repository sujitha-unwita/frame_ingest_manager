"""Validated trigger and frame exchange models."""

from frame_ingest_manager.models.camera_frame import CameraFrame
from frame_ingest_manager.models.trigger_frame_bundle import TriggerFrameBundle
from frame_ingest_manager.models.trigger_telemetry import TriggerTelemetry

__all__ = ["CameraFrame", "TriggerFrameBundle", "TriggerTelemetry"]
