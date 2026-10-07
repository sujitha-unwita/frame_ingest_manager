"""Map frame-local detections into calibrated inspection coordinates.

Active camera coverage and stencil widths define the boundaries. X is relative to the
active inspection area; global pixels retain the camera-bar reference.
"""

from __future__ import annotations

import logging

from config.models import InferenceLayout
from frame_ingest_manager.models.trigger_telemetry import TriggerTelemetry
from inference_pipeline.models import BoundingBoxPx, Detection, MappedDetection


LOGGER = logging.getLogger(__name__)


class CoordinateMapper:
    """Own the configured camera-layout and encoder-coordinate calculation."""

    def __init__(
        self,
        camera_ids: list[str],
        layout_config: InferenceLayout,
        encoder_mm_per_count: float,
    ) -> None:
        """Store validated physical layout inputs without opening any device."""
        LOGGER.info("CoordinateMapper.__init__: started")
        try:
            missing = set(camera_ids) - layout_config.camera_position_indices.keys()
            if missing:
                raise ValueError(f"physical positions missing: {sorted(missing)}")
            self._camera_ids = tuple(camera_ids)
            self._layout_config = layout_config
            self._encoder_mm_per_count = encoder_mm_per_count
            if not camera_ids or len(set(camera_ids)) != len(camera_ids):
                raise ValueError("active camera IDs must be nonempty and unique")
            positions = sorted(
                layout_config.camera_position_indices[camera_id]
                for camera_id in camera_ids
            )
            # Gaps must not be advertised as one continuously inspected width.
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise ValueError("active cameras must occupy consecutive physical slots")
            camera_width_mm = layout_config.frame_width_px * layout_config.mm_per_pixel_x
            step_mm = camera_width_mm * (1 - layout_config.horizontal_overlap_ratio)
            self._active_left_mm = (
                positions[0] * step_mm + layout_config.left_stencil_mm
            )
            self._active_right_mm = (
                positions[-1] * step_mm + camera_width_mm
                - layout_config.right_stencil_mm
            )
            if self._active_right_mm <= self._active_left_mm:
                raise ValueError("stencils must leave positive active camera coverage")
        except Exception:
            # Preserve the failure for the caller; never return fake success.
            LOGGER.exception("CoordinateMapper.__init__: failed")
            raise

    @property
    def active_width_mm(self) -> float:
        """Combined active-camera width after removing the two outer stencils."""
        return self._active_right_mm - self._active_left_mm

    def map_detection(
        self,
        detection: Detection,
        trigger: TriggerTelemetry,
    ) -> MappedDetection | None:
        """Build one complete mapped detection from local pixels and trigger.

        This public method finds the camera's global pixel offset, converts
        horizontal edges with ``_map_x`` and encoder-based vertical edges
        with ``_map_y``, and returns the complete typed object.
        Fully masked boxes return None. Boundary boxes are clipped to their
        inspected portion; the original camera-local box remains unchanged.
        """
        LOGGER.info("CoordinateMapper.map_detection: started trigger=%s camera=%s",
                    detection.trigger_id,
                    detection.camera_id)
        try:
            if detection.trigger_id != trigger.trigger_id:
                raise ValueError("detection and telemetry trigger IDs differ")
            if detection.camera_id not in self._camera_ids:
                raise ValueError(f"inactive camera: {detection.camera_id}")
            if detection.frame_width_px != self._layout_config.frame_width_px:
                raise ValueError("frame width does not match inference.layout.frame_width_px")
            camera_index = self._layout_config.camera_position_indices[detection.camera_id]

            horizontal_step_px = detection.frame_width_px * (
                1 - self._layout_config.horizontal_overlap_ratio
            )
            box = detection.bounding_box_px
            original_x1 = box.x1 + camera_index * horizontal_step_px
            original_x2 = box.x2 + camera_index * horizontal_step_px
            scale_x = self._layout_config.mm_per_pixel_x
            x1 = max(original_x1, self._active_left_mm / scale_x)
            x2 = min(original_x2, self._active_right_mm / scale_x)
            if x2 <= x1:
                LOGGER.info("Detection outside active inspection area; excluded")
                return None
            world_x1_mm, world_x2_mm = self._map_x(x1, x2)
            world_y1_mm, world_y2_mm = self._map_y(
                box.y1, box.y2, trigger.encoder_count
            )
            # The merger needs global pixels. Y already includes encoder
            # travel, so convert units here without adding that travel again.
            global_box = BoundingBoxPx(
                x1=x1,
                y1=world_y1_mm / self._layout_config.mm_per_pixel_y,
                x2=x2,
                y2=world_y2_mm / self._layout_config.mm_per_pixel_y,
            )
            return MappedDetection(
                **detection.model_dump(),
                global_bounding_box_px=global_box,
                clipped_to_active_area=(x1 != original_x1 or x2 != original_x2),
                world_x1_mm=world_x1_mm,
                world_y1_mm=world_y1_mm,
                world_x2_mm=world_x2_mm,
                world_y2_mm=world_y2_mm,
            )
        except Exception:
            # Preserve the failure for the caller; never return fake success.
            LOGGER.exception("CoordinateMapper.map_detection: failed")
            raise

    def _map_x(
        self,
        x1_px: float,
        x2_px: float,
    ) -> tuple[float, float]:
        """Convert global pixels into millimetres from the active area's edge.

        It is private because it is one small calculation used by
        ``map_detection``; it cannot produce a complete ``MappedDetection``
        by itself.
        """
        LOGGER.info("CoordinateMapper._map_x: started")
        try:
            scale = self._layout_config.mm_per_pixel_x
            if self._layout_config.origin.endswith("right"):
                return (
                    self._active_right_mm - x2_px * scale,
                    self._active_right_mm - x1_px * scale,
                )
            return (
                x1_px * scale - self._active_left_mm,
                x2_px * scale - self._active_left_mm,
            )
        except Exception:
            # Preserve the failure for the caller; never return fake success.
            LOGGER.exception("CoordinateMapper._map_x: failed")
            raise

    def _map_y(
        self,
        local_y1_px: float,
        local_y2_px: float,
        encoder_count: int,
    ) -> tuple[float, float]:
        """Add this trigger's travel to local Y converted using vertical GSD.

        Encoder zero is the longitudinal reference. TriggerTelemetry validates
        the count; checking movement between triggers belongs to the pipeline.
        """
        LOGGER.info("CoordinateMapper._map_y: encoder_count=%s", encoder_count)
        try:
            encoder_travel_mm = encoder_count * self._encoder_mm_per_count
            scale_y = self._layout_config.mm_per_pixel_y
            return (
                encoder_travel_mm + local_y1_px * scale_y,
                encoder_travel_mm + local_y2_px * scale_y,
            )
        except Exception:
            LOGGER.exception("CoordinateMapper._map_y: failed")
            raise

    def inspection_coverage(
        self, frame_width_px: int
    ) -> dict[str, tuple[float, float] | None]:
        """Report each camera's inspected interval, measured from active zero.

        None means this camera sees no active fabric. Intervals may overlap;
        their union determines coverage. Uncovered areas remain uninspected.
        All cameras are assumed to share the supplied frame width and GSD.
        """
        LOGGER.info("CoordinateMapper.inspection_coverage: started")
        if frame_width_px != self._layout_config.frame_width_px:
            raise ValueError("frame width does not match inference.layout.frame_width_px")
        step_px = frame_width_px * (1 - self._layout_config.horizontal_overlap_ratio)
        scale = self._layout_config.mm_per_pixel_x
        coverage = {}
        for camera_id in self._camera_ids:
            start = self._layout_config.camera_position_indices[camera_id] * step_px
            left = max(start, self._active_left_mm / scale)
            right = min(
                start + frame_width_px, self._active_right_mm / scale
            )
            if right <= left:
                coverage[camera_id] = None
            else:
                coverage[camera_id] = self._map_x(left, right)
        return coverage


__all__ = ["CoordinateMapper"]
