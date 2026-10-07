"""Score settled physical defects using the configured project poster policy."""

import logging
import math

from config.models import FourPointGrading
from inference_pipeline.models.defect_group import DefectGroup

LOGGER = logging.getLogger(__name__)


class FourPointGradingEngine:
    """Calculate per-defect points; own no database or roll-total state."""

    def __init__(
        self,
        grading_config: FourPointGrading,
        active_width_mm: float,
    ) -> None:
        """Keep policy and the same active-width reference used by the mapper."""
        LOGGER.info("FourPointGradingEngine.__init__: started")
        self._grading_config = grading_config
        if not math.isfinite(active_width_mm) or active_width_mm <= 0:
            raise ValueError("active inspection width must be positive and finite")
        self._active_width_mm = active_width_mm
        if grading_config.full_width_tolerance_mm >= self._active_width_mm / 2:
            raise ValueError("full-width tolerance must be below half active width")
        self._hole_classes = {
            label.strip().casefold() for label in grading_config.hole_classes
        }

    def grade(self, group: DefectGroup) -> DefectGroup:
        """Return a scored copy without changing the merger's open group.

        Hole and full-active-width rules take precedence over length bands.
        This is not a roll grade: no per-yard cap or inspected-area total is
        applied here. Dimensions describe the mapped, possibly clipped region.
        """
        LOGGER.info("FourPointGradingEngine.grade: class=%s", group.defect_class)
        try:
            coordinates = (
                group.world_x1_mm, group.world_y1_mm,
                group.world_x2_mm, group.world_y2_mm,
            )
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("grading requires finite world coordinates")
            width_mm = group.world_x2_mm - group.world_x1_mm
            height_mm = group.world_y2_mm - group.world_y1_mm
            if width_mm <= 0 or height_mm <= 0:
                raise ValueError("grading requires a positive rectangle")

            config = self._grading_config
            tolerance = config.full_width_tolerance_mm
            if group.defect_class.strip().casefold() in self._hole_classes:
                points = config.hole_points
            elif (
                group.world_x1_mm <= tolerance
                and group.world_x2_mm >= self._active_width_mm - tolerance
            ):
                points = config.full_width_points
            else:
                length_inches = max(width_mm, height_mm) / config.mm_per_inch
                points = config.oversize_points
                for limit, band_points in zip(
                    config.length_limits_inches, config.length_points
                ):
                    if length_inches <= limit:
                        points = band_points
                        break

            LOGGER.info("FourPointGradingEngine.grade: points=%s", points)
            return group.model_copy(update={"four_point_score": points})
        except Exception:
            LOGGER.exception("FourPointGradingEngine.grade: failed")
            raise
