"""Merge complete camera rows and release settled, unnumbered defects.

Adapted from ../inference/05_dedupe_nmm_final/dedup.py: SAHI NMM first,
then directional gap merging. Input coordinates must already be global.
"""

import logging
from operator import attrgetter
from typing import Literal

import numpy as np
from sahi.postprocess.combine import batched_nmm

from config.models import Deduplication
from inference_pipeline.engines.four_point_grading_engine import (
    FourPointGradingEngine,
)
from inference_pipeline.models.contracts import BoundingBoxPx, MappedDetection
from inference_pipeline.models.defect_group import DefectGroup

LOGGER = logging.getLogger(__name__)


class DeduplicationEngine:
    """Own open defect groups for one roll, with no database or numbering.

    Call process_trigger once with all cameras, including empty detection
    lists. The caller supplies the current frame-row top in global pixels.
    Rows must move forward. finish is for an actual end of roll, not a pause.
    """

    def __init__(
        self, config: Deduplication, camera_ids: list[str],
        grading_engine: FourPointGradingEngine,
    ) -> None:
        """Store YAML thresholds and the expected complete camera set."""
        LOGGER.info("DeduplicationEngine.__init__: started")
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("camera IDs must be nonempty and unique")
        self._config = config
        self._grading_engine = grading_engine
        self._camera_ids = tuple(camera_ids)
        self._open: list[DefectGroup] = []
        self._last_row_top: float | None = None
        self._last_trigger: int | None = None
        self._finished = False

    def process_trigger(
        self,
        trigger_id: int,
        row_top_px: float,
        detections_by_camera: dict[str, list[MappedDetection]],
    ) -> list[DefectGroup]:
        """Merge a complete row and return only groups now safe to finalize.

        Settlement uses the observed row top minus the vertical gap budget.
        This is more conservative than predicting the NEXT trigger position:
        it needs no assumption about future encoder travel. An empty row also
        advances settlement. All state commits only after processing succeeds.
        """
        LOGGER.info("DeduplicationEngine.process_trigger: trigger=%s", trigger_id)
        try:
            self._validate_row(trigger_id, row_top_px, detections_by_camera)
            row: list[DefectGroup] = []
            for camera_id in self._camera_ids:
                incoming = [
                    self._from_detection(item)
                    for item in detections_by_camera[camera_id]
                ]
                row = self._merge(row + incoming, "horizontal")
            merged = self._merge(self._open + row, "vertical")
            frontier = row_top_px - self._config.vertical_gap_px
            settled = []
            pending = []
            for group in merged:
                if group.global_bounding_box_px.y2 < frontier:
                    settled.append(self._grading_engine.grade(group))
                else:
                    pending.append(group)
            self._open = pending
            self._last_row_top = row_top_px
            self._last_trigger = trigger_id
            LOGGER.info("trigger=%s settled=%s open=%s",
                        trigger_id, len(settled), len(pending))
            return settled
        except Exception:
            LOGGER.exception("merging failed for trigger=%s", trigger_id)
            raise

    def finish(self) -> list[DefectGroup]:
        """Release remaining groups once at roll end; subsequent calls are empty."""
        LOGGER.info("DeduplicationEngine.finish: open=%s", len(self._open))
        # Grade everything before clearing state, so a failure loses no groups.
        settled = [self._grading_engine.grade(group) for group in self._open]
        self._open = []
        self._finished = True
        return settled

    @property
    def open_count(self) -> int:
        """Number of groups still waiting for possible later observations."""
        return len(self._open)

    def _validate_row(
        self,
        trigger_id: int,
        row_top_px: float,
        detections_by_camera: dict[str, list[MappedDetection]],
    ) -> None:
        """Reject incomplete, repeated, or backward rows before consuming state."""
        LOGGER.info("DeduplicationEngine._validate_row: started")
        if self._finished:
            raise ValueError("roll has finished; create a merger for the next roll")
        if trigger_id < 0:
            raise ValueError("trigger ID must be nonnegative")
        if set(detections_by_camera) != set(self._camera_ids):
            raise ValueError("one detection list is required for every camera")
        if not np.isfinite(row_top_px) or row_top_px < 0:
            raise ValueError("row top must be finite and nonnegative")
        if self._last_row_top is not None and row_top_px < self._last_row_top:
            raise ValueError("backward encoder movement requires a new roll")
        if self._last_trigger is not None and trigger_id <= self._last_trigger:
            raise ValueError("trigger IDs must increase within this roll")
        seen = {
            member for group in self._open for member in group.member_detection_ids
        }
        for camera_id, detections in detections_by_camera.items():
            for detection in detections:
                if (
                    detection.camera_id != camera_id
                    or detection.trigger_id != trigger_id
                ):
                    raise ValueError("detection camera/trigger does not match row")
                if detection.detection_id is None:
                    raise ValueError("persist raw detections before merging")
                if detection.detection_id in seen:
                    raise ValueError("a raw detection ID appears more than once")
                seen.add(detection.detection_id)
                box = detection.global_bounding_box_px
                coordinates = (box.x1, box.y1, box.x2, box.y2)
                if not all(np.isfinite(value) for value in coordinates):
                    raise ValueError("global coordinates must be finite")
                if box.y1 < row_top_px:
                    raise ValueError("detection lies before its frame row")

    @staticmethod
    def _from_detection(detection: MappedDetection) -> DefectGroup:
        """Retain the raw UUID as merge provenance; assign no defect number."""
        LOGGER.info("DeduplicationEngine._from_detection: started")
        return DefectGroup(
            defect_class=detection.defect_class,
            confidence=detection.confidence,
            global_bounding_box_px=detection.global_bounding_box_px,
            member_detection_ids=(detection.detection_id,),
            world_x1_mm=detection.world_x1_mm,
            world_y1_mm=detection.world_y1_mm,
            world_x2_mm=detection.world_x2_mm,
            world_y2_mm=detection.world_y2_mm,
        )

    def _merge(
        self, groups: list[DefectGroup], direction: Literal["horizontal", "vertical"]
    ) -> list[DefectGroup]:
        """Use SAHI's member indices, then close directional gaps to convergence."""
        LOGGER.info("DeduplicationEngine._merge: direction=%s", direction)
        if len(groups) < 2:
            return list(groups)
        classes = sorted({group.defect_class for group in groups})
        records = []
        for group in groups:
            box = group.global_bounding_box_px
            records.append([box.x1, box.y1, box.x2, box.y2,
                            group.confidence, classes.index(group.defect_class)])
        membership = batched_nmm(
            np.asarray(records),
            match_metric=self._config.match_metric,
            match_threshold=self._config.ios_threshold,
        )
        remaining = []
        for keep, members in membership.items():
            group = groups[keep]
            for member in members:
                group = self._union(group, groups[member])
            remaining.append(group)
        changed = True
        while changed:
            changed = False
            for index, candidate in enumerate(remaining):
                for other in range(index):
                    if self._close(remaining[other], candidate, direction):
                        remaining[other] = self._union(remaining[other], candidate)
                        remaining.pop(index)
                        changed = True
                        break
                if changed:
                    break
        return sorted(remaining, key=attrgetter("world_y1_mm", "world_x1_mm"))

    def _close(
        self,
        first: DefectGroup,
        second: DefectGroup,
        direction: Literal["horizontal", "vertical"],
    ) -> bool:
        """Apply the original same-class directional gap and axis-overlap rule."""
        LOGGER.info("DeduplicationEngine._close: direction=%s", direction)
        if first.defect_class != second.defect_class:
            return False
        a, b = first.global_bounding_box_px, second.global_bounding_box_px
        if direction == "horizontal":
            gap = max(0, a.x1 - b.x2, b.x1 - a.x2)
            overlap = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
            ratio = overlap / min(a.y2 - a.y1, b.y2 - b.y1)
            budget = self._config.horizontal_gap_px
        else:
            gap = max(0, a.y1 - b.y2, b.y1 - a.y2)
            overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
            ratio = overlap / min(a.x2 - a.x1, b.x2 - b.x1)
            budget = self._config.vertical_gap_px
        return gap <= budget and ratio >= self._config.min_axis_overlap_ratio

    @staticmethod
    def _union(first: DefectGroup, second: DefectGroup) -> DefectGroup:
        """Union edges and UUID provenance, keeping the higher confidence."""
        LOGGER.info("DeduplicationEngine._union: started")
        a, b = first.global_bounding_box_px, second.global_bounding_box_px
        return DefectGroup(
            defect_class=first.defect_class,
            confidence=max(first.confidence, second.confidence),
            global_bounding_box_px=BoundingBoxPx(
                x1=min(a.x1, b.x1), y1=min(a.y1, b.y1),
                x2=max(a.x2, b.x2), y2=max(a.y2, b.y2),
            ),
            member_detection_ids=tuple(sorted(set(first.member_detection_ids)
                                              | set(second.member_detection_ids))),
            world_x1_mm=min(first.world_x1_mm, second.world_x1_mm),
            world_y1_mm=min(first.world_y1_mm, second.world_y1_mm),
            world_x2_mm=max(first.world_x2_mm, second.world_x2_mm),
            world_y2_mm=max(first.world_y2_mm, second.world_y2_mm),
        )
