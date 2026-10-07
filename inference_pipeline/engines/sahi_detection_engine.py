"""Explicit Ultralytics tile inference with SAHI-only postprocessing."""

from __future__ import annotations

import asyncio
import logging

from config.models import Inference
from frame_ingest_manager.models.camera_frame import CameraFrame
from inference_pipeline.models import BoundingBoxPx, Detection

import numpy as np
from ultralytics import YOLO
from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import NMMPostprocess, NMSPostprocess
from inference_pipeline.engines.tile_generator import ImageTile, TileGenerator
import torch
import cv2

from utils.timing import Timing


LOGGER = logging.getLogger(__name__)


class SahiDetectionEngine:
    """Own direct model inference; delegate tiling and use SAHI to merge boxes."""

    def __init__(
        self, inference_config: Inference, tile_generator: TileGenerator,
        *, timing: Timing | None = None,
    ) -> None:
        """Store validated settings; model loading is deferred until startup."""
        LOGGER.info("SahiDetectionEngine.__init__: started")
        try:
            self._inference_config = inference_config
            self._model: object | None = None
            self._tile_generator = tile_generator
            self._timing = timing if timing is not None else Timing(False)
            self._device: str | None = None
            postprocessor = (
                NMMPostprocess if inference_config.sahi_postprocess_type == "NMM"
                else NMSPostprocess
            )
            self._postprocessor = postprocessor(
                match_threshold=inference_config.sahi_match_threshold,
                match_metric=inference_config.sahi_match_metric,
                class_agnostic=inference_config.postprocess_class_agnostic,
            )
        except Exception:
            # Preserve the failure for the caller; never return fake success.
            LOGGER.exception("SahiDetectionEngine.__init__: failed")
            raise

    async def start(self) -> None:
        """Load the configured model once, outside the asyncio event loop.

        ``start`` is the public lifecycle method. It stores the loaded model
        on this object so every later ``detect`` call reuses it.
        """
        LOGGER.info("SahiDetectionEngine.start: started")
        try:
            if self._model is not None:
                return
            LOGGER.info("loading inference model from configured model_path")
            loading = asyncio.create_task(asyncio.to_thread(self._load_model))
            try:
                self._model = await asyncio.shield(loading)
            except asyncio.CancelledError:
                # Cancelling an await cannot stop a running worker thread.
                # Wait before allowing shutdown to release model resources.
                try:
                    await loading
                finally:
                    self._model = None
                raise
            LOGGER.info("inference model loaded")
        except Exception:
            # Preserve the failure for the caller; never return fake success.
            LOGGER.exception("SahiDetectionEngine.start: failed")
            raise

    async def detect(self, frame: CameraFrame) -> list[Detection]:
        """Read one folder image and return its SAHI/NMM detections."""
        LOGGER.info("SahiDetectionEngine.detect: started trigger=%s camera=%s",
                    frame.trigger_id,
                    frame.camera_id)
        try:
            if self._model is None:
                raise RuntimeError("SahiDetectionEngine.start() must run first")
            detecting = asyncio.create_task(
                asyncio.to_thread(self._detect_sync, frame)
            )
            try:
                return await asyncio.shield(detecting)
            except asyncio.CancelledError:
                await detecting
                raise
        except Exception:
            # Preserve the failure for the caller; never return fake success.
            LOGGER.exception("SahiDetectionEngine.detect: failed")
            raise

    async def stop(self) -> None:
        """Release model ownership after the orchestrator drains inference.

        This is shutdown, not a roll-end signal. It does not finalize defects.
        """
        LOGGER.info("SahiDetectionEngine.stop: releasing model")
        self._model = None

    def _load_model(self) -> object:
        """Load Ultralytics directly and warm up the configured tensor shape.

        TensorRT actually initializes on prediction. Warmup makes incompatible
        channels, input sizes, batches or runtime versions fail before capture.
        """
        LOGGER.info("SahiDetectionEngine._load_model: started")
        config = self._inference_config
        if not config.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {config.model_path}")
        self._device = self._select_device(torch.cuda.is_available())
        if config.model_path.suffix.lower() == ".engine" and self._device == "cpu":
            raise RuntimeError("TensorRT engines require a CUDA-enabled runtime")
        model = YOLO(str(config.model_path), task="detect")
        warmup = torch.zeros(
            (config.tile_batch_size, config.input_channels,
             config.tile_size_px, config.tile_size_px),
            dtype=torch.float32,
        )
        self._predict(model, warmup)
        return model

    def _predict(self, model: object, tensor: torch.Tensor) -> list:
        """Use the same prediction settings for warmup and real tile batches."""
        config = self._inference_config
        LOGGER.info("SahiDetectionEngine._predict: shape=%s", tuple(tensor.shape))
        with torch.inference_mode():
            return model.predict(
                source=tensor,
                device=self._device,
                imgsz=config.tile_size_px,
                conf=config.confidence_threshold,
                iou=config.prediction_iou_threshold,
                max_det=config.prediction_max_detections,
                agnostic_nms=config.prediction_class_agnostic,
                verbose=config.prediction_verbose,
                save=False,
                stream=False,
            )

    def _select_device(self, cuda_available: bool) -> str:
        """Resolve the YAML device policy against currently available hardware."""
        LOGGER.info("SahiDetectionEngine._select_device: started")
        try:
            preference = self._inference_config.device_preference
            if preference == "auto":
                return "cuda:0" if cuda_available else "cpu"
            if preference == "cuda:0" and not cuda_available:
                raise RuntimeError("inference requests cuda:0 but CUDA is unavailable")
            return preference
        except Exception:
            # Preserve the failure for the caller; never return fake success.
            LOGGER.exception("SahiDetectionEngine._select_device: failed")
            raise

    def _prepare_batch(self, tiles: list[ImageTile]) -> torch.Tensor:
        """Create normalized BCHW input and pad an incomplete fixed batch.

        Three-channel tile images are already RGB. One-channel images have
        shape HW, so insert their channel dimension before transposing.
        """
        config = self._inference_config
        images = [tile.image for tile in tiles]
        while len(images) < config.tile_batch_size:
            images.append(np.full_like(images[0], config.padding_value))
        array = np.stack(images)
        if config.input_channels == 1:
            array = array[:, None, :, :]
        else:
            array = array.transpose(0, 3, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(array)).float().div(
            config.normalization_divisor
        )

    @staticmethod
    def _restore_predictions(result: object, tile: ImageTile) -> list[ObjectPrediction]:
        """Clip off padding and translate tile-local boxes to frame coordinates."""
        objects = []
        if result.boxes is None:
            return objects
        coordinates = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(coordinates, scores, classes):
            if not np.isfinite(box).all() or not np.isfinite(score):
                raise ValueError("model returned nonfinite prediction values")
            x1 = max(0.0, float(box[0]))
            y1 = max(0.0, float(box[1]))
            x2 = min(tile.valid_width, float(box[2]))
            y2 = min(tile.valid_height, float(box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            objects.append(ObjectPrediction(
                bbox=[x1 + tile.x_offset, y1 + tile.y_offset,
                      x2 + tile.x_offset, y2 + tile.y_offset],
                score=float(score),
                category_id=int(class_id),
                category_name=result.names[int(class_id)],
            ))
        return objects

    def _detect_sync(self, frame: CameraFrame) -> list[Detection]:
        """Decode, tile, infer batches, restore coordinates and merge with SAHI."""
        LOGGER.info(
            "SahiDetectionEngine._detect_sync: trigger=%s camera=%s",
            frame.trigger_id, frame.camera_id,
        )
        try:
            config = self._inference_config
            read_mode = cv2.IMREAD_GRAYSCALE if config.input_channels == 1 else cv2.IMREAD_COLOR
            with self._timing.measure(
                "image_read_decode", trigger_id=frame.trigger_id,
                camera_id=frame.camera_id,
            ):
                if getattr(frame, "image", None) is None:
                    if frame.frame_reference.startswith("mvs://"):
                        raise RuntimeError("live camera frame lease has already been released")
                    image = cv2.imread(str(frame.frame_reference), read_mode)
                else:
                    image = frame.image
                    if image.dtype != np.uint8 or image.ndim not in (2, 3):
                        raise ValueError("live frame must be uint8 mono or BGR")
                    if image.ndim == 3 and image.shape[2] != 3:
                        raise ValueError("live color frame must have three BGR channels")
                    if config.input_channels == 1 and image.ndim == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    elif config.input_channels == 3 and image.ndim == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if image is None:
                    raise RuntimeError(
                        f"cannot read frame {frame.frame_reference!r}"
                    )
            if config.input_channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height_px, width_px = image.shape[:2]
            tiles = self._tile_generator.generate(image)
            objects = []
            for start in range(0, len(tiles), config.tile_batch_size):
                batch = tiles[start:start + config.tile_batch_size]
                results = self._predict(self._model, self._prepare_batch(batch))
                if len(results) != config.tile_batch_size:
                    raise RuntimeError("model result count does not match configured batch")
                # Ignore results from the padded entries of the last batch.
                for result, tile in zip(results[:len(batch)], batch):
                    objects.extend(self._restore_predictions(result, tile))
            merged = self._postprocessor(objects) if objects else []
            detections = []
            for prediction in merged:
                x1, y1, x2, y2 = prediction.bbox.to_xyxy()
                detections.append(Detection(
                    trigger_id=frame.trigger_id,
                    camera_id=frame.camera_id,
                    frame_reference=frame.frame_reference,
                    defect_class=prediction.category.name,
                    confidence=float(prediction.score.value),
                    bounding_box_px=BoundingBoxPx(x1=x1, y1=y1, x2=x2, y2=y2),
                    frame_width_px=width_px,
                    frame_height_px=height_px,
                    # SAHI ObjectPrediction merging does not preserve tile membership.
                    # Keep this unknown rather than inventing contributing tile IDs.
                    tile_ids=None,
                ))
            LOGGER.info(
                "SahiDetectionEngine._detect_sync: tiles=%s detections=%s",
                len(tiles), len(detections),
            )
            return detections
        except Exception:
            LOGGER.exception("SahiDetectionEngine._detect_sync: failed")
            raise


__all__ = ["SahiDetectionEngine"]
