"""Exercise the real preprocessing and SAHI postprocessor with mocked model I/O."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from config.config_loader import load_config
from inference_pipeline.engines.sahi_detection_engine import SahiDetectionEngine
from inference_pipeline.engines.tile_generator import TileGenerator
from utils.timing import Timing


class ExplicitInferenceTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        model = Path(folder.name) / "model.pt"
        model.touch()
        self.config = load_config(overrides={
            "audit_write": {"enabled": False},
            "inference": {
                "model_path": str(model), "device_preference": "cpu",
                "tile_size_px": 4, "tile_batch_size": 2,
                "tile_overlap_ratio": 0.5, "input_channels": 1,
                "padding_value": 0,
            },
        }).inference
        self.tiles = TileGenerator(self.config)
        self.engine = SahiDetectionEngine(self.config, self.tiles)

    @staticmethod
    def result(boxes):
        return SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                conf=torch.full((len(boxes),), 0.9),
                cls=torch.zeros(len(boxes)),
            ),
            names={0: "hole"},
        )

    def test_grayscale_batch_normalization_and_dummy_padding(self):
        tiles = self.tiles.generate(np.full((4, 4), 255, dtype=np.uint8))
        tensor = self.engine._prepare_batch(tiles)
        self.assertEqual(tuple(tensor.shape), (2, 1, 4, 4))
        self.assertTrue(torch.all(tensor[0] == 1))
        self.assertTrue(torch.all(tensor[1] == 0))

    def test_rgb_channel_order_is_preserved_in_tensor(self):
        self.config.input_channels = 3
        image = np.full((4, 4, 3), (255, 0, 127), dtype=np.uint8)
        tensor = self.engine._prepare_batch(self.tiles.generate(image))
        self.assertEqual(tuple(tensor.shape), (2, 3, 4, 4))
        self.assertEqual(tensor[0, 0, 0, 0].item(), 1)
        self.assertEqual(tensor[0, 1, 0, 0].item(), 0)

    def test_boxes_in_padding_are_removed_and_partial_boxes_clipped(self):
        tile = self.tiles.generate(np.zeros((2, 3), dtype=np.uint8))[0]
        result = self.result([[0, 0, 4, 4], [3, 0, 4, 1]])
        restored = self.engine._restore_predictions(result, tile)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].bbox.to_xyxy(), [0, 0, 3, 2])

    def test_warmup_initializes_configured_batch_shape(self):
        model = Mock()
        with patch("inference_pipeline.engines.sahi_detection_engine.YOLO", return_value=model):
            self.assertIs(self.engine._load_model(), model)
        self.assertEqual(tuple(model.predict.call_args.kwargs["source"].shape), (2, 1, 4, 4))
        self.assertEqual(model.predict.call_args.kwargs["device"], "cpu")

    def test_frame_offsets_are_restored_then_duplicate_boxes_merged(self):
        image = np.zeros((4, 6), dtype=np.uint8)
        frame = SimpleNamespace(trigger_id=1, camera_id="camera_3", frame_reference="test.jpg")
        with patch("inference_pipeline.engines.sahi_detection_engine.cv2.imread", return_value=image), \
                patch.object(self.engine, "_predict", return_value=[
                    self.result([[2, 1, 4, 3]]), self.result([[0, 1, 2, 3]]),
                ]):
            detections = self.engine._detect_sync(frame)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bounding_box_px.x1, 2)
        self.assertEqual(detections[0].bounding_box_px.x2, 4)
        self.assertEqual(detections[0].frame_width_px, 6)

    def test_dummy_batch_predictions_are_discarded(self):
        frame = SimpleNamespace(trigger_id=1, camera_id="camera_3", frame_reference="test.jpg")
        with patch("inference_pipeline.engines.sahi_detection_engine.cv2.imread", return_value=np.zeros((4, 4), dtype=np.uint8)), \
                patch.object(self.engine, "_predict", return_value=[
                    self.result([]), self.result([[0, 0, 2, 2]]),
                ]):
            self.assertEqual(self.engine._detect_sync(frame), [])

    def test_frame_read_decode_records_camera_and_preserves_read_failure(self):
        path = self.config.model_path.parent / "image-read.csv"
        engine = SahiDetectionEngine(
            self.config, self.tiles, timing=Timing(True, path),
        )
        frame = SimpleNamespace(
            trigger_id=7, camera_id="camera_3", frame_reference="missing.jpg",
        )
        with patch(
            "inference_pipeline.engines.sahi_detection_engine.cv2.imread",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot read frame"):
                engine._detect_sync(frame)
        with path.open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["stage"], "image_read_decode")
        self.assertEqual(row["trigger_id"], "7")
        self.assertEqual(row["camera_id"], "camera_3")
        self.assertEqual(row["status"], "error")
