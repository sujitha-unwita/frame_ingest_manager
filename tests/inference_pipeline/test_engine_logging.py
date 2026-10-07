"""Check failure visibility without loading GPU or image dependencies."""

import asyncio
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from config.config_loader import load_config
from inference_pipeline.engines.sahi_detection_engine import SahiDetectionEngine
from inference_pipeline.engines.tile_generator import TileGenerator


class EngineLoggingTests(unittest.IsolatedAsyncioTestCase):
    """Logging must preserve failures and cancellation semantics."""

    def setUp(self) -> None:
        """Use a temporary model file; model loading itself is mocked below."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        model = Path(folder.name) / "test-model.pt"
        model.touch()
        self.config = load_config(overrides={
            "audit_write": {"enabled": False},
            "inference": {
                "model_path": str(model),
                "layout": {"mm_per_pixel_x": 0.5},
            },
        }).inference

    async def test_failed_start_logs_and_preserves_exception(self) -> None:
        """A failed model load stays retryable and raises the original error."""
        engine = SahiDetectionEngine(self.config, TileGenerator(self.config))
        failure = RuntimeError("model unavailable")
        logger = "inference_pipeline.engines.sahi_detection_engine"
        with patch.object(engine, "_load_model", side_effect=failure):
            with self.assertLogs(logger, level="INFO") as captured:
                with self.assertRaises(RuntimeError) as raised:
                    await engine.start()
        self.assertIs(raised.exception, failure)
        self.assertTrue(any(record.exc_info for record in captured.records))
        model = object()
        with patch.object(engine, "_load_model", return_value=model) as load:
            await engine.start()
            await engine.start()
        load.assert_called_once()

    async def test_cancellation_is_not_logged_as_failure(self) -> None:
        """Shutdown cancellation must pass through the async logging boundary."""
        engine = SahiDetectionEngine(self.config, TileGenerator(self.config))
        with patch(
            "inference_pipeline.engines.sahi_detection_engine.asyncio.to_thread",
            side_effect=asyncio.CancelledError,
        ):
            with patch(
                "inference_pipeline.engines.sahi_detection_engine.LOGGER"
            ) as logger:
                with self.assertRaises(asyncio.CancelledError):
                    await engine.start()
                logger.exception.assert_not_called()
