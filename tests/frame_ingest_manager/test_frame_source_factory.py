"""Tests for selecting a frame-source implementation from configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config.models import FrameIngest
from frame_ingest_manager.frame_sources.folder_frame_source import (
    FolderFrameSource,
)
from frame_ingest_manager.frame_sources.frame_source_factory import (
    FrameSourceFactory,
    FrameSourceFactoryError,
)


class FrameSourceFactoryTests(unittest.TestCase):
    """Verify source selection stays outside FrameIngestManager."""

    def test_create_selects_folder_source(self) -> None:
        """The folder selection creates the folder implementation."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = FrameIngest.model_validate(
                {
                    "source_kind": "folder",
                    "folder_path": temporary_directory,
                    "camera_ids": ["camera_1"],
                    "expected_camera_count": 1,
                }
            )

            source = FrameSourceFactory.create(config)

            self.assertIsInstance(source, FolderFrameSource)

    def test_create_mvs_is_lazy_and_requires_settings(self):
        from pydantic import ValidationError
        from frame_ingest_manager.frame_sources.mvs_frame_source import MvsFrameSource
        with self.assertRaises(ValidationError):
            FrameIngest(source_kind="mvs", camera_ids=["camera_1"], expected_camera_count=1)
        config = FrameIngest(source_kind="mvs", camera_ids=["camera_1"], expected_camera_count=1,
            mvs={"cameras": [{"id": "camera_1", "serial_number": "test"}]})
        source = FrameSourceFactory.create(config)
        self.assertIsInstance(source, MvsFrameSource)
        self.assertIsNone(source._sdk)
        with self.assertRaisesRegex(ValueError, "verified"):
            source.start()


if __name__ == "__main__":
    unittest.main()
