"""Tests for temporary sequential trigger/frame bundle creation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config.models import FrameIngest
from frame_ingest_manager.frame_sources.folder_frame_source import (
    FolderFrameSource,
    FolderFrameSourceError,
)
from frame_ingest_manager.services.frame_ingest_manager import (
    FrameIngestManager,
)
from frame_ingest_manager.models.trigger_telemetry import TriggerTelemetry


class FrameIngestManagerTests(unittest.TestCase):
    """Verify one trigger consumes exactly one configured camera set."""

    def test_create_bundle_assigns_sorted_files_to_configured_cameras(self) -> None:
        """Each trigger gets the next files in configured camera-ID order."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder_path = Path(temporary_directory)
            for filename in (
                "image_003.jpg",
                "image_001.jpg",
                "image_004.jpg",
                "image_002.jpg",
            ):
                (folder_path / filename).touch()

            config = FrameIngest.model_validate(
                {
                    "source_kind": "folder",
                    "folder_path": str(folder_path),
                    "camera_ids": ["camera_1", "camera_2"],
                    "expected_camera_count": 2,
                }
            )
            manager = FrameIngestManager(
                camera_ids=config.camera_ids,
                expected_camera_count=config.expected_camera_count,
                frame_source=FolderFrameSource(folder_path),
            )

            first_bundle = manager.create_bundle(
                TriggerTelemetry(
                    trigger_id=101,
                    encoder_count=10_000,
                    timestamp_us=1_000_000,
                )
            )
            second_bundle = manager.create_bundle(
                TriggerTelemetry(
                    trigger_id=102,
                    encoder_count=12_875,
                    timestamp_us=1_500_000,
                )
            )

            self.assertEqual(first_bundle.trigger_id, 101)
            self.assertEqual(
                first_bundle.camera_frames["camera_1"].capture_timestamp_us,
                1_000_000,
            )
            self.assertEqual(
                first_bundle.camera_frames["camera_1"].trigger_id,
                101,
            )
            self.assertEqual(
                {
                    camera_id: Path(reference).name
                    for camera_id, reference in first_bundle.frame_references.items()
                },
                {"camera_1": "image_001.jpg", "camera_2": "image_002.jpg"},
            )
            self.assertEqual(
                {
                    camera_id: Path(reference).name
                    for camera_id, reference in second_bundle.frame_references.items()
                },
                {"camera_1": "image_003.jpg", "camera_2": "image_004.jpg"},
            )

    def test_create_bundle_propagates_source_exhaustion(self) -> None:
        """A trigger cannot silently reuse a file after mock input ends."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder_path = Path(temporary_directory)
            (folder_path / "image_001.jpg").touch()
            config = FrameIngest.model_validate(
                {
                    "source_kind": "folder",
                    "folder_path": str(folder_path),
                    "camera_ids": ["camera_1", "camera_2"],
                    "expected_camera_count": 2,
                }
            )
            manager = FrameIngestManager(
                camera_ids=config.camera_ids,
                expected_camera_count=config.expected_camera_count,
                frame_source=FolderFrameSource(folder_path),
            )

            with self.assertRaises(FolderFrameSourceError):
                manager.create_bundle(
                    TriggerTelemetry(
                        trigger_id=101,
                        encoder_count=10_000,
                        timestamp_us=1_000_000,
                    )
                )


if __name__ == "__main__":
    unittest.main()
