"""Contract tests for the temporary sequential folder frame source."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frame_ingest_manager.frame_sources.folder_frame_source import (
    FolderFrameSource,
    FolderFrameSourceError,
)


class FolderFrameSourceTests(unittest.TestCase):
    """Verify deterministic file consumption without camera/trigger logic."""

    def test_take_next_returns_files_in_sorted_order(self) -> None:
        """The source consumes the next requested regular files in name order."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder_path = Path(temporary_directory)
            for filename in ("image_003.jpg", "image_001.jpg", "image_002.jpg"):
                (folder_path / filename).touch()

            source = FolderFrameSource(folder_path)

            first_group = source.take_next(count=2)
            second_group = source.take_next(count=1)

            self.assertEqual(
                [Path(reference).name for reference in first_group],
                ["image_001.jpg", "image_002.jpg"],
            )
            self.assertEqual(
                [Path(reference).name for reference in second_group],
                ["image_003.jpg"],
            )

    def test_take_next_rejects_insufficient_files(self) -> None:
        """The source reports exhaustion instead of reusing a mock image."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder_path = Path(temporary_directory)
            (folder_path / "image_001.jpg").touch()
            source = FolderFrameSource(folder_path)

            with self.assertRaises(FolderFrameSourceError):
                source.take_next(count=2)


if __name__ == "__main__":
    unittest.main()
