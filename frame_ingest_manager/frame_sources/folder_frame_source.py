"""Temporary sequential image source for folder-based mock testing.

This source deliberately knows nothing about MCU triggers or camera IDs.  It
only returns the next regular files in deterministic filename order.  The
future FrameIngestManager will assign those files to configured cameras for a
specific trigger.
"""

from __future__ import annotations

from pathlib import Path


class FolderFrameSourceError(RuntimeError):
    """Raised when the folder source cannot supply the requested files."""


class FolderFrameSource:
    """Consume mock image files once, in deterministic filename order."""

    def __init__(self, folder_path: Path) -> None:
        """Discover source files and initialise the next unread position.

        Args:
            folder_path: Directory containing only mock image files.

        Raises:
            FolderFrameSourceError: If the configured path is not a directory.
        """
        self._folder_path = Path(folder_path)
        self._files = self._discover_files()
        self._next_index = 0

    def start(self):
        pass

    def stop(self):
        pass

    def check_health(self):
        pass

    def get_frames(self, trigger, camera_ids):
        from frame_ingest_manager.models.camera_frame import CameraFrame
        return {
            camera_id: CameraFrame(camera_id=camera_id, frame_reference=reference,
                                   trigger_id=trigger.trigger_id,
                                   capture_timestamp_us=trigger.timestamp_us)
            for camera_id, reference in zip(camera_ids, self.take_next(len(camera_ids)))
        }

    def take_next(self, count: int) -> list[str]:
        """Return file references in order, repeating from the beginning."""
        if count <= 0:
            raise FolderFrameSourceError("count must be greater than zero")

        if not self._files:
            raise FolderFrameSourceError("mock image folder is empty")

        files: list[str] = []

        for _ in range(count):
            files.append(str(self._files[self._next_index]))
            self._next_index += 1

            if self._next_index >= len(self._files):
                self._next_index = 0

        return files

    def _discover_files(self) -> list[Path]:
        """Return regular files sorted by case-insensitive filename.

        Sorting makes the mock repeatable across operating systems.  No image
        extension is hardcoded: the configured mock folder must contain only
        files intended for this temporary source.
        """
        if not self._folder_path.is_dir():
            raise FolderFrameSourceError(
                f"mock image folder does not exist: {self._folder_path}"
            )

        return sorted(
            (path for path in self._folder_path.iterdir() if path.is_file()),
            key=lambda path: path.name.casefold(),
        )

    def read_bytes(self, reference: str) -> bytes:
        """Read a whole folder frame without cropping or re-encoding it."""
        return Path(reference).read_bytes()


__all__ = ["FolderFrameSource", "FolderFrameSourceError"]
