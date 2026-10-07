"""Create the frame-source implementation selected by validated YAML."""

from __future__ import annotations

from pathlib import Path

from config.models import FrameIngest
from frame_ingest_manager.frame_sources.folder_frame_source import (
    FolderFrameSource,
)
from frame_ingest_manager.frame_sources.frame_source import FrameSource


class FrameSourceFactoryError(RuntimeError):
    """Raised when the selected frame source cannot be created."""


class FrameSourceFactory:
    """Composition-only factory for configured frame-source implementations."""

    @staticmethod
    def create(config: FrameIngest) -> FrameSource:
        """Create the implementation selected by ``config.source_kind``.

        Args:
            config: Validated frame-ingest settings from the application config.

        Raises:
            FrameSourceFactoryError: If the selected implementation is absent.
        """
        if config.source_kind == "folder":
            return FolderFrameSource(Path(config.folder_path))

        if config.source_kind == "mvs":
            from .mvs_frame_source import MvsFrameSource
            return MvsFrameSource(config)

        raise FrameSourceFactoryError(
            f"unsupported frame source: {config.source_kind!r}"
        )


__all__ = ["FrameSourceFactory", "FrameSourceFactoryError"]
