"""Create deterministic overlapping tiles with their full-frame offsets."""

import logging
from dataclasses import dataclass

import numpy as np

from config.models import Inference

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageTile:
    """One tile; valid dimensions exclude padding outside the original frame."""

    tile_id: int
    image: np.ndarray
    x_offset: int
    y_offset: int
    valid_width: int
    valid_height: int


class TileGenerator:
    """Own tiling only, with no model, GPU, or database dependency."""

    def __init__(self, inference_config: Inference) -> None:
        self._inference_config = inference_config

    @staticmethod
    def _starts(length: int, size: int, stride: int) -> list[int]:
        """Include the final image edge without introducing uncovered strips."""
        if length <= size:
            return [0]
        starts = list(range(0, length - size + 1, stride))
        if starts[-1] != length - size:
            starts.append(length - size)
        return starts

    def generate(self, image: np.ndarray) -> list[ImageTile]:
        """Return row-major tiles; pad only images smaller than a tile."""
        LOGGER.info("TileGenerator.generate: shape=%s", image.shape)
        if image.ndim not in (2, 3) or not image.size:
            raise ValueError("expected a nonempty grayscale or colour image")
        config = self._inference_config
        size = config.tile_size_px
        stride = max(1, int(size * (1 - config.tile_overlap_ratio)))
        height, width = image.shape[:2]
        tiles = []
        for y in self._starts(height, size, stride):
            for x in self._starts(width, size, stride):
                valid_height = min(size, height - y)
                valid_width = min(size, width - x)
                crop = image[y:y + valid_height, x:x + valid_width]
                if valid_height != size or valid_width != size:
                    padded = np.full(
                        (size, size) + image.shape[2:], config.padding_value,
                        dtype=image.dtype,
                    )
                    padded[:valid_height, :valid_width] = crop
                    crop = padded
                tiles.append(ImageTile(
                    len(tiles), crop, x, y, valid_width, valid_height,
                ))
        LOGGER.info("TileGenerator.generate: tiles=%s", len(tiles))
        return tiles
