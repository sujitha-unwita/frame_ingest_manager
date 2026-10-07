"""Test tiling geometry and padding without model or GPU dependencies."""

import unittest
from types import SimpleNamespace

import numpy as np

from inference_pipeline.engines.tile_generator import TileGenerator


class TileGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = TileGenerator(SimpleNamespace(
            tile_size_px=4, tile_overlap_ratio=0.25, padding_value=17,
        ))

    def test_tiles_cover_edges_in_row_major_order(self):
        image = np.arange(70, dtype=np.uint8).reshape(7, 10)
        tiles = self.generator.generate(image)
        self.assertEqual(
            [(t.x_offset, t.y_offset) for t in tiles],
            [(0, 0), (3, 0), (6, 0), (0, 3), (3, 3), (6, 3)],
        )
        self.assertEqual([t.tile_id for t in tiles], list(range(6)))
        covered = np.zeros(image.shape, dtype=bool)
        for tile in tiles:
            covered[tile.y_offset:tile.y_offset + tile.valid_height,
                    tile.x_offset:tile.x_offset + tile.valid_width] = True
        self.assertTrue(covered.all())

    def test_non_stride_aligned_last_tile_reaches_edge(self):
        tiles = self.generator.generate(np.zeros((4, 9), dtype=np.uint8))
        self.assertEqual([t.x_offset for t in tiles], [0, 3, 5])

    def test_small_grayscale_frame_is_padded_and_valid_extent_retained(self):
        tile = self.generator.generate(np.ones((2, 3), dtype=np.uint8))[0]
        self.assertEqual((tile.valid_height, tile.valid_width), (2, 3))
        self.assertEqual(tile.image.shape, (4, 4))
        self.assertTrue((tile.image[:2, :3] == 1).all())
        self.assertTrue((tile.image[2:, :] == 17).all())
        self.assertTrue((tile.image[:, 3:] == 17).all())

    def test_colour_channels_are_preserved(self):
        image = np.full((2, 2, 3), (1, 2, 3), dtype=np.uint8)
        tile = self.generator.generate(image)[0]
        np.testing.assert_array_equal(tile.image[0, 0], [1, 2, 3])
        self.assertEqual(tile.image.shape, (4, 4, 3))

    def test_exact_tile_size_is_not_duplicated(self):
        self.assertEqual(len(self.generator.generate(np.zeros((4, 4)))), 1)

    def test_empty_image_is_rejected(self):
        with self.assertRaises(ValueError):
            self.generator.generate(np.zeros((0, 4)))
