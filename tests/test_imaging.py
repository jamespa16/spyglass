import numpy as np

from spyglass.imaging import DEFAULT_GAP_COLOR, compose_tile_grid


def test_compose_tile_grid_places_tiles_and_gap():
    tiles = np.array(
        [
            np.full((2, 3, 3), (10, 20, 30), dtype=np.uint8),
            np.full((2, 3, 3), (40, 50, 60), dtype=np.uint8),
            np.full((2, 3, 3), (70, 80, 90), dtype=np.uint8),
            np.full((2, 3, 3), (100, 110, 120), dtype=np.uint8),
        ]
    )

    result = compose_tile_grid(tiles, gap=1)

    # 2x2 grid of (2, 3) tiles with a 1px gap around and between them.
    assert result.shape == (2 * 2 + 3 * 1, 2 * 3 + 3 * 1, 3)

    # Every pixel is either one tile's RGB color or the gap color -- nothing
    # else should appear on the canvas.
    flat = result.reshape(-1, 3)
    allowed = {(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120), DEFAULT_GAP_COLOR}
    assert set(map(tuple, flat)) <= allowed

    # Border and seams are the gap color.
    assert tuple(result[0, 0]) == DEFAULT_GAP_COLOR
    assert tuple(result[-1, -1]) == DEFAULT_GAP_COLOR

    # First tile occupies its expected interior block.
    assert tuple(result[1, 1]) == (10, 20, 30)


def test_compose_tile_grid_single_tile_has_border_gap():
    tiles = np.full((1, 4, 4, 3), (200, 150, 100), dtype=np.uint8)
    result = compose_tile_grid(tiles, gap=2, gap_color=(0, 255, 0))

    assert result.shape == (4 + 2 * 2, 4 + 2 * 2, 3)
    assert tuple(result[0, 0]) == (0, 255, 0)
    assert tuple(result[2, 2]) == (200, 150, 100)
