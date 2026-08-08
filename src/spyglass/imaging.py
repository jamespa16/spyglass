from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

# Distinct from anything the diverging weight colormap produces, so the seams
# between tiles read unambiguously as "not weight data" rather than as a
# saturated positive/negative value.
DEFAULT_GAP_COLOR = (255, 64, 0)


def write_rgb_png(arr_u8_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8_rgb, mode="RGB").save(path)


def compose_tile_grid(
    tiles_rgb: np.ndarray,
    *,
    gap: int = 4,
    gap_color: tuple[int, int, int] = DEFAULT_GAP_COLOR,
) -> np.ndarray:
    """Arranges a stack of RGB tiles (n, rows, cols, 3) into one RGB grid
    image, roughly square, with a solid-color gap between and around tiles
    so each tile (e.g. one MoE expert's weight matrix) stays visually
    distinct instead of blurring into its neighbors.
    """
    n, tile_rows, tile_cols, _ = tiles_rgb.shape
    n_cols = max(1, math.ceil(math.sqrt(n)))
    n_rows = math.ceil(n / n_cols)

    out_h = n_rows * tile_rows + (n_rows + 1) * gap
    out_w = n_cols * tile_cols + (n_cols + 1) * gap

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, :] = gap_color

    for idx in range(n):
        r, c = divmod(idx, n_cols)
        y0 = gap + r * (tile_rows + gap)
        x0 = gap + c * (tile_cols + gap)
        canvas[y0 : y0 + tile_rows, x0 : x0 + tile_cols] = tiles_rgb[idx]

    return canvas
