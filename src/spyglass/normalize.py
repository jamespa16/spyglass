from __future__ import annotations

from typing import Sequence

import numpy as np

from spyglass.dequant import DequantError, dequantize_tensor
from spyglass.reading import TensorInfo

# Diverging colormap endpoints: negative weights fade in toward NEGATIVE_COLOR,
# positive weights toward POSITIVE_COLOR, and zero is black for both -- so
# magnitude reads as brightness and sign reads as hue, distinct from each other
# instead of sharing one grayscale axis.
NEGATIVE_COLOR = (168, 85, 247)  # purple
POSITIVE_COLOR = (250, 204, 21)  # yellow


def estimate_clip(
    candidates: Sequence[TensorInfo],
    *,
    sample_tensors: int,
    sample_elements: int,
    clip_std: float,
    rng_seed: int = 0,
) -> float:
    """Pool a strided element sample from a handful of tensors spread evenly
    through `candidates` and return clip_std * pooled_std.

    Each sampled tensor is fully dequantized one at a time (same peak-memory
    bound as the main render loop) and immediately dropped after its sample
    is taken, so this never holds more than one tensor's worth of extra
    memory regardless of how many tensors are sampled.
    """
    if not candidates:
        return 0.0

    rng = np.random.default_rng(rng_seed)
    n_pick = min(sample_tensors, len(candidates))
    indices = np.linspace(0, len(candidates) - 1, num=n_pick, dtype=int)
    indices = sorted(set(int(i) for i in indices))

    pooled: list[np.ndarray] = []
    for i in indices:
        info = candidates[i]
        try:
            arr = dequantize_tensor(info)
        except DequantError:
            continue
        flat = arr.reshape(-1)
        if flat.size > sample_elements:
            take = rng.choice(flat.size, size=sample_elements, replace=False)
            flat = flat[take]
        pooled.append(flat.astype(np.float64, copy=False))
        del arr

    if not pooled:
        return 0.0

    all_values = np.concatenate(pooled)
    std = float(np.std(all_values))
    return clip_std * std


def to_diverging_rgb(arr_f32: np.ndarray, clip: float) -> np.ndarray:
    """Maps signed weights to an RGB heatmap against a fixed clip range,
    shared across every image in a run so brightness and hue are both
    comparable across layers: black at zero, saturating toward
    NEGATIVE_COLOR as values go more negative and POSITIVE_COLOR as they
    go more positive.
    """
    if not np.isfinite(clip) or clip <= 0:
        return np.zeros(arr_f32.shape + (3,), dtype=np.uint8)

    t = np.clip(arr_f32, -clip, clip) / clip  # in [-1, 1]
    magnitude = np.abs(t)[..., np.newaxis]
    negative = np.array(NEGATIVE_COLOR, dtype=np.float64)
    positive = np.array(POSITIVE_COLOR, dtype=np.float64)
    color = np.where(t[..., np.newaxis] < 0, negative, positive)
    return np.rint(magnitude * color).astype(np.uint8)


def downsample_block_mean(arr_f32: np.ndarray, max_dim: int) -> np.ndarray:
    """Block-averages the last two axes down to at most max_dim each.

    Any leading axes (e.g. an expert axis on a 3D MoE tensor) are left
    alone and downsampled independently, so a stack of tiles keeps its
    tile count while each tile shrinks the same way a plain 2D tensor would.
    """
    *lead, rows, cols = arr_f32.shape
    if rows <= max_dim and cols <= max_dim:
        return arr_f32

    fy = max(1, -(-rows // max_dim))  # ceil division
    fx = max(1, -(-cols // max_dim))

    pad_rows = (-rows) % fy
    pad_cols = (-cols) % fx
    if pad_rows or pad_cols:
        fill_value = float(np.mean(arr_f32))
        pad_width = [(0, 0)] * len(lead) + [(0, pad_rows), (0, pad_cols)]
        arr_f32 = np.pad(arr_f32, pad_width, mode="constant", constant_values=fill_value)

    *_, padded_rows, padded_cols = arr_f32.shape
    reshaped = arr_f32.reshape(*lead, padded_rows // fy, fy, padded_cols // fx, fx)
    return reshaped.mean(axis=(len(lead) + 1, len(lead) + 3))
