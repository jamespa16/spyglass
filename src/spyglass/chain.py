from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from spyglass.imaging import DEFAULT_GAP_COLOR
from spyglass.reading import TensorInfo

BLOCK_TENSOR_RE = re.compile(r"^blk\.(\d+)\.(.+)\.weight$")
Z_OUTLIER_THRESHOLD = 3.0
MAX_REPORTED_CHANNELS = 100


def infer_hidden_size(infos: Sequence[TensorInfo]) -> Optional[int]:
    """Guesses the model's residual-stream width from tensor shapes alone.

    Every per-layer weight matrix has exactly one axis tied to the residual
    stream (its input or output projection dimension); the other axis varies
    per tensor family (intermediate size, head_count * head_dim, ...). That
    makes the width the one dimension value that recurs across nearly every
    blk.N.*.weight tensor, so its mode is a robust, architecture-agnostic
    estimate that needs no per-family special-casing (works the same for a
    plain transformer block and a hybrid attention/SSM block).
    """
    dim_counts: Counter[int] = Counter()
    for info in infos:
        if not BLOCK_TENSOR_RE.match(info.name) or info.ndim not in (2, 3):
            continue
        for d in info.shape[-2:]:
            dim_counts[d] += 1
    if not dim_counts:
        return None
    hidden_size, _ = dim_counts.most_common(1)[0]
    return hidden_size


def channel_magnitude(arr_f32: np.ndarray, hidden_size: int) -> Optional[np.ndarray]:
    """Per-channel mean |weight|, or None if neither trailing axis is
    hidden_size (e.g. an SSM conv kernel, unrelated to the residual stream).

    For a 3D tiled tensor (stacked MoE experts), magnitude is averaged over
    the expert axis too, so every tensor still contributes one length-
    hidden_size vector.
    """
    ndim = arr_f32.ndim
    rows, cols = arr_f32.shape[-2], arr_f32.shape[-1]
    if cols == hidden_size:
        channel_axis = ndim - 1
    elif rows == hidden_size:
        channel_axis = ndim - 2
    else:
        return None
    reduce_axes = tuple(i for i in range(ndim) if i != channel_axis)
    return np.mean(np.abs(arr_f32), axis=reduce_axes)


@dataclass
class ChainSample:
    layer: int
    family: str
    magnitude: np.ndarray  # shape (hidden_size,), mean |weight| per channel


@dataclass
class ChainReport:
    hidden_size: int
    n_layers: int
    cells: int  # number of (layer, family) tensors sampled
    outlier_hits: np.ndarray  # (hidden_size,) int, cells where |z-score| > threshold
    heat: np.ndarray  # (n_layers, hidden_size) float, mean |z-score| per layer
    top_channels: list[tuple[int, int, float]]  # (channel, hits, hit_rate), descending


def _zscore(vec: np.ndarray) -> np.ndarray:
    std = vec.std()
    if std == 0:
        return np.zeros_like(vec)
    return (vec - vec.mean()) / std


def build_chain_report(
    samples: Sequence[ChainSample], hidden_size: int, *, z_threshold: float = Z_OUTLIER_THRESHOLD
) -> Optional[ChainReport]:
    if not samples:
        return None

    per_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    outlier_hits = np.zeros(hidden_size, dtype=np.int64)
    for sample in samples:
        z = _zscore(sample.magnitude)
        per_layer[sample.layer].append(z)
        outlier_hits += np.abs(z) > z_threshold

    n_layers = max(per_layer) + 1
    heat = np.zeros((n_layers, hidden_size), dtype=np.float64)
    for layer, zs in per_layer.items():
        heat[layer] = np.abs(np.stack(zs)).mean(axis=0)

    order = np.argsort(-outlier_hits)
    top_channels = [
        (int(c), int(outlier_hits[c]), float(outlier_hits[c]) / len(samples))
        for c in order[:MAX_REPORTED_CHANNELS]
        if outlier_hits[c] > 0
    ]

    return ChainReport(
        hidden_size=hidden_size,
        n_layers=n_layers,
        cells=len(samples),
        outlier_hits=outlier_hits,
        heat=heat,
        top_channels=top_channels,
    )


def top_outlier_channels(report: ChainReport) -> set[int]:
    """The busiest 1% of channels by outlier-hit count -- the same set used
    to mark the heatmap's bottom strip, reused as-is to flag points on the
    scatter, so the two artifacts agree on which channels are "the chain."
    """
    top_k = max(1, report.hidden_size // 100)
    return {c for c, _, _ in report.top_channels[:top_k]}


def render_chain_heatmap(report: ChainReport, path: Path) -> None:
    """Layer (row) x channel (col) grid, brightness = mean |z-score| for that
    channel in that layer. A channel that stays bright top-to-bottom is an
    outlier in nearly every layer -- a "chain" running the model's full depth.
    A marker strip along the bottom flags the single busiest 1% of channels
    in the tool's own gap-orange, so they're easy to spot even when the
    heatmap itself is mostly noise.
    """
    vmax = np.percentile(report.heat, 99.5) or 1.0
    gray = np.clip(report.heat / vmax * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    top_channels = top_outlier_channels(report)
    marker_h = max(4, report.n_layers // 20)
    marker = np.full((marker_h, report.hidden_size, 3), 30, dtype=np.uint8)
    for c in top_channels:
        marker[:, c, :] = DEFAULT_GAP_COLOR
    canvas = np.concatenate([rgb, marker], axis=0)

    scale_x = max(1, 900 // report.hidden_size)
    scale_y = max(2, 600 // report.n_layers)
    img = Image.fromarray(canvas, mode="RGB").resize(
        (report.hidden_size * scale_x, canvas.shape[0] * scale_y), Image.NEAREST
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


SCATTER_SIZE = (1400, 700)
SCATTER_MARGIN = {"left": 70, "right": 20, "top": 40, "bottom": 50}
SCATTER_BG_COLOR = (20, 20, 20)
SCATTER_AXIS_COLOR = (90, 90, 90)
SCATTER_POINT_COLOR = (130, 130, 130)
SCATTER_TEXT_COLOR = (200, 200, 200)


def _nice_ticks(vmin: float, vmax: float, target_count: int = 5) -> list[float]:
    """Evenly spaced tick values landing on round numbers (1/2/5 x10^k)
    within [vmin, vmax], the standard "nice numbers" axis algorithm --
    plain fractions of the data range (e.g. 1/4, 1/2, 3/4) tend to land on
    values like 6.85 that read as noise rather than a scale.
    """
    if vmax <= vmin:
        return [vmin]
    raw_step = (vmax - vmin) / max(1, target_count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    step = (1 if residual <= 1 else 2 if residual <= 2 else 5 if residual <= 5 else 10) * magnitude

    start = math.ceil(vmin / step) * step
    ticks = []
    v = start
    while v <= vmax + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks or [vmin]


def _rank_channels(report: ChainReport) -> np.ndarray:
    """Each channel's 1-indexed rank by outlier-hit count (1 = most
    persistent), shared by the scatter PNG and its CSV export so both agree.
    """
    order = np.argsort(-report.outlier_hits, kind="stable")
    rank_of_channel = np.empty(report.hidden_size, dtype=np.int64)
    rank_of_channel[order] = np.arange(1, report.hidden_size + 1)
    return rank_of_channel


def render_chain_scatter(report: ChainReport, gamma: np.ndarray, path: Path) -> None:
    """Scatter of each channel's outlier-hit rank (x, 1 = the most persistent
    chain channel) against its final-norm gamma (y, how much that channel's
    value actually reaches the output logits, after the model's last RMSNorm).

    Persistence through depth and influence on the final readout are
    different things -- a channel can dominate every mid-network matrix and
    still be scaled down to nearly nothing before it reaches the logits (see
    README). This is the chart that shows whether that's actually happening
    for a given model, rather than assuming it from the heatmap alone.
    """
    hidden_size = report.hidden_size
    if gamma.shape != (hidden_size,):
        raise ValueError(f"gamma shape {gamma.shape} != (hidden_size,)=({hidden_size},)")

    rank_of_channel = _rank_channels(report)
    top_channels = top_outlier_channels(report)

    m = SCATTER_MARGIN
    width, height = SCATTER_SIZE
    plot_w = width - m["left"] - m["right"]
    plot_h = height - m["top"] - m["bottom"]

    gmin, gmax = float(gamma.min()), float(gamma.max())
    if gmin == gmax:
        gmin, gmax = gmin - 1.0, gmax + 1.0
    pad = (gmax - gmin) * 0.05
    gmin, gmax = gmin - pad, gmax + pad

    def x_px(rank: float) -> int:
        return m["left"] + round((rank - 1) / max(1, hidden_size - 1) * plot_w)

    def y_px(value: float) -> int:
        frac = (value - gmin) / (gmax - gmin)
        return m["top"] + round((1.0 - frac) * plot_h)

    img = Image.new("RGB", SCATTER_SIZE, SCATTER_BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw.line([(m["left"], m["top"]), (m["left"], height - m["bottom"])], fill=SCATTER_AXIS_COLOR)
    draw.line(
        [(m["left"], height - m["bottom"]), (width - m["right"], height - m["bottom"])],
        fill=SCATTER_AXIS_COLOR,
    )
    if gmin < 0 < gmax:
        y0 = y_px(0.0)
        draw.line([(m["left"], y0), (width - m["right"], y0)], fill=SCATTER_AXIS_COLOR)

    # Non-flagged channels first, then the chain channels on top so they're
    # never hidden under the general scatter.
    for c in range(hidden_size):
        if c in top_channels:
            continue
        x, y = x_px(int(rank_of_channel[c])), y_px(float(gamma[c]))
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=SCATTER_POINT_COLOR)
    for c in top_channels:
        x, y = x_px(int(rank_of_channel[c])), y_px(float(gamma[c]))
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=DEFAULT_GAP_COLOR)

    for rank in _nice_ticks(1, hidden_size):
        x = x_px(rank)
        draw.line([(x, height - m["bottom"]), (x, height - m["bottom"] + 4)], fill=SCATTER_AXIS_COLOR)
        draw.text((x, height - m["bottom"] + 8), f"{rank:g}", fill=SCATTER_TEXT_COLOR, anchor="ma")
    for value in _nice_ticks(gmin, gmax):
        y = y_px(value)
        draw.line([(m["left"] - 4, y), (m["left"], y)], fill=SCATTER_AXIS_COLOR)
        draw.text((m["left"] - 8, y), f"{value:g}", fill=SCATTER_TEXT_COLOR, anchor="rm")

    draw.text(
        (m["left"], 12),
        "x: outlier-hit rank (1 = most persistent chain channel)   y: final-norm gamma",
        fill=SCATTER_TEXT_COLOR,
    )
    draw.text(
        (width - m["right"], 12),
        f"orange = top {len(top_channels)} chain channels",
        fill=DEFAULT_GAP_COLOR,
        anchor="ra",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_chain_scatter_csv(report: ChainReport, gamma: np.ndarray, path: Path) -> None:
    """Same per-channel data as render_chain_scatter (rank, gamma, chain-channel
    flag), one row per channel including its id, for anyone who wants to
    re-plot or filter it outside the fixed PNG.
    """
    hidden_size = report.hidden_size
    if gamma.shape != (hidden_size,):
        raise ValueError(f"gamma shape {gamma.shape} != (hidden_size,)=({hidden_size},)")

    rank_of_channel = _rank_channels(report)
    top_channels = top_outlier_channels(report)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "rank", "outlier_hits", "gamma", "is_chain_channel"])
        for c in range(hidden_size):
            writer.writerow([
                c,
                int(rank_of_channel[c]),
                int(report.outlier_hits[c]),
                float(gamma[c]),
                c in top_channels,
            ])


def write_chain_report_json(report: ChainReport, path: Path) -> None:
    payload = {
        "hidden_size": report.hidden_size,
        "n_layers": report.n_layers,
        "cells": report.cells,
        "z_outlier_threshold": Z_OUTLIER_THRESHOLD,
        "top_channels": [
            {"channel": c, "outlier_hits": hits, "hit_rate": rate}
            for c, hits, rate in report.top_channels
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
