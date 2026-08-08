#!/usr/bin/env python3
"""Analyze qwen_modality_probe.cpp's output: for each of the three channel
bands found by spyglass --chain-report ([0,1024), [1024,4096), [4096,5120)),
compute how differently each channel responds to image-token positions vs.
text-token positions ("modality contrast"). If a band were vision-specialized
(H1 in docs/qwen-channel-banding.md), it should show a much higher contrast
than the rest of the residual stream. It doesn't -- see that doc for results.

Usage: python3 analyze_modality_probe.py probe_output.json
"""
import json
import sys

import numpy as np

BANDS = {
    "edge_low":  (0, 1024),
    "middle":    (1024, 4096),
    "edge_high": (4096, 5120),
}
LAYERS = [0, 8, 16, 24, 32, 40, 48, 56, 63]


def band_masks(n_embd: int) -> dict[str, np.ndarray]:
    idx = np.arange(n_embd)
    return {name: (idx >= lo) & (idx < hi) for name, (lo, hi) in BANDS.items()}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "probe_output.json"
    data = json.load(open(path))
    n_embd = data["n_embd"]
    masks = band_masks(n_embd)

    print(f"{'layer':>6} {'band':>10} {'mean|text|':>11} {'mean|img|':>11} {'contrast':>9}")
    agg_contrast = {name: [] for name in masks}
    for il in LAYERS:
        tensor = data["tensors"].get(f"l_out-{il}")
        if tensor is None:
            continue
        text_abs = np.array(tensor["text"]["mean_abs"])
        img_abs = np.array(tensor["image"]["mean_abs"])
        contrast = np.abs(img_abs - text_abs) / (img_abs + text_abs + 1e-8)
        for band, mask in masks.items():
            mt, mi, mc = text_abs[mask].mean(), img_abs[mask].mean(), contrast[mask].mean()
            agg_contrast[band].append(mc)
            print(f"{il:6d} {band:>10} {mt:11.5f} {mi:11.5f} {mc:9.4f}")
        print()

    print("=== averaged across sampled layers ===")
    for band, vals in agg_contrast.items():
        print(f"{band:>10}: mean modality-contrast = {np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
