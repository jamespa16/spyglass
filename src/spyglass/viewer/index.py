"""Builds the JSON index the viewer frontend consumes from a spyglass results folder."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from spyglass.chain import BLOCK_TENSOR_RE

_FILENAME_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)\.weight\.png$")
_FILENAME_GLOBAL_RE = re.compile(r"^(.+)\.weight\.png$")

CHAIN_HEATMAP_NAME = "chain_heatmap.png"
CHAIN_SCATTER_NAME = "chain_scatter.png"
CHAIN_SCATTER_CSV_NAME = "chain_scatter.csv"
CHAIN_REPORT_NAME = "chain_report.json"


def build_index(results_dir: Path) -> dict:
    manifest_path = results_dir / "manifest.json"
    if manifest_path.is_file():
        index = _build_from_manifest(results_dir, manifest_path)
    else:
        index = _build_from_directory_scan(results_dir)

    index["chain"] = _build_chain_section(results_dir)
    index["title"] = index.get("title") or results_dir.name
    index["layer_indices"] = sorted(int(i) for i in index["layers"].keys())
    index["n_layers"] = (
        (index["layer_indices"][-1] + 1) if index["layer_indices"] else 0
    )
    index["families"] = sorted(index["families"])
    return index


def _new_layer_entry() -> dict:
    return {}


def _infer_hidden_size(outcomes: list[dict]) -> Optional[int]:
    """Guesses the model's residual-stream width the same way `chain.infer_hidden_size`
    does, but from manifest.json shapes rather than live TensorInfo -- it's the one
    dimension size shared across nearly every 2D `blk.N.*.weight` tensor."""
    dim_counts: Counter[int] = Counter()
    for outcome in outcomes:
        shape = outcome.get("shape")
        if not shape or len(shape) not in (2, 3) or not BLOCK_TENSOR_RE.match(outcome["name"]):
            continue
        for d in shape[-2:]:
            dim_counts[d] += 1
    if not dim_counts:
        return None
    hidden_size, _ = dim_counts.most_common(1)[0]
    return hidden_size


def _needs_rotation(shape: Optional[list], hidden_size: Optional[int]) -> bool:
    """True when this tensor should be transposed so it reads horizontally.

    Priority 1: if either axis ties to the residual stream, put *that* axis
    horizontal even if it isn't the longer one (e.g. ffn_gate/up, where the
    intermediate size outgrows hidden_size but hidden_size is already on the
    column axis) -- that's the axis that's actually comparable channel-for-
    channel across every other tensor, which is the whole point of aligning it.

    Priority 2 (fallback): a tensor with no axis tied to the residual stream
    at all -- an SSM conv1d kernel, a low-rank projection -- has nothing to
    line up against other tensors, so there's no alignment reason to rotate
    it. But left alone, one with a long, thin native shape (e.g. a (10240, 4)
    conv1d kernel) renders as an absurdly tall sliver once a viewer stretches
    it to a row's full width. Orienting its long axis horizontal instead --
    the same way `ssm_alpha`/`ssm_beta` already happen to sit, with their
    5120-wide axis as the column one -- keeps every row in the stack reading
    the same "wide, short" way regardless of what the tensor means.
    """
    if not shape or len(shape) != 2:
        return False
    rows, cols = shape
    if hidden_size is not None and (rows == hidden_size or cols == hidden_size):
        return rows == hidden_size and cols != hidden_size
    return rows > cols


def _build_from_manifest(results_dir: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    layers: dict[str, dict] = {}
    global_tensors: list[dict] = []
    families: set[str] = set()

    written = [
        o
        for o in manifest.get("outcomes", [])
        if o.get("status") == "written" and o.get("output_path")
    ]
    hidden_size = _infer_hidden_size(written)

    for outcome in written:
        shape = outcome.get("shape")
        entry = {
            "name": outcome["name"],
            "ggml_type": outcome.get("ggml_type"),
            "shape": shape,
            "stats": outcome.get("stats"),
            "image": Path(outcome["output_path"]).name,
            "rotate": _needs_rotation(shape, hidden_size),
        }

        match = BLOCK_TENSOR_RE.match(outcome["name"])
        if match:
            layer_index, family = match.group(1), match.group(2)
            families.add(family)
            layers.setdefault(layer_index, _new_layer_entry())[family] = entry
        else:
            global_tensors.append(entry)

    return {
        "title": Path(manifest.get("input_path", "")).stem or None,
        "input_path": manifest.get("input_path"),
        "clip_value": manifest.get("clip_value"),
        "clip_source": manifest.get("clip_source"),
        "hidden_size": hidden_size,
        "layers": layers,
        "global_tensors": global_tensors,
        "families": families,
    }


def _build_from_directory_scan(results_dir: Path) -> dict:
    layers: dict[str, dict] = {}
    global_tensors: list[dict] = []
    families: set[str] = set()

    for png_path in sorted(results_dir.glob("*.png")):
        name = png_path.name
        if name in (CHAIN_HEATMAP_NAME, CHAIN_SCATTER_NAME):
            continue

        block_match = _FILENAME_BLOCK_RE.match(name)
        if block_match:
            layer_index, family = block_match.group(1), block_match.group(2)
            families.add(family)
            entry = {
                "name": f"blk.{layer_index}.{family}.weight",
                "ggml_type": None,
                "shape": None,
                "stats": None,
                "image": name,
                "rotate": False,  # shape unknown without manifest.json -- can't infer the hidden axis
            }
            layers.setdefault(layer_index, _new_layer_entry())[family] = entry
            continue

        global_match = _FILENAME_GLOBAL_RE.match(name)
        if global_match:
            global_tensors.append(
                {
                    "name": global_match.group(1),
                    "ggml_type": None,
                    "shape": None,
                    "stats": None,
                    "image": name,
                    "rotate": False,
                }
            )

    return {
        "title": None,
        "hidden_size": None,
        "input_path": None,
        "clip_value": None,
        "clip_source": None,
        "layers": layers,
        "global_tensors": global_tensors,
        "families": families,
    }


def _build_chain_section(results_dir: Path) -> dict:
    heatmap = results_dir / CHAIN_HEATMAP_NAME
    scatter = results_dir / CHAIN_SCATTER_NAME
    scatter_csv = results_dir / CHAIN_SCATTER_CSV_NAME
    report_json = results_dir / CHAIN_REPORT_NAME

    if not heatmap.is_file() and not report_json.is_file():
        return {"available": False}

    report: Optional[dict] = None
    if report_json.is_file():
        report = json.loads(report_json.read_text())

    return {
        "available": True,
        "heatmap": CHAIN_HEATMAP_NAME if heatmap.is_file() else None,
        "scatter": CHAIN_SCATTER_NAME if scatter.is_file() else None,
        "scatter_csv": CHAIN_SCATTER_CSV_NAME if scatter_csv.is_file() else None,
        "hidden_size": report.get("hidden_size") if report else None,
        "n_layers": report.get("n_layers") if report else None,
        "z_outlier_threshold": report.get("z_outlier_threshold") if report else None,
        "top_channels": report.get("top_channels") if report else [],
    }
