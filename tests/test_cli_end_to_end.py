from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from spyglass.cli import build_arg_parser, run


def _run(gguf_path: Path, out_dir: Path, extra_args: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(
        [str(gguf_path), "-o", str(out_dir), *(extra_args or [])]
    )
    return run(args)


def test_end_to_end_writes_expected_pngs_and_skips_unsupported_ndim(gguf_fixture, tmp_path):
    gguf_path, arrays = gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir)

    written = {o.name: o for o in report.outcomes if o.status == "written"}
    assert set(written) == {"float_tensor.weight", "quant_tensor.weight", "moe_tensor.weight"}

    for name in ("float_tensor.weight", "quant_tensor.weight"):
        outcome = written[name]
        png_path = Path(outcome.output_path)
        assert png_path.exists()
        assert png_path.name == f"{name}.png"

        img = Image.open(png_path)
        assert img.mode == "RGB"
        # PIL size is (width, height) == (cols, rows); numpy shape is (rows, cols)
        expected_rows, expected_cols = arrays[name].shape
        assert img.size == (expected_cols, expected_rows)

        pixels = np.array(img)
        assert pixels.std() > 0

    skipped = {o.name for o in report.outcomes if o.status == "skipped_unsupported_ndim"}
    assert skipped == {"norm_tensor.weight"}
    norm_outcome = next(o for o in report.outcomes if o.name == "norm_tensor.weight")
    assert norm_outcome.shape == (32,)


def test_moe_tensor_renders_as_separated_tile_grid(gguf_fixture, tmp_path):
    gguf_path, arrays = gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir)

    moe_outcome = next(o for o in report.outcomes if o.name == "moe_tensor.weight")
    assert moe_outcome.status == "written"
    assert moe_outcome.shape == (4, 8, 16)
    assert moe_outcome.stats["tiles"] == {"count": 4, "tile_shape": [8, 16]}

    png_path = Path(moe_outcome.output_path)
    img = Image.open(png_path)
    assert img.mode == "RGB"

    # 4 tiles of (8, 16) arrange into a 2x2 grid with a gap border around
    # and between each tile (default gap=4), so the canvas is larger than
    # the tiles alone and every pixel on the seams is the gap color.
    pixels = np.array(img)
    assert pixels.shape[0] > 2 * 8
    assert pixels.shape[1] > 2 * 16

    from spyglass.imaging import DEFAULT_GAP_COLOR

    assert tuple(pixels[0, 0]) == DEFAULT_GAP_COLOR
    # Interior of the top-left tile should not be the gap color.
    assert tuple(pixels[6, 6]) != DEFAULT_GAP_COLOR


def test_manifest_records_clip_value_and_outcomes(gguf_fixture, tmp_path):
    gguf_path, _ = gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir)

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())

    assert payload["clip_source"] == "auto_sampled"
    assert payload["clip_value"] == report.clip_value
    assert payload["clip_value"] > 0
    assert len(payload["outcomes"]) == 4


def test_explicit_clip_value_is_used_verbatim_and_reproducible(gguf_fixture, tmp_path):
    gguf_path, _ = gguf_fixture

    out_dir_a = tmp_path / "a"
    out_dir_b = tmp_path / "b"
    report_a = _run(gguf_path, out_dir_a, ["--clip-value", "0.25"])
    report_b = _run(gguf_path, out_dir_b, ["--clip-value", "0.25"])

    assert report_a.clip_value == 0.25
    assert report_a.clip_source == "manual"

    img_a = np.array(Image.open(out_dir_a / "float_tensor.weight.png"))
    img_b = np.array(Image.open(out_dir_b / "float_tensor.weight.png"))
    assert np.array_equal(img_a, img_b)


def test_dry_run_writes_nothing(gguf_fixture, tmp_path):
    gguf_path, _ = gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir, ["--dry-run"])

    assert not out_dir.exists()
    planned = {o.name for o in report.outcomes if o.status == "planned"}
    assert planned == {"float_tensor.weight", "quant_tensor.weight", "moe_tensor.weight"}


def test_include_exclude_filters(gguf_fixture, tmp_path):
    gguf_path, _ = gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir, ["--exclude", "quant_tensor"])

    written = {o.name for o in report.outcomes if o.status == "written"}
    assert written == {"float_tensor.weight", "moe_tensor.weight"}
    filtered = {o.name for o in report.outcomes if o.status == "skipped_filtered"}
    assert "quant_tensor.weight" in filtered


def test_existing_output_is_skipped_unless_overwrite(gguf_fixture, tmp_path):
    gguf_path, _ = gguf_fixture
    out_dir = tmp_path / "out"

    _run(gguf_path, out_dir)
    second = _run(gguf_path, out_dir)
    skipped_exists = {o.name for o in second.outcomes if o.status == "skipped_exists"}
    assert skipped_exists == {"float_tensor.weight", "quant_tensor.weight", "moe_tensor.weight"}

    overwritten = _run(gguf_path, out_dir, ["--overwrite"])
    written = {o.name for o in overwritten.outcomes if o.status == "written"}
    assert written == {"float_tensor.weight", "quant_tensor.weight", "moe_tensor.weight"}


def test_max_dim_downsamples_image(gguf_fixture, tmp_path):
    gguf_path, arrays = gguf_fixture
    out_dir = tmp_path / "out"

    _run(gguf_path, out_dir, ["--max-dim", "16"])

    img = Image.open(out_dir / "float_tensor.weight.png")
    assert max(img.size) <= 16


def test_parallel_jobs_produce_identical_output_to_sequential(gguf_fixture, tmp_path):
    gguf_path, _ = gguf_fixture

    seq_dir = tmp_path / "sequential"
    par_dir = tmp_path / "parallel"

    seq_report = _run(gguf_path, seq_dir, ["--clip-value", "0.3"])
    par_report = _run(
        gguf_path, par_dir,
        ["--clip-value", "0.3", "--jobs", "4", "--max-memory-gb", "1"],
    )

    # Manifest content (outcome order, stats, statuses) must be identical
    # and independent of scheduling, even though completion order isn't.
    # output_path is deliberately excluded: it legitimately differs because
    # the two runs write to different output directories.
    def _comparable(outcomes):
        return [
            (o.name, o.status, o.shape, Path(o.output_path).name if o.output_path else None, o.stats)
            for o in outcomes
        ]

    assert _comparable(seq_report.outcomes) == _comparable(par_report.outcomes)

    for name in ("float_tensor.weight.png", "quant_tensor.weight.png", "moe_tensor.weight.png"):
        seq_img = np.array(Image.open(seq_dir / name))
        par_img = np.array(Image.open(par_dir / name))
        assert np.array_equal(seq_img, par_img)


def test_parallel_jobs_with_more_workers_than_tensors(gguf_fixture, tmp_path):
    gguf_path, _ = gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir, ["--jobs", "16", "--max-memory-gb", "0.5"])

    written = {o.name for o in report.outcomes if o.status == "written"}
    assert written == {"float_tensor.weight", "quant_tensor.weight", "moe_tensor.weight"}


def test_parallel_jobs_respects_tiny_memory_budget_without_hanging(gguf_fixture, tmp_path):
    # A budget smaller than any single tensor should still make progress
    # (each oversized tensor runs alone) rather than deadlocking.
    gguf_path, _ = gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir, ["--jobs", "4", "--max-memory-gb", "0.0000001"])

    written = {o.name for o in report.outcomes if o.status == "written"}
    assert written == {"float_tensor.weight", "quant_tensor.weight", "moe_tensor.weight"}
