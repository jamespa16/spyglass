from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spyglass.chain import (
    ChainSample,
    build_chain_report,
    channel_magnitude,
    infer_hidden_size,
    render_chain_scatter,
    top_outlier_channels,
)
from spyglass.cli import build_arg_parser, run
from spyglass.reading import TensorInfo
import gguf


def _info(name: str, shape: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(name=name, ggml_type=gguf.GGMLQuantizationType.F32, shape=shape, raw=None)


def test_infer_hidden_size_picks_the_recurring_axis():
    # 8 is the only size shared by every family (attn_q's "other" axis is 12,
    # ffn_down's is 20 -- neither recurs elsewhere), so it must win even
    # though it's never the larger dimension in any single tensor.
    infos = [
        _info("blk.0.attn_q.weight", (12, 8)),
        _info("blk.0.ffn_down.weight", (8, 20)),
        _info("blk.1.attn_q.weight", (12, 8)),
        _info("blk.1.ffn_down.weight", (8, 20)),
        _info("blk.0.ssm_conv1d.weight", (5, 5)),  # neither axis is the hidden size
        _info("not_a_block_tensor.weight", (8, 8)),  # ignored: no blk.N. prefix
    ]
    assert infer_hidden_size(infos) == 8


def test_infer_hidden_size_returns_none_without_block_tensors():
    assert infer_hidden_size([_info("token_embd.weight", (100, 8))]) is None


def test_channel_magnitude_picks_matching_axis_and_skips_otherwise():
    arr = np.zeros((3, 4), dtype=np.float32)
    arr[:, 2] = 5.0
    mag = channel_magnitude(arr, hidden_size=4)
    assert mag.shape == (4,)
    assert mag[2] == 5.0
    assert mag[0] == 0.0

    assert channel_magnitude(arr, hidden_size=99) is None


def test_channel_magnitude_averages_moe_expert_axis():
    arr = np.zeros((2, 3, 4), dtype=np.float32)  # (experts, rows, cols)
    arr[0, :, 1] = 2.0
    arr[1, :, 1] = 6.0
    mag = channel_magnitude(arr, hidden_size=4)
    assert mag[1] == 4.0  # averaged across both experts


def test_build_chain_report_flags_a_consistent_outlier_channel():
    # With one outlier among otherwise-constant values, the outlier's own
    # z-score is capped at sqrt(hidden_size - 1) no matter how large it gets
    # (it dominates the population std it's being measured against) -- so
    # hidden_size has to be comfortably above z_threshold**2 + 1 for the
    # outlier to actually cross the threshold.
    hidden_size = 20
    outlier_channel = 7
    samples = []
    for layer in range(4):
        vec = np.full(hidden_size, 1.0)
        vec[outlier_channel] = 500.0
        samples.append(ChainSample(layer=layer, family="attn_q", magnitude=vec))

    report = build_chain_report(samples, hidden_size=hidden_size)

    assert report is not None
    assert report.n_layers == 4
    assert report.cells == 4
    assert report.top_channels[0][0] == outlier_channel
    assert report.top_channels[0][1] == 4  # outlier in every layer
    assert report.heat.shape == (4, hidden_size)


def test_build_chain_report_returns_none_for_no_samples():
    assert build_chain_report([], hidden_size=8) is None


def test_top_outlier_channels_matches_the_reported_ranking():
    samples = []
    hidden_size = 300  # so 1% (top_k=3) is unambiguous
    for layer in range(4):
        vec = np.full(hidden_size, 1.0)
        vec[5] = 500.0
        vec[6] = 400.0
        vec[7] = 300.0
        samples.append(ChainSample(layer=layer, family="attn_q", magnitude=vec))

    report = build_chain_report(samples, hidden_size=hidden_size)

    assert top_outlier_channels(report) == {5, 6, 7}


def test_render_chain_scatter_writes_an_image(tmp_path):
    hidden_size = 40
    samples = [
        ChainSample(layer=layer, family="attn_q", magnitude=np.full(hidden_size, 1.0))
        for layer in range(3)
    ]
    samples[0].magnitude[3] = 500.0
    samples[1].magnitude[3] = 500.0
    samples[2].magnitude[3] = 500.0
    report = build_chain_report(samples, hidden_size=hidden_size)
    assert report is not None

    gamma = np.ones(hidden_size, dtype=np.float32)
    out_path = tmp_path / "chain_scatter.png"

    render_chain_scatter(report, gamma, out_path)

    assert out_path.exists()
    img = Image.open(out_path)
    assert img.mode == "RGB"
    assert img.size[0] > 0 and img.size[1] > 0


def test_render_chain_scatter_rejects_mismatched_gamma_shape(tmp_path):
    samples = [ChainSample(layer=0, family="attn_q", magnitude=np.full(10, 1.0))]
    report = build_chain_report(samples, hidden_size=10)
    assert report is not None

    with pytest.raises(ValueError):
        render_chain_scatter(report, np.ones(5, dtype=np.float32), tmp_path / "out.png")


def _run(gguf_path: Path, out_dir: Path, extra_args: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args([str(gguf_path), "-o", str(out_dir), *(extra_args or [])])
    return run(args)


def test_cli_chain_report_end_to_end(chain_gguf_fixture, tmp_path):
    gguf_path, info = chain_gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir, ["--chain-report"])

    assert report.chain_report_path is not None
    assert report.chain_heatmap_path is not None
    assert report.chain_scatter_path is not None

    heatmap_path = Path(report.chain_heatmap_path)
    assert heatmap_path.exists()
    img = Image.open(heatmap_path)
    assert img.mode == "RGB"

    scatter_path = Path(report.chain_scatter_path)
    assert scatter_path.exists()
    scatter_img = Image.open(scatter_path)
    assert scatter_img.mode == "RGB"

    payload = json.loads(Path(report.chain_report_path).read_text())
    assert payload["hidden_size"] == info["hidden_size"]
    assert payload["n_layers"] == info["n_layers"]
    assert payload["top_channels"], "expected at least one outlier channel"
    assert payload["top_channels"][0]["channel"] == info["outlier_channel"]
    assert payload["top_channels"][0]["hit_rate"] == 1.0

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["chain_report_path"] == report.chain_report_path
    assert manifest["chain_heatmap_path"] == report.chain_heatmap_path
    assert manifest["chain_scatter_path"] == report.chain_scatter_path


def test_cli_without_chain_report_flag_writes_nothing_extra(chain_gguf_fixture, tmp_path):
    gguf_path, _ = chain_gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir)

    assert report.chain_report_path is None
    assert report.chain_heatmap_path is None
    assert report.chain_scatter_path is None
    assert not (out_dir / "chain_report.json").exists()
    assert not (out_dir / "chain_heatmap.png").exists()
    assert not (out_dir / "chain_scatter.png").exists()


def test_cli_chain_report_on_gguf_without_block_tensors_is_a_noop(gguf_fixture, tmp_path):
    gguf_path, _ = gguf_fixture
    out_dir = tmp_path / "out"

    report = _run(gguf_path, out_dir, ["--chain-report"])

    assert report.chain_report_path is None
    assert report.chain_heatmap_path is None
    assert report.chain_scatter_path is None
