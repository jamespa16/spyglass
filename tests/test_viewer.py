from __future__ import annotations

import io
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

from spyglass.cli import build_arg_parser, run
from spyglass.viewer.index import _needs_rotation, build_index
from spyglass.viewer.server import ViewerRequestHandler


def _run(gguf_path: Path, out_dir: Path, extra_args: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args([str(gguf_path), "-o", str(out_dir), *(extra_args or [])])
    return run(args)


def test_needs_rotation_prioritizes_hidden_axis_over_long_axis():
    hidden_size = 2560
    # ffn_gate/up-shaped: intermediate size (10240) is the longer axis, but
    # hidden_size is already on the column axis -- must NOT rotate, since that
    # would misalign the one axis actually comparable across every tensor.
    assert _needs_rotation([10240, 2560], hidden_size) is False
    # attn_output/proj-shaped: hidden_size is on the row axis -- must rotate.
    assert _needs_rotation([2560, 2048], hidden_size) is True


def test_needs_rotation_falls_back_to_long_axis_when_hidden_size_absent():
    hidden_size = 5120
    # An ssm_conv1d-shaped kernel: neither axis is hidden_size, so there's no
    # channel-alignment reason to rotate -- but its long axis (10240) is
    # vertical, which would otherwise stretch to an absurd height in the
    # stack views, so the fallback rule rotates it long-axis-horizontal.
    assert _needs_rotation([10240, 4], hidden_size) is True
    # ssm_alpha/beta-shaped: also untied to hidden_size, but the long axis
    # (5120) is already horizontal, so leave it alone.
    assert _needs_rotation([48, 5120], hidden_size) is False
    # A square, unrelated tensor: no basis to prefer either orientation.
    assert _needs_rotation([5, 5], hidden_size) is False


def test_build_index_from_manifest_groups_by_layer_and_family(chain_gguf_fixture, tmp_path):
    gguf_path, _ = chain_gguf_fixture
    out_dir = tmp_path / "out"
    _run(gguf_path, out_dir, ["--chain-report"])

    index = build_index(out_dir)

    assert index["layer_indices"] == [0, 1, 2]
    assert index["n_layers"] == 3
    assert index["families"] == ["attn_q", "ffn_down", "ssm_conv1d"]

    # ssm_conv1d.weight only exists on layer 0 in the fixture.
    assert set(index["layers"]["0"].keys()) == {"attn_q", "ffn_down", "ssm_conv1d"}
    assert set(index["layers"]["1"].keys()) == {"attn_q", "ffn_down"}

    entry = index["layers"]["0"]["attn_q"]
    assert entry["name"] == "blk.0.attn_q.weight"
    assert entry["image"] == "blk.0.attn_q.weight.png"
    assert entry["stats"]["std"] > 0

    # output_norm.weight is 1D (skipped) so no global tensors get written here.
    assert index["global_tensors"] == []

    # HIDDEN_SIZE=32 in the fixture. attn_q is (16, 32) -- hidden already on the
    # column/horizontal axis, so it needs no rotation. ffn_down is (32, 20) --
    # hidden is on the row/vertical axis there, so it does. ssm_conv1d (5, 5)
    # doesn't touch the hidden size on either axis, so it's left alone too.
    assert index["hidden_size"] == 32
    assert index["layers"]["0"]["attn_q"]["rotate"] is False
    assert index["layers"]["0"]["ffn_down"]["rotate"] is True
    assert index["layers"]["0"]["ssm_conv1d"]["rotate"] is False

    assert index["chain"]["available"] is True
    assert index["chain"]["heatmap"] == "chain_heatmap.png"
    assert index["chain"]["scatter"] == "chain_scatter.png"
    top_channels = index["chain"]["top_channels"]
    assert top_channels[0]["channel"] == 17  # OUTLIER_CHANNEL in the fixture


def test_build_index_falls_back_to_directory_scan_without_manifest(chain_gguf_fixture, tmp_path):
    gguf_path, _ = chain_gguf_fixture
    out_dir = tmp_path / "out"
    _run(gguf_path, out_dir, ["--chain-report"])
    (out_dir / "manifest.json").unlink()

    index = build_index(out_dir)

    assert index["layer_indices"] == [0, 1, 2]
    assert index["families"] == ["attn_q", "ffn_down", "ssm_conv1d"]
    assert index["layers"]["0"]["attn_q"]["stats"] is None
    assert index["layers"]["0"]["attn_q"]["image"] == "blk.0.attn_q.weight.png"
    assert index["chain"]["available"] is True

    # No manifest means no shape info, so rotation can't be inferred either.
    assert index["hidden_size"] is None
    assert index["layers"]["0"]["ffn_down"]["rotate"] is False


def test_transpose_query_param_swaps_png_dimensions(chain_gguf_fixture, tmp_path):
    gguf_path, _ = chain_gguf_fixture
    out_dir = tmp_path / "out"
    _run(gguf_path, out_dir, ["--chain-report"])

    ViewerRequestHandler.results_dir = out_dir
    httpd = HTTPServer(("127.0.0.1", 0), ViewerRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}/data/blk.0.ffn_down.weight.png"
        with urlopen(base_url) as resp:
            original_size = Image.open(io.BytesIO(resp.read())).size
        with urlopen(f"{base_url}?transpose=1") as resp:
            transposed_size = Image.open(io.BytesIO(resp.read())).size

        assert transposed_size == (original_size[1], original_size[0])
    finally:
        httpd.shutdown()
        thread.join()


def test_build_index_handles_missing_chain_report(gguf_fixture, tmp_path):
    gguf_path, _ = gguf_fixture
    out_dir = tmp_path / "out"
    _run(gguf_path, out_dir)

    index = build_index(out_dir)

    assert index["chain"] == {"available": False}
    assert index["layer_indices"] == []
    names = {t["name"] for t in index["global_tensors"]}
    assert names == {"float_tensor.weight", "quant_tensor.weight", "moe_tensor.weight"}
