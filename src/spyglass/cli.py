from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from spyglass import __version__
from spyglass.chain import (
    BLOCK_TENSOR_RE,
    ChainSample,
    build_chain_report,
    channel_magnitude,
    infer_hidden_size,
    render_chain_heatmap,
    render_chain_scatter,
    write_chain_report_json,
    write_chain_scatter_csv,
)
from spyglass.dequant import DequantError, dequantize_tensor
from spyglass.imaging import compose_tile_grid, write_rgb_png
from spyglass.naming import safe_filename
from spyglass.normalize import downsample_block_mean, estimate_clip, to_diverging_rgb
from spyglass.reading import TensorInfo, iter_tensor_infos, open_reader
from spyglass.report import RunReport, TensorOutcome, print_summary, write_manifest
from spyglass.scheduling import default_memory_budget_bytes, run_memory_aware

logger = logging.getLogger("spyglass")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spyglass",
        description=(
            "Render every 2D weight tensor in a GGUF file as a purple/black/yellow "
            "diverging-colormap PNG heatmap (3D tensors, e.g. stacked MoE experts, "
            "render as a grid of per-slice tiles)."
        ),
        epilog=(
            "Memory note: by default tensors are dequantized and written one at a "
            "time, so peak extra memory is bounded by the largest single tensor "
            "(often the token embedding / output projection matrix). Use --exclude "
            "or --max-dim to avoid rendering the largest tensors at full resolution, "
            "or --jobs to process multiple tensors concurrently (bounded by "
            "--max-memory-gb as well, so total in-flight memory stays capped)."
        ),
    )
    parser.add_argument("gguf_path", type=Path, help="path to the .gguf file")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="default: '<gguf-stem>_weights' in the current directory",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-render PNGs that already exist (default: skip)",
    )
    parser.add_argument("--no-manifest", action="store_true", help="disable manifest.json")

    clip_group = parser.add_mutually_exclusive_group()
    clip_group.add_argument(
        "--clip-std", type=float, default=4.0,
        help="multiplies the auto-sampled pooled std to get the brightness clip (default: 4.0)",
    )
    clip_group.add_argument(
        "--clip-value", type=float, default=None,
        help="explicit fixed brightness clip in raw weight units; skips sampling",
    )

    parser.add_argument("--sample-tensors", type=int, default=12)
    parser.add_argument("--sample-elements", type=int, default=1_000_000)
    parser.add_argument("--sample-seed", type=int, default=0)

    parser.add_argument(
        "--include", action="append", default=[], metavar="REGEX",
        help="only render tensors whose name matches (repeatable)",
    )
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="REGEX",
        help="skip tensors whose name matches (repeatable)",
    )

    parser.add_argument(
        "--max-dim", type=int, default=None,
        help="block-average downsample tensors above this size in either dimension "
        "(default: unset, full resolution)",
    )

    parser.add_argument(
        "--jobs", type=int, default=1,
        help="number of tensors to dequantize/render concurrently (default: 1, sequential). "
        "Bounded by --max-memory-gb as well, so a value larger than what the memory "
        "budget allows won't all run at once.",
    )
    parser.add_argument(
        "--max-memory-gb", type=float, default=None,
        help="total bytes of dequantized tensor data allowed in flight at once, in GiB "
        "(only relevant with --jobs > 1). Default: half of detected system RAM, or "
        "4 GiB if it can't be detected.",
    )

    parser.add_argument(
        "--chain-report", action="store_true",
        help="also write chain_heatmap.png and chain_report.json, tracing which "
        "residual-stream channels have outlier weight magnitude across nearly every "
        "layer (a 'chain' running the model's full depth -- see README). Covers only "
        "tensors written in this run, so combine with --overwrite for a complete "
        "picture on a directory that was already rendered.",
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="list tensors and planned actions without dequantizing or writing anything",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true")
    verbosity.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=f"spyglass {__version__}")
    return parser


def _matches_filters(name: str, include: list[re.Pattern], exclude: list[re.Pattern]) -> bool:
    if include and not any(p.search(name) for p in include):
        return False
    if any(p.search(name) for p in exclude):
        return False
    return True


def _process_tensor(
    info: TensorInfo,
    out_path: Path,
    clip: float,
    max_dim: Optional[int],
    chain_hidden_size: Optional[int] = None,
    chain_samples: Optional[list[ChainSample]] = None,
) -> TensorOutcome:
    """Dequantizes, normalizes, and writes a single tensor's PNG.

    2D tensors render as a single diverging-colormap heatmap. 3D tensors (e.g. a
    stacked MoE expert weight, shape (n_expert, rows, cols)) render as a
    grid of one heatmap tile per slice along axis 0, separated by a solid
    color gap so experts stay visually distinct instead of blurring
    together as if they were one contiguous matrix.

    Never raises: every failure mode is captured as an "error" or
    "skipped_dequant_unsupported" TensorOutcome instead, since the parallel
    scheduler relies on this to always return rather than throw (an
    exception here would leave that task's memory/worker slot un-freed).
    """
    try:
        arr_f32 = dequantize_tensor(info)
    except DequantError as exc:
        logger.warning("skip (dequant unsupported) %s: %s", info.name, exc)
        return TensorOutcome(
            name=info.name, ggml_type=info.ggml_type.name, shape=info.shape,
            status="skipped_dequant_unsupported", reason=str(exc),
        )

    if chain_hidden_size is not None and chain_samples is not None:
        block_match = BLOCK_TENSOR_RE.match(info.name)
        if block_match:
            magnitude = channel_magnitude(arr_f32, chain_hidden_size)
            if magnitude is not None:
                # list.append is atomic under the GIL, so threads sharing
                # chain_samples (via --jobs > 1) need no extra lock here.
                chain_samples.append(
                    ChainSample(layer=int(block_match.group(1)), family=block_match.group(2), magnitude=magnitude)
                )

    try:
        if max_dim is not None:
            arr_f32 = downsample_block_mean(arr_f32, max_dim)

        stats = {
            "min": float(np.min(arr_f32)),
            "max": float(np.max(arr_f32)),
            "mean": float(np.mean(arr_f32)),
            "std": float(np.std(arr_f32)),
        }
        img_rgb = to_diverging_rgb(arr_f32, clip)

        if img_rgb.ndim == 4:
            n_tiles, tile_rows, tile_cols, _ = img_rgb.shape
            tiled = compose_tile_grid(img_rgb)
            write_rgb_png(tiled, out_path)
            stats["tiles"] = {"count": n_tiles, "tile_shape": [tile_rows, tile_cols]}
        else:
            write_rgb_png(img_rgb, out_path)

        logger.info("wrote %s <- %s (%s)", out_path, info.name, info.ggml_type.name)
        return TensorOutcome(
            name=info.name, ggml_type=info.ggml_type.name, shape=info.shape,
            status="written", output_path=str(out_path), stats=stats,
        )
    except Exception as exc:  # noqa: BLE001 - one bad tensor must not kill the run
        logger.warning("error rendering %s: %s", info.name, exc)
        return TensorOutcome(
            name=info.name, ggml_type=info.ggml_type.name, shape=info.shape,
            status="error", reason=str(exc),
        )
    finally:
        del arr_f32


def run(args: argparse.Namespace) -> RunReport:
    output_dir: Path = args.output_dir or Path(f"{args.gguf_path.stem}_weights")

    include_patterns = [re.compile(p) for p in args.include]
    exclude_patterns = [re.compile(p) for p in args.exclude]

    reader = open_reader(args.gguf_path)
    all_infos = list(iter_tensor_infos(reader))

    kept_infos: list[TensorInfo] = []
    outcome_by_name: dict[str, TensorOutcome] = {}
    for info in all_infos:
        if not _matches_filters(info.name, include_patterns, exclude_patterns):
            outcome_by_name[info.name] = TensorOutcome(
                name=info.name, ggml_type=info.ggml_type.name, shape=info.shape,
                status="skipped_filtered",
            )
            continue
        if info.ndim not in (2, 3):
            outcome_by_name[info.name] = TensorOutcome(
                name=info.name, ggml_type=info.ggml_type.name, shape=info.shape,
                status="skipped_unsupported_ndim", reason=f"ndim={info.ndim}",
            )
            continue
        kept_infos.append(info)

    if args.clip_value is not None:
        clip = args.clip_value
        clip_source = "manual"
    else:
        clip = estimate_clip(
            kept_infos,
            sample_tensors=args.sample_tensors,
            sample_elements=args.sample_elements,
            clip_std=args.clip_std,
            rng_seed=args.sample_seed,
        )
        clip_source = "auto_sampled"

    logger.warning("clip value: %.6g (source: %s)", clip, clip_source)

    chain_hidden_size: Optional[int] = None
    chain_samples: list[ChainSample] = []
    if args.chain_report:
        chain_hidden_size = infer_hidden_size(kept_infos)
        if chain_hidden_size is None:
            logger.warning(
                "--chain-report: no blk.N.*.weight tensor found to infer a "
                "residual-stream width from; skipping chain analysis"
            )

    # Cheap, sequential decisions first (dry-run / already-exists), same as
    # before parallelism existed. Only tensors that actually need
    # dequantizing go into to_process, which is what gets parallelized.
    to_process: list[TensorInfo] = []
    out_paths: dict[str, Path] = {}
    for info in kept_infos:
        out_path = output_dir / safe_filename(info.name)
        out_paths[info.name] = out_path

        if args.dry_run:
            outcome_by_name[info.name] = TensorOutcome(
                name=info.name, ggml_type=info.ggml_type.name, shape=info.shape,
                status="planned", output_path=str(out_path),
            )
            logger.info("would write %s <- %s (%s)", out_path, info.name, info.ggml_type.name)
            continue

        if out_path.exists() and not args.overwrite:
            outcome_by_name[info.name] = TensorOutcome(
                name=info.name, ggml_type=info.ggml_type.name, shape=info.shape,
                status="skipped_exists", output_path=str(out_path),
            )
            logger.info("skip (exists) %s", out_path)
            continue

        to_process.append(info)

    if args.jobs <= 1:
        for info in to_process:
            outcome_by_name[info.name] = _process_tensor(
                info, out_paths[info.name], clip, args.max_dim, chain_hidden_size, chain_samples
            )
    else:
        max_memory_bytes = (
            int(args.max_memory_gb * 1024 ** 3)
            if args.max_memory_gb is not None
            else default_memory_budget_bytes()
        )
        results = run_memory_aware(
            to_process,
            lambda info: _process_tensor(
                info, out_paths[info.name], clip, args.max_dim, chain_hidden_size, chain_samples
            ),
            max_workers=args.jobs,
            max_memory_bytes=max_memory_bytes,
        )
        outcome_by_name.update(results)

    # Rebuilt in original GGUF tensor order so the manifest is deterministic
    # and human-browsable regardless of scheduling, whatever args.jobs is.
    outcomes = [outcome_by_name[info.name] for info in all_infos]

    chain_report_path: Optional[str] = None
    chain_heatmap_path: Optional[str] = None
    chain_scatter_path: Optional[str] = None
    chain_scatter_csv_path: Optional[str] = None
    if chain_hidden_size is not None:
        chain_report = build_chain_report(chain_samples, chain_hidden_size)
        if chain_report is None:
            logger.warning("--chain-report: no tensors were written this run; skipping chain analysis")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            heatmap_path = output_dir / "chain_heatmap.png"
            report_path = output_dir / "chain_report.json"
            render_chain_heatmap(chain_report, heatmap_path)
            write_chain_report_json(chain_report, report_path)
            chain_heatmap_path = str(heatmap_path)
            chain_report_path = str(report_path)
            logger.warning(
                "chain report: %s outlier-heavy channels across %d layers -> %s",
                len(chain_report.top_channels), chain_report.n_layers, report_path,
            )

            # A channel that dominates every mid-network matrix can still be
            # scaled to near-nothing by the final RMSNorm before it reaches
            # the logits, so pair the heatmap with what actually survives to
            # the readout -- if the model has the tensor for it.
            gamma_info = next((i for i in all_infos if i.name == "output_norm.weight"), None)
            if gamma_info is None:
                logger.warning("--chain-report: no 'output_norm.weight' tensor found; skipping gamma scatter")
            else:
                try:
                    gamma = dequantize_tensor(gamma_info)
                except DequantError as exc:
                    logger.warning("--chain-report: could not dequantize output_norm.weight: %s", exc)
                    gamma = None
                if gamma is not None and gamma.shape != (chain_hidden_size,):
                    logger.warning(
                        "--chain-report: output_norm.weight shape %s doesn't match inferred "
                        "hidden_size %d; skipping gamma scatter",
                        gamma.shape, chain_hidden_size,
                    )
                elif gamma is not None:
                    scatter_path = output_dir / "chain_scatter.png"
                    scatter_csv_path = output_dir / "chain_scatter.csv"
                    render_chain_scatter(chain_report, gamma, scatter_path)
                    write_chain_scatter_csv(chain_report, gamma, scatter_csv_path)
                    chain_scatter_path = str(scatter_path)
                    chain_scatter_csv_path = str(scatter_csv_path)
                    logger.warning("chain scatter: %s", scatter_path)
                    logger.warning("chain scatter csv: %s", scatter_csv_path)

    report = RunReport(
        input_path=str(args.gguf_path),
        output_dir=str(output_dir),
        clip_value=clip,
        clip_source=clip_source,
        outcomes=outcomes,
        chain_report_path=chain_report_path,
        chain_scatter_path=chain_scatter_path,
        chain_scatter_csv_path=chain_scatter_csv_path,
        chain_heatmap_path=chain_heatmap_path,
    )

    if not args.dry_run and not args.no_manifest:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(report, output_dir / "manifest.json")

    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose:
        level = logging.INFO
    elif args.quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)

    report = run(args)
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
