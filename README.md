# spyglass

Renders every 2D weight tensor in a GGUF file as a grayscale PNG heatmap.
Pixel `(x, y)` brightness encodes the weight connecting neuron `x` to neuron `y`:
black is the most negative weight, mid-gray is zero, white is the most positive
weight, all scaled against one fixed range shared across every image in a run
so brightness is comparable across layers.

## Install

```sh
pip install -e ".[dev]"
```

## Usage

```sh
spyglass model.gguf                       # writes model_weights/*.png
spyglass model.gguf --dry-run             # list tensors, write nothing
spyglass model.gguf --clip-value 0.25     # fixed, reproducible brightness scale
spyglass model.gguf --exclude 'token_embd\.weight|output\.weight'  # skip huge tensors
spyglass model.gguf --max-dim 2048        # block-average downsample large matrices
spyglass model.gguf --jobs 4              # dequantize/render up to 4 tensors at once
spyglass model.gguf --chain-report        # also trace outlier channels across every layer
```

Run `spyglass --help` for the full flag list.

### Notes

- By default (`--jobs 1`) tensors are dequantized and processed one at a time, so
  peak memory is bounded by the largest single tensor (typically the token
  embedding / output projection matrix), not the whole model file.
- `--jobs N` processes up to N tensors concurrently, bounded by `--max-memory-gb`
  (default: half of detected system RAM) so total in-flight dequantized data stays
  capped. Tensors are admitted largest-fits-first against the remaining budget, so
  smaller tensors opportunistically fill in while a big one waits for room instead
  of stalling behind it. Output is identical to a sequential run either way.
- 2D tensors render as a single grayscale heatmap. 3D tensors (e.g. a stacked
  MoE expert weight, shape `(n_expert, rows, cols)`) render as an RGB PNG: one
  grayscale heatmap tile per slice along axis 0, arranged in a roughly square
  grid and separated by a solid orange gap so each expert's matrix stays
  visually distinct instead of blurring into its neighbors. Tile count and
  per-tile shape are recorded in `manifest.json` under that tensor's `stats.tiles`.
- 1D tensors (layer norms, biases) and tensors with 4+ dimensions are skipped
  and noted in `manifest.json` as `skipped_unsupported_ndim`.
- A pixel's coordinates are `array[y, x]` (or `array[tile, y, x]` for a 3D/tiled
  tensor) where `y` is the tensor's first (outer) numpy dimension after the
  tile axis and `x` is the second — check `manifest.json` or `--dry-run` output
  for each tensor's shape if you need to map a pixel back to a specific
  `(source_neuron, dest_neuron)` pair for a given architecture.

### Chain report (`--chain-report`)

Every per-layer weight matrix has exactly one axis tied to the model's
residual-stream width (its input or output projection dimension) — the model's
width is auto-detected as the one dimension size shared across nearly every
`blk.N.*.weight` tensor, no architecture-specific config needed. `--chain-report`
computes each channel's mean `|weight|` along that axis, per layer and per
tensor family, and z-scores it against its own tensor so magnitude is
comparable across families with very different scales. It then writes:

- `chain_heatmap.png` — layer (row) × channel (column) grid, brightness = mean
  `|z-score|` for that channel in that layer, averaged over every tensor
  family present at that layer. A channel that stays bright top-to-bottom is
  an outlier in nearly every layer: a "chain" running the model's full depth
  (the "massive activation" / rogue-dimension phenomenon reported in LLM
  interpretability work). The bottom strip marks the busiest 1% of channels
  in the tool's gap-orange.
- `chain_report.json` — the channels ranked by how often (across every
  `(layer, family)` pair) their z-score exceeds 3, with a hit rate so runs on
  different models are comparable.
- `chain_scatter.png` — one point per channel: x is its rank in the report
  above (1 = the most persistent chain channel), y is the model's
  final-norm (`output_norm.weight`) gamma for that channel, i.e. how much of
  it actually survives to the output logits after the last RMSNorm. Written
  only when the model has that tensor. A channel dominating every
  mid-network matrix doesn't imply it dominates the readout too — those are
  different things, and this is the chart that shows whether they happen to
  coincide for a given model rather than assuming it from the heatmap alone.
  The same top-1% channels are highlighted in gap-orange here as in the
  heatmap's marker strip.
- `chain_scatter.csv` — the same per-channel data behind `chain_scatter.png`
  (one row per channel: `channel`, `rank`, `outlier_hits`, `gamma`,
  `is_chain_channel`), for re-plotting or filtering outside the fixed PNG.
  Written alongside it, whenever the model has `output_norm.weight`.

Channel magnitudes are computed from the full-resolution dequantized tensor,
before `--max-dim` downsampling, so `--chain-report` keeps single-channel
precision even on a run that downsamples its PNGs. It only covers tensors
actually written during the run, so on a directory that was already rendered,
pair it with `--overwrite` for a complete picture.

#### Why this might be interesting

A channel that's an outlier in nearly every layer means the same residual-stream
coordinate carries unusually large weight all the way through the network,
instead of each layer routing information through its own independent set of
dimensions. That's not noise — it lines up with a few things reported in LLM
interpretability and quantization work:

- **Massive activations.** [Sun et al., 2024](https://arxiv.org/abs/2402.17762)
  found that a handful of hidden-state dimensions in trained LLMs carry
  activation magnitudes ~1000x the median, persist across most of the model's
  depth, and — when zeroed out — collapse the model's output. Weight-magnitude
  outliers are a plausible downstream cause: a channel that every layer's
  input/output projections weight heavily is exactly the kind of dimension
  that would blow up in activation space too.
- **Super weights.** [Yu et al., 2024](https://arxiv.org/abs/2411.07191) went a
  step further and showed the effect can trace back to a tiny number of
  *individual weight coordinates* — pruning a single one is enough to destroy
  the model's ability to generate coherent text, far out of proportion to its
  size. A "chain" is the weight-level shadow of that: a coordinate a
  disproportionate number of layers depend on.
- **Outlier features and quantization.** [Dettmers et al., 2022](https://arxiv.org/abs/2208.07339)
  (LLM.int8()) found that naively quantizing these outlier channels to low
  precision measurably degrades quality, which is why many quantization
  schemes give outlier channels special (higher-precision) treatment instead
  of quantizing everything uniformly.
- **Attention sinks.** Separately, [Xiao et al., 2023](https://arxiv.org/abs/2309.17453)
  found models dump disproportionate attention onto a few fixed positions
  (often the first token) regardless of content. Persistent channels and
  attention sinks are studied at different levels (residual-stream dimension
  vs. sequence position) but both describe the same broader pattern: a small,
  fixed part of the model absorbing a disproportionate share of its capacity.

One honest caveat: this is all inferred from **weights**, not activations —
the papers above mostly measure activations on real input. A channel with
outlier weight magnitude is a good candidate for also having outlier
activations (that's the mechanism the super-weights work describes), but
`--chain-report` doesn't run the model to confirm it. Treat a persistent
channel as a lead worth investigating (e.g. before pruning/quantizing it
aggressively), not a proven activation outlier.

`chain_scatter.png` exists because persistence and readout influence turned
out not to be the same thing in practice: running `--chain-report` on Gemma
3n E4B, its two most persistent channels (outliers in ~25% and ~24% of every
layer's tensors) had final-norm gamma of 4.2 and 0.005 respectively — one
channel actually reaches the logits, the other is scaled almost to nothing
right before the readout despite dominating nearly every layer up to that
point. The general population of channels clusters tightly around gamma ≈
7–9; the flagged chain channels scatter across the *entire* range, from
inside that cluster down to zero. So no, being a persistent chain channel
doesn't reliably predict how much a channel influences the final output —
some chain channels write straight to the logits, others look like they're
doing something purely internal (routing, an attention-sink-style role, a
bias term consumed by later layers) that the final projection ignores
entirely. Confirming *which* would still need activations.
