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
