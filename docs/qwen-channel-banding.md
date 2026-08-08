# Qwen3.6-27B channel banding: findings and hypotheses

Working notes from a `--chain-report` investigation of two local GGUF renders
(`gemma-4-E4B-it_weights/`, `Qwen3.6-27B-UD-Q2_K_XL_weights/`). Written up so
the next session (or an activation-level probe) doesn't have to re-derive any
of this from scratch.

## 1. Background: what `--chain-report` measures

Every per-layer weight matrix in a transformer has exactly one axis tied to
the model's residual-stream width — its input or output projection
dimension. `spyglass --chain-report`:

1. auto-detects that width (`hidden_size`) as the one dimension shared across
   nearly every `blk.N.*.weight` tensor,
2. computes each channel's mean `|weight|` along that axis, per layer and per
   tensor family, from the **full-resolution dequantized tensor** (not the
   PNG, so it isn't degraded by `--max-dim` downsampling),
3. z-scores each tensor's channel vector against itself, and flags a channel
   as an "outlier" wherever `|z| > 3`,
4. ranks channels by how often (across every `(layer, family)` pair) they're
   flagged — a channel that's an outlier in nearly every layer is a
   "chain": the same coordinate carrying unusually large weight all the way
   through the network's depth.

Output per run: `chain_heatmap.png` (layer × channel, brightness = outlier
strength), `chain_report.json` (ranked channels), and `chain_scatter.png`
(x = outlier-hit rank, y = the model's final-norm `output_norm.weight` gamma
for that channel — i.e. how much of it survives to the output logits after
the last RMSNorm).

Literature this connects to: "massive activations" ([Sun et al.
2024](https://arxiv.org/abs/2402.17762)), "super weights" ([Yu et al.
2024](https://arxiv.org/abs/2411.07191)), outlier features in LLM.int8()
([Dettmers et al. 2022](https://arxiv.org/abs/2208.07339)), and attention
sinks ([Xiao et al. 2023](https://arxiv.org/abs/2309.17453)). See the
project README's "Why this might be interesting" section for the full
writeup — this doc only covers the Qwen-specific follow-up.

## 2. Finding 1 — persistence ≠ readout influence (both models)

Being a persistent chain outlier in the weights does not reliably predict
how much a channel actually reaches the final logits.

**Gemma 3n E4B** (`hidden_size=2560`, 42 layers, 378 `(layer, family)`
cells): the two most persistent channels are **611** (outlier in 94/378 =
24.9% of cells) and **2302** (90/378 = 23.8%). Their final-norm gammas are
wildly different — channel 611: gamma ≈ **4.22** (reaches the logits at
real strength); channel 2302: gamma ≈ **0.005** (scaled to almost nothing
right before the readout, despite dominating nearly a quarter of every
layer's tensors up to that point). `chain_scatter.png` for this model shows
the flagged channels scattered across the *entire* gamma range, from deep
inside the general population's cluster (gamma ≈ 7–9) down to zero.

**Qwen3.6-27B** (`hidden_size=5120`, 65 layers, 504 cells): top channels are
far more persistent than Gemma's — **310** (401/504 = 79.6%), **3986**
(399/504 = 79.2%), **3994** (384/504 = 76.2%), **4939** (366/504 = 72.6%),
**3321** (362/504 = 71.8%). Unlike Gemma, Qwen's flagged channels are *not*
scattered across the whole gamma range — nearly all of them sit in a **tight
band below** the general population's baseline cluster. So in Qwen,
persistence correlates with suppression at the readout; in Gemma it doesn't
correlate with anything in particular. Different models, different
relationship — this by itself was the trigger for looking more closely at
Qwen's scatter, which is where finding 2 turned up.

## 3. Finding 2 — a precise 1024/3072/1024 channel banding in Qwen

`chain_scatter.png` for Qwen shows a visible step in the general
population's baseline gamma partway across the plot, plus a smaller, noisier
one near the start. Investigated directly against the real dequantized
`output_norm.weight` (not the PNG):

| channel-index band | width | mean gamma | std |
|---|---|---|---|
| `[0, 1024)` | 1024 | 1.912 | 0.114 |
| `[1024, 4096)` | 3072 | 1.997 | 0.140 |
| `[4096, 5120)` | 1024 | 1.908 | 0.110 |

Both boundaries land exactly on **1024** and **4096** — confirmed with
256-wide binning (band `[768,1024)` mean 1.919 → band `[1024,1280)` mean
2.009; band `[3840,4096)` mean 1.988 → band `[4096,4352)` mean 1.900), no
drift. Individual channel gammas are noisy (std ≈ 0.12–0.14), but the gap
between region means is ~20 standard errors given the sample sizes — real,
not noise. Fine-grained scans of the first ~300 and last ~220 channels found
no additional sub-structure: the "smaller, noisier" step near the start of
the scatter is the *same* 1024 boundary, just seen through the plot's rank
axis rather than raw channel index. Rank isn't a clean proxy for index near
the low-rank end specifically because that's where the flagged chain
channels (scattered across all three bands, not evenly) get pulled forward
out of index order — of the top 100 flagged channels, **71 fall inside the
middle 3072-wide band** vs. only ~15 in each 1024-wide edge band, so the
local index order gets scrambled most right where the plot starts.

**Cross-checks:**

- `token_embd.weight` per-channel stats across the vocab axis: **no signal
  at all** (mean `|value|` 0.00952 / 0.00963 / 0.00953 across the three
  bands; std 0.01203 / 0.01218 / 0.01203). Rules out anything baked into the
  embedding table's own initialization/statistics.
- A handful of interior block-layer tensors (`blk.2.ssm_out.weight`,
  `blk.2.attn_gate.weight`, `blk.30.ffn_gate.weight`, `blk.30.ffn_down.weight`,
  `blk.60.attn_qkv.weight`) show the **same direction** of effect — middle
  band ~1–2% heavier weighted — in most (not all) of them, much fainter than
  output_norm's ~4.5% gap. Consistent with a real, if faint, network-wide
  bias toward the middle band that gets consolidated into a much cleaner
  signal specifically at the final norm (whose whole job is calibrating
  per-channel importance for the readout).

## 4. Hypotheses, ranked by current confidence

### H1 — Modality specialization (image/video channels). Tested and refuted.

Qwen3.6-27B is confirmed **natively multimodal** (text, image, video) with
its own vision projector — `mmproj-F32.gguf` sits right next to the main
GGUF in the local model folder, so this was the first hypothesis we could
actually test rather than just argue about.

**Sources:** [Qwen3.6-27B overview,
MarkTechPost](https://www.marktechpost.com/2026/04/22/alibaba-qwen-team-releases-qwen3-6-27b-a-dense-open-weight-model-outperforming-397b-moe-on-agentic-coding-benchmarks/)
· [Qwen 3.6 27B, GroqDocs](https://console.groq.com/docs/model/qwen/qwen3.6-27b)

**Probe run:** `probes/qwen_modality_probe.cpp` — a standalone C++ program
against a local `llama.cpp` build (`mtmd` + a custom `ggml_backend_sched_eval_callback`)
that runs one real image+text prompt through the actual model (llama.cpp's
own `tools/mtmd/test-1.jpeg` test asset — a NYT "Men Walk on Moon" front
page — plus the prompt "Describe this newspaper front page in detail,
including the headline text."), captures every layer's residual-stream
output (`l_out-N`, all 64 layers, ne = `[5120, n_tokens]`), and accumulates
per-channel activation statistics split by whether each token position came
from the image chunk (300 tokens) or a text chunk (15 tokens: 1 before the
image, 14 after). Full build/run instructions in `probes/README.md`; raw
output saved at `probes/results/qwen_modality_probe_test-1.json`; analysis
in `probes/analyze_modality_probe.py`.

**Result: no signal.** For each of the three channel bands, we computed a
"modality contrast" score per channel — `|mean_abs_image − mean_abs_text| /
(mean_abs_image + mean_abs_text)` — and averaged it within the band, at 9
layers spread across the model's depth (0, 8, 16, 24, 32, 40, 48, 56, 63).
If the edge bands were vision-specialized, they should show a much higher
image/text contrast than the middle band. They don't, at any depth checked:

| band | mean modality-contrast (layers 0–63) |
|---|---|
| `[0, 1024)` (edge) | 0.2982 |
| `[1024, 4096)` (middle) | 0.2930 |
| `[4096, 5120)` (edge) | 0.2971 |

Image tokens *do* activate every channel more strongly than text tokens
(expected — dense visual content generally produces higher-norm activations
than text, well documented elsewhere), but that elevation is essentially
uniform across all three bands, not concentrated in the edges. **H1 is
ruled out**, at least for this checkpoint under this test: the edge bands
are not disproportionately doing image/video work.

One honest limit: this is one image and one short prompt. It's a clean null
result, not an exhaustive one — a systematic sweep over many images/videos
could still turn up something this single run missed, though the
consistency across all 9 sampled layers (never even close between bands)
makes a hidden effect that only this exact test failed to catch fairly
unlikely.

### H2 — Model growth / width extension (edge bands undertrained). Plausible, untested.

A model built by width-extending a smaller checkpoint (concatenating fresh
capacity onto an existing model and continuing training) commonly leaves
exactly this signature: a "mature" core band with higher learned magnitude
flanked by less-established edge bands. No direct evidence this specific
checkpoint was built that way — nothing found in public
descriptions/searches — but it isn't ruled out either.

**Probe:** (a) activation-richness check on a diverse text-only corpus —
variance/entropy of edge-band channels vs. middle-band channels; undertrained
channels typically look quieter and less input-dependent. (b) An ablation:
zero the edge bands vs. an equal-sized random band elsewhere, compare
perplexity/benchmark degradation. (c) Cheapest of all: check whether Qwen
published training details mentioning a growth/upcycling technique for this
checkpoint (nothing found so far in a quick search — could look more
specifically, e.g. for a technical report rather than news coverage).

### H3 — GQA/attention-structural link. Weakest support, hardest to test.

1024 = `qwen35.attention.head_count_kv (4) × key_length (256)` — the exact
width of one Gated-Attention layer's total K or V projection
(`attn_k.weight`/`attn_v.weight` are both `(1024, 5120)`). Striking
numerical coincidence for a boundary to land on exactly, but no mechanism
identified for why the *residual stream itself* — shared by both the 1-in-4
Gated Attention layers and the 3-in-4 Gated DeltaNet (linear attention)
layers — would inherit that width for a contiguous channel range.

**Probe:** hardest of the three to test directly with activations. Best
paths: (a) Qwen's own architecture/training code or report, if one
surfaces (not found yet); (b) a targeted ablation comparing how much zeroing
the edge bands hurts tasks that stress full-attention layers specifically
vs. tasks that stress the Gated DeltaNet layers specifically — asymmetric
damage would support a real functional link.

### Reference: confirmed architecture details (from local GGUF metadata + public sources)

- `qwen35`: dense (not MoE), 27B params, 65 blocks per this GGUF's
  `block_count` (public descriptions say "64 layers" — reconciles as 64 main
  transformer blocks + 1 MTP layer, matching `nextn_predict_layers=1` and the
  `nextn.eh_proj.weight` tensor, a DeepSeek-V3-style multi-token-prediction
  module).
- Repeating unit: 3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)
  (`full_attention_interval=4`).
- Gated Attention: 24 Q heads, 4 KV heads, head dim 256
  (`qwen35.attention.head_count=24`, `head_count_kv=4`, `key_length=256`).
- Gated DeltaNet (linear attention): 48 V heads, 16 QK heads, head dim 128
  (`qwen35.ssm.time_step_rank=48`, `ssm.group_count=16`, `ssm.inner_size=6144`
  = 48 × 128).
- `embedding_length=5120`, `feed_forward_length=17408`, `context_length=262144`.
- Natively multimodal (text/image/video); local files include a companion
  `mmproj-F32.gguf` vision projector.

## 5. What we haven't done yet

H1 (modality specialization) has now been tested with a real activation
probe and refuted — see section 4. H2 (undertrained edge bands from a
width-extension/growth technique) and H3 (a GQA-structural link) are still
untested, and both need something other than the probe we already built:

- **H2** needs either (a) an activation-richness comparison on a diverse
  *text-only* corpus (does the edge bands' variance/entropy look
  "undertrained" relative to the middle band?), (b) an ablation — zero the
  edge bands vs. an equal-sized random band, compare perplexity/benchmark
  degradation — or (c) just finding Qwen's own training/architecture
  writeup, if one becomes available, and checking whether it mentions a
  growth technique.
- **H3** needs an ablation comparing how much zeroing the edge bands hurts
  tasks that stress the 1-in-4 Gated Attention layers specifically vs. tasks
  that stress the 3-in-4 Gated DeltaNet layers specifically, or Qwen's own
  training code/report turning up a documented reason for the exact
  1024-channel width.

The `qwen_modality_probe.cpp` infrastructure (mtmd + a custom `cb_eval`
capturing named residual-stream tensors, split by an externally-tracked
label) generalizes reasonably directly to an ablation: instead of just
*reading* `l_out-N`, a similar callback could *zero* specific channels in
place before the graph continues, then compare generation quality or
perplexity with vs. without the edge bands zeroed — that's the more
promising next step for H2/H3 over building something new from scratch.
