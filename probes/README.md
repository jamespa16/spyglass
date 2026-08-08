# probes

Activation-level probes that go beyond what spyglass itself does (spyglass
only ever reads static GGUF weights; it never runs a model). These require a
built [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout with `mtmd`
support and are not part of spyglass's own build — see
`../docs/qwen-channel-banding.md` for the investigation these came out of.

## qwen_modality_probe.cpp

Tests whether Qwen3.6-27B's residual-stream channel banding
(`[0,1024)` / `[1024,4096)` / `[4096,5120)`, found via `spyglass --chain-report`)
corresponds to image/video-token vs. text-token specialization. Runs one
image+text prompt through the real model via `mtmd`, and for every layer's
residual-stream output (`l_out-N`) plus the final post-`output_norm` hidden
state (`result_norm`, when reachable — see caveat below), accumulates
per-channel activation statistics (mean, mean`|.|`, std) split by whether the
token position came from the image chunk or a text chunk.

### Build

Needs a llama.cpp checkout built with `mtmd` support (this repo assumes one
exists at `../llama.cpp` with `build/bin/` populated):

```sh
LLAMA=../llama.cpp
LIBDIR="$LLAMA/build/bin"
clang++ -std=c++17 -O2 \
  -I "$LLAMA/include" -I "$LLAMA/ggml/include" -I "$LLAMA/common" -I "$LLAMA/tools/mtmd" -I "$LLAMA/vendor" \
  -c qwen_modality_probe.cpp -o qwen_modality_probe.o
clang++ -std=c++17 -O2 qwen_modality_probe.o \
  -L "$LIBDIR" -Wl,-rpath,"$LIBDIR" \
  -lllama -lmtmd -lggml -lggml-base -lggml-cpu -lllama-common \
  -o qwen_modality_probe
```

### Run

```sh
PROBE_OUT=probe_output.json ./qwen_modality_probe \
  -m /path/to/Qwen3.6-27B-UD-Q2_K_XL.gguf \
  --mmproj /path/to/mmproj-F32.gguf \
  --image /path/to/image.jpg \
  -ngl 99 -c 4096
```

`-p "..."` overrides the default prompt; include `<__media__>` if you want
the image placed somewhere other than the start. Any image works; the
original run used llama.cpp's own `tools/mtmd/test-1.jpeg` test asset.

### Analyze

```sh
python3 analyze_modality_probe.py probe_output.json
```

Prints, per sampled layer and per channel band, the mean activation
magnitude for text vs. image token positions and a normalized "modality
contrast" score. See `../docs/qwen-channel-banding.md` for the actual
result (short version: the three bands show statistically indistinguishable
contrast — the modality-specialization hypothesis didn't hold up).

### Caveat: `result_norm` is usually empty

`result_norm` is gated by `inp_out_ids` (llama.cpp only computes it for
positions that were asked for logits). The convenience helpers used here
(`mtmd_helper_eval_chunk_single`, `mtmd_helper_decode_image_chunk`) don't
expose a way to request logits at every position, only the last one — so in
practice `result_norm` ends up empty unless you replace those helpers with
manual `llama_batch` construction that sets `logits[i] = true` for every
position. The probe still gets everything it needs from `l_out-N`, which is
uncensored at every layer and every position.
