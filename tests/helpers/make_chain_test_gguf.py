from __future__ import annotations

from pathlib import Path

import gguf
import numpy as np

HIDDEN_SIZE = 32
OUTLIER_CHANNEL = 17
N_LAYERS = 3
# Deliberately different sizes per family (as in a real model, e.g. attn_q's
# head_count*head_dim vs ffn_down's intermediate size) so HIDDEN_SIZE is the
# only dimension shared by every family -- otherwise it can tie with
# whichever "other" axis size happens to be reused across families too.
ATTN_OTHER_DIM = 16
FFN_OTHER_DIM = 20
OUTLIER_SCALE = 40.0
# Mirrors a real finding (see README): a channel can dominate every
# mid-network matrix and still be scaled to near-nothing by the final
# RMSNorm before it reaches the logits.
OUTLIER_GAMMA = 0.01


def make_chain_test_gguf(path: Path, *, seed: int = 0) -> dict:
    """Writes a small real .gguf file shaped like a tiny transformer: N_LAYERS
    blocks, each with an input-side family ("attn_q.weight", hidden on the
    column axis) and an output-side family ("ffn_down.weight", hidden on the
    row axis), plus a final-norm gamma and one non-block tensor that should
    be ignored by chain analysis.

    OUTLIER_CHANNEL is deliberately scaled up in every block tensor so a
    chain report should single it out as the top hit in every layer, and its
    final-norm gamma is deliberately tiny relative to the rest.
    """
    rng = np.random.default_rng(seed)
    writer = gguf.GGUFWriter(str(path), arch="spyglass-chain-test")
    writer.add_block_count(N_LAYERS)

    for layer in range(N_LAYERS):
        attn_q = rng.normal(0, 0.02, size=(ATTN_OTHER_DIM, HIDDEN_SIZE)).astype(np.float32)
        attn_q[:, OUTLIER_CHANNEL] *= OUTLIER_SCALE
        writer.add_tensor(f"blk.{layer}.attn_q.weight", attn_q)

        ffn_down = rng.normal(0, 0.02, size=(HIDDEN_SIZE, FFN_OTHER_DIM)).astype(np.float32)
        ffn_down[OUTLIER_CHANNEL, :] *= OUTLIER_SCALE
        writer.add_tensor(f"blk.{layer}.ffn_down.weight", ffn_down)

    output_norm = np.ones(HIDDEN_SIZE, dtype=np.float32)
    output_norm[OUTLIER_CHANNEL] = OUTLIER_GAMMA
    writer.add_tensor("output_norm.weight", output_norm)

    # Unrelated to the residual stream (neither axis is HIDDEN_SIZE) --
    # should be skipped by chain analysis without affecting hidden_size
    # inference.
    unrelated = rng.normal(0, 0.02, size=(5, 5)).astype(np.float32)
    writer.add_tensor("blk.0.ssm_conv1d.weight", unrelated)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "hidden_size": HIDDEN_SIZE,
        "outlier_channel": OUTLIER_CHANNEL,
        "n_layers": N_LAYERS,
        "outlier_gamma": OUTLIER_GAMMA,
    }
