from __future__ import annotations

from pathlib import Path

import gguf
import numpy as np


def make_test_gguf(path: Path, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Writes a small real .gguf file with a known mix of tensor shapes/types
    and returns the original float arrays keyed by tensor name, for assertions.
    """
    rng = np.random.default_rng(seed)

    float_arr = rng.normal(0, 0.05, size=(48, 96)).astype(np.float32)
    quant_src = rng.normal(0, 0.05, size=(64, 128)).astype(np.float32)
    norm_arr = rng.normal(0, 0.05, size=(32,)).astype(np.float32)
    moe_arr = rng.normal(0, 0.05, size=(4, 8, 16)).astype(np.float32)

    writer = gguf.GGUFWriter(str(path), arch="spyglass-test")
    writer.add_block_count(1)

    writer.add_tensor("float_tensor.weight", float_arr)

    quant_bytes = gguf.quantize(quant_src, gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor(
        "quant_tensor.weight", quant_bytes, raw_dtype=gguf.GGMLQuantizationType.Q8_0
    )

    writer.add_tensor("norm_tensor.weight", norm_arr)
    writer.add_tensor("moe_tensor.weight", moe_arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "float_tensor.weight": float_arr,
        "quant_tensor.weight": quant_src,
        "norm_tensor.weight": norm_arr,
        "moe_tensor.weight": moe_arr,
    }
