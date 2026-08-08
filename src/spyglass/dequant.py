from __future__ import annotations

import numpy as np
import gguf

from spyglass.reading import TensorInfo


class DequantError(RuntimeError):
    def __init__(self, tensor_name: str, ggml_type: gguf.GGMLQuantizationType, cause: Exception):
        super().__init__(f"{tensor_name} ({ggml_type.name}): {cause}")
        self.tensor_name = tensor_name
        self.ggml_type = ggml_type
        self.cause = cause


def dequantize_tensor(info: TensorInfo) -> np.ndarray:
    try:
        arr = gguf.dequantize(info.raw.data, info.ggml_type)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised typed
        raise DequantError(info.name, info.ggml_type, exc) from exc
    return np.asarray(arr, dtype=np.float32)
