from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import gguf


@dataclass(frozen=True)
class TensorInfo:
    name: str
    ggml_type: gguf.GGMLQuantizationType
    shape: tuple[int, ...]
    raw: "gguf.ReaderTensor"

    @property
    def ndim(self) -> int:
        return len(self.shape)


def open_reader(path: Path) -> gguf.GGUFReader:
    return gguf.GGUFReader(str(path))


def iter_tensor_infos(reader: gguf.GGUFReader) -> Iterator[TensorInfo]:
    for tensor in reader.tensors:
        # tensor.shape holds ggml's ne[] dims (fastest-varying first); the
        # logical row-major numpy shape is its reverse. This must be used
        # instead of tensor.data.shape, which for quantized types is a
        # block-padded *byte* shape, not the logical element shape (e.g. a
        # (64, 128) Q8_0 tensor reports tensor.data.shape == (64, 136)).
        # gguf.dequantize() independently produces an array in this same
        # logical shape, so the two stay consistent.
        logical_shape = tuple(int(d) for d in reversed(tensor.shape))
        yield TensorInfo(
            name=tensor.name,
            ggml_type=tensor.tensor_type,
            shape=logical_shape,
            raw=tensor,
        )
