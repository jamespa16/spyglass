import numpy as np
import pytest

from spyglass.dequant import DequantError, dequantize_tensor
from spyglass.reading import iter_tensor_infos, open_reader


def _infos_by_name(path):
    reader = open_reader(path)
    return {info.name: info for info in iter_tensor_infos(reader)}


def test_dequantize_float_tensor_matches_original(gguf_fixture):
    path, arrays = gguf_fixture
    infos = _infos_by_name(path)
    result = dequantize_tensor(infos["float_tensor.weight"])
    assert result.shape == arrays["float_tensor.weight"].shape
    assert np.allclose(result, arrays["float_tensor.weight"], atol=1e-6)


def test_dequantize_quantized_tensor_close_to_original(gguf_fixture):
    path, arrays = gguf_fixture
    infos = _infos_by_name(path)
    result = dequantize_tensor(infos["quant_tensor.weight"])
    original = arrays["quant_tensor.weight"]
    assert result.shape == original.shape
    # Q8_0 is lossy; a loose tolerance confirms it round-tripped, not that
    # it's exact.
    assert np.max(np.abs(result - original)) < 0.01


def test_dequantize_unsupported_type_raises_dequant_error(gguf_fixture, monkeypatch):
    path, _ = gguf_fixture
    infos = _infos_by_name(path)
    info = infos["float_tensor.weight"]

    import gguf as gguf_module

    def _boom(data, qtype):
        raise NotImplementedError("simulated unsupported quant type")

    monkeypatch.setattr(gguf_module, "dequantize", _boom)
    with pytest.raises(DequantError):
        dequantize_tensor(info)
