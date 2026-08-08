from __future__ import annotations

from pathlib import Path

import pytest

from helpers.make_chain_test_gguf import make_chain_test_gguf
from helpers.make_test_gguf import make_test_gguf


@pytest.fixture
def gguf_fixture(tmp_path: Path):
    path = tmp_path / "test.gguf"
    arrays = make_test_gguf(path)
    return path, arrays


@pytest.fixture
def chain_gguf_fixture(tmp_path: Path):
    path = tmp_path / "chain_test.gguf"
    info = make_chain_test_gguf(path)
    return path, info
