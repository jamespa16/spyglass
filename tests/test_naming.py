from spyglass.naming import safe_filename


def test_preserves_dots():
    assert safe_filename("blk.0.attn_q.weight") == "blk.0.attn_q.weight.png"


def test_replaces_unsafe_characters():
    assert safe_filename("weird/name:here?") == "weird_name_here_.png"


def test_empty_name_falls_back():
    assert safe_filename("") == "tensor.png"


def test_long_name_is_truncated_with_hash_suffix():
    long_name = "blk." + "x" * 400 + ".weight"
    result = safe_filename(long_name)
    assert result.endswith(".png")
    assert len(result.encode("utf-8")) < len(long_name.encode("utf-8"))
    # truncation is deterministic for the same input
    assert safe_filename(long_name) == result
