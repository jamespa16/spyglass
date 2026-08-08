from __future__ import annotations

import hashlib
import re

_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_MAX_STEM_BYTES = 200  # leave room for suffix + hash, well under 255-byte fs limits


def safe_filename(tensor_name: str, suffix: str = ".png") -> str:
    stem = _UNSAFE_CHARS.sub("_", tensor_name).strip()
    if not stem:
        stem = "tensor"

    if len(stem.encode("utf-8")) > _MAX_STEM_BYTES:
        digest = hashlib.sha1(tensor_name.encode("utf-8")).hexdigest()[:8]
        truncated = stem.encode("utf-8")[: _MAX_STEM_BYTES - len(digest) - 1].decode(
            "utf-8", errors="ignore"
        )
        stem = f"{truncated}_{digest}"

    return f"{stem}{suffix}"
