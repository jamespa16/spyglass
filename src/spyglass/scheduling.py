from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import subprocess
import sys
import threading
from typing import Callable, Optional, Sequence, TypeVar

from spyglass.reading import TensorInfo

logger = logging.getLogger("spyglass")

T = TypeVar("T")

DEFAULT_MEMORY_FRACTION = 0.5
FALLBACK_MEMORY_BYTES = 4 * 1024 ** 3  # used only if system memory can't be detected


def estimate_tensor_bytes(info: TensorInfo) -> int:
    """Peak float32 size of the dequantized tensor, in bytes.

    This is the size that matters for memory budgeting: --max-dim downsampling
    happens on the already-dequantized float32 array, so it doesn't reduce
    this peak.
    """
    return math.prod(info.shape) * 4


def detect_total_memory_bytes() -> Optional[int]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            return int(out.stdout.strip())
        except Exception:
            pass
    return None


def default_memory_budget_bytes() -> int:
    total = detect_total_memory_bytes()
    if total is None:
        logger.warning(
            "could not detect system memory; defaulting to a %.0fGiB budget "
            "(override with --max-memory-gb)",
            FALLBACK_MEMORY_BYTES / 1024 ** 3,
        )
        return FALLBACK_MEMORY_BYTES
    return int(total * DEFAULT_MEMORY_FRACTION)


def run_memory_aware(
    infos: Sequence[TensorInfo],
    worker_fn: Callable[[TensorInfo], T],
    *,
    max_workers: int,
    max_memory_bytes: int,
) -> dict[str, T]:
    """Runs worker_fn(info) for every info, bounded by a worker-count cap and
    a total-in-flight-bytes budget.

    Pending tasks are admitted largest-fits-first against the remaining
    budget: whenever a slot is free, the biggest pending tensor that fits is
    dispatched, and if none fit, smaller ones are tried so they can
    opportunistically fill in while a big tensor waits for room, rather than
    blocking behind it in FIFO order. If a single tensor's estimated size
    exceeds the entire budget, it is run alone once nothing else is in
    flight, rather than deadlocking or being silently skipped.

    worker_fn is expected not to raise (report failures via its return value
    instead, as _process_tensor does) but this is enforced defensively: if it
    raises anyway, that task's memory/slot is still released so the
    scheduler can't hang, and the first such exception is re-raised from
    this function once every in-flight task has finished.
    """
    pending = sorted(infos, key=estimate_tensor_bytes, reverse=True)
    results: dict[str, T] = {}
    first_exception: list[BaseException] = []

    cond = threading.Condition()
    used_bytes = 0
    in_flight = 0

    def on_done(info: TensorInfo, size: int, future: "concurrent.futures.Future[T]") -> None:
        nonlocal used_bytes, in_flight
        try:
            results[info.name] = future.result()
        except BaseException as exc:  # noqa: BLE001 - must not prevent slot release below
            logger.error("worker_fn raised for %s: %s", info.name, exc)
            first_exception.append(exc)
        with cond:
            used_bytes -= size
            in_flight -= 1
            cond.notify_all()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with cond:
            while (pending or in_flight > 0) and not first_exception:
                idx = None
                if in_flight < max_workers:
                    remaining = max_memory_bytes - used_bytes
                    for i, info in enumerate(pending):
                        if estimate_tensor_bytes(info) <= remaining:
                            idx = i
                            break
                    if idx is None and in_flight == 0 and pending:
                        # Nothing fits and nothing is running: force the
                        # largest pending tensor through alone so the run
                        # can make progress instead of deadlocking.
                        idx = 0
                        oversized = pending[0]
                        logger.warning(
                            "%s (%.2f GiB dequantized) exceeds the %.2f GiB "
                            "memory budget; running it alone",
                            oversized.name,
                            estimate_tensor_bytes(oversized) / 1024 ** 3,
                            max_memory_bytes / 1024 ** 3,
                        )

                if idx is None:
                    cond.wait()
                    continue

                info = pending.pop(idx)
                size = estimate_tensor_bytes(info)
                used_bytes += size
                in_flight += 1
                future = executor.submit(worker_fn, info)
                future.add_done_callback(
                    lambda f, info=info, size=size: on_done(info, size, f)
                )
        # exiting the executor's `with` block waits for whatever was already
        # submitted (including the task that raised) to finish before we get
        # here, so on_done has already run for all of them.

    if first_exception:
        raise first_exception[0]

    return results
