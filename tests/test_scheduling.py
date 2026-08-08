from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from spyglass.scheduling import estimate_tensor_bytes, run_memory_aware


@dataclass(frozen=True)
class FakeGgmlType:
    name: str = "F32"


@dataclass(frozen=True)
class FakeTensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: FakeGgmlType = FakeGgmlType()


def test_estimate_tensor_bytes_is_float32_element_count():
    info = FakeTensorInfo("t", (100, 200))
    assert estimate_tensor_bytes(info) == 100 * 200 * 4


def test_all_tasks_complete_and_results_keyed_by_name():
    infos = [FakeTensorInfo(f"t{i}", (10, 10)) for i in range(20)]

    def worker(info):
        return info.name.upper()

    results = run_memory_aware(
        infos, worker, max_workers=4, max_memory_bytes=10 * 1024 * 1024
    )
    assert set(results) == {info.name for info in infos}
    for info in infos:
        assert results[info.name] == info.name.upper()


def test_respects_worker_count_cap():
    infos = [FakeTensorInfo(f"t{i}", (10, 10)) for i in range(10)]
    concurrent = 0
    max_concurrent = 0
    lock = threading.Lock()

    def worker(info):
        nonlocal concurrent, max_concurrent
        with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.02)
        with lock:
            concurrent -= 1
        return info.name

    run_memory_aware(infos, worker, max_workers=3, max_memory_bytes=10**12)
    assert max_concurrent <= 3
    assert max_concurrent > 1  # actually ran concurrently, not accidentally serial


def test_respects_memory_budget():
    # Each task claims ~1 unit; budget only allows 2 in flight at once.
    unit = 1024
    infos = [FakeTensorInfo(f"t{i}", (16, 16)) for i in range(8)]  # 16*16*4 = 1024 bytes
    assert estimate_tensor_bytes(infos[0]) == unit

    concurrent = 0
    max_concurrent = 0
    lock = threading.Lock()

    def worker(info):
        nonlocal concurrent, max_concurrent
        with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.02)
        with lock:
            concurrent -= 1
        return info.name

    run_memory_aware(infos, worker, max_workers=8, max_memory_bytes=unit * 2)
    assert max_concurrent <= 2


def test_small_tasks_fill_in_while_oversized_task_waits():
    # One huge task that alone exceeds the entire budget, plus several small
    # tasks that comfortably fit together within it. The huge task can only
    # be admitted through the "nothing fits, nothing in flight" fallback, so
    # it must wait for every small task to finish first rather than
    # blocking them behind it in submission order (pending is sorted
    # largest-first, so naive FIFO-behind-the-front-item behavior would
    # wrongly put huge first).
    huge = FakeTensorInfo("huge", (2000, 2000))  # 2000*2000*4 = 16,000,000 bytes
    small = [FakeTensorInfo(f"small{i}", (10, 10)) for i in range(5)]  # 400 bytes each
    budget = 2000  # fits all 5 small tasks at once (5*400); huge never fits alongside anything

    order = []
    order_lock = threading.Lock()

    def worker(info):
        if info.name != "huge":
            time.sleep(0.05)
        with order_lock:
            order.append(info.name)
        return info.name

    infos = [huge, *small]
    results = run_memory_aware(
        infos, worker, max_workers=len(infos), max_memory_bytes=budget
    )

    assert set(results) == {info.name for info in infos}
    assert order[-1] == "huge"


def test_worker_exception_does_not_hang_scheduler():
    infos = [FakeTensorInfo(f"t{i}", (10, 10)) for i in range(5)]

    def worker(info):
        if info.name == "t2":
            raise ValueError("simulated failure")
        return "ok"

    with pytest.raises(ValueError):
        run_memory_aware(infos, worker, max_workers=2, max_memory_bytes=10**9)
