"""CPU tests for the F19 read-order pool swap and its staged packed_phase edit (no MLX)."""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "deepseek_v41"))

from f2 import read_order  # noqa: E402
from f2 import stage_f2_runner as stager  # noqa: E402

_PACKED_PHASE = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed/packed_phase.py"


class _Reader:
    def __init__(self, workers=15):
        self._fanout_pool_workers = workers
        self._fanout_executor = ThreadPoolExecutor(max_workers=workers)


def test_install_swaps_the_pool_and_shuts_the_old_one_down():
    reader = _Reader()
    old = reader._fanout_executor
    report = read_order.install(reader, workers=3)
    assert report == {"installed": True, "workers_before": 15, "workers": 3}
    assert reader._fanout_executor is not old and reader._fanout_executor._max_workers == 3
    with pytest.raises(RuntimeError):
        old.submit(lambda: None)          # old pool is shut down
    reader._fanout_executor.shutdown(wait=True)


def test_small_pool_serves_in_submission_order():
    reader = _Reader()
    read_order.install(reader, workers=2)
    done, lock = [], threading.Lock()

    def job(tag):
        time.sleep(0.01)
        with lock:
            done.append(tag)

    futures = [reader._fanout_executor.submit(job, ("A", i)) for i in range(6)]
    futures += [reader._fanout_executor.submit(job, ("B", i)) for i in range(6)]
    for f in futures:
        f.result()
    # every A job was submitted before every B job; with 2 workers at most one B may overtake the last A pair
    last_a = max(i for i, t in enumerate(done) if t[0] == "A")
    first_b = min(i for i, t in enumerate(done) if t[0] == "B")
    assert first_b >= last_a - 1
    reader._fanout_executor.shutdown(wait=True)


@pytest.mark.parametrize("bad", [0, 65, True, 2.5, "3"])
def test_bad_worker_counts_are_refused(bad):
    with pytest.raises(RuntimeError):
        read_order.install(_Reader(), workers=bad)


def test_reader_without_a_fanout_pool_is_refused():
    reader = _Reader()
    reader._fanout_executor.shutdown(wait=True)
    reader._fanout_executor = None
    with pytest.raises(RuntimeError):
        read_order.install(reader, workers=3)


def test_env_unset_is_a_noop(monkeypatch):
    monkeypatch.delenv(read_order.ENV, raising=False)
    reader = _Reader()
    old = reader._fanout_executor
    assert read_order.install_from_env(reader) == {"installed": False}
    assert reader._fanout_executor is old
    old.shutdown(wait=True)


def test_staged_edit_roundtrips_on_the_archived_packed_phase_and_refuses_double_apply():
    src = _PACKED_PHASE.read_text()
    staged = stager.stage_read_order(src)
    lines = [ln for ln in staged.splitlines() if "_f19_read_order" not in ln]
    assert "\n".join(lines) + ("\n" if src.endswith("\n") else "") == src
    idx = staged.splitlines().index(stager._RO_INSERT)
    assert staged.splitlines()[idx + 1] == stager._RO_ANCHOR   # immediately before the lane import
    compile(staged, "packed_phase_staged", "exec")
    with pytest.raises(RuntimeError):
        stager.stage_read_order(staged)
