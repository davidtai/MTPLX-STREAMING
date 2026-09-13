"""W121: the gpu_window guard's phys_footprint tree reader (scripts/deepseek_v41/
tree_footprint.py).  Read-only, CPU-only, no MLX, no model, no GPU.

phys_footprint (proc_pid_rusage RUSAGE_INFO_V4) is the per-process total that
INCLUDES Metal/IOAccelerator pages -- the quantity the guard sums over the step
tree instead of ps RSS (which undercounts unified Metal) or a vm_stat bucket
(which cannot separate GPU memory from the reclaimable file cache).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

_TF_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "deepseek_v41" / "tree_footprint.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("tree_footprint_w121", _TF_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tf = _load()


def test_phys_footprint_self_is_positive():
    fp = tf.phys_footprint(os.getpid())
    assert fp is not None
    assert fp > 8 * 1024 * 1024  # this interpreter is at least a few MiB


def test_phys_footprint_missing_pid_is_none():
    # A pid that (almost certainly) does not exist -> None, and tree sum -> 0.
    assert tf.phys_footprint(2_000_000_000) is None
    assert tf.tree_footprint([2_000_000_000]) == 0


def test_tree_pids_includes_root_and_children():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # Give the child a moment to appear in the ps snapshot.
        for _ in range(20):
            pids = tf.tree_pids([os.getpid()])
            if child.pid in pids:
                break
            time.sleep(0.1)
        pids = tf.tree_pids([os.getpid()])
        assert os.getpid() in pids
        assert child.pid in pids  # the descendant is walked from the ps snapshot
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_tree_footprint_at_least_self_and_grows_with_child():
    self_only = tf.phys_footprint(os.getpid())
    # A child that actually touches ~200 MiB so its footprint is non-trivial.
    code = (
        "import time;"
        "b=bytearray(200*1024*1024);"
        "b[::4096]=b'\\x01'*len(b[::4096]);"
        "time.sleep(30)"
    )
    child = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.time() + 10
        tree = 0
        while time.time() < deadline:
            if child.pid in tf.tree_pids([os.getpid()]):
                tree = tf.tree_footprint([os.getpid()])
                if tree >= self_only + 100 * 1024 * 1024:
                    break
            time.sleep(0.2)
        # The tree sum includes the child's ~200 MiB on top of this process.
        assert tree >= self_only
        assert tree >= self_only + 100 * 1024 * 1024
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_cli_pid_mode_matches_function():
    out = subprocess.run(
        [sys.executable, str(_TF_PATH), "--pid", str(os.getpid())],
        capture_output=True, text=True, timeout=15,
    )
    assert out.returncode == 0
    cli = int(out.stdout.strip())
    # Same process, sampled microseconds apart -> within a small tolerance.
    assert abs(cli - tf.phys_footprint(os.getpid())) < 64 * 1024 * 1024


def test_cli_tree_sum_is_positive_integer():
    out = subprocess.run(
        [sys.executable, str(_TF_PATH), str(os.getpid())],
        capture_output=True, text=True, timeout=15,
    )
    assert out.returncode == 0
    assert int(out.stdout.strip()) > 0


def test_cli_tree_root_unreadable_exits_2():
    # MEDIUM-2: a bogus/unreadable ROOT pid must FAIL CLOSED (exit 2), not print 0 rc 0 --
    # the gpu_window guard treats rc != 0 as "footprint reader broke" and aborts, so an
    # unreadable step root can never silently read as box_used == baseline.
    out = subprocess.run(
        [sys.executable, str(_TF_PATH), "2000000000"],
        capture_output=True, text=True, timeout=15,
    )
    assert out.returncode == 2, f"expected exit 2 for an unreadable root, got {out.returncode}: {out.stdout!r}"
