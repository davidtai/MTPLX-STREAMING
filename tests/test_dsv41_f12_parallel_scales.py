"""CPU tests for the DSV4.1 F12 parallel growth-transition scales loader.

Pins MLX to CPU (a "no GPU" worker task must not touch Metal). Compares the
parallel loader byte-for-byte against the retained serial ``packed_storage.load_layer``
on a small synthetic artifact, exercises every loud-failure path through
``finish()``, and checks the stager (round-trip on the real archived
``packed_phase.py``, env-off == original behavior, double-apply refused).
"""
import errno
import hashlib
import os
import sys
from pathlib import Path

import mlx.core as mx
mx.set_default_device(mx.cpu)

import pytest

# --- locate the sources under test (worktree-relative, absolute) ---------------
_ROOT = Path(__file__).resolve().parents[1]
_PACKED = _ROOT / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
_F12 = _ROOT / "scripts/deepseek_v41/f12"
for _p in (str(_PACKED), str(_F12)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import packed_storage          # retained serial loader (source of truth)
import parallel_scales         # F12 parallel loader under test
import stage_f12_runner        # F12 stager under test

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_FIELDS = ("descriptors", "payload", "bases")
# Small, distinct per-file shapes (uint32); byte counts stay tiny.
_SHAPES = {
    "gate_proj": {"descriptors": (2, 3), "payload": (7,), "bases": (4,)},
    "up_proj":   {"descriptors": (3, 2), "payload": (5,), "bases": (4,)},
    "down_proj": {"descriptors": (2, 5), "payload": (9,), "bases": (4,)},
}


@pytest.fixture(autouse=True)
def _drain_pending():
    """Never leak submitted futures across tests."""
    yield
    try:
        parallel_scales.finish()
    except BaseException:
        pass


def _build_artifact(tmp: Path):
    """Write a one-layer synthetic artifact; return (entry, {(proj,field): bytes})."""
    entry = {"components": {}}
    contents = {}
    for proj in _PROJECTIONS:
        comp = {}
        for field in _FIELDS:
            shape = _SHAPES[proj][field]
            nelem = 1
            for d in shape:
                nelem *= d
            nbytes = nelem * 4  # uint32
            data = os.urandom(nbytes)
            fname = f"{proj}-{field}.bin"
            (tmp / fname).write_bytes(data)
            comp[field] = {"shape": list(shape), "file": fname, "bytes": nbytes,
                           "sha256": hashlib.sha256(data).hexdigest(), "dtype": "uint32"}
            contents[(proj, field)] = data
        entry["components"][proj] = comp
    return entry, contents


def _as_bytes(array) -> bytes:
    mv = memoryview(array).cast("B")
    try:
        return bytes(mv)
    finally:
        mv.release()


# --- byte-identical parity ------------------------------------------------------
def test_parallel_matches_serial_byte_for_byte(tmp_path):
    entry, contents = _build_artifact(tmp_path)
    serial = packed_storage.load_layer(tmp_path, entry, mx=mx)
    par = parallel_scales.load_layer(tmp_path, entry, mx=mx)
    parallel_scales.finish()  # reads/hashes complete here

    assert set(par) == set(serial) == set(_PROJECTIONS)
    for proj in _PROJECTIONS:
        assert len(par[proj]) == len(serial[proj]) == len(_FIELDS)
        for field, a_ser, a_par in zip(_FIELDS, serial[proj], par[proj]):
            b_ser, b_par = _as_bytes(a_ser), _as_bytes(a_par)
            assert b_par == b_ser, f"{proj}.{field} parallel != serial"
            assert b_par == contents[(proj, field)], f"{proj}.{field} != file bytes"
            assert a_par.dtype == mx.uint32


def test_finish_is_noop_when_nothing_pending():
    parallel_scales.finish()
    parallel_scales.finish()  # idempotent, no error


# --- loud failures all surface through finish() ---------------------------------
def test_digest_mismatch_raises(tmp_path):
    entry, _ = _build_artifact(tmp_path)
    entry["components"]["up_proj"]["payload"]["sha256"] = "0" * 64  # correct size, wrong digest
    parallel_scales.load_layer(tmp_path, entry, mx=mx)
    with pytest.raises(RuntimeError, match="digest differs"):
        parallel_scales.finish()


def test_short_file_raises(tmp_path):
    entry, _ = _build_artifact(tmp_path)
    bad = tmp_path / entry["components"]["gate_proj"]["descriptors"]["file"]
    bad.write_bytes(b"\x00" * 4)  # far short of the inventoried byte count
    parallel_scales.load_layer(tmp_path, entry, mx=mx)
    with pytest.raises(RuntimeError, match="size differs"):
        parallel_scales.finish()


def test_missing_file_raises(tmp_path):
    entry, _ = _build_artifact(tmp_path)
    entry["components"]["down_proj"]["bases"]["file"] = "does-not-exist.bin"
    parallel_scales.load_layer(tmp_path, entry, mx=mx)
    with pytest.raises(FileNotFoundError):
        parallel_scales.finish()


def test_symlinked_file_refused_by_nofollow(tmp_path):
    entry, _ = _build_artifact(tmp_path)
    target = tmp_path / entry["components"]["gate_proj"]["payload"]["file"]
    link = tmp_path / "gate_proj-payload-symlink.bin"
    os.symlink(target, link)
    entry["components"]["gate_proj"]["payload"]["file"] = link.name
    parallel_scales.load_layer(tmp_path, entry, mx=mx)
    with pytest.raises(OSError) as ei:
        parallel_scales.finish()
    assert ei.value.errno == errno.ELOOP  # O_NOFOLLOW refused the symlink, not ENOENT


def test_failure_still_joins_every_worker(tmp_path):
    """All-or-nothing: after a failure, no future is left pending/running."""
    entry, _ = _build_artifact(tmp_path)
    entry["components"]["up_proj"]["descriptors"]["sha256"] = "f" * 64
    parallel_scales.load_layer(tmp_path, entry, mx=mx)
    with pytest.raises(RuntimeError):
        parallel_scales.finish()
    assert parallel_scales._pending == []  # drained even on failure


# --- routing: env-off == original serial behavior -------------------------------
def test_resolve_env_off_is_original_serial(monkeypatch):
    monkeypatch.delenv(parallel_scales._ENV_ENABLE, raising=False)
    sentinel = object()
    load, fin = parallel_scales.resolve(sentinel)
    assert load is sentinel            # the caller's ORIGINAL serial load_layer
    assert fin() is None               # no-op finish


def test_resolve_env_on_is_parallel(monkeypatch):
    monkeypatch.setenv(parallel_scales._ENV_ENABLE, "1")
    load, fin = parallel_scales.resolve(object())
    assert load is parallel_scales.load_layer
    assert fin is parallel_scales.finish


def test_resolve_env_zero_is_serial(monkeypatch):
    monkeypatch.setenv(parallel_scales._ENV_ENABLE, "0")
    sentinel = object()
    load, fin = parallel_scales.resolve(sentinel)
    assert load is sentinel and fin() is None


def test_bad_worker_count_raises_loudly(monkeypatch):
    monkeypatch.setattr(parallel_scales, "_executor", None)  # force re-create
    monkeypatch.setenv(parallel_scales._ENV_WORKERS, "0")
    with pytest.raises(RuntimeError, match="must be >= 1"):
        parallel_scales._get_executor()
    monkeypatch.setattr(parallel_scales, "_executor", None)


# --- stager on the real archived packed_phase.py --------------------------------
def _real_packed_phase() -> str:
    return (_PACKED / "packed_phase.py").read_text()


def test_stager_round_trip_and_compiles():
    src = _real_packed_phase()
    out = stage_f12_runner.stage(src)  # round-trip asserted inside stage()
    assert out != src
    # independent reverse recovers the byte-for-byte original
    rec = out.replace(stage_f12_runner._A3_NEW, stage_f12_runner._A3)
    rec = rec.replace(stage_f12_runner._A2_NEW, stage_f12_runner._A2)
    rec = rec.replace(stage_f12_runner._I1 + "\n" + stage_f12_runner._A1, stage_f12_runner._A1)
    assert rec == src
    compile(out, "packed_phase_staged.py", "exec")  # staged output is valid Python


def test_stager_edits_are_the_expected_three():
    out = stage_f12_runner.stage(_real_packed_phase())
    assert "_f12_load_layer, _f12_finish = _f12.resolve(load_layer)" in out
    assert "owners[layer] = _f12_load_layer(ROOT / 'artifact'" in out
    assert "owners[layer] = load_layer(ROOT / 'artifact'" not in out  # loop call rerouted
    # finish() lands after the loop and before the first post-loop consumer
    join = out.index("_f12_finish()  # F12 join")
    forphys = out.index("                for physical in (*pool._persistent.values(), *pool._transient):",
                        out.index("for layer, switch in zip(layers, switches):"))
    assert join < forphys


def test_stager_refuses_double_apply():
    out = stage_f12_runner.stage(_real_packed_phase())
    with pytest.raises(RuntimeError, match="already applied"):
        stage_f12_runner.stage(out)


def test_stager_refuses_missing_anchor():
    with pytest.raises(RuntimeError, match="anchor is not unique"):
        stage_f12_runner.stage("def transition():\n    pass\n")
