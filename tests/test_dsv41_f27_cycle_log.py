"""CPU tests for the F27 per-cycle draft log (host-only) and its staged packed_phase edit.

Drives the three wraps with a FAKE decode module + the REAL retained lookup.py (loaded by path):
a scripted sequence of DSpark cycles produces exactly the expected records; the k_cap == 0 cycle,
the leading primary-token call, a second generate's marker, double install, a missing target, a bad
path, an unset variable and the atexit JSON writer are all checked.  The staging edit round-trips,
compiles and composes with the F19/F21 hook like the other stagers.  MLX is pinned to CPU before any
array op (the conf conversion is host numpy on a CPU array); no GPU.  Run under nice -n 19.
"""
from __future__ import annotations

import atexit
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)          # host-only: no Metal, set before any array op

import numpy as np                     # noqa: E402
import pytest                          # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from f2 import cycle_log               # noqa: E402
from f2 import stage_f2_runner as st   # noqa: E402

_PACKED = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
_LOOKUP_PY = _PACKED / "lookup.py"
_PACKED_PHASE = _PACKED / "packed_phase.py"


def _load_real_lookup():
    """The retained lookup.py imports only collections; load it by path (no MLX, no sys.path)."""
    spec = importlib.util.spec_from_file_location("f27_test_lookup", _LOOKUP_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lookup = _load_real_lookup()
LookupExtension = _lookup.LookupExtension


def _fake_effective_draft_len(conf_row, k, threshold):
    """Mirror of deepseek_v41_dspark_decode._effective_draft_len: keep the leading run of drafts
    whose sigmoid confidence clears the threshold (>= 1); keep all k when the threshold is off."""
    if threshold is None or k <= 0:
        return k
    conf = np.asarray(mx.sigmoid(conf_row.astype(mx.float32))).reshape(-1)
    keep = 0
    for i in range(k):
        if float(conf[i]) >= threshold:
            keep += 1
        else:
            break
    return max(1, keep)


def _fake_decode():
    return types.SimpleNamespace(_effective_draft_len=_fake_effective_draft_len)


def _hybrid_decode_module():
    """A fake decode module whose _decode_cycles is the exec'd hybrid copy (filename
    '<hybrid_lookup_decode>'), so its bare _effective_draft_len resolves via the copy's OWN globals dict --
    exactly the retained seam (the growth-transition hook runs after this rewrite). Replacing the module
    attribute would NOT reach it; the fix wraps the name inside the live function's __globals__."""
    mod = types.SimpleNamespace(_effective_draft_len=_fake_effective_draft_len)
    src = (
        "def _decode_cycles(inst, conf_row, k, threshold, native, delta):\n"
        "    inst.append_committed([native[0]])                  # leading primary (not a cycle)\n"
        "    kn = _effective_draft_len(conf_row, k, threshold)   # resolved via THIS globals dict\n"
        "    res = inst.extend(list(native[:kn]))\n"
        "    inst.append_committed(list(delta))                  # close the cycle\n"
        "    return kn, len(res)\n"
    )
    namespace = dict(mod.__dict__)                               # what hybrid_install.install does (line 54)
    exec(compile(src, "<hybrid_lookup_decode>", "exec"), namespace)
    mod._decode_cycles = namespace["_decode_cycles"]
    return mod


def _sig6(logits):
    """The wrapper's conf list: sigmoid of the float32 row, 6 decimals (float32->float64, like it)."""
    x = np.asarray(logits, dtype=np.float32).astype(np.float64)
    return [round(float(v), 6) for v in 1.0 / (1.0 + np.exp(-x))]


def _row(logits):
    return mx.array(logits, dtype=mx.float32)


@pytest.fixture
def restore():
    """Save/restore the real lookup class methods and clear the module install state per test."""
    saved = (LookupExtension.extend, LookupExtension.append_committed)
    try:
        yield
    finally:
        LookupExtension.extend, LookupExtension.append_committed = saved
        if cycle_log._LOG is not None:
            try:
                atexit.unregister(cycle_log._LOG.write)
            except Exception:
                pass
            cycle_log._LOG = None


# --------------------------------------------------------------------------- scripted cycles
def test_scripted_cycles_produce_exact_records(restore, tmp_path):
    decode = _fake_decode()
    report = cycle_log.install(decode, _lookup, path=str(tmp_path / "cycles.json"))
    # a fake decode has no _decode_cycles, so the sink is the module attribute
    assert report == {"installed": True, "path": str(tmp_path / "cycles.json"), "conf_seam": "module_attribute"}
    log = cycle_log._LOG

    # G=[1,2,3,4,5] appears once, preceded by [8,7] and followed by [90,91]; after the primary the
    # tail is [...,8,7], so the causal context matches and extend returns 5 -> 7 tokens.
    prompt = [50, 51, 8, 7, 1, 2, 3, 4, 5, 90, 91, 60, 61, 62, 63, 8]
    inst = LookupExtension(prompt, minimum_context=2, extra_tokens=2)
    inst.append_committed([7])                         # leading primary -- NOT a cycle
    assert log.cycles == []

    # Cycle A: full draft, threshold off, WITH a lookup extension (k_total > k_native).
    logits_a = [3.0, 2.0, 1.0, 0.5, 2.5]
    assert decode._effective_draft_len(_row(logits_a), 5, None) == 5
    ra = inst.extend([1, 2, 3, 4, 5])
    assert len(ra) == 7                                # the real lookup extended 5 -> 7
    inst.append_committed([1, 2, 3, 101])              # committed = 4

    # Cycle B: full draft, threshold off, NO extension (k_total == k_native).
    logits_b = [2.0, 2.0, 2.0, 1.0, 1.0]
    assert decode._effective_draft_len(_row(logits_b), 5, None) == 5
    assert len(inst.extend([2001, 2002, 2003, 2004, 2005])) == 5
    inst.append_committed([2001, 2002, 2003, 2004, 2005, 777])   # committed = 6

    # Cycle C: truncated draft under a threshold (k_native < k_cap; conf still holds k_cap floats).
    logits_c = [2.0, 1.5, -0.5, 2.0, 2.0]
    assert decode._effective_draft_len(_row(logits_c), 5, 0.6) == 2
    assert len(inst.extend([3001, 3002])) == 2
    inst.append_committed([3001, 3002, 888])           # committed = 3

    # Cycle D: k_cap == 0 (no draft) -- only append_committed fires; opened lazily, closed here.
    inst.append_committed([999])                       # committed = 1

    assert log.cycles == [
        {"conf": _sig6(logits_a), "k_cap": 5, "threshold": None, "k_native": 5, "k_total": 7, "committed": 4},
        {"conf": _sig6(logits_b), "k_cap": 5, "threshold": None, "k_native": 5, "k_total": 5, "committed": 6},
        {"conf": _sig6(logits_c), "k_cap": 5, "threshold": 0.6, "k_native": 2, "k_total": 2, "committed": 3},
        {"committed": 1},
    ]
    # keys land in schema order for a full record
    assert list(log.cycles[0].keys()) == ["conf", "k_cap", "threshold", "k_native", "k_total", "committed"]
    assert len(log.cycles[0]["conf"]) == 5 and len(log.cycles[2]["conf"]) == 5


def test_second_dspark_generate_gets_a_marker(restore, tmp_path):
    decode = _fake_decode()
    cycle_log.install(decode, _lookup, path=str(tmp_path / "cycles.json"))
    log = cycle_log._LOG

    inst1 = LookupExtension([1, 2, 3, 4, 5, 6, 7], minimum_context=2, extra_tokens=2)
    inst1.append_committed([8])                        # gen 1 primary
    decode._effective_draft_len(_row([1.0, 1.0]), 2, None)
    inst1.extend([100, 101])
    inst1.append_committed([100, 101, 9])              # gen 1: one cycle

    inst2 = LookupExtension([1, 2, 3, 4, 5, 6, 7], minimum_context=2, extra_tokens=2)
    inst2.append_committed([8])                        # gen 2 primary -> marker, no cycle
    decode._effective_draft_len(_row([2.0, 2.0]), 2, None)
    inst2.extend([200, 201])
    inst2.append_committed([200, 201, 9])              # gen 2: one cycle

    assert [c.get("generate") for c in log.cycles] == [None, 2, None]
    assert log.cycles[1] == {"generate": 2}
    assert log.cycles[0]["k_native"] == 2 and log.cycles[2]["k_native"] == 2


def test_hybrid_live_records_conf_via_function_globals(restore, tmp_path):
    # Reproduce the REAL order: the hybrid copy is already live, so the module-attribute wrap would be a
    # no-op; the fix wraps _effective_draft_len inside the live function's globals. (Regression for the bug.)
    decode = _hybrid_decode_module()
    report = cycle_log.install(decode, _lookup, path=str(tmp_path / "cycles.json"))
    assert report["conf_seam"] == "function_globals"       # corrected seam, not module_attribute
    log = cycle_log._LOG
    inst = LookupExtension([1, 2, 3, 4, 5, 6], minimum_context=2, extra_tokens=2)
    logits = [2.0, 1.0, 0.5]
    kn, ktot = decode._decode_cycles(inst, _row(logits), 3, None, [1, 2, 3], [1, 2, 99])
    assert kn == 3
    assert log.cycles == [
        {"conf": _sig6(logits), "k_cap": 3, "threshold": None, "k_native": 3, "k_total": ktot, "committed": 3}
    ]


# ------------------------------------------------------------------ install refusals + env
def test_double_install_refuses(restore, tmp_path):
    decode = _fake_decode()
    cycle_log.install(decode, _lookup, path=str(tmp_path / "cycles.json"))
    with pytest.raises(RuntimeError, match="already wrapped"):
        cycle_log.install(decode, _lookup, path=str(tmp_path / "cycles2.json"))


def test_missing_decode_target_refuses(restore, tmp_path):
    decode = types.SimpleNamespace()                   # no _effective_draft_len
    with pytest.raises(RuntimeError, match="target missing"):
        cycle_log.install(decode, _lookup, path=str(tmp_path / "cycles.json"))


def test_missing_lookup_class_refuses(restore, tmp_path):
    fake_lookup = types.SimpleNamespace()              # no LookupExtension
    with pytest.raises(RuntimeError, match="LookupExtension"):
        cycle_log.install(_fake_decode(), fake_lookup, path=str(tmp_path / "cycles.json"))


def test_bad_path_refuses(restore, tmp_path):
    with pytest.raises(RuntimeError, match="must end in .json"):
        cycle_log.install(_fake_decode(), _lookup, path=str(tmp_path / "cycles.txt"))
    with pytest.raises(RuntimeError, match="parent directory does not exist"):
        cycle_log.install(_fake_decode(), _lookup, path=str(tmp_path / "nope" / "cycles.json"))


def test_unset_env_installs_nothing(restore, monkeypatch):
    monkeypatch.delenv(cycle_log.ENV, raising=False)
    assert cycle_log.install_from_env() == {"installed": False}
    assert cycle_log._LOG is None
    # the real lookup class was never touched
    assert not getattr(LookupExtension.extend, "_f27_cycle_log", False)


def test_atexit_writer_emits_valid_json(restore, tmp_path):
    decode = _fake_decode()
    path = tmp_path / "cycles.json"
    cycle_log.install(decode, _lookup, path=str(path))
    log = cycle_log._LOG
    inst = LookupExtension([1, 2, 3, 4, 5, 6], minimum_context=2, extra_tokens=2)
    inst.append_committed([7])                         # primary
    decode._effective_draft_len(_row([1.0, 1.0]), 2, None)
    inst.extend([10, 11])
    inst.append_committed([10, 11, 8])                 # one cycle
    log.write()                                        # the function atexit would call at exit
    payload = json.loads(path.read_text())
    assert payload["schema"] == 1
    assert isinstance(payload["cycles"], list) and len(payload["cycles"]) == 1
    assert payload["cycles"][0]["committed"] == 3


# --------------------------------------------------------------- staged packed_phase edit
def test_stage_cycle_log_roundtrips_compiles_and_composes():
    src = _PACKED_PHASE.read_text()
    assert src.count(st._CL_ANCHOR) == 1 and "_f27_cycle_log" not in src
    out = st.stage_cycle_log(src)
    assert out != src and "_f27_cycle_log" in out
    lines = out.splitlines()
    idx = lines.index(st._CL_ANCHOR)
    assert lines[idx + 1] == st._CL_INSERT              # hook sits immediately AFTER the plane-lane call
    # round-trip: drop the one inserted line -> the archived source byte-for-byte
    assert out.replace(st._CL_ANCHOR + "\n" + st._CL_INSERT, st._CL_ANCHOR) == src
    compile(out, "packed_phase_f27", "exec")
    with pytest.raises(RuntimeError):
        st.stage_cycle_log(out)                         # double apply refused
    # composes with the F19/F21 read-order hook in either order (disjoint anchors)
    a = st.stage_cycle_log(st.stage_read_order(src))
    b = st.stage_read_order(st.stage_cycle_log(src))
    assert a == b
    assert "_f27_cycle_log" in a and "_f19_read_order" in a and "_f21_submit_yield" in a
    compile(a, "packed_phase_f27_f19", "exec")
