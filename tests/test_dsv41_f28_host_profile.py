"""CPU tests for the F28 generation-thread host profile hook and its staged packed_phase edit.

Two seams are exercised. DIRECT: when the exec'd hybrid ``_decode_cycles`` (co_filename
``'<hybrid_lookup_decode>'``) is already live -- the real order, since the growth-transition hook runs after the
hybrid install -- the hook wraps ``_decode_cycles`` directly. FALLBACK (not yet live): it wraps
``hybrid_install.install`` so each rewrite is re-wrapped. Pure-Python fakes drive both; MLX (pinned to CPU) is used
only by the ``+prof+cl`` order test, which checks F28's ``__wrapped__`` lets the F27 conf hook reach the hybrid
copy's globals in either install order. Run under nice -n 19.
"""
from __future__ import annotations

import importlib.util
import pstats
import sys
import types
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)          # host-only: no Metal, set before any array op

import pytest                          # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from f2 import host_profile            # noqa: E402
from f2 import cycle_log               # noqa: E402
from f2 import stage_f2_runner as st   # noqa: E402

_PACKED = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
_PACKED_PHASE = _PACKED / "packed_phase.py"
_LOOKUP_PY = _PACKED / "lookup.py"
_HYBRID = "<hybrid_lookup_decode>"


def _load_real_lookup():
    spec = importlib.util.spec_from_file_location("f28_test_lookup", _LOOKUP_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_rewritten():
    calls = []

    def rewritten(*a, **kw):
        calls.append((a, kw))
        return ("decoded", a, kw)

    rewritten.calls = calls
    return rewritten


def _fakes(rewritten=None):
    """A fake ``hybrid_install`` module + a fake decode module with NO ``_decode_cycles`` yet, so install takes the
    FALLBACK seam (wrap ``install``); the fake install then reassigns ``module._decode_cycles`` like the real one."""
    if rewritten is None:
        rewritten = _make_rewritten()
    hyb = types.ModuleType("f28_fake_hybrid")
    inst_calls = []

    def install(module, model, prompt, *, requested_depth=None, verify_chunks=None, confidence_threshold=None):
        module._decode_cycles = rewritten
        inst_calls.append((module, model, prompt))
        return {"hybrid_report": True}

    install.calls = inst_calls
    hyb.install = install
    hyb._install_calls = inst_calls   # survives the F28 wrapper replacing hyb.install
    decode = types.SimpleNamespace(_decode_cycles=None)
    return hyb, decode, rewritten


def _hybrid_decode_simple():
    """A decode module whose ``_decode_cycles`` is the exec'd hybrid copy (co_filename '<hybrid_lookup_decode>')."""
    mod = types.SimpleNamespace()
    ns = dict(mod.__dict__)
    exec(compile("def _decode_cycles(x):\n    return ('decoded', x)\n", _HYBRID, "exec"), ns)
    mod._decode_cycles = ns["_decode_cycles"]
    return mod


def _dummy_hyb():
    hyb = types.ModuleType("f28_dummy_hybrid")
    hyb.install = lambda *a, **k: None
    return hyb


def _hybrid_decode_with_edl():
    """A hybrid-live decode whose ``_decode_cycles`` runs one full F27 cycle (primary/edl/extend/commit),
    resolving ``_effective_draft_len`` via its OWN globals -- for the +prof+cl order test."""
    def _effective_draft_len(conf_row, k, threshold):
        return k

    mod = types.SimpleNamespace(_effective_draft_len=_effective_draft_len)
    src = (
        "def _decode_cycles(inst, conf_row, k, native, delta):\n"
        "    inst.append_committed([native[0]])\n"
        "    kn = _effective_draft_len(conf_row, k, None)\n"
        "    inst.extend(list(native[:kn]))\n"
        "    inst.append_committed(list(delta))\n"
        "    return ('done', kn)\n"
    )
    ns = dict(mod.__dict__)
    exec(compile(src, _HYBRID, "exec"), ns)
    mod._decode_cycles = ns["_decode_cycles"]
    return mod


# ------------------------------------------------------------------ direct seam (real order)
def test_direct_wrap_when_hybrid_live(tmp_path):
    decode = _hybrid_decode_simple()
    hyb = _dummy_hyb()
    report = host_profile.install(hyb, decode, out_dir=str(tmp_path))
    assert report == {"installed": True, "out_dir": str(tmp_path), "seam": "direct"}
    assert decode._decode_cycles.__wrapped__.__code__.co_filename == _HYBRID   # wraps the hybrid copy directly
    assert not getattr(hyb.install, "_f28_host_profile", False)                # fallback install NOT touched
    out = decode._decode_cycles(42)
    assert out == ("decoded", 42)
    assert (tmp_path / "hostprof.pstats").exists()
    loaded = pstats.Stats(str(tmp_path / "hostprof.pstats"))
    assert len(loaded.stats) > 0
    txt = (tmp_path / "hostprof.txt").read_text()
    assert "greenlet" in txt.lower() and "strip_dirs() is NOT applied" in txt
    assert "top 80 by tottime" in txt and "top 80 by cumulative" in txt and "callers of the 10 largest tottime" in txt


def test_direct_double_install_refuses(tmp_path):
    decode = _hybrid_decode_simple()
    host_profile.install(_dummy_hyb(), decode, out_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="already wrapped"):
        host_profile.install(_dummy_hyb(), decode, out_dir=str(tmp_path))


# ------------------------------------------------------ fallback seam (hook before hybrid install)
def test_fallback_wraps_install_and_profiles(tmp_path):
    hyb, decode, rewritten = _fakes()
    report = host_profile.install(hyb, decode, out_dir=str(tmp_path))
    assert report == {"installed": True, "out_dir": str(tmp_path), "seam": "hybrid_install_wrap"}
    assert getattr(hyb.install, "_f28_host_profile", False)

    hyb.install(decode, "model", [1, 2, 3], requested_depth=5, verify_chunks=None, confidence_threshold=None)
    assert decode._decode_cycles is not rewritten                  # now the profiled wrapper
    assert len(hyb._install_calls) == 1                            # original install ran once
    assert decode._decode_cycles.__wrapped__ is rewritten          # F27 can unwrap to the real function

    out = decode._decode_cycles([9, 9], cache="c")
    assert out == ("decoded", ([9, 9],), {"cache": "c"})           # same return value
    assert rewritten.calls == [(([9, 9],), {"cache": "c"})]        # original ran once, same args/kwargs
    assert (tmp_path / "hostprof.pstats").exists()
    assert len(pstats.Stats(str(tmp_path / "hostprof.pstats")).stats) > 0
    assert "greenlet" in (tmp_path / "hostprof.txt").read_text().lower()


def test_exception_propagates_and_still_writes(tmp_path):
    def boom(*a, **kw):
        raise ValueError("boom")

    hyb, decode, _ = _fakes(rewritten=boom)
    host_profile.install(hyb, decode, out_dir=str(tmp_path))
    hyb.install(decode, "m", [1])
    with pytest.raises(ValueError, match="boom"):
        decode._decode_cycles(1, 2)
    assert (tmp_path / "hostprof.pstats").exists() and (tmp_path / "hostprof.txt").exists()
    pstats.Stats(str(tmp_path / "hostprof.pstats"))


def test_two_generates_write_hostprof_1_and_2(tmp_path):
    hyb, decode, _ = _fakes()
    host_profile.install(hyb, decode, out_dir=str(tmp_path))
    hyb.install(decode, "m", [1]); decode._decode_cycles(1)        # generate 1
    hyb.install(decode, "m", [2]); decode._decode_cycles(2)        # generate 2
    assert (tmp_path / "hostprof-1.pstats").exists() and (tmp_path / "hostprof-1.txt").exists()
    assert (tmp_path / "hostprof-2.pstats").exists() and (tmp_path / "hostprof-2.txt").exists()
    assert not (tmp_path / "hostprof.pstats").exists()             # the first pair was renamed to -1


def test_double_install_refuses(tmp_path):
    hyb, decode, _ = _fakes()
    host_profile.install(hyb, decode, out_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="already wrapped"):
        host_profile.install(hyb, decode, out_dir=str(tmp_path))


def test_missing_dir_refuses(tmp_path):
    hyb, decode, _ = _fakes()
    with pytest.raises(RuntimeError, match="directory does not exist"):
        host_profile.install(hyb, decode, out_dir=str(tmp_path / "nope"))


def test_missing_install_refuses(tmp_path):
    empty = types.ModuleType("f28_no_install")           # no .install, and decode has no hybrid _decode_cycles
    decode = types.SimpleNamespace(_decode_cycles=None)
    with pytest.raises(RuntimeError, match="hybrid_install.install"):
        host_profile.install(empty, decode, out_dir=str(tmp_path))


def test_unset_env_installs_nothing(monkeypatch):
    monkeypatch.delenv(host_profile.ENV, raising=False)
    assert host_profile.install_from_env() == {"installed": False}


# --------------------------------------------------- +prof+cl compose in either install order
@pytest.mark.parametrize("cl_first", [True, False])
def test_prof_and_cl_compose_in_either_order(tmp_path, cl_first):
    lookup = _load_real_lookup()
    LE = lookup.LookupExtension
    decode = _hybrid_decode_with_edl()
    saved = (LE.extend, LE.append_committed)
    cycle_log._LOG = None
    try:
        if cl_first:
            cycle_log.install(decode, lookup, path=str(tmp_path / "cycles.json"))
            host_profile.install(_dummy_hyb(), decode, out_dir=str(tmp_path))
        else:
            host_profile.install(_dummy_hyb(), decode, out_dir=str(tmp_path))
            cycle_log.install(decode, lookup, path=str(tmp_path / "cycles.json"))
        inst = LE([1, 2, 3, 4, 5, 6], minimum_context=2, extra_tokens=2)
        decode._decode_cycles(inst, mx.array([2.0, 1.0], dtype=mx.float32), 2, [1, 2], [1, 9])
        # F27 reached the hybrid copy's globals -> conf recorded; F28 profiled -> file written.
        assert len(cycle_log._LOG.cycles) == 1
        cyc = cycle_log._LOG.cycles[0]
        assert cyc["k_cap"] == 2 and cyc["k_native"] == 2 and len(cyc["conf"]) == 2 and cyc["committed"] == 2
        assert (tmp_path / "hostprof.pstats").exists()
    finally:
        LE.extend, LE.append_committed = saved
        cycle_log._LOG = None


# --------------------------------------------------------------- staged packed_phase edit
def test_stage_host_profile_roundtrips_compiles_and_composes():
    src = _PACKED_PHASE.read_text()
    assert src.count(st._PROF_ANCHOR) == 1 and "_f28_host_profile" not in src
    out = st.stage_host_profile(src)
    assert out != src and "_f28_host_profile" in out
    lines = out.splitlines()
    idx = lines.index(st._PROF_ANCHOR)
    assert lines[idx + 1] == st._PROF_INSERT                       # hook sits immediately after the anchor
    assert out.replace(st._PROF_ANCHOR + "\n" + st._PROF_INSERT, st._PROF_ANCHOR) == src
    compile(out, "packed_phase_f28", "exec")
    with pytest.raises(RuntimeError):
        st.stage_host_profile(out)                                 # double apply refused
    a = st.stage_host_profile(st.stage_cycle_log(src))
    b = st.stage_cycle_log(st.stage_host_profile(src))
    assert a == b
    assert "_f28_host_profile" in a and "_f27_cycle_log" in a
    compile(a, "packed_phase_f28_f27", "exec")
