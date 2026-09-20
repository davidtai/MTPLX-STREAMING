"""CPU tests for the F28 generation-thread host profile hook and its staged packed_phase edit.

Pure-Python fakes (no MLX, no GPU): a fake ``hybrid_install`` module whose ``install`` reassigns
``module._decode_cycles`` (as the retained one does) and a fake decode module.  After the hook, calling
the fake hybrid install swaps in the profiled ``_decode_cycles``; calling that runs the original exactly
once with the same args/return, propagates exceptions while still writing the files, and produces a
loadable ``pstats`` file + a text summary carrying the greenlet caveat.  Unset env installs nothing,
double install refuses, a missing directory refuses at install.  The staging edit round-trips, compiles
and composes with the F27 cycle-log hook.  Run under nice -n 19.
"""
from __future__ import annotations

import pstats
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from f2 import host_profile               # noqa: E402
from f2 import stage_f2_runner as st      # noqa: E402

_PACKED_PHASE = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed/packed_phase.py"


def _make_rewritten():
    calls = []

    def rewritten(*a, **kw):
        calls.append((a, kw))
        return ("decoded", a, kw)

    rewritten.calls = calls
    return rewritten


def _fakes(rewritten=None):
    """A fake ``hybrid_install`` module + fake decode module. The fake install reassigns
    ``module._decode_cycles`` to ``rewritten`` exactly like the retained rewrite+reassign step."""
    if rewritten is None:
        rewritten = _make_rewritten()
    hyb = types.ModuleType("f28_fake_hybrid")
    inst_calls = []

    def install(module, model, prompt, *, requested_depth=None, verify_chunks=None, confidence_threshold=None):
        module._decode_cycles = rewritten
        inst_calls.append((module, model, prompt, requested_depth, verify_chunks, confidence_threshold))
        return {"hybrid_report": True}

    install.calls = inst_calls
    hyb.install = install
    hyb._install_calls = inst_calls   # survives the F28 wrapper replacing hyb.install
    decode = types.SimpleNamespace(_decode_cycles=None)   # pre-rewrite placeholder
    return hyb, decode, rewritten


# ------------------------------------------------------------------ the hook / profiling
def test_hook_profiles_the_final_decode_cycles(tmp_path):
    hyb, decode, rewritten = _fakes()
    report = host_profile.install(hyb, out_dir=str(tmp_path))
    assert report == {"installed": True, "out_dir": str(tmp_path)}
    assert getattr(hyb.install, "_f28_host_profile", False)

    # simulate a generate: the (wrapped) hybrid install rewrites + reassigns _decode_cycles, then the
    # hook re-wraps it with the profiled version.
    hyb.install(decode, "model", [1, 2, 3], requested_depth=5, verify_chunks=None, confidence_threshold=None)
    assert decode._decode_cycles is not rewritten                  # now the profiled wrapper
    assert len(hyb._install_calls) == 1                            # original install ran once

    out = decode._decode_cycles([9, 9], cache="c")                 # the profiled decode call
    assert out == ("decoded", ([9, 9],), {"cache": "c"})           # same return value
    assert rewritten.calls == [(([9, 9],), {"cache": "c"})]        # original ran once, same args/kwargs

    assert (tmp_path / "hostprof.pstats").exists()
    loaded = pstats.Stats(str(tmp_path / "hostprof.pstats"))       # loadable pstats
    assert len(loaded.stats) > 0                                   # the decode call was profiled
    txt = (tmp_path / "hostprof.txt").read_text()
    assert "greenlet" in txt.lower()                               # the caveat line
    assert "strip_dirs() is NOT applied" in txt
    assert "top 80 by tottime" in txt and "top 80 by cumulative" in txt
    assert "callers of the 10 largest tottime" in txt


def test_exception_propagates_and_still_writes(tmp_path):
    def boom(*a, **kw):
        raise ValueError("boom")

    hyb, decode, _ = _fakes(rewritten=boom)
    host_profile.install(hyb, out_dir=str(tmp_path))
    hyb.install(decode, "m", [1])
    with pytest.raises(ValueError, match="boom"):
        decode._decode_cycles(1, 2)
    # the finally-block dumped the profile despite the exception
    assert (tmp_path / "hostprof.pstats").exists()
    assert (tmp_path / "hostprof.txt").exists()
    pstats.Stats(str(tmp_path / "hostprof.pstats"))                # still loadable


def test_two_generates_write_hostprof_1_and_2(tmp_path):
    hyb, decode, _ = _fakes()
    host_profile.install(hyb, out_dir=str(tmp_path))
    hyb.install(decode, "m", [1]); decode._decode_cycles(1)        # generate 1
    hyb.install(decode, "m", [2]); decode._decode_cycles(2)        # generate 2
    assert (tmp_path / "hostprof-1.pstats").exists() and (tmp_path / "hostprof-1.txt").exists()
    assert (tmp_path / "hostprof-2.pstats").exists() and (tmp_path / "hostprof-2.txt").exists()
    assert not (tmp_path / "hostprof.pstats").exists()             # the first pair was renamed to -1


def test_double_install_refuses(tmp_path):
    hyb, _, _ = _fakes()
    host_profile.install(hyb, out_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="already wrapped"):
        host_profile.install(hyb, out_dir=str(tmp_path))


def test_missing_dir_refuses(tmp_path):
    hyb, _, _ = _fakes()
    with pytest.raises(RuntimeError, match="directory does not exist"):
        host_profile.install(hyb, out_dir=str(tmp_path / "nope"))


def test_missing_install_refuses(tmp_path):
    empty = types.ModuleType("f28_no_install")
    with pytest.raises(RuntimeError, match="hybrid_install.install"):
        host_profile.install(empty, out_dir=str(tmp_path))


def test_unset_env_installs_nothing(monkeypatch):
    monkeypatch.delenv(host_profile.ENV, raising=False)
    assert host_profile.install_from_env() == {"installed": False}


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
    # composes with the F27 cycle-log hook in either order (disjoint anchors)
    a = st.stage_host_profile(st.stage_cycle_log(src))
    b = st.stage_cycle_log(st.stage_host_profile(src))
    assert a == b
    assert "_f28_host_profile" in a and "_f27_cycle_log" in a
    compile(a, "packed_phase_f28_f27", "exec")
