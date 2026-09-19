"""CPU tests for the F2 window stager (scripts/deepseek_v41/f2/stage_f2_runner.py).

Pure text edits against the REAL archived retained sources (docs/deepseek-v41/receipts/
extension-bank-20260919/full/sources/packed/*.py): anchors unique, edits round-trip
byte-for-byte, staged output compiles. No MLX, no GPU. Run under nice -n 19.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import f2.stage_f2_runner as st  # noqa: E402

_ARCH = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
_ADMISSION = _ARCH / "packed_admission.py"
_PACKED_PHASE = _ARCH / "packed_phase.py"


def _compiles(text: str, name: str, tmp_path: Path) -> None:
    p = tmp_path / name
    p.write_text(text)
    py_compile.compile(str(p), doraise=True)


def test_admission_cap_roundtrips_and_compiles(tmp_path):
    src = _ADMISSION.read_text()
    out = st.stage_admission(src, max_rows=108)
    assert "range(108, old_capacity, -1)" in out
    assert out != src
    # round-trip: strip the F2 marker comment back to the retained line.
    recovered = out.replace(
        "    for capacity in range(108, old_capacity, -1):  # F2 equal-capacity cap",
        st._CAP_ANCHOR,
    )
    assert recovered == src
    _compiles(out, "packed_admission.py", tmp_path)


def test_admission_ring_charge_roundtrips_and_compiles(tmp_path):
    src = _ADMISSION.read_text()
    out = st.stage_admission(src, ring_records=32)
    assert "active += 32 * WEIGHTS" in out
    recovered = out.replace(
        st._ACTIVE_ANCHOR + "\n        active += 32 * WEIGHTS  # F2 speculative ring reserve (R records)",
        st._ACTIVE_ANCHOR,
    )
    assert recovered == src
    _compiles(out, "packed_admission.py", tmp_path)


def test_admission_cap_plus_ring_together(tmp_path):
    src = _ADMISSION.read_text()
    out = st.stage_admission(src, max_rows=107, ring_records=32)
    assert "range(107, old_capacity, -1)" in out and "active += 32 * WEIGHTS" in out
    _compiles(out, "packed_admission.py", tmp_path)


def test_packed_phase_install_swap_roundtrips_and_compiles(tmp_path):
    src = _PACKED_PHASE.read_text()
    out = st.stage_packed_phase(src)
    assert "install_f2_growth(rt, dict(zip(layers, switches)), owners, model=model)" in out
    assert "install_plane_lane" not in out
    assert out.replace(st._INSTALL_F2, st._INSTALL_ANCHOR) == src
    _compiles(out, "packed_phase.py", tmp_path)


@pytest.mark.parametrize("bad", [84, 113, 0])
def test_admission_cap_rejects_out_of_range(bad):
    with pytest.raises(RuntimeError):
        st.stage_admission(_ADMISSION.read_text(), max_rows=bad)


def test_admission_ring_records_bounds():
    with pytest.raises(RuntimeError):
        st.stage_admission(_ADMISSION.read_text(), ring_records=65)


def test_anchors_unique_in_real_sources():
    a = _ADMISSION.read_text()
    p = _PACKED_PHASE.read_text()
    assert a.count(st._CAP_ANCHOR) == 1
    assert a.count(st._ACTIVE_ANCHOR) == 1
    assert p.count(st._INSTALL_ANCHOR) == 1


# --- window_preflight helper logic (hermetic; no MLX -- these fns don't import it) ---
import hashlib  # noqa: E402
import json  # noqa: E402

import f2.window_preflight as wp  # noqa: E402


def test_preflight_compile_files_flags_broken(tmp_path):
    good = tmp_path / "good.py"; good.write_text("x = 1\n")
    bad = tmp_path / "bad.py"; bad.write_text("def (:\n")
    assert wp._compile_files([str(good)]) == []
    probs = wp._compile_files([str(bad)])
    assert probs and "does not compile" in probs[0]


def test_preflight_verify_helpers_matches_and_flags(tmp_path):
    arch = tmp_path / "arch"; (arch / "packed").mkdir(parents=True)
    helper = arch / "packed" / "run_full.py"; helper.write_bytes(b"print('hi')\n")
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    inst = tmp_path / "packed_installation.json"
    inst.write_text(json.dumps({"helper_sha256": {"run_full.py": digest}}))
    assert wp._verify_helpers(str(arch), str(inst), None) == []
    # a changed helper is flagged.
    helper.write_bytes(b"print('changed')\n")
    probs = wp._verify_helpers(str(arch), str(inst), None)
    assert probs and "differs from pin" in probs[0]


def test_preflight_source_pin_flags_wrong_commit(tmp_path):
    # a non-git dir yields a clear "cannot read HEAD" refusal, not a crash.
    inst = tmp_path / "compat.json"
    inst.write_text(json.dumps({"source_commit": "deadbeef", "runtime_source_sha256": {}}))
    probs = wp._source_pin(str(tmp_path / "not-a-worktree"), str(inst))
    assert probs and ("HEAD" in probs[0] or "commit" in probs[0])
