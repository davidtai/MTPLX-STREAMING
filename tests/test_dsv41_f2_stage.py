"""CPU tests for the F2b window stager + preflight helpers.

Pure text edits against the REAL archived retained sources (docs/deepseek-v41/receipts/
extension-bank-20260919/full/sources/packed/*.py): anchors unique, edits round-trip
byte-for-byte, staged output compiles. No MLX, no GPU. Run under nice -n 19.
"""
from __future__ import annotations

import hashlib
import json
import py_compile
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import f2.stage_f2_runner as st  # noqa: E402
import f2.window_preflight as wp  # noqa: E402

_ARCH = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
_ADMISSION = _ARCH / "packed_admission.py"
_RUN_FULL = _ARCH / "run_full.py"


def _compiles(text: str, name: str, tmp_path: Path) -> None:
    p = tmp_path / name
    p.write_text(text)
    py_compile.compile(str(p), doraise=True)


def test_admission_cap_roundtrips_and_compiles(tmp_path):
    src = _ADMISSION.read_text()
    out = st.stage_admission(src, max_rows=107)
    assert "range(107, old_capacity, -1)" in out and out != src
    recovered = out.replace(
        "    for capacity in range(107, old_capacity, -1):  # F2b equal-capacity cap",
        st._CAP_ANCHOR,
    )
    assert recovered == src
    _compiles(out, "packed_admission.py", tmp_path)


@pytest.mark.parametrize("bad", [84, 113, 0])
def test_admission_cap_rejects_out_of_range(bad):
    with pytest.raises(RuntimeError):
        st.stage_admission(_ADMISSION.read_text(), max_rows=bad)


def test_run_full_install_and_traceback_roundtrip_and_compile(tmp_path):
    src = _RUN_FULL.read_text()
    out = st.stage_run_full(src)
    assert "f2.install as _f2b_install" in out and "install_from_env(target)" in out
    assert "traceback.print_exc()" in out
    assert out != src
    # round-trip: reverse both insertions -> retained source byte-for-byte.
    step = out.replace(st._TRACE_INSERT + "\n" + st._TRACE_ANCHOR, st._TRACE_ANCHOR)
    recovered = step.replace(st._INSTALL_ANCHOR + "\n" + st._INSTALL_INSERT, st._INSTALL_ANCHOR)
    assert recovered == src
    _compiles(out, "run_full.py", tmp_path)


def test_run_full_install_after_prime_model(tmp_path):
    # the install must be inserted AFTER prime_model (so switch._run/reader are final).
    out = st.stage_run_full(_RUN_FULL.read_text())
    lines = out.splitlines()
    prime_i = next(i for i, l in enumerate(lines) if l.strip() == "projection_owner_report.update(prime_model(target))")
    install_i = next(i for i, l in enumerate(lines) if "install_from_env(target)" in l)
    assert install_i > prime_i


def test_anchors_unique_in_real_sources():
    a = _ADMISSION.read_text()
    r = _RUN_FULL.read_text()
    assert a.count(st._CAP_ANCHOR) == 1
    assert r.count(st._INSTALL_ANCHOR) == 1
    assert r.count(st._TRACE_ANCHOR) == 1


# --- window_preflight helper logic (hermetic; these fns don't import MLX) ---
def test_preflight_compile_files_flags_broken(tmp_path):
    good = tmp_path / "good.py"; good.write_text("x = 1\n")
    bad = tmp_path / "bad.py"; bad.write_text("def (:\n")
    assert wp._compile_files([str(good)]) == []
    assert wp._compile_files([str(bad)])


def test_preflight_verify_helpers_matches_and_flags(tmp_path):
    arch = tmp_path / "arch"; (arch / "packed").mkdir(parents=True)
    helper = arch / "packed" / "run_full.py"; helper.write_bytes(b"print('hi')\n")
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    inst = tmp_path / "packed_installation.json"
    inst.write_text(json.dumps({"helper_sha256": {"run_full.py": digest}}))
    assert wp._verify_helpers(str(arch), str(inst), None) == []
    helper.write_bytes(b"print('changed')\n")
    assert wp._verify_helpers(str(arch), str(inst), None)


def test_preflight_source_pin_flags_missing_worktree(tmp_path):
    inst = tmp_path / "compat.json"
    inst.write_text(json.dumps({"source_commit": "deadbeef", "runtime_source_sha256": {}}))
    probs = wp._source_pin(str(tmp_path / "not-a-worktree"), str(inst))
    assert probs and ("HEAD" in probs[0] or "commit" in probs[0])
