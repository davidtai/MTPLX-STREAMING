"""F6 runner-stager tests: round-trip on the real archived run_full.py, and
composition with the F5 stager (apply F5 then F6; both round-trips must hold).
Pure text staging -- no mlx, no Metal.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

# conftest adds scripts/deepseek_v41/f6 to sys.path
import stage_f6_runner as f6

# worktree root: .../f6/tests/<this> -> parents[4]
_WT_ROOT = Path(__file__).resolve().parents[4]
_ARCHIVED = _WT_ROOT / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed/run_full.py"
_F5_STAGER = Path(
    "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f5-compile"
    "/scripts/deepseek_v41/f5_compile/stage_f5_runner.py"
)


@pytest.fixture(scope="module")
def archived_text():
    if not _ARCHIVED.is_file():
        pytest.skip(f"archived run_full.py not found at {_ARCHIVED}")
    return _ARCHIVED.read_text()


def _load_f5():
    if not _F5_STAGER.is_file():
        return None
    spec = importlib.util.spec_from_file_location("stage_f5_runner", _F5_STAGER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("stage_f5_runner", mod)
    spec.loader.exec_module(mod)
    return mod


def test_f6_alone_roundtrips_on_archived(archived_text):
    staged = f6.stage(archived_text)
    assert staged != archived_text
    # both install points present exactly once, growth_transition still unique
    assert staged.count("_f6ep.install_from_env(resident.model, site='load')") == 1
    assert staged.count("_f6ep.install_from_env(target, site='decode')") == 1
    assert staged.count(f6._D_ANCHOR) == 1
    assert staged.count(f6._L_ANCHOR) == 1
    # explicit reverse recovers the original byte-for-byte
    rev = staged.replace(f6._D_ANCHOR + "\n" + f6._D_INSERT, f6._D_ANCHOR)
    rev = rev.replace(f6._L_ANCHOR + "\n" + f6._L_INSERT, f6._L_ANCHOR)
    assert rev == archived_text


def test_f6_refuses_double_apply(archived_text):
    staged = f6.stage(archived_text)
    with pytest.raises(RuntimeError, match="already applied"):
        f6.stage(staged)


def test_f6_requires_unique_anchors(archived_text):
    # remove the load anchor -> count 0 -> raise
    broken = archived_text.replace(f6._L_ANCHOR, "        pass  # anchor removed", 1)
    with pytest.raises(RuntimeError, match="INSERT L anchor"):
        f6.stage(broken)


def test_compose_f5_then_f6_on_archived(archived_text):
    f5 = _load_f5()
    if f5 is None:
        pytest.skip(f"F5 stager not found at {_F5_STAGER}")

    # F5 first (asserts its own round-trip internally)
    f5_text = f5.stage(archived_text)
    assert f5_text != archived_text
    # then F6 on the F5-staged file (asserts its own round-trip internally)
    composed = f6.stage(f5_text)

    # F5's block is intact and present in the composed file
    assert composed.count("_f5.enable_from_env(_f5_dv)") == 1
    assert composed.count(f5._E3_NEW.split("\n")[1].strip()) >= 1  # traceback line present
    # F6's two inserts present exactly once each
    assert composed.count("_f6ep.install_from_env(resident.model, site='load')") == 1
    assert composed.count("_f6ep.install_from_env(target, site='decode')") == 1
    # shared anchor still unique after both stagers
    assert composed.count(f6._D_ANCHOR) == 1

    # INSERT D lands immediately after growth_transition(), before F5's EDIT-2 block
    idx_anchor = composed.index(f6._D_ANCHOR)
    seg = composed[idx_anchor: idx_anchor + 400]
    assert seg.index("_f6ep.install_from_env(target, site='decode')") < seg.index(
        "_f5.enable_from_env(_f5_dv)"
    )

    # reverse F6 -> recovers the F5-staged text exactly
    rev = composed.replace(f6._D_ANCHOR + "\n" + f6._D_INSERT, f6._D_ANCHOR)
    rev = rev.replace(f6._L_ANCHOR + "\n" + f6._L_INSERT, f6._L_ANCHOR)
    assert rev == f5_text
    # then reverse F5 -> recovers the original archived runner exactly
    rev = rev.replace(f5._E3_NEW, f5._E3_OLD)
    rev = rev.replace(f5._E2_ANCHOR + "\n" + f5._E2_INSERT, f5._E2_ANCHOR)
    assert rev == archived_text


def test_compose_is_order_documented_f6_after_f5(archived_text):
    """Applying F6 before F5 also composes (both anchors independent of order),
    but the receipt convention is F5-then-F6; prove F6-then-F5 still round-trips."""
    f5 = _load_f5()
    if f5 is None:
        pytest.skip("F5 stager not found")
    f6_text = f6.stage(archived_text)
    both = f5.stage(f6_text)
    # reverse F5 then F6 -> original
    rev = both.replace(f5._E3_NEW, f5._E3_OLD)
    rev = rev.replace(f5._E2_ANCHOR + "\n" + f5._E2_INSERT, f5._E2_ANCHOR)
    rev = rev.replace(f6._D_ANCHOR + "\n" + f6._D_INSERT, f6._D_ANCHOR)
    rev = rev.replace(f6._L_ANCHOR + "\n" + f6._L_INSERT, f6._L_ANCHOR)
    assert rev == archived_text
