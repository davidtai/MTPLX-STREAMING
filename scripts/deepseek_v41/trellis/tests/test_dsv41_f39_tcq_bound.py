"""CPU test for the F39 Task-8 fix: the bounded-MTP-workload model-dir gate (run_full.py ~:515) accepts the +tcq
arm's model dir.  Stages the tree (rewrite_run_full on the REAL retained run_full.py), extracts the model-dir
predicate from the STAGED source, and evaluates it with the tcq arm's EXACT --model from the failed window's
command.txt (and the mxfp4 control + a mismatch).  No serve, no model, no GPU.
"""
import os
import re
import sys
from pathlib import Path

import pytest

_DSV41 = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_DSV41), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tcq import stage_tcq_runner as S      # noqa: E402

_RETAINED_RUN_FULL = _ROOT / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed/run_full.py"
# EXACT --model from the failed window's command.txt (arm-pipe_bal_egl_gt_plx_la_ct50_fi_tcq)
TCQ3_MODEL = "/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-tcq3"
MXFP4_MODEL = "/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"


def _staged_source() -> str:
    if not _RETAINED_RUN_FULL.exists():
        pytest.skip(f"retained run_full.py absent: {_RETAINED_RUN_FULL}")
    return S.rewrite_run_full(_RETAINED_RUN_FULL.read_text())   # applies all edits + asserts round-trip internally


def test_staged_run_full_has_env_gated_model_dir_gate():
    staged = _staged_source()
    assert S._MODEL_ROUTED in staged                            # the committed model-dir fix is in the staged tree
    # the original hardcoded gate is gone (replaced by the env-gated ternary)
    assert "!= Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')\n" not in staged


def _model_dir_fires(staged: str, *, model: str, tcq3_armed: bool) -> bool:
    """Evaluate the STAGED model-dir disjunct (the first term of the bounded-workload `if`) for the given args.

    Extracts the exact `(Path(tcq3) if os.environ.get('MTPLX_DSV41_TCQ3')=='1' else Path(mxfp4))` allowed-dir
    expression from the staged source and evaluates `Path(args.model).resolve() != allowed` — True => the bound
    check fires (raises)."""
    m = re.search(r"Path\(args\.model\)\.resolve\(\) != (\(Path\('[^']*-tcq3'\) if os\.environ\.get\("
                  r"'MTPLX_DSV41_TCQ3'\) == '1' else Path\('[^']*-mxfp4'\)\))", staged)
    assert m, "staged model-dir predicate not found"
    saved = os.environ.get("MTPLX_DSV41_TCQ3")
    try:
        if tcq3_armed:
            os.environ["MTPLX_DSV41_TCQ3"] = "1"
        else:
            os.environ.pop("MTPLX_DSV41_TCQ3", None)
        allowed = eval(m.group(1), {"Path": Path, "os": os})    # the actual staged allowed-dir expression
        return Path(model).resolve() != allowed
    finally:
        if saved is None:
            os.environ.pop("MTPLX_DSV41_TCQ3", None)
        else:
            os.environ["MTPLX_DSV41_TCQ3"] = saved


def test_bound_predicate_accepts_tcq_arm_and_control_rejects_mismatch():
    staged = _staged_source()
    # the +tcq arm's exact args (command.txt: --model .../tcq3, MTPLX_DSV41_TCQ3=1): the gate must NOT fire
    assert _model_dir_fires(staged, model=TCQ3_MODEL, tcq3_armed=True) is False
    # the mxfp4 control (flag unset): identical to the original -- must NOT fire
    assert _model_dir_fires(staged, model=MXFP4_MODEL, tcq3_armed=False) is False
    # a mismatch still fires (the check is not weakened): wrong dir for the armed lane
    assert _model_dir_fires(staged, model=MXFP4_MODEL, tcq3_armed=True) is True
    assert _model_dir_fires(staged, model=TCQ3_MODEL, tcq3_armed=False) is True


def test_control_arm_model_gate_byte_identical_when_flag_unset():
    # with the flag unset the staged ternary evaluates to the mxfp4 Path -> behaviorally identical to the original
    # gate. (The mxfp4 control arm never gets the staged edits anyway -- its tree is the unpatched source, Task 4 --
    # this asserts the tcq staged edit's OFF-path matches the original semantics.)
    staged = _staged_source()
    assert _model_dir_fires(staged, model=MXFP4_MODEL, tcq3_armed=False) is False   # accepts mxfp4 (original behavior)
