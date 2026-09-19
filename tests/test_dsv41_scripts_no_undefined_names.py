"""W121 LOW: scripts/deepseek_v41/*.py must have NO undefined names (ruff F821).

The W121 bench cleanup deleted dead functions by name and its transform swallowed a
module-level constant (``_PROFILE_PLAN_FIELDS``) that sat between two deleted defs -- a
NameError only on the real ``_run_arm -> _load_model`` path, which no test exercised, so
window 48 crashed at load.  A cheap static lint over every dsv41 script catches that
class of regression (a reference to a name nothing defines) before a GPU window opens.

CPU-only, no model, no MLX device.  Run under ``nice -n 19``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_WT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _WT / "scripts" / "deepseek_v41"

def test_dsv41_scripts_have_no_undefined_names():
    # LOW: ruff is expected in the venv (it is the lint used here); FAIL LOUDLY if it is
    # absent rather than skipping -- a silent skip would let the _PROFILE_PLAN_FIELDS
    # class of regression through unlinted.
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True
    )
    assert ruff.returncode == 0, (
        "ruff is not importable in this venv (`python -m ruff`); install it so the F821 "
        f"undefined-name lint runs: {ruff.stderr.strip()}"
    )
    scripts = sorted(str(p) for p in _SCRIPTS_DIR.glob("*.py"))
    assert scripts, f"no scripts found under {_SCRIPTS_DIR}"
    r = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821", "--no-cache", *scripts],
        cwd=str(_WT), capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        "ruff F821 found undefined name(s) in scripts/deepseek_v41/*.py:\n"
        f"{r.stdout}\n{r.stderr}"
    )
