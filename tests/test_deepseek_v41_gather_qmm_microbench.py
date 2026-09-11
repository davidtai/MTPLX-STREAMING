"""W43 CPU tests for scripts/deepseek_v41/gather_qmm_microbench.py.

Proves the microbench's gather_qmm shape construction is the DSV4.1 streamed
switch's ACTUAL all-hit call: it drives the real
``mtplx.models.expert_mlx._gather_component_bank`` with ``mx.gather_qmm``
monkeypatched to a shape-capturing stub (NO Metal, NO GPU) and asserts the x /
rhs_indices shapes the production code passes equal what ``plan_arm`` says the
microbench builds.  Also checks the plan geometry is the canonical DSV4.1
geometry and that ``--help`` / ``--dry-run`` / the never-overwrite guard are
CPU-safe (no mlx, no GPU).

CPU-pinned per the worker-test rule (memory/worker-tests-must-pin-mlx-cpu.md).
Run under ``nice -n 19``, without ``pytest -n auto`` (host-encode sensitivity).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)  # worker-test rule: pin MLX to CPU

from mtplx import deepseek_v41_convert as dc

_WT = Path(__file__).resolve().parents[1]
_SCRIPT = _WT / "scripts" / "deepseek_v41" / "gather_qmm_microbench.py"
_PY = sys.executable
_BUDGET = 8 * 1024**3


def _load():
    spec = importlib.util.spec_from_file_location(
        "dsv41_gather_qmm_microbench", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # register: keeps dataclasses._is_type happy
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gqm():
    return _load()


def _switch_flat_plan(gqm, m: int, arm: str):
    return gqm.plan_arm(
        arm,
        m=m,
        top_k=dc.TOP_K,
        hidden=dc.HIDDEN_SIZE,
        inter=dc.MOE_INTERMEDIATE,
        requested_slots=92,
        mxfp4_group=dc.MXFP4_GROUP,
        budget_bytes=_BUDGET,
    )


def _tiny_bank(gqm, arm: str, slots: int = 8):
    """Build a small synthetic bank for `arm` via the microbench's own builder."""
    plan = gqm.plan_arm(
        arm,
        m=1,
        top_k=dc.TOP_K,
        hidden=dc.HIDDEN_SIZE,
        inter=dc.MOE_INTERMEDIATE,
        requested_slots=slots,
        mxfp4_group=dc.MXFP4_GROUP,
        budget_bytes=_BUDGET,
    )
    key = mx.random.key(0)

    def rng(shape):
        nonlocal key
        key, sub = mx.random.split(key)
        return mx.random.normal(shape, key=sub)

    arrays = gqm.build_bank(mx, plan, rng)
    return gqm._BankShim(arrays), plan.slots


# ==========================================================================
# 1. plan geometry is the canonical DSV4.1 geometry (single source of truth)
# ==========================================================================
def test_module_constants_match_canonical(gqm):
    assert gqm.HIDDEN_SIZE == dc.HIDDEN_SIZE == 5120
    assert gqm.MOE_INTERMEDIATE == dc.MOE_INTERMEDIATE == 2304
    assert gqm.TOP_K == dc.TOP_K == 6
    assert gqm.MXFP4_GROUP == dc.MXFP4_GROUP == 32
    assert gqm.MXFP4_BITS == dc.MXFP4_BITS == 4
    assert gqm.SWIGLU_LIMIT == 10.0


def test_mxfp4_plan_bytes_reproduce_record(gqm):
    plan = _switch_flat_plan(gqm, 1, "mxfp4_switch")
    # per-expert record math == the canonical 18,800,640 B mxfp4 record.
    assert plan.per_slot_bytes == dc.MXFP4_EXPERT_RECORD_BYTES == 18_800_640
    assert plan.rows == dc.TOP_K  # M=1 -> top_k GEMVs
    assert plan.gathered_bytes == dc.TOP_K * 18_800_640 == 112_803_840
    plan4 = _switch_flat_plan(gqm, 4, "mxfp4_switch")
    assert plan4.rows == 4 * dc.TOP_K == 24
    assert plan4.gathered_bytes == 24 * 18_800_640


def test_convention_arm_is_canonical_convention_shape(gqm):
    # The W43 audit found NO violation; this arm is the [rows,1,K]+[rows,top_k]
    # convention form (token not duplicated, top_k folded into the index dim).
    p1 = _switch_flat_plan(gqm, 1, "mxfp4_convention")
    assert p1.x_shape == (1, 1, dc.HIDDEN_SIZE)
    assert p1.rhs_indices_shape == (1, dc.TOP_K)
    assert p1.lhs_indices_shape == (1, dc.TOP_K)
    p4 = _switch_flat_plan(gqm, 4, "mxfp4_convention")
    assert p4.x_shape == (4, 1, dc.HIDDEN_SIZE)
    assert p4.rhs_indices_shape == (4, dc.TOP_K)


def test_budget_fit_shrinks_slots_never_below_rows_when_it_fits(gqm):
    # A 2 GiB budget cannot hold 92 bf16 slots (~6 GiB) but must keep >= rows.
    tight = gqm.plan_arm(
        "bf16_dense", m=4, top_k=dc.TOP_K, hidden=dc.HIDDEN_SIZE,
        inter=dc.MOE_INTERMEDIATE, requested_slots=92, mxfp4_group=dc.MXFP4_GROUP,
        budget_bytes=2 * 1024**3,
    )
    assert tight.slots < 92
    assert tight.slots >= tight.rows  # no wrap needed at this budget
    assert tight.indices_wrapped is False
    assert tight.bank_bytes <= 2 * 1024**3


# ==========================================================================
# 2. THE contract: microbench shapes == the switch's real gather_qmm call
# ==========================================================================
@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize(
    "arm,codec,bits,group",
    [
        ("mxfp4_switch", "mxfp4", 4, 32),
        ("affine_q4_gs64", "affine", 4, 64),
        ("affine_q8_gs64", "affine", 8, 64),
    ],
)
def test_switch_flat_shapes_match_real_gather_component_bank(
    gqm, monkeypatch, m, arm, codec, bits, group
):
    import mtplx.models.expert_mlx as em

    plan = _switch_flat_plan(gqm, m, arm)
    bank, slots = _tiny_bank(gqm, arm, slots=8)
    rows = m * dc.TOP_K

    # Build x exactly as HotExpertSwitchGLU._run builds the all-hit
    # assignment_inputs (expert_mlx.py _run, "wave.positions == range(...)"
    # branch): broadcast each token across top_k, then flatten to [rows, K].
    tokens = mx.zeros((m, dc.HIDDEN_SIZE), dtype=mx.bfloat16)
    assignment_inputs = mx.broadcast_to(
        tokens[:, None, :], (m, dc.TOP_K, dc.HIDDEN_SIZE)
    ).reshape((-1, dc.HIDDEN_SIZE))
    assert tuple(assignment_inputs.shape) == (rows, dc.HIDDEN_SIZE) == (plan.rows, dc.HIDDEN_SIZE)

    # slot_indices as _run_component_bank_q4 builds them: [rows, 1] int32.
    slot_indices = mx.array([i % slots for i in range(rows)], dtype=mx.int32).reshape((-1, 1))

    captured: list[tuple] = []

    def stub(*args, **kwargs):
        x = args[0]
        w = args[1]
        captured.append((tuple(int(d) for d in x.shape),
                         tuple(int(d) for d in kwargs["rhs_indices"].shape),
                         len(args)))
        # transpose=True -> output feature dim is w.shape[1] (the N of [N,Kp]).
        return mx.zeros((*x.shape[:-1], int(w.shape[1])), dtype=mx.bfloat16)

    monkeypatch.setattr(em.mx, "gather_qmm", stub)

    out = em._gather_component_bank(
        assignment_inputs,
        bank,
        slot_indices,
        group_size=group,
        bits=bits,
        swiglu_limit=gqm.SWIGLU_LIMIT,
        codec=codec,
    )
    mx.eval(out)

    # gate + up + down = exactly three gather_qmm calls (no per-expert loop).
    assert len(captured) == 3
    gate_x, gate_rhs, gate_n = captured[0]
    up_x, up_rhs, up_n = captured[1]
    down_x, down_rhs, down_n = captured[2]

    # gate/up take the switch's entry x reshaped to [rows,1,1,hidden] -- exactly
    # plan.x_shape.  down takes the SwiGLU output [rows,1,1,inter].
    assert gate_x == up_x == plan.x_shape == (rows, 1, 1, dc.HIDDEN_SIZE)
    assert down_x == (rows, 1, 1, dc.MOE_INTERMEDIATE)

    for x_shape, rhs_shape, nargs in captured:
        # The [rows,1,K] convention: batch dims [rows,1], GEMV M-dim (x[-2]) == 1,
        # rhs_indices [rows,1] -> exactly `rows` single-row GEMVs, no 8x broadcast.
        assert len(x_shape) == 4 and x_shape[0] == rows and x_shape[-2] == 1
        assert rhs_shape == plan.rhs_indices_shape == (rows, 1)
        # affine passes biases as a 4th positional; mxfp4 does not.
        assert nargs == (4 if codec == "affine" else 3)

    # final output is [rows, hidden] (down projection back to hidden).
    assert tuple(int(d) for d in out.shape) == (rows, dc.HIDDEN_SIZE)


# ==========================================================================
# 3. CPU-safe surfaces: --dry-run, --help, no top-level mlx, overwrite guard
# ==========================================================================
def test_dry_run_returns_zero_in_process(gqm):
    assert gqm.main(["--dry-run"]) == 0
    assert gqm.main(["--dry-run", "--arms", "mxfp4_switch", "--m-values", "1"]) == 0


def test_source_has_no_toplevel_mlx_import():
    src = _SCRIPT.read_text()
    assert "\nimport mlx" not in src
    assert "\nfrom mlx" not in src
    # mlx is used only via the deferred _mx() / lazy imports inside runners.
    assert "import mlx.core as mx" in src


def test_dry_run_subprocess_imports_no_mlx():
    code = (
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('g',r'{_SCRIPT}');"
        "m=importlib.util.module_from_spec(spec);sys.modules['g']=m;"
        "spec.loader.exec_module(m);"
        "rc=m.main(['--dry-run']);"
        "print('RC',rc);"
        "print('MLX', 'mlx.core' in sys.modules or 'mlx' in sys.modules)"
    )
    env = {"PYTHONPATH": str(_WT), "PATH": "/usr/bin:/bin"}
    res = subprocess.run([_PY, "-c", code], capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    assert "RC 0" in res.stdout
    assert "MLX False" in res.stdout, res.stdout


def test_help_is_cpu_safe():
    env = {"PYTHONPATH": str(_WT), "PATH": "/usr/bin:/bin"}
    res = subprocess.run(
        [_PY, str(_SCRIPT), "--help"], capture_output=True, text=True, env=env
    )
    assert res.returncode == 0
    assert "gather_qmm microbench" in res.stdout


def test_never_overwrite_guard_before_any_gpu(gqm, tmp_path):
    existing = tmp_path / "receipt.json"
    existing.write_text("{}")
    # run_real checks out.exists() and returns 2 BEFORE importing mlx / touching
    # the GPU (never-overwrite-a-measurement). CPU-safe.
    rc = gqm.main(["--arms", "mxfp4_switch", "--m-values", "1", "--out", str(existing)])
    assert rc == 2


def test_out_required_for_timing_path(gqm):
    # Without --out and without --dry-run, run_real refuses before any GPU work.
    assert gqm.main(["--arms", "mxfp4_switch", "--m-values", "1"]) == 2
