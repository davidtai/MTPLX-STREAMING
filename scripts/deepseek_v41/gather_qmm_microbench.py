#!/usr/bin/env python3
"""W43 gather_qmm microbench for the DSV4.1-Flash streamed-switch all-hit path.

Times the routed-expert component-bank gather at M=1 (AR decode) and M=4 (MTP
verify) against the real DSV4.1-Flash geometry, so the window can settle whether
the ~2.2 ms/layer ``moe.routed_switch`` cost is the gather itself or (per W42) the
per-layer blocking ``mx.eval`` fence.  W42 argues the gather moves only
6 x 18,800,640 B = 112.8 MB from a resident bank (~0.19 ms @ 600 GB/s); this
bench measures the *actual* mxfp4 gs32 M=1 GEMV time, which the
``metal-sub4bit-alu-bound`` note warns may be far off bandwidth (the mxfp4 gather
has 144 template variants; the M=1 variant may be ALU-bound).

Arms (each times one full routed-expert MLP: gate + up + ClampedSwiGLU(limit) +
down, exactly as ``mtplx.models.expert_mlx._gather_component_bank`` runs it):

  mxfp4_switch      native mxfp4 gs32, the switch's EXACT call
                    (x -> [rows,1,1,K], rhs_indices [rows,1], transpose=True,
                    mode="mxfp4"; rows = M*top_k).  Runs the real
                    ``_gather_component_bank`` so the timed code IS production.
  mxfp4_convention  native mxfp4 gs32 in the canonical [rows,1,K] + [rows,top_k]
                    convention form (token NOT duplicated; top_k folded into the
                    rhs_indices second dim via lhs_indices).  The W43 audit found
                    NO convention violation in the switch (see
                    docs/deepseek-v41/W43_GATHER_QMM_AUDIT.md), so this arm is an
                    EQUIVALENCE check: does folding top_k into the index dim pick
                    a different / faster kernel variant than the flat form?
  affine_q4_gs64    affine q4 gs64 gather_qmm (weight+scales+biases), switch layout
  affine_q8_gs64    affine q8 gs64 gather_qmm (weight+scales+biases), switch layout
  bf16_dense        bf16 gather_mm over gathered dense slices -- the bandwidth
                    reference (no dequant ALU; reads 4x the mxfp4 bytes)

CPU-safe: ``--help`` and ``--dry-run`` import NO mlx and touch NO GPU/Metal --
all shapes/bytes/the window command are pure integer geometry.  The real timing
path imports ``mlx.core`` lazily and MUST run inside the exclusive GPU window
(David runs it).  This worker never runs the timing path.

Never-overwrite (memory/never-overwrite-a-measurement): refuses to clobber an
existing ``--out`` receipt.

Run the timing path only inside the guarded window; see the command printed by
``--dry-run`` and recorded in W43_GATHER_QMM_AUDIT.md.

NOTE: deliberately NO ``from __future__ import annotations`` -- with it every
dataclass field annotation becomes a string that ``dataclasses._is_type``
resolves via ``sys.modules[cls.__module__]``, which is None when the module is
loaded by file path without registration (the repo's bench-script test loader
does exactly that, tests/test_deepseek_v41_bench_scripts.py::_load).  Eager
annotations avoid that crash; every annotation here evaluates under Python 3.12.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Canonical DSV4.1-Flash geometry -- single source of truth (mlx-free import,
# verified in tests/test_deepseek_v41_gather_qmm_microbench.py against these).
from mtplx.deepseek_v41_convert import (
    HIDDEN_SIZE,
    MOE_INTERMEDIATE,
    MXFP4_BITS,
    MXFP4_EXPERT_RECORD_BYTES,
    MXFP4_GROUP,
    N_ROUTED_EXPERTS,
    TOP_K,
)

# DeepSeek-V4.1-Flash clamps its experts at +/-10.0 (model.py Expert.forward;
# spec.swiglu_limit == 10.0, asserted in tests/test_deepseek_v41_mxfp4_bank.py).
SWIGLU_LIMIT = 10.0

ARMS = (
    "mxfp4_switch",
    "mxfp4_convention",
    "affine_q4_gs64",
    "affine_q8_gs64",
    "bf16_dense",
)

_GIB = 1024**3


# ---------------------------------------------------------------------------
# Pure-integer geometry (no mlx) -- the shape/byte plan every arm is built from.
# ---------------------------------------------------------------------------
@dataclass
class ProjGeom:
    """One projection's stacked-bank component geometry for a codec."""

    name: str
    out_size: int
    in_size: int
    # component -> (dtype_str, per-slot shape (without the leading slot axis))
    components: dict[str, tuple[str, tuple[int, ...]]]
    per_slot_bytes: int


def _dtype_bytes(dt: str) -> int:
    return {"uint32": 4, "uint8": 1, "bfloat16": 2, "float16": 2}[dt]


def projection_geoms(
    *, hidden: int, inter: int, codec: str, bits: int, group: int
) -> list[ProjGeom]:
    """Per-slot component shapes/dtypes for gate/up/down under a codec.

    Mirrors mtplx.models.expert_mlx.make_mlx_component_bank_allocator's
    ``expected_signature`` (the stacked component bank gather_qmm reads).
    """
    geoms: list[ProjGeom] = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        out_size = inter if proj in ("gate_proj", "up_proj") else hidden
        in_size = hidden if proj in ("gate_proj", "up_proj") else inter
        comps: dict[str, tuple[str, tuple[int, ...]]] = {}
        if codec == "bf16":
            # Dense reference: gather_mm reads x @ w with w = [in, out] per slot.
            comps["weight"] = ("bfloat16", (in_size, out_size))
        elif codec == "mxfp4":
            comps["weight"] = ("uint32", (out_size, in_size * bits // 32))
            comps["scales"] = ("uint8", (out_size, in_size // group))
        elif codec == "affine":
            comps["weight"] = ("uint32", (out_size, in_size * bits // 32))
            comps["scales"] = ("bfloat16", (out_size, in_size // group))
            comps["biases"] = ("bfloat16", (out_size, in_size // group))
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"unknown codec {codec!r}")
        per_slot = sum(
            _dtype_bytes(dt) * math.prod(shape) for dt, shape in comps.values()
        )
        geoms.append(
            ProjGeom(
                name=proj,
                out_size=out_size,
                in_size=in_size,
                components=comps,
                per_slot_bytes=per_slot,
            )
        )
    return geoms


@dataclass
class ArmPlan:
    arm: str
    codec: str
    bits: int
    group: int
    layout: str  # "switch_flat" | "convention" | "bf16_gather_mm"
    m: int
    top_k: int
    hidden: int
    inter: int
    rows: int  # M * top_k GEMVs the gather issues
    requested_slots: int
    slots: int  # actual, after fitting to the memory budget
    indices_wrapped: bool
    # gather_qmm/gather_mm input shapes (what the kernel receives)
    x_shape: tuple[int, ...]
    rhs_indices_shape: tuple[int, ...]
    lhs_indices_shape: tuple[int, ...] | None
    per_slot_bytes: int
    bank_bytes: int
    gathered_bytes: int  # bytes the gather reads: rows * per-expert bytes
    per_slot_bytes_by_proj: dict[str, int] = field(default_factory=dict)


def _fit_slots(requested_slots: int, per_slot_bytes: int, rows: int, budget_bytes: int):
    """Largest slot count <= requested that fits the budget; >= rows if possible."""
    max_by_budget = int(budget_bytes // per_slot_bytes) if per_slot_bytes else requested_slots
    slots = min(requested_slots, max_by_budget)
    wrapped = False
    if slots < rows:
        # Can't hold one distinct expert per GEMV within budget: wrap the
        # indices (repeats hit cache and INFLATE GB/s -- flagged in the receipt).
        slots = max(slots, 1)
        wrapped = True
    if slots < 1:
        slots = 1
        wrapped = rows > 1
    return slots, wrapped


def plan_arm(
    arm: str,
    *,
    m: int,
    top_k: int,
    hidden: int,
    inter: int,
    requested_slots: int,
    mxfp4_group: int,
    budget_bytes: int,
) -> ArmPlan:
    rows = m * top_k
    if arm == "mxfp4_switch":
        codec, bits, group, layout = "mxfp4", MXFP4_BITS, mxfp4_group, "switch_flat"
    elif arm == "mxfp4_convention":
        codec, bits, group, layout = "mxfp4", MXFP4_BITS, mxfp4_group, "convention"
    elif arm == "affine_q4_gs64":
        codec, bits, group, layout = "affine", 4, 64, "switch_flat"
    elif arm == "affine_q8_gs64":
        codec, bits, group, layout = "affine", 8, 64, "switch_flat"
    elif arm == "bf16_dense":
        codec, bits, group, layout = "bf16", 16, 0, "bf16_gather_mm"
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown arm {arm!r}")

    geoms = projection_geoms(hidden=hidden, inter=inter, codec=codec, bits=bits, group=group)
    per_slot_bytes = sum(g.per_slot_bytes for g in geoms)
    slots, wrapped = _fit_slots(requested_slots, per_slot_bytes, rows, budget_bytes)
    bank_bytes = per_slot_bytes * slots
    gathered_bytes = per_slot_bytes * rows

    # Kernel input shapes, per layout.
    if layout == "switch_flat":
        # _gather_component_bank: selected = x.reshape((rows, 1, 1, K)); rhs [rows,1].
        x_shape = (rows, 1, 1, hidden)
        rhs_shape: tuple[int, ...] = (rows, 1)
        lhs_shape: tuple[int, ...] | None = None
    elif layout == "convention":
        # canonical [rows=M, 1, K] with lhs/rhs [M, top_k] -> output [M, top_k, N]
        x_shape = (m, 1, hidden)
        rhs_shape = (m, top_k)
        lhs_shape = (m, top_k)
    else:  # bf16_gather_mm
        # gather_mm computes a @ b (no transpose): a=[rows,1,1,K], b=[slots,K,N].
        x_shape = (rows, 1, 1, hidden)
        rhs_shape = (rows, 1)
        lhs_shape = (rows, 1)

    return ArmPlan(
        arm=arm,
        codec=codec,
        bits=bits,
        group=group,
        layout=layout,
        m=m,
        top_k=top_k,
        hidden=hidden,
        inter=inter,
        rows=rows,
        requested_slots=requested_slots,
        slots=slots,
        indices_wrapped=wrapped,
        x_shape=x_shape,
        rhs_indices_shape=rhs_shape,
        lhs_indices_shape=lhs_shape,
        per_slot_bytes=per_slot_bytes,
        bank_bytes=bank_bytes,
        gathered_bytes=gathered_bytes,
        per_slot_bytes_by_proj={g.name: g.per_slot_bytes for g in geoms},
    )


def build_plans(cfg: argparse.Namespace) -> list[ArmPlan]:
    budget_bytes = int(cfg.memory_limit_gib * _GIB)
    plans: list[ArmPlan] = []
    for arm in cfg.arms:
        for m in cfg.m_values:
            plans.append(
                plan_arm(
                    arm,
                    m=m,
                    top_k=cfg.top_k,
                    hidden=cfg.hidden,
                    inter=cfg.inter,
                    requested_slots=cfg.bank_slots,
                    mxfp4_group=cfg.group_size_mxfp4,
                    budget_bytes=budget_bytes,
                )
            )
    return plans


# ---------------------------------------------------------------------------
# Window command (printed by --dry-run and echoed into the W43 report).
# ---------------------------------------------------------------------------
def window_command(cfg: argparse.Namespace, *, out: str) -> str:
    # Canonical form from scripts/deepseek_v41/README.md: the step is the DIRECT
    # command of gpu_window.sh (which execs "$@" once it holds the exclusive
    # lock).  $WT = the worktree that has this branch checked out, $PY the venv
    # interpreter.  gpu_window.sh itself does no MLX/GPU; the step does.
    return (
        'WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w43; '
        'PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3; '
        "cd $WT && bash scripts/deepseek_v41/gpu_window.sh "
        "env PYTHONPATH=$WT $PY scripts/deepseek_v41/gather_qmm_microbench.py "
        f"--arms {' '.join(cfg.arms)} "
        f"--m-values {' '.join(str(m) for m in cfg.m_values)} "
        f"--top-k {cfg.top_k} --hidden {cfg.hidden} --inter {cfg.inter} "
        f"--bank-slots {cfg.bank_slots} --group-size-mxfp4 {cfg.group_size_mxfp4} "
        f"--iters {cfg.iters} --warmup {cfg.warmup} "
        f"--memory-limit-gib {cfg.memory_limit_gib} --seed {cfg.seed} "
        f"--out {out}"
    )


def dry_run(cfg: argparse.Namespace) -> int:
    plans = build_plans(cfg)
    out = cfg.out or "docs/deepseek-v41/receipts/gpu-windows/window-NN/gather-qmm-microbench.json"
    print("W43 gather_qmm microbench -- DRY RUN (no mlx, no GPU) ")
    print(f"geometry: hidden={cfg.hidden} inter={cfg.inter} top_k={cfg.top_k} "
          f"n_routed={N_ROUTED_EXPERTS} mxfp4_group={cfg.group_size_mxfp4} bits={MXFP4_BITS}")
    print(f"mxfp4 expert record bytes (canonical) = {MXFP4_EXPERT_RECORD_BYTES:,}")
    print(f"memory budget = {cfg.memory_limit_gib} GiB  iters={cfg.iters} warmup={cfg.warmup}")
    print()
    hdr = (f"{'arm':<17}{'M':>2}{'rows':>6}{'slots':>7}{'x_shape':>18}"
           f"{'rhs':>10}{'lhs':>10}{'bank_GiB':>10}{'gathered_MB':>13}{'wrap':>6}")
    print(hdr)
    print("-" * len(hdr))
    for p in plans:
        print(
            f"{p.arm:<17}{p.m:>2}{p.rows:>6}{p.slots:>7}"
            f"{str(p.x_shape):>18}{str(p.rhs_indices_shape):>10}"
            f"{str(p.lhs_indices_shape):>10}"
            f"{p.bank_bytes / _GIB:>10.3f}{p.gathered_bytes / 1e6:>13.1f}"
            f"{('yes' if p.indices_wrapped else 'no'):>6}"
        )
    print()
    print("window command (run inside the exclusive GPU lock):")
    print(f"  {window_command(cfg, out=out)}")
    return 0


# ---------------------------------------------------------------------------
# Real timing path -- imports mlx lazily; runs on the GPU inside the window.
# ---------------------------------------------------------------------------
def _mx():
    import mlx.core as mx  # deferred: keeps --help/--dry-run mlx-free

    return mx


def _clear_cache(mx) -> None:
    for fn in (getattr(mx, "clear_cache", None),
               getattr(getattr(mx, "metal", None), "clear_cache", None)):
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            return


def build_bank(mx, plan: ArmPlan, rng) -> dict[str, Any]:
    """Random synthetic stacked bank for one arm (values irrelevant to timing)."""
    geoms = projection_geoms(
        hidden=plan.hidden, inter=plan.inter, codec=plan.codec, bits=plan.bits,
        group=plan.group,
    )
    bank: dict[str, Any] = {}
    for g in geoms:
        for comp, (dt, shape) in g.components.items():
            full = (plan.slots, *shape)
            key = f"{g.name}.{comp}"
            if dt == "uint32":
                bank[key] = mx.random.randint(0, 2**31 - 1, full, dtype=mx.uint32)
            elif dt == "uint8":
                # E8M0 exponents: keep them mid-range so dequant stays finite.
                bank[key] = mx.random.randint(110, 130, full).astype(mx.uint8)
            elif dt == "bfloat16":
                bank[key] = (rng(full) * 0.02).astype(mx.bfloat16)
            else:  # pragma: no cover
                raise ValueError(dt)
    return bank


class _BankShim:
    """Minimal stand-in for MlxComponentBank (only ``.arrays`` is read)."""

    def __init__(self, arrays: dict[str, Any]) -> None:
        self.arrays = arrays


def _slot_indices(mx, rows: int, slots: int):
    # Distinct experts when the bank holds >= rows slots; wrap otherwise.
    return mx.array([i % slots for i in range(rows)], dtype=mx.int32).reshape((-1, 1))


def make_runner(mx, plan: ArmPlan, arrays: dict[str, Any], rng):
    """Return a zero-arg callable that issues one full expert-MLP gather call."""
    bank = _BankShim(arrays)
    rows = plan.rows
    K = plan.hidden

    if plan.layout in ("switch_flat",):
        # Run the REAL production code path so the timed op IS the switch's call.
        from mtplx.models.expert_mlx import _gather_component_bank

        x = (rng((rows, K)) * 0.02).astype(mx.bfloat16)
        slot_indices = _slot_indices(mx, rows, plan.slots)

        def run():
            return _gather_component_bank(
                x,
                bank,
                slot_indices,
                group_size=plan.group,
                bits=plan.bits,
                swiglu_limit=SWIGLU_LIMIT,
                codec=plan.codec,
            )

        return run

    if plan.layout == "convention":
        from mlx_lm.models.activations import swiglu

        m, top_k = plan.m, plan.top_k
        x = (rng((m, 1, K)) * 0.02).astype(mx.bfloat16)  # [M,1,K], batch size M
        # gate/up: x batch is M -> lhs picks token m for each of its top_k experts
        # (output [M,top_k,1,N]); down: its input batch is M*top_k -> lhs is the
        # identity into that batch. rhs picks the expert slot per (token,k).
        lhs_gateup = mx.array([[mm] * top_k for mm in range(m)], dtype=mx.int32)
        lhs_down = mx.array(
            [[mm * top_k + k for k in range(top_k)] for mm in range(m)], dtype=mx.int32
        )
        rhs = mx.array(
            [[(mm * top_k + k) % plan.slots for k in range(top_k)] for mm in range(m)],
            dtype=mx.int32,
        )

        def qmm(values, proj, lhs):
            return mx.gather_qmm(
                values,
                arrays[f"{proj}.weight"],
                arrays[f"{proj}.scales"],
                lhs_indices=lhs,
                rhs_indices=rhs,
                transpose=True,
                group_size=plan.group,
                bits=plan.bits,
                mode="mxfp4",
            )

        def run():
            gate = qmm(x, "gate_proj", lhs_gateup)  # [M, top_k, 1, N_inter]
            up = qmm(x, "up_proj", lhs_gateup)
            up = mx.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
            gate = mx.minimum(gate, SWIGLU_LIMIT)
            h = swiglu(gate, up)  # batch [M, top_k]
            return qmm(h, "down_proj", lhs_down)

        return run

    # bf16_gather_mm bandwidth reference: gather_mm does a @ b (b=[slots,K,N]),
    # one [1,K]@[K,N] GEMV per row -- the mxfp4 gather's work with no dequant ALU.
    from mlx_lm.models.activations import swiglu

    x = (rng((rows, 1, 1, K)) * 0.02).astype(mx.bfloat16)  # [rows,1,1,K]
    lhs = mx.array([[r] for r in range(rows)], dtype=mx.int32)  # a's batch, in order
    rhs = _slot_indices(mx, rows, plan.slots)  # [rows,1] expert per row

    def gmm(values, proj):
        return mx.gather_mm(
            values, arrays[f"{proj}.weight"], lhs_indices=lhs, rhs_indices=rhs
        )

    def run():
        gate = gmm(x, "gate_proj")  # [rows,1,N_inter]
        up = gmm(x, "up_proj")
        hidden = mx.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        gate = mx.minimum(gate, SWIGLU_LIMIT)
        h = swiglu(gate, hidden)
        return gmm(h, "down_proj")

    return run


def time_arm(mx, plan: ArmPlan, iters: int, warmup: int, seed: int) -> dict[str, Any]:
    key = mx.random.key(seed + hash(plan.arm) % 100000 + plan.m)

    def rng(shape):
        nonlocal key
        key, sub = mx.random.split(key)
        return mx.random.normal(shape, key=sub)

    arrays = build_bank(mx, plan, rng)
    mx.eval(list(arrays.values()))
    run = make_runner(mx, plan, arrays, rng)

    for _ in range(max(1, warmup)):
        mx.eval(run())

    times_us: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(run())
        times_us.append((time.perf_counter() - t0) * 1e6)

    times_us.sort()
    median_us = times_us[len(times_us) // 2]
    gbps = plan.gathered_bytes / (median_us / 1e6) / 1e9 if median_us > 0 else 0.0
    peak = getattr(mx, "get_peak_memory", lambda: 0)()

    # Free before the next arm to respect the budget.
    del arrays, run
    _clear_cache(mx)

    return {
        **asdict(plan),
        "iters": iters,
        "warmup": warmup,
        "median_us": median_us,
        "min_us": times_us[0],
        "max_us": times_us[-1],
        "mean_us": sum(times_us) / len(times_us),
        "effective_gbps": gbps,
        "peak_gb_at_arm": peak / _GIB,
    }


def run_real(cfg: argparse.Namespace) -> int:
    if not cfg.out:
        print("ERROR: --out is required for the timing path", file=sys.stderr)
        return 2
    out = Path(cfg.out)
    if out.exists():
        # never-overwrite a measurement (memory/never-overwrite-a-measurement)
        print(f"ERROR: refusing to overwrite existing receipt {out}", file=sys.stderr)
        return 2

    mx = _mx()
    dev = mx.default_device()
    print(f"[microbench] default device = {dev}")

    plans = build_plans(cfg)
    results = [time_arm(mx, p, cfg.iters, cfg.warmup, cfg.seed) for p in plans]
    for r in results:
        print(
            f"[microbench] {r['arm']:<17} M={r['m']} rows={r['rows']} "
            f"slots={r['slots']} median={r['median_us']:.1f} us "
            f"GB/s={r['effective_gbps']:.1f} "
            f"peak={r['peak_gb_at_arm']:.2f} GB"
            f"{'  WRAPPED' if r['indices_wrapped'] else ''}"
        )

    receipt = {
        "bench": "W43_gather_qmm_microbench",
        "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "device": str(dev),
        "geometry": {
            "hidden": cfg.hidden,
            "inter": cfg.inter,
            "top_k": cfg.top_k,
            "n_routed_experts": N_ROUTED_EXPERTS,
            "mxfp4_group": cfg.group_size_mxfp4,
            "mxfp4_bits": MXFP4_BITS,
            "mxfp4_expert_record_bytes": MXFP4_EXPERT_RECORD_BYTES,
            "swiglu_limit": SWIGLU_LIMIT,
        },
        "config": vars(cfg),
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"[microbench] wrote {out}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="W43 DSV4.1 all-hit gather_qmm microbench (GPU-window timing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--m-values", nargs="+", type=int, default=[1, 4],
                   help="row batch M: 1=AR decode, 4=MTP verify")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--hidden", type=int, default=HIDDEN_SIZE)
    p.add_argument("--inter", type=int, default=MOE_INTERMEDIATE)
    p.add_argument("--bank-slots", type=int, default=92,
                   help="synthetic bank slot count (real resident slots/layer ~92)")
    p.add_argument("--group-size-mxfp4", type=int, default=MXFP4_GROUP)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--memory-limit-gib", type=float, default=8.0,
                   help="per-arm synthetic bank allocation budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                   help="JSON receipt path (required for the timing path; "
                        "refuses to overwrite)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the shape/byte plan + window command; no mlx, no GPU")
    return p


def main(argv: list[str] | None = None) -> int:
    cfg = build_parser().parse_args(argv)
    if cfg.dry_run:
        return dry_run(cfg)
    return run_real(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
