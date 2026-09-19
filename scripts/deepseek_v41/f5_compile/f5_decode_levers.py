"""F5 decode-only lever enablement at the quiescent post-prefill boundary.

The three "explicit_bound_model_levers" (HC_COMPILE, ATTN_COMPILE, ATTN_WIN_MEMO)
are FROZEN False at construction by
``scripts/deepseek_v41/bench_standard_shape.py:bind_model_levers`` -- it reads the
arm env once and ``setattr``s the module globals
``deepseek_v41._HC_COMPILE`` / ``_ATTN_COMPILE`` / ``_ATTN_WIN_MEMO`` (the arm
preset forces those env keys to "0" in run_full.py:507-510, and run_full.py:587
asserts the binding).  Because ``_hc_use_compile`` / ``_attn_use_compile`` read the
MODULE GLOBAL (not the env) at call time, an env change AFTER construction does not
move them -- ONLY writing the global does.  This mirrors the in-tree precedent
``deepseek_v41_dspark.py:_moe_compile_window`` (sets ``_dv41._ATTN_COMPILE = True``
around a scope) and the K22 unit test's ``_attn_flag`` (sets ``dv41._ATTN_COMPILE``).

The other candidate levers (SMALL_STAGES_FUSED, ATTN_CORE_COMPILE, HC_PREMIX_KERNEL,
DRAFT_COMPILE) are read at USE from the env (or a module override for DRAFT), so a
boundary env write engages them; their row caps (7/8/-/32) already keep them out of
the large prefill chunks regardless.

Enabling for DECODE ONLY, prefill left EXACTLY as retained:
  1. keep the arm env for the three bound levers at "0" so ``bind_model_levers``
     freezes them False and PREFILL runs identically (the 84-row / 110e9-budget
     prefill is unchanged -- crucial for ATTN_WIN_MEMO, whose per-forward retained
     window mask is the prefill-memory interaction Codex bound out for provenance);
  2. call :func:`enable_decode_levers` from the runner's post-prefill callback
     (run_full.py:768-785 ``observe_prefill_boundary``, AFTER ``callback(info)`` and
     ``growth_transition()``), which flips the module globals and/or env for the
     chosen set.  Prefill already completed, so nothing prefill-side changes.

This module performs NO MLX ops and is import-safe on CPU (the unit tests exercise
:func:`enable_decode_levers` on the tiny CPU model).
"""
from __future__ import annotations

import os
from typing import Iterable


# Env keys the window arms use to declare the F5 selection (read once at the
# boundary, so the same shim serves every arm).
F5_ENABLE_ENV = "MTPLX_DSV41_F5_ENABLE"        # csv of lever names, see _NAMES
F5_CAPS8_ENV = "MTPLX_DSV41_F5_CAPS8"          # "1" -> raise HC/SMALL_STAGES caps 7->8
F5_TIMED_PROBE_ENV = "MTPLX_DSV41_F5_TIMED_PROBE"   # "1" -> arm A2 stamp probe

# lever name -> (kind, target).  kind "global": write the import-bound module
# global.  kind "env": write the read-at-use env var.
_NAMES = {
    "hc_compile": ("global", "_HC_COMPILE"),
    "attn_compile": ("global", "_ATTN_COMPILE"),
    "attn_win_memo": ("global", "_ATTN_WIN_MEMO"),
    "small_stages": ("env", "MTPLX_DSV41_SMALL_STAGES_FUSED"),
    "attn_core_compile": ("env", "MTPLX_DSV41_ATTN_CORE_COMPILE"),
    "hc_premix_kernel": ("env", "MTPLX_DSV41_HC_PREMIX_KERNEL"),
    # sinkhorn_metal is a GPU rounding-class kernel the retained arm already runs
    # (arm_env MTPLX_DSV41_SINKHORN_METAL=1); listed so an arm can assert it stays on.
    "sinkhorn_metal": ("env", "MTPLX_DSV41_SINKHORN_METAL"),
}

# Exactness class per lever at the M<=8 verify regime, for the receipt (CPU tiny
# vs native width -- native is NOT bit-identical for the compiled HC arithmetic,
# per deepseek_v41.py:2769-2772 and the HC screen max|delta| 3.7e-4).
_EXACTNESS = {
    "hc_compile": "rounding-class at native width (not bit-identical; HC screen 3.7e-4); mx.array_equal on tiny CPU only",
    "attn_compile": "rounding-class at native width; mx.array_equal on tiny CPU only",
    "attn_win_memo": "byte-identical (pure host-dispatch mask reuse; same array object)",
    "small_stages": "rounding-class at native width (shares the compiled HC premix); mx.array_equal on tiny CPU only",
    "attn_core_compile": "rounding-class at native width; mx.array_equal on tiny CPU only",
    "hc_premix_kernel": "GPU-only; parity-gated 1e-6 + argmax-exact (test_hc_premix_kernel_parity_gpu); INERT on CPU",
    "sinkhorn_metal": "rounding-class Metal (K3); INERT on CPU",
}


def _truthy(value) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no", "off", "auto")


def parse_enable(spec: str | Iterable[str] | None) -> list[str]:
    """Parse a csv (or iterable) of lever names, validating against ``_NAMES``."""
    if spec is None:
        return []
    names = spec.split(",") if isinstance(spec, str) else list(spec)
    out = []
    for raw in names:
        name = raw.strip().lower()
        if not name:
            continue
        if name not in _NAMES:
            raise ValueError(f"unknown F5 lever {name!r}; choose from {sorted(_NAMES)}")
        out.append(name)
    return out


def enable_decode_levers(
    dv41,
    enable: str | Iterable[str] | None,
    *,
    caps_at_8: bool = False,
) -> dict:
    """Enable the chosen levers for DECODE, at the post-prefill quiescent boundary.

    ``dv41`` is the imported ``mtplx.models.deepseek_v41`` module.  Import-bound
    levers are set by writing the module global; read-at-use levers by writing the
    env.  ``caps_at_8`` is the SEPARATE Task-4 construction option -- it raises the
    HC and SMALL_STAGES row caps 7 -> 8 so the M=8 verify batch also compiles; this
    trades the byte-identical-on-tiny guarantee for rounding-class (the >=8 band
    reassociates ~5e-7..1.2e-6 on tiny; native width is already rounding-class), so
    it NEVER changes defaults and must be requested explicitly.  Returns a receipt
    dict describing exactly what was set (so ``arm_env`` is not the only evidence)."""
    chosen = parse_enable(enable)
    applied = {}
    for name in chosen:
        kind, target = _NAMES[name]
        if kind == "global":
            if not hasattr(dv41, target):
                raise RuntimeError(f"model module lacks lever global {target}")
            setattr(dv41, target, True)
            applied[name] = {"mechanism": "module_global", "target": target,
                             "value": bool(getattr(dv41, target))}
        else:  # env
            os.environ[target] = "1"
            applied[name] = {"mechanism": "env", "target": target, "value": "1"}
        applied[name]["exactness"] = _EXACTNESS[name]

    caps = {"hc_compile_max_rows": int(dv41._HC_COMPILE_MAX_ROWS),
            "small_stages_max_rows": int(dv41._SMALL_STAGES_MAX_ROWS)}
    if caps_at_8:
        dv41._HC_COMPILE_MAX_ROWS = 8
        dv41._SMALL_STAGES_MAX_ROWS = 8
        # tape caches keyed partly on shape -- clearing is harmless and avoids a
        # stale 7-row tape being reused for a newly admitted 8-row shape.
        dv41._HC_COMPILED.clear()
        dv41._SMALL_STAGES_COMPILED.clear()
        caps_after = {"hc_compile_max_rows": 8, "small_stages_max_rows": 8}
    else:
        caps_after = dict(caps)

    return {
        "enabled": chosen,
        "applied": applied,
        "caps_at_8": bool(caps_at_8),
        "caps_before": caps,
        "caps_after": caps_after,
        "caps_at_8_exactness": (
            "M=8 verify batch (>7) now compiles -> rounding-class (was eager/exact "
            "at M=8); the M<=7 rows are unchanged. Defaults untouched."
        ),
        "boundary": "post-prefill quiescent (observe_prefill_boundary), decode-only",
        "prefill_unchanged": (
            "bound levers stay env=0 -> bind_model_levers freezes them False for "
            "construction+prefill; globals flipped only after growth_transition()"
        ),
    }


def enable_from_env(dv41) -> dict:
    """Read ``MTPLX_DSV41_F5_ENABLE`` / ``MTPLX_DSV41_F5_CAPS8`` and apply.

    Intended as the one line a window shim calls inside the post-prefill callback:
    ``f5_decode_levers.enable_from_env(deepseek_v41)``.  A no-op if the enable env
    is empty (so an unarmed arm is exactly the retained control)."""
    spec = os.environ.get(F5_ENABLE_ENV)
    caps8 = _truthy(os.environ.get(F5_CAPS8_ENV))
    if not spec and not caps8:
        return {"enabled": [], "caps_at_8": False, "boundary_applied": False}
    report = enable_decode_levers(dv41, spec, caps_at_8=caps8)
    report["boundary_applied"] = True
    return report


def timed_probe_requested() -> bool:
    """Whether arm A2's ``TimedPackedDecode`` stamp probe is armed for this arm."""
    return _truthy(os.environ.get(F5_TIMED_PROBE_ENV))


F5_TIMED_OUT_ENV = "MTPLX_DSV41_F5_TIMED_OUT"   # receipt stem for the probe dump


def install_and_arm_probe(model, *, scheduled: bool) -> dict:
    """Arm A2: swap each retained ``PackedDecode`` runner for the timed lane and
    register the decode-end dump.

    Discovers the runners from the already-installed model
    (``layer.mlp.switch_mlp._run.__self__`` is the bound ``PackedDecode`` after
    plane_lane.install + projection_install), so it needs no runtime internals.
    MUST be called at the post-prefill boundary AFTER projection_install (so
    ``scheduled=True`` matches the installed ``self.issue_next()`` run body).  The
    dump fires at process exit to the ``MTPLX_DSV41_F5_TIMED_OUT`` stem."""
    import atexit

    import timed_plane_lane

    runners, switches = {}, {}
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        switch = getattr(mlp, "switch_mlp", None)
        run = getattr(switch, "_run", None)
        runner = getattr(run, "__self__", None)
        if runner is None:
            continue
        runners[i] = runner
        switches[i] = switch
    if not runners:
        raise RuntimeError("A2 probe: no PackedDecode runners found on the model")

    report = timed_plane_lane.install_timed(runners, switches, scheduled=scheduled)
    sink = report.pop("sink")
    out_stem = os.environ.get(F5_TIMED_OUT_ENV)
    if out_stem:
        n_layers = len(runners)
        atexit.register(
            lambda: timed_plane_lane.dump_probe(
                sink, out_stem, verify_forward_layers=n_layers
            )
        )
        report["dump_stem"] = out_stem
    report["armed_layers"] = len(runners)
    return report
