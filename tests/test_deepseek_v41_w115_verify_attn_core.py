"""W115: the honest verify-attention CORE A/B (`cell16k_ring_v2_draft_attn_eager`)
and the dspark-scoped engagement plumbing.

Premise correction (red-team): the DSpark verify is NOT running an eager attention
core by default.  ``ab_decode_env_levers._run_arm`` (and the served
``arm_dspark_decode_kernels``) ``os.environ.setdefault`` K29 (`DECODE_ATTN_KERNEL`) +
K30 (`SELECTED_KEYS`) to "1" for EVERY ``--decode-mode dspark`` arm, so
``cell16k_ring_v2_draft_attn`` runs the K+1 verify through the K29 fused
decode-attention core (window-43: ``arm_env.DECODE_ATTN_KERNEL="1"``,
``decode_attn_kernel_engagement.calls>0``).  The ~M× verify cost is STRUCTURAL (per-row
gathered sparse cores), not a missing lever.

So there is no auto-arm lever to add.  W115 instead ships:
  * `cell16k_ring_v2_draft_attn_eager` = `cell16k_ring_v2_draft_attn` with the verify
    core K29 pinned OFF (`decode_attn_kernel="0"` — the runtime knob, an explicit "0"
    beats the setdefault which only fills unset keys — AND `dspark_verify_k29="0"` so
    the setdefault drops K29 entirely).  Everything else identical, so the A/B isolates
    the verify SDPA core: K29 (draft_attn) vs the eager gathered core (this arm);
  * the dspark-scoped engagement plumbing so `fused_proj` / `decode_attn_kernel` /
    `attn_core_compile` counters cover the K+1 VERIFY rows (the top-level blocks are
    AR-scoped — the window-43 "rows = AR only" gap).

CPU tests, no Metal (the gate GPU-branch is spied — no dispatch), no model download,
<1.5 GB RSS.  MLX pinned to CPU (memory/worker-tests-must-pin-mlx-cpu.md).  Run under
``nice -n 19``, pytest one file per process (no ``-n auto``).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41 as V41  # noqa: E402
from mtplx.models.deepseek_v41_dspark_decode import (  # noqa: E402
    dspark_decode_kernel_env_defaults,
)

_WT = Path(__file__).resolve().parents[1]
_SCRIPTS = _WT / "scripts" / "deepseek_v41"
_K29_ENV = "MTPLX_DSV41_DECODE_ATTN_KERNEL"
_VK29_ENV = "MTPLX_DSV41_DSPARK_VERIFY_K29"
_SEL_ENV = "MTPLX_DSV41_SELECTED_KEYS"


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dsv41_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ab():
    return _load("ab_decode_env_levers")


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: v for k, v in os.environ.items() if k.startswith("MTPLX_DSV41_")}
    for k in list(saved):
        del os.environ[k]
    try:
        yield
    finally:
        for k in [k for k in os.environ if k.startswith("MTPLX_DSV41_")]:
            del os.environ[k]
        os.environ.update(saved)


def _apply_dspark_arm(ab, arm):
    """Mimic ``_run_arm`` for a ``--decode-mode dspark`` cell: apply the arm preset,
    then the K29/K30 setdefault the lane runs for every dspark arm."""
    for k in [k for k in os.environ if k.startswith("MTPLX_DSV41_")]:
        del os.environ[k]
    ab._apply_arm_env(arm)
    for k, v in dspark_decode_kernel_env_defaults().items():
        os.environ.setdefault(k, v)


def _use_on_gpu(monkeypatch, rows):
    """`_decode_attn_kernel_use` for a `rows`-wide verify batch on a spied GPU (the
    gate only reads is_available()/default_device(); no Metal is dispatched)."""
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    return V41._decode_attn_kernel_use(mx.zeros((1, rows, 8, 512)))


# ---------------------------------------------------------------------------
# 1. the honest A/B: verify core K29 ON (draft_attn) vs OFF (eager)
# ---------------------------------------------------------------------------
def test_eager_arm_turns_verify_core_off_under_dspark_defaults(ab, monkeypatch):
    assert "cell16k_ring_v2_draft_attn_eager" in ab.ARM_PRESETS

    # draft_attn: the setdefault arms K29 -> the verify runs the fused decode core.
    _apply_dspark_arm(ab, "cell16k_ring_v2_draft_attn")
    assert os.environ.get(_K29_ENV) == "1"          # setdefault armed it
    assert V41._resolve_decode_attn_kernel() is True
    assert _use_on_gpu(monkeypatch, 6) is True       # K+1=6 verify -> K29

    # eager: both knobs pinned off -> the setdefault cannot re-arm -> eager core.
    _apply_dspark_arm(ab, "cell16k_ring_v2_draft_attn_eager")
    assert os.environ.get(_K29_ENV) == "0"           # explicit 0 survived setdefault
    assert os.environ.get(_VK29_ENV) == "0"          # dropped from the defaults too
    assert os.environ.get(_SEL_ENV) == "1"           # SELECTED_KEYS still on (A/B holds all else)
    assert V41._resolve_decode_attn_kernel() is False
    assert _use_on_gpu(monkeypatch, 6) is False       # the verify runs the eager core
    assert _use_on_gpu(monkeypatch, 1) is False       # M=1 too (whole arm)


def test_dspark_defaults_drop_k29_only_when_verify_k29_off():
    """The knob that governs the setdefault: `DSPARK_VERIFY_K29=0` drops K29 from the
    lane defaults (K30 stays)."""
    if _VK29_ENV in os.environ:
        del os.environ[_VK29_ENV]
    assert dspark_decode_kernel_env_defaults() == {_SEL_ENV: "1", _K29_ENV: "1"}
    os.environ[_VK29_ENV] = "0"
    assert dspark_decode_kernel_env_defaults() == {_SEL_ENV: "1"}   # no K29


def test_eager_preset_pins_zero_not_none(ab):
    """`setdefault` only fills UNSET keys, so the eager preset must pin the RUNTIME
    knob to an explicit "0" (a None would be popped by `_apply_arm_env` and then
    re-armed to "1")."""
    preset = ab.ARM_PRESETS["cell16k_ring_v2_draft_attn_eager"]
    assert preset[_K29_ENV] == "0"
    assert preset[_VK29_ENV] == "0"
    # sanity: the base arm leaves it unset (None) -> the setdefault arms it.
    assert ab.ARM_PRESETS["cell16k_ring_v2_draft_attn"][_K29_ENV] is None


# ---------------------------------------------------------------------------
# 2. classification: an OFF-pinned lever is not a rounding reason
# ---------------------------------------------------------------------------
def test_eager_arm_rounding_class_does_not_credit_off_kernel(ab):
    keys = ab._rounding_class_keys("cell16k_ring_v2_draft_attn_eager")
    # still rounding-class (fused proj + bf16 draft head), but K29="0" is NOT a reason.
    assert ab._is_rounding_class("cell16k_ring_v2_draft_attn_eager") is True
    assert _K29_ENV not in keys, keys
    assert ab.ATTN_FUSED_PROJ_ENV in keys
    # and the base arm's preset also does not list K29 (it is setdefault-armed, not
    # preset-armed) -- fused proj / draft-head keep it rounding-class.
    assert _K29_ENV not in ab._rounding_class_keys("cell16k_ring_v2_draft_attn")


def test_off_values_not_counted_generally(ab):
    for off in (None, "", "0", "false", "off", "no"):
        ab.ARM_PRESETS.setdefault("_probe_off", dict(ab.ARM_PRESETS["control"]))
        ab.ARM_PRESETS["_probe_off"][ab.ATTN_FUSED_PROJ_ENV] = off
        assert ab.ATTN_FUSED_PROJ_ENV not in ab._rounding_class_keys("_probe_off"), off
    ab.ARM_PRESETS["_probe_off"][ab.ATTN_FUSED_PROJ_ENV] = "1"
    assert ab.ATTN_FUSED_PROJ_ENV in ab._rounding_class_keys("_probe_off")
    del ab.ARM_PRESETS["_probe_off"]


# ---------------------------------------------------------------------------
# 3. env-lever bookkeeping (drift guards)
# ---------------------------------------------------------------------------
def test_verify_k29_env_in_lever_lists(ab):
    assert ab.DSPARK_VERIFY_K29_ENV in ab.ALL_LEVER_ENVS
    from mtplx.server.openai import _DSV41_LEVER_ENV_KEYS
    assert ab.DSPARK_VERIFY_K29_ENV in _DSV41_LEVER_ENV_KEYS
    assert set(ab.ALL_LEVER_ENVS) <= set(_DSV41_LEVER_ENV_KEYS)   # superset guard
    # every preset pins the new key (None or a value), never missing.
    for arm, preset in ab.ARM_PRESETS.items():
        assert ab.DSPARK_VERIFY_K29_ENV in preset, arm


# ---------------------------------------------------------------------------
# 4. dspark-scoped engagement plumbing
# ---------------------------------------------------------------------------
def test_engagement_reset_and_capture(ab):
    ok = ab._reset_dspark_engagement_counters()
    assert ok.get("dsv41") and ok.get("k29") and ok.get("fp")
    # a fused-proj + core-compile call after the reset shows up in the capture.
    from mtplx.models import deepseek_v41_fused_proj_kernels as fp
    fp.note_qkv(6)
    fp.note_out()
    V41._note_attn_core_call(True)
    cap = ab._capture_dspark_engagement(ok)
    assert set(cap) == {
        "attn_core_compile_engagement",
        "decode_attn_kernel_engagement",
        "fused_proj_engagement",
    }
    assert cap["fused_proj_engagement"]["rows"] == 6
    assert cap["fused_proj_engagement"]["qkv_calls"] == 1
    assert cap["attn_core_compile_engagement"]["compiled"] == 1
