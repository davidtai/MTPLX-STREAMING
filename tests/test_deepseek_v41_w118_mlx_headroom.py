"""W118 (H7): the MLX allocator-limit headroom lever.

A lever that raises ONLY the MLX allocator soft limit (``mx.set_memory_limit``) by N
GiB ABOVE the residency plan, WITHOUT changing what is resident or the expert-cache
slot plan -- so bytes/routing/outputs stay byte-identical.  See
``docs/deepseek-v41/W118_MLX_LIMIT_HEADROOM.md``.

CPU-only: a fake ``mx`` (no Metal, no model load) exercises ``apply_mlx_memory_cap``;
the budget/forecast + preset assertions are pure math on a file-path load of the A/B
script (``scripts/`` is not a package), mirroring the W46/W90 drift-guard test.
"""

from __future__ import annotations

import importlib.util
import tempfile
import types
from pathlib import Path

import pytest

from mtplx.expert_runtime import (
    ExpertStreamingConfigurationError,
    MLX_LIMIT_HEADROOM_ENV,
    apply_mlx_memory_cap,
    reconcile_mlx_memory_cap,
    resolve_mlx_limit_headroom_bytes,
)

GIB = 1024**3
_HEADROOM_ENV = "MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB"


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


def _fake_plan(
    *,
    total_limit_bytes: int = 100_000,
    runtime_reserve_bytes: int = 10_000,
    io_staging_bytes: int = 5_000,
) -> types.SimpleNamespace:
    """A duck-typed ExpertMemoryPlan carrying only the fields the memory-cap
    reconciliation reads.  ``mmap_islands_wired=True`` keeps the paged-band branch
    out (no mmap island in the DSV4.1 plan), so reconcile = total - reserve - io."""

    return types.SimpleNamespace(
        total_limit_bytes=total_limit_bytes,
        runtime_reserve_bytes=runtime_reserve_bytes,
        io_staging_bytes=io_staging_bytes,
        mmap_islands_wired=True,
        mmap_island_bytes=0,
        fixed_bytes=40_000,
        persistent_cache_bytes=20_000,
    )


class _FakeMX:
    """Captures the single value handed to ``set_memory_limit`` -- no Metal touched."""

    def __init__(self) -> None:
        self.value: int | None = None

    def set_memory_limit(self, value: int) -> int:
        prev = self.value
        self.value = int(value)
        return 0 if prev is None else prev


def _load_ab_module():
    ab_path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "deepseek_v41" / "ab_decode_env_levers.py"
    )
    spec = importlib.util.spec_from_file_location("dsv41_ab_env_levers_w118", ab_path)
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    return ab


# --------------------------------------------------------------------------
# Part A: resolve_mlx_limit_headroom_bytes (read at use, default 0)
# --------------------------------------------------------------------------


def test_headroom_env_name_is_canonical() -> None:
    assert MLX_LIMIT_HEADROOM_ENV == _HEADROOM_ENV


def test_resolve_headroom_default_zero() -> None:
    assert resolve_mlx_limit_headroom_bytes(env={}) == 0
    assert resolve_mlx_limit_headroom_bytes(env={_HEADROOM_ENV: ""}) == 0
    assert resolve_mlx_limit_headroom_bytes(env={_HEADROOM_ENV: "  "}) == 0


def test_resolve_headroom_parses_gib() -> None:
    assert resolve_mlx_limit_headroom_bytes(env={_HEADROOM_ENV: "8"}) == 8 * GIB
    assert resolve_mlx_limit_headroom_bytes(env={_HEADROOM_ENV: "0"}) == 0
    assert resolve_mlx_limit_headroom_bytes(env={_HEADROOM_ENV: "0.5"}) == int(
        round(0.5 * GIB)
    )


def test_resolve_headroom_rejects_bad_values() -> None:
    with pytest.raises(ExpertStreamingConfigurationError, match="non-negative"):
        resolve_mlx_limit_headroom_bytes(env={_HEADROOM_ENV: "-1"})
    with pytest.raises(ExpertStreamingConfigurationError, match="number of GiB"):
        resolve_mlx_limit_headroom_bytes(env={_HEADROOM_ENV: "lots"})


# --------------------------------------------------------------------------
# Part B: apply_mlx_memory_cap -- headroom raises ONLY the set_memory_limit value
# --------------------------------------------------------------------------


def test_default_headroom_zero_is_today() -> None:
    """Env unset (default) -> the value passed to set_memory_limit is the plan limit
    exactly, and the report shape is unchanged (backward compatible)."""

    plan = _fake_plan()
    plan_limit = 100_000 - 10_000 - 5_000  # 85_000
    assert reconcile_mlx_memory_cap(plan, env={}) == plan_limit

    fake = _FakeMX()
    env: dict[str, str] = {}
    report = apply_mlx_memory_cap(plan, mx_module=fake, env=env)

    assert report == {"applied": True, "limit": plan_limit}
    assert fake.value == plan_limit
    assert env["MTPLX_MEMORY_LIMIT_BYTES"] == str(plan_limit)


def test_headroom_raises_only_the_allocator_limit() -> None:
    """With headroom=8: set_memory_limit gets plan + 8 GiB, but the plan-derived
    engine budget (MTPLX_MEMORY_LIMIT_BYTES) and the reconcile result are UNCHANGED --
    the proof that residents / expert-cache slots do not move."""

    plan = _fake_plan()
    plan_limit = 85_000

    # reconcile NEVER reads the headroom -- the plan limit is identical with or without
    # the env set (residents/slots are planned from this value, so they cannot move).
    assert reconcile_mlx_memory_cap(plan, env={_HEADROOM_ENV: "8"}) == plan_limit
    assert reconcile_mlx_memory_cap(plan, env={}) == plan_limit

    fake = _FakeMX()
    env = {_HEADROOM_ENV: "8"}
    report = apply_mlx_memory_cap(plan, mx_module=fake, env=env)

    assert fake.value == plan_limit + 8 * GIB
    assert report == {"applied": True, "limit": plan_limit + 8 * GIB}
    # The engine budget that bounds residency stays the PLAN value, not plan+headroom.
    assert env["MTPLX_MEMORY_LIMIT_BYTES"] == str(plan_limit)


def test_headroom_does_not_mutate_the_plan() -> None:
    """apply_mlx_memory_cap never writes back to the plan object (no field moves)."""

    plan = _fake_plan()
    before = dict(vars(plan))
    apply_mlx_memory_cap(plan, mx_module=_FakeMX(), env={_HEADROOM_ENV: "8"})
    assert dict(vars(plan)) == before


def test_headroom_via_metal_namespace_setter() -> None:
    """Falls back to mx.metal.set_memory_limit when mx has no top-level setter."""

    plan = _fake_plan()
    captured: dict[str, int] = {}

    class _Metal:
        @staticmethod
        def set_memory_limit(value: int) -> int:
            captured["value"] = int(value)
            return 0

    mx = types.SimpleNamespace(metal=_Metal())
    report = apply_mlx_memory_cap(plan, mx_module=mx, env={_HEADROOM_ENV: "2"})
    assert captured["value"] == 85_000 + 2 * GIB
    assert report["limit"] == 85_000 + 2 * GIB


# --------------------------------------------------------------------------
# Part C: A/B harness registration + presets
# --------------------------------------------------------------------------


def test_env_registered_in_all_lever_envs_and_preset() -> None:
    ab = _load_ab_module()
    assert ab.MLX_LIMIT_HEADROOM_ENV == _HEADROOM_ENV
    assert _HEADROOM_ENV in ab.ALL_LEVER_ENVS

    # _preset names the key (None = force-unset) so applying an arm fully determines it.
    default = ab._preset()
    assert _HEADROOM_ENV in default
    assert default[_HEADROOM_ENV] is None
    assert ab._preset(mlx_limit_headroom="8")[_HEADROOM_ENV] == "8"


def test_served_log_snapshot_is_superset_of_ab_levers() -> None:
    """W46/W90 drift guard extended to W118: the served-log lever snapshot
    (_DSV41_LEVER_ENV_KEYS) must contain the new headroom env, and stay a superset of
    ALL_LEVER_ENVS."""

    from mtplx.server.openai import _DSV41_LEVER_ENV_KEYS

    ab = _load_ab_module()
    assert _HEADROOM_ENV in _DSV41_LEVER_ENV_KEYS
    missing = set(ab.ALL_LEVER_ENVS) - set(_DSV41_LEVER_ENV_KEYS)
    assert not missing, f"A/B levers absent from the served-log snapshot: {sorted(missing)}"


def test_hr8_arms_are_base_arms_plus_headroom_only() -> None:
    """cell16k_ring_v2_attn_hr8 == cell16k_ring_v2_attn with headroom=8 and NOTHING
    else changed (byte-identity of every other lever key); same for the DSpark arm."""

    ab = _load_ab_module()
    for base_name, hr_name in (
        ("cell16k_ring_v2_attn", "cell16k_ring_v2_attn_hr8"),
        ("cell16k_ring_v2_draft_attn", "cell16k_ring_v2_draft_attn_hr8"),
    ):
        base = ab.ARM_PRESETS[base_name]
        hr = ab.ARM_PRESETS[hr_name]
        assert hr[_HEADROOM_ENV] == "8"
        assert base[_HEADROOM_ENV] is None
        # Every OTHER key is identical -> the A/B isolates only the headroom.
        assert {k: v for k, v in hr.items() if k != _HEADROOM_ENV} == {
            k: v for k, v in base.items() if k != _HEADROOM_ENV
        }


# --------------------------------------------------------------------------
# Part D: budget derivation prices the headroom (forecast + sidecar + receipt keys)
# --------------------------------------------------------------------------


def _derive(ab, headroom):
    return ab.derive_budget_total_plan(
        budget_total_gb=100.0,
        system_used_at_start_gb=6.0,
        non_metal_overhead_gb=2.0,
        kv_growth_to_max_kv_gb=4.0,
        safety_gb=3.0,
        plan_overshoot_gib=6.0,
        floor_gib=10.0,
        mlx_limit_headroom_gib=headroom,
    )


def test_derive_default_headroom_zero_unchanged() -> None:
    ab = _load_ab_module()
    bt = _derive(ab, 0.0)
    assert bt.mlx_limit_headroom_gib == 0.0
    # headroom 0 -> allocator_extra = max(overshoot 6, min(0, 6+cache 6)) = 6, so the
    # lever-off derivation is unchanged: plan = 100 - 6 - 2 - 4 - 3 - 6 = 79.
    assert bt.plan_limit_gib == pytest.approx(79.0)
    # forecast = baseline + plan + overhead + allocator_extra = 6+79+2+6 = 93.
    assert bt.forecast_system_peak_gib() == pytest.approx(93.0)
    keys = bt.memory_keys()
    assert keys["mlx_limit_headroom_gib"] == pytest.approx(0.0)
    assert keys["mlx_limit_gib_effective"] == pytest.approx(79.0)
    assert keys["budget_headroom_forecast_extra_gib"] == pytest.approx(6.0)
    assert keys["budget_cache_limit_gib"] == pytest.approx(6.0)


def test_derive_prices_headroom_into_forecast_and_keys() -> None:
    ab = _load_ab_module()
    bt = _derive(ab, 8.0)
    # review HIGH-1: allocator_extra = max(overshoot 6, min(headroom 8, 6+cache 6=12))
    # = 8 (NOT overshoot + headroom = 14).  On the derive path the extra term comes off
    # the plan: plan = 100 - 6 - 2 - 4 - 3 - allocator_extra(8) = 77 (drops by 2, not 8).
    assert bt.plan_limit_gib == pytest.approx(77.0)
    assert bt.mlx_limit_headroom_gib == pytest.approx(8.0)
    assert bt.headroom_forecast_extra_gib() == pytest.approx(8.0)
    # forecast = 6 + 77 + 2 + allocator_extra(8) = 93 = budget - kv - safety (invariant).
    assert bt.forecast_system_peak_gib() == pytest.approx(93.0)
    assert bt.forecast_system_peak_gib() <= 100.0
    keys = bt.memory_keys()
    assert keys["mlx_limit_headroom_gib"] == pytest.approx(8.0)
    # effective allocator limit handed to set_memory_limit = plan_eff + headroom = 85.
    assert keys["mlx_limit_gib_effective"] == pytest.approx(85.0)
    assert keys["budget_forecast_system_peak_gb"] == pytest.approx(93.0)
    assert keys["budget_headroom_forecast_extra_gib"] == pytest.approx(8.0)
    # the human formula shows the allocator_extra term (not overshoot + headroom).
    assert "allocator_extra(8" in bt.formula()
    assert "headroom 8" in bt.formula()


def test_forecast_rises_by_extra_minus_overshoot_on_a_pinned_plan() -> None:
    """The PINNED-plan A/B path keeps plan_limit fixed and ADDS the headroom on top.
    review HIGH-1: the forecast box peak then rises by (allocator_extra - overshoot),
    NOT by the full headroom -- raising the soft limit lets the allocator RETAIN cache,
    it does not add a fresh headroom GiB.  With overshoot 6, cache 6, headroom 8 the
    extra term is 8, so the rise is 8 - 6 = 2 (forecast 93 -> 95)."""

    ab = _load_ab_module()
    base = _derive(ab, 0.0)  # plan 79, forecast 93, allocator_extra 6
    pinned_hr = base.replace(mlx_limit_headroom_gib=8.0)
    # plan_limit UNCHANGED (residents identical -> byte-identical A/B) ...
    assert pinned_hr.plan_limit_gib == pytest.approx(79.0)
    assert pinned_hr.headroom_forecast_extra_gib() == pytest.approx(8.0)
    # ... the forecast rises by extra - overshoot = 2 (NOT the full headroom 8).
    assert pinned_hr.forecast_system_peak_gib() == pytest.approx(95.0)
    assert pinned_hr.forecast_system_peak_gib() == pytest.approx(
        base.forecast_system_peak_gib() + 2.0
    )
    keys = pinned_hr.memory_keys()
    # plan_limit_gib_effective is equal for control vs hr8 (the plan-equality guard);
    # only mlx_limit_gib_effective differs.
    assert keys["plan_limit_gib_derived"] == pytest.approx(79.0)
    assert keys["mlx_limit_gib_effective"] == pytest.approx(79.0 + 8.0)


def test_headroom_survives_sidecar_roundtrip() -> None:
    ab = _load_ab_module()
    bt = _derive(ab, 8.0)
    restored = ab.BudgetTotalDerivation.from_plan_dict(bt.to_plan_dict())
    assert restored.mlx_limit_headroom_gib == pytest.approx(8.0)
    assert restored.plan_limit_gib == pytest.approx(bt.plan_limit_gib)
    assert restored.forecast_system_peak_gib() == pytest.approx(
        bt.forecast_system_peak_gib()
    )


def test_derive_rejects_negative_headroom() -> None:
    ab = _load_ab_module()
    with pytest.raises(ValueError, match="mlx_limit_headroom_gib must be non-negative"):
        _derive(ab, -1.0)


def test_explicit_path_forecast_none_but_effective_includes_headroom() -> None:
    """The explicit (non-budget) derivation has no budget to forecast against, so the
    forecast term is None; the effective allocator limit still surfaces the headroom
    (matching what the runtime hands set_memory_limit)."""

    ab = _load_ab_module()
    bt = ab._explicit_plan_derivation(69.0).replace(mlx_limit_headroom_gib=8.0)
    assert bt.forecast_system_peak_gib() is None
    keys = bt.memory_keys()
    assert keys["budget_forecast_system_peak_gb"] is None
    assert keys["plan_limit_gib_effective"] == pytest.approx(69.0)
    assert keys["mlx_limit_gib_effective"] == pytest.approx(69.0 + 8.0)


# --------------------------------------------------------------------------
# Part E: _resolve_mlx_limit_headroom_gib flag/env precedence
# --------------------------------------------------------------------------


def test_resolve_arm_headroom_flag_over_env(monkeypatch) -> None:
    ab = _load_ab_module()
    monkeypatch.setenv(_HEADROOM_ENV, "8")
    # Explicit flag wins over the preset env.
    assert ab._resolve_mlx_limit_headroom_gib(
        types.SimpleNamespace(mlx_limit_headroom_gib=4.0)
    ) == pytest.approx(4.0)
    # No flag -> the env the preset stamped is used.
    assert ab._resolve_mlx_limit_headroom_gib(
        types.SimpleNamespace(mlx_limit_headroom_gib=None)
    ) == pytest.approx(8.0)


def test_resolve_arm_headroom_default_zero(monkeypatch) -> None:
    ab = _load_ab_module()
    monkeypatch.delenv(_HEADROOM_ENV, raising=False)
    assert ab._resolve_mlx_limit_headroom_gib(
        types.SimpleNamespace(mlx_limit_headroom_gib=None)
    ) == pytest.approx(0.0)


def test_resolve_arm_headroom_rejects_negative() -> None:
    # MEDIUM-3: the harness resolver raises a CLEAN SystemExit (not a traceback) so a
    # bad value fails before the in-window crash.
    ab = _load_ab_module()
    with pytest.raises(SystemExit):
        ab._resolve_mlx_limit_headroom_gib(
            types.SimpleNamespace(mlx_limit_headroom_gib=-2.0)
        )


# --------------------------------------------------------------------------
# Part F: review HIGH-1 -- window-44 pin accepts hr8 at budget 93 (no double-count)
# --------------------------------------------------------------------------


def _pinned_bt(ab, *, headroom):
    """A window-44-shaped pinned derivation (plan 69, budget 93)."""
    return ab.BudgetTotalDerivation(
        source="budget",
        budget_total_gb=93.0,
        system_used_at_start_gb=8.7,
        non_metal_overhead_gb=6.0,
        kv_growth_to_max_kv_gb=0.72,
        safety_gb=3.0,
        plan_overshoot_gib=6.0,
        floor_gib=20.0,
        plan_limit_gib=69.0,
        cache_limit_gib=6.0,
        mlx_limit_headroom_gib=headroom,
    )


def test_w44_pin_accepts_hr8_at_budget_93() -> None:
    """review HIGH-1: at the pinned window-44 plan (69) and budget 93, the hr8 forecast
    with the CORRECT allocator extra term is 91.7 <= 93 (ACCEPTED); the old
    overshoot+headroom double-count was 97.7 > 93 and would have REFUSED the documented
    run.  Also exercises _validate_pinned_plan end-to-end (no raise)."""

    ab = _load_ab_module()
    bt = _pinned_bt(ab, headroom=8.0)
    # extra = max(6, min(8, 6+6)) = 8; forecast = 8.7 + 69 + 6.0 + 8 = 91.7 <= 93.
    assert bt.headroom_forecast_extra_gib() == pytest.approx(8.0)
    assert bt.forecast_system_peak_gib() == pytest.approx(91.7)
    assert bt.forecast_system_peak_gib() <= 93.0
    # the OLD (double-counting) forecast would have been over budget:
    naive = 8.7 + 69.0 + 6.0 + 6.0 + 8.0  # baseline + plan + overshoot + overhead + hr
    assert naive == pytest.approx(97.7)
    assert naive > 93.0

    # End-to-end: _validate_pinned_plan ACCEPTS the pin (returns the live baseline).
    import json as _json

    with tempfile.TemporaryDirectory() as d:
        model = Path(d)
        (model / "config.json").write_text(_json.dumps({"num_hidden_layers": 1}))
        args = ab.build_parser().parse_args([
            "--out", str(model / "out.jsonl"),
            "--model", str(model),
            "--memory-budget-total-gib", "93",
            "--context-tokens", "16384",
            "--max-kv", "16704",
        ])
        args._dsv41_system_used_at_start_bytes = int(8.7 * GIB)  # live baseline = 8.7
        stamp = {
            "config_sha": ab._config_sha(model),
            "max_kv": 16704,
            "context_tokens": 16384,
            "budget_total_gb": 93.0,
        }
        live = ab._validate_pinned_plan(
            args, types.SimpleNamespace(), 16704, bt, stamp
        )
        assert live == pytest.approx(8.7)


def test_w44_pin_refuses_hr_that_does_not_fit() -> None:
    """A headroom large enough to push the forecast over budget IS refused (the guard
    still fires -- the fix removed the double-count, not the guard)."""

    ab = _load_ab_module()
    import json as _json

    bt = _pinned_bt(ab, headroom=20.0)  # extra = max(6, min(20, 12)) = 12 -> forecast +12
    with tempfile.TemporaryDirectory() as d:
        model = Path(d)
        (model / "config.json").write_text(_json.dumps({"num_hidden_layers": 1}))
        args = ab.build_parser().parse_args([
            "--out", str(model / "out.jsonl"),
            "--model", str(model),
            "--memory-budget-total-gib", "93",
            "--context-tokens", "16384",
            "--max-kv", "16704",
        ])
        args._dsv41_system_used_at_start_bytes = int(8.7 * GIB)
        stamp = {
            "config_sha": ab._config_sha(model),
            "max_kv": 16704,
            "context_tokens": 16384,
            "budget_total_gb": 93.0,
        }
        # forecast = 8.7 + 69 + 6 + 12 = 95.7 > 93 -> refuse.
        with pytest.raises(ValueError, match="would exceed the budget"):
            ab._validate_pinned_plan(args, types.SimpleNamespace(), 16704, bt, stamp)


# --------------------------------------------------------------------------
# Part G: review MEDIUM-3 finite guards (runtime raises config error, harness SysExit)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1_0", "abc"])
def test_runtime_resolver_rejects_nonfinite_and_obfuscated(bad) -> None:
    with pytest.raises(ExpertStreamingConfigurationError):
        resolve_mlx_limit_headroom_bytes(env={_HEADROOM_ENV: bad})


@pytest.mark.parametrize("bad", ["nan", "inf", "1_0", "abc", "-1"])
def test_harness_resolver_systemexit_on_bad_env(monkeypatch, bad) -> None:
    ab = _load_ab_module()
    monkeypatch.setenv(_HEADROOM_ENV, bad)
    with pytest.raises(SystemExit):
        ab._resolve_mlx_limit_headroom_gib(
            types.SimpleNamespace(mlx_limit_headroom_gib=None)
        )


def test_harness_resolver_systemexit_on_nan_flag() -> None:
    ab = _load_ab_module()
    with pytest.raises(SystemExit):
        ab._resolve_mlx_limit_headroom_gib(
            types.SimpleNamespace(mlx_limit_headroom_gib=float("nan"))
        )


# --------------------------------------------------------------------------
# Part H: review MEDIUM-2 allocator readback proof keys (fake mx)
# --------------------------------------------------------------------------


def test_readback_keys_from_fake_mx() -> None:
    ab = _load_ab_module()

    class _Metal:
        @staticmethod
        def device_info():
            return {"max_recommended_working_set_size": int(80 * GIB)}

    class _MX:
        metal = _Metal()

        @staticmethod
        def get_memory_limit():
            return int(90 * GIB)

    keys = ab._mlx_headroom_readback_keys(
        _MX(),
        mlx_peak_bytes=int(94 * GIB),
        active_start_bytes=int(70 * GIB),
        active_end_bytes=int(75 * GIB),
        cache_end_bytes=int(1 * GIB),
    )
    assert keys["mlx_limit_gib_readback"] == pytest.approx(90.0)
    # gc_limit = min(readback 90, 0.95 * 80 = 76) = 76.
    assert keys["mlx_gc_limit_gib_effective"] == pytest.approx(76.0)
    assert keys["mlx_active_gb_at_decode_start"] == pytest.approx(70.0)
    assert keys["mlx_active_gb_at_decode_end"] == pytest.approx(75.0)
    assert keys["mlx_cache_gb_at_decode_end"] == pytest.approx(1.0)
    # peak_over_limit = peak 94 - readback 90 = 4 (POSITIVE -> went over the soft limit).
    assert keys["mlx_peak_over_limit_gb"] == pytest.approx(4.0)


def test_readback_keys_none_when_getters_absent() -> None:
    ab = _load_ab_module()
    keys = ab._mlx_headroom_readback_keys(
        types.SimpleNamespace(),  # no getters at all
        mlx_peak_bytes=int(10 * GIB),
        active_start_bytes=None,
        active_end_bytes=None,
        cache_end_bytes=None,
    )
    assert keys["mlx_limit_gib_readback"] is None
    assert keys["mlx_gc_limit_gib_effective"] is None
    assert keys["mlx_peak_over_limit_gb"] is None


# --------------------------------------------------------------------------
# Part I: review HIGH-2 preflight prices preset-carried headroom
# --------------------------------------------------------------------------


def test_preflight_headroom_reads_preset_carried_value() -> None:
    ab = _load_ab_module()
    # The preset carries 8 on the hr8 arm; the pre-flight (which runs BEFORE the preset
    # env is applied) must price max(flag, max preset headroom over the arms) = 8.
    args = types.SimpleNamespace(
        mlx_limit_headroom_gib=None,
        arms=["cell16k_ring_v2_attn", "cell16k_ring_v2_attn_hr8"],
    )
    assert ab._preflight_headroom_gib(args) == pytest.approx(8.0)


def test_preflight_headroom_flag_beats_preset() -> None:
    ab = _load_ab_module()
    args = types.SimpleNamespace(
        mlx_limit_headroom_gib=12.0,
        arms=["cell16k_ring_v2_attn_hr8"],
    )
    assert ab._preflight_headroom_gib(args) == pytest.approx(12.0)


def test_preflight_headroom_zero_when_no_hr_arm() -> None:
    ab = _load_ab_module()
    args = types.SimpleNamespace(
        mlx_limit_headroom_gib=None,
        arms=["cell16k_ring_v2_attn"],
    )
    assert ab._preflight_headroom_gib(args) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Part J: review MEDIUM-1 auto-pin arms >=2 to arm-1's sidecar (same plan_limit)
# --------------------------------------------------------------------------


class _FakeBench:
    def __init__(self, system_used_bytes: int):
        self._sys = int(system_used_bytes)

    def _system_used_bytes(self) -> int:
        return self._sys


def test_multiarm_derive_autopins_second_arm_to_first_plan(monkeypatch) -> None:
    """review MEDIUM-1: two arms in ONE invocation on the derive path must run the SAME
    plan_limit.  Arm 1 (control, headroom 0) derives + writes the sidecar; arm 2 (hr8)
    auto-pins it, so arm 2's plan_limit == arm 1's even though its headroom differs (a
    per-arm re-derive would have shifted the plan and confounded the A/B)."""

    import json as _json

    ab = _load_ab_module()
    with tempfile.TemporaryDirectory() as d:
        model = Path(d)
        (model / "config.json").write_text(_json.dumps(
            {"num_hidden_layers": 1, "head_dim": 512, "qk_rope_head_dim": 64,
             "index_head_dim": 128, "sliding_window": 128, "compress_ratios": [0]}
        ))
        args = ab.build_parser().parse_args([
            "--out", str(model / "out.jsonl"),
            "--model", str(model),
            "--memory-budget-total-gib", "100",
            "--context-tokens", "16384",
            "--max-kv", "1000",
        ])
        args._dsv41_system_used_at_start_bytes = int(20 * GIB)
        bench = _FakeBench(int(20 * GIB))

        # Arm 1 (control): derives + writes the sidecar, records it for auto-pin.
        monkeypatch.delenv(_HEADROOM_ENV, raising=False)
        ab._resolve_derivation(args, bench=bench, max_kv=1000)
        p0 = args._dsv41_budget_total.plan_limit_gib
        assert args._dsv41_budget_total.mlx_limit_headroom_gib == pytest.approx(0.0)
        assert getattr(args, "_dsv41_autopin_sidecar", None) is not None

        # Arm 2 (hr8): the preset would set the env in-window; simulate it here.
        monkeypatch.setenv(_HEADROOM_ENV, "8")
        ab._resolve_derivation(args, bench=bench, max_kv=1000)
        bt2 = args._dsv41_budget_total
        # SAME plan_limit as arm 1 (auto-pinned, residents identical) ...
        assert bt2.plan_limit_gib == pytest.approx(p0)
        # ... but this arm carries its own headroom (applied at set_memory_limit).
        assert bt2.mlx_limit_headroom_gib == pytest.approx(8.0)
