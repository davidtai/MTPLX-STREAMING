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
    # plan = 100 - 6 - 2 - 4 - 3 - 6 - 0 = 79
    assert bt.plan_limit_gib == pytest.approx(79.0)
    # forecast = baseline + plan + overshoot + overhead + headroom = 6+79+6+2+0 = 93
    assert bt.forecast_system_peak_gib() == pytest.approx(93.0)
    keys = bt.memory_keys()
    assert keys["mlx_limit_headroom_gib"] == pytest.approx(0.0)
    assert keys["mlx_limit_gib_effective"] == pytest.approx(79.0)


def test_derive_prices_headroom_into_forecast_and_keys() -> None:
    ab = _load_ab_module()
    bt = _derive(ab, 8.0)
    # On the derive-from-budget path the headroom comes OFF the plan so the forecast
    # still fits: plan = 100 - 6 - 2 - 4 - 3 - 6 - 8 = 71.
    assert bt.plan_limit_gib == pytest.approx(71.0)
    assert bt.mlx_limit_headroom_gib == pytest.approx(8.0)
    # forecast = 6 + 71 + 6 + 2 + 8 = 93 = budget - kv - safety (invariant preserved).
    assert bt.forecast_system_peak_gib() == pytest.approx(93.0)
    assert bt.forecast_system_peak_gib() <= 100.0
    keys = bt.memory_keys()
    assert keys["mlx_limit_headroom_gib"] == pytest.approx(8.0)
    # effective allocator limit = plan_eff + headroom = 71 + 8 = 79.
    assert keys["mlx_limit_gib_effective"] == pytest.approx(79.0)
    assert keys["budget_forecast_system_peak_gb"] == pytest.approx(93.0)
    # "mlx_limit_headroom" appears in the human formula only when non-zero.
    assert "mlx_limit_headroom(8" in bt.formula()


def test_forecast_rises_by_headroom_on_a_pinned_plan() -> None:
    """The PINNED-plan A/B path keeps plan_limit fixed and ADDS the headroom on top:
    the forecast box peak then rises by exactly the headroom (what the pin validator
    checks against the budget)."""

    ab = _load_ab_module()
    base = _derive(ab, 0.0)  # plan 79, forecast 93
    pinned_hr = base.replace(mlx_limit_headroom_gib=8.0)
    # plan_limit UNCHANGED (residents identical -> byte-identical A/B) ...
    assert pinned_hr.plan_limit_gib == pytest.approx(79.0)
    # ... but the forecast rises by the headroom.
    assert pinned_hr.forecast_system_peak_gib() == pytest.approx(93.0 + 8.0)
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
    ab = _load_ab_module()
    with pytest.raises(ValueError, match="non-negative"):
        ab._resolve_mlx_limit_headroom_gib(
            types.SimpleNamespace(mlx_limit_headroom_gib=-2.0)
        )
