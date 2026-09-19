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
    """Captures the values handed to ``set_memory_limit`` and ``set_wired_limit``
    (W121: the DSV4.1 cap now wires the working set too) -- no Metal touched."""

    def __init__(self) -> None:
        self.value: int | None = None
        self.wired: int | None = None

    def set_memory_limit(self, value: int) -> int:
        prev = self.value
        self.value = int(value)
        return 0 if prev is None else prev

    def set_wired_limit(self, value: int) -> int:
        prev = self.wired
        self.wired = int(value)
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

    # W121: the report now also carries the wired-limit outcome (the DSV4.1 cap
    # wires the Metal working set to the SAME value it hands set_memory_limit, so
    # its IOAccelerator pages land in wire_count instead of the swappable LRU).
    assert report == {
        "applied": True,
        "limit": plan_limit,
        "limit_source": "plan",
        "wired_limit_applied": True,
        "wired_limit_bytes": plan_limit,
        "wired_limit_api": "mx.set_wired_limit",
        "previous_wired_limit_bytes": 0,
    }
    assert fake.value == plan_limit
    assert fake.wired == plan_limit  # wired == the allocation cap
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
    # W121: the wired limit tracks the SAME effective allocator limit (plan + headroom).
    assert report == {
        "applied": True,
        "limit": plan_limit + 8 * GIB,
        "limit_source": "plan",
        "wired_limit_applied": True,
        "wired_limit_bytes": plan_limit + 8 * GIB,
        "wired_limit_api": "mx.set_wired_limit",
        "previous_wired_limit_bytes": 0,
    }
    assert fake.wired == plan_limit + 8 * GIB
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
# Part F: the headroom resolvers reject bad values (runtime + harness)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["nan", "inf", "1_0", "abc", "-1"])
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

