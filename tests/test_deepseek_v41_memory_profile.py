"""W62 -- DeepSeek-V4.1 explicit, self-limiting memory profile (CPU-only).

Pins four things the GPU window relies on, none of which need Metal or the
195 GB artifact:

1. The budget->plan derivation arithmetic (budget 100 -> plan 78 GiB) and its
   env/override paths and guards.
2. The snapshot helper's schema against MOCKED MLX allocator stats.
3. That the allocator cache-limit fix calls ``mx.set_cache_limit`` with the
   derived bytes.
4. That the served ``mtplx_openai_generation`` event carries peak/active/cache.

MLX is pinned to CPU only because the served-event helper lives in
``mtplx.server.openai``, which imports MLX at module scope.  Run under
``nice -n 19``; no ``pytest -n auto``.
"""

from __future__ import annotations

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest

from mtplx import deepseek_v41_memory_profile as mp

GIB = 1024**3


# --------------------------------------------------------------------------
# (1) Derivation arithmetic.
# --------------------------------------------------------------------------


def test_derivation_budget_100_gives_78_gib():
    # plan = budget(100) - macOS_floor(6) - host_overhead(10) - cache_limit(6).
    d = mp.derive_plan_from_budget(env={})
    assert d.source == "budget"
    assert d.box_budget_gib == 100.0
    assert d.plan_gib == pytest.approx(78.0)
    assert d.memory_limit_bytes == int(round(78.0 * GIB))
    assert d.cache_limit_bytes == int(round(6.0 * GIB))
    assert d.runtime_reserve_bytes == int(round(7.0 * GIB))
    # The formula string carries every term with its number.
    assert "budget(100)" in d.formula()
    assert "= 78 GiB" in d.formula()


def test_derivation_arithmetic_is_pure_subtraction():
    d = mp.derive_plan_from_budget(
        box_budget_gib=120.0,
        macos_floor_gib=6.0,
        host_overhead_gib=10.0,
        cache_limit_gib=6.0,
    )
    assert d.plan_gib == pytest.approx(120.0 - 6.0 - 10.0 - 6.0)
    assert d.plan_gib == pytest.approx(98.0)


def test_derivation_reads_box_budget_env():
    d = mp.derive_plan_from_budget(env={mp.ENV_BOX_BUDGET: "110"})
    # 110 - 6 - 10 - 6 = 88.
    assert d.plan_gib == pytest.approx(88.0)


def test_derivation_env_deduction_overrides():
    d = mp.derive_plan_from_budget(
        env={
            mp.ENV_BOX_BUDGET: "100",
            mp.ENV_MACOS_FLOOR: "8",
            mp.ENV_HOST_OVERHEAD: "12",
            mp.ENV_CACHE_LIMIT: "4",
        }
    )
    assert d.plan_gib == pytest.approx(100.0 - 8.0 - 12.0 - 4.0)
    assert d.cache_limit_bytes == int(round(4.0 * GIB))


def test_explicit_memory_limit_is_an_override():
    d = mp.derive_plan_from_budget(override_memory_limit_gib=82.0, env={})
    assert d.source == "override"
    assert d.plan_gib == pytest.approx(82.0)
    assert d.memory_limit_bytes == int(round(82.0 * GIB))
    # The overhang fix still applies to an override run.
    assert d.cache_limit_bytes == int(round(6.0 * GIB))
    assert "override" in d.formula()


def test_derivation_rejects_non_positive_plan():
    with pytest.raises(ValueError):
        mp.derive_plan_from_budget(box_budget_gib=10.0, env={})


def test_derivation_rejects_plan_below_reserve_plus_transient():
    # 20 - 6 - 10 - 6 = -2 -> already non-positive; use a case that is positive
    # but under reserve+transient: budget 24, floor 1, host 1, cache 1 -> 21;
    # reserve 7 + transient 0.57 fits, so pick a tighter one.
    with pytest.raises(ValueError):
        mp.derive_plan_from_budget(
            box_budget_gib=15.0,
            macos_floor_gib=1.0,
            host_overhead_gib=1.0,
            cache_limit_gib=6.0,
            env={},
        )


def test_derivation_as_dict_is_json_shaped():
    d = mp.derive_plan_from_budget(env={})
    row = d.as_dict()
    for key in (
        "source",
        "box_budget_gib",
        "macos_floor_gib",
        "host_overhead_gib",
        "cache_limit_gib",
        "plan_gib",
        "memory_limit_bytes",
        "cache_limit_bytes",
        "runtime_reserve_bytes",
        "formula",
    ):
        assert key in row


# --------------------------------------------------------------------------
# (2) Snapshot schema with mocked MLX stats.
# --------------------------------------------------------------------------


class _FakeMx:
    """Records set_cache_limit and returns fixed allocator stats."""

    def __init__(self, active=11 * GIB, cache=3 * GIB, peak=42 * GIB):
        self._active = active
        self._cache = cache
        self._peak = peak
        self.set_cache_limit_calls: list[int] = []
        self._prev_cache_limit = 99 * GIB

    def get_active_memory(self):
        return self._active

    def get_cache_memory(self):
        return self._cache

    def get_peak_memory(self):
        return self._peak

    def set_cache_limit(self, value):
        self.set_cache_limit_calls.append(int(value))
        return self._prev_cache_limit


def test_mlx_memory_snapshot_with_mock():
    fake = _FakeMx()
    snap = mp.mlx_memory_snapshot(mx_module=fake)
    assert snap["ok"] is True
    assert snap["active_bytes"] == 11 * GIB
    assert snap["cache_bytes"] == 3 * GIB
    assert snap["peak_bytes"] == 42 * GIB


def test_memory_profile_snapshot_schema():
    fake = _FakeMx()
    d = mp.derive_plan_from_budget(env={})
    snap = mp.memory_profile_snapshot(
        phase="load_end", token=None, mx_module=fake, derivation=d
    )
    assert snap["phase"] == "load_end"
    assert snap["token"] is None
    # MLX sub-snapshot from the mock.
    assert snap["mlx"]["active_bytes"] == 11 * GIB
    # Process + box sub-snapshots always present (real host; best-effort).
    assert "process" in snap and "peak_maxrss_bytes" in snap["process"]
    assert "box" in snap
    # Derivation echoed only where requested.
    assert snap["derivation"]["plan_gib"] == pytest.approx(78.0)


def test_memory_profile_snapshot_per_token_label():
    fake = _FakeMx()
    snap = mp.memory_profile_snapshot(phase="decode", token=128, mx_module=fake)
    assert snap["phase"] == "decode"
    assert snap["token"] == 128
    assert "derivation" not in snap


def test_process_rss_snapshot_is_psutil_free():
    snap = mp.process_rss_snapshot()
    assert "peak_maxrss_bytes" in snap
    assert snap["peak_maxrss_bytes"] > 0
    # On this darwin box the mach reader returns a current resident figure.
    assert snap["source"] in {"mach_task_vm_info", "getrusage_only"}


def test_format_memory_profile_table_renders():
    fake = _FakeMx()
    d = mp.derive_plan_from_budget(env={})
    rows = [
        mp.memory_profile_snapshot(phase="load_end", mx_module=fake, derivation=d),
        mp.memory_profile_snapshot(phase="after_prefill", mx_module=fake),
        mp.memory_profile_snapshot(phase="decode", token=64, mx_module=fake),
    ]
    table = mp.format_memory_profile_table(rows)
    assert "load_end" in table
    assert "after_prefill" in table
    assert "decode" in table
    assert "derivation:" in table


# --------------------------------------------------------------------------
# (3) Cache-limit call.
# --------------------------------------------------------------------------


def test_apply_allocator_cache_limit_calls_set_cache_limit_with_bytes():
    fake = _FakeMx()
    d = mp.derive_plan_from_budget(env={})
    report = mp.apply_allocator_cache_limit(d.cache_limit_bytes, mx_module=fake)
    assert report["applied"] is True
    assert fake.set_cache_limit_calls == [d.cache_limit_bytes]
    assert report["cache_limit_bytes"] == d.cache_limit_bytes
    assert report["previous_cache_limit_bytes"] == 99 * GIB


def test_apply_allocator_cache_limit_rejects_negative():
    with pytest.raises(ValueError):
        mp.apply_allocator_cache_limit(-1, mx_module=_FakeMx())


def test_apply_allocator_cache_limit_no_setter_is_soft():
    class _NoSetter:
        pass

    report = mp.apply_allocator_cache_limit(6 * GIB, mx_module=_NoSetter())
    assert report["applied"] is False
    assert report["cache_limit_bytes"] == 6 * GIB


# --------------------------------------------------------------------------
# (2b) Plan breakdown from a real spec (no model load, no artifact).
# --------------------------------------------------------------------------


def test_plan_breakdown_from_real_spec():
    from mtplx.expert_streaming_models import get_model_spec, plan_expert_memory

    spec = get_model_spec("deepseek-v41-flash-expert-q2")
    d = mp.derive_plan_from_budget(env={})
    plan = plan_expert_memory(
        spec,
        total_limit_bytes=d.memory_limit_bytes,
        context_tokens=1024,
        runtime_reserve_bytes=d.runtime_reserve_bytes,
    )
    bd = mp.plan_breakdown(plan)
    assert bd["total_limit_bytes"] == d.memory_limit_bytes
    assert bd["runtime_reserve_bytes"] == d.runtime_reserve_bytes
    assert bd["kv_planned_bytes"] == 1024 * spec.kv_bytes_per_token
    # residents + reserve + transient + kv all sit inside the fixed side.
    assert bd["fits_fixed"] is True
    assert bd["expert_cache_planned_bytes"] > 0
    # engram host LRU is reported and is NOT folded into resident_bytes.
    assert bd["engram_host_lru_bytes"] is not None
    assert bd["engram_host_lru_bytes"] > 0


# --------------------------------------------------------------------------
# (3b) Budget child-env propagation.
# --------------------------------------------------------------------------


def test_budget_child_env_stamps_when_unset():
    added = mp.budget_child_env({})
    assert added == {mp.ENV_BOX_BUDGET: "100"}


def test_budget_child_env_respects_operator_value():
    added = mp.budget_child_env({mp.ENV_BOX_BUDGET: "120"})
    assert added == {}


# --------------------------------------------------------------------------
# (4) Served event carries the memory fields.
# --------------------------------------------------------------------------


def test_generation_event_memory_fields_from_stats():
    from mtplx.server.openai import _generation_event_memory_fields

    fields = _generation_event_memory_fields(
        {
            "peak_memory_bytes": 42 * GIB,
            "active_memory_bytes": 30 * GIB,
            "cache_memory_bytes": 5 * GIB,
            "other": "ignored",
        }
    )
    assert fields == {
        "peak_memory_bytes": 42 * GIB,
        "active_memory_bytes": 30 * GIB,
        "cache_memory_bytes": 5 * GIB,
    }


def test_generation_event_memory_fields_partial_stats():
    from mtplx.server.openai import _generation_event_memory_fields

    fields = _generation_event_memory_fields({"peak_memory_bytes": 7})
    assert fields == {"peak_memory_bytes": 7}


def test_generation_event_memory_fields_falls_back_to_live_read():
    # Empty stats -> a fresh _mlx_allocator_public_stats() read; on CPU MLX the
    # accessors exist and return ints (>= 0), so the keys are present.
    from mtplx.server.openai import _generation_event_memory_fields

    fields = _generation_event_memory_fields({})
    for key in ("peak_memory_bytes", "active_memory_bytes", "cache_memory_bytes"):
        assert key in fields
        assert isinstance(fields[key], int)
