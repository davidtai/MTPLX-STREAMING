"""CPU-pinned unit tests for the W106 item-3 --memory-budget-total-gb derivation.

Covers ``scripts/deepseek_v41/ab_decode_env_levers.py``:

  * ``derive_budget_total_plan`` -- the pure derivation
        plan_limit = total - system_used_at_start - non_metal_overhead
                           - kv_growth_to_max_kv - safety
    exercised with INJECTED measurements (no vm_stat, no MLX, no model), plus the
    floor refusal;
  * ``BudgetTotalDerivation.memory_keys`` -- the exact receipt ``memory``-block
    keys (budget + explicit paths);
  * ``_kv_bytes_at_max_kv`` -- the LOCAL KV growth estimator (now the FALLBACK)
    against small fake configs;
  * ``_kv_growth_estimate`` -- prefers the exact W107 helper
    (``mtplx.models.deepseek_v41_cache.kv_bytes_at_max_kv``), falls back to the
    local estimator when that import is unavailable, and reports which ran
    (``budget_kv_estimator``);
  * ``_read_kv_config_dims`` -- flat and ``text_config``-nested ``config.json``;
  * ``_resolve_derivation`` -- the budget path end-to-end with an injected
    system-used baseline and a temp ``config.json`` (still no MLX/model).

MLX is imported ONLY to pin the default device to CPU
(memory/worker-tests-must-pin-mlx-cpu.md: MLX defaults to Metal). No GPU, no
model, no server, no network. Run under ``nice -n 19`` and WITHOUT ``pytest -n
auto`` (host-encode sensitivity).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import pytest

# HARD rule: pin MLX to CPU before anything can touch Metal.
mx.set_default_device(mx.cpu)

_WT = Path(__file__).resolve().parents[1]
_SCRIPTS = _WT / "scripts" / "deepseek_v41"

GIB = 1024**3

_EXPECTED_MEMORY_KEYS = {
    "memory_plan_source",
    "budget_total_gb",
    "plan_limit_gib_derived",
    "plan_limit_gib_effective",
    "budget_system_used_at_start_gb",
    "budget_non_metal_overhead_gb",
    "budget_non_metal_overhead_measured_gb",
    "budget_kv_growth_to_max_kv_gb",
    "budget_kv_estimator",
    "budget_safety_gb",
    "budget_floor_gib",
}


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dsv41_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mod():
    return _load("ab_decode_env_levers")


# --------------------------------------------------------------------------
# pure derivation math + floor
# --------------------------------------------------------------------------


def test_derivation_math_with_injected_measurements():
    mod = _mod()
    d = mod.derive_budget_total_plan(
        budget_total_gb=100.0,
        system_used_at_start_gb=20.0,
        non_metal_overhead_gb=10.0,
        kv_growth_to_max_kv_gb=2.0,
        safety_gb=3.0,
        floor_gib=20.0,
    )
    # 100 - 20 - 10 - 2 - 3 = 65
    assert d.plan_limit_gib == pytest.approx(65.0)
    assert d.source == "budget"
    assert d.non_metal_overhead_measured_gb is None  # not measured yet (pre-load)
    # the formula string names every term
    for term in ("budget_total", "system_used_at_start", "non_metal_overhead",
                 "kv_growth_to_max_kv", "safety"):
        assert term in d.formula()


def test_floor_refusal_raises_clear_error():
    mod = _mod()
    # 50 - 20 - 10 - 2 - 3 = 15, below the 20 floor -> refuse.
    with pytest.raises(ValueError) as exc:
        mod.derive_budget_total_plan(
            budget_total_gb=50.0,
            system_used_at_start_gb=20.0,
            non_metal_overhead_gb=10.0,
            kv_growth_to_max_kv_gb=2.0,
            safety_gb=3.0,
            floor_gib=20.0,
        )
    msg = str(exc.value)
    assert "below the floor" in msg.lower()
    assert "--memory-budget-total-gib" in msg  # actionable


def test_negative_term_rejected():
    mod = _mod()
    with pytest.raises(ValueError, match="non-negative"):
        mod.derive_budget_total_plan(
            budget_total_gb=100.0,
            system_used_at_start_gb=-1.0,
            non_metal_overhead_gb=10.0,
            kv_growth_to_max_kv_gb=2.0,
        )


def test_exactly_at_floor_is_allowed():
    mod = _mod()
    # 55 - 20 - 10 - 2 - 3 = 20 == floor -> allowed (not below).
    d = mod.derive_budget_total_plan(
        budget_total_gb=55.0,
        system_used_at_start_gb=20.0,
        non_metal_overhead_gb=10.0,
        kv_growth_to_max_kv_gb=2.0,
        safety_gb=3.0,
        floor_gib=20.0,
    )
    assert d.plan_limit_gib == pytest.approx(20.0)


# --------------------------------------------------------------------------
# receipt memory-block keys
# --------------------------------------------------------------------------


def test_memory_keys_budget_path_has_all_terms():
    mod = _mod()
    d = mod.derive_budget_total_plan(
        budget_total_gb=100.0,
        system_used_at_start_gb=20.0,
        non_metal_overhead_gb=10.0,
        kv_growth_to_max_kv_gb=2.0,
        safety_gb=3.0,
        floor_gib=20.0,
        kv_estimator="w107",
    )
    keys = d.memory_keys()
    assert set(keys) == _EXPECTED_MEMORY_KEYS
    assert keys["memory_plan_source"] == "budget"
    assert keys["budget_total_gb"] == 100.0
    assert keys["plan_limit_gib_derived"] == 65.0
    assert keys["budget_system_used_at_start_gb"] == 20.0
    assert keys["budget_non_metal_overhead_gb"] == 10.0
    assert keys["budget_non_metal_overhead_measured_gb"] is None
    assert keys["budget_kv_growth_to_max_kv_gb"] == 2.0
    assert keys["budget_kv_estimator"] == "w107"
    assert keys["budget_safety_gb"] == 3.0
    assert keys["budget_floor_gib"] == 20.0


def test_memory_keys_explicit_path_nulls_the_budget_terms():
    mod = _mod()
    e = mod._explicit_plan_derivation(70.0)
    keys = e.memory_keys()
    assert set(keys) == _EXPECTED_MEMORY_KEYS  # SAME key set, self-describing
    assert keys["memory_plan_source"] == "explicit"
    assert keys["budget_total_gb"] is None
    assert keys["plan_limit_gib_derived"] == 70.0
    assert keys["plan_limit_gib_effective"] == 70.0
    assert keys["budget_non_metal_overhead_measured_gb"] is None
    assert keys["budget_kv_estimator"] is None  # no KV growth priced on this path


# --------------------------------------------------------------------------
# local KV estimator + config reader
# --------------------------------------------------------------------------


def test_kv_estimator_window_only_when_no_kv_source_layers():
    mod = _mod()
    dims = {
        "num_hidden_layers": 2,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "window_size": 128,
        "compress_ratios": [0, 0],  # no kv-source layers
    }
    got = mod._kv_bytes_at_max_kv(dims, 1000)
    # only the window ring, priced conservatively at max_kv rows.
    assert got == 2 * (1000 * 512 * 2)


def test_kv_estimator_ratio_one_prices_every_lane():
    mod = _mod()
    dims = {
        "num_hidden_layers": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "window_size": 128,
        "compress_ratios": [1],  # kv-source, no pooling
    }
    got = mod._kv_bytes_at_max_kv(dims, 1000)
    window = 1000 * 512 * 2
    latent = 1000 * 512 * 2
    rope = 1000 * 64 * 2
    index = 1000 * 128 * 2
    assert got == window + latent + rope + index


def test_kv_estimator_ratio_pools_rows_by_ceil():
    mod = _mod()
    dims = {
        "num_hidden_layers": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "window_size": 128,
        "compress_ratios": [4],
    }
    got = mod._kv_bytes_at_max_kv(dims, 1000)
    rows = -(-1000 // 4)  # ceil(1000/4) = 250
    window = 1000 * 512 * 2
    latent = rows * 512 * 2
    rope = rows * 64 * 2
    index = rows * 128 * 2
    assert got == window + latent + rope + index


def test_kv_estimator_zero_max_kv_is_zero():
    mod = _mod()
    assert mod._kv_bytes_at_max_kv(dict(mod._KV_CONFIG_DEFAULTS), 0) == 0


def test_kv_estimator_honors_kv_source_layer_ids():
    # W106 LOW-1: only the layers in kv_source_layer_ids hold the compressed/index
    # lanes; every layer keeps the window ring.
    mod = _mod()
    dims = {
        "num_hidden_layers": 4, "head_dim": 512, "qk_rope_head_dim": 64,
        "index_head_dim": 128, "window_size": 128,
        "compress_ratios": [0, 0, 4, 0], "kv_source_layer_ids": [2],
    }
    got = mod._kv_bytes_at_max_kv(dims, 1000)
    window = 4 * (1000 * 512 * 2)          # every layer
    rows = -(-1000 // 4)                    # layer 2 ratio 4 -> ceil = 250
    src = rows * 512 * 2 + rows * 64 * 2 + rows * 128 * 2
    assert got == window + src


def test_kv_estimator_kv_source_ids_override_compress_ratios():
    # kv_source_layer_ids is authoritative: layer 0 has compress_ratios=1 but is NOT
    # in kv_source_layer_ids, so it is window-only.
    mod = _mod()
    dims = {
        "num_hidden_layers": 2, "head_dim": 512, "qk_rope_head_dim": 64,
        "index_head_dim": 128, "window_size": 128,
        "compress_ratios": [1, 1], "kv_source_layer_ids": [1],
    }
    got = mod._kv_bytes_at_max_kv(dims, 1000)
    window = 2 * (1000 * 512 * 2)
    src = 1000 * 512 * 2 + 1000 * 64 * 2 + 1000 * 128 * 2  # only layer 1, ratio 1
    assert got == window + src


def test_kv_estimator_default_config_prices_only_four_source_layers():
    # The released default kv_source_layer_ids has 4 entries -> only 4 of the 40
    # layers get the compressed/index lanes (LOW-1 regression: not all 38/40).
    mod = _mod()
    dims = dict(mod._KV_CONFIG_DEFAULTS)  # 40 layers, kv_source_layer_ids=[2,8,14,20]
    got = mod._kv_bytes_at_max_kv(dims, 1000)
    window = 40 * (1000 * 512 * 2)
    rows = 1000  # compress_ratios empty -> ratio defaults to 1
    per_src = rows * 512 * 2 + rows * 64 * 2 + rows * 128 * 2
    assert got == window + 4 * per_src


# --------------------------------------------------------------------------
# _kv_growth_estimate: prefer the exact W107 helper, fall back to local
# --------------------------------------------------------------------------


def test_kv_growth_estimate_prefers_w107():
    mod = _mod()
    dims = dict(mod._KV_CONFIG_DEFAULTS)
    kv_bytes, estimator = mod._kv_growth_estimate(dims, 1000)
    assert estimator == "w107"
    # delegates EXACTLY to mtplx.models.deepseek_v41_cache.kv_bytes_at_max_kv,
    # reading the flat dims dict via a SimpleNamespace.
    from types import SimpleNamespace

    from mtplx.models.deepseek_v41_cache import kv_bytes_at_max_kv

    assert kv_bytes == kv_bytes_at_max_kv(SimpleNamespace(**dims), 1000)


def test_kv_growth_estimate_falls_back_to_local_when_import_fails(monkeypatch):
    mod = _mod()
    import sys
    import types as _types

    # A stand-in cache module WITHOUT kv_bytes_at_max_kv: the ``from ... import``
    # inside _kv_growth_estimate raises ImportError -> the local estimator runs.
    fake = _types.ModuleType("mtplx.models.deepseek_v41_cache")
    monkeypatch.setitem(sys.modules, "mtplx.models.deepseek_v41_cache", fake)
    dims = dict(mod._KV_CONFIG_DEFAULTS)
    kv_bytes, estimator = mod._kv_growth_estimate(dims, 1000)
    assert estimator == "local"
    assert kv_bytes == mod._kv_bytes_at_max_kv(dims, 1000)


def test_read_config_dims_flat(tmp_path):
    mod = _mod()
    cfg = {
        "num_hidden_layers": 3,
        "head_dim": 256,
        "qk_rope_head_dim": 32,
        "index_head_dim": 64,
        "sliding_window": 64,
        "compress_ratios": [0, 2, 4],
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    dims = mod._read_kv_config_dims(tmp_path)
    assert dims["num_hidden_layers"] == 3
    assert dims["head_dim"] == 256
    assert dims["compress_ratios"] == [0, 2, 4]


def test_read_config_dims_text_config_nested_wins(tmp_path):
    mod = _mod()
    cfg = {
        "model_type": "deepseek_v41",
        "head_dim": 999,  # top-level decoy
        "text_config": {
            "num_hidden_layers": 5,
            "head_dim": 512,
            "index_head_dim": 128,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    dims = mod._read_kv_config_dims(tmp_path)
    assert dims["num_hidden_layers"] == 5
    assert dims["head_dim"] == 512  # text_config key wins over the top-level decoy


def test_read_config_dims_missing_file_falls_back_to_defaults(tmp_path):
    mod = _mod()
    dims = mod._read_kv_config_dims(tmp_path / "does-not-exist")
    assert dims == mod._KV_CONFIG_DEFAULTS


# --------------------------------------------------------------------------
# _resolve_derivation integration (still CPU-only: injected baseline + temp cfg)
# --------------------------------------------------------------------------


class _FakeBench:
    def __init__(self, system_used_bytes: int):
        self._sys = int(system_used_bytes)

    def _system_used_bytes(self) -> int:
        return self._sys


def _budget_args(mod, tmp_path, **over):
    """A parsed args namespace for the budget path (real defaults via build_parser)."""
    cfg = {"num_hidden_layers": 1, "head_dim": 512, "qk_rope_head_dim": 64,
           "index_head_dim": 128, "sliding_window": 128, "compress_ratios": [0]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    argv = [
        "--out", str(tmp_path / "out.jsonl"),
        "--model", str(tmp_path),
        "--memory-budget-total-gib", "100",
        "--max-kv", "1000",
    ]
    args = mod.build_parser().parse_args(argv)
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_resolve_derivation_budget_path(tmp_path):
    mod = _mod()
    args = _budget_args(mod, tmp_path)
    # Inject the system-used baseline so the test is deterministic (no vm_stat).
    args._dsv41_system_used_at_start_bytes = int(20 * GIB)
    bench = _FakeBench(int(20 * GIB))

    derivation = mod._resolve_derivation(args, bench=bench, max_kv=1000)

    bt = args._dsv41_budget_total
    assert bt is not None and bt.source == "budget"
    # W107 merge: the plan prices KV growth with the exact per-lane helper
    # (mtplx.models.deepseek_v41_cache.kv_bytes_at_max_kv), recorded as
    # budget_kv_estimator == "w107".  The config has no kv_source_layer_ids, so the
    # only lane priced is the bounded window ring (independent of max_kv).
    dims = mod._read_kv_config_dims(tmp_path)
    kv_bytes, estimator = mod._kv_growth_estimate(dims, 1000)
    assert estimator == "w107"
    kv_gb = kv_bytes / GIB
    assert bt.kv_estimator == "w107"
    # non_metal_overhead default = 10, safety default = 3.
    expected = 100.0 - 20.0 - 10.0 - kv_gb - 3.0
    assert bt.plan_limit_gib == pytest.approx(expected)
    # the W62 derivation the loader consumes uses the derived limit as its plan.
    assert derivation.plan_gib == pytest.approx(expected)
    keys = mod._budget_memory_keys(args)
    assert keys["memory_plan_source"] == "budget"
    assert keys["budget_total_gb"] == 100.0
    assert keys["budget_kv_growth_to_max_kv_gb"] == pytest.approx(round(kv_gb, 4))
    assert keys["budget_kv_estimator"] == "w107"


def test_resolve_derivation_budget_overrides_memory_limit(tmp_path):
    mod = _mod()
    args = _budget_args(mod, tmp_path, memory_limit_gib=42.0)
    args._dsv41_system_used_at_start_bytes = int(20 * GIB)
    bench = _FakeBench(int(20 * GIB))
    derivation = mod._resolve_derivation(args, bench=bench, max_kv=1000)
    # --memory-budget-total-gib overrides --memory-limit-gib (derives, not literal).
    assert args._dsv41_budget_total.source == "budget"
    assert derivation.plan_gib != pytest.approx(42.0)


def test_resolve_derivation_explicit_path(tmp_path):
    mod = _mod()
    args = _budget_args(mod, tmp_path, memory_budget_total_gib=None, memory_limit_gib=70.0)
    derivation = mod._resolve_derivation(args, bench=None, max_kv=None)
    assert derivation.plan_gib == pytest.approx(70.0)
    bt = args._dsv41_budget_total
    assert bt.source == "explicit"
    keys = mod._budget_memory_keys(args)
    assert keys["memory_plan_source"] == "explicit"
    assert keys["budget_total_gb"] is None
    assert keys["plan_limit_gib_derived"] == pytest.approx(70.0)


def test_resolve_derivation_floor_refusal_propagates(tmp_path):
    mod = _mod()
    # budget 40 - 20 - 10 - kv - 3 < 20 floor -> refuse.
    args = _budget_args(mod, tmp_path, memory_budget_total_gib=40.0)
    args._dsv41_system_used_at_start_bytes = int(20 * GIB)
    bench = _FakeBench(int(20 * GIB))
    with pytest.raises(ValueError, match="(?i)below the floor"):
        mod._resolve_derivation(args, bench=bench, max_kv=1000)


# --------------------------------------------------------------------------
# HIGH-1: _remeasure_non_metal_overhead must NEVER call mx.set_memory_limit
# post-load, must measure the CURRENT footprint (phys_footprint), and must ABORT
# (not lower the limit) when the measured overhead blows the budget.
# --------------------------------------------------------------------------


class _FakeMx:
    def __init__(self, active_bytes):
        self._active = int(active_bytes)
        self.set_memory_limit_calls = []
        self.metal = None

    def get_active_memory(self):
        return self._active

    def set_memory_limit(self, n):  # must NEVER be called post-load
        self.set_memory_limit_calls.append(int(n))
        return 0


def _budget_bt(mod, *, estimate_gib):
    return mod.BudgetTotalDerivation(
        source="budget",
        budget_total_gb=93.0,
        system_used_at_start_gb=20.0,
        non_metal_overhead_gb=float(estimate_gib),
        kv_growth_to_max_kv_gb=2.0,
        safety_gb=3.0,
        floor_gib=20.0,
        plan_limit_gib=50.0,
    )


def _patch_footprint(mod, monkeypatch, footprint_gib):
    import mtplx.deepseek_v41_memory_profile as mp

    monkeypatch.setattr(
        mp, "process_rss_snapshot",
        lambda: {"phys_footprint_bytes": int(footprint_gib * GIB),
                 "resident_bytes": int(footprint_gib * GIB),
                 "peak_maxrss_bytes": int(footprint_gib * GIB)},
    )


def test_remeasure_never_calls_set_memory_limit_and_aborts_over_budget(monkeypatch):
    mod = _mod()
    # active 70 GiB, footprint 90 GiB -> measured overhead 20 GiB, estimate 10 ->
    # overage 10 GiB > tolerance -> ABORT, and set_memory_limit NEVER called.
    _patch_footprint(mod, monkeypatch, 90.0)
    mx = _FakeMx(active_bytes=int(70 * GIB))

    class _A:
        pass

    args = _A()
    args._dsv41_budget_total = _budget_bt(mod, estimate_gib=10.0)
    with pytest.raises(RuntimeError, match="(?i)ABORT .*before decode"):
        mod._remeasure_non_metal_overhead(args, mx)
    assert mx.set_memory_limit_calls == []  # the core HIGH-1 assertion


def test_remeasure_within_tolerance_records_measured_no_limit_change(monkeypatch):
    mod = _mod()
    # active 70 GiB, footprint 80 GiB -> measured 10 == estimate 10 -> no abort.
    _patch_footprint(mod, monkeypatch, 80.0)
    mx = _FakeMx(active_bytes=int(70 * GIB))

    class _A:
        pass

    args = _A()
    args._dsv41_budget_total = _budget_bt(mod, estimate_gib=10.0)
    mod._remeasure_non_metal_overhead(args, mx)  # no raise
    bt = args._dsv41_budget_total
    assert bt.non_metal_overhead_measured_gb == pytest.approx(10.0, abs=0.01)
    assert bt.plan_limit_gib_effective == pytest.approx(50.0)  # NEVER lowered
    assert mx.set_memory_limit_calls == []


def test_remeasure_footprint_unavailable_keeps_estimate(monkeypatch):
    mod = _mod()
    import mtplx.deepseek_v41_memory_profile as mp
    monkeypatch.setattr(mp, "process_rss_snapshot",
                        lambda: {"phys_footprint_bytes": None, "resident_bytes": None})
    mx = _FakeMx(active_bytes=int(70 * GIB))

    class _A:
        pass

    args = _A()
    args._dsv41_budget_total = _budget_bt(mod, estimate_gib=10.0)
    mod._remeasure_non_metal_overhead(args, mx)  # no raise, no crash
    bt = args._dsv41_budget_total
    assert bt.non_metal_overhead_measured_gb is None
    assert bt.plan_limit_gib_effective == pytest.approx(50.0)
    assert mx.set_memory_limit_calls == []


def test_remeasure_noop_for_explicit_plan():
    mod = _mod()
    mx = _FakeMx(active_bytes=int(70 * GIB))

    class _A:
        pass

    args = _A()
    args._dsv41_budget_total = mod._explicit_plan_derivation(50.0)
    mod._remeasure_non_metal_overhead(args, mx)  # explicit -> noop
    assert mx.set_memory_limit_calls == []


# --------------------------------------------------------------------------
# HIGH-2: the receipt memory block carries an rss_semantics_note stating that
# RSS-vs-mlx_peak on Metal is unverified.
# --------------------------------------------------------------------------


def test_memory_block_extra_keys_carry_rss_semantics_note(tmp_path):
    mod = _mod()
    args = _budget_args(mod, tmp_path)
    args._dsv41_system_used_at_start_bytes = int(20 * GIB)
    mod._resolve_derivation(args, bench=_FakeBench(int(20 * GIB)), max_kv=1000)
    extra = mod._memory_block_extra_keys(args)
    assert "rss_semantics_note" in extra
    assert "UNVERIFIED" in extra["rss_semantics_note"]
    assert "mlx_peak_gb" in extra["rss_semantics_note"]
    # the budget keys are still present alongside the note
    assert extra["memory_plan_source"] == "budget"


def test_memory_block_extra_keys_explicit_still_has_note():
    mod = _mod()

    class _A:
        pass

    args = _A()
    args._dsv41_budget_total = mod._explicit_plan_derivation(50.0)
    extra = mod._memory_block_extra_keys(args)
    assert extra["memory_plan_source"] == "explicit"
    assert "rss_semantics_note" in extra


# --------------------------------------------------------------------------
# MEDIUM-1: flags are GiB (-gib canonical); -gb is a deprecated alias that
# converts decimal GB -> GiB at the boundary.
# --------------------------------------------------------------------------


def test_gb_to_gib_conversion():
    mod = _mod()
    # 100 decimal GB = 100e9 bytes = 100e9 / 2**30 GiB ~= 93.13 GiB.
    assert mod._gb_to_gib(100.0) == pytest.approx(93.1322574, abs=1e-4)


def test_deprecated_gb_alias_converts(tmp_path):
    mod = _mod()
    cfg = {"num_hidden_layers": 1, "head_dim": 512, "qk_rope_head_dim": 64,
           "index_head_dim": 128, "sliding_window": 128, "compress_ratios": [0],
           "kv_source_layer_ids": []}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    argv = ["--out", str(tmp_path / "o.jsonl"), "--model", str(tmp_path),
            "--memory-budget-total-gb", "100", "--max-kv", "1000"]
    args = mod.build_parser().parse_args(argv)
    args._dsv41_system_used_at_start_bytes = int(20 * GIB)
    mod._resolve_derivation(args, bench=_FakeBench(int(20 * GIB)), max_kv=1000)
    # the -gb 100 is decimal GB -> ~93.13 GiB (NOT 100 GiB).
    assert args._dsv41_budget_total.budget_total_gb == pytest.approx(93.1322574, abs=1e-3)


def test_gib_and_gb_both_set_refused(tmp_path):
    mod = _mod()
    (tmp_path / "config.json").write_text(json.dumps(
        {"num_hidden_layers": 1, "kv_source_layer_ids": []}))
    argv = ["--out", str(tmp_path / "o.jsonl"), "--model", str(tmp_path),
            "--memory-budget-total-gib", "93", "--memory-budget-total-gb", "100",
            "--max-kv", "1000"]
    args = mod.build_parser().parse_args(argv)
    args._dsv41_system_used_at_start_bytes = int(20 * GIB)
    with pytest.raises(ValueError, match="only one of"):
        mod._resolve_derivation(args, bench=_FakeBench(int(20 * GIB)), max_kv=1000)


def test_safety_and_overhead_gib_flags(tmp_path):
    mod = _mod()
    cfg = {"num_hidden_layers": 1, "kv_source_layer_ids": [], "compress_ratios": [0],
           "head_dim": 512, "qk_rope_head_dim": 64, "index_head_dim": 128,
           "sliding_window": 128}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    argv = ["--out", str(tmp_path / "o.jsonl"), "--model", str(tmp_path),
            "--memory-budget-total-gib", "100", "--max-kv", "1000",
            "--memory-safety-gib", "5", "--non-metal-overhead-gib", "8"]
    args = mod.build_parser().parse_args(argv)
    args._dsv41_system_used_at_start_bytes = int(20 * GIB)
    mod._resolve_derivation(args, bench=_FakeBench(int(20 * GIB)), max_kv=1000)
    bt = args._dsv41_budget_total
    assert bt.safety_gb == pytest.approx(5.0)
    assert bt.non_metal_overhead_gb == pytest.approx(8.0)


# --------------------------------------------------------------------------
# LOW-4: --memory-plan-preflight derives from a dry snapshot and exits 0/3
# BEFORE any model load / GPU window.
# --------------------------------------------------------------------------


class _FakeBenchPF:
    def __init__(self, system_used_bytes):
        self._sys = int(system_used_bytes)

    def _system_used_bytes(self):
        return self._sys

    def resolve_max_kv(self, cells, steps, max_kv):
        return int(max_kv)


def _pf_args(mod, tmp_path, **over):
    cfg = {"num_hidden_layers": 40, "head_dim": 512, "qk_rope_head_dim": 64,
           "index_head_dim": 128, "sliding_window": 128,
           "kv_source_layer_ids": [2, 8, 14, 20], "compress_ratios": []}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    argv = ["--out", str(tmp_path / "o.jsonl"), "--model", str(tmp_path),
            "--context-tokens", "16384", "--decode-tokens", "256",
            "--max-kv", "17408", "--memory-plan-preflight"]
    for k, v in over.items():
        if k == "argv_extra":
            argv += v
    args = mod.build_parser().parse_args(argv + over.get("argv_extra", []))
    args._dsv41_system_used_at_start_bytes = int(20 * GIB)
    return args


def test_preflight_ok_returns_0(tmp_path):
    mod = _mod()
    args = _pf_args(mod, tmp_path, argv_extra=["--memory-budget-total-gib", "93"])
    rc = mod._preflight_memory_plan(args, _FakeBenchPF(int(20 * GIB)))
    assert rc == 0


def test_preflight_below_floor_returns_3(tmp_path):
    mod = _mod()
    args = _pf_args(mod, tmp_path, argv_extra=["--memory-budget-total-gib", "40"])
    rc = mod._preflight_memory_plan(args, _FakeBenchPF(int(20 * GIB)))
    assert rc == 3  # 40 - 20 - 10 - kv - 3 < 20 floor


def test_preflight_no_budget_flag_returns_0(tmp_path):
    mod = _mod()
    args = _pf_args(mod, tmp_path)  # no --memory-budget-total-*
    rc = mod._preflight_memory_plan(args, _FakeBenchPF(int(20 * GIB)))
    assert rc == 0
