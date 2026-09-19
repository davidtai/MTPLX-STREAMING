"""W121 regression: the REAL load path resolves arguments without a model.

Window 48 crashed at load with ``NameError: _PROFILE_PLAN_FIELDS`` --
``main() -> _run_arm -> _load_model -> _resolve_plan_overrides`` (the argument-resolution
path) references a module constant the W121 bench cleanup deleted, and no test covered it.
The DEFAULT ``--expert-profile deepseek-v41-mxfp4-75`` takes the ``for field in
_PROFILE_PLAN_FIELDS`` branch, so a plain launch tripped it.

These drive the whole chain with a FAKE bench + a patched
``load_deepseek_v41_streaming`` that raises a sentinel the instant argument resolution
finishes (before any model / MLX allocation), so ``_resolve_derivation`` +
``dspark_bench_loader_overrides`` + ``_resolve_plan_overrides`` are all exercised for:
  * ``--box-target-gb 100 --box-baseline-gb 10.42`` (AR),
  * the same under ``--decode-mode dspark`` (+ ``--dspark-depth 5``),
  * ``--memory-limit-gib`` override, and
  * ``--memory-plan-from`` (pinned target components).
Reaching the sentinel (not a NameError) is the pass; before the fix these raise NameError.

CPU-only: MLX pinned to CPU (memory/worker-tests-must-pin-mlx-cpu.md); no model, no
tokenizer load (``_tokenizer`` patched + ``--eos-id`` given), no server, no network.  Run
under ``nice -n 19`` and WITHOUT ``pytest -n auto``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import types
from pathlib import Path

import mlx.core as mx

# HARD rule: pin MLX to CPU before anything can touch Metal.
mx.set_default_device(mx.cpu)

import pytest

_WT = Path(__file__).resolve().parents[1]
GIB = 1024**3
GB = 1_000_000_000

_BOX_ENV = (
    "MTPLX_DSV41_BOX_TARGET_GB",
    "MTPLX_DSV41_BOX_BASELINE_GB",
    "MTPLX_DSV41_MLX_CACHE_LIMIT_GIB",
    "MTPLX_DSV41_TRANSIENT_BAND_GIB",
    "MTPLX_MEMORY_LIMIT_BYTES",
    "MTPLX_DSV41_KV_BOUNDED",
    "MTPLX_DSV41_KV_BOUNDED_MAXKV",
    "MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB",
)


def _ab():
    path = _WT / "scripts" / "deepseek_v41" / "ab_decode_env_levers.py"
    spec = importlib.util.spec_from_file_location("_dsv41_ab_loadpath", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LoadReached(Exception):
    """Raised by the patched loader once argument resolution has completed."""


@pytest.fixture(autouse=True)
def _clean_box_env(monkeypatch):
    # A fresh env each test; _resolve_target_plan writes raw os.environ (not via
    # monkeypatch), so snapshot the whole environ and restore it afterwards.
    snapshot = dict(os.environ)
    for k in _BOX_ENV:
        monkeypatch.delenv(k, raising=False)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def _drive(ab, argv, arm, tmp_path, monkeypatch, *, max_kv=2048):
    """Run _run_arm through _load_model up to the (patched) loader; return
    (args, captured-loader-kwargs)."""
    import mtplx.models.deepseek_v41_loader as loader

    captured: dict = {}

    def _fake_load(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        raise _LoadReached()

    monkeypatch.setattr(loader, "load_deepseek_v41_streaming", _fake_load)

    def _no_tokenizer(_args, _bench):  # never touch the model dir / network
        raise RuntimeError("tokenizer disabled in this CPU test")

    monkeypatch.setattr(ab, "_tokenizer", _no_tokenizer)

    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([1, 2, 3, 4]))
    full = list(argv) + [
        "--eos-id", "0",
        "--prompt-ids-file", str(ids),
        "--model", str(tmp_path / "no-such-model"),
        "--out", str(tmp_path / "r.json"),
    ]
    args = ab.build_parser().parse_args(full)
    bench = types.SimpleNamespace(
        resolve_max_kv=lambda ctx, dec, mk: int(mk or max_kv),
        _load_build_prompt=lambda: (lambda *a, **k: [1, 2, 3, 4]),
        _resolve_prompt=lambda a, tok, bp, ctx: ([1, 2, 3, 4], {"prompt_tokens": 4}),
    )
    with pytest.raises(_LoadReached):
        ab._run_arm(args, arm, bench, mx=object())
    return args, captured


def test_box_target_ar_end_to_end(tmp_path, monkeypatch):
    ab = _ab()
    args, cap = _drive(
        ab, ["--box-target-gb", "100", "--box-baseline-gb", "10.42", "--max-kv", "2048"],
        "cell16k_ring_v2_attn", tmp_path, monkeypatch,
    )
    tp = args._dsv41_target_plan
    assert tp is not None
    assert tp["box_target_gb"] == 100.0
    assert abs(tp["box_baseline_gb"] - 10.42) < 1e-9
    # Current defaults: allocator = target - baseline - host(2 GiB);
    # engine leaves the 10 GiB measured prefill reserve, including the 2 GiB cache.
    assert tp["allocator_limit_bytes"] == int(round((100 - 10.42) * GB)) - int(round(2.0 * GIB))
    assert tp["engine_budget_bytes"] == (
        tp["allocator_limit_bytes"] - 10 * GIB
    )
    assert tp["allocator_cache_limit_gib"] == 2.0
    # the loader is handed the engine budget (grows the persistent slots to the target)
    assert abs(cap["memory_limit_bytes"] - tp["engine_budget_bytes"]) < 4096
    assert cap["max_live_kv_tokens"] == 2048
    # the DEFAULT profile seeded the plan fields -- the crash-site loop ran clean.
    assert "transient_slots" in args._dsv41_plan_overrides
    assert "transient_slots" in cap  # forwarded as **plan_overrides


def test_box_target_dspark_end_to_end(tmp_path, monkeypatch):
    ab = _ab()
    args, cap = _drive(
        ab,
        ["--box-target-gb", "100", "--box-baseline-gb", "10.42",
         "--decode-mode", "dspark", "--dspark-depth", "5"],
        "cell16k_ring_v2_draft_attn", tmp_path, monkeypatch,
    )
    assert args._dsv41_target_plan is not None
    assert cap["with_mtp"] is True  # the DSpark head is loaded
    assert cap["memory_limit_bytes"] > 0  # repriced for the MTP residents, still positive


def test_memory_limit_gib_override_end_to_end(tmp_path, monkeypatch):
    ab = _ab()
    args, cap = _drive(
        ab, ["--memory-limit-gib", "69.18"], "cell16k_ring_v2_attn", tmp_path, monkeypatch,
    )
    assert args._dsv41_target_plan is None  # target not armed -> explicit override
    assert cap["memory_limit_bytes"] == int(round(69.18 * GIB))


def test_memory_plan_from_pins_components_end_to_end(tmp_path, monkeypatch):
    ab = _ab()
    monkeypatch.setenv("MTPLX_DSV41_BOX_BASELINE_GB", "10")
    sidecar = tmp_path / "pin.json"
    sidecar.write_text(json.dumps({
        "box_target_gb": 100.0,
        "box_baseline_gb": 10.42,
        "host_overhead_gib": 2.0,
        "allocator_cache_limit_gib": 6.0,
        "transient_band_gib": 5.54,
        "active_overshoot_gib": 1.45,
    }))
    args, cap = _drive(
        ab, ["--memory-plan-from", str(sidecar)],
        "cell16k_ring_v2_attn", tmp_path, monkeypatch,
    )
    tp = args._dsv41_target_plan
    assert tp is not None
    assert tp["pinned_from"] == str(sidecar)
    # Historical pinned components override defaults: host(2), max(5.54, 1.45+6).
    assert tp["engine_budget_bytes"] == (
        int(round((100 - 10.42) * GB)) - int(round(2.0 * GIB))
        - max(int(round(5.54 * GIB)), int(round(1.45 * GIB)) + 6 * GIB)
    )
    assert abs(cap["memory_limit_bytes"] - tp["engine_budget_bytes"]) < 4096


def test_dspark_depth_over_latent_slack_raises(tmp_path, monkeypatch):
    ab = _ab()
    # MEDIUM-7: bounded-KV DSpark with depth + 1 > the bounded latent verify slack (8)
    # must refuse with a clean error (the *_bounded arm arms KV_BOUNDED).
    import os
    snapshot = dict(os.environ)
    monkeypatch.setattr(ab, "_tokenizer",
                        lambda a, b: (_ for _ in ()).throw(RuntimeError("no tok")))
    (tmp_path / "ids.json").write_text(json.dumps([1, 2, 3]))
    args = ab.build_parser().parse_args(
        ["--decode-mode", "dspark", "--dspark-depth", "8", "--eos-id", "0",
         "--prompt-ids-file", str(tmp_path / "ids.json"),
         "--out", str(tmp_path / "r.json")]
    )
    bench = types.SimpleNamespace(
        resolve_max_kv=lambda c, d, m: int(m or 2048),
        _load_build_prompt=lambda: (lambda *a, **k: [1, 2, 3]),
        _resolve_prompt=lambda a, t, b, c: ([1, 2, 3], {}),
    )
    try:
        with pytest.raises(SystemExit, match="latent verify slack"):
            ab._run_arm(args, "cell16k_ring_v2_draft_attn_bounded", bench, mx=object())
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def test_memory_plan_from_missing_raises(tmp_path, monkeypatch):
    ab = _ab()
    # MEDIUM-5: a --memory-plan-from that does not exist must RAISE, not fall through.
    import mtplx.models.deepseek_v41_loader as loader
    monkeypatch.setattr(loader, "load_deepseek_v41_streaming",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not load")))
    args = ab.build_parser().parse_args(
        ["--memory-plan-from", str(tmp_path / "nope.json"),
         "--eos-id", "0", "--out", str(tmp_path / "r.json")]
    )
    import types as _t
    bench = _t.SimpleNamespace(resolve_max_kv=lambda c, d, m: int(m or 2048))
    with pytest.raises(FileNotFoundError):
        ab._resolve_derivation(args, bench=bench, max_kv=2048)


# --------------------------------------------------------------------------
# Window-49 fixes: receipt box_used / mlx_limit_gib_effective / memory_cap on the
# EXPLICIT --memory-limit-gib path (no box target armed).
# --------------------------------------------------------------------------


def test_inject_box_used_reads_exported_baseline_on_explicit_path(monkeypatch):
    ab = _ab()
    # explicit path: no target plan on args, but gpu_window exported the baseline env.
    monkeypatch.setenv("MTPLX_DSV41_BOX_BASELINE_GB", "10.81")
    args = types.SimpleNamespace(_dsv41_target_plan=None, box_baseline_gb=None)
    mem = {"process_footprint_peak_gb": 84.07}
    ab._inject_box_used(mem, args)
    assert mem["box_baseline_gb"] == 10.81
    assert mem["baseline_plus_process_peak_estimate_gb"] == round(10.81 + 84.07, 4)  # was 0.00 before the fix


def test_inject_box_used_target_plan_wins_over_env(monkeypatch):
    ab = _ab()
    monkeypatch.setenv("MTPLX_DSV41_BOX_BASELINE_GB", "10.81")
    args = types.SimpleNamespace(_dsv41_target_plan={"box_baseline_gb": 11.0},
                                 box_baseline_gb=None)
    mem = {"process_footprint_peak_gb": 80.0}
    ab._inject_box_used(mem, args)
    assert mem["box_baseline_gb"] == 11.0  # target plan preferred
    assert mem["baseline_plus_process_peak_estimate_gb"] == 91.0


def test_inject_box_used_none_when_no_baseline_anywhere(monkeypatch):
    ab = _ab()
    monkeypatch.delenv("MTPLX_DSV41_BOX_BASELINE_GB", raising=False)
    args = types.SimpleNamespace(_dsv41_target_plan=None, box_baseline_gb=None)
    mem = {"process_footprint_peak_gb": 80.0}
    ab._inject_box_used(mem, args)
    assert mem["box_baseline_gb"] is None
    assert mem["box_used_gb"] is None


def test_effective_limit_is_the_passed_value_not_the_clamped_readback():
    ab = _ab()
    # window-49: set_memory_limit passed 77.18 GiB but get_memory_limit reads back the
    # OS-clamped 70.18; mlx_limit_gib_effective must be the PASSED value.
    mem = {"mlx_limit_gib_effective": None, "mlx_gc_limit_gib_readback": 70.18}
    cap = {"limit": int(round(77.18 * GIB))}
    ab._apply_effective_limit(mem, cap)
    assert abs(mem["mlx_limit_gib_effective"] - 77.18) < 1e-6


def test_effective_limit_falls_back_to_readback_without_cap():
    ab = _ab()
    mem = {"mlx_limit_gib_effective": None, "mlx_gc_limit_gib_readback": 70.18}
    ab._apply_effective_limit(mem, None)
    assert mem["mlx_limit_gib_effective"] == 70.18


def test_memory_cap_block_reads_report_and_snapshot_fallback():
    ab = _ab()
    rep = {"applied": True, "limit": 123, "wired_limit_applied": True,
           "cache_limit_applied": True}
    # primary: runtime.memory_cap_report
    rt = types.SimpleNamespace(memory_cap_report=rep)
    assert ab._memory_cap_block(rt) is rep
    # fallback: resource_telemetry_snapshot()['memory_cap'] when the attr is missing
    rt2 = types.SimpleNamespace(
        memory_cap_report=None,
        resource_telemetry_snapshot=lambda: {"memory_cap": rep},
    )
    assert ab._memory_cap_block(rt2) is rep
    # neither -> None
    assert ab._memory_cap_block(types.SimpleNamespace()) is None
