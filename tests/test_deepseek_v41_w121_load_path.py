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
    # allocator limit = target - baseline - cache(6 GiB); engine = allocator - band(4.1)
    assert tp["allocator_limit_bytes"] == int(round((100 - 10.42) * GB)) - 6 * GIB
    assert tp["engine_budget_bytes"] == tp["allocator_limit_bytes"] - int(round(4.1 * GIB))
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
    sidecar = tmp_path / "pin.json"
    sidecar.write_text(json.dumps({
        "box_target_gb": 100.0,
        "box_baseline_gb": 10.42,
        "allocator_cache_limit_gib": 6.0,
        "transient_band_gib": 4.1,
    }))
    args, cap = _drive(
        ab, ["--memory-plan-from", str(sidecar)],
        "cell16k_ring_v2_attn", tmp_path, monkeypatch,
    )
    tp = args._dsv41_target_plan
    assert tp is not None
    assert tp["pinned_from"] == str(sidecar)
    assert tp["engine_budget_bytes"] == (
        int(round((100 - 10.42) * GB)) - 6 * GIB - int(round(4.1 * GIB))
    )
    assert abs(cap["memory_limit_bytes"] - tp["engine_budget_bytes"]) < 4096
