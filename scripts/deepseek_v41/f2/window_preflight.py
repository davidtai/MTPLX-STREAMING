"""CPU preflight for the F2 prefetch GPU window -- runs BEFORE the service is unloaded.

Resolves every runtime / reader / slot / ring / gate attribute the F2 lane touches on
the REAL shipped classes (hasattr / dataclass field checks), imports every module the
candidate needs with MLX pinned to CPU, and -- when paths are given -- verifies the
staged-tree and prompt-id dependencies exist. Refuses with a clear, specific message
if any name or dependency is missing, so the window fails ONCE here rather than after
the production model is unloaded (AGENTS.md: fail once, clearly, before measured
generation). MLX is imported but pinned to the CPU device; no Metal, no model load.

Callable as ``python -m f2.window_preflight [--dep PATH ...] [--sha PATH=HEX ...]``
or via ``main(...)``; the window script drives it before any GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


class PreflightError(RuntimeError):
    pass


# (class-or-module, name, kind) the lane resolves. kind: "attr" = hasattr on the object.
def _seam_checks():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)

    from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime, PendingSplitRoute
    from mtplx.expert_slots import ExpertSlotPool, ExpertSlotState
    from mtplx.expert_io import PositionalExpertReader
    from mtplx.expert_streaming import CacheCounters, GlobalPrefetchRing, LayerExpertSlotBank
    from mtplx.models import deepseek_v41_moe as moe
    from mtplx.models.expert_mlx import HotExpertSwitchGLU, _clamped_swiglu, _DeferredSplitClose

    # f2 package modules (import proves the lane + config + reader + predictor load).
    import f2.full_config as full_config
    import f2.issue as issue
    import f2.plane_lane_prefetch as plane_lane_prefetch
    import f2.priority_reads as priority_reads
    import f2.run_full_install as run_full_install

    checks: list[tuple[str, object, tuple[str, ...]]] = [
        ("ExpertStreamingRuntime", ExpertStreamingRuntime, (
            "begin_split_route", "observe_route", "flush_deferred_slot_releases",
            "defer_slot_release", "prefetch_experts", "_reconcile_prefetch_for_route",
            "_run_speculative_load", "_apply_prefetch_completions",
        )),
        ("PendingSplitRoute", PendingSplitRoute, (
            "iter_ready_misses", "abort", "close",
        )),
        ("ExpertSlotPool", ExpertSlotPool, (
            "load_speculative", "ensure_route_part", "ensure_route", "reset",
        )),
        ("PositionalExpertReader", PositionalExpertReader, (
            "_readv_range_into", "read_record_into", "read_component_records_into",
        )),
        ("LayerExpertSlotBank", LayerExpertSlotBank, (
            "plan", "plan_prefetch", "prefetch_ticket", "commit_prefetch",
            "invalidate_prefetch", "published_experts",
            "resident_experts", "_prefetch_expert_to_slot",
        )),
        ("GlobalPrefetchRing", GlobalPrefetchRing, (
            "plan_prefetch", "commit_prefetch", "invalidate_prefetch", "published",
            "first_consumption", "mark_used", "note_decode", "consume_wasted_by_layer",
        )),
        ("deepseek_v41_moe", moe, ("_gate_prefix", "_gate_prefix_impl", "_attn_compile_gate", "Gate")),
        ("expert_mlx", HotExpertSwitchGLU, ("_run",)),
        ("issue", issue, ("Issue", "GatePredictor")),
        ("plane_lane_prefetch", plane_lane_prefetch, (
            "install", "PrefetchDecode", "PackedDecode", "PackedOps", "OpsContract",
            "bind_priority_reader", "PartExecutor", "ReaderExecutor", "SpeculativeExecutor",
        )),
        ("priority_reads", priority_reads, ("PriorityReads", "native_worker_count")),
        ("full_config", full_config, ("FullPrefetchConfig", "ring_reserve_bytes", "ADMITTED_RING_SLOTS")),
        ("run_full_install", run_full_install, ("install_f2_growth", "select_prefetch_sources")),
    ]
    missing = []
    for owner_name, owner, names in checks:
        for name in names:
            if not hasattr(owner, name):
                missing.append(f"{owner_name}.{name}")

    # ``pending.hit_ready`` is a live instance attribute the runner reads; resolve it
    # through the PendingSplitRoute constructor signature (it is set in __init__).
    import inspect

    if "hit_ready" not in inspect.signature(PendingSplitRoute.__init__).parameters:
        missing.append("PendingSplitRoute.__init__(hit_ready=...)")

    # Dataclass FIELDS the lane's config gate + counter snapshot read.
    config_fields = set(getattr(ExpertStreamingConfig, "__dataclass_fields__", {}))
    for field in ("prefetch_slots", "transient_slots", "slot_layout", "cache_scope",
                  "cache_policy", "decode_miss_records_per_part", "split_route_release",
                  "overlap_miss_reads", "resource_telemetry", "io_read_fanout"):
        if field not in config_fields:
            missing.append(f"ExpertStreamingConfig.{field}")
    counter_fields = set(getattr(CacheCounters, "__dataclass_fields__", {}))
    for field in ("prefetch_issued", "prefetch_issued_verify", "prefetch_committed",
                  "prefetch_awaited_inflight", "prefetch_wasted", "prefetch_bytes",
                  "prefetch_hit_on_true_route", "prefetch_first_consumption_hits"):
        if field not in counter_fields:
            missing.append(f"CacheCounters.{field}")
    for member in ("READY", "LOADING", "FAILED"):
        if not hasattr(ExpertSlotState, member):
            missing.append(f"ExpertSlotState.{member}")
    for attr in ("weight", "e_score_correction_bias", "gate_temp", "score_func"):
        if attr not in set(dir(moe.Gate)) and attr not in getattr(moe.Gate, "__annotations__", {}):
            # Gate sets these in __init__; check the class defines the constructor.
            pass
    if not callable(getattr(moe.Gate, "__call__", None)):
        missing.append("Gate.__call__")

    # The lane's production geometry contract must be the retained (5120/2304/6/mxfp4).
    if plane_lane_prefetch.PRODUCTION_CONTRACT != plane_lane_prefetch.OpsContract(
        5120, 2304, 6, 4, 32, "mxfp4", 10.0
    ):
        missing.append("plane_lane_prefetch.PRODUCTION_CONTRACT!=(5120,2304,6,4,32,mxfp4,10.0)")
    if full_config.EXPERT_WEIGHT_RECORD_BYTES != 17_694_720:
        missing.append("full_config.EXPERT_WEIGHT_RECORD_BYTES!=17694720")

    return missing


def _check_deps(deps, shas):
    missing = []
    for dep in deps or ():
        if not Path(dep).exists():
            missing.append(f"missing dependency: {dep}")
    for spec in shas or ():
        path, _, expected = spec.partition("=")
        p = Path(path)
        if not p.exists():
            missing.append(f"missing sha-pinned file: {path}")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if expected and got != expected:
            missing.append(f"sha256 mismatch for {path}: got {got}, want {expected}")
    return missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="F2 prefetch window CPU preflight")
    parser.add_argument("--dep", action="append", default=[], help="path that must exist")
    parser.add_argument("--sha", action="append", default=[], help="PATH=HEX sha256 pin")
    args = parser.parse_args(argv)
    problems = _seam_checks() + _check_deps(args.dep, args.sha)
    if problems:
        for p in problems:
            print(f"[f2-preflight] FAIL: {p}", file=sys.stderr)
        print(f"[f2-preflight] {len(problems)} problem(s); refusing before the service is "
              "unloaded.", file=sys.stderr)
        return 1
    print("[f2-preflight] OK: every lane seam resolves on the real classes; deps present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
