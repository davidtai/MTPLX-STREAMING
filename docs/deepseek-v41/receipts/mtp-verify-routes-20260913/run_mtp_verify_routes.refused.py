"""Bounded capture of actual MTP target-verify routes and warm expert banks."""

import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
from collections import defaultdict

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

signal.alarm(900)
GIB = 1024 ** 3
RECORD = 18_800_640
FIXED_MTP = 31_931_661_128
PRIOR_ACTIVE_PEAK = 91_897_033_560
PRIOR_SLOTS = 78
GRAPH_MARGIN = 2 * GIB
PRIOR_BASELINE = 10_292_700_000
PREFIX = Path("/tmp/dsv41-110-preflight/mtp-verify-routes-16k-1024")
ROOT = Path("/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4")
FIXTURE = Path("docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json")
PRIOR_RECEIPT = Path("docs/deepseek-v41/receipts/target-slot-band6-20260913")
source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
    raise RuntimeError("tracked source must be clean for the diagnostic")
base = float(os.environ["MTPLX_DSV41_BOX_BASELINE_GB"]) * 1e9
if not 0 <= base <= 20e9:
    raise RuntimeError("invalid measured baseline")
if os.environ.get("MTPLX_DSV41_IO_READ_FANOUT") != "4":
    raise RuntimeError("the controlled fanout must be 4")

prior = json.loads((PRIOR_RECEIPT / "summary.json").read_text())
prior_raw = json.loads((PRIOR_RECEIPT /
                        "python-16k-1024-band6.jsonl").read_text())
assert prior["source_commit"] == "5c7661db48a1bed22cef753e336f9aba911add24"
assert prior["slots_per_layer"] == PRIOR_SLOTS
assert prior["budget"]["mlx_peak_bytes"] == PRIOR_ACTIVE_PEAK
assert prior_raw["prompt_ids_sha256"] == "38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2"
assert not subprocess.call(
    ["git", "diff", "--quiet", prior["source_commit"], "HEAD", "--", "mtplx", "scripts"]
), "production source changed since the measured full run"

# Same M6/T5 geometry as the complete 78-slot measured run. Capture only the
# expert IDs the target switch has already moved to Python after its existing
# routing barrier. The maximum raw route payload is 1023 * 40 * 36 * 4 bytes;
# 2 GiB separately covers graph variation, Python objects and warm-bank state.
# No residency, KV or prefill saving is credited. Diagnostic timing is void.
engine = int(110e9 - base - 2 * GIB - 6 * GIB)
expected_slots = (engine - FIXED_MTP) // (40 * RECORD)
if not 72 <= expected_slots <= 80:
    raise RuntimeError("unadmitted full-workload slot geometry")
slot_delta = max(0, expected_slots - PRIOR_SLOTS) * 40 * RECORD
active_bound = PRIOR_ACTIVE_PEAK + slot_delta + GRAPH_MARGIN
physical_bound = max(
    base + active_bound + 2 * GIB + 2 * GIB,
    prior["budget"]["sampled_physical_peak_bytes_250ms"]
    + base - PRIOR_BASELINE + slot_delta + GRAPH_MARGIN,
)
wired = host_memory_snapshot()["box"]["wired_bytes"]
if physical_bound > 109e9 or wired + active_bound + 2 * GIB > 100 * GIB:
    raise RuntimeError("candidate lacks bounded physical/wired headroom")

bounds = {
    "source_commit": source,
    "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "baseline_bytes": base,
    "engine_budget_bytes": engine,
    "expected_slots_per_layer": expected_slots,
    "active_bound_bytes": active_bound,
    "physical_bound_bytes": physical_bound,
    "wired_before_bytes": wired,
    "graph_and_capture_margin_bytes": GRAPH_MARGIN,
    "maximum_raw_route_payload_bytes": 1023 * 40 * 36 * 4,
    "target_verify_rows": 6,
    "target_max_assignments": 36,
    "shared_transient_slots": 48,
    "requested_decode_steps": 1023,
    "scope": "diagnostic only, native artifact and exact 16K/1024 Python workload; no AR pass or comparable TPS claim",
}
PREFIX.with_suffix(".bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")
print("MTP_ROUTE_BOUND", json.dumps(bounds), flush=True)

phase = "startup"
stop = threading.Event()

def sample():
    with PREFIX.with_suffix(".os.jsonl").open("a") as out:
        out.write(json.dumps({"phase": phase, "snapshot": host_memory_snapshot()}) + "\n")

def monitor():
    while not stop.wait(0.25):
        sample()

sample()
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
try:
    spec = importlib.util.spec_from_file_location(
        "ab", "scripts/deepseek_v41/ab_decode_env_levers.py"
    )
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    args = ab.build_parser().parse_args()
    if (Path(args.model).resolve() != ROOT or
        Path(args.prompt_ids_file).resolve() != FIXTURE.resolve() or
        args.arms != ["cell16k_ring_v2_draft_attn_pf0"] or
        args.context_tokens != 16384 or args.decode_tokens != 1023 or
        args.decode_mode != "dspark" or args.dspark_depth != 5 or
        args.max_kv != 17664 or args.prompt_seed != 20260829 or
        args.box_target_gb != 110 or args.transient_band_gib != 6 or
        args.host_overhead_gib != 2 or args.allocator_cache_gib != 2 or
        args.slot_layout != "component-banks" or
        args.expert_profile != "deepseek-v41-mxfp4-75" or
        args.memory_limit_gib is not None or args.expert_cache_limit_gib is not None or
        args.memory_plan_from or args.dry_run or not args.stop_on_eos or
        args.stage_timing or args.prefill_stage_timing or args.syncs or
        args.warm_repeat or not args.apply_memory_cap or
        ab._device_sample_resolved(args)):
        raise RuntimeError("arguments differ from admitted diagnostic")

    from mtplx.models import deepseek_v41_loader as loader
    from mtplx.expert_runtime import text_only_resident_discount

    original_allocator = loader._component_bank_allocator_for

    def checked_allocator(config, spec, root, manifest_path, manifest=None, *,
                          additional_resident_bytes=None):
        if (config.slot_layout != "component-banks" or config.transient_slots != 48 or
            config.prefetch_slots != 0 or config.max_live_kv_tokens != 17664 or
            config.memory_limit_bytes != engine or
            config.runtime_reserve_bytes != 7 * GIB or not spec.mtp_included or
            spec.key != "deepseek-v41-flash-expert-mxfp4"):
            raise RuntimeError("resolved geometry differs before allocation")
        if manifest is None:
            manifest = loader.load_expert_manifest(manifest_path)
        plan = config.memory_plan(
            spec, additional_resident_bytes=additional_resident_bytes,
            resident_discount_bytes=text_only_resident_discount(manifest, spec),
        )
        if plan.fixed_bytes != FIXED_MTP or plan.slots_per_layer != expected_slots:
            raise RuntimeError("resolved allocation plan differs before allocation")
        return original_allocator(config, spec, root, manifest_path, manifest,
                                  additional_resident_bytes=additional_resident_bytes)

    loader._component_bank_allocator_for = checked_allocator
    prior_mtp_ids = prior_raw["dspark"]["token_ids"]
    assert len(prior_mtp_ids) == 1024

    def diagnostic_mtp(*, model, ops, mem_probe, prompt_ids, steps, **kwargs):
        global phase
        if len(prompt_ids) != 16384 or steps != 1023 or model.mtp.block_size != 5:
            raise RuntimeError("loaded diagnostic model has wrong dimensions")
        digest = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
        if digest != prior["prompt_ids_sha256"]:
            raise RuntimeError("pinned prompt changed")
        runtime = model._mtplx_expert_runtime
        if runtime.plan.slots_per_layer != expected_slots or not runtime.spec.mtp_included:
            raise RuntimeError("loaded pool plan differs")
        # Observe only the target expert IDs already converted by the streamed
        # switch after its existing mx.eval(indices) routing barrier. The wrapper
        # forwards the original route untouched. This is diagnostic-only and its
        # execution time is excluded from performance comparisons.
        from mtplx.expert_streaming import RoutingPhase
        target_routes = defaultdict(list)
        original_observe = runtime.observe_route

        def capture_route(layer, routing_phase, expert_ids, *, token_count):
            original_observe(layer, routing_phase, expert_ids, token_count=token_count)
            if RoutingPhase(routing_phase) is RoutingPhase.DECODE:
                target_routes[int(layer)].append(tuple(int(x) for x in expert_ids))

        runtime.observe_route = capture_route

        # _generate_dspark calls this once at the exact prefill->decode boundary.
        # Snapshot only policy state and physical slot mappings; no extra GPU op.
        initial_banks = {}
        original_snapshot = ab._stream_counters_snapshot

        def capture_warm_banks(snapshot_model):
            if not initial_banks:
                for layer, bank in runtime._banks.items():
                    if bank._prefetch_ring is not None:
                        raise RuntimeError("offline replay requires pf0")
                    initial_banks[str(layer)] = {
                        "expert_count": bank.expert_count,
                        "persistent_slots": bank.persistent_slots,
                        "transient_slots": bank.transient_slots,
                        "single_pool": bank.single_pool,
                        "cache_policy": bank.cache_policy,
                        "frequency_decay": bank.frequency_decay,
                        "_expert_to_slot": dict(bank._expert_to_slot),
                        "_slot_to_expert": list(bank._slot_to_expert),
                        "_pool_recency": dict(bank._pool_recency),
                        "_pool_clock": bank._pool_clock,
                        "_protected": sorted(bank._protected),
                        "_decode_epoch": bank._decode_epoch,
                        "_history": [dataclasses.asdict(h) for h in bank._history],
                        "_prefill_seed_candidates": sorted(bank._prefill_seed_candidates),
                        "_prefill_route_freq": dict(bank._prefill_route_freq),
                        "_saw_decode_since_prefill": bank._saw_decode_since_prefill,
                    }
            return original_snapshot(snapshot_model)

        ab._stream_counters_snapshot = capture_warm_banks
        phase = "dspark"
        sample()
        import mlx.core as mx
        try:
            with runtime.admit_kv_tokens(len(prompt_ids) + steps + 5 + 1):
                result = ab._generate_dspark(
                    model=model, mx=mx, mem_probe=mem_probe, prompt_ids=prompt_ids,
                    steps=steps, depth=5, stage_timing=False,
                    ar_reference=None,
                    stop_ids=({int(kwargs["eos_id"])} if kwargs["stop_on_eos"] else None),
                )
        finally:
            runtime.observe_route = original_observe
            ab._stream_counters_snapshot = original_snapshot
        if runtime._live_kv_tokens != 0:
            raise RuntimeError("KV admission did not release")
        phase = "complete"
        sample()
        ids = result["generated"]
        first_diff = next((i for i, (a, b) in enumerate(zip(prior_mtp_ids, ids))
                           if a != b), None)
        if len(ids) != 1024 and first_diff is None:
            first_diff = min(len(ids), 1024)
        cycles = result["stats"]["cycles"]
        complete = (
            set(target_routes) == set(runtime.spec.routed_layer_indices)
            and set(initial_banks) == {str(layer) for layer in runtime.spec.routed_layer_indices}
            and all(len(seq) == cycles for seq in target_routes.values())
            and all(0 < len(step) <= 36 and len(step) % 6 == 0
                    for seq in target_routes.values() for step in seq)
        )
        io_before = result["stream_after_prefill"]["io"]
        io_after = result["stream_end"]["io"]
        cache_before = result["stream_after_prefill"]["expert_cache"]
        cache_after = result["stream_end"]["expert_cache"]
        report = {
            "source_commit": source,
            "wrapper_sha256": bounds["wrapper_sha256"],
            "diagnostic_only": True,
            "input_tokens": len(prompt_ids),
            "decode_steps": steps,
            "output_tokens": len(ids),
            "block_size": model.mtp.block_size,
            "slots_per_layer": runtime.plan.slots_per_layer,
            "generated_ids": ids,
            "prior_mtp_ids_sha256": hashlib.sha256(json.dumps(prior_mtp_ids).encode()).hexdigest(),
            "matches_prior_mtp_ids": first_diff is None,
            "first_difference_index": first_diff,
            "cycles": cycles,
            "complete": complete,
            "record_bytes": RECORD,
            "target_routes_by_layer": {str(layer): seq for layer, seq in target_routes.items()},
            "initial_banks": initial_banks,
            "mlx_peak_bytes": result["memory"]["mlx_peak_bytes"],
            "decode_records_read": io_after["records_read"] - io_before["records_read"],
            "decode_bytes_read": io_after["read_bytes"] - io_before["read_bytes"],
            "decode_expert_misses": cache_after["expert_misses"] - cache_before["expert_misses"],
        }
        PREFIX.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        if not complete:
            raise RuntimeError("target verify route capture incomplete; raw trace preserved")
        print("MTP_VERIFY_ROUTES_CAPTURED", json.dumps({
            "cycles": cycles,
            "target_layers": len(target_routes),
            "matches_prior_mtp_ids": first_diff is None,
            "first_difference_index": first_diff,
            "mlx_peak_gb": result["peak_gb"],
        }), flush=True)
        raise SystemExit(0)

    ab._generate = diagnostic_mtp
    raise SystemExit(ab.main())
finally:
    stop.set()
    thread.join(2)
    sample()
