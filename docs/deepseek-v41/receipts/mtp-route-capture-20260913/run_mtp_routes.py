"""Bounded diagnostic capture of native MTP expert routes on the exact workload."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

signal.alarm(1200)
GIB = 1024 ** 3
RECORD = 18_800_640
FIXED_MTP = 31_931_661_128
PRIOR_ACTIVE_PEAK = 87_384_896_236
PRIOR_SLOTS = 72
GRAPH_MARGIN = 2 * GIB
PRIOR_BASELINE = 9_432_500_000
PREFIX = Path("/tmp/dsv41-110-preflight/mtp-routes-16k-1024")
ROOT = Path("/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4")
FIXTURE = Path("docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json")
PRIOR_RECEIPT = Path("docs/deepseek-v41/receipts/dspark-layer-prefill-20260913")
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
                        "python-16k-1024-dspark-layer-prefill.jsonl").read_text())
assert prior["source_commit"] == "1f3b9bca7ae5aab6a4bf30eb969b1f6dd0368711"
assert prior["slots_per_layer"] == PRIOR_SLOTS
assert round(prior["dspark"]["peak_gb"] * 1e9) == PRIOR_ACTIVE_PEAK
assert prior_raw["prompt_ids_sha256"] == "38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2"
assert not subprocess.call(
    ["git", "diff", "--quiet", prior["source_commit"], "HEAD", "--", "mtplx", "scripts"]
), "production source changed since the measured full run"

# Same M6/T5 geometry as the complete measured run. The 2 GiB margin covers
# graph variation and the retained, already-fenced int32 route arrays; the
# maximum raw route payload is 3 * 1023 * 5 * 3 * 4 = 184,140 bytes. No model
# residency or KV saving is credited. This is a diagnostic, not a TPS result.
engine = int(110e9 - base - 2 * GIB - 11 * GIB)
expected_slots = (engine - FIXED_MTP) // (40 * RECORD)
if not 1 <= expected_slots <= 75:
    raise RuntimeError("unadmitted full-workload slot geometry")
slot_delta = max(0, expected_slots - PRIOR_SLOTS) * 40 * RECORD
active_bound = PRIOR_ACTIVE_PEAK + slot_delta + GRAPH_MARGIN
physical_bound = max(
    base + active_bound + 2 * GIB + 2 * GIB,
    prior["physical_peak_bytes"] + base - PRIOR_BASELINE + slot_delta + GRAPH_MARGIN,
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
    "maximum_raw_route_payload_bytes": 3 * 1023 * 5 * 3 * 4,
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
        args.box_target_gb != 110 or args.transient_band_gib != 11 or
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
        # Diagnostic-only Gate wrapper. It retains the native lazy int32 arrays
        # without evaluating or converting them in any draft/target hot path.
        # The normal draft fence resolves each array before serialization below.
        from mtplx.models.deepseek_v41_moe import Gate
        stage_for_gate = {id(stage.mlp.gate): i for i, stage in enumerate(model.mtp.layers)}
        if len(stage_for_gate) != 3:
            raise RuntimeError("expected three distinct native MTP gates")
        captured = []
        original_gate = Gate.__call__

        def capture_gate(self, *a, **kw):
            output = original_gate(self, *a, **kw)
            stage = stage_for_gate.get(id(self))
            if stage is not None:
                captured.append((stage, output[1]))
            return output

        Gate.__call__ = capture_gate
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
            Gate.__call__ = original_gate
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
        if len(captured) != 3 * cycles or any(captured[i][0] != i % 3 for i in range(len(captured))):
            raise RuntimeError("MTP route capture did not cover all stages in order")
        import numpy as np
        ordered = []
        for stage, route in captured:
            values = np.asarray(route)
            if values.shape != (5, 3) or values.dtype.kind not in "iu":
                raise RuntimeError("unexpected native MTP route geometry")
            ordered.append(values.tolist())
        routes = [ordered[i:i + 3] for i in range(0, len(ordered), 3)]
        io_before = result["stream_after_prefill"]["io"]
        io_after = result["stream_end"]["io"]
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
            "draft_routes_by_cycle_stage_row": routes,
            "mlx_peak_bytes": result["memory"]["mlx_peak_bytes"],
            "decode_records_read": io_after["records_read"] - io_before["records_read"],
            "decode_bytes_read": io_after["read_bytes"] - io_before["read_bytes"],
        }
        PREFIX.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print("MTP_ROUTES_CAPTURED", json.dumps({
            "cycles": cycles,
            "route_arrays": len(captured),
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
