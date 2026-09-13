"""Diagnostic-only 16K Python / 64-step native block-6 MTP acceptance screen."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

signal.alarm(900)
GIB = 1024 ** 3
RECORD = 18_800_640
FIXED_MTP = 31_931_661_128
PRIOR_ACTIVE_PEAK = 87_384_896_236
PRIOR_SLOTS = 72
GRAPH_MARGIN = 3 * GIB
PRIOR_BASELINE = 9_432_500_000
PREFIX = Path("/tmp/dsv41-110-preflight/dspark-block6-short")
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

# Target M7 remains inside the 48-record shared pool (7*6=42 assignments).
# It allocates no extra persistent expert slots or KV backing. The previous
# full-workflow peak, plus the exact slot delta and 3 GiB, covers a complete
# additional real-shape attention-layer active peak (measured 1.133 GB), native
# T6 draft compile increment (21 MB), new row outputs, and graph variation.
# The bound credits no saving for a shorter decode or for Qwen cache release.
engine = int(110e9 - base - 2 * GIB - 11 * GIB)
expected_slots = (engine - FIXED_MTP) // (40 * RECORD)
if not 1 <= expected_slots <= 75:
    raise RuntimeError("unadmitted short-screen slot geometry")
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
    "graph_shape_margin_bytes": GRAPH_MARGIN,
    "mtp_head_only_T6_peak_bytes": 15_415_845_136,
    "target_one_layer_M7_peak_bytes": 1_132_996_826,
    "target_verify_rows": 7,
    "target_max_assignments": 42,
    "shared_transient_slots": 48,
    "requested_decode_steps": 64,
    "scope": "diagnostic only, native artifact and 16K pinned Python prompt; no AR pass or full-workload TPS claim",
}
PREFIX.with_suffix(".bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")
print("BLOCK6_BOUND", json.dumps(bounds), flush=True)

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
        args.context_tokens != 16384 or args.decode_tokens != 64 or
        args.decode_mode != "dspark" or args.dspark_depth != 6 or
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

    from mtplx.models import deepseek_v41_dspark as dsp
    from mtplx.models import deepseek_v41_loader as loader
    from mtplx.expert_runtime import text_only_resident_discount

    original_head_init = dsp.DSparkHead.__init__

    def block6_head(self, config):
        if int(config.dspark_block_size) != 5:
            raise RuntimeError("native head geometry changed")
        altered = copy.copy(config)
        altered.dspark_block_size = 6
        original_head_init(self, altered)
        if self.block_size != 6 or any(layer.block_size != 6 for layer in self.layers):
            raise RuntimeError("block length did not reach all MTP stages")

    dsp.DSparkHead.__init__ = block6_head
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
    prior_ar_ids = prior_raw["token_ids"][:65]
    assert len(prior_ar_ids) == 65

    def diagnostic_mtp(*, model, ops, mem_probe, prompt_ids, steps, **kwargs):
        global phase
        if len(prompt_ids) != 16384 or steps != 64 or model.mtp.block_size != 6:
            raise RuntimeError("loaded diagnostic model has wrong dimensions")
        digest = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
        if digest != prior["prompt_ids_sha256"]:
            raise RuntimeError("pinned prompt changed")
        runtime = model._mtplx_expert_runtime
        if runtime.plan.slots_per_layer != expected_slots or not runtime.spec.mtp_included:
            raise RuntimeError("loaded pool plan differs")
        phase = "dspark"
        sample()
        import mlx.core as mx
        with runtime.admit_kv_tokens(len(prompt_ids) + steps + 6 + 1):
            result = ab._generate_dspark(
                model=model, mx=mx, mem_probe=mem_probe, prompt_ids=prompt_ids,
                steps=steps, depth=6, stage_timing=False,
                ar_reference=prior_ar_ids,
                stop_ids=({int(kwargs["eos_id"])} if kwargs["stop_on_eos"] else None),
            )
        if runtime._live_kv_tokens != 0:
            raise RuntimeError("KV admission did not release")
        phase = "complete"
        sample()
        ids = result["generated"]
        first_diff = next((i for i, (a, b) in enumerate(zip(prior_ar_ids, ids))
                           if a != b), None)
        if len(ids) != 65 and first_diff is None:
            first_diff = min(len(ids), 65)
        result.pop("divergence", None)  # full logit row belongs only in memory
        report = {
            "source_commit": source,
            "diagnostic_only": True,
            "input_tokens": len(prompt_ids),
            "decode_steps": steps,
            "output_tokens": len(ids),
            "block_size": model.mtp.block_size,
            "slots_per_layer": runtime.plan.slots_per_layer,
            "generated_ids": ids,
            "ar_reference_first65_ids_sha256": hashlib.sha256(json.dumps(prior_ar_ids).encode()).hexdigest(),
            "matches_prior_ar_prefix": first_diff is None,
            "first_difference_index": first_diff,
            "result": result,
        }
        PREFIX.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print("BLOCK6_DIAGNOSTIC", json.dumps({
            "decode_tok_s": result["decode_tok_s"],
            "cycles": result["stats"]["cycles"],
            "matches_prior_ar_prefix": first_diff is None,
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
