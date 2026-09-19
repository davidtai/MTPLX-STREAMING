"""Summarize the two completed component runs without importing MLX."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
SOURCE = "3dcc1605404c860342601479dad1d4a5bc10f9bf"
REPO = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


summary = {
    "source_commit": SOURCE,
    "scope": "Three real Q4 expert layers, saved routes, synthetic inputs and omitted attention. No full-model throughput claim.",
    "decision": "Neither schedule is promoted; no full candidate or optimization regression tests follow.",
    "box_budget_bytes": 110_000_000_000,
    "canonical_component_accounting": {
        "static_incremental_bound_bytes": 14 * 1024**3,
        "metal_cache_compiler_bound_bytes": 10 * 1024**3,
        "host_reader_compiler_bound_bytes": 4 * 1024**3,
        "persistent_slots_per_layer": 105,
        "layers": [30, 31, 32],
        "shared_transient_slots": 48,
        "shared_prefetch_slots": 16,
        "total_slots": 379,
        "raw_bank_bytes": 379 * 18_800_640,
        "packed_bank_bytes": 379 * 17_694_720,
        "retained_output_bytes": 192 * 6 * 6 * 5120 * 2,
        "ridge_parameter_payload_bytes": 2 * (384 * 384 + 3 * 384) * 4,
        "accounting_note": "Bank, output and adapter payloads are inside the Metal envelope; do not add them again. CPU preparation finishes before GPU loading.",
    },
    "receipt_metadata_correction": {
        "raw_receipts_unchanged": True,
        "authoritative_admission_fields": ["static_incremental_bound_bytes", "budget_components", "plan"],
        "obsolete_inherited_fields": ["bound_components", "bound_scope", "native_slot_bytes", "packed_slot_bytes", "memory_components", "comparison", "layer_selection", "experimental_prefetch", "allocator_sources.rule"],
        "reason": "Copied descriptive fields still describe earlier 157/274-slot and row-permutation screens. Actual preflight, 379-slot geometry, 14 GiB admission, 10 GiB allocator and command cap agree; canonical values above supersede only those stale descriptions.",
    },
    "ownership_review": {
        "issue_targets": {"30": 31, "31": 32},
        "current_layer": "GlobalPrefetchRing.plan_prefetch protects every target-1 tenant, including the source layer whose resident GPU work has just been enqueued.",
        "previous_layer": "The probe evaluates each layer output before the next layer flushes deferred leases. The full lane requires a covering dependency evaluation; this screen does not establish full integration.",
        "physical_writer": "ExpertSlotPool._prepare_load waits for LOADING and pinned slots before ownership changes. load_speculative keeps its lifecycle claim through all component writers.",
        "publication": "Ticket-bound completion and native route transactions are unchanged; callback workers do not publish directly.",
        "failure": "The lane synchronizes submitted GPU work before abort/close. The supervisor reaps its child before file-cache reclamation.",
        "source_files": {name: sha(REPO / name) for name in (
            "mtplx/expert_streaming.py", "mtplx/expert_slots.py",
            "mtplx/expert_runtime.py", "mtplx/models/expert_mlx.py")},
    },
    "runs": [],
}

for variant, point in (("v1", "first missing-part GU witness"),
                       ("v2", "after demand and resident/shared GPU submission")):
    root = ROOT / variant
    result = json.loads((root / "probe.json").read_text())
    proof = json.loads((root / "installation.json").read_text())
    child = json.loads((root / "child.json").read_text())
    reclaim = json.loads((root / "reclamation.json").read_text())
    assert result["complete"] and result["source_commit"] == SOURCE
    assert child["returncode"] == 0 and reclaim["child_terminal"]
    assert reclaim["cached_page_bytes_after"] == 0
    assert result["construction"] == proof
    assert proof["static_incremental_bound_bytes"] == 14 * 1024**3
    assert proof["bank_capacity"] == 379
    assert proof["plan"]["persistent_slots"] == 315
    for key in ("raw_banks_bytes", "packed_banks_bytes"):
        canonical = key.replace("banks", "bank")
        assert proof["budget_components"][key] == summary["canonical_component_accounting"][canonical]
    assert proof["budget_components"]["retained_output_bytes"] == summary["canonical_component_accounting"]["retained_output_bytes"]
    for name, expected in proof["helper_sha256"].items():
        assert sha(root / name) == expected, (variant, name)
    for name, expected in proof["runtime_source_sha256"].items():
        assert sha(REPO / name) == expected, name
    prep = json.loads((root / "preparation.json").read_text())
    assert prep["complete"] and not prep["mlx_imported"]
    assert sha(root / "ridge-parameters.npz") == prep["parameters_sha256"]
    assert all(a["all_outputs_exact"] and len(a["cases"]) == 192 for a in result["arms"])
    refs = result["arms"][0]["cases"]
    assert all(a["cases"] == refs for a in result["arms"])
    native = [a for a in result["arms"] if a["mode"] == "native"]
    candidate = next(a for a in result["arms"] if a["mode"] == "prefetch")
    median = statistics.median(a["heldout_total_ns"] for a in native)
    ratio = candidate["heldout_total_ns"] / median
    assert ratio == result["latency_ratio"]
    log = (root / "guard.log").read_text()
    assert "GPU step exited with code 0;" in log
    memory = re.search(r"sampled peak step footprint [^(]+\((\d+) bytes\), sampled peak guard accounted [^(]+\((\d+) bytes\), sampled peak physical used [^(]+\((\d+) bytes\), sampled peak compressor delta [^(]+\((\d+) bytes\); complete step memory samples (\d+)", log)
    assert memory is not None
    restored = next(line for line in log.splitlines() if "healthy, model identity" in line and "warmup ready" in line)
    released = next(line for line in log.splitlines() if "released exclusive GPU lock" in line)
    row = {
        "variant": variant,
        "issue_point": point,
        "result_sha256": sha(root / "probe.json"),
        "heldout_candidate_over_native_median": ratio,
        "heldout_latency_change_percent": (ratio - 1) * 100,
        "control_spread_fraction": result["control_spread_fraction"],
        "all64_candidate_over_native_median": candidate["continuous_all64_ns"] / statistics.median(a["continuous_all64_ns"] for a in native),
        "all_192_layer_outputs_exact_per_arm": True,
        "final_active_bytes": result["active_after_close_bytes"],
        "memory": dict(zip(("child_tree_footprint_peak_bytes", "guard_accounted_peak_bytes", "whole_machine_physical_peak_bytes", "compressor_growth_peak_bytes", "guard_samples"), map(int, memory.groups()))),
        "arms": [{key: a[key] for key in ("mode", "sequence", "heldout_total_ns", "continuous_all64_ns", "mlx_peak_bytes", "active_after_close_bytes")} | {
            "reader": {key: a["reader_metrics"][key] for key in ("records_read", "read_bytes", "read_wall_ns")},
            "prefetch": {key: value for key, value in a["runtime_counters"].items() if "prefetch" in key},
        } for a in result["arms"]],
        "counter_scope": "Whole 64-cycle run, including initial bank seeding for reader counters; not heldout-only. Assignment hits and physical records are different units.",
        "guard_restore_evidence": restored,
        "guard_release_evidence": released,
        "guard_exit_code": 0,
        "child": child,
        "source_cache_after_bytes": reclaim["cached_page_bytes_after"],
    }
    summary["runs"].append(row)

assert sha(ROOT / "v1/ridge-parameters.npz") == sha(ROOT / "v2/ridge-parameters.npz")
assert sha(ROOT / "v1/routes.json") == sha(ROOT / "v2/routes.json")
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"complete": True, "variants": [{"variant": r["variant"], "ratio": r["heldout_candidate_over_native_median"], "all64_ratio": r["all64_candidate_over_native_median"]} for r in summary["runs"]]}))
