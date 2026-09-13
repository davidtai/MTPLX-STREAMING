"""Offline global-cache relaxation using the same tested bypass oracle."""
import argparse
import json
from pathlib import Path

from scripts.deepseek_v41.analyze_route_cache import optimal_batch_misses

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trace", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
trace = json.loads(args.trace.read_text())
layers = sorted(trace["sequences"], key=int)
width = max(state["expert_count"] for state in trace["initial_banks"].values())
sequence = [
    [int(layer) * width + expert for expert in trace["sequences"][layer][step]]
    for step in range(trace["decode_steps"])
    for layer in layers
]
initial = [
    int(layer) * width + int(expert)
    for layer, state in trace["initial_banks"].items()
    for expert in state["_expert_to_slot"]
]
capacity = sum(state["persistent_slots"] for state in trace["initial_banks"].values())
reads = optimal_batch_misses(sequence, capacity, initial)
result = {
    "semantics": "clairvoyant global cache with bypass, equal-size records and complete sequential layer order; not implemented or a speed claim",
    "capacity": capacity,
    "reads": reads,
    "misses_per_token": reads / trace["decode_steps"],
    "bytes_per_token": reads * trace["record_bytes"] / trace["decode_steps"],
    "required_ssd_gb_s_at_20_tps": reads * trace["record_bytes"] / trace["decode_steps"] * 20 / 1e9,
}
args.out.write_text(json.dumps(result, indent=2) + "\n")
