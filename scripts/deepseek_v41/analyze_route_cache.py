#!/usr/bin/env python3
"""Offline cache lower bounds for captured DeepSeek routes; never imports MLX.

Each batch can use temporary service slots, then retain at most ``capacity``
experts for the next batch. This permits bypass: a one-use expert need not evict
a useful resident. The runtime's older mandatory-admission oracle instead keeps
every expert from the current batch. Results here assume clairvoyant eviction,
fixed per-layer capacity, and sufficient temporary slots for a complete route.
They are diagnostic bounds, not a deployable policy or throughput measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def optimal_batch_misses(sequence, capacity, initial=()):
    """Minimum record reads with batch service and optional cache admission."""
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")
    initial = set(initial)
    if len(initial) > capacity:
        raise ValueError("initial cache exceeds capacity")
    batches = [set(step) for step in sequence]
    never = len(batches) + 1
    next_use = {}
    after = [{} for _ in batches]
    for index in range(len(batches) - 1, -1, -1):
        for expert in batches[index]:
            after[index][expert] = next_use.get(expert, never)
            next_use[expert] = index
    resident = {expert: next_use.get(expert, never) for expert in initial}
    misses = 0
    for index, needed in enumerate(batches):
        misses += len(needed - resident.keys())
        resident.update(after[index])
        if len(resident) > capacity:
            keep = sorted(resident, key=lambda expert: (resident[expert], expert))[:capacity]
            resident = {expert: resident[expert] for expert in keep}
    return misses


def analyze(payload):
    sequences = payload["sequences"]
    banks = payload["initial_banks"]
    steps = payload["decode_steps"]
    if steps <= 0 or set(sequences) != set(banks):
        raise ValueError("trace must contain matching layers and positive decode steps")
    if any(len(sequence) != steps for sequence in sequences.values()):
        raise ValueError("incomplete layer sequence")
    capacities = {bank["persistent_slots"] for bank in banks.values()}
    if len(capacities) != 1:
        raise ValueError("this diagnostic requires a uniform per-layer capacity")
    capacity = capacities.pop()
    record_bytes = payload["record_bytes"]
    for layer, sequence in sequences.items():
        bank = banks[layer]
        if any(len(set(step)) > bank["transient_slots"] for step in sequence):
            raise ValueError("temporary storage cannot serve a complete batch")
        if any(not 0 <= expert < bank["expert_count"] for step in sequence for expert in step):
            raise ValueError("route expert outside the captured bank")

    def metrics(reads):
        return {
            "record_reads": reads,
            "misses_per_decode_token": reads / steps,
            "bytes_per_decode_token": reads * record_bytes / steps,
            "required_ssd_gb_s_at_20_tps": reads * record_bytes / steps * 20 / 1e9,
        }

    warm = sum(
        optimal_batch_misses(sequence, capacity, map(int, banks[layer]["_expert_to_slot"]))
        for layer, sequence in sequences.items()
    )
    curve = {}
    for slots in sorted({0, 16, 27, capacity, 48, 64, 96}):
        reads = sum(optimal_batch_misses(sequence, slots) for sequence in sequences.values())
        curve[str(slots)] = {
            "persistent_bytes": slots * len(sequences) * record_bytes,
            **metrics(reads),
        }
    return {
        "semantics": "clairvoyant per-layer eviction with temporary service and bypass; no speed claim",
        "decode_steps": steps,
        "layers": len(sequences),
        "actual_slots_per_layer": capacity,
        "actual_initial_cache_lower_bound": metrics(warm),
        "cold_start_capacity_curve": curve,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(analyze(json.loads(args.trace.read_text())), indent=2) + "\n")


if __name__ == "__main__":
    main()
