"""Construction-only draft expert selection and router-nearest aliases.

This approximates draft proposals only. The native target must verify them.
The screen retains the original physical bank; savings are a projection.
"""
from collections import Counter

SIZES = {"one_band": (80, 40, 24), "two_bands": (64, 28, 12)}
TRAIN_CYCLES = 103
EXPERT_BYTES = 18_800_640
TARGET_BAND_BYTES = 707_788_800


def choose(routes, selected):
    if len(routes) != 206 or tuple(map(len, selected)) != (93, 58, 32):
        raise RuntimeError("draft training geometry differs")
    training = routes[:TRAIN_CYCLES]
    counts = [Counter(e for cycle in training for row in cycle[s] for e in row)
              for s in range(3)]
    result = {}
    for name, sizes in SIZES.items():
        keep = [sorted(sorted(c, key=lambda e: (-c[e], e))[:n])
                for c, n in zip(counts, sizes)]
        if any(len(ids) != n or not set(ids) <= set(old)
               for ids, n, old in zip(keep, sizes, selected)):
            raise RuntimeError("selected experts lack verified resident weights")
        savings = (sum(map(len, selected)) - sum(sizes)) * EXPERT_BYTES
        result[name] = {
            "selected_experts_by_stage": keep,
            "projected_retired_payload_bytes": savings,
            "projected_target_bands": savings // TARGET_BAND_BYTES,
            "training_removed_assignments": [
                sum(v for e, v in c.items() if e not in ids)
                for c, ids in zip(counts, keep)],
        }
    return result


def alias_slots(gate_weights, retained, original):
    import numpy as np
    weights = np.asarray(gate_weights, dtype=np.float32)
    if weights.shape != (128, 5120) or not np.isfinite(weights).all():
        raise RuntimeError("router weight geometry or values differ")
    norms = np.linalg.norm(weights, axis=1)
    if np.any(norms == 0):
        raise RuntimeError("zero router vector cannot define cosine similarity")
    normalized = weights / norms[:, None]
    # Sorted IDs make exact cosine ties choose the lowest expert ID.
    keep = np.asarray(sorted(retained), dtype=np.int32)
    nearest = keep[np.argmax(normalized @ normalized[keep].T, axis=1)]
    nearest[keep] = keep
    slots = {expert: slot for slot, expert in enumerate(original)}
    return [slots[int(e)] for e in nearest], nearest.tolist()
