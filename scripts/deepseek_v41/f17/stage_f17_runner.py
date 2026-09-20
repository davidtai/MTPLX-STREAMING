"""Stage the F17 per-layer extension-row edits into a STAGED copy of ``extension.py``.

Like the F6/F12 stagers, this applies anchored, unique-string replacements to a
STAGED COPY of the retained ``extension.py`` (never the committed receipt, never
anything under ``mtplx/``) and asserts that reversing every edit recovers the input
byte-for-byte, so the staged runner is a provable minimal delta.  It does NO MLX work
and is CPU-safe.

The retained ``grow_rows`` gives every routed layer the same ``capacity-old`` extension
rows.  These six edits route that count through ``allocation.allocate`` so each layer
can get a DIFFERENT number of extension rows, while the TOTAL stays
``len(layers)*(capacity-old)`` -- so every plan/accounting total (persistent_slots,
persistent_cache_bytes, total_limit_bytes, allocated_bytes, and the final physical
accounting check) is byte-identical to the uniform path, and the retained receipt gates
(run_full.py:735-736 and :839-840, which key off ``plan.slots_per_layer`` /
``plan.persistent_slots_by_layer`` -- both LEFT UNIFORM here) still pass.

  EDIT 1  Route ONCE, right after ``added=capacity-old``:
            import allocation as _f17_alloc
            _f17_added = _f17_alloc.allocate(runtime, capacity=capacity, old=old)
          ``allocate`` reads ``MTPLX_DSV41_F17_ALLOC`` once and returns
          ``{layer: added_L}``.  Unset/``uniform`` -> ``capacity-old`` for every layer,
          so one staged tree serves both the control and the candidate arms.

  EDIT 2  resize bank grows to the per-layer capacity:
            grow_bank(bank,capacity,...)  ->  grow_bank(bank,old+_f17_added[layer],...)
  EDIT 3  extension bank is sized per layer:
            MlxComponentBank(capacity=added,...)  ->  MlxComponentBank(capacity=_f17_added[layer],...)
  EDIT 4  the per-layer physical-slot loop count is per layer:
            for offset in range(added):  ->  for offset in range(_f17_added[layer]):
  EDIT 5  per-layer policy fields (slot labels, persistent_slots, _persistent_capacity,
          slot_count, _protected_cap) use ``c_L = old + _f17_added[layer]``.
  EDIT 6  the per-layer route-capacity map uses ``old + _f17_added[layer]``.

The scalar ``pool._persistent_route_capacity`` (line 93) is LEFT UNCHANGED at the
uniform-equivalent ``capacity``: it is report-only (read only by the pool telemetry
snapshot in mtplx/expert_slots.py) and mirrors ``plan.slots_per_layer``, the documented
uniform-equivalent scalar; the decode hot path (mtplx/expert_slots.py::_physical) reads
the per-layer ``_persistent_route_capacities`` map that EDIT 6 fills.

When ``MTPLX_DSV41_F17_ALLOC`` is unset/``uniform`` every ``_f17_added[layer]`` equals
``capacity-old``, so ``old+_f17_added[layer]==capacity`` and every edited expression
reduces to the retained text -- the staged path then behaves EXACTLY like uniform
``grow_rows``.

The runner needs ``scripts/deepseek_v41/f17`` on PYTHONPATH so ``import allocation``
resolves, alongside the staged sources dir so ``from extension import grow_rows`` does.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# A marker unique to F17's inserts, used to refuse a double-apply.
_MARKER = "_f17_added"

# EDIT 1 -- route once, right after ``added=capacity-old`` (4-space indent).
_E1_OLD = "    added=capacity-old"
_E1_NEW = (
    "    added=capacity-old\n"
    "    import allocation as _f17_alloc  # F17: per-layer extension-row split, routed once by MTPLX_DSV41_F17_ALLOC\n"
    "    _f17_added = _f17_alloc.allocate(runtime, capacity=capacity, old=old)  # unset/'uniform' -> uniform split (retained behaviour)"
)

# EDIT 2 -- resize bank grows to the per-layer capacity (16-space indent).
_E2_OLD = "                grow_bank(bank,capacity,mx=mx)"
_E2_NEW = "                grow_bank(bank,old+_f17_added[layer],mx=mx)"

# EDIT 3 -- extension bank sized per layer (16-space indent; first line of the call).
_E3_OLD = "                bank=MlxComponentBank(capacity=added,record=pool._record_map[layer,0],"
_E3_NEW = "                bank=MlxComponentBank(capacity=_f17_added[layer],record=pool._record_map[layer,0],"

# EDIT 4 -- per-layer physical-slot loop count (12-space indent).
_E4_OLD = "            for offset in range(added):"
_E4_NEW = "            for offset in range(_f17_added[layer]):"

# EDIT 5 -- per-layer policy fields (12-space indent, 4-line block -> 5-line block).
_E5_OLD = (
    "            policy._slot_to_expert.extend([None]*added)\n"
    "            policy.persistent_slots=policy._persistent_capacity=capacity\n"
    "            policy.slot_count=capacity+48\n"
    "            policy._protected_cap=max(1,int(capacity*.8))"
)
_E5_NEW = (
    "            policy._slot_to_expert.extend([None]*_f17_added[layer])\n"
    "            _f17_c=old+_f17_added[layer]\n"
    "            policy.persistent_slots=policy._persistent_capacity=_f17_c\n"
    "            policy.slot_count=_f17_c+48\n"
    "            policy._protected_cap=max(1,int(_f17_c*.8))"
)

# EDIT 6 -- per-layer route-capacity map (8-space indent). The scalar on the line
# ABOVE (pool._persistent_route_capacity=capacity) is intentionally untouched.
_E6_OLD = "        pool._persistent_route_capacities={layer:capacity for layer in layers}"
_E6_NEW = "        pool._persistent_route_capacities={layer:old+_f17_added[layer] for layer in layers}"

_EDITS = (
    ("EDIT 1 route-once allocate", _E1_OLD, _E1_NEW),
    ("EDIT 2 resize bank capacity", _E2_OLD, _E2_NEW),
    ("EDIT 3 extension bank capacity", _E3_OLD, _E3_NEW),
    ("EDIT 4 physical-slot loop count", _E4_OLD, _E4_NEW),
    ("EDIT 5 per-layer policy fields", _E5_OLD, _E5_NEW),
    ("EDIT 6 per-layer route-capacity map", _E6_OLD, _E6_NEW),
)


def stage(source_text: str) -> str:
    """Apply the six F17 edits; assert reversing all six recovers ``source_text``."""
    if _MARKER in source_text:
        raise RuntimeError("F17 staging already applied (_f17_added marker present)")
    for label, old, _new in _EDITS:
        n = source_text.count(old)
        if n != 1:
            raise RuntimeError(f"{label} anchor is not unique ({n} occurrences)")

    updated = source_text
    for _label, old, new in _EDITS:
        updated = updated.replace(old, new)

    # Round-trip: reversing every edit must recover the input byte-for-byte.
    recovered = updated
    for _label, old, new in reversed(_EDITS):
        recovered = recovered.replace(new, old)
    if recovered != source_text:
        raise RuntimeError("F17 staging changed extension.py beyond its six edits")
    return updated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the F17-patched extension.py copy")
    ap.add_argument("--extension", required=True,
                    help="input extension.py (retained/archived; never overwritten)")
    ap.add_argument("--out", required=True,
                    help="staged patched extension.py path (must differ from --extension)")
    args = ap.parse_args(argv)
    src_path, out_path = Path(args.extension), Path(args.out)
    if src_path.resolve() == out_path.resolve():
        raise RuntimeError("--out must differ from --extension (never patch the committed receipt in place)")
    src = src_path.read_text()
    staged = stage(src)
    out_path.write_text(staged)
    print("input_sha256", hashlib.sha256(src.encode()).hexdigest())
    print("staged_sha256", hashlib.sha256(staged.encode()).hexdigest())
    print("staged_path", str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
