"""F39: re-derive the decode capacity ladder for the tcq3 slot so the ~33% headroom is actually admitted.

The retained mxfp4 ladder (packed_admission.resolve_admission) sizes the bank off the 17,694,720-B mxfp4 weight
slot and charges the 3,086,136,060-B resident packed-scale bank.  A tcq3 slot is the whole record (13,290,496 B,
routs included) and the only resident side-table is the routs (298,844,160 B).  So for the SAME live memory
envelope more decode rows fit.

This module does NOT change the 110 GB / 100 GiB / allocator-limit accounting rules — it re-runs the SAME gates
with two substitutions:
  * per-slot bytes: WEIGHTS (17,694,720) -> TCQ3_SLOT (13,290,496)
  * resident side-table: PACKED scales (3,086,136,060) -> ROUTS (298,844,160)
and raises the capacity search ceiling (the mxfp4 ceiling of 112 was set by the mxfp4 slot size).  It starts from
the mxfp4 admission RESULT dict (so every proven fixed bound — KV, engine, host reserve, cache allowance, page
padding, embedding/lookup/expansion credits — is inherited unchanged) and re-solves only the row-dependent terms.

``retarget`` is a pure function of the mxfp4 result dict + (base, wired) and is unit-tested.  ``resolve_admission``
is the runtime entry the staged runner routes to when the tcq3 lane is armed: it calls the mxfp4 resolver first
(inheriting its bounds) then retargets.
"""
from __future__ import annotations

from mtplx.deepseek_v41_memory_profile import DEFAULT_BOX_BUDGET_BYTES   # 110 GB decimal box ceiling

# retained mxfp4 constants (packed_admission.py / packed_storage.py)
WEIGHTS = 17_694_720            # mxfp4 weight-code bytes per slot (scales resident/separate)
PACKED = 3_086_136_060          # resident packed-scale bank bytes
# tcq3 (F38 record) constants
TCQ3_SLOT = 13_290_496          # whole tcq3 record per slot (code + routs)
ROUTS = 298_844_160             # whole-bank routs resident (15,360 x (2304+2304+5120) x 2)
N_LAYERS = 40
TRANSIENT_SLOTS = 48
WIRED_CEILING = 100 * 1024 ** 3       # 100 GiB wired ceiling (never raised)
WIRED_HEADROOM = 1024 ** 3            # +1 GiB headroom, as in packed_admission


def _slot_term(capacity: int, slot_bytes: int) -> int:
    """Resident expert bytes at ``capacity`` decode slots/layer: (40*cap + 48 transient) * slot_bytes."""
    return (N_LAYERS * capacity + TRANSIENT_SLOTS) * slot_bytes


def retarget(mxfp4_admission: dict, *, base: int, wired: int, search_ceiling: int = 160) -> dict:
    """Re-derive the tcq3 decode capacity from the mxfp4 admission result, under the SAME memory ceilings.

    Model: the mxfp4 STEADY decode active bound decomposes as
        steady_mxfp4 = fixed + slot_term(mxfp4_cap, WEIGHTS) + PACKED
    where ``fixed`` (KV, engine, transient scheduler state, margins) is codec-independent.  The tcq3 steady bound
    at capacity ``c`` is  fixed + slot_term(c, TCQ3_SLOT) + ROUTS.  We admit the largest ``c`` (down from
    ``search_ceiling``, never below the mxfp4 capacity) whose steady bound clears every unchanged gate.
    """
    orig = mxfp4_admission
    mxfp4_cap = orig["decode_slots_per_layer"]
    steady_mxfp4 = orig["steady_decode_active_bound_bytes"]
    cache = orig["decode_cache_allowance_bytes"]
    allocator_limit = orig["allocator_limit_bytes"]
    host_reserve = orig["host_reserve_bytes"]           # includes embedding/lookup/expansion host allowances
    # codec-independent fixed part of the steady active bound
    fixed = steady_mxfp4 - _slot_term(mxfp4_cap, WEIGHTS) - PACKED
    chosen = None
    for capacity in range(search_ceiling, mxfp4_cap - 1, -1):
        steady = fixed + _slot_term(capacity, TCQ3_SLOT) + ROUTS
        active = steady                                  # steady dominates as rows grow; tcq3 has no packed-scale transition peak
        physical = base + host_reserve + active + cache
        if (physical <= DEFAULT_BOX_BUDGET_BYTES
                and active + cache <= allocator_limit
                and wired + active + cache + WIRED_HEADROOM <= WIRED_CEILING):
            chosen = capacity
            break
    if chosen is None:
        raise RuntimeError("no tcq3 decode capacity fits the live memory envelope")
    steady = fixed + _slot_term(chosen, TCQ3_SLOT) + ROUTS
    net_growth = _slot_term(chosen, TCQ3_SLOT) + ROUTS - (_slot_term(mxfp4_cap, WEIGHTS) + PACKED) \
        + orig.get("growth_payload_bytes", 0)
    result = dict(orig)
    result.update(
        expert_codec="tcq3",
        decode_weight_record_bytes=TCQ3_SLOT,
        source_record_bytes=TCQ3_SLOT,
        resident_packed_scales_bytes=0,
        resident_rout_bytes=ROUTS,
        retired_scale_credit_bytes=PACKED,
        mxfp4_decode_slots_per_layer=mxfp4_cap,
        decode_slots_per_layer=chosen,
        extra_rows_vs_mxfp4=chosen - mxfp4_cap,
        capacity_search_ceiling=search_ceiling,
        steady_decode_active_bound_bytes=steady,
        active_bound_bytes=max(orig.get("prefill_active_bound_bytes", 0), steady),
        physical_bound_bytes=base + host_reserve + steady + cache,
        growth_payload_bytes=net_growth,
        tcq3_slot_bytes=TCQ3_SLOT,
        tcq3_scale_credit_vs_mxfp4_bytes=PACKED - ROUTS,
        bound_scope=("tcq3 re-derivation of the mxfp4 ladder: per-slot 17,694,720->13,290,496 B, resident side-table "
                     "packed scales 3,086,136,060->routs 298,844,160 B; all fixed bounds (KV, engine, host reserve, "
                     "cache, padding, embedding/lookup/expansion credits) inherited from the mxfp4 admission; the "
                     "110 GB physical, 100 GiB wired and allocator-limit gates are unchanged."),
    )
    return result


def resolve_admission(base, wired, *, grow, expected_receipt_hash, strict_allocator, search_ceiling: int = 160):
    """Runtime entry: inherit the mxfp4 bounds then retarget for the tcq3 slot.  Routed to by the staged runner."""
    from packed_admission import resolve_admission as mxfp4_resolve
    mxfp4 = mxfp4_resolve(base, wired, grow=grow, expected_receipt_hash=expected_receipt_hash,
                          strict_allocator=strict_allocator)
    return retarget(mxfp4, base=base, wired=wired, search_ceiling=search_ceiling)
