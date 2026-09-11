#!/usr/bin/env python3
"""W85 residency program — plan/hit/SSD-bound projection calculator (CPU-only).

This is an *analysis* tool, not a runtime path.  It needs **no MLX, no model
load, no GPU** and reads no large receipts: every measured input is embedded
below as a constant with its provenance so the numbers in
``docs/deepseek-v41/W85_RESIDENCY_PROGRAM.md`` are reproducible from arithmetic
alone.  (If a future caller imports MLX here, pin it to CPU first —
``mx.set_default_device(mx.cpu)`` — per the box worker rule; this module never
imports it.)

It reproduces three things:

1. ``plan_slots`` — the uniform component-bank persistent-slot math from
   ``mtplx.expert_streaming_models.plan_expert_memory`` for the DSV4.1-Flash
   mxfp4 spec, so the 60 GiB profile composition and the transient-slot reclaim
   are exact.  Calibrated against W36's real-manifest table (72->79, 82->93,
   92->108 slots at transient=6/kv=4096).

2. ``misses_per_token`` / ``ssd_bound_tok_s`` — convert an expert-cache hit rate
   into AR misses/token and the *implied SSD-bound* decode ceiling at a given
   realized bandwidth.  (The receipts show decode is NOT SSD-bound at today's
   operating point; these are upper bounds — see the W85 report §6.)

3. ``dspark_verify_misses`` — the DSpark depth-3 verify miss projection from the
   per-layer union (gate_0) and a resident-hit fraction.

Run ``python3 scripts/deepseek_v41/w85_residency_sim.py`` for the tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Measured constants (provenance in comments) — DSV4.1-Flash mxfp4, 40 MoE      #
# layers, 384 experts/layer, top_k=6.                                           #
# --------------------------------------------------------------------------- #
GiB = 2 ** 30

# Real-manifest record size (W36_PERSISTENT_SLOT_CAPACITY.md §"70"): the physical
# per-expert record is 18,800,640 B (17.93 MiB).  The served cache byte counter
# accounts a miss at 18,800,000 B exactly (bytes_read_per_token / misses on the
# window-32 / W83 serve logs), so byte->tok/s uses BYTES_PER_MISS_ACCOUNTED.
EXPERT_RECORD_BYTES = 18_800_640          # physical record (plan math)
BYTES_PER_MISS_ACCOUNTED = 18_800_000     # served counter accounting (tok/s math)
N_MOE_LAYERS = 40
N_EXPERTS = 384
TOP_K = 6
AR_REQ_PER_TOKEN = N_MOE_LAYERS * TOP_K   # 240 (matches served expert_requests/token)

# Fixed-side terms from W36 §"70" (real manifest) + expert_profiles.json profile
# deepseek-v41-mxfp4-75.
SPEC_RESIDENT_BYTES = 18_649_658_184      # spec.resident_bytes (17.369 GiB)
SWA_WINDOW_BYTES = 5_242_880              # additional_resident (~5 MiB)
TEXT_ONLY_DISCOUNT_BYTES = 8_920_505_736  # W21 text-only resident skip (8.308 GiB)
RUNTIME_RESERVE_BYTES = 7_516_192_768     # 7 GiB (profile runtime_reserve_bytes)
KV_BYTES_PER_TOKEN = 3_200               # W62_MEMORY_PROFILE.md (16384*3200=52.4MB@16K)

# The per-slot cost is ONE expert record in EACH of the 40 streamed layers
# (uniform component banks, cache_scope=layer); a persistent "slot/layer" unit
# therefore costs:
PER_SLOT_LAYER_BYTES = N_MOE_LAYERS * EXPERT_RECORD_BYTES   # ~0.700 GiB

# TWO resident regimes (this was the W85-brief correction, W81_VERIFY_SWITCH.md):
#  * SERVED / DSpark-capable production daemon wires the FULL dense residents
#    (MTP draft head + vision NOT skipped) -> no text-only discount -> 17.37 GiB.
#    This is the regime of the 16K standard cell.  60 GiB -> 49 slots/layer.
#  * TEXT-ONLY AR load (routing-census CPU forward, W36 phase-1 gate) skips the
#    MTP/vision residents -> 8.308 GiB discount -> 9.07 GiB.  60 GiB -> 61 slots.
RESIDENT_BYTES_SERVED = SPEC_RESIDENT_BYTES + SWA_WINDOW_BYTES               # 17.37 GiB
RESIDENT_BYTES_TEXTONLY = RESIDENT_BYTES_SERVED - TEXT_ONLY_DISCOUNT_BYTES   # 9.07 GiB

# Profile deepseek-v41-mxfp4-75 (expert_profiles.json):
PROFILE_TOTAL_GIB = 60          # memory_limit_bytes = 64,424,509,440
PROFILE_TRANSIENT_SLOTS = 48
PROFILE_KV_TOKENS = 16384       # max_live_kv_tokens (cache_policy lru = static plan)

# Box ceiling (memory: "Box 110 GB hard limit"; usable buffer ~100 GB TOTAL).
BOX_USABLE_TOTAL_GIB = 100

# SSD bandwidth (docs TEST_MACHINES.md; task brief).
SSD_CEILING_GIB_S = 12.5        # ~13.42 GB/s
SSD_REALIZED_SWITCH_GB_S = 9.8  # decimal GB/s, drive busy during the switch

# --------------------------------------------------------------------------- #
# Measured routing-census anchors (routing_census_1024.json, context=1024,     #
# decode=64, CPU forward captured the trace then simulated LRU/Belady offline). #
# budget key = persistent slots/layer.                                          #
# --------------------------------------------------------------------------- #
CENSUS_CONTEXT_TOKENS = 1024
CENSUS_DECODE_TOKENS = 64
# slots/layer -> (cold_lru_hit, warm_prefill_primed_lru_hit)
CENSUS_HIT = {
    115: (0.7330729, 0.8347005),
    205: (0.7331380, 0.9412109),
    384: (0.7331380, 0.9885417),
}
CENSUS_COLD_CEILING = 0.7331    # flat for every budget >= 115 (reuse dist << cap)

# Served 16K-cell ground truth (window-32 / W83 serve logs), 60 GiB plan:
SERVED_AR_16K = dict(hit=0.741078, miss_per_token=61.9111, gb_per_token=1.163968512,
                     tok_s=1.831, completion_tokens=270)
SERVED_DSPARK_16K = dict(hit=0.921639, miss_per_accepted=94.0333,
                         gb_per_accepted=1.687461888, tok_s=3.938,
                         accepted_tokens=180, accepted_per_cycle=3.5)
# Anomalous short/warm AR-16K run (same 60 GiB plan): 3.4x fewer bytes, SAME tok/s.
SERVED_AR_16K_WARMISH = dict(hit=0.923388, miss_per_token=18.2536,
                             gb_per_token=0.34317980, tok_s=1.884, completion_tokens=138)

# gate_0_verify_union (per-layer union of top-6 sets across W consecutive
# positions; max = W*6).  decode_consecutive rows MEASURED at W=2,3,4:
VERIFY_UNION_MEASURED = {2: 10.8706, 3: 15.0556, 4: 18.6857}   # per-layer mean u
# The marginal "new experts per added row" decays ~0.86x geometrically
# (u(2)-u(1)=4.87, u(3)-u(2)=4.19, u(4)-u(3)=3.63; ratios 0.86, 0.87), so
# W=5,6 are EXTRAPOLATED (no receipt beyond W=4 — a real value needs re-running
# verify_union_stats with --mtp-widths 5 6 on a fresh trace).
def _verify_union_mean(positions: int) -> float:
    u = {1: 6.0, **VERIFY_UNION_MEASURED}
    if positions in u:
        return u[positions]
    last = max(k for k in u)
    val = u[last]
    marg = u[last] - u[last - 1]
    for _ in range(last, positions):
        marg *= 0.86
        val += marg
    return val
# depth d speculative decode verifies d+1 rows (d draft + 1 bonus/verify row).
VERIFY_ROWS = {3: 4, 5: 6}  # depth 3 -> 4-row; depth 5 -> 6-row (W82)


@dataclass
class Plan:
    total_gib: float
    transient_slots: int
    kv_tokens: int
    resident_gib: float
    kv_gib: float
    reserve_gib: float
    transient_gib: float
    fixed_gib: float
    available_gib: float
    slots_per_layer: int
    expert_cache_gib: float
    unallocated_gib: float


def plan_slots(total_gib: float, transient_slots: int, kv_tokens: int,
               resident_bytes: int = RESIDENT_BYTES_SERVED) -> Plan:
    """Uniform persistent slots/layer for a TOTAL memory budget (component banks).

    Mirrors plan_expert_memory: fixed = resident + kv + reserve + transient;
    persistent budget = total - fixed, floored into whole slots/layer.  Default
    ``resident_bytes`` is the SERVED (full-resident) regime; pass
    ``RESIDENT_BYTES_TEXTONLY`` to reproduce W36's text-only phase-1 table.
    """
    total = total_gib * GiB
    kv = kv_tokens * KV_BYTES_PER_TOKEN
    transient = transient_slots * EXPERT_RECORD_BYTES
    fixed = resident_bytes + kv + RUNTIME_RESERVE_BYTES + transient
    available = max(0, total - fixed)
    slots = min(N_EXPERTS, int(available // PER_SLOT_LAYER_BYTES))
    cache = slots * PER_SLOT_LAYER_BYTES
    return Plan(
        total_gib=total_gib, transient_slots=transient_slots, kv_tokens=kv_tokens,
        resident_gib=resident_bytes / GiB, kv_gib=kv / GiB,
        reserve_gib=RUNTIME_RESERVE_BYTES / GiB, transient_gib=transient / GiB,
        fixed_gib=fixed / GiB, available_gib=available / GiB,
        slots_per_layer=slots, expert_cache_gib=cache / GiB,
        unallocated_gib=(total - fixed - cache) / GiB,
    )


def slots_to_cache_gib(slots_per_layer: int) -> float:
    return slots_per_layer * PER_SLOT_LAYER_BYTES / GiB


def slots_to_total_plan_gib(slots_per_layer: int, transient_slots: int = PROFILE_TRANSIENT_SLOTS,
                            kv_tokens: int = PROFILE_KV_TOKENS,
                            resident_bytes: int = RESIDENT_BYTES_SERVED) -> float:
    kv = kv_tokens * KV_BYTES_PER_TOKEN
    transient = transient_slots * EXPERT_RECORD_BYTES
    fixed = resident_bytes + kv + RUNTIME_RESERVE_BYTES + transient
    return (slots_per_layer * PER_SLOT_LAYER_BYTES + fixed) / GiB


def peak_gb_estimate(total_plan_gib: float) -> float:
    """Rough process peak RSS with the W80 window ring (~+6 GB over the plan;
    W62 host_overhead+alloc-cache minus macOS floor, tightened by W80)."""
    return total_plan_gib + 6.0


def misses_per_token(hit_rate: float) -> float:
    """AR: 240 expert requests/token; miss = (1-hit)*240."""
    return (1.0 - hit_rate) * AR_REQ_PER_TOKEN


def ssd_bound_tok_s(bytes_per_token: float, bw_gib_s: float | None = None,
                    bw_gb_s: float | None = None) -> float:
    """Decode tok/s IF perfectly SSD-bound (upper bound only)."""
    if bw_gib_s is not None:
        bw = bw_gib_s * GiB
    else:
        bw = bw_gb_s * 1e9
    return bw / bytes_per_token


def dspark_verify_misses(union_per_layer: float, resident_hit_fraction: float) -> float:
    """Distinct record reads that MISS across a 4-row verify (all 40 layers)."""
    naive = union_per_layer * N_MOE_LAYERS
    return naive * (1.0 - resident_hit_fraction)


# Capacities the W85 program simulates (coordinator directive), slots/layer:
PROGRAM_CAPACITIES = (49, 78, 100, 120)


def _print_plan_table() -> None:
    print("== SERVED plan (full residents 17.37 GiB, transient=48 global, kv=16384) ==")
    print(f"{'total':>6} {'slots/L':>8} {'cache GiB':>10} {'fixed GiB':>10} "
          f"{'resid':>6} {'kv':>6} {'resv':>5} {'trans':>6}")
    for g in (60, 70, 80, 90, 100):
        p = plan_slots(g, PROFILE_TRANSIENT_SLOTS, PROFILE_KV_TOKENS)
        print(f"{g:>6} {p.slots_per_layer:>8} {p.expert_cache_gib:>10.2f} "
              f"{p.fixed_gib:>10.2f} {p.resident_gib:>6.2f} {p.kv_gib:>6.3f} "
              f"{p.reserve_gib:>5.2f} {p.transient_gib:>6.3f}")
    print("\n== transient reclaim @60 GiB (transient is GLOBAL, ~free) ==")
    base = plan_slots(60, 48, PROFILE_KV_TOKENS).slots_per_layer
    for t in (48, 24, 8, 6):
        p = plan_slots(60, t, PROFILE_KV_TOKENS)
        print(f"  transient={t:>2}: {p.slots_per_layer} slots/L "
              f"(+{p.slots_per_layer - base} vs 48), transient tier {p.transient_gib * 1024:.0f} MiB")

    print("\n== program capacities -> plan/peak (110 GiB hard, ~100 GiB buffer) ==")
    for s in PROGRAM_CAPACITIES:
        tot = slots_to_total_plan_gib(s)
        peak = peak_gb_estimate(tot)
        if peak > 110:
            reach = "INFEASIBLE (peak>110 hard)"
        elif peak > BOX_USABLE_TOTAL_GIB:
            reach = "MARGINAL (peak>100 buffer)"
        else:
            reach = "OK"
        print(f"  {s:>3} slots/L -> cache {slots_to_cache_gib(s):>5.1f} GiB, "
              f"plan ~{tot:>5.1f} GiB, peak ~{peak:>5.1f} GB  [{reach}]")


def _hit_for_capacity_16k(slots: int) -> float:
    """AR decode hit for the 16K cell.  MEASURED-flat: served 0.741 @49 slots and
    in-process decode is flat 60->88 GiB; the 16K prefill saturates the bank
    (~all 384 experts/layer) so 49-120 slots retain <32% -> warm≈cold.  Returns
    the measured cold-session value; NOT the 1024-census warm curve."""
    return SERVED_AR_16K["hit"]


def _print_hit_and_tok_s() -> None:
    print("\n== AR: 16K-cell hit / misses / implied SSD-bound tok/s by capacity ==")
    print("   (hit is MEASURED-flat 0.741 for the 16K cold-session cell; the "
          "1024-census warm\n    curve 0.835@115 does NOT transfer — see report §2)")
    print(f"{'slots/L':>7}{'hit16k':>8}{'miss/tok':>9}{'GB/tok':>8}"
          f"{'@12.5GiB/s':>11}{'@9.8GB/s':>10}")
    for s in PROGRAM_CAPACITIES:
        hit = _hit_for_capacity_16k(s)
        m = misses_per_token(hit)
        bpt = m * BYTES_PER_MISS_ACCOUNTED
        print(f"{s:>7}{hit:>8.3f}{m:>9.1f}{bpt/1e9:>8.3f}"
              f"{ssd_bound_tok_s(bpt, bw_gib_s=SSD_CEILING_GIB_S):>11.1f}"
              f"{ssd_bound_tok_s(bpt, bw_gb_s=SSD_REALIZED_SWITCH_GB_S):>10.1f}")
    need = SSD_CEILING_GIB_S * GiB / 20 / BYTES_PER_MISS_ACCOUNTED
    print(f"  20 tok/s @12.5GiB/s needs <= {need:.0f} miss/tok (hit >= "
          f"{1 - need / AR_REQ_PER_TOKEN:.3f}) -> UNREACHABLE (flat at 0.741)")
    print("  1024-census warm CEILING (theoretical, not the 16K cell):")
    for s, (cold, warm) in CENSUS_HIT.items():
        m = misses_per_token(warm)
        print(f"    {s} slots/L warm={warm:.3f} -> {m:.1f} miss/tok, "
              f"{ssd_bound_tok_s(m*BYTES_PER_MISS_ACCOUNTED, bw_gib_s=SSD_CEILING_GIB_S):.1f} tok/s @12.5")

    print("\n== DSpark verify misses: depth-3 (4-row) and depth-5 (6-row) ==")
    print("   resident-hit frac held at served ~0.56 (capacity-flat, see §2); "
          "depth-5 union EXTRAPOLATED")
    for depth, rows in VERIFY_ROWS.items():
        u = _verify_union_mean(rows)
        naive = u * N_MOE_LAYERS
        acc = SERVED_DSPARK_16K["accepted_per_cycle"] if depth == 3 else 5.0
        misses = dspark_verify_misses(u, 0.56)
        gb_acc = misses * BYTES_PER_MISS_ACCOUNTED / acc / 1e9
        tag = "(measured)" if depth == 3 else "(extrapolated union, est. accept 5/cycle)"
        print(f"  depth-{depth} {rows}-row: union/layer {u:.1f}, naive {naive:.0f}/verify, "
              f"~{misses:.0f} miss/verify, ~{misses/acc:.0f} miss/accepted -> "
              f"{gb_acc:.2f} GB/acc -> {SSD_CEILING_GIB_S*GiB/1e9/gb_acc:.1f} tok/s @12.5  {tag}")
    umax4, umax6 = 4 * TOP_K, 6 * TOP_K
    print(f"  transient sizing: depth-3 max union {umax4} (need ts>=24); "
          f"depth-5 max union {umax6} (need ts>=36 -> ts=48 OK, ts=24 BREAKS depth-5)")
    print(f"  target <=120 miss/verify needs resident-hit >= "
          f"{1 - 120 / (_verify_union_mean(4) * N_MOE_LAYERS):.2f} (served 0.56) -> "
          "UNREACHABLE without a big DSpark-lane capacity/dedup win")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    ap.parse_args()
    _print_plan_table()
    _print_hit_and_tok_s()


if __name__ == "__main__":
    main()
