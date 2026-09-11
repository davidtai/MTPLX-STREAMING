# W85 — DSV4.1-Flash expert-residency program (CPU analysis, no GPU, no model load)

**Headline verdict: 20 tok/s on the 16K cell is UNREACHABLE by any expert-residency /
streaming-policy change at a box-feasible plan, for two independent reasons that the
measured receipts already show.** (1) The decode miss floor is **compulsory and flat
against capacity** in the reachable range: served AR-16K sits at **0.741 hit / 61.9
misses/token** at 49 slots/layer, and both the census (cold LRU == Belady == 0.733 at
every budget ≥115 slots) and the in-process 60→88 GiB flatness say more cache does not
move it — because the 16K prefill saturates the bank (~all 384 experts/layer touched),
so 49–120 slots retain <32 % and warm ≈ cold. (2) **Decode is not SSD-bound at the
operating point**: a 3.4× swing in bytes/token (61.9 vs 18.25 misses/token on two real
AR-16K runs) produced the *same* ~1.85 tok/s, and realized decode bandwidth is
2.1 GB/s (AR) / 6.65 GB/s (DSpark) — far below the 9.8–12.5 GiB/s SSD ceiling. Per-layer
allocation and working-set pins both **lose to uniform LRU** on the measured trace, and
token-to-token route predictability is ~19 %, so no policy manufactures the missing
hits. Best achievable *SSD-bound ceiling* (if the compute bound were lifted): AR
~11.5 tok/s, DSpark ~8 tok/s — still under 20. The path to 20 tok/s is fewer bytes at
the source (smaller record / top-k) **plus** the decode compute/dispatch lane
(W73/W76/W78), not residency.

Author: Opus 4.8 worker (`w85/residency-program`, off `6647e0003`). CPU-only; no MLX
import, no GPU, no model load; ≤1.5 GB. Calculator + CPU test:
`scripts/deepseek_v41/w85_residency_sim.py`, `tests/test_w85_residency_sim.py`
(8 tests green).

---

## 0. Provenance and the trace-availability caveat

| Input | Source | Regime |
|---|---|---|
| Routing census (LRU/Belady/frequency/verify curves) | `docs/deepseek-v41/receipts/routing_census_1024.json` | **context 1024, decode 64**, CPU forward, text-only residents |
| Served 16K cell counters | `gpu-windows/window-32/` + W83 serve logs (`../dsv41-w83/.../w83/serve-*.log`) | **16K served**, full residents (60 GiB) |
| In-process 16K cell | `gpu-windows/window-30/ar-16k-cell16k{,-60g}.json` | 16K, cold session |
| Plan / resident bytes | `W36_PERSISTENT_SLOT_CAPACITY.md`, `W62_MEMORY_PROFILE.md`, `W81_VERIFY_SWITCH.md`, `mtplx/expert_streaming_models.py::plan_expert_memory`, `expert_profiles.json` | code audit |

**Raw per-(layer,token) expert-id traces are NOT persisted.** The census receipt is 55 KB
of *aggregate* analysis; it captured a trace in-process, computed the curves, and wrote
only the summaries (the `<out>.trace-progress.json` is transient and gone). I cannot
regenerate a trace — that is a model load (barred here: no GPU, no model, ≤1.5 GB). So
per the brief I **use the aggregate curves** and say where a number needs the raw trace
or a fresh forward. Concretely, the census simulated only budgets **115 / 205 / 384
slots/layer at context 1024**; there is no 16K census and no capacity point below 115.
Where the 16K cell differs from the 1024 census I anchor on the **served 16K counters**,
which are direct measurements.

**Two resident regimes (this corrected the W85 brief; W81 code audit).** The *served /
DSpark-capable* daemon wires the **full** dense residents (MTP draft head + vision not
skipped) = **17.37 GiB**; the *text-only AR* load used by the census / W36 phase-1 gate
skips them (−8.31 GiB) = 9.07 GiB. **The 16K standard cell runs served → 17.37 GiB
residents.** Both regimes share the same expert-cache slot math (0.700 GiB per
slot/layer), so census curves indexed by slots/layer transfer across regimes.

---

## 1. The 60 GiB plan composition (Q1)

`plan_expert_memory` removes the fixed footprint first, then floors the remainder into
uniform per-layer persistent slots. Exact split of the shipped profile
**`deepseek-v41-mxfp4-75`** (memory_limit 60 GiB, reserve 7 GiB, transient_slots 48,
cache_policy lru, cache_scope layer, split_route_release deferred), served regime:

| Term | Bytes | GiB | Note |
|---|---:|---:|---|
| residents (full dense) | 18,654,901,064 | **17.37** | `spec.resident_bytes` + SWA 5 MiB, **no** text-only discount on the served path |
| **persistent LRU cache** | 36,849,254,400 | **34.32** | **1,960 slots = 49/layer** (per-layer, `slots_per_layer × 40`) |
| transient service pool | 902,430,720 | **0.84** | **ONE GLOBAL pool of 48 slots**, reused one layer at a time — *not* ×40 |
| KV (16,384 × 3,200 B) | 52,428,800 | 0.05 | MLA-compressed; negligible |
| runtime reserve | 7,516,192,768 | 7.00 | prefill-transient headroom |
| **total** | | **59.6** | ✓ (0.4 GiB unallocated) |

One persistent slot/layer costs `40 × 18,800,640 B = 0.700 GiB` (one record in each of the
40 streamed MoE layers). In **decode**, every miss loads a **persistent LRU victim**
(`expert_streaming.py` ~793–810); the global transient pool is prefill/verify scratch and
is never touched by AR decode — matching the served counters (`transient_loads = 0`,
`persistent_loads == misses`, every miss a persistent load, 30/40 layers with ≥1 miss).

**There is no ~34 GiB to reclaim from transient** — the "48 × 40 layers" reading was
wrong; transient is a single 0.84 GiB global pool. Dropping it frees ~1 slot/layer:

| transient_slots | slots/layer @60 GiB | transient tier | verify coverage |
|---:|---:|---:|---|
| 48 (current) | 49 | 861 MiB | covers depth-3 (≤24) **and** depth-5 (≤36) |
| 24 | 50 (+1) | 430 MiB | depth-3 only (≤24); **breaks depth-5** (needs ≥36) |
| 8 | 50 (+1) | 143 MiB | AR only (top-6); breaks any verify |

So the transient lever is a **red herring for capacity** (±1 slot/layer, ≈0.7 GiB). Keep
48 if depth-5 drafting is used; 24 is the floor for depth-3.

Slot counts by served plan (calculator; text-only regime reproduces W36's 72→79, 82→93,
92→108 exactly): **60→49, 70→63, 80→78, 90→92, 100→106 slots/layer.**

---

## 2. LRU hit vs persistent capacity (Q2)

### Census (context 1024) — the only simulated curve
Decode per layer touches ~102 distinct experts (of 384) over 64 tokens with a **median
reuse distance of 12**, far below any budget ≥115, so:

| slots/layer | expert cache | cold LRU | cold Belady | warm (prefill-primed) LRU |
|---:|---:|---:|---:|---:|
| 115 | 80.5 GiB | 0.7331 | 0.7331 | 0.8347 |
| 205 | 143.6 GiB | 0.7331 | 0.7331 | 0.9412 |
| 384 (full) | 268.9 GiB | 0.7331 | 0.7331 | 0.9885 |

**Cold LRU == Belady == 0.733, flat** for every budget ≥115: eviction policy has zero
headroom (Belady, the oracle, ties LRU) and capacity above the working set does nothing.
The 0.733 "within-prompt ceiling" is compulsory — 26.7 % of decode requests are to
experts not yet seen in the decode window. Warming with the prefill working set lifts hit
rate, and *that* scales with capacity (retains more of the ~261 prefill-distinct/layer).

### The 16K cell does NOT realize the warm curve
The census warm gains are a **1024-context artifact** and do not transfer to the 16K cell:

- **Served AR-16K @49 slots (60 GiB): hit 0.741, 61.9 misses/token, 1.164 GB/token** —
  i.e. essentially the *cold* ceiling, not warm@115 (0.835).
- **In-process 16K decode is flat vs capacity**: 2.24 tok/s @60 GiB (peak 67.5 GB) vs
  2.21 tok/s @~88 GiB (peak 87.8 GB) — more cache, no gain (the "ballast 60/88 flat"
  result).
- **Why**: `gate_r5` measured 261 distinct experts/layer touched by a *1024*-token
  prefill (68 % of the bank); a 16K prefill runs ~13 chunks and its union approaches all
  384/layer. 49–120 slots retain ≤32 % of that, and decode re-uses primed experts only
  weakly, so **warm ≈ cold at 16K**.

### Hit / misses / implied SSD-bound tok/s at the program capacities
AR, 240 requests/token; hit held at the measured-flat 0.741 (justified above):

| slots/layer | plan (peak) | AR hit (16K) | miss/tok | GB/tok | tok/s @12.5 GiB/s | @9.8 GB/s |
|---:|---|---:|---:|---:|---:|---:|
| 49 | 60 GiB (~66 GB) ✓ | 0.741 | 61.9 | 1.164 | 11.5 | 8.4 |
| 78 | 80 GiB (~86 GB) ✓ | 0.741 | 61.9 | 1.164 | 11.5 | 8.4 |
| 100 | 95 GiB (~101 GB) marginal | 0.741 | 61.9 | 1.164 | 11.5 | 8.4 |
| 120 | 109 GiB (~115 GB) **infeasible** | 0.741 | 61.9 | 1.164 | 11.5 | 8.4 |

Flat. 20 tok/s @12.5 GiB/s needs ≤36 misses/token (hit ≥0.851) — unreachable at 0.741.
For reference, the *theoretical* 1024-warm ceiling (which the 16K cell does not reach):
115→18.0, 205→50.6, 384→259.6 tok/s @12.5 GiB/s; but 205/384 slots require **160 / 286
GiB total plans — physically impossible on a 110 GiB box.**

### DSpark verify unique-miss per verify (`gate_0`, per-layer union of top-6 sets)
| depth | rows | union/layer | naive/verify | misses/verify | misses/accepted | GB/accepted | tok/s @12.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 4 | 18.7 (median 19, max 24) | 747 | ~330 (served) | 94.0 (served) | 1.687 (served) | 7.95 |
| 5 | 6 | ~24.5 *(extrapolated)* | ~980 | ~431 | ~86 | ~1.62 | ~8.3 |

The depth-5 union is **extrapolated** (gate_0 measured only W=2/3/4; the marginal new
experts/row decays ~0.86× — a real value needs `verify_union_stats --mtp-widths 5 6` on a
fresh trace). Served DSpark implies a resident-hit fraction of only ~0.56 of the union;
≤120 misses/verify needs **≥0.84** — unreachable at the flat, capacity-insensitive hit
rate. Depth-5 lowers misses *per accepted token* only marginally (bigger union, more
accepts) and does not change the SSD-bound ceiling.

---

## 3. Per-layer capacity allocation (Q3)

Routing **is** heterogeneous across layers (census per-layer, 1024):

- distinct experts/layer over decode: mean 102, range **69–132**.
- Layers **20–24 concentrated** (distinct mean 84, oracle-infinite hit ~0.80); layers
  **0–7 and 37–39 diffuse** (distinct mean 109.5, oracle ~0.69). Because the all-6-resident
  ("all-hit") probability scales ~hit⁶, this ~0.11 gap in per-request hit becomes roughly
  **3× in all-hit rate** (0.80⁶≈0.26 vs 0.69⁶≈0.11) — consistent with the "2× all-hit"
  observation in the brief.

**But allocating slots by that heterogeneity LOSES to uniform on the measured trace.**
`gate_1` water-fill (allocate 50–88 slots/layer by routing popularity, same total bytes):

| budget (uniform slots/layer) | uniform miss-rate | frequency-alloc miss-rate | relative Δ |
|---:|---:|---:|---:|
| 115 | 0.189 | 0.277 | **−46.7 %** (worse) |
| 205 | 0.058 | 0.277 | −380 % |
| 384 | 0.009 | 0.277 | −3120 % |

Reasons: decode routing is near-uniform within a layer and unstable train→eval, so
popularity mis-predicts; the distinct-count spread (69–132) is small enough that uniform
is already near-optimal; and every layer sits at its own compulsory floor, so shifting
bytes between layers cannot buy hits. A coarse marginal check on the per-layer warm-miss
deltas (115→205 slot value 0.31–0.52 misses/slot across layers) confirms the spread is
too small to overcome the instability. **Per-layer / miss-rate-weighted allocation yields
no total-miss reduction at equal bytes — the measured result is a regression.**

---

## 4. Frequency-aware / prompt-working-set pins (Q4)

**Static pins lose to LRU at every size** (`gate_1` top-N pinning, hit iff routed expert
in the pin set):

| pins/layer | static-pin miss | LRU miss | Belady miss | pin vs LRU | Belady vs LRU |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.516 | 0.338 | 0.191 | **−52 %** | +43 % |
| 64 | 0.380 | 0.194 | 0.102 | **−96 %** | +47 % |
| 96 | 0.283 | 0.130 | 0.067 | **−118 %** | +49 % |

Pinning roughly *doubles* misses vs a plain LRU of the same size. Belady beats LRU by
~45 %, but Belady is an unimplementable oracle. Cross-layer top-N coverage stdev
(0.037–0.064) is above hy3's dead 0.027 — routing is *more* concentrated than hy3 — yet
still nowhere near enough for pins to win.

**W64 `pin_working_set`** (landed, default-off) exists to give a device route
**slot-stability** guarantee (safety for the W44 deferred gather), **not** a hit-rate win;
its own design note and this census agree that static pins lose to LRU. **Prompt
working-set pinning is not a win here**: after a 16K prefill the per-layer set covering a
useful fraction of decode routes is ~the whole bank (prefill touches ~all 384/layer), so
pinning it *is* full residency — impossible on the box. So the answer to "does pinning the
post-prefill working set beat LRU" is **no** at every reachable capacity.

---

## 5. Route predictability for prefetch (Q5)

The census computes no temporal (t vs t−1) prediction and the raw trace is not persisted,
so precision/recall of a t−1 prefetch cannot be computed directly; the best available
proxy is the `gate_0` consecutive-position dedup:

- **Temporal (token t vs t−1), per layer**: 2-position union = 10.87 of 12 → the two top-6
  sets share only **1.13 experts** → predicting token t's top-6 from t−1's gives
  **precision ≈ recall ≈ 0.19**. Token-to-token routing is weakly predictable.
- **Within a DSpark verify**: 4-row per-layer dedup factor 1.28 (18.7 union vs 24 naive) →
  ~22 % of a verify's per-layer requests repeat across its rows — real, and already
  exploited by reading the union once. Cross-verify structure (a verify's bonus row → the
  next verify) is not separately traced.

Implication: a prefetcher can at best hide latency for the ~19 %-predictable next-token
experts; it cannot manufacture the compulsory hits, and decode is not SSD-bound anyway
(§6), so prefetch does not move the cell.

---

## 6. Verdict (Q6)

**20 tok/s is not reachable by an expert-residency / streaming-policy change at any
box-feasible plan.** Two independent walls:

1. **Compulsory miss floor, flat vs capacity in the reachable range.** AR-16K decode is
   0.741 hit / 61.9 misses/token at 49 slots and stays there through 120 slots (measured
   flatness + prefill-bank saturation + cold==Belady). Reaching AR ≤33 misses (hit ≥0.86)
   needs the census warm points at 205/384 slots — 160/286 GiB plans, impossible on a
   110 GiB box. Per-layer allocation (§3) and pins (§4) lose to uniform LRU; prefetch
   recall is ~19 % (§5). The DSpark verify is the same story: 330 misses/verify needs to
   fall to ≤120 (resident-hit 0.56→0.84), unreachable at the flat, capacity-insensitive
   hit rate.

2. **Decode is not SSD-bound at the operating point.** Two real AR-16K runs at the *same*
   60 GiB plan: 61.9 misses/token (1.164 GB, 1.83 tok/s) vs 18.25 misses/token (0.343 GB,
   1.88 tok/s) — a 3.4× byte difference with **no** tok/s change. Realized decode
   bandwidth is 2.1 GB/s (AR) and 6.65 GB/s (DSpark), far below the 9.8–12.5 GiB/s SSD
   ceiling. So even if the miss target were hit, tok/s would not rise until the
   compute/dispatch bound (the W73/W76/W78 decode-attention lane) is lifted.

**Requested-capacity feasibility (110 GiB hard, ~100 GiB buffer, W80 ring):** 49 slots
(60 GiB, ~66 GB peak) ✓; 78 slots (80 GiB, ~86 GB) ✓; 100 slots (~95 GiB, ~101 GB peak)
**marginal**; 120 slots (~109 GiB, ~115 GB peak) **infeasible**.

**Best achievable (SSD-bound ceiling, if the compute bound were removed):** AR ~11.5 tok/s
@12.5 GiB/s (8.4 @9.8) at the flat 62-miss floor — identical at 49/78/100/120 slots;
DSpark depth-3 ~7.95/5.8, depth-5 ~8.3 (extrapolated). All below 20. Actual, compute-bound
today: **AR 1.83 tok/s, DSpark 3.94 tok/s** (both 60 GiB, 16K cell, TTFT 191 s / 149 s,
peak ~70 GB).

**The one residency lever with unmeasured upside:** a larger plan for the **DSpark lane
only** — 80 GiB / 78 slots fits the box (~86 GB peak) and DSpark's cross-cycle reuse (hit
0.922 vs AR 0.741) *might* convert +29 slots into hits better than AR's near-random decode
does. But AR's measured flatness makes the prior weak; it needs one served-DSpark-16K
window at an 80 GiB plan to settle. That is the single next experiment worth GPU time; a
uniform-vs-per-layer split is not (per-layer loses).

**Real path to 20 tok/s (out of this task's scope, flagged for planning):** cut
bytes/token at the source — a smaller expert record (sub-mxfp4 / on-the-fly requant) or
fewer distinct experts touched (top-k reduction, router-temperature) — **and** lift the
decode compute/dispatch bound. Residency reshuffling is not the lever.
