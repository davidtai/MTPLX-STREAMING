# W87 — DeepSeek-V4.1 single slot pool: one scan-resistant bank, warm at decode start

Flag: `MTPLX_DSV41_SINGLE_SLOT_POOL` (default **off**; the two-tier path is
byte-for-byte unchanged when off, gated on `cache_scope == "layer"`). Arm:
`cell16k_ring_pool` (= `cell16k_ring` + the flag). **The window A/B must run in
`--decode-mode ar`** (see §5). This design was hardened after an adversarial review
that found three flag-on defects in the first cut; §7 records what changed and why.

Evidence for the problem: routing census verdict ([[dsv41-routing-census-verdict]])
and the 16,384-token cell decode counters — 40 streamed MoE layers, top_k 6, 49
slots/layer at the 60 GiB plan. Decode hit rate **0.741** (62 misses/token of 240
assignments); at 20 tok/s the SSD budget is **≤33 misses/token**, so residency is the
last-mile lever ([[dsv41-decode-not-ssd-bound]]).

---

## 1. Why there were two pools

`LayerExpertSlotBank` runs **two tiers**: a per-layer **persistent** tier (49
slots/layer) that **learns only from DECODE routes**, and a global **transient**
scratch (48 slots, shared across layers) through which **every PREFILL miss** is
served. The transient tier made prefill scan-resistant: a 16K prefill streams ~all
384 experts per layer, and routing those misses through the never-recorded transient
scratch guaranteed a prefill scan could not evict the decode hot set.

The cost: the persistent tier learns **nothing** from 16K tokens of prefill, so
**decode starts cold** and re-streams the prompt's working set. Served counters
confirm it — `persistent_loads == expert_misses`, `transient_loads == 0` (an LRU with
a bypassed prefill).

---

## 2. The single-pool admission policy

Under the flag each layer runs **one** resident pool (the `slots_per_layer` physical
slots) with a **2Q / segmented-LRU** policy, warmed by the PREFILL **frequency seed**:

- **Frequency seed → protected.** The switch calls `prepare_prefill_seed(route)`
  before each prefill route (it always did, for the two-tier seed branch); it ranks
  the prompt's routed experts by frequency and picks the top `empty` (= the top
  `slots_per_layer` for a fresh pool). Under the flag those seed experts are admitted
  **PROTECTED**, and **seed-first within each wave** so a seed that appears late in an
  id-sorted wave still claims a slot instead of overflowing to transient.
- **Every other prefill miss → PROBATIONARY** (the eviction end).
- **Decode hit → promote** probationary → protected (2Q); the protected segment is
  capped (~80%) so a probation landing zone survives.
- **PREFILL never evicts a protected expert** — it overflows to transient instead.
  So WITHIN a request's prefill a wide multi-wave scan cannot scan out the seed.
- **Per-request re-warming** (served daemon, no reset between requests): the FIRST
  prefill after a decode route (a new request) **demotes** the prior request's
  protected set — `_protected` is cleared but `_pool_recency` (LRU order) is kept —
  and the seed budget under the flag is `capacity − len(_protected)` (the whole pool
  once demoted), not the zero empty slots a full pool would report. Without this the
  daemon warms only the FIRST request; request 2 (a different hot set) would start
  COLDER than the two-tier LRU.
- **W64/W71 pins are never victims** (unchanged).
- **Seed recency is stamped by frequency rank** (least-frequent seed admitted first =
  lowest recency), so the first decode eviction drops the least-frequent seed.
- Prefill advances only a **pool-local recency clock**, never `_decode_epoch` or the
  decayed-frequency `_history` — decode-frequency purity is preserved.

The result: after prefill the pool holds the prompt's top-`slots_per_layer` experts
(protected), so decode starts warm; decode 2Q then refines toward the live hot set.

---

## 3. Wave width is `transient_slots` on both paths (merged capacity retired)

The first cut widened the single-fence capacity to `slots_per_layer + transient_slots`
(a "merged pool" of 97). **That is unsafe and was retired** (review HIGH-1): a route
is only serviceable if the experts that cannot get a persistent slot fit the transient
scratch, and during prefill the protected/pinned slots are never evictable, so a wave
wider than `transient_slots` overflows the 48-slot scratch the moment any slot is
protected. Worked example on the real geometry (P=49, T=48): a second 97-wide sorted
wave with 4 protected seed → `97 − (49 − 4) = 52 > 48` experts spill → physically
unserviceable. And DSV4.1's top_k=6 verify is at most `8×6 = 48 = transient_slots`
unique, so the widening bought **nothing** for this model.

So the wave width — `route_waves` `max_unique_experts`, the `expert_mlx` verify
single-fence `_capacity`, and the bank's route-capacity check — is **`transient_slots`
on both paths** (byte-identical off). The single-pool win is the admission policy, not
wave width. Two defensive guards remain:

- `_validate_experts` rejects `unique > transient_slots` (as before, both paths).
- `plan()` raises a loud `ValueError` if a route ever spills more than
  `transient_slots` experts to the scratch — a tripwire that faults **at policy time**
  instead of deep in `ExpertSlotPool._physical` with an out-of-plan slot index.

`plan.batch_admission_slots` is `transient_slots` on both paths (kept for telemetry;
the widening term was removed from `plan_expert_memory`).

---

## 4. Exactness — residency never changes the math

A slot bank is a **pure cache**: a routed-MoE layer's output is `Σ gate·expert(x)`
over the router-selected experts, independent of which slot holds an expert, whether
it hit or missed, or the order it was admitted (seed-first reordering changes only the
slot map, which `resolved` keys by identity; the gather recombines by original
position). So single-pool vs two-tier is **byte-identical by construction**. Proven on
the real small component-bank runtime (CPU) in
`tests/test_deepseek_v41_single_slot_pool.py`:
`test_prefill_decode_byte_identical_on_vs_off` and
`test_dspark_verify_byte_identical_on_vs_off` (`mx.array_equal`).

---

## 5. The measured signal, and how to read it in the window

Pure-policy simulation at the real geometry (384 experts, top_k 6, 49 slots, ONE
layer-major route, `sort_unique` waves at `transient_slots`, `prepare_prefill_seed`,
Zipf prompt). **The shipped baseline is `cache_policy="frequency"`** (the
`ExpertStreamingConfig` default; the loader/bench never override it), so that is the
control below (LRU shown for contrast). **The single pool ignores `cache_policy`** —
it is a 2Q policy and never consults the frequency/LRU score.

| metric | two-tier `frequency` (shipped) | two-tier `lru` | single pool |
|---|---|---|---|
| overlap with top-49 (request 1) | 48 | 48 | **48** |
| decode hit, first-64 steps (request 1) | 0.570 | 0.477 | **0.584** |
| decode hit, steady/200 (request 1) | 0.568 | 0.445 | **0.571** |
| request 2, **different** hot set, no reset | **0.320** | ~0.32 | **0.585** |
| request 2, **same** hot set, no reset | 0.585 | — | 0.569 (48/49 retained) |

Reading it honestly: **vs the shipped `frequency` baseline the single-request gain is
small** — request-1 first-64 ~+1.4 pts, steady ~tied. **The durable, robust win is the
served MULTI-REQUEST case with a NEW hot set: 0.585 vs the frequency two-tier's 0.320**
— the two-tier persistent tier learned request 1's decode set and request 2's prefill
cannot repopulate it, so it starts cold, whereas the pool re-warms per request (the
demote + seed). Same-hot-set request 2 is a small trade (0.569 vs 0.585) because the
demote discards the decode-tuned set to re-seed from prefill frequency; the fix keeps
it from collapsing (was 0.355, 12/49 → now 0.569, 48/49). Against a plain LRU the 2Q
steady win is large (~0.57 vs ~0.45), which is what the reviewer's independent run
(0.550 vs 0.454) measured. **The window A/B (`cell16k_ring` vs `cell16k_ring_pool`)
must run in `--decode-mode ar`**: one decode step is one token / one layer sweep, the
basis the cold-start counter uses; a DSpark verify's accepted-tokens-per-cycle do not
map to layer sweeps (MED-4). Counter fields are labelled `..._first_64_steps`,
single-request.

Counters (`serve_stream_counters`): `expert_cache` gains `pool_loads` (misses admitted
to the pool), `scan_inserts` (of those, the prefill probationary inserts), and
`promotions` (seed-protected + decode-hit promotions). A new **`cold_start`** block
(always on — both paths populate the same fields) carries
`decode_hit_rate_first_64_steps` vs `decode_hit_rate_steady_state`, the raw
`first_64_steps_*` / `steady_*` counts, `decode_steps_observed`, and a
`measurement_basis` note (decode-STEPS, single-request, AR). The cold window
re-opens on the first PREFILL after decode, so each request is
measured fresh (MED-4). The `cell16k_ring_pool` DSpark cell also **cold-resets the
expert-streaming residency between the AR reference pass and the DSpark pass**
(recorded as `dspark.cold_reset_before_pass`) so a warm-from-AR bank does not confound
the pool A/B (HIGH-3).

---

## 6. Plan arithmetic — the 60 GiB profile (`deepseek-v41-mxfp4-75`)

`R` = one expert record = **18,800,640 B** (≈17.9 MiB). 40 streamed MoE layers.

| quantity | before (two-tier) | after (single pool) |
|---|---|---|
| resident slots / layer | 49 | 49 (**unchanged**) |
| persistent slots total | 1,960 | 1,960 (**unchanged**) |
| transient scratch (global) | 48 (0.84 GiB) | 48 physical; 0 as a policy tier |
| `persistent_cache_bytes` / `transient_bytes` / `fixed_bytes` | — | **identical** |
| single-fence wave width (`batch_admission_slots`) | 48 | **48** (widening retired) |
| decode start | cold | **warm** (prompt-frequency seed protected) |

Memory does not change; the flag alters only which loads happen. Verified in
`test_60gib_plan_arithmetic_and_allocation_neutral`.

### The physical fold (documented follow-up, unchanged from the first cut)
The physical two-region allocation is kept (1,960 per-layer + 48 global transient);
transient is ~1 slot/layer (48 ÷ 40). A physical fold (`transient_slots → 0`,
`+1 slot/layer`) reclaims `48·R` and spends `40·R`, giving `slots_per_layer 49 → 50`
at equal-or-less memory — **not** a one-liner (relaxes floors, `ExpertSlotPool` guards
and worker sizing `min(workers, transient_slots)`, `_physical` overflow, and the
W44/W81 pin-safety recycle), so it stays a follow-up. Its residency win (+1 slot/layer)
is small next to warming the whole pool, so it is low priority.

---

## 7. What the adversarial review changed (all flag-on; default-off stayed identical)

### Round 2 (MERGE WITH FIXES)
- **HIGH-1 (bench counters + cold reset were no-ops):** the DSV4.1 loader attaches a
  BARE `ExpertStreamingRuntime` (`snapshot()`/`reset()`) as `model._mtplx_expert_runtime`,
  not an `MTPLXRuntime` (`expert_streaming_snapshot()`/`.expert_streaming`), so
  `snapshot_stream_counters` swallowed an AttributeError → `serve_stream_counters` was
  None in every bench receipt (the W81 hit-rate/bytes counters never worked in the
  bench either), and the DSpark cold reset dereferenced a missing `.expert_streaming`
  → no-op. Fix: both resolve `es = getattr(rt, "expert_streaming", None) or rt` and use
  `es.snapshot()` / `es.reset()` (the served MTPLXRuntime path is preferred first, so
  it is unchanged). Test: a bare-runtime stub asserts the receipt gets `expert_cache`
  + `cold_start` and reset fires once.
- **HIGH-2 (warming was first-request-only on a served daemon):** `prepare_prefill_seed`
  budget was `capacity − occupancy` = 0 on request 2 (pool full), and protected was
  never demoted, so request 2 (a different hot set) started colder than two-tier (0.363
  vs 0.442). Fix: the first PREFILL after decode **demotes** the protected set (clears
  `_protected`, keeps recency) and the seed budget under the flag is
  `capacity − len(_protected)` — per-request re-warming. Request-2 first-64 now 0.546 ≥
  0.445 (§5). Tests: second-request warming ≥ two-tier, and a seed-first byte-identity
  test (top-8 = ids 24..31, `plan.misses[0]` in the seed, `mx.array_equal` on/off).
- **MEDIUM:** cold-start fields renamed `..._first_64_steps` (they count verify calls
  under DSpark), `measurement_basis` states single-request / decode-steps / AR.
- **LOW:** seed recency is stamped by frequency rank (least-frequent seed evicted first).

### Round 1 (DO NOT MERGE)
- **HIGH-1 (crash):** the 97-wide wave overflowed the 48-slot transient scratch under
  W64 pins or a lowered derived capacity → `ExpertSlotError` in `ExpertSlotPool`. Fix:
  retire the widening (§3) — wave width is `transient_slots`; add the policy-time
  overflow guard and clean `invalidate_expert`. `pool_admission_capacity` and the
  min-over-banks live bound were removed as no longer needed.
- **HIGH-2 (premise inverted):** on the real layer-major single sorted route,
  "promote on a later-wave hit" never fired and the pool held the 49 lowest expert
  IDs of the last wave, not the prompt tail (7/49 overlap). Fix: drive prefill
  promotion by the **frequency seed** (`_prefill_route_freq` / `prepare_prefill_seed`),
  admitted PROTECTED and **seed-first** within the wave → 49/49 overlap, first-64
  0.563 (§5).
- **HIGH-3 (2Q defeated by wave width; DSpark cell confounded):** a 97-wide wave
  evicted every non-hit resident incl. protected, and the DSpark cell re-prefilled a
  warm bank. Fix: **PREFILL never evicts protected** (overflow to transient), and the
  ab script cold-resets between the AR and DSpark passes.
- **MED-4:** cold-start counters now re-open per request (first PREFILL after decode),
  are labelled decode-STEPS with an AR-basis note, and the A/B runs in `--decode-mode ar`.
- **MED-5:** `invalidate_expert` now drops the expert from `_protected` and
  `_pool_recency` (was leaving stale pool bookkeeping).
- **MED-6:** the flag is gated on `cache_scope == "layer"` in `open()` (single-pool is
  a per-layer-bank policy); under global scope it is ignored with a warning, not a
  fault.

---

## 8. Changed files (behind the flag; default-off byte-identical)

- `mtplx/expert_streaming.py` — `LayerExpertSlotBank(single_pool=...)`: seed-first 2Q
  admission (`_pool_admit(protect, allow_protected)`, `_pool_victim_slot`,
  `_pool_touch`), prefill-never-evicts-protected, the overflow guard,
  `invalidate_expert` pool cleanup; `CacheCounters`/`RoutePlan` gain
  `pool_loads`/`scan_inserts`/`promotions`.
- `mtplx/expert_runtime.py` — reads the flag at `open()` gated on layer scope; threads
  it to the banks; `route_waves` uses `transient_slots`; always-on cold-start telemetry
  with per-request reset; snapshot exposure.
- `mtplx/expert_streaming_models.py` — `ExpertMemoryPlan.batch_admission_slots`
  (= `transient_slots`, allocation-neutral).
- `mtplx/models/expert_mlx.py` — verify `_capacity` reads the runtime's live bound
  (= `transient_slots`).
- `mtplx/serve_stream_counters.py` — the three cache keys + the `cold_start` delta.
- `scripts/deepseek_v41/ab_decode_env_levers.py` — env key + arm `cell16k_ring_pool`;
  DSpark cold-reset between passes.
- Tests: `tests/test_deepseek_v41_single_slot_pool.py` (rewritten for the seed design +
  the reviewer's overflow / sorted-wave-seeded / scan-resistance / gating tests) and
  the composite-arms row in `tests/test_deepseek_v41_ab_env_levers.py`.

---

## 9. Ledger notes (for the receipt log)

- **DSpark receipts from this commit on are COLD-BANK and not comparable to earlier
  DSpark cells.** The `cell16k_ring_pool` DSpark cell cold-resets the expert-streaming
  residency between the AR reference pass and the DSpark pass (HIGH-3;
  `dspark.cold_reset_before_pass` in the receipt), so its DSpark prefill starts cold
  rather than warmed by the AR pass. Earlier DSpark cells did not reset, so their
  decode numbers are not directly comparable to post-W87 DSpark receipts.
- **The prefill seed budget ignores W64 pins.** Under the flag the per-request seed
  budget is `capacity − len(_protected)`; it does not subtract `len(_pinned)`, so with
  a W64 pinned working set active the seed can nominally exceed the unpinned capacity
  (the excess simply overflows to transient, no crash). This is a non-issue for the
  `cell16k_ring_pool` arm, which does NOT enable `pin_working_set`; it matters only if
  the single-slot pool is ever stacked with the W64/W71 pin flags, and is left as a
  documented limitation rather than a fix.
