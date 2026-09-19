# W81 — DSpark verify routed-switch: why the single-barrier fast path declined, and the batched fix

Evidence receipt: `docs/deepseek-v41/receipts/gpu-windows/window-31/dspark-16k-cell16k.json`
(`dspark` block). Base: `c01a540c5`; rebased onto integration HEAD `6647e0003`
(W79 profile → 60 GiB, W80 window_ring/cell16k_ring arms).

Real-model measurement being explained: DeepSeek-V4.1-Flash streaming, 16,384-token
cell, DSpark depth 3, arm `cell16k`, 73 cycles, 3.52 tok/cycle, accept 0.916. Per
cycle: draft 184 ms, verify 1,423 ms. Verify `moe.routed_switch` = **733.8 ms/token**
(vs the AR 1-row census **112 ms/token** in the same process → 6.5×). `w61_engagement`:
`verify_single_barrier` 14, `all_hit` 1616, `begin_split_route` 10615 over ~2,920
layer-verifies (73×40) → the fast path took ~14 layer-verifies, `begin_split_route`
~3.6×/layer-verify.

---

## Task 1 — Why the fast paths did not engage (decline reason)

**Verdict: transient capacity, not pin state and not the env flag. The window-31
bench ran the verify switch with `plan.transient_slots = spec.top_k = 6`, far below
the ~20 unique experts of a 4-row verify, so the W66 single-admission gate
`len(unique) <= transient_slots` failed and every miss verify fell to the legacy
per-wave-fenced `route_waves` loop.**

Chain, from the code:

1. **The `_verify_single_barrier` gate PASSES** for the 4-row verify
   (`mtplx/models/expert_mlx.py`): env default `"1"` (arm_env recorded it `null` =
   unset = default on), `phase is DECODE` (the DSpark verify wraps the forward in
   `expert_routing_phase("decode")`; `current_expert_routing_phase` returns the
   explicit override, so the 4-row forward is DECODE not PREFILL —
   `expert_mlx.py:268`), `slot_layout == "component-banks"`, `codec == "mxfp4"`,
   `_shadow_bank is None`, `2 <= rows(4) <= 8`, `len(expert_ids) == rows*top_k`.
   So W61's all-hit probe runs every verify.

2. **`top_k = 6`** for `deepseek-v41-flash-expert-mxfp4`
   (`mtplx/expert_streaming_models.py:696,727`), so a 4-row verify routes
   `4*6 = 24` assignments with ~20 unique experts.

3. **W61 all-hit rarely fires.** `try_all_hit_route` returns non-None only when ALL
   ~20 unique are already persistent-resident; during verify (which explores
   experts off the accepted-draft path) that is rare → **14** engagements.

4. **W66 declines on capacity.** `_verify_can_defer` is True (proven by the 14
   `hot.allhit_defer` in the receipt; `defer_slot_release` + `flush_deferred_slot_releases`
   are both callable), and `_deferred_pin_active` is False (proven by 1,602
   `hot.allhit_fence_eval` — the 1-row AR all-hits fence because the config leaves
   `deferred_pin_release` false). So the only W66 blocker is the pre-fix gate
   `len(unique) <= _capacity` where `_capacity = plan.transient_slots`. In the bench
   that value was **6**: `20 > 6` → W66 declined on **every** miss verify
   (`hot.verify_single_barrier_split` = 0 in the receipt).

5. **Falls to the legacy loop.** `route_waves` partitions by
   `max_unique = plan.transient_slots = 6` (`expert_runtime.py:3248`), so ~20 unique
   → ceil(~20/6) ≈ 3–4 transient-bounded waves, each admitted with its own
   `begin_split_route` and fenced per wave part → `begin_split_route` ~3.6×/layer-verify,
   the 733.8 ms switch.

Why `plan.transient_slots` was 6, not the profile's 48: the window-31 launcher runs
the **CLI bench** (`scripts/deepseek_v41/ab_decode_env_levers.py` →
`load_deepseek_v41_streaming` → `build_streaming_config`), which never sets
`transient_slots`, so it defaults `None` and `plan_expert_memory` sets
`service_slots = spec.top_k = 6` (`expert_streaming_models.py:991`). The profile
`deepseek-v41-mxfp4-75` (`transient_slots: 48`) is applied ONLY by the served daemon
(`expert_cli.py` → `build_expert_streaming_config(profile, …)` →
`ExpertStreamingConfig(**profile.config)`, `expert_profiles.py:359`). **Bench and
served ran different slot plans** — the root of the whole discrepancy.

`pin_working_set` and `device_route_pinned` were both `enabled: false` in the run, so
W64/W71 are not factors.

---

## Task 2 — Where the 734 ms goes (host syncs / Metal submissions per layer-verify)

Measured on the fake component-bank runtime at the real shape (4 rows, top_k 6, ~20
unique, capacity 6, all-miss), counting blocking `mx.eval` and `begin_split_route`:

| path | blocking host syncs / layer-verify | admissions |
|---|---|---|
| **legacy** (flag off, bounded loop) | **21** | 4 |
| **W81 batched** (flag on) | **4** | 4 |
| ideal single admission (capacity ≥ unique) | **1** | 1 |
| 1-row AR all-hit (reference) | 2 | 0 |

The legacy path pays **~one blocking device→host fence per MISSING expert**, not per
wave: for a DECODE route each wave's misses are split one part per expert
(`begin_split_route` → `_miss_route_parts`, `expert_runtime.py:3108`) and the bounded
loop force-syncs every hit part and every miss part
(`evaluate_component_bindings(force_sync=True)`, `expert_mlx.py`). ~20 miss experts
→ ~20 fences + the 1 routing barrier = **21**, each a full round-trip that serialises
decode (no overlap into the next layer). That is the 6.5× over the 1-row switch: the
1-row route is all-hit (≈2 syncs) whereas the 4-row verify misses on ~20 experts and
fences each.

**Ideal** (one admission of all unique + one row-independent `gather_qmm`, deferred
release): exactly **1** blocking sync — the routing barrier — with the gather
async-submitted and released at the next barrier. The gather is 4× the rows of the
1-row switch but decode is host-sync/dispatch-bound at these sizes, so the wall
approaches the 1-row switch (112 ms/tok ≈ 2.8 ms/layer). Expected verify switch with
the fix + a capacity that fits: 733 ms/tok → order ~130–170 ms/tok (near AR 112).

---

## Task 3 — The fix (byte-identical batched single-fence)

`mtplx/models/expert_mlx.py`, in the `_verify_single_barrier` block: the W66 path
(admit whole route in one `begin_split_route`, gather deferred, one barrier) is kept
for `unique <= capacity`, and **generalised** so `unique > capacity` no longer falls
to the per-part-fenced loop:

- Partition the route into the SAME capacity-bounded waves `route_waves` produces
  (DECODE → `sort_unique` off, so the partition, the per-assignment gathers and the
  output-position recombination are identical to the bounded loop).
- Each wave = ONE `begin_split_route`; hits + all miss parts gathered **deferred**
  (`evaluate_component_bindings(defer=True)`, async-submitted) behind a **single**
  `synchronous_fence` — one fence per batch, never one per wave part.
- **Pin-safety (W44 / issue #120):** the layer lock is not reentrant
  (`begin_split_route` does `lock.acquire()`, released by `close()`,
  `expert_runtime.py:3081`) and non-final waves must recycle their transient slots,
  so each NON-final wave fences → `release_hits` + `release_miss` + `close` (freeing
  the lock) BEFORE the next wave re-enters. Only the FINAL wave defers its whole
  release via `_DeferredSplitClose` to the next routing barrier's covering eval
  (the W42/W61 deferred-pin proof — pins held until the eval, no unpinned recycle).
- Net: `(num_waves − 1)` blocking fences + the one routing barrier;
  `unique <= capacity` collapses to a single wave = the original W66 single-barrier.
- **Byte-identical**: only fence/release TIMING moves; the gather is row-independent
  (same token × same per-assignment expert weights, recombined by output position).
  Asserted flag-on vs flag-off for 1-miss/multi-miss/all-miss at M = 2/4/8.
- A runtime without the defer/flush seam (a fake double) leaves `_verify_can_defer`
  False and falls through to the bounded loop unchanged.

### Slot arithmetic (corrected) and the capacity lever

The earlier "48 transient slots × 40 layers = 33.6 GiB" premise is **wrong**. The
transient pool is a **single GLOBAL pool** of `plan.transient_slots` slots reused
one layer at a time (`expert_slots.py:830` `for slot_index in range(plan.transient_slots):
"global-transient-{i}"`), whereas the persistent LRU cache is per-layer
(`slots_per_layer × routed_layers`, `expert_slots.py:793`). Exact 60 GiB plan split
(`plan_expert_memory`, mxfp4 record = 17.93 MiB):

| transient_slots | persistent slots (per-layer) | persistent cache | transient pool (GLOBAL) |
|---|---|---|---|
| 6 (bench default = top_k) | 2000 (50/layer) | 35.02 GiB | 0.105 GiB |
| **24** (fits 4-row verify) | 2000 (50/layer) | 35.02 GiB | 0.420 GiB |
| 48 (current profile / served) | 1960 (49/layer) | 34.32 GiB | 0.840 GiB |

Plus resident dense weights 17.37 GiB + runtime reserve 7 GiB + KV(16384×3200B) ≈
0.05 GiB. At ts=48: 17.37 + 34.32 + 0.84 + 7 + 0.05 ≈ **59.6 GiB** ✓.

So raising transient_slots is nearly free (< 1 GiB), NOT 34 GiB. **transient_slots =
24** fits a 4-row (depth-3) verify's ≤24 unique in ONE admission AND keeps the
persistent LRU at the full 2000 slots — strictly better than 48 for decode (48
costs 40 persistent slots / 0.7 GiB for no depth-3 benefit; 48 is only needed for an
8-row/depth-7 verify's ≤48 unique). The batched path is the fallback whenever a
verify's unique still exceeds the capacity.

Recommendation: **the profile's current 48 already makes a depth-3 verify
single-admission** (24 ≤ 48), so the served path was never the problem. For a minor
decode-hit-rate gain, propose `transient_slots: 24` for the depth-3 16K cell (left
unchanged here at W79's 48 to preserve window-32 served comparability — a David-facing
call, not a silent worker edit). The bench now resolves this value from the profile,
so it follows whatever the profile ships.

---

## Task 4 — Tests (fake component-bank runtime, CPU-pinned, no GPU)

`tests/test_deepseek_v41_verify_switch_batched.py` (new, 12 tests, all pass):
- **byte-identity** flag-on (batched) vs flag-off (bounded loop) for 1-miss /
  multi-miss / all-miss 4-row routes with 20 unique at capacity 6, M = 2/4/8.
- **engagement ≥95%**: over 40 4-row verifies at capacity 6 the batched split path
  (`hot.verify_single_barrier_split`) fires on ≥95% (100% in practice); the W61
  all-hit + batched split together also ≥95%.
- **sync/admission counts**: batched = `1 + (waves−1)` blocking evals and `waves`
  admissions (ceil(20/6)=4 → 4 evals, 4 admissions); legacy strictly more.
- **receipt counters**: `hot.verify_single_barrier_split` (+`…_batched` when
  multi-wave) tick on the batched path.

Existing suites still green: `test_deepseek_v41_verify_single_barrier.py` (8),
`test_deepseek_v41_verify_single_barrier_split.py` (14),
`test_deepseek_v41_ab_env_levers.py` (65, incl. new composite-arm test).

Concrete before/after (fake runtime, 4-row/20-unique/cap-6 all-miss): legacy **21**
blocking evals → batched **4**; single-admission (cap ≥ unique) → **1**.

---

## Task 5 — Per-verify engagement census in the ab receipt

`ab_decode_env_levers.py` `w61_engagement` now carries a per-verify census derived
from new probe counters in `expert_mlx.py`:
`verify_candidates` (every 2..8-row DECODE component-bank route), `verify_engaged`
(all_hit W61 + split W66/W81), `verify_declined`, `verify_engaged_pct`, and
`verify_decline_reasons` (`flag_off` / `codec` / `shadow_bank` / `assignment_shape` /
`no_defer_seam` / `other`) — plus a console `[ab] verify engagement:` line. Verified
on the fake runtime: cap 6 → 10/10 engaged (batched); flag off → 10/10 declined
`flag_off`. The next window reads this to prove engagement directly.

---

## Coordinator slot analysis (window-32 served counters)

Served counters given: AR decode hit 0.741, DSpark decode hit 0.922, **transient_loads
= 0**, persistent_loads == misses, evictions == misses.

**Q1 — what are the transient slots doing during decode?** They are a small GLOBAL
prefill/streaming scratch pool (48 slots = **0.84 GiB**, not 33.6 GiB), idle during
decode. In DECODE every miss loads into a **persistent LRU** slot: decode route
planning takes an LRU victim in the persistent cache and emits
`SlotLoad(persistent=True)`, only spilling to transient when NO persistent victim
remains — i.e. when a single layer's route unique exceeds the persistent capacity
`slots_per_layer` (`expert_streaming.py:780-810`; slot selection
`expert_slots.py:1210` routes logical slots `< slots_per_layer` to persistent). A
decode route (AR 6 unique, verify ≤24 unique) is far below `slots_per_layer` (49–50),
so transient is never touched → `transient_loads = 0`, `persistent_loads == misses`.
`persistent_loads` counts loads into the persistent LRU pool; `transient_loads` counts
loads into the transient pool (`expert_streaming.py:145-146`, keyed by
`SlotLoad.persistent`). **There is no ~34 GiB to reclaim** — the "×40 layers" was a
misreading; transient is global. Exact 60 GiB split at ts=48: resident 17.37 +
persistent 34.32 (1960 slots) + transient 0.84 + KV ~0.05 + reserve 7.0 ≈ 59.6 GiB.

**Q2 — slot plan that maximises decode hit rate while keeping a single-submission
4-row verify:** since transient is ~free, the decode hit-rate lever is persistent LRU
capacity (`slots_per_layer`), not transient. `transient_slots = 24` fits the depth-3
verify's ≤24 unique in one admission AND keeps persistent at the full 2000 slots
(50/layer) — vs 48 which drops it to 1960 (49/layer). So **24 transient + the rest
persistent** maximises decode residency subject to the single-submission verify. The
gain over 48 is +40 persistent slots (+0.7 GiB LRU, ~1 slot/layer at 60 GiB) — a
small hit-rate lever, not the large one the 34 GiB premise implied. (The routing
census within-prompt hit-vs-capacity curve should be consulted to quantify the
0.741→? delta of +40 slots; it is small.)

**Served AR already runs with transient_slots = 48** (`build_expert_streaming_config`
applies `profile.config`), so the served DSpark verify (24 ≤ 48) is already
single-admission (`hot.verify_single_barrier_split` should fire on served); window
32's served numbers are read against a runtime that already had the fast path — the
bench, not served, was the outlier.

---

## Bench = served slot plan; serve_stream_counters in the receipt; composite arms

- **Slot-plan resolution** (`ab_decode_env_levers.py` + `bench_standard_shape.py`):
  new `--transient-slots` and `--expert-profile` (default `deepseek-v41-mxfp4-75`).
  Unset transient_slots now resolves from the profile (transient_slots,
  split_route_release, prefetch_slots) so the in-process bench runs the SAME plan as
  served; explicit `--transient-slots` overrides. Every receipt records
  `resolved_plan` (transient/persistent slot counts, bytes, split_route_release,
  source).
- **serve_stream_counters in the ab receipt**: DECODE-scoped (bracketed at the
  prefill→decode boundary, prefill excluded) expert hits/misses/bytes-per-token,
  persistent/transient loads and the slot plan, for the AR reference decode
  (top-level `serve_stream_counters`) and the DSpark decode
  (`dspark.serve_stream_counters`) — matching the served daemon's block so bench vs
  served is directly comparable, and giving David hit rate + bandwidth/token per run.
- **Two composite arms** (append-only, exact `cell16k_ring` key set + one lever
  group; covered by `test_cell16k_ring_composite_arms`):
  `cell16k_ring_draft` = cell16k_ring + K33 DSpark draft compile;
  `cell16k_ring_pinned` = cell16k_ring + W64 pin + W71 pinned device route. Both run
  at the profile's transient_slots by default (slot-plan resolution), so their verify
  switch is single-admission (≤24 unique ≤ 48).
