# W95 — DeepSeek-V4.1-Flash streaming decode runner: the TO-BE design (spec)

**Status:** DESIGN / SPEC. This is the runner the v2 path is built to (single switch
`MTPLX_DSV41_RUNNER=v2`, default = current path until the parity window passes). It is
not a flag-gated lever stacked on the current path — it is the per-token *structure* the
current path lacks, absorbing the W93 lanes (A config, B ring, C issue, D predictor;
all committed on `a5e162ee5`) as mechanisms rather than as env keys. Author: Opus 4.8
worker (`w95/barrier-free-resident`, off `1e6222811`, integration HEAD `cb47a2c35`).
CPU-only design authored without a GPU; MLX pinned to CPU; the real numbers below are
read from the cited windows/W-docs, not re-measured here.

**Provenance (measured):** windows 30/33/36/37 receipts; W82 (cost model), W85
(residency/compulsory-miss floor), W87 (single pool), W89 (route predictor), W92 (switch
census). Reference architecture: `/Users/davidtai/models/DeepSeek-V4.1-Flash-src/inference/model.py`.
Model facts (config): **40 routed-MoE layers, 384 routed experts, top-6, +1 resident
shared expert**, hidden 5120, `moe_intermediate` 2304, routed record **18.8 MB** (mxfp4
gs32, 3 weights); `n_mtp_layers`=3, `dspark_block_size`=**5**, `dspark_target_layer_ids`
=[37,38,39], draft MoE 128 experts top-3; served plan 49 slots/layer (60 GiB → 1,960
persistent) + 48 global transient.

---

## Headline

> **syncs per token: today 40+ → design 1 (unavoidable); expected tok/s: AR 2.2 → ~4–6
> (rebuild + ring + prefetch); DSpark depth-5 → 14–20 (the 20 tok/s goal sits at the
> W82 aggressive corner, not the center).**

- **The one unavoidable sync** is the token-boundary covering barrier (the sampler eval,
  which also carries the batched route read-back and the planner hand-off). Autoregression
  + streaming forces exactly one host round-trip per token: the sampled id must return to
  seed the next token, and the routes must reach the host to plan the next loads. The
  reference resident runner already runs at this floor (1 sync/token); the streaming runner
  today runs at 40+ because it blocks on `mx.eval(indices)` **per layer** to plan that
  layer's loads.
- **The design removes the per-layer routing barrier on every layer whose route is fully
  covered by the resident pool ∪ the prefetch ring** (all-resident + correctly-prefetched):
  those layers gather device-side over a slot LUT with **zero** host syncs, race-free by a
  token-scoped eviction epoch. A layer whose route touches an *un-prefetched* expert still
  pays one sync — but that sync is the SSD demand-load the layer had to do anyway (W92: the
  ~70% miss layers "block on SSD reads regardless of any fence"), not new drain.
- **What this cannot fix (honest, §7):** the compulsory-miss floor (novel experts the
  predictor cannot name a layer ahead — W85's ~19% temporal floor, worst on the first ~4
  layers) and the SSD wall (1.084 GiB/AR-token, 1.69 GiB/DSpark-accepted-token). The design
  turns the additive `T = compute + io` into `compute + (1−r)·io` and, as prefetch coverage
  `r → 1`, into `max(compute, io)` — it keeps the SSD *off* the critical path; it does not
  make the bytes free.

---

## 0. What is wrong today (diagnosis, measured)

AR-16K decode is **447 ms/token unfenced** (window 30, 2.238 tok/s); the windows 36/37
framing decomposes the ~460 ms token as **~240 ms kernel+SSD + ~220 ms drain/refill**.
The 220 ms is not compute — it is 40 per-layer routing barriers. Each MoE layer runs
`HotExpertSwitchGLU._run → mx.eval(indices)`, a blocking device→host round-trip so the host
can `.tolist()` the routed ids, check residency, plan SSD loads, and build the
`gather_qmm` `rhs_indices`. That barrier (a) drains the GPU pipeline and (b) stalls host
encode of the *next* layer while it waits. 40 layers → 40 serialized barriers, GPU idle
across each (W44 §0; W92 census: all-hit layer = 2 blocking `mx.eval`, miss layer = 3–5).

Three structural faults compound it:

1. **The route is planned on the generation thread, inline, per layer.** The `.tolist()`,
   the residency lookup, the `route_waves` partition, `begin_split_route`, the LRU touch,
   and the receipts all run between `mx.eval(indices)` and the gather — on the critical
   path, per layer.
2. **Residency is reactive, not predictive.** A miss is discovered *at* the layer and only
   then issued to SSD; the read cannot overlap earlier compute. W89 shows the io (87 ms
   flat-out) is 19% of the token and fully hideable, but today it is additive.
3. **Two failed attempts to remove the barrier prove the trap.** W44 (barrier-free device
   LUT gather, deferred, unpinned) decoded all-zero garbage on window 19: the deferred
   `gather_qmm` reads the bank at *eval* time, and an admission recycled a read slot in
   place before the eval → wrong weights. W71 (pin the slots so they cannot recycle) fixed
   the race but collapsed on window 34: pinning the whole resident set left only **0.25%**
   of routes all-pinned (a route's 6 experts are rarely all in the pinned set), so
   10,318/10,319 layer-calls "recovered" (double compute) → 1.04 tok/s **and** a wrong token
   stream (sha mismatch, never diagnosed). The lesson: **correctness cannot rest on pins,
   and the barrier-free set must be the whole resident set (30–51% of layers), not the
   pinned subset (0.25%).**

The design below removes the barrier without pins, makes the barrier-free set the whole
resident (+prefetched) set, and moves route planning off the generation thread.

---

## 1. The per-token structure, as it should be

### 1.1 Where the host MUST sync (exactly one, and why)

**Unavoidable syncs per token = 1.** Autoregressive streaming decode has one hard
serialization: the token sampled at step *t* is the input at step *t+1*, so the sampled id
must cross to the host once per token; and a *streaming* model must additionally learn the
step's routes on the host to plan the next step's SSD loads. Both needs are satisfied by a
**single covering barrier at the token boundary** — the sampler's `mx.eval(logits)` (or the
device-sampled id read) — which, because every layer's routed `indices` is an ancestor of
the logits, simultaneously materializes all 40 routes for the planner **in one
device→host round-trip, not 40**. This is precisely the sync count of the resident
reference runner. Everything else the current path does per layer is either moved off the
critical path (planning → the planner thread, §2) or eliminated (the barrier itself on
covered layers, §1.2).

**Why it cannot be zero (streaming).** A resident model could in principle chain several
tokens with device sampling and 0 host syncs. A *streaming* model cannot: the host must see
the routes to issue SSD reads for the misses the next token will need. So the floor is one
sync per token — the covering barrier that both reads the sampled id and feeds the routes to
the planner. (A future device-side load-scheduler could push even this off, but that is out
of scope and not assumed here.)

**Why it is not more (on covered layers).** See §1.2. **When it *is* more (on
un-prefetched-miss layers):** a layer whose route names an expert that is neither resident
nor in the prefetch ring cannot be gathered correctly without loading that expert, and the
load cannot be issued without the id — so that layer pays one demand sync. This is *the same
SSD stall the current path pays on every miss layer* (W92), now the only residual sync, and
prefetch (§1.3) drives its count toward the compulsory-miss floor.

### 1.2 How all-resident layers proceed with no sync (epoch + device LUT + exact fallback)

**Device slot LUT over ALL residents.** Each layer keeps a device array
`lut[expert] = slot` (int32 `[384]`, −1 = non-resident), built on the host **only when the
layer's resident set changes** — which now happens only at token boundaries (§1.4) plus the
planner's mid-token prefetch commits. This is the existing `device_route_lut` /
`device_route_snapshot`, but built over **the whole resident set** (persistent pool ∪
committed prefetch ring), not W71's pinned subset. The switch gathers device-side:

```
slot   = mx.take(lut, indices.reshape(-1))     # device gather, NO mx.eval(indices)
covered = (slot.min() >= 0)                    # device-side all-covered flag → per-token probe
routed = gather_qmm(x_bcast, bank.weights, rhs_indices=maximum(slot,0), transpose=True)
```

`indices` is never `.tolist()`'d on this path; the host races ahead to the next layer while
the GPU pipelines the gather. **0 host syncs on a covered layer.**

**Token-scoped eviction epoch — the correctness primitive that replaces pins.** A decode
token opens an *eviction epoch*. Invariant, enforced in the bank's victim selection:

> **Within one decode token, no slot that the device LUT can reference is recycled.**

Concretely: a mid-token miss admitted by the planner goes into a **free** slot (a slot the
current LUT maps to no expert) — a free persistent slot if any, else a prefetch-ring /
transient slot — **never** by evicting a resident the LUT already points at. LRU
eviction/promotion of the token's churn is deferred to the token boundary (§1.4), where the
covering `mx.eval` has already evaluated every deferred gather of the token. Therefore a
deferred device gather issued at layer L and evaluated at the boundary reads the *same* slot
bytes it referenced at issue — **the W44 recycle race is impossible by construction**, with
no pins, and the barrier-free set is the whole resident set (W71's collapse is gone).

**Exact fallback on a miss.** A layer whose route names an expert with `lut[e] = −1` (not
resident, not yet prefetched) cannot be served correctly barrier-free — the missing weights
are simply not on the device. The device `covered` flag folds into a small per-token probe
array; at the covering barrier the planner reads it (`flush_device_route_probes`, one
batched eval) and the layer is repaired on the **fenced path** (the current, byte-identical
`try_all_hit_route`/`begin_split_route`: demand-admit the miss experts into free slots under
the same epoch, gather with real weights). Two framings, both exact:

- **Prefetched-covered layer:** the miss was loaded a layer early (§1.3) into a free slot
  and is in the LUT before the gather is *built*, so the gather is correct and barrier-free
  — no fallback, no sync.
- **Un-prefetched-miss layer:** repaired on the fenced path. Because a layer's MoE output
  feeds the residual stream (reference `Block.forward`: `x = hc_post(ffn(x), …)`), a wrong
  MoE output corrupts everything downstream — so the fallback must run the miss layer's MoE
  **before** the next layer consumes its residual. The v2 runner does this by **issuing the
  demand load and the fenced gather for that layer inline** (the one demand sync), keeping
  the residual exact layer-for-layer. It does **not** speculate-then-recover the whole token
  (W44/W71's re-forward): re-forward is 2× compute and, at today's 30% all-hit, the first
  miss is at layer ~1, so re-forward is ~full and a net loss (this is exactly what sank W71,
  1.04 tok/s). **Inline fenced repair on the miss layer is the exact, bounded fallback;
  whole-token recovery is retired from the v2 path** (kept only behind the current path for
  the shelved device-route arms).

### 1.3 How layer L+1's reads are issued during layer L (predictor riding the one sync)

The read for a miss cannot start until the host knows the expert id; the true route for L is
known only after L's attention. The **gate-oracle predictor** (W89 §2.1; lane D) breaks the
serialization: it runs **layer L+1's own gate** on layer L's collapsed hidden `h`
(`gate_predict_topk(next_gate, mean_hc(h), k)`) — the residual barely turns per layer
(cos 0.921), so the previous layer's hidden ranks L+1's true top-6 at **prec@6 0.622**, and
a width-`k` prefetch covers **missRed@10 0.736 / @12 0.766** of L+1's true experts (30/39
deep layers clear the 0.70 overlap threshold; the first ~4 layers are the residual-turnover
floor and stay on demand, `min_layer`). The prediction is a pure read of the compiled K22
gate tape (`a'=1.0` vs the router in both regimes) and costs ~0.24 ms/token (<0.1%).

**Riding the one sync (no extra barrier).** The predicted ids for L+1 are *stashed on L+1's
switch* during L's forward (`DecoderLayer._maybe_stash_gate_prefetch` → `switch.
_mtplx_gate_prefetch_pending = (next_layer, predicted)`); they are evaluated on the switch's
*own* covering eval, never on a fresh barrier. The planner issues them as **speculative
reads into the global prefetch ring** (lane B `GlobalPrefetchRing.plan_prefetch(layer, ids,
resident=…, is_slot_pinned=…)`): lower SSD-queue priority than demand reads, the
unconditional **target−1** victim rule, **pinned/resident-victim refusal**, and eviction
attributed to the victim's layer. Lane C's issue point (after `begin_split_route(L)`, a
fire-once closure) and speculative-concurrency cap (default 4) with a 2 s reconcile timeout
+ demand fallback are the *issue mechanics*; v2 keeps them but folds them into the planner
thread rather than the switch return path. **The ring is sized globally** (lane A: env-
authoritative sizing in the loader + served builder, one `prefetch_slots` budget shared
across layers), so a burst of deep-layer misses draws from one pool, not 40 per-layer pools.

**Correctness of prefetch:** a prefetched expert that the true route does **not** use is a
wasted read (evicted at lowest priority); a true-route expert the prefetch **missed** is a
demand read (identical to today). The gather always uses the **true** `indices`. So prefetch
only warms the cache — **it never changes a result** (§4).

### 1.4 Eviction / promotion only at token boundaries

Under the epoch (§1.2), no LRU eviction or protected-promotion happens *during* a token.
At the covering barrier the planner, off-thread, applies the whole token's residency delta
at once: promote decode-hit experts (2Q, W87), evict the LRU tail down to capacity, commit
the surviving prefetch-ring entries into the persistent pool, drop wasted speculative reads,
and rebuild any dirtied `lut[layer]`. This is the only point the resident set (hence the
LUT) changes, which is why the barrier-free deferred gathers of the *next* token are exact
against a stable snapshot. The bank's decode-frequency history (`_decode_epoch`, W87) ticks
once per token here, not per layer.

### 1.5 DSpark verify (M = K+1 rows, depth 5) on the same structure

DSpark drafts `K = min(speculative_depth, block_size=5)` tokens (markov rollout over the
block; confidence early-stop) and **verifies them in one target forward over the `1+K` rows
`[primary, d1..dK]`** (`deepseek_v41_dspark_decode.py`). The verify is the *same* 40-layer
backbone the AR path runs, with `M = K+1` rows instead of 1 — so it uses the *same* per-token
structure:

- **One covering barrier per verify** (not per layer): the verify's logits eval carries all
  40 routes of the `M`-row batch to the planner.
- **The route is the union of the M rows' top-6** (up to `U(K+1)` distinct experts/layer —
  independence estimate U(4)=23.6 at depth 3, U(6)=34.9 at depth 5; the served plan's 48
  transient slots hold the union). The device LUT + epoch cover it exactly as for M=1; a
  covered layer is barrier-free, an uncovered-miss layer is inline-fenced.
- **Depth 5 is free on the draft side** (W82 §1: block_size=5 is the trained config; the
  markov recurrence, not a 4th/5th transformer block, produces d4/d5). It widens the verify
  *union* (more SSD bytes, §5) but not the *structure*.
- **Accept / rollback reuse the existing cache lanes** (`LayerAttentionCache.mark/rollback/
  trim`, `trim_verified_window_to_prefix`): the verify appends `K+1` rows; keep the committed
  prefix `[primary, d1..da]`, trim the `K−a` rejected tail. The eviction epoch spans the
  **whole verify** (open at draft, close at commit), so a rejected-tail rollback cannot have
  recycled a slot a kept row's gather referenced — the same by-construction safety as AR.

---

## 2. What leaves the generation thread → the planner thread

Today the generation thread does route planning inline, per layer, on the critical path.
v2 splits the work:

- **Generation thread (hot):** builds the forward graph and issues gathers only. Per layer:
  attention → gate → `async_eval(indices)` (non-blocking submit) → device-LUT gather (§1.2)
  → next layer. It touches no residency map, runs no `route_waves`, no `.tolist()`, no
  receipts. One covering `mx.eval` per token.
- **Planner thread (cold, off critical path):** consumes the `async_eval`'d routes as they
  complete, and does **route planning** (residency lookup, miss set, 2Q promote/evict
  decisions — applied at the boundary, §1.4), **load scheduling** (issue demand reads for
  un-prefetched misses; issue speculative reads for the predictor's next-layer ids; drive the
  global ring's completion drain, lane B), and **counters** (all telemetry, §6). It hands the
  generation thread back only two things it needs: the updated device LUTs (rebuilt at the
  boundary) and, for an un-prefetched-miss layer, the signal to take the inline fenced path.

**The one cross-thread rule that keeps it deadlock-free.** The current deferred-release
machinery holds a layer's `threading.Lock` across the covering flush, and
`_apply_derived_allowance` (KV-boundary allowance) takes every layer lock — a self-deadlock
latent on the DSpark lane today (W92 CRITICAL fix). v2's planner never holds a layer lock
across a generation-thread dependency: it drains its own submitted work
(`flush_deferred_slot_releases(evaluate=True)`) before any all-layer lock acquisition, and
the ring drain (lane B) acquires layer locks **non-blocking** and skips on contention
(counter `skipped_lock_held`). The generation thread never blocks on the planner except at
the covering barrier (where it is supposed to).

---

## 3. What is removed from the production path

| removed | why | replaced by |
|---|---|---|
| **Per-layer `mx.eval(indices)` routing barrier** (on covered layers) | the 220 ms drain | device-LUT gather + one covering barrier |
| **The second all-hit wave fence** (`hot.allhit_fence_eval`) | redundant sync (W92) | deferred release under the epoch (no fence needed — the covering eval covers it) |
| **`route_waves` / `begin_split_route` split machinery** on covered layers | per-layer host partition + per-part fences | one grouped `gather_qmm` over the LUT slots (already the minimal 3 dispatches, W92); split path kept only for the inline fenced fallback |
| **Per-stage probe brackets** (`_route_probe.bracket` per stage) in the hot path | host bookkeeping per stage | cheap atomic counters only (§6), read at the boundary |
| **Per-layer Python bookkeeping** (`.tolist()`, residency touch, receipts) on the gen thread | critical-path host work | moved to the planner thread (§2) |
| **The transient-vs-persistent tier split** | prefill-scan-resistance tier that leaves decode cold (W87 §1) | **single scan-resistant pool** (W87 2Q + prefill frequency seed; merged at HEAD) — one `slots_per_layer` pool per layer, prefetch ring drawn from the same budget |
| **Pins (W64 working-set / W71 device-route-pinned)** | correctness-via-pins is fragile and collapsed the barrier-free set to 0.25% (W71/window 34) | the token-scoped eviction epoch (§1.2) — race-freedom by scheduling, not by pinning |

The current path (all of the above) stays intact and **default**; v2 is the single
`MTPLX_DSV41_RUNNER=v2` switch. No stacked env keys — the W93 lane keys
(`gate_prefetch`, ring sizing, issue, predictor) are subsumed into the one switch.

---

## 4. Exactness argument (per part)

**The invariant:** a routed-MoE layer's output is
`y = Σ_{e∈top6(x)} weight_e(x)·expert_e(x) + shared(x)` (reference `MoE.forward`). It depends
only on (a) the **true route** `(weights, indices)` — a pure function of the layer's
post-attention hidden `x` via the gate — and (b) each routed expert's **actual weights**. It
is **independent of which slot holds an expert, whether it hit or missed, and the order of
admission** (the slot map is a pure indirection the gather resolves by identity; W87 §4).
So the runner is byte-identical to the fenced reference **iff every routed expert's real
weights are gathered against the true indices.** Each v2 part preserves exactly this:

- **Device-LUT gather (covered layer):** `lut[e]` is by construction the same
  `bank_index` the fenced `try_all_hit_route` would use (both read `_expert_to_slot`), and
  every routed `e` is resident (covered), so `maximum(slot,0)==slot` (no sentinel) and the
  gather runs the identical kernel over the identical rows → **bit-for-bit** the fenced
  output. The only thing skipped is the host round-trip, which produces no array (W44 §2.1,
  CPU-proven byte-identical).
- **Token-scoped epoch:** does not touch math — it only constrains *when* a slot may be
  recycled. It guarantees the deferred gather reads the weights that were resident at issue
  (no in-place recycle), which is the precondition the invariant needs and W44 §8 lacked.
  **The W44 race is impossible by construction:** no LUT-referenced slot is recycled within
  the token, so there is no interval in which a deferred gather can read a recycled slot.
- **Prefetch / predictor:** the gather uses the **true** `indices`, never the predicted ones;
  a mispredict evicts a wasted read, a missed prediction becomes a demand read. Output is
  independent of prefetch (W89 §4). Byte-identity holds regardless of coverage `r`.
- **Inline fenced fallback (uncovered-miss layer):** *is* the current fenced path (demand-
  admit + gather with real weights), run before the next layer reads the residual → exact
  layer-for-layer, hence exact token.
- **DSpark verify:** the accept/reject/rollback math is unchanged (standard speculative
  decoding over the M rows); residency/epoch only affect *how* the M-row union is gathered,
  not the logits. The epoch spanning the whole verify makes the kept-prefix gathers exact
  after a rejected-tail rollback.
- **Single pool vs two-tier:** byte-identical (W87 §4, `mx.array_equal` on/off) — residency
  policy changes which loads happen, never the result.

**Test-enforced (§6):** v2-on decodes are `mx.array_equal` to v2-off (the current runner)
over 256 AR tokens with adversarial hit/miss/overflow mixes, and over DSpark
accept/reject/rollback sequences.

---

## 5. Cost model (measured anchors; expected after the rebuild)

**Anchors (measured).** AR-16K unfenced **447 ms/token** (window 30) ≈ **240 ms
kernel+SSD + 220 ms drain** (windows 36/37). Decomposed (W82 §2, unfenced, per token):
attention **277 ms** (mem-pressure, not O(T) — W73/W76; owned by the **W80 window ring**,
already in the base `cell16k_ring` arm), switch **119 ms** (routing barriers; owned by this
rebuild), gate/shared/combine/Sinkhorn/HC/head ~110 ms. SSD io **I = 1.084 GiB / 12.5 GiB/s
= 87 ms** flat-out (61.9 misses/token × 17.93 MiB, hit 0.741); realized single-stream today
**2.1–5.6 GB/s** → 190–516 ms if issued serially (SSD device ceiling 13.42 GB/s; AR uses only
**19%**). 1K forward floor **~165 ms** unfenced. Drain per sync ≈ 220/40 ≈ **5.5 ms**.

**Drain removed = 220 × (barrier-free fraction b).** `b` = fraction of layers whose full
top-6 is covered (resident, or correctly prefetched). io is overlapped by prefetch:
`I → (1−r)·I` (W89 overlap model). The two scenarios the coordinator asked for:

| scenario | all-resident a | prefetch | b (covered) | drain kept 220·(1−b) | io kept | **T (ms)** | **AR tok/s** |
|---|---:|---|---:|---:|---:|---:|---:|
| today (26% miss) | 0.30 | none | 0.30 | 154 | 87 | **394** | **2.54** (+17%) |
| today (26% miss) | 0.30 | r≈0.65 @k12 | ~0.57 | 95 | ~30 | **~278** | **~3.6** |
| 80 GiB (15% miss) | 0.51 | none | 0.51 | 108 | ~54 | **~315** | **~3.2** |
| 80 GiB (15% miss) | 0.51 | r≈0.75 | ~0.80 | 44 | ~14 | **~211** | **~4.7** |
| ceiling (b→1, ring→~165 attn, io hidden) | — | r→1 | 1.0 | 0 | 0 | **~165** | **~6.0** |

(kernel floor held at ~153 ms + the ring-recovered attention; `b` from
`a + (1−a)·r^{avg misses/miss-layer≈2.2}`; approximate, ±.) **Reading:** removing drain alone
(no prefetch) is a modest +17–47% because AR-16K is compute-bound and the miss layers keep
their SSD stall. **The prefetch + the ring are what turn it into a real win** — they raise
`b` (fewer syncs) *and* hide io. The AR ceiling after the full rebuild is **~6 tok/s** (the
1K-forward floor), consistent with W82.

**DSpark depth 5.** L(d) = expected tokens/cycle: **L(3)=3.55** (measured accept
[0.9315,0.9118,0.9032]), **L(5)=4.83** (extrapolated a4≈0.889/a5≈0.875 — **UNMEASURED, a
gate**). Cycle = draft + verify(M=d+1 rows) + commit. With the rebuild (barrier-free verify +
ring-recovered attention + within-cycle dedup R2), W82 §5:

| corner | verify base+per-row | cycle d3 | tok/s d3 | cycle d5 | **tok/s d5** |
|---|---|---:|---:|---:|---:|
| central | 140 + 15R | 200 ms | 17.8 | 230 ms | **~15.4** |
| aggressive | 120 + 8R | 152 ms | 23.4 | 168 ms | **~20** |

**So the 20 tok/s target is DSpark depth-5 at the aggressive corner** (barrier-free verify
+ ring + R2 dedup keeping per-row ≤8 ms + d4/d5 acceptance holding + verify hit ≥0.955). The
central estimate is **14–16 tok/s** (W82's 13–16). The runner rebuild is the **precondition**
for all of it (it does nothing while the verify costs 1.4 s), but it is not sufficient alone.

**SSD wall (co-binding, §7).** DSpark streams **1.69 GiB/accepted token** (94 misses, hit
0.922). SSD binds at `BW / 1.69`: device ceiling → 7.94 tok/s, realized ~10.7 GB/s → ~6.4.
So the SSD is a **second wall at ~6–8 tok/s** that bites only after the compute floor clears
it; 20 tok/s needs the accepted-token byte cost cut 2.5× — **verify hit ≥ 0.955–0.97** (up
from 0.922: fewer misses via residency + R2 within-cycle dedup, 0.71 unique → ~29% request
cut). AR hits its SSD wall only at **11.5 tok/s**, so AR stays compute-bound with headroom.

---

## 6. Implementation plan — ONE integrated change

### 6.1 Module boundaries (kept vs replaced)

**Kept as-is (imported, not touched):**
- **Loader + bank I/O:** `deepseek_v41_loader.py`, `MlxComponentBank` / `MappedExpertStore`
  / `_gather_component_bank` (the mxfp4 gs32 gather kernels — already the minimal 3
  `gather_qmm` per layer, W92). v2 gathers through the same kernels.
- **Cache lanes:** `deepseek_v41_cache.py` (`LayerAttentionCache.mark/rollback/trim/advance`,
  `DeepseekV41Cache`) — reused verbatim for DSpark rollback and any fenced repair.
- **Attention (CSA2), gate math, HC/Sinkhorn, engram, DSpark draft head** — unchanged; the
  rebuild is only the MoE routed-switch execution + residency + threading.
- **The W93 lane mechanisms as building blocks:** the global prefetch ring (lane B), the
  gate-oracle predictor + `_GatePrefetchLink` stash (lane D), the issue closure + concurrency
  cap + reconcile timeout (lane C), env-authoritative ring sizing (lane A). v2 *calls* these;
  it does not re-implement them and does not expose their env keys.

**Replaced (the v2 path, behind `MTPLX_DSV41_RUNNER=v2`):**
- `models/expert_mlx.py` `HotExpertSwitchGLU._run` — the env-key branch forest
  (fastpath/submit/device-route/pinned/shared-hoist) collapses to two paths: **covered →
  device-LUT gather (no sync)**; **uncovered-miss → inline fenced repair**. The all-hit wave
  fence and the per-stage probe brackets are gone (counters only).
- `expert_runtime.py` — the token-scoped eviction epoch (`begin_token_epoch` /
  `end_token_epoch`, victim selection refuses LUT-referenced slots mid-token; counters
  `mid_token_evictions_avoided`, `transient_overflow_fallbacks`); the LUT built over the
  whole resident set; the planner-thread handoff; retire the pins path from v2.
- `expert_streaming.py` — the single-pool 2Q admission (W87, drop pins from the v2 path); the
  bank victim honors the epoch.
- `models/deepseek_v41.py` — `_forward_span` runs the v2 loop (gen-thread gathers + one
  covering barrier); the whole-token `_device_route_recover` is **not** used by v2 (inline
  repair instead); DSpark verify uses the same span with `M=K+1`.
- **The planner thread** — a new small module (`expert_planner.py` or a runtime member): a
  worker fed the `async_eval`'d routes, doing planning/scheduling/counters (§2).

### 6.2 Test plan (CPU-only, fake component-bank runtime + tiny model, `nice -n 19`, ≤1.5 GB)

1. **Byte-identity vs the current runner over 256 AR tokens** (`mx.array_equal` on logits +
   sampled ids), on the fake runtime + tiny model, across adversarial residency mixes:
   all-hit, mixed hit/miss, all-miss, and **prefetch-ring/transient overflow** (more
   mid-token misses than free slots → forced inline fenced fallback). This is the correctness
   gate — v2 must equal v1 exactly.
2. **DSpark accept/reject/rollback byte-identity:** M=K+1 at depth 3 and 5, full-accept /
   partial-accept / full-reject, logits + committed prefix + cache state (`_buf`/`_len`) +
   engram equal to the fenced verify.
3. **The W44 race is impossible by construction:** the CPU test that recycles a read slot
   mid-token under a 20-expert churn must show the epoch **refuses** the recycle
   (`mid_token_evictions_avoided` ≥ 1) and the deferred gather is isolated — the exact
   scenario that returned |Δ|≈3e4 garbage in W44 now returns byte-identical.
4. **No-sync count per token asserted = the design's N:** wrap the module-level blocking
   `mx.eval` (the W92 method) and assert a fully-covered token does **1** covering sync
   (0 per-layer barriers), a token with *m* uncovered-miss layers does `m+1`, and the covered
   fraction matches the LUT coverage.
5. **Planner-thread invariants:** the deadlock regression (deferred route → KV-boundary
   allowance off-thread < 5 s, W92); the ring drain acquires locks non-blocking; counters are
   monotonic and match the route trace.

### 6.3 Real-model parity harness (GPU window, `MTPLX_GPU_PARITY=1`; run by David, not this worker)

`test_gpu_parity_runner_v2_vs_fenced`: decode N tokens on the real artifact, running **each
layer both ways** (v2 device-LUT gather vs the fenced reference gather) and comparing
per-layer — report the **first divergent (layer, slot, expert)** with the routed-index diff,
plus the **token-id sha256 vs a reference receipt** (the ring/paired reference). This is the
instrument W71's window-34 failure lacked (its wrong stream was never diagnosed). Gate: the
harness must be clean (0 divergent layers, sha match) before v2 is promoted from default-off.

### 6.4 Window plan (paired reference vs v2 on the standard cell)

The standard cell (16,384-token real prefill, `--memory-limit-gib 60`, `--max-kv 17408`,
window-28b prompt ids, `--prompt-seed 20260829`, `--decode-mode ar`). Arms:
`runner_v2` (= `cell16k_ring` + `MTPLX_DSV41_RUNNER=v2`) vs the paired `cell16k_ring`
reference (the window-37 `ar-ring-ref` shape), and `cell16k_ring_v2` for the DSpark cell.
Report control-vs-v2 prefill tok/s, decode tok/s, peak GB, wall, TTFT, deltas, whether
token ids are sha-identical, and the receipt counters below. Also the 80 GiB plan
(`--memory-limit-gib 80`) to measure `b` at 51% all-resident, and a depth-5 DSpark cell to
gate a4/a5 acceptance.

### 6.5 Receipt counters (v2 block)

`syncs_per_token` (the headline N), `barrier_free_layers` / `fenced_layers` /
`inline_repairs`, `covered_fraction`; `prefetch_issued` / `prefetch_hit` / `prefetch_wasted`
(lane B/C: `speculative_bytes_read`, `demand_bytes_read`, `dropped_no_slot`,
`skipped_lock_held`, `skipped_backlog`); `mid_token_evictions_avoided`,
`transient_overflow_fallbacks`; `planner_thread_wait_s` (gen-thread time blocked on the
planner — should be ~0 except at the boundary); decode hit rate first-64 vs steady (W87
`cold_start`). One flush = one batched `mx.eval` (assert the "+1").

### 6.6 Integration/sequencing note (must resolve before coding)

The four W93 lanes (A/B/C/D) are committed on `a5e162ee5` — a **rewrite** of
`expert_streaming.py` (±820), `deepseek_v41.py` (−603), `expert_runtime.py`, `expert_mlx.py`
that the coordinator directs v2 to absorb. The build base named for v2 is `cb47a2c35`
(feat/streaming HEAD, W91 relabel) — which is **pre-rewrite**. Building v2 on `cb47a2c35`
while the lanes rewrite the same files guarantees a large merge conflict and re-implements
the lanes' committed work. **The clean base is the merge of the four lanes onto one
integration point**, and v2 builds on that (calling their ring/predictor/issue/sizing). This
is the one decision to confirm before implementation, because it determines whether v2 calls
the lanes' code or duplicates it.

---

## 7. What this design cannot fix (honest)

- **The compulsory-miss floor.** A genuinely novel expert (one the gate oracle cannot name a
  layer ahead) must be demand-loaded, costing one sync + one SSD read. W85 put the temporal
  floor at ~19%; the gate oracle lifts covered layers to missRed@12 0.766 on the deep layers
  but the **first ~4 layers stay on demand** (residual turnover, L1 missRed@24 only 0.44). So
  N never reaches a hard 1 in practice — it reaches `1 + (compulsory-miss layers)`, ~5–12
  today, ~3–6 at 80 GiB, → 1 only as residency + prediction → full coverage.
- **The SSD wall.** The design overlaps and hides io; it does not shrink it. AR streams
  1.084 GiB/token, DSpark 1.69 GiB/accepted token. The SSD co-binds DSpark at ~6–8 tok/s and
  AR at 11.5. Past that, only **fewer misses** (higher residency: the 80 GiB plan, R3
  frequency residency), **fewer bytes** (R2 within-cycle dedup, 0.71 unique), or **more
  realized bandwidth** (concurrent issue toward the 13.42 GB/s ceiling, vs today's 19% AR
  utilization) move the wall. 20 tok/s DSpark needs verify hit ≥0.955–0.97; this design
  enables that (residency + dedup) but does not deliver it by itself.
- **20 tok/s is the optimistic edge, not the center.** W82's honest read — central 13–16
  tok/s (depth 5, full compute stack), 20 only when every lever lands at its good end. The
  runner rebuild is the precondition that unlocks the rest; it is not the whole 20.
