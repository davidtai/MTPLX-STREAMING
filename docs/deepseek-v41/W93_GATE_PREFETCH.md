# W93 — DSV4.1 gate-oracle one-layer-ahead expert prefetch

**Job.** Turn W89's training-free gate oracle into a runtime lever: during the
decode forward of layer `L−1`, apply layer `L`'s OWN router to the residual
*entering* `L−1` (W89's `b'` predictor), take its top-`k` expert ids, and issue
those as **speculative SSD reads** for layer `L` into a bounded prefetch ring, so
that when `L`'s true route arrives its cold experts are already resident (hit) or
in flight (awaited, not re-read). The model still gathers on the **true** route,
so a mispredict only wastes a read — the output is byte-identical by
construction. Behind `MTPLX_DSV41_GATE_PREFETCH=<k>` (default off).

Author: Opus 4.8 worker (`w93/gate-oracle-prefetch`, off `1c4f72e37`).

Evidence read first: `../dsv41-w89/docs/deepseek-v41/W89_ROUTE_PREDICTOR.md` and
`../dsv41-w89/docs/deepseek-v41/receipts/w89-window35-eval/gate-oracle.md`.
W89 measured, on the real 16K trace (40 layers × 2,304 rows), the one-layer-ahead
gate oracle `gate_L(layer_in_{L-1})`: prec@6 **0.622**, missRed@10 **0.736**,
@12 0.766, @16 0.805; residual cosine `cos(layer_in_L, layer_in_{L-1})` mean
**0.921**; the first ~4 layers are the floor (L1 missRed@24 0.44, cos 0.766).

---

## 0. What is new here vs what already shipped

The prefetch **plumbing is already built** and driven by hy3/glm52
(`mtplx/models/lookahead_prefetch.py` → `runtime.prefetch_experts`). W93 reuses
it wholesale and adds only the DSV4.1-specific pieces:

* the **predictor** — `gate_L` applied to the residual *entering* `L−1`
  (pre-attention), top-`k` by the port's exact routing transform;
* the **eval piggyback** — the prediction rides layer `L−1`'s existing
  `mx.eval(indices)` routing barrier (item 2, §3);
* the **ring sizing** — the `deepseek-v41-mxfp4-75` profile ships
  `prefetch_slots=0`; the flag sizes a `k`-slot ring (§4);
* the **reconcile counters** — `hit_on_true_route` / `wasted` /
  `awaited_inflight` / `predicted` / `bytes_prefetched` (§6);
* the **await-inflight** guarantee on the demand route (item 3, §5).

Everything below is inert unless the flag is armed: with the flag off
`prefetch_slots` stays 0, no ring is allocated, and every touched path collapses
to its shipped expression (byte-identical, A/B-clean).

---

## 1. The predictor (engaged on the real path)

At the top of `DecoderLayer.__call__(h, pre_mix, …)` (`deepseek_v41.py`), `h` is
the residual **entering** this layer with the hyper-connection copies still
present: shape `[B, T, hc_mult, dim]`. This is exactly W89's `layer_in` before it
collapses the copies. The predictor:

1. collapses the hc copies EXACTLY as the trace collector's `layer_in`
   (`scripts/deepseek_v41/collect_route_traces.py:175`):
   `mx.mean(h.astype(float32), axis=2)` — the f32 upcast is applied **before** the
   mean, so the predicted route is bit-for-bit the tensor W89's gate oracle scored
   (missRed@10 0.736); collapsing in bf16 first would differ at the bf16 ULP on the
   real bf16 residual. The collector records `layer_in` = the **mean over the hc
   streams**, not the HC-combined MoE input (`router_in`) nor stream 0;
2. applies layer `L = L−1+1`'s **real** `Gate` via the port's exact
   `_gate_prefix_impl` (`deepseek_v41_moe.gate_predict_topk`): the bf16 score
   GEMM `/temp`, `sqrtsoftplus`, `+ e_score_correction_bias` — the identical
   scoring the shipped `Gate.__call__` runs (L810-822) — then `argpartition` for
   the **top-`k`** (`k` = prefetch width, 10 or 12; the shipped gate takes top-6,
   the prefetch takes the wider `k`). No `argsort` and no weight gather: prefetch
   needs the *set*, not the order or the weights.

This is computed **before `L−1`'s attention** (the program point where `h` is
available), as a lazy MLX array. It is never an ancestor of `L−1`'s output — it
reads only `h` (already fixed as the layer input) and `L`'s frozen gate weights —
so it does not perturb the true route in any way.

**Eligibility.** A prediction is formed only when: the flag is armed
(`k > 0`); it is single-token AR decode (`T == 1`); the bound switch has a
runtime with `prefetch_slots > 0`; the target `L` is a routed streamed layer with
a bound gate; `L ≥ MTPLX_DSV41_GATE_PREFETCH_MIN_LAYER` (default 4, W89's early
floor); and `L` is not the last routed layer (skip, per the brief). Otherwise the
layer stays on the pure demand path.

The predicted ids are stashed on `L−1`'s streamed switch
(`switch_mlp._mtplx_gate_prefetch_pending = (L, predicted_ids_array)`) for the
piggybacked eval below.

---

## 2. Correctness: exact by construction

The MoE gather in `MoE.__call__` reads `self.gate(moe_input)`'s **true**
`indices`; the prefetch prediction feeds only `runtime.prefetch_experts` (cache
warming) and is never read by any compute that produces a logit. Therefore:

* a correct prediction pre-warms a slot the true route then reads in place;
* a mis-prediction reads bytes into a ring slot the true route never consumes —
  a wasted read, evicted round-robin — and changes no output;
* the prediction's own matvec is a dead branch of the graph as far as the model
  output is concerned.

So logits/tokens are `mx.array_equal` with the flag on vs off, for AR and for a
DSpark verify (M=4) sequence with rejections. Proven in
`tests/test_deepseek_v41_w93_gate_prefetch_reconcile.py` (switch-level: the
gather output is bit-identical with the ring warmed vs cold, at M=1 and M=4,
all-hit / split / all-miss) and
`tests/models/test_deepseek_v41_w93_gate_prefetch_predict.py` (model-level: 64
decode steps byte-identical on/off).

---

## 3. Sync argument (item 2) — the prediction rides the switch's own barrier

**The design decision.** DSV4.1's fenced decode switch
(`HotExpertSwitchGLU._run`, `expert_mlx.py`) performs exactly one host sync per
layer — `mx.eval(indices)` (`hot.eval_indices`) — to read the true route to the
host so it can plan the SSD reads. The prediction for the **next** layer needs
its ids on the host for the same reason (to call `prefetch_experts`). Because the
prediction subgraph (`gate_L(mean(h_{L-1}))`) is **independent** of `indices`
(which depends on `L−1`'s attention + `L−1`'s gate), the two materialize together
in one device→host round-trip:

```
with _route_probe.bracket("hot.eval_indices"):
    mx.eval(indices, *pending_gate_prefetch)   # one call, two independent subgraphs
```

This adds **zero** new host syncs — the barrier already existed; we only widen
what it forces. The ids are then already resident, so
`prefetch_experts(L, ids)` reads them off the (already-synced) array with no
further sync. This is exactly the property `lookahead_prefetch.py` documents for
hy3 ("materialized together with the layer's own indices in the single sync the
streamed switch would perform anyway"), specialized to DSV4.1's pre-attention
residual source.

**Where it does NOT ride, it drops.** If the switch takes the barrier-free
device-route path (`MTPLX_DSV41_DEVICE_ROUTE[_PINNED]`, no `mx.eval(indices)`) or
it is not single-token decode, there is no barrier to piggyback and the pending
prediction is discarded unevaluated — gate-prefetch engages **only** where the
fenced barrier it rides already exists (the standard AR cell). Adding a barrier
where the device route removed one would violate item 2, so we never do.

Counted before/after on the fake runtime: `test_gate_prefetch_adds_no_host_sync`
asserts the generation-thread `mx.eval` count in `_run` is identical with a
pending prediction present vs absent (exactly one routing barrier per layer).

---

## 4. Ring sizing — one GLOBAL ring shared across layers

The `deepseek-v41-mxfp4-75` profile ships `prefetch_slots=0`. When
`MTPLX_DSV41_GATE_PREFETCH=k` is armed, `build_streaming_config` sets
`prefetch_slots=2·k` (clamped to `[1, 32]`) — the size of ONE ring **shared
across all routed layers** (`GlobalPrefetchRing`, `expert_streaming.py`), not a
per-layer ring. The config already requires `cache_scope="layer"` (the DSV4.1
default) and the component-banks layout the mxfp4 profile uses; the
mixed-official-bank MVP lane still rejects the ring (`expert_streaming_models.py`
guard).

**Why global, and why 2·k.** Decode runs the layers sequentially and prefetches
exactly one step ahead, so at any instant the ring holds at most two layers'
predictions: layer L's committed hits still resolving while layer L+1's `k`
predictions fill. `2·k` slots double-buffer that; sharing one ring across layers
means the resident ring is `2·k` records **total**, a small fixed reserve like
the transient scratch pool — not `k × n_layers` carved out of the persistent LRU
budget:

```
record (AR, mxfp4)              = 17.93 MiB
GLOBAL ring, 2·k = 20 (k=10)    = 20 × 17.93 MiB = 358.6 MiB  ≈ 0.36 GiB
GLOBAL ring, 2·k = 24 (k=12)    = 24 × 17.93 MiB = 430.3 MiB
GLOBAL ring, 2·k = 32 cap       = 32 × 17.93 MiB = 573.8 MiB
```

Contrast the naive per-layer ring: `k × R_routed × record = 10 × 40 × 17.93 MiB
= 7.0 GiB`, ~20% of the 60 GiB plan's expert cache, which would shrink the LRU
and its hit rate. The global ring is **~20× smaller** and comes out of the fixed
reserve (`ExpertMemoryPlan.prefetch_bytes = prefetch_ring_slots ×
expert_record_bytes`, subtracted alongside the transient scratch before the
persistent budget), so the LRU is untouched. Physically it is one shared
component bank of `2·k` buffers (`global-prefetch-{i}` labels), resolved by slot
index like the shared transient tier — never per-layer buffers. Entries are keyed
by `(layer, expert)`; round-robin replacement spans the whole ring, so layer
L+2's predictions recycle layer L's already-consumed slots. Only allocated when
the flag is armed (off → 0, shipped profile byte-identical). The speculative I/O
admission (`speculative_io_fraction=0.25`) still caps concurrent ring reads so a
prediction burst never queues ahead of a demand miss at the SSD.

On a true hit a committed ring entry resolves directly to its ring slot and the
gather reads it in place — no re-read, no byte copy into the persistent tier
(§5).

---

## 5. Pin / lock argument (item 3)

All four safety properties are held by the **existing** ring, plus one addition:

* **Honours pins / never evicts an earned resident.** `plan_prefetch`
  (`expert_streaming.py`) replaces round-robin **within the ring only**; a
  persistent/pinned resident is never a ring victim. Ring reads target ring slots
  (`base = persistent + transient + ring_index`), physically disjoint from the
  persistent and transient tiers, so a speculative fill can never overwrite a
  pinned or route-hit slot.
* **Never takes a lock a deferred split holds (the ~3827 hazard).**
  `prefetch_experts` acquires the layer lock **non-blocking**
  (`lock.acquire(blocking=False)`); under deferred split-route release the
  previous token's split holds that lock until the next covering flush, so
  speculation skips rather than waits (`expert_runtime.py:3827-3834`). The worker
  read (`_run_speculative_load`) uses only the slot state machine's own locks, not
  the runtime layer lock, so a demand route holding the layer lock cannot deadlock
  a worker.
* **In-flight read awaited, not duplicated, not overwritten (new).** On the
  **demand** route, `_reconcile_prefetch_for_route` runs under the layer lock the
  route already holds (`begin_split_route`, before planning): it (a) applies any
  settled ring completions so a just-finished read becomes a committed hit, then
  (b) for a requested expert still genuinely in flight in the ring, blocks on that
  read's future and commits it — so the true route reads the already-issued bytes
  instead of issuing a second demand read (`awaited_inflight++`). "Not
  overwritten" already holds (ring vs transient slots are disjoint). The await is
  a *demand* operation (the route needs that expert regardless) and is ≤ the cost
  of a fresh demand read, so it never adds latency vs today.
* **Promotable on a true hit without a copy.** A committed ring entry resolves
  through `bank.plan()` (`_prefetch_expert_to_slot`, `expert_streaming.py:748-763`)
  directly to its ring slot index; `begin_split_route` pins it as a hit and the
  gather reads it in place — no re-read, no byte copy into the persistent tier.

---

## 6. Counters (the `gate_prefetch` receipt block)

`resource_telemetry_snapshot()["gate_prefetch"]`:

| field | meaning |
|---|---|
| `predicted` | total predicted expert-ids handed to `prefetch_experts` (unique/layer/token) |
| `issued` | speculative reads actually started (`prefetch_issued`; after skipping resident/published/inflight/recent-miss/backlog) |
| `committed` | ring reads that settled and published (`prefetch_committed`) |
| `hit_on_true_route` | ring commits consumed by a true route as a hit (`RoutePlan.prefetch_hits`) |
| `wasted` | ring commits evicted round-robin without ever being hit |
| `awaited_inflight` | true-route experts awaited in flight instead of re-read |
| `bytes_prefetched` | `issued × expert_record_bytes` |
| `hit_rate` | `hit_on_true_route / max(1, committed)` |
| `per_layer` | `{layer: {predicted, issued, hit_on_true_route, wasted, awaited_inflight, hit_rate}}` |

Plus a one-line **census** string (`gate_prefetch.census`) for the ab receipt:
`gate_prefetch k=<k> min_layer=<m>: predicted=… issued=… hit=… (rate …) wasted=… awaited=… bytes=…MiB`.

---

## 7. Overlap arithmetic (why, and how much)

From W89 §3 (AR-16K, measured W82/W85): record 17.93 MiB, 61.9 misses/token →
`I = 1.084 GiB / 12.5 GiB/s = 86.7 ms`; the unfenced wall is 447 ms/token, so
`C = 360 ms` under the additive `T = C + I`. A one-layer-ahead prefetch overlaps
the fraction `r = missRed@k` of `I` with `L−1`'s compute:

```
T_overlap(r) ≈ C + (1 − r)·I          →  max(C, I)  as r → 1
```

| r = missRed@k | exposed I | token T | tok/s | Δ vs 447 ms |
|---:|---:|---:|---:|---:|
| 0.000 (today)      | 86.7 ms | 447.0 ms | 2.237 | — |
| **0.736** (k=10)   | 22.9 ms | 382.9 ms | 2.612 | **+16.7 %** |
| 0.766 (k=12)       | 20.3 ms | 380.3 ms | 2.629 | +17.5 % |
| 0.805 (k=16)       | 16.9 ms | 376.9 ms | 2.653 | +18.6 % |
| 1.000 (ceiling)    | 0.0 ms  | 360.3 ms | 2.776 | +24.1 % |

So the **expected hidden fraction is `0.736 × I ≈ 64 ms`** at k=10 (the brief's
`0.74 × I`), banking ~+17% of the +24% ceiling. Per W89 this is **second-order on
AR today** (I is only 19% of the wall — decode is compute-bound) and becomes
first-order once the compute stack (W80 ring + W71 barrier-free route) cuts `C`
toward the ~165 ms floor, at which point `C + I = 252 → max = 165 ms` (~34%). On
DSpark the prize is larger (`I ≈ 132 ms/accepted`, ~37% of the cycle). The
predictor's own cost is a 5120×384 matvec/layer ≈ 0.24 ms/token (<0.1% of the
wall), and it rides an eval that already existed — it never has to pay for itself
in compute, only prefetch correctly.

---

## 8. Deliverables

| artifact | path |
|---|---|
| predictor top-k helper (reuses `_gate_prefix_impl`) | `mtplx/models/deepseek_v41_moe.py::gate_predict_topk` |
| env resolvers + DecoderLayer prediction stash + ref install | `mtplx/models/deepseek_v41.py` |
| switch eval piggyback + prefetch call + ref wiring + shared-ring labels | `mtplx/models/expert_mlx.py` |
| `GlobalPrefetchRing` (shared ring) + bank delegation + counters + `prefetch_hits` | `mtplx/expert_streaming.py` |
| shared prefetch tier (physical) | `mtplx/expert_slots.py` |
| global `prefetch_bytes` accounting + `prefetch_ring_slots` | `mtplx/expert_streaming_models.py` |
| predicted/awaited/bytes counters, reconcile, ring sizing, receipt block | `mtplx/expert_runtime.py` |
| flag → `prefetch_slots` sizing | `mtplx/models/deepseek_v41_loader.py` |
| ab arms `gate_prefetch`, `cell16k_ring_prefetch` | `scripts/deepseek_v41/ab_decode_env_levers.py` |
| reconcile / exactness / sync / pin-lock tests | `tests/test_deepseek_v41_w93_gate_prefetch_reconcile.py` |
| predictor / offline-replay / model-exactness tests | `tests/models/test_deepseek_v41_w93_gate_prefetch_predict.py` |
| this design | `docs/deepseek-v41/W93_GATE_PREFETCH.md` |
