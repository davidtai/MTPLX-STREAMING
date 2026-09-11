# W44 — Barrier-free all-hit device route (KERNEL_LEDGER K24)

Status: **implemented end to end** (switch + runtime + decode-forward cold recovery),
behind env `MTPLX_DSV41_DEVICE_ROUTE` (default off), byte-identity — output, cache,
and engram — proven vs the fenced path on the tiny backbone with a controllable fake
bank. Author: Opus 4.8 worker (`feat/deepseek-v41-w44`, off `feat/deepseek-v41-
streaming` @ 5b6b8e7d2). CPU-only; MLX pinned to CPU; no `experts.bin` load; ≤3 GB.
mlx 0.32.2.

## 0. The cost this removes

W37/W44 stage timing (window 15, `head_bf16`): `moe.routed_switch` is still the
top decode stage at **~108 ms/token fenced** (~90 ms net of the ~0.43 ms probe
fence × 40 layers), i.e. **~2.2 ms per layer** on an **all-hit** gather that moves
**113 MB** from the resident component bank — ~0.2 ms at 600 GB/s. The gather is not
the cost. The cost is the **per-layer routing barrier**: every layer's streamed
switch runs `mx.eval(indices)` (`HotExpertSwitchGLU._run`), a blocking device→host
round-trip that (a) drains the GPU and (b) stalls host encode of the **next** layer
while it waits. 40 layers → **40 barriers/token**, serialised, GPU idle across each.

Why the barrier exists today: the host needs the concrete expert ids (`.tolist()`
the routed `indices`) to (1) check residency (hit vs SSD miss), (2) build the
per-assignment slot indices (`binding.buffer.bank_index`) that `gather_qmm`'s
`rhs_indices` consumes. Both are **host-side** uses of the ids, so the ids must
cross to the host → the barrier.

## 1. Mechanism — a device-side expert→slot LUT

Neither host use is fundamental if the **expert→slot map lives on the device**.

`LayerExpertSlotBank._expert_to_slot: dict[int,int]` already maps every resident
expert to its persistent component-bank row (the same row `binding.buffer.bank_index`
names). Snapshot it into a per-layer device array:

```
lut[layer]  : mx.array int32, shape [n_routed_experts]   (384 for DSV4.1)
lut[layer][e] = slot(e)  if e resident,  else  -1 (sentinel)
```

The LUT is **built on the host only when the cache contents change** (an admission
or eviction for that layer flips a `_lut_dirty[layer]` flag; the next
`device_route_lut(layer)` rebuilds and re-`mx.array`s it, otherwise returns the
cached array). Between cache changes — which for a warm decode layer is *most
tokens* — it is a constant device array, reused with zero host work.

The all-hit gather then issues **entirely on-device, without evaluating `indices`
on the host**:

```
slot_idx = mx.take(lut, indices.reshape(-1))     # device gather, no host sync
safe     = mx.maximum(slot_idx, 0)               # sentinel -1 -> row 0 (valid mem)
routed   = gather_qmm(x_bcast, bank.weights, rhs_indices=safe.reshape(-1,1), …)
```

`indices` is never `mx.eval`'d or `.tolist()`'d on this path. The host does **not**
block on the gate compute; it races ahead to encode the next layer while the GPU
pipelines the gather. **0 host syncs on an all-hit layer.**

## 2. Exactness

### 2.1 All-hit layer (every routed expert resident): byte-identical, unconditionally

When every expert in `indices` is resident, `lut[e] = slot(e)` is exactly the
`bank_index` the fenced `try_all_hit_route` would pin for `e` (both read the same
`_expert_to_slot`). So `slot_idx` equals the fenced path's `slot_indices` **value
for value**, `safe == slot_idx` (no sentinel), and `gather_qmm` runs the identical
kernel over the identical rows with the identical `x`. The routed output is
**bit-for-bit** the fenced output. The only thing skipped is the *host round trip*,
which produces no array — pure scheduling. (Test: `all_hit` byte-identity + 0-sync.)

### 2.2 Miss layer: the optimistic output is wrong → must be recovered

If any expert `e*` in `indices` is not resident, `lut[e*] = -1`, `safe` clamps it
to row 0, and the gather reads the **wrong expert's** weights for that assignment.
The missing expert's weights are simply **not on the device**, so no device-only
computation can produce the correct output — recovery requires **admitting** `e*`
(an SSD read into a slot) and re-running the gather with real weights, i.e. the
**fenced path**. Detecting that `e*` missed requires the host to see the ids, i.e.
a barrier — but that barrier lands **only on miss layers**, and it is deferred (§3),
not paid up-front on every layer.

### 2.3 Verification is deferred, not skipped

The host bookkeeping the fenced barrier used to do inline — residency check, LRU
touch, receipts — is moved **off** the layer's critical path: the switch does
`mx.async_eval(indices)` (a non-blocking submit) and **enqueues a probe**
`(layer, indices, resident_snapshot)`. The ids are read back **later** — at the
next layer's entry, or at token end (where the sampler's `mx.eval(logits)` has
already evaluated `indices` as an ancestor of the routed output, so the read is a
`.tolist()` of an already-computed array, **not a fresh GPU drain**). The probe
compares each id against the snapshot the LUT was built from; any id outside it is
a miss for that layer.

## 3. Recovery — final token byte-identical in all cases

Two equivalent framings; both make the **final token** identical to the fenced path:

**(a) Per-layer (the primitive W44 lands and tests).** Each layer's routed output
is either (i) an all-hit LUT gather — already byte-identical (§2.1) — or (ii) a miss,
in which case the enqueued probe flags it and the layer is **re-gathered on the
fenced path** (admit the miss experts, `try_all_hit`/`begin_split_route`, gather with
real weights) — byte-identical (it *is* the fenced path). Reconciling a token =
{all-hit layers kept, miss layers replaced by their fenced re-gather} is byte-
identical layer for layer, hence the token is identical. The CPU tests hold each
layer's `x` fixed and prove `reconcile(device_route) == fenced` for all-hit / mixed
/ all-miss sequences at M=1 and M=4.

**(b) Span-level, LANDED in the decode forward.** `DeepseekV41Backbone._forward_span`
(the decode M=1 and verify M=K+1 path) runs the optimistic device-route span with
**0 routing barriers**, snapshotting each layer's pre-token cache length (`mark`) and
the span's initial `(h, pre_mix)`. At span end it calls `flush_device_route_probes()`.
If no layer missed, the span output/cache/engram are byte-identical (§2.1 composed)
and stand. If layers missed, `_device_route_recover` **rewinds every layer's cache to
pre-token** (the existing length-based `LayerAttentionCache.rollback` — window /
compressed / index / compressor frontier; the engram is untouched because its offset
delta is zero while `offset` is still pre-advance) and **re-runs the whole span from
the saved initial state on a fresh `shared` runtime**, forcing ONLY the miss layers
onto the fenced path (barrier + admit + gather → correct + admitted) while every other
layer stays on the device path (0 barriers). It loops until a pass has no misses
(bounded; the final fallback forces the fenced path for every layer), then `cache.
advance(s)` runs once. Result: **span output, per-layer KV/compress/index cache, and
engram state all byte-identical to the fenced path, paying exactly `m` = (miss layers)
routing barriers** (asserted for all-hit / single-miss / multi-miss / all-miss at M=1
and M=4 in `tests/test_deepseek_v41_device_route_recovery.py`).

**Why re-run from layer 0, not from the first miss `m1`.** The preferred per-layer
scheme re-runs only `m1..n-1`, but DSV4.1's CSA layer menu threads state across layers
through the `shared` runtime — the compressor/index **source** layers
(`kv_source_layer_ids`, `index_source_layer_ids`) and the **candidate source**
(`candidate_source_layer_id`) feed later reindex/reuse layers. A correct partial
restart from `m1` on a fresh `shared` would have to back up to the earliest such
source that feeds any layer `≥ m1` — which for this model's menu sits near the top of
the stack — so it saves little compute over a full re-run while adding real fragility.
Re-running the whole span from layer 0 on a fresh `shared` is trivially correct and
pays the **same `m` barriers** (only miss layers are fenced; the re-run of the
already-correct prefix is all device → 0 extra barriers). Its cost is **one extra
span of compute on a cold (miss) token** — priced in §5.

## 4. Barrier count per token (before → after)

| regime | fenced (today) | device-route (this lever) |
|---|---:|---:|
| per layer, all-hit | 1 | **0** |
| per layer, miss (fenced on the recovery pass) | 1 | 1 (only miss layers) |
| span-end verify (one batched `mx.eval` of all probed indices) | 0 | **1** / token |
| **all-hit token (warm / cross-prompt primed)** | **40** | **1** (verify only) |
| **token with *m* miss layers** | **40** | **m + 1** |

The per-layer routing barriers drop **40 → m** (only the miss layers are fenced, on
the recovery pass); the single span-end verify read is the "+1" (`flush_device_route_
probes` batches all probed indices into **one** device→host sync — never one per
layer, which would reintroduce the ~40). **Warm token: 40 → 1.** The counting test
asserts the `m` recovery-fenced routing barriers (`hot.eval_indices`); the batched
verify sync is the constant "+1".

## 5. Cost when misses are frequent (honest)

Window-12 measured a per-expert-slot hit rate of **0.17**, but **1558/2560 layer-calls
were all-hit** (≈**0.61** of layer-calls). So within a token of 40 layers, ≈24 layers
are all-hit and ≈**16 miss** — device-route saves the 24 barriers and still pays ~16.
**40 → ~16 barriers/token** at cold window-12 rates: a real cut, but not the ~0 the
warm case gives.

The landed recovery pays **`m` barriers, monotonic in the miss count** — never the
all-or-nothing 40 of a naive whole-token-fenced recompute. But it does re-run the
span once on a cold token, so the honest cost model is:

| | host syncs / token | decode-forward compute / token |
|---|---:|---:|
| fenced (control) | 40 | 1× |
| device route, **warm** (all-hit token) | **1** (verify only) | 1× |
| device route, **cold** (`m` miss layers) | **`m` + 1** | **2×** (pass-1 + one recovery span) |

So the lever is an unambiguous win **warm** (0 barriers, 1× compute) — the target
regime: cross-prompt residency, or deep into a long decode where the per-layer working
set has settled (W24's within-prompt hit rate rises to 0.835 warm → few miss layers).
**Cold**, it trades 40 barriers for `m` barriers **plus** one extra span of compute;
whether that nets faster depends on the barrier-stall vs decode-compute ratio at the
window-12 rate (≈16 miss layers/token) — **the GPU A/B (`device_route` / `stack_a`
arms, window 16+) is authoritative.** If cold nets negative, the follow-up is the
partial `m1`-restart (§7) to cut the 2× compute toward 1× + tail. **Default off.**

Interaction with K23 (switch fast-path): K23 removed the *second* per-layer sync (the
all-hit wave fence) by deferring the slot release; K24 removes the *first* (the
routing barrier itself) on all-hit layers. They compose — K24 subsumes K23's win on
all-hit layers (no barrier means no fence to defer) and is the larger lever.

## 6. What W44 lands (end to end)

- `expert_runtime.py`: `device_route_lut(layer) -> mx.array` (cached int32 LUT,
  rebuilt only when `_lut_dirty[layer]` is set by an admission/eviction/reset);
  `device_route_snapshot`; `enqueue_device_route_probe` / `flush_device_route_probes()
  -> misses`; `set_device_route_force_fenced(layers)` (recovery override); the
  per-layer component-bank capture; LUT-dirty marks at the cache-change sites.
- `expert_mlx.py`: `HotExpertSwitchGLU._run_device_route` + the `_run` branch (env
  `MTPLX_DSV41_DEVICE_ROUTE=1`, decode + component-banks): barrier-free LUT gather +
  `async_eval(indices)` + probe enqueue, 0 host syncs; skipped for layers the backbone
  has forced onto the fenced path this recovery pass.
- `deepseek_v41.py` (`DeepseekV41Backbone`): `_forward_span` marks + saves + the
  span-end `_device_route_recover` (rollback-all + fresh-`shared` re-run with the miss
  layers forced fenced); `cache.advance` moved after recovery. `_device_route_active`
  gates it to a DECODE span with a streamed runtime present.
- `deepseek_v41_cache.py`: **no change** — the existing length-based
  `LayerAttentionCache.rollback` / `mark` already rewind window/compress/index/frontier
  and leave the engram untouched (offset delta zero pre-advance).
- `scripts/deepseek_v41/ab_decode_env_levers.py`: `device_route` arm (nine keys now,
  every preset pins all nine) + `device_route` folded into `stack_a`.
- Tests: `tests/test_deepseek_v41_device_route.py` (switch-level: all-hit byte-identity
  + 0-sync + probe/verify + LUT refresh) and `tests/test_deepseek_v41_device_route_
  recovery.py` (backbone end to end: output + cache + engram byte-identical to fenced
  and barriers = miss layers, across all-hit / single-miss / multi-miss / all-miss at
  M=1 and M=4).

## 7. Follow-up (optional optimisation)

Partial `m1`-restart to cut the cold-token 2× compute toward 1× + tail: re-run only
`min(m1, source_floor)..n-1`, where `source_floor` is the earliest CSA source
(`kv_source_layer_ids` / `index_source_layer_ids` / `candidate_source_layer_id`) that
feeds any layer `≥ m1`, reconstructing exactly that prefix of the `shared` runtime.
Barriers are already `m` without it, so this is a compute optimisation for the cold
regime, priced only if the window-16 A/B shows cold device-route compute-bound.
