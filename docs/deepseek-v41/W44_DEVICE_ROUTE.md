# W44 — Barrier-free all-hit device route (KERNEL_LEDGER K24)

Status: design note + switch/runtime primitives, behind env `MTPLX_DSV41_DEVICE_ROUTE`
(default off), byte-identity proven on the fake bank. Author: Opus 4.8 worker
(`feat/deepseek-v41-w44`, off `feat/deepseek-v41-streaming` @ 5b6b8e7d2). CPU-only;
MLX pinned to CPU; no `experts.bin` load; ≤3 GB. mlx 0.32.2.

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

**(b) Token-level (the production forward).** The decode loop runs the optimistic
device-route forward with **0 routing barriers**, then at token end reads the probes.
If no layer missed, the logits are byte-identical (§2.1 composed over 40 layers) and
the sampled token stands. If any layer missed, the optimistic logits are void and
the token is **recomputed on the fenced path** (missing experts now admitted). The
recompute is authoritative → byte-identical.

Framing (b) needs a token-boundary hook in the decode loop (`mtplx/generation.py`) —
**outside the W44 allowlist** (`expert_mlx.py` / `expert_runtime.py`). W44 therefore
lands the **switch + runtime primitives** — the device LUT, the barrier-free issue,
the async probe queue, the `flush_device_route_probes()` verify, and the fenced
re-gather recovery — and proves them byte-identical at the switch level. Wiring the
token-end verify/recompute into the generation loop (framing (b), and the
resume-from-first-miss optimisation that gives "barriers only on miss layers" in the
backbone) is the follow-up (a `generation.py` change), specified here, not landed.

## 4. Barrier count per token (before → after)

| regime | fenced (today) | device-route (this lever) |
|---|---:|---:|
| all-hit token (warm / cross-prompt primed) | 40 | **0** routing barriers¹ |
| per layer, all-hit | 1 | **0** |
| per layer, miss | 1 | 1 (deferred; only miss layers) |
| token with *m* miss layers (per-layer recovery) | 40 | **m** (+ the token-end verify read, covered by the sampler eval) |

¹ the token-end sampler `mx.eval` exists in the fenced path too and is not a routing
barrier. **Target: barriers only on miss layers**, achieved by the per-layer
primitive (framing a). Before→after headline: **40 → m** barriers/token, `m` = miss
layers that token.

## 5. Cost when misses are frequent (honest)

Window-12 measured a per-expert-slot hit rate of **0.17**, but **1558/2560 layer-calls
were all-hit** (≈**0.61** of layer-calls). So within a token of 40 layers, ≈24 layers
are all-hit and ≈**16 miss** — device-route saves the 24 barriers and still pays ~16.
**40 → ~16 barriers/token** at cold window-12 rates: a real cut, but not the ~0 the
warm case gives.

The trap to document: under **token-level** recovery (framing b, naive), *any* miss
in the token voids the optimistic pass and forces a **full 40-barrier fenced
recompute** — and `P(all-hit token) = 0.61^40 ≈ 4e-9` at cold rates, so **almost
every token recomputes** and device-route is **pure overhead** (the optimistic pass
is wasted). Token-level recovery is therefore only a win when the working set is
**warm** (P(all-hit token) → 1: cross-prompt residency, or deep into a long decode
where the per-layer working set has settled — W24's within-prompt hit rate rises to
0.835 warm). The **per-layer** recovery (framing a / resume-from-first-miss) avoids
the all-or-nothing cliff — it pays exactly `m` barriers, monotonic in the miss count
— and is the form to wire into the backbone. **Default off; the GPU A/B (`device_route`
arm, window 16+) prices it against control at the standard shape.**

Interaction with K23 (switch fast-path): K23 removed the *second* per-layer sync (the
all-hit wave fence) by deferring the slot release; K24 removes the *first* (the
routing barrier itself) on all-hit layers. They compose — K24 subsumes K23's win on
all-hit layers (no barrier means no fence to defer) and is the larger lever.

## 6. What W44 lands

- `expert_runtime.py`: `device_route_lut(layer) -> mx.array` (cached int32 LUT,
  rebuilt only when `_lut_dirty[layer]` is set by an admission/eviction/reset);
  `enqueue_device_route_probe(...)` / `flush_device_route_probes() -> misses`;
  LUT-dirty marks at the cache-change sites.
- `expert_mlx.py`: `HotExpertSwitchGLU._run` device-route branch (env
  `MTPLX_DSV41_DEVICE_ROUTE=1`, decode + component-banks): barrier-free LUT gather +
  `async_eval(indices)` + probe enqueue, 0 host syncs; fenced re-gather recovery on a
  flagged miss.
- `scripts/deepseek_v41/ab_decode_env_levers.py`: `device_route` arm (nine keys now;
  every preset pins all nine).
- `tests/test_deepseek_v41_device_route.py`: byte-identity (all-hit / mixed /
  all-miss, M=1 and M=4), 0-host-sync-on-all-hit census, LUT-refresh-on-change,
  probe/verify correctness.

## 7. Follow-up (out of W44 allowlist)

Framing (b) in `mtplx/generation.py`: run the optimistic device-route forward, call
`flush_device_route_probes()` at token end, and on any miss admit + recompute the
token (ideally resume from the first miss layer, holding the residual stream at that
layer, to pay only `m` barriers rather than 40). This is where the token-level 40→0
(warm) / 40→m (cold) headline is actually realised end to end.
