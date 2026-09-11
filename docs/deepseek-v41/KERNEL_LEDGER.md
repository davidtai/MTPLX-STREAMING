# DeepSeek-V4.1-Flash — Kernel / byte / fusion Ledger (GPU-compute side)

Status: living document (worker feat/deepseek-v41-w27). Analysis only; no code lands from this file.
Author: Opus 4.8 worker. CPU-only static analysis of the ported forward
(`mtplx/models/deepseek_v41.py`, `deepseek_v41_moe.py`, `expert_mlx.py`); no model runs, no GPU.

**This file is the compute-side companion to `OPTIMIZATION_LEDGER.md`.** That ledger prices every
lever against the **SSD wall** (13.42 GB/s, the decode-binding constraint *today*). This file prices
the **DRAM / dispatch / host-sync wall** (614 GB/s M5 Max, plus per-dispatch host-encode and the
per-layer routing barrier) — the floor that is **hidden under the SSD wall today and becomes the
binding constraint the moment the streaming levers (OPT_LEDGER R1–R5) cut SSD bytes**. Its §1 cost
model is the pricing basis; this file **extends** it with the per-component GPU accounting David asked
for and does **not** restate the SSD arithmetic. Where a lever is already ranked there (R1, R2, R5,
R6, R7), this file adds only its GPU-visible cost/gain and cross-references — it does not re-rank it.

David's ask (verbatim): "what about kernel and byte optimizations and fusions, you should create a
list of candidates and rank them by potential improvements before you start."

---

## 0. The headline (read this first)

**After the SSD wall falls, DSV4.1 decode is host-sync / dispatch-bound, not DRAM-bandwidth-bound.**

Three independent GPU-side floors, per accepted token, measured against the actual ported forward:

| Floor | Value | Ceiling it imposes |
|---|---|---|
| Resident-weight DRAM read (mxfp8, once/token) | ~9.4 GiB = **10.1 GB** → 16.5 ms @ 614 GB/s | ~60 tok/s |
| + routed experts if DRAM-resident (240 recs, 4.51 GB) | 24 ms total DRAM | ~42 tok/s |
| **Per-layer routing barrier** `mx.eval(indices)` × 40 | **~1 ms p50 × 40 = ~40 ms** (measured, `expert_mlx.py:1959`) | **~25 tok/s** |
| Dispatch host-encode (~6.8 k dispatches, Sinkhorn-dominated — cf. V4's 6,794) | ~20 ms naive × 2.9 µs, **production-discounted ~45 % → ~9 ms** (V4 note) | folds into the chain; K3 kills most |

The routing barrier is a **serial** chain (layer N's route needs layer N−1's output) with the GPU
idle during each round-trip, so it does **not** overlap the DRAM read — the exposed decode floor is
`barrier(40) + non-overlapped compute ≈ 40–50 ms/token ≈ 20–25 tok/s`. **That sits exactly on the
20 tok/s target.** Consequence: **the SSD levers alone cannot reach 20 tok/s** — even with infinite
SSD bandwidth, the 40 ms barrier chain caps AR decode near 20. The GPU-side levers below (barrier
reduction, Sinkhorn kernel, verify-batched dispatch) are **on the critical path to 20, co-equal with
the SSD levers**, not the "behind-the-wall +2–5 %" refinement OPT_LEDGER R6 correctly calls them
*at today's SSD-bound state*. This file's whole job is to price the wall behind the wall.

**⚠ One measured caution up front (governs K1/K14):** on A3B, removing a per-cycle decision sync was
**+3.27 % *slower*** and the `MLX_MAX_MB_PER_BUFFER`/`MAX_OPS` env sweep was **≈dead** (200 → −0.9 %,
2000 → +9 % worse, promotion rejected) because MLX's async-submission backpressure (10 outstanding
buffers × 50 ops / 50 MB force-commit) **conserves the block regardless of the Python sync** — the
cv-wait is genuine GPU-dependency time, not recoverable by cadence changes
([[a3b-decode-roundtrip-is-the-lever]]). DSV4.1's barrier differs in one load-bearing way — it is a
genuine device→host **route materialization** (`.tolist()` the ids to issue an SSD/DRAM gather), not
a bare decision drain — so its host-side portion *is* real work with a real GPU-idle window to fill.
But the a3b result forbids assuming the fill is free: every K1/K14 arm counts `mx.eval` and is
measured against the backpressure floor, not credited on the roofline.

**And V4's "road to 40" finding, which this file inherits:** on V4 (experts resident), once the
Sinkhorn kernel + fused-CSA/HC-tape stack landed, decode was **no longer** attention- or
dispatch-bound — the residual binding term was the **MoE gather/dequant GPU-forward** itself, and
"2-bit is ALU-bound, hand kernels lose there" ([[deepseek-v4-kernel-verdicts]],
[[iq2xxs-kernel-loses-to-stock]]). DSV4.1's routed forward is the **native mxfp4** gather (K7): so
after K1/K3/K4 exhaust the barrier and dispatch, the mxfp4 `gather_qmm` efficiency (K7) is the residual
decode lever — via the *stock* kernel, never a hand one.

MTP (OPT_LEDGER R1) is the multiplier on this side too: one K+1 verify forward runs the 40 barriers
**once** and covers ~2.85 accepted tokens → **~14 ms/accepted-token of barrier**, not 40. Verify is
GPU-cheaper per accepted token than 4 AR steps for the same reason it is SSD-cheaper (amortization) —
see K10.

Everything below RANKS levers; it does not predict absolute e2e (the hy3 roofline overshot the whole
by 42–71 % summing parts, [[hy3-decode-roofline]]). Every candidate carries a measured-A/B gate (§7).

---

## 1. Constants and method (extends OPT_LEDGER §1, does not duplicate it)

| Constant | Value | Source |
|---|---|---|
| DRAM bandwidth (M5 Max) | **614 GB/s** (12.47 GiB/s SSD is the *other* wall) | [[test-machines-bandwidth-file]] |
| Host-encode per dispatch | ~2.9 µs (exposed, B=1) | [[deepseek-v4-kernel-verdicts]] / docs/perf/deepseek-v4-dispatch-levers.md |
| Per-streamed-layer routing barrier | **~1 ms p50** (`mx.eval(indices)` + `.tolist()`) | `expert_mlx.py:1959`, measured |
| Resident dense (mxfp8, text-only) | 10.65 GiB (W18) | OPT_LEDGER §1.1 |
| Routed record (mxfp4 gs32) | 18.80 MB (w1+w2+w3) | OPT_LEDGER §1.1 |
| Layers / experts / top-k | 40 text / 384 / 6 (+1 shared) | PORT_PLAN §0 |
| hidden / head_dim / heads / KV heads | 5120 / 512 / 64 / 1 | PORT_PLAN §0 |
| q_lora / o_lora / o_groups | 1280 / 1024 / 8 | PORT_PLAN §0 |
| moe_intermediate | 2304 | PORT_PLAN §0 |
| vocab | 129280 | config |
| hc_mult / Sinkhorn iters | 4 / 20 | config |
| index_n_heads / index_head_dim / index_topk | 32 / 128 / 512 | config |
| window | 128 | config |
| kv_source (Full) layers | [2, 8, 14, 20] (4) | PORT_PLAN §0 |
| index_source layers | [2,8,14,20,24,28,32,36] (8) | PORT_PLAN §0 |
| engram layers | 1, 14 | PORT_PLAN §1 |

**Method guardrails carried from OPT_LEDGER §7 (they gate every A/B here):** paired in-window control;
**count `mx.eval` per arm** ([[moe-exec-fusion-25-26-27]]); queued-not-eager microbench
([[queued-vs-eager-metal-microbench]]); `gather_qmm` x must be `[rows,1,K]→[rows,1,N]` or it silently
does 8× work and fakes a win ([[gather-qmm-calling-convention-trap]]); rank on measured e2e, never on
a component roofline; IOV_MAX/preadv slicing on any union gather.

---

## 2. Per-component GPU cost model — DECODE

### 2.1 Per token, AR (M=1)

DRAM @ 614 GB/s. "BW-time" = bytes/614 if bandwidth-bound; at M=1 the quantized matmuls are
**ALU/dispatch-bound, not BW-bound** ([[metal-sub4bit-alu-bound]]), so BW-time is a *floor* the real
kernel misses by the ALU factor noted.

| Component | DRAM bytes/token | BW-time floor | Dispatch / sync | Notes |
|---|---:|---:|---|---|
| **Resident dense read** (mxfp8, ex-embed) | 9.42 GiB = 10.11 GB | **16.5 ms** | ~600–900 disp | hard floor; only byte-cuts move it (K8/K9) |
| — head GEMV (bf16, 129280×5120) | 1.323 GB | 2.15 ms | 1 | **biggest single read**; q8/mxfp8 → 1.08 ms (K9) |
| — embed | ~10 KB (gather 1 row) | ~0 | 1 | *not* a full read — the 10.65 GiB "floor" counts it, real read is ~9.4 GiB |
| — attn wq_b/wo_b/wo_a/wkv/wq_a (×40) | ~5.0 GB | 8.1 ms | ~5/layer | M=1 GEMVs |
| — shared expert w1/w2/w3 (×40) | 1.50 GB | 2.4 ms | 3/layer | every token, every layer |
| — gate/hc/compressor/indexer/norms | ~0.8 GB | 1.3 ms | many small | hc_fn is [24,20480]×2/layer |
| **Routed experts** (240 recs = 6×40) | 4.51 GB | 7.35 ms *(if DRAM-resident)* | 40 gather_qmm | M=1 ALU-bound ×1.3–1.75 → **9.5–12.9 ms** (K7, unmeasured) |
| — **today**: ~77 % miss → SSD | ~3.0–3.5 GB **off SSD** | **224–260 ms** @13.42 | — | THE WALL — OPT_LEDGER §1.3, not re-priced here |
| **Per-layer routing barrier** | — | **40 × ~1 ms = ~40 ms** | 40 host syncs | `mx.eval(indices)`+`.tolist()`; serial; exposed after SSD falls (K1) |
| Attention softmax (M=1, D512) | small (KV window 128 + ≤512 comp rows) | <1 ms | ~25–30/layer | D512 ∉ MLX fused-SDPA dims → full score materialize; tiny at M=1 (K6 is a *prefill* lever) |
| HC mix + Sinkhorn (×2/layer) | small (fn GEMV) | <0.5 ms compute | **~80 disp/mix w/o kernel; 1 w/ kernel** | 20-iter 4×4 recurrence ×2×40 = the top dispatch source (K3) |
| Engram gather+wkv (L1,L14) | 24 rows ×264 B ×2 = 12.4 KiB | <0.1 ms | small | latency, not BW; OPT_LEDGER R9 = ~0 tok/s |
| Sampling / argmax | 129280 f32 = 0.52 MB | ~1 µs | 1 host sync | greedy argmax; forces the token decision |
| **Dispatch host-encode (all)** | — | ~9 ms (prod-discounted; ~20 ms naive) | ~6.8 k × 2.9 µs (Sinkhorn-dominated) | Sinkhorn kernel removes most (K3/K4) |

**AR exposed floor once SSD is hidden ≈ barrier(40) + resident-read not overlapped(~16) ≈ 40–50 ms
→ 20–25 tok/s.** DRAM-bandwidth ceiling (42 tok/s) is *looser* than the barrier ceiling (25 tok/s):
**decode is host-sync-bound, not BW-bound, after streaming.**

### 2.2 Per K+1-row verify cycle (MTP, K=3 → 4 rows in one forward)

The verify forward is one pass with M=4 query rows. Cross-ref OPT_LEDGER R1/R2 for the SSD side;
GPU side:

| Component | Per cycle | Per accepted token (÷~2.85) | Notes |
|---|---:|---:|---|
| Resident dense read | 10.65 GiB (read once, M=4 GEMM) | **5.8 ms** (vs 16.5 AR) | ~3× amortization — M=4 GEMM ≈ same bytes as M=1 GEMV |
| Routing barrier | 40 × ~1 ms = 40 ms (once) | **14 ms** (vs 40 AR) | the single biggest GPU-visible amortization MTP buys |
| Routed experts | union `u`/layer × 40 × 18.8 MB | depends on R2 dedup | u≈6 → 1.58 GB/acc-tok; u≈24 → 6.3 GB/acc-tok (worse than AR) — OPT_LEDGER §1.4 |
| gather_qmm | M=4 (4 verify rows at once) | better ALU/byte than 4×M=1 | K10 — dequant amortized over 4 rows |

**Verify is GPU-cheaper per accepted token than 4 AR steps** because the resident read and the 40
barriers run once and cover ~2.85 tokens. This is the GPU-side of OPT_LEDGER R1 and is *only* a win
if K10 (batched M=4 dispatch) and R2 (dedup) fire; without them the barrier still fires 40× but the
expert union widens (GLM trap).

---

## 3. Per-component GPU cost model — PREFILL

Prefill is **SSD-bank-read bound**, not compute bound. Routing over many tokens covers ~all 384
experts/layer, so prefill reads (up to) the **whole 269 GiB bank once** = ~20 s @ 13.42 GB/s. The GPU
compute must be *overlapped under* that read, and the read must not be repeated. Two shapes:

### 3.1 1,024 tokens (one-shot — chunk auto-derives > s, so single pass)

`_prefill_score_bytes_per_row(1024)` = 64×(1024+512)×4 = 0.39 MB/row → derived chunk ≈ 20 k > 1024 →
**one-shot**, byte-for-byte the single forward.

| Component | Cost | Notes |
|---|---:|---|
| Bank read (≈all 384/layer touched, coupon-collector at 6144 draws/384) | ~19–21 s @ 13.42 | dominates; ≈ measured TTFT 27 s minus compute |
| Resident dense read (M=1024 GEMM, **once**) | 10.65 GB / 614 = 17 ms | trivial vs bank read |
| Attention score transient | [1024,64,1536] f32 = 402 MB (one-shot) | under the buffer cap; K6 shrinks it |
| MoE / attn FLOPs (M=1024) | compute-bound, ~few s | gather_qmm M=1024 = good ALU util (unlike M=1) |
| Head logits (all rows) | [1,1024,129280] f32 = 0.53 GB | K19: head last row only → decode needs one row |
| Routing barriers | 40 (one pass) | negligible vs 20 s |

**1024 TTFT lever = overlap compute under the ~20 s bank read + don't re-read the bank** (OPT_LEDGER
R5). GPU-kernel levers (K6/K17) shave only the ~5–7 s of compute *not* hidden under the read.

### 3.2 16,384 tokens (W20 token-chunked)

`_prefill_score_bytes_per_row(16384)` = 64×(16384+8192)×4 = **6.29 MB/row** → derived chunk ≈ 8e9/6.29e6
≈ **1,272 rows → ~13 chunks**.

**⚠ The dominant 16K TTFT term is the per-chunk expert re-gather.** W20 chunks the *query* rows to
bound the attention transient, but each chunk's MoE routes its ~1,272 tokens over ~all 384
experts/layer, and the expert cache (~85–100 slots/layer under 82 GiB) cannot hold 384/layer — so
**each of the ~13 chunks re-streams ~the whole bank**:

```
13 chunks × 384 experts/layer × 40 layers × 18.8 MB  ≈  13 × 269 GiB  ≈  up to ~260 s SSD read
```

That is **up to ~13× the one-shot bank read** and, if realized, *dominates* 16K TTFT — far above the
compute. This is the single biggest prefill lever and it is a **today** problem (16K uses chunking
now). Fix = **K16 (expert-major "read the bank once" across chunks)**: accumulate all chunks' routing
per layer, then stream each expert **once** and apply it to every chunk-row that routed to it →
13× → 1× ≈ 20 s.

| Component | Cost (as-is, chunked) | With K16 |
|---|---:|---:|
| Bank read | up to ~13 × 20 s ≈ **~260 s** | ~20 s (once) |
| Attention score (per chunk) | [1271,64,24576] f32 = 8 GB transient/chunk | K6 two-pass split-K shrinks it |
| Resident read | 10.65 GB × 13 (re-read per chunk!) = 138 GB / 614 = 0.23 s | fold into K16 layer-major pass |
| **Head logits (all rows)** | [1, 16385, 129280] f32 = **8.47 GB** (one-shot and chunked alike, W20 §7) | **K19: head only the last row at prefill** — decode seeds from the last token; the 8.47 GB + the head GEMM over 16 384 useless rows is pure waste |

**16K TTFT is bank-re-gather bound; K16 is the lever, K6 shrinks the attention transient, K17/K18
tune the compute.**

---

## 4. `mx.eval` / dispatch / host-sync census (counted from the ported forward)

- **AR decode, per token:** ~**40** forced `mx.eval` — one `mx.eval(indices)` per layer in the
  streamed switch (`expert_mlx.py` `HotExpertSwitchGLU.__call__:1976`, `MappedExpertSwitchGLU:1802`),
  each a host round-trip that `.tolist()`s the 6 routed ids to issue the SSD/DRAM gather — **plus 1**
  at sampling. Resident/island experts (`DenseIslandSwitchGLU:642`) take `rhs_indices` as an mx.array
  and need **no** host sync; the barrier is a *streaming* tax.
- **Verify cycle:** the same ~40 `mx.eval(indices)` run **once** for all 4 rows (M=4) → ~14 ms/acc-tok.
- **Prefill (chunked):** ~40 `mx.eval(indices)` **per chunk** → ~13×40 = ~520 barriers at 16K (trivial
  vs the bank read, but K16's layer-major pass collapses them to 40).
- **Dispatch count/token:** dominated by the **2 × 40 = 80 Sinkhorn recurrences** (20-iter 4×4 each,
  `hc_split_sinkhorn` via `_mixes`), ~80 dispatches each without the V4 `_sinkhorn_metal_kernel` →
  ~6.4 k dispatch just for HC-mix normalization (matches V4's 6,794 → 86 with the kernel). This is the
  compute-floor lever that becomes visible post-streaming (K3).
- **A second per-layer sync on the DSV4.1 lane (K23):** besides the barrier, the *fenced* default
  (`deferred_pin_release=False`, `split_route_release="fenced"` — DSV4.1's `build_streaming_config`
  leaves both at their dataclass defaults) forces one blocking `mx.eval(wave_output)` per all-hit
  layer (`synchronous_fence`; W37 route bracket `hot.allhit_fence_eval` = 0.65 ms × 1558 all-hit
  layer-calls ≈ 15.9 ms/token) plus one per wave part on split routes. hy3/glm defer both, so their
  profiles pay **only** the barrier per layer. K23 (`MTPLX_DSV41_SWITCH_FASTPATH`) removes this second
  sync for DSV4.1 by promoting the same deferred release.
- **No whole-forward `mx.eval`** in `_forward_span` (decode path); the only forced evals are the
  per-layer streamed-switch barrier, the (fenced-default) all-hit/split wave fence above, and (in
  chunked prefill) `_eval_cache_state` per chunk.

---

## 5. Candidate list (kernel / byte / fusion), ranked by GPU-visible EV

Each: **mechanism · where · gain now (SSD-bound) vs after streaming levers land · exactness risk ·
effort · precedent (measured) · rank.** "Now" is almost always ~0 because these live behind the SSD
wall; the point of the column is that they **switch on together** when OPT_LEDGER R1–R5 expose GPU
time, and several are then **on the critical path to 20 tok/s**, not refinements.

### K1 — Per-layer routing-barrier reduction / decision-sync overlap — **Rank 1**
- **Mechanism:** the 40× `mx.eval(indices)` p50 ~1 ms barrier is the largest exposed decode cost.
  Levers: (a) `async_eval` the barrier and fill its GPU-idle window with work that does *not* depend
  on the routed ids — the shared expert (already flag-gated `MTPLX_HY3_SHARED_HOIST`, ~0.44 ms
  recovered, `expert_mlx.py:1956`), the next layer's HC-mix/attn-norm, the head-independent prologue;
  (b) route the **DRAM-resident** fraction via `gather_qmm(rhs_indices=mx.array)` with **no** host
  list (the `DenseIslandSwitchGLU` path), syncing only the **miss** set — as residency (R3) rises the
  sync'd id count shrinks toward the miss count.
- **Where:** decode + verify. **Now:** ~0 (hidden). **After:** cuts ~40 ms → ~15–20 ms → the single
  biggest post-streaming decode gain; **necessary to clear 20 tok/s** (§0).
- **Exactness:** none (execution reorder; shared-hoist is bitwise-identical).
- **⚠ Effort/risk:** medium, and **measure — do not assume.** [[a3b-decode-roundtrip-is-the-lever]]:
  removing the Python sync was **+3.27 % *slower*** because MLX's async submission backpressure
  (10-buffer / 50-op force-commit) conserves the blocking regardless of the Python sync. So the win is
  **overlap-fill**, not sync-deletion; price it as utilization and A/B with `mx.eval` counted per arm.
  **The DSV4.1-specific reason it may still pay where a3b did not:** a3b's barrier was a bare decision
  `_eval` drain (nothing to hide behind); DSV4.1's barrier additionally does a device→host `.tolist()`
  to build the SSD-gather id list — real host work with a genuine GPU-idle window that the shared
  branch (x-only dependency) provably fills. Recover the *proven* piece first (shared-hoist), then
  test whether async_eval of the next layer's route-independent prologue clears more without tripping
  the backpressure floor.
- **Precedent:** shared-hoist ~0.44 ms already coded (`expert_mlx.py:1956`); a3b one-sync-per-cycle
  KILL (the ceiling, not a promise).

### K23 — DSV4.1 all-hit switch fence removal (deferred-release, W42) — **Rank 1b (companion to K1; the second per-layer sync K1 does not touch)**
> **⚠ GPU window 14 result (integration 7770960a9, 1,024/256, 82 GiB):** pure defer
> (`switch_fastpath`) decoded **3.48 vs control 4.02 tok/s = −13.4 %**, byte-identical
> (same token sha), same 77.7 GB peak. The **a3b trap fired** (worse than a3b's
> −3.27 %). Diagnosis (W42_SWITCH_HOT_PATH.md §8): the all-hit deferred branch
> submitted **no** GPU work (unlike the split defer branch, which `async_eval`s its
> wave — "the GPU still needs the part submitted now"), and DSV4.1's backbone has no
> `MTPLX_HY3_SUBMIT_CADENCE` equivalent (hy3_mlx.py:1165, out of this allowlist), so
> on all-hit layers the lazy graph accrued and the device idled until the next
> barrier drained it in one lump. **Variant B** (`switch_fastpath_b` =
> `MTPLX_DSV41_SWITCH_FASTPATH=1` + `MTPLX_DSV41_SWITCH_SUBMIT=1`) `async_eval`s each
> all-hit wave output — a non-blocking per-layer submit that keeps the GPU fed
> without the blocking round-trip, matching the split path. **A/B-pending; default
> off.** If variant B also fails to win, K23 is dead-on-this-lane (the barrier
> already drains every layer, leaving no accumulated-graph window a within-switch
> submit can fill — the real cadence lever is in the backbone, W41's allowlist).
- **Mechanism:** the DSV4.1 lane pays a **second** per-layer device→host sync besides the K1 routing
  barrier — the all-hit **wave fence** `synchronous_fence` → `mx.eval(wave_output)` in
  `HotExpertSwitchGLU._run` (route bracket `hot.allhit_fence_eval`). W37 window-13 measured it at
  **1016.9 ms / 1558 all-hit layer-calls = 0.65 ms each ≈ 15.9 ms/token**; split routes add one
  blocking fence per wave part (1 hit + N miss). **hy3/glm never pay it:** their profiles ship
  `deferred_pin_release=True` + `split_route_release="deferred"` (`mtplx/data/expert_profiles.json`),
  so the slot release **defers to the next layer's barrier** — the wave output is an ancestor of the
  next `mx.eval(indices)`, which materializes it for free — and split waves dispatch via `async_eval`.
  DSV4.1's `build_streaming_config` (`deepseek_v41_loader.py:284`) leaves **both at the dataclass
  default (`False` / `"fenced"`, `expert_runtime.py:180,205`)**, so this lane — and only this lane —
  eats the extra sync. **W37 proof this is the cost, not miss I/O:** warm-repeat decode == cold
  (3.80 vs 3.73 tok/s), so every-expert-resident does not speed decode; the exposed per-layer host
  round-trips do. The all-hit gather itself reads 6 × **18,800,640 B** (manifest, W24-consistent) =
  ~113 MB/layer from the resident bank ≈ ~0.19 ms @ 600 GB/s — **not** the 2.5 ms.
- **Lever:** env `MTPLX_DSV41_SWITCH_FASTPATH` (default off) promotes the identical, already-shipped
  deferred mechanism for the DSV4.1 lane **without mutating the config** (other lanes/callers
  untouched; A/Bs cleanly, W28 env-isolation convention). It **removes** the second sync (not fill —
  the next barrier is the covering eval, so **no blocking sync is added**), and stops serialising the
  decode so layer N+1's dispatch overlaps layer N's gather. **Distinct from K1:** K1 *fills* the
  barrier's idle window (barrier stays); K23 *removes* the separate wave/slot fence. They compose.
- **Where:** decode + verify (all-hit + split). **Now:** exposed **today** (window-13 measured), not
  behind the SSD wall — the fence fires every all-hit layer regardless of residency. **After:**
  removes ~40 × 0.65 ms ≈ **26 ms/token** of exposed all-hit fence plus the split per-part fences;
  expected all-hit switch **~2.5 → ~1 ms/layer** (toward hy3's barrier-only floor). GPU A/B prices
  the net decode delta.
- **Exactness:** none — pure fence/release-timing reorder; gather math unchanged, output byte-identical.
  Proven: fake-bank all-hit/split/all-miss at M=1 and M=4 + a real streamed runtime whose logits equal
  the fenced default with pins held until the covering flush (`tests/test_deepseek_v41_switch_
  fastpath.py`); the underlying deferred mechanism's slot-safety is
  `test_streamed_models.py::test_deferred_split_route_release_matches_fenced_bitwise`.
- **⚠ Effort/risk:** low; guarded — engages only when the runtime implements `defer_slot_release` /
  `flush_deferred_slot_releases` (real `ExpertStreamingRuntime` does; a fake without it falls back to
  the shipped fence, never crashing). The a3b sync-conservation caution
  ([[a3b-decode-roundtrip-is-the-lever]]) does **not** bite here: this is the pin/release **lifecycle**
  fence, not the decision sync — it is genuinely redundant with the next barrier, so its removal adds
  no blocking `mx.eval` (asserted per arm, [[moe-exec-fusion-25-26-27]] "COUNT mx.eval PER ARM").
- **Precedent:** hy3-oq2e profiles ship this exact pair in production (`expert_profiles.json`);
  W28's shared-overlap uses the same env-isolation pattern.

### K24 — Barrier-free all-hit device route (W44) — **SHELVED (window-19: NOT exact on the real model; default off, out of stack_a)**
> **⚠ GPU window 19 (integration 0b35a8bc2, 1,024/256):** `device_route` decoded
> 3.19 tok/s vs control 4.05 (**−21%**) and was **NOT byte-identical** (tokens
> collapsed to 0 from the first decode step; sha `c0a892a0…`). `stack_a`+device_route
> 3.77 (−7%, also non-identical) vs 6.24 without it (window 15). So on the real model
> it costs more than it saves **and** is inexact.
> **Root cause (proven on CPU, `test_deepseek_v41_device_route_parity.py`):** the
> barrier-free gather reads a component-bank slot via the LUT **without pinning it**
> and **defers** execution (async; forced only at the token-end flush), while
> `gather_qmm` reads the bank at **eval time**. Any admission that recycles that slot
> **in place** — the cold-recovery pass's fenced admissions, or the next token's LRU
> eviction/admission over a 256-token decode — overwrites the slot's bytes before the
> pending gather runs, so it reads the wrong expert's weights → catastrophic garbage
> → all-zero logits. The snapshot-based miss check cannot catch it (it verifies
> expert *membership* at LUT-build, not slot *stability* through eval). The fenced
> path is safe precisely because it **pins** the route's slots and **fences** the
> gather immediately (the wave fence) before the slot can be reused. CPU proof:
> a deferred gather over a real component bank reflects an in-place slot mutation
> applied after issue (max|Δ| 3e4), i.e. it is not isolated from recycling.
> **No barrier-free-and-exact fix exists for this LRU bank:** safety needs either a
> per-layer fence or pinning the read slots, and both require the host-side slot ids
> the lever removed (= the barrier). device_route is exact **only** when the resident
> set is static for the whole decode (no miss, no eviction) — not the window-12 rate.
> **Kept as a standalone arm (default off); removed from stack_a** until a redesign
> (e.g. persistently pinning each layer's resident set for the decode, at a memory
> cost W24 already priced against) clears the `MTPLX_GPU_PARITY` window.
- **Mechanism:** the per-layer `mx.eval(indices)` routing barrier exists only because the
  **host** needs the routed ids to (1) check residency and (2) build the gather's slot indices.
  Put the expert→slot map **on the device** — a per-layer LUT `lut[layer]` (int32 `[n_experts]`,
  slot for resident / −1 for miss, snapshot of `LayerExpertSlotBank._expert_to_slot`, rebuilt on the
  host **only when residency changes**) — and issue `gather_qmm` over `lut[indices]` **without
  evaluating `indices` on the host**. Residency verification is deferred: `async_eval(indices)` +
  an enqueued probe read back at the next flush / token end (covered by the sampler's eval). **0
  host syncs on an all-hit layer.**
- **Where:** decode (all-hit). **Now:** exposed today (the barrier is the top decode-stage cost,
  W44 §0). **After:** per-layer routing barriers **40 → m** (`m` = miss layers, fenced on the recovery
  pass) **+ 1** batched span-end verify sync. Warm (all-hit token) → **1**; window-12 cold (≈16 miss
  layers/tok) → ≈17; the barrier-free path itself is 0-sync per all-hit layer.
- **Exactness:** all-hit layer is byte-identical (LUT slot == fenced `bank_index`, same kernel, same
  rows). A miss reads a void row (clamped) and is caught by the deferred probe; recovery admits the
  miss experts and re-runs the fenced path → **final token byte-identical** (W44_DEVICE_ROUTE.md §2–3).
  Proven on the fake bank: all-hit/mixed/all-miss reconcile to fenced at M=1 & M=4; 0-sync-on-all-hit
  census; LUT-refresh-only-on-change (`tests/test_deepseek_v41_device_route.py`).
- **⚠ Effort/risk:** medium. **W44 lands it end to end** (switch + runtime + the decode-forward
  cold recovery in `DeepseekV41Backbone._forward_span` / `_device_route_recover`): on a cold token it
  rolls back every layer's cache to pre-token, re-runs the span on a fresh `shared` with only the miss
  layers forced fenced, and leaves output + cache + engram **byte-identical to fenced** while paying
  exactly `m` barriers. Byte-identity + barrier count are asserted on the tiny backbone with a
  controllable fake bank (all-hit/single/multi/all-miss × M=1,4). Real-artifact slot-numbering
  (`_expert_to_slot` row == `bank_index`) is validated by the GPU A/B sha.
- **Cost when misses frequent (honest):** warm (all-hit token) = **0 barriers, 1× compute** — the
  win. Cold (`m` miss layers) = **`m` barriers + 2× compute** (pass-1 + one recovery span; re-run
  from layer 0 because DSV4.1's CSA source layers sit near the top of the stack, so a partial
  `m1`-restart saves little and is fragile). At window-12 (≈16 miss layers/token) cold trades 40
  barriers for 16 + one extra span — net sign is the GPU A/B's call; if compute-bound, the partial
  `m1`-restart (W44 §7) is the follow-up. **Default off; GPU A/B prices it.**
- **Composes with / subsumes K23:** K23 deferred the second per-layer sync (the wave fence); K24
  removes the first (the barrier) on all-hit layers — no barrier means no fence to defer.

### K2 — MTP verify amortizes barrier + resident read (Factor A, GPU side) — **Rank (owned by R1)**
- **Mechanism:** one K+1 forward runs the 40 barriers and the 10.65 GB resident read **once** for
  ~2.85 accepted tokens (§2.2). **Where:** decode. **After:** barrier 40→14 ms/acc-tok, resident
  16.5→5.8 ms/acc-tok — the dominant GPU multiplier. **Exactness:** greedy verify = pure-AR argmax
  (PORT_PLAN P3.0). Ranked as OPT_LEDGER R1; here only its GPU cost is added. Pairs mandatorily with
  K10 + R2.

### K3 — Sinkhorn Metal kernel port from V4 — **Rank 2**
- **Mechanism:** collapse the 20-iter 4×4 Sinkhorn recurrence (`hc_split_sinkhorn`, ×2/layer ×40) into
  one Metal dispatch (V4 `_sinkhorn_metal_kernel`, 6,794 → 86 dispatches). **Where:** decode + verify +
  prefill. **Now:** ~0 (hidden). **After:** removes the top dispatch source (~5–8 ms/token of
  host-encode) — and the V4 measurement was taken in *exactly this regime* (V4 decode was
  dispatch-bound because its experts were resident): **AR +29.3 % (→28.86), K3 →32.50**
  ([[deepseek-v4-kernel-verdicts]]). **Exactness:** it is a normalizer; gate argmax parity (V4:
  bit-identical bf16). **Effort:** **low** — carry `_sinkhorn_metal_kernel` (1206) + `hc_split_sinkhorn`
  (1345) + `_install_sinkhorn_normaliser` (1312) from `deepseek_v4.py`; hc math identical (mult 4,
  iters 20, eps 1e-6). **Precedent:** measured +29.3 % AR on V4. `metal_kernel` composes with
  `mx.compile` in 0.31 ([[metal-kernel-compiles-in-031]]).
- **Status (W32, `feat/deepseek-v41-w32`):** ported behind `MTPLX_DSV41_SINKHORN_METAL` (default OFF,
  read-at-use, GPU-only). V4's `_sinkhorn_metal_kernel` / `_sinkhorn_kernel_apply` / `_sinkhorn_ops`
  are **imported and reused** (shapes identical, no fork); `deepseek_v41._hc_split_sinkhorn` carries the
  pre/post/comb split and routes the Sinkhorn tail through `_sinkhorn_normalise` (kernel on GPU+flag,
  recurrence on CPU / flag-off). Composes with chunked-prefill (W20) and layer-major (W30) — rows = any
  n. **Dispatch/token:** 80 Sinkhorn calls × 119 primitives (78 reduce+divide core) = **9,520 → 80**
  (one Metal launch/call). CPU dispatch gates green (8 pass / 1 skip; 115 pass / 23 skip across the
  V4.1 suites, 0 fail), flag-off bit-identical to the stock split. GPU numeric parity
  (`tests/models/test_deepseek_v41_sinkhorn_metal.py::test_sinkhorn_kernel_parity_gpu`, fp32 1e-6 +
  argmax, bf16 argmax) present, **skipped unless `MTPLX_GPU_PARITY=1`** — awaiting a GPU window. See
  [W32_K3_SINKHORN_METAL.md](W32_K3_SINKHORN_METAL.md).
- **Status (W38, `feat/deepseek-v41-w38` off `bfd361424`):** window-12 parity failure **root-caused** —
  the V4 kernel source **fails to build for a bf16 buffer** (`out[off+i]=c[i]`: `float`→`bfloat16_t`, no
  store cast), so the W32 test's raw `_sinkhorn_kernel_apply(comb_bf16)` raised a Metal build error;
  **fp32 is bit-identical (`max|d| 8.9e-8`)** and was never the issue. **Fix:** `_sinkhorn_normalise`
  upcasts a non-fp32 comb to fp32 for the kernel and casts back (production comb is fp32 → no-op).
  **Also fixed a live merge regression:** W32's `_hc_split_sinkhorn` rename orphaned the name W33's
  compiled path (`_hc_mixes_split`) + its test call, so on `bfd361424` the K4 HC-compile suite was **6
  RED** (`NameError`/`AttributeError`) and the compiled decode path bypassed the K3 kernel; renamed back
  to canonical `hc_split_sinkhorn` so **both eager and compiled paths route through the kernel**
  (K3×K4 compose; K4 suite 8/8 green). Added **engagement counters** (`_SINKHORN_KERNEL_CALLS` /
  `_SINKHORN_RECURRENCE_CALLS` + probe stages `hc.sinkhorn_kernel`/`hc.sinkhorn_recurrence`) surfaced
  per arm as `sinkhorn_engagement` in `ab_decode_env_levers.py`, and a **self-diagnosing** parity test
  (JSON receipt via `MTPLX_PARITY_RECEIPT`; bf16 gated vs fp32-recurrence-rounded, raw-bf16 build
  failure recorded). CPU tests 14/1; broad V4.1 sweep 131 pass / 0 fail. See
  [W38_K3_SINKHORN_DIAGNOSTICS.md](W38_K3_SINKHORN_DIAGNOSTICS.md).

### K10 — Verify K+1 rows in one batched dispatch per layer — **Rank 3**
- **Mechanism:** run the 4 verify rows through each layer's MoE as one M=4 `gather_qmm` over the
  **deduped union** (R2), not 4 M=1 gathers, and collapse the 4 per-position barriers into 1/layer.
  **Where:** verify. **After:** one barrier/layer (not 4), dequant ALU amortized over 4 rows (M=4 is
  materially better than M=1 on the ALU-bound sub-4-bit path, [[metal-sub4bit-alu-bound]]).
  **Exactness:** none (same records, same rows). **Effort:** medium (verify path, W23-owned; this
  prices it). **Precedent:** [[qwen38-verify-band-sdpa-dead-band]] batched-verify; pairs R1/R2.
  **Deps:** R2 dedup (union), W23 DSpark.

### K16 — Prefill expert-major "read the bank once" across chunks — **Rank 4 (TTFT)**
- **Mechanism:** W20's per-chunk MoE re-streams ~the whole bank per chunk → up to ~13× at 16K (§3.2).
  Restructure to **layer-major**: chunk attention to bound the score transient, but accumulate all
  chunks' post-attention hidden for a layer, then stream each expert **once** and apply it to every
  routed row across the full prompt (expert-major gather, M = all rows that routed to it). **Where:**
  prefill (16K decisively; 1024 already ~1×). **Now/after:** ~**13× on the 16K bank-read term**
  (~260 s → ~20 s) — the dominant 16K TTFT lever, live *today*. **Exactness:** reorder; bit-exact if
  the f32 routed-sum accumulation order is preserved (spot-check long-prompt A/B). **Effort:** high
  (MoE restructure + peak-activation accounting: 16384×4×5120×2 ≈ 0.67 GB/layer hc stream, under
  budget). **Precedent:** [[deepseek-v4-longcontext-prefill]] block-shared top-512 gather = +11 %,
  byte-exact >1024; OPT_LEDGER R5 (this is R5's GPU-structural realization).
- **STATUS (W30, `feat/deepseek-v41-w30`):** IMPLEMENTED on CPU as **option (a)** —
  layer-major driver (`_forward_layer_major` in `deepseek_v41.py`) iterates every
  layer over all chunks and issues **one** row-capped `switch_mlp` call over the
  concatenated chunk rows per layer, so `partition_route_waves` gathers each of the
  layer's experts once. Option (b) is impossible (transient slot pool defaults to
  `top_k`=6 ≪ 384/layer; the 269 GiB bank is streamed, never resident). Proven
  CPU-exact vs one-shot (logits + full cache state ≤ 1e-5, engram history + KV lanes
  identical), and a counting fake switch shows **max fetches/(layer,expert) = 1**
  (layer-major) vs **C** (chunk-major) — the read-once claim. Resident hc state
  0.671 GB @ 16 K (< 1 GB); one-call routed transient 2.01 GB; MoE row cap 65,104
  rows @ 8 GB. Opt-in `MTPLX_DSV41_PREFILL_LAYER_MAJOR` (default OFF — chunk-major
  stays the default so W20 tests are unchanged).
- **STATUS (W30, GPU window 14 follow-up, 2026-09-11):** window 14 confirmed the
  **prefill win is real** — TTFT 494.88→**372.98 s** (−24.6 %), prefill
  33.11→**43.93 tok/s** (+32.7 %) — but the first cut flipped **one greedy token**
  (decode position 4: control 1449 vs 18) and ran +4.7 GB peak / −10 % decode.
  ROOT CAUSE (CPU-localised): the *resident* router gate (`xf @ weight.T`) and
  shared `Expert` are **not M-invariant**, so batching the whole MoE reassociated
  the gate scores ~1e-6 and flipped a top-k near-tie (different expert selected);
  `SwitchGLU`/the streamed gather IS M-invariant. FIX: compute the gate + shared
  **per chunk** (byte-identical routing), batch **only** the streamed `switch_mlp`
  (still bank-read-once). Now **byte-for-byte == chunk-major** on CPU
  (`mx.array_equal` logits + greedy argmax + full cache state). Peak (2.01 GB
  batched routed transient; row-cap is the knob) and decode-seed differences are
  documented (decode census is decode-only, so no residency-policy regression).
  Re-run **KG-b** on the fixed branch to confirm greedy tokens now match control.
  The ~13× 16 K-TTFT win holds. See `W30_K16_LAYER_MAJOR_PREFILL.md` §8.

### K4 — HC-compile + fused CSA attention (V4 dispatch stack) — **Rank 5**
- **Mechanism:** V4's HC-tape collapse + fused CSA attention: **AR +31.3 % (17.37→22.80), K3 +17.4 %,
  −26.1 % dispatches** ([[deepseek-v4-kernel-verdicts]]). **Where:** decode + prefill. **After:** the
  same post-streaming dispatch-bound regime → material; **now:** ~0. **Exactness:** parity-gated (V4
  bit-identical). **Effort:** low (V4 reuse). **⚠** keep the *default fused* SDPA — do **not** chase a
  hand MLA kernel (K-dead below). **Precedent:** measured on V4.
- **STATUS (W33, `feat/deepseek-v41-w33` @ `3b89fb61f`):** HC-tape half IMPLEMENTED
  on CPU. `mx.compile` tapes (`_hc_attn_prep_impl` / `_hc_ffn_prep_impl` /
  `_hc_post_impl` in `deepseek_v41.py`) replay the per-sublayer Hyper-Connection
  pre/post chains — layer HC weights as tape INPUTS (one tape shared across all
  `2*n_layers` HCs), the **Sinkhorn kept as an opaque `hc_split_sinkhorn` boundary
  so K3's (W32) Metal kernel drops in at the tape tail**, attention + MoE switch
  OUTSIDE the tapes (pure, no cache mutation). Opt-in `MTPLX_DSV41_HC_COMPILE`
  (default OFF). Fixed-shape + row-cap `_HC_COMPILE_MAX_ROWS` **exactly as V4**
  (shapeless REJECTED, measured: W32's `hc_split_sinkhorn` reshape can't trace
  shapeless, and shapeless adds ~1e-6 at batch>1). CPU-exact vs eager: flag on vs
  off is `mx.array_equal` over decode / K+1 verify / chunked + layer-major prefill;
  the dispatch collapse is asserted (eager rebuilds the HC graph 2×/layer/token,
  warm compiled replay rebuilds ZERO; Sinkhorn ~119→~79 dispatches/mix from
  elementwise fusion, `312→~180` HC dispatches/layer). **NOT carried:** the head
  HC (DSV4.1 has none — threaded `pre_mix` collapse + RMSNorm, no `fn`/sigmoid),
  the fused-CSA/MLA attention (mutates KV / D512 ∉ fused-SDPA → K6, hand MLA
  Dead-here). Realized GPU decode/dispatch delta unmeasured — that is **KG-f**;
  V4 measured AR +31.3 %, −26.1 % dispatches. See `W33_K4_HC_COMPILE.md`.

### K6 — D512 two-pass SDPA split-K port (gemma4) — **Rank 6 (prefill)**
- **Mechanism:** head_dim 512 is not an MLX fused-SDPA dim (64/96/128/192/256), so attention falls
  back to full `[b,s,H,T]` f32 score materialization (the 8 GB/chunk transient at 16K, §3.2). Gemma4's
  D512 two-pass SDPA split-K avoids the full-T materialize: **+18 %** on the attention term
  ([[gemma4-dflash-cycle-program]] / gemma4 D512 two-pass). **Where:** prefill (M and T large); decode
  M=1 is too small to benefit. **After/now:** shrinks the 16K attention transient and its compute;
  helps peak (protects the knob). **Exactness:** split-K reassociates the softmax denom — validate
  parity (near-tie risk, same caution as the V4 MLA kernel). **Effort:** medium. **Precedent:** gemma4
  +18 % measured.

### K7 — mxfp4 native gather kernel vs affine-q4 (MEASURE) — **Rank 7**
- **Mechanism:** the routed bank is **native mxfp4 gs32**; MLX 0.32.2 ships `gather_qmm(mode="mxfp4")`
  (144 Metal variants) but its **speed vs affine gs32 is UNMEASURED** ([[mlx-native-float-quant-modes]],
  OPT_LEDGER §8). At M=1 sub-4-bit gathers are ALU/occupancy-bound ([[metal-sub4bit-alu-bound]]); the
  native codec may cost more (or less) ALU than affine. **Where:** decode (M=1, ALU-bound) + prefill
  (M=chunk, compute-bound). This is the **residual decode lever** once K1/K3/K4 exhaust
  barrier+dispatch — V4 found the binding term after the Sinkhorn+CSA stack is the MoE gather/dequant
  forward itself ([[deepseek-v4-kernel-verdicts]] "road to 40"). **Measure plan:** queued-lane
  microbench, `gather_qmm(mode="mxfp4")` on the shipped native records vs an affine-q4 gs32 repack of
  the same records (a **speed reference on equivalent shapes, not a shippable alternative** — affine q4
  gs64 is only cos 0.995 / 285 GiB per W16; native mxfp4 stays the bit-exact bank), at
  M in {1, 4, 1271}, **x shape `[rows,1,K]` enforced** ([[gather-qmm-calling-convention-trap]]),
  `mx.eval` counted per arm. **Decides:** whether the native bank carries a decode/prefill compute tax;
  **~0 today** (behind SSD). **Effort:** low (microbench only). **Precedent:**
  [[iq2xxs-kernel-loses-to-stock]] (hand sub-4-bit loses 1.74× to stock) → use the stock native kernel;
  the microbench only picks native-vs-affine among *stock* codecs, never a hand kernel.

### K8 — mxfp8 resident GEMV vs affine-q8 (MEASURE) — **Rank 8**
- **Mechanism:** the 10.65 GiB residents are **mxfp8**; every decode token GEMVs all of them (§2.1).
  W18 flagged an mxfp8 **GEMM-precision** finding (OPT_LEDGER §8) — is there a **speed** delta vs the
  original affine q8 gs64 at M=1? If mxfp8 `quantized_matmul` is slower at M=1, the whole 16.5 ms
  resident read pays a tax on every token. **Where:** decode (M=1) + verify (M=4). **Measure:**
  microbench mxfp8 vs affine-q8 on the wq_b / wo_b / head shapes at M∈{1,4}, queued lane. **Decides:**
  resident-read tax; **~0 today**. **Exactness:** format is fixed by the artifact (native repack);
  this only informs whether to keep native or affine residents. **Effort:** low. **Precedent:**
  W18 precision note; speed unmeasured.

### K9 — Head GEMV byte cut — **Rank 9**
- **Mechanism:** head is **bf16 [129280×5120] = 1.323 GB read/token** (12 % of the resident floor),
  kept dense by the native predicate (`deepseek_v41.py:963`). q8/mxfp8 head → ~0.66 GB → save ~1 ms
  (~+2–4 % at a 40–50 ms decode). Top-k-head tricks (compute only candidate logits) are **incompatible
  with greedy argmax exactness** — argmax needs the full 129280 row — so only the byte-cut, not the
  compute-cut, is in play. **Where:** decode + verify + prefill. **After:** ~+2–4 % decode; **now:**
  ~0. **Exactness:** **risk** — q8 head can tip near-tie argmax; gate the streamed==resident argmax
  (PORT_PLAN P1.7) and a HumanEval cell before shipping. **Effort:** low. **Precedent:** [[report-fastest-of-seeds]]-class byte cuts; q8 head is standard but exactness-gated here.
- **⚠ premise superseded by K21 (measured W40):** the head is **not** a clean 1.32 GB
  bf16 read — the call-site `astype(float32)` makes it the **fp32-cast trap (6.6 GB /
  70.9 ms/token, the #2 decode stage)**. The byte cut here is real but the bigger,
  free win is removing the trap (`MTPLX_DSV41_HEAD_MODE=bf16`). See K21.

### K21 — Output-head fp32-cast trap fix + head codec lever (`MTPLX_DSV41_HEAD_MODE`) — **measured root cause of K9; promoted, landed W40 (CPU)**
- **Supersedes K9's premise.** K9 priced the head as a clean **1.32 GB bf16 read
  (~2 ms, +2–4 %)**. GPU window 13 stage timing measures the `head` stage at
  **70.9 ms/token** (64 calls, 1,024 ctx) — the **#2 decode stage (20.3 % of the
  fenced frame)**, ~35× a clean bf16 M=1 GEMV. **Root cause (read from the code, not
  guessed):** the residual stream `h` reaches the head as **bf16** (the HC merge at
  `_forward_span` sums in f32 but casts back to `h.dtype`; `_rmsnorm` returns the
  input dtype), but the head call site did `self.head(source.astype(mx.float32))` —
  an **explicit upcast of the hidden to float32**. `f32 @ bf16.T` has no
  mixed-precision matmul in MLX, so it **materialises a 2.648 GB float32 copy of the
  1.324 GB bf16 head weight every token** and GEMVs over it. This is exactly Hy3's
  [[hy3-resident-path-breakthrough]] "lm_head fp32-cast trap". (The trap is the
  call-site `astype`, **not** the HC merge — which already casts back to bf16.)
- **Byte traffic / token (M=1):** control (trap) = read bf16 1.324 + write f32 temp
  2.648 + read f32 GEMV 2.648 = **6.619 GB**; `bf16` (cast hidden to weight dtype) =
  **1.324 GB (5.0× less)**; `mxfp8` (native gs32, E8M0) = **0.683 GB (9.7×)**; `q8`
  (affine gs64) = **0.703 GB (9.4×)**. If the stage is BW-bound (its 6.6 GB and the
  35× gap say it is), scaling the 70.9 ms head: `bf16`→~14.2 ms (**−80 %**),
  `mxfp8`/`q8`→~7.3/7.5 ms (**−90 %**) — a first-order ~−16–18 % of the fenced decode
  frame, most of it from the **free lossless `bf16` fix alone**.
- **Lever:** `MTPLX_DSV41_HEAD_MODE ∈ {bf16, mxfp8, q8}` (default = current
  behaviour, byte-identical). Resolved at construction; the weight repack applied
  **once post-load** (`Model.apply_head_mode`, called by the loader after the real
  head weight loads — it does not exist at `__init__`). `mxfp8`/`q8` also free
  **~0.64 GB resident**, priced into `_mtplx_resident_load_report`
  (`head_resident_saved_bytes`). **Where:** decode + verify + prefill.
- **Exactness (CPU, `tests/models/test_deepseek_v41_head_lever.py`):** every codec
  flips the greedy argmax **only** within ~3× its own `max|Δ|` (genuine near-ties):
  over 256 random hidden vectors on a real-hidden (5120) head, `bf16` matches
  249/256 (max|Δ| 0.028, misses ≤ 1.63× max|Δ|), `q8` 254/256 (max|Δ| 0.054),
  `mxfp8` 239/256 (max|Δ| 0.188 — coarsest, E8M0 gs32). DSpark MTP verify (K+1 rows)
  yields `[1,K+1,vocab]` f32 logits in every mode; `bf16` keeps the verify argmax
  identical to default. Random head + random hidden is a worst case (near-uniform
  logits); a trained head with a dominant top-1 flips none. **Gate `mxfp8`/`q8` on a
  full HumanEval/MBPP eval** ([[task-evals-decide-bank-verdicts]]) before serving;
  **`bf16` is a free lossless bug fix** (ship as default). **Effort:** low.
- **W40 — LANDED (CPU-only; ms/token is a KG-class GPU window, arms `head_bf16` /
  `head_mxfp8` / `head_q8` in `ab_decode_env_levers.py`, not measured here).**
  Report: [`W40_HEAD_LEVER.md`](W40_HEAD_LEVER.md).

### K14 — `MLX_MAX_MB_PER_BUFFER` / `MAX_OPS_PER_BUFFER` sweep — **Rank 10 (cheap falsifier, low expectation)**
- **Mechanism:** command-buffer throttling knob — caps ops/bytes per command buffer, changing commit
  granularity and host-encode/GPU overlap (and peak). **Where:** decode + prefill.
- **⚠ measured ≈dead on a3b, do not expect a win:** [[a3b-decode-roundtrip-is-the-lever]] swept it —
  `200 → −0.9 %`, `2000 → +9 % worse`, and a promotion A/B was **rejected (+1.93 % worse under
  interleaved pairs)** — because the 10-buffer cap **was never binding** (the cv-wait is genuine
  GPU-dependency time). So this is not a fresh idea; it is a **cheap re-falsifier** in the DSV4.1
  streaming regime (which differs — the per-layer preadv issue may change the buffer pressure), run
  only to confirm the a3b verdict transfers, at ~0 expected. It bounds prefill peak as a side effect,
  which is the one place it may earn its keep. **Exactness:** none. **Effort:** trivial (env sweep).
  Run alongside K1 (they share the backpressure mechanism); keep buffer knobs at defaults unless a
  clean 3-pair A/B beats drift.

### K12 — Fused HC-mix + RMSNorm — **Rank 11**
- **Mechanism:** `_mixes` does rsqrt-normalize → `flat @ fn.T` → Sinkhorn split as separate dispatches
  ×2/layer. Fuse the rsqrt-norm + fn-matmul (and feed K3's Sinkhorn kernel) into one kernel to cut
  ~3–4 dispatches/mix. **Where:** decode + prefill. **After:** small dispatch cut, complements K3.
  **Exactness:** none (same math). **Effort:** medium. **Precedent:** [[moe-exec-fusion-25-26-27]]
  (count `mx.eval` per arm — fused-MoE looked 1.85× but was 75 % accounting artifact; hold this fusion
  to the same bar).

### K19 — Head only the last row at prefill — **Rank 12 (prefill peak + TTFT)**
- **Mechanism:** `Model.__call__` runs `self.head(h)` over **all** prefill rows → a **8.47 GB** f32
  logits transient at 16K (W20 §7) plus a 129280×5120 GEMM over 16 384 rows whose logits are never
  read — decode seeds only from the **last** token's logits. Slice `h[:, -1:]` before the head at
  prefill (keep all rows for chunked-forward continuation; only the head input narrows). **Where:**
  prefill (1024: 0.53 GB saved; 16384: **8.47 GB + a 16 384-row GEMM** saved). **After/now:** cuts the
  single largest prefill transient and a full head GEMM — helps TTFT *and* peak (protects the knob at
  16K). **Exactness:** none for AR (same last-token logits); if MTP/DSpark needs multi-row prefill
  logits, gate on which rows the draft head consumes. **Effort:** low (one slice, guard the
  cache-continuation path). **Precedent:** standard prefill last-token head; W20 §7 identifies the
  8.47 GB transient explicitly.

- **W29 — LANDED (CPU-only; magnitude is a KG-g GPU window, not measured here).** Report:
  [`W29_K19_HEAD_LAST_ROW.md`](W29_K19_HEAD_LAST_ROW.md). The last-row head already existed as the
  runtime `forward_ar` contract `logits_keep` (W23) — `logits_keep=1` slices `h[:, -1:]` before the
  head; W29 adds the explicit alias `logits_rows="last"` (same single slice path) and, crucially,
  **wires the mxfp4 lane to use it** by setting `MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS=0` in the
  `deepseek-v41-mxfp4-75` profile `child_env`. Until W29 the lane ran W20 chunking (which bounds the
  *attention/score* transient) but the head still built the full-row logits — the 8.47 GB was live at
  16K. The generic prefill runner (`generation._prefill*`) already emits **no** logits for the prompt
  body (`emit_logits=False`) and `logits_keep=1` for the last token when the gate is on, so no
  `generation.py` change was needed. **Verify (K+1 rows during MTP decode-verify) is not prefill and
  is untouched.**

  *Analytical transient reduction* (released config: `vocab=129,280`, `hidden=5,120`, f32 logits;
  head input narrows `[1, s, 5120] → [1, 1, 5120]`):

  | prefill tokens *s* | all-rows logits `[1,s,129280]` f32 | last-row `[1,1,129280]` f32 | transient saved | head-GEMM rows |
  |---|---|---|---|---|
  | 1,024  | 529,530,880 B = **0.530 GB** | 517,120 B = 0.49 MB | 0.529 GB (99.90 %) | 1,024 → 1 (1,024×) |
  | 16,384 | 8,472,494,080 B = **8.472 GB** | 517,120 B = 0.49 MB | 8.472 GB (99.99 %) | 16,384 → 1 (16,384×) |

  The head GEMM drops from `s·5120·129280` MACs (≈21.7 TFLOP at 16,384) to `5120·129280` (≈1.36 GFLOP),
  a factor-*s* cut. **Exactness:** `logits_rows="last"` is bit-identical to `logits_keep=1` (same M=1
  head); the surviving row matches the all-rows tail to ~1.8e-7 (measured) — the lm head is a matmul
  whose rounding depends on its row count M, so heading M=1 vs M=s differs sub-ULP-scale; **argmax is
  identical**, so AR decode is unchanged. **Call-site audit:** every DSV4.1 prefill consumer reads only
  the last row *except* `generation.score_prompt_logprobs` (prompt-logprobs / echo), which is
  independently chunked (`chunk_size=256`) and passes `emit_logits=True` unconditionally — it is
  unaffected by the gate and correctly keeps all rows. Session-cache prefill is off for this lane
  (`MTPLX_SESSION_STORE_ON_PREFILL=0`, W22); DSpark reads `main_hidden`, never the lm head. Full audit
  table in the W29 report.

### K17 — Prefill GEMM tiling for M=chunk — **Rank 13**
- **Mechanism:** at M=1271 the expert gather is compute-bound; tile to keep it so (avoid the M=1
  ALU-stall regime). **Where:** prefill. **After/now:** shaves the ~5–7 s of non-hidden prefill
  compute. **Exactness:** none. **Effort:** medium. **Precedent:** [[qwen27b-compile-profile]] small-T
  qmm ALU ramp; folds into K16's layer-major pass.

### K11 — Expert-record dedup at the gather call site (GPU side of R2) — **Rank 14 (owned by R2)**
- **Mechanism:** union the per-cycle expert-id set before the gather so each distinct record feeds one
  `gather_qmm` slot and one preadv. **Where:** verify. GPU side: fewer rhs slots + fewer syscalls;
  SSD side is OPT_LEDGER R2 (the binding win). **Exactness:** none. **⚠** slice the union preadv per
  IOV_MAX (a 40-layer union can exceed 1024 iovecs → EINVAL, [[hy3-c5-dense-islands]]). Ranked as R2.

### K13 — In-place verify-KV writes (footprint, GPU side of R7) — **Rank 15 (owned by R7)**
- **Mechanism:** avoid `mx.slice_update` KV copies during verify (Qwen3.8 256K: 6.4 GB/verify copy,
  [[qwen38-verify-band-sdpa-dead-band]]). CSA2 shares KV across 36 layers (only 4 own it), so fewer
  caches than Qwen3.8, but SWA windows + compressed caches are copy-prone. **Where:** verify.
  **After:** prevents a peak spike that steals expert-cache slots / trips the knob (~4× collapse,
  [[never-exceed-the-memory-knob]]). Not a tok/s add — a regression guard. Ranked as R7.

### K5 — `mx.compile` of the per-layer decode step (MEASURE, may be dead) — **Rank 16**
- **Mechanism:** compile the per-layer *non-expert* subgraph (HC-mix + attn + norms), leaving the
  host-sync'd expert gather outside the compiled region, to collapse its dispatches. **Where:** decode.
  **⚠ likely dead:** [[hy3-decode-roofline]] compile-**the-forward** was DEAD (async_eval already
  overlaps the graph rebuild); [[metal-kernel-compiles-in-031]] reopens *kernel*-compile but not
  necessarily step-compile. **Measure** a per-layer compiled step vs K3+K4 hand-fusion; if async_eval
  already hides the rebuild, mark dead. **Exactness:** none. **Effort:** low to test.

### K18 — Prefill attention chunk sizing sweep — **Rank 17 (cheap)**
- **Mechanism:** sweep `MTPLX_DSV41_PREFILL_CHUNK_TARGET_GB` (default 8 GB) — bigger chunks =
  fewer re-gathers (helps K16's problem) but larger transients; with K16 landed the chunk only bounds
  the attention score, so raise it to the K6 two-pass ceiling. **Where:** prefill. **Exactness:** none
  (W13 pooling is chunk-independent). **Effort:** trivial. Run after K16/K6.

### K20 — Engram 48-row gather + dequant + wkv GEMV fusion — **Rank 18 (latency, ~0 tok/s)**
- **Mechanism:** each token, the 2 engram layers (L1, L14) each gather 24 disk-backed rows (264 B
  mxfp8), dequant them, and run the `wkv` [25600,6144] GEMV + sigmoid gate (`engram_v41.py`). Fuse the
  per-row dequant + the gated add so the 48 small random reads issue as one sliced `preadv` and the
  dequant/gate is one kernel. **Where:** decode + verify + prefill. **Now/after:** the byte volume is
  ~12.4 KiB/token — **~5 orders below** the expert read, so this is **not** a bandwidth or tok/s lever
  (OPT_LEDGER R9 = ~0). It matters only if the 48 random small reads **serialize** badly against the
  compute (latency), which the engram-hash prefetch (OPT_LEDGER New #2 — hash the DSpark draft tokens
  to warm the rows before verify) addresses more directly. **Exactness:** none (same rows, same math);
  do **not** requantize further (PORT_PLAN §4: a second lossy step on a gate-damped memory buys
  nothing). **⚠** slice the 48-row `preadv` under IOV_MAX ([[hy3-c5-dense-islands]]). **Effort:** low.
  **Rank 18** — include for completeness; ship only if a probe shows the engram reads serialize.

### K22 — Attention-chain tape collapse (+ gate-prefix / MoE-combine folds) — **Rank (W41, sibling of K4)**
- **Mechanism:** window-13 stage timing (W37) put decode attention at **attn.reuse 50.2 ms/tok (30
  layers → 1.7 ms/layer for an M=1 MLA step) + swa_only 9.8 (2 layers → 4.9 ms/layer!) + full 8.9 (4)
  + reindex 8.0 (4)** — a single-row attention step costing 1.7–4.9 ms is a **dispatch-chain** problem
  (dozens of tiny projection/RMSNorm/RoPE ops), the regime where whole-chain `mx.compile` pays and
  single fusions do not ([[b1-decode-dispatch-removal-hides]]; unfenced decode 246 ms/tok, warm-repeat
  == cold ⇒ dispatch/compute-bound, not I/O). K22 replays the two **pure** attention chains from an
  `mx.compile` tape (K4's fixed-shape + row-cap design, `MTPLX_DSV41_ATTN_COMPILE`, default OFF): the
  pre-SDPA q/kv projection+norm+RoPE prep **up to (not incl.) the KV-cache write and the SDPA**, and
  the post-SDPA output chain (query-RoPE removal + grouped o-LoRA einsum + `wo_b`). Also folds the pure
  **MoE gate prefix** (score GEMM + sqrtsoftplus + correction bias; the argpartition/top-k routing
  barrier stays eager) and the **MoE combine** (weighted routed sum + shared add) under the same flag.
  **Where:** decode + verify (+ small prefill chunks ≤ cap). **Exactness:** projection weights are tape
  INPUTS applied by `_apply_lin` exactly as `nn.Linear`/`nn.QuantizedLinear` (dense **or** quantized —
  `mx.quantized_matmul` replays as one primitive, so compile never reassociates it), so byte-identical
  to eager off / above the cap; on and rows ≤ cap it is `mx.array_equal` (matmul reassociates at ≥ 8
  rows exactly as K4 — cap confines the tape to decode/verify).
- **STATUS (W41, `feat/deepseek-v41-w41`):** IMPLEMENTED + CPU-proven. Two shared tapes
  (`_attn_qkv_prep` / `_attn_out_prep` in `deepseek_v41.py`, one each across all 40 layers — same
  codec/geometry, weights as inputs) + `_gate_prefix` / `_moe_combine` in `deepseek_v41_moe.py`. The
  KV-cache write, the window mask, the data-dependent CSA index selection (Indexer top-k) and the SDPA
  stay OUTSIDE the tapes (pure). Unlike K4 it does **not** force-eager under the W37 probe (`_attend`
  is one `attn.<mode>` stage, no sub-stage fence). **Not compiled:** the SDPA/`_sparse_attend` (score
  materialisation, dynamic KV length; D512 ∉ fused-SDPA = K6), the Indexer score/top-k (dynamic n_comp
  + data-dependent), the streamed routed switch (W42 / `expert_mlx.py`), the lm head (W40), and
  `hc.premix_sinkhorn`/`hc.combine` (K4's `MTPLX_DSV41_HC_COMPILE` already folds them — the dominant
  ~4.1k prim/tok Sinkhorn is K3/K4 territory). **Dispatch census** (CPU, `scripts/deepseek_v41/
  dispatch_census.py`, `mx.export_to_dot` node count per W37 stage): per attention call **−40
  primitives every CSA mode** (qkv-prep 81→48, out-prep 38→31), gate prefix 8→3, combine 6→5; whole
  decode token **6,631 → 6,263 primitives/token**. Flag on/off `mx.array_equal` over decode / K+1
  verify / chunked + layer-major prefill; cache-state identical; census reduction asserted
  (`tests/models/test_deepseek_v41_attn_compile.py`, 11 CPU tests). Realized GPU decode/dispatch delta
  is **KG-i** (unmeasured). See `W41_DISPATCH_CENSUS_ATTN_COMPILE.md`.

### K24 — Sliding-window attend-mask memo (`MTPLX_DSV41_ATTN_WIN_MEMO`) — **Rank (W45, further attn dispatch cut)**
- **Mechanism:** after K22 the census puts the biggest *remaining* mode-invariant per-layer attention
  dispatch chunk in the causal sliding-window mask `broadcast((wp<=qp)&(wp>qp-window_size),[b,s,T])`
  (~11 graph nodes: `Arange`, 2 compares, `Subtract`, `BitwiseAnd`, broadcasts). It is a pure function
  of `(positions, window length T, window_size)`, all **invariant across the 40 backbone layers of one
  `_forward_span`** (positions and the per-forward `shared` runtime are made once and handed to every
  layer; every layer's window grows in lockstep to the same `T`) — yet each layer rebuilds it. K24
  memoizes it on the per-forward `shared` runtime (`Attention._window_attend`), computing it ONCE and
  reusing the identical array for the other `n_layers-1` layers. **Where:** decode + verify + prefill.
  **Exactness:** byte-identical (the reused array is the same object; reuse fires only when `positions`
  **is** the same object and `(T, window_size, b, s)` match, else per-layer recompute — never wrong),
  independent of and composing with K22. `shared` has no `__slots__`, so it is a plain attribute set
  from `deepseek_v41.py` (no edit to the W13 cache module).
- **STATUS (W45, `feat/deepseek-v41-w45`):** IMPLEMENTED + CPU-proven, default OFF. **Census (tiny
  8-layer):** attention primitives/token eager 1698.7 → K22 1378.7 → **K22+K24 1301.7** (−77/tok here =
  ~11 × 7 reusing layers; **~11 × 39 ≈ 429/tok at 40 layers**, larger than K22's 368); whole token
  6631 → 6186. Flag on/off `mx.array_equal` over decode / K+1 verify / chunked + layer-major prefill,
  with the K22 tapes on **and** off; reduction asserted from the census
  (`tests/models/test_deepseek_v41_attn_win_memo.py`, 13 CPU tests). **Options ruled out (measured):**
  (a) a shapeless SDPA tape is dead — shapeless fails on the sink-drop dynamic slice + the value-einsum
  reshape (T≥20), and even where a padded-KV rewrite traces it does NOT reduce dispatches (the SDPA is
  2 einsums + softmax + masked where — reductions/matmuls compile can't fuse: 22→24); a sink-in-denom
  rewrite diverges even eager. (b) functional-KV append reduces no dispatch alone (only enables the dead
  a). (c) wq_a⊕wkv weight-concat is bit-exact for **quantized** residents (−1 matmul/layer, serving) but
  NOT dense (GEMM output-tiling ~1 ULP) — recorded for a quantized-gated add-on, not bundled. (d) the
  identical-shape shared compiled callable is **already** K22 (one qkv + one out tape across all layers).
  Realized GPU decode delta is **KG-j** (unmeasured). See `W45_ATTN_DISPATCH_REDUCTION.md`.

### K25 — Prefill score-path precision + split-K online softmax (`MTPLX_DSV41_PREFILL_SCORE_DTYPE` / `..._KEY_CHUNK`) — **W50 (prefill, sibling of K6)**
- **Mechanism:** W47's stage timing puts **`attn.*.score` at 201 s of the 370 s 16K TTFT**
  (reuse 153 + reindex 23 + full 19 + swa_only 6; layer-major, 60 GiB plan) — the single largest
  prefill term. head_dim **512 is not a fused-SDPA dim** (64/96/128/192/256), so the score path
  materializes the full `[rows,64,T]` f32 transient (§3.2 / K6). The port computes QK^T **and** PV with
  **both matmul inputs cast to f32** (`_sparse_attend`, `deepseek_v41.py`), matching the pure-f32
  torch oracle (`ref_forward._k_sparse_attn`, `torch.einsum(q.float(), kvg.float())`) — but at ~153 s
  the reuse score runs **≈7 TFLOPS, i.e. f32-matmul territory; the M5 Max does ~2× that in bf16**.
  K25 splits the lever in two, both **prefill-only** (gated `q.shape[1] > 1`; decode/M=1 always runs
  the shipped f32 one-shot, byte-identical):
  1. **`MTPLX_DSV41_PREFILL_SCORE_DTYPE=bf16`** — cast the QK^T / PV matmul **inputs** to bf16; the
     scale, mask, per-head sink and softmax stay f32 (as the oracle). MLX matmul **accumulates in f32
     and rounds the result back to bf16** — proven on CPU: a bf16 QK^T over K=512 equals
     `round_bf16(f32-accumulated)` *exactly* (`array_equal`), so the matmul error **== the bf16 output
     rounding** (0.2039 == 0.2039), not a K-growing accumulation error. Unset/`f32` inserts no casts
     (byte-identical to control).
  2. **`MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK=<n>`** (default off) — the gemma4 D512 two-pass split-K
     **online softmax** over `n`-wide key blocks (running max/denom/value with per-chunk rescale; the
     value-0 sink seeds `m=attn_sink, denom=1, acc=0`, so a fully-masked chunk never hits
     `-inf−(−inf)`). Caps the score transient at `[rows,64,n]` instead of `[rows,64,T]`.
- **FLOP / byte arithmetic (16K, layer-major, rows=1024/chunk):**
  - Score matmuls per call ≈ `4·rows·H·T·d` FLOP (QK^T `2·rows·H·T·d` + PV `2·rows·H·T·d`, d=512, H=64);
    summed over 17 chunks × 30 reuse layers ≈ the W47 **1.12 PFLOP** attention estimate. **bf16 2×** on
    the two matmuls (the compute-bound majority of the 201 s score term) → **est. −60 to −90 s off the
    score stages, TTFT 370 → ~290–310 s (−16 to −22 %)** — a GPU-window estimate (softmax/mask/dispatch
    stay f32, so not the full 2×). Byte cut per element: 4 B (f32) → 2 B (bf16) on the two matmuls' i/o.
  - Score **transient** at the last 16K chunk: one-shot f32 **6 GiB** (W47); bf16 one-shot ~half; split-K
    `n=2048` → `[1024,64,2048]·4 B ≈ 0.54 GB` (~8× smaller → protects the 100 GiB knob / admits a larger
    prefill chunk); **bf16 + n=2048 ≈ 0.27 GB**. Split-K is **compute-neutral** (same FLOP, extra
    per-chunk host dispatch) — its win is peak GB, possibly a small TTFT cost.
- **Exactness (CPU-proven, `tests/models/test_deepseek_v41_prefill_score_precision.py`, 16 tests):**
  - `f32` one-shot **byte-identical** to the pre-W50 inline formula (`array_equal`), and to control
    end-to-end (tiny-model logits `array_equal`).
  - **split-K f32 == one-shot f32 up to reassociation only**: scores are bit-identical (the QK^T reduces
    over head_dim, **not** the chunked T axis), so only the softmax denom + value sum reorder — real
    DSV4.1 shape (H=64, d=512, rows 128, T 4096) **max |Δ| = 6.1e-7 (rel 2e-6)**, greedy argmax identical
    on every row; NaN-free on a fully-masked chunk (collapses to the value-0 sink → 0).
  - **bf16 is LOSSY by design** (like head_bf16, K21): real-shape attention-output **max |Δ| ≈ 1.8–3.9e-3
    (rel 6e-3 to 1.1e-2)**, at bf16 rounding scale. Greedy tokens **can flip on near-ties** — the untrained
    tiny double flips intermediate-row argmax under bf16 while chunked-f32 does not — so exactness is
    **not** the ship bar; a task eval (HumanEval, [[deepseek-v4-quality-verdict]]) gates bf16, and the
    real deployment runs bf16 anyway (the f32 oracle is stricter than the reference's own precision).
- **STATUS (W50, `feat/deepseek-v41-w50`):** IMPLEMENTED + CPU-proven, all levers default OFF (f32 /
  one-shot). Peak RSS of the real-shape exactness test 1.62 GB (<3 GB). Arms `score_bf16`,
  `score_chunked` (`n=2048`), `score_bf16_chunked` in `ab_decode_env_levers.py`.
- **⚠ WINDOW-20 MEASURED — the K25 roofline was WRONG (the score stage is pass/bandwidth-bound, NOT
  matmul-FLOP-bound).** Integration `7a789731d`, 16,384-token prompt, layer-major, 60 GiB,
  `PREFILL_CHUNK=1024`, all tokens byte-identical (receipt `receipts/gpu-windows/window-20/
  prefill-16384-ladder.json`): `layer_major` baseline **TTFT 352.8 s** (46.4 tok/s, peak 76.6 GB);
  **`score_chunked` 418.8 s (−16 %), peak 65.1 GB (−11.5 GB)**; **`score_bf16` 535.1 s (−34 %)**, peak
  71.9 GB; `prefill_fast` (bf16+chunk+dense) 346.5 s (~baseline). The FLOP model predicted bf16 ≈ −2×
  on the matmul; instead it **regressed**. Root cause, from the code:
  - **bf16 (−34 %):** `_sparse_attend_oneshot` casts the `[rows,64,T]` QK^T output **bf16→f32** for the
    f32 softmax (one full T-wide pass, ~6.4 GB traffic at rows=1024/T=16K) **and** casts `w` **f32→bf16**
    before the PV matmul (a second T-wide pass) — two passes the f32 path never runs. The matmul I/O is
    halved to 2 B but the stage is bandwidth-bound (softmax stays f32), and MLX 0.32.2's bf16 GEMM at
    K=512 / large-N with the transpose flag does not beat f32 here — so the two cast passes are pure loss.
  - **split-K (−16 %, −11 GB):** the online-softmax **rescale of the running output** `acc = acc*corr + pv`
    is **O(rows·64·hd) = 2 passes over the 134 MB `acc` (rows=1024) EVERY chunk** (`⌈T/n⌉≈8` at n=2048),
    vs one-shot's single accumulation — plus `corr=exp(m−m_new)`/`denom` rescales and **≈8× the kernel
    dispatches** for qk/mask/softmax/pv. The −11 GB peak is real (transient `[rows,64,n]` not
    `[rows,64,T]`), so split-K is a **PEAK-GB lever, not a throughput lever**.
- **`lean` (W50, post-window-20): cut PASSES over the `[rows,64,T]` transient, not FLOPs** — the f32
  pass-cut one-shot (`MTPLX_DSV41_PREFILL_SCORE_PATH=lean`): (1) **scale `q` once** (`[rows,64,512]`)
  instead of the scores (`[rows,64,T]`) — one fewer T-wide pass; (2) **fold the value-0 sink into the
  denominator** (reference `_k_sparse_attn`) — no sink concatenate (`[rows,64,T+1]` alloc) and no
  post-softmax slice (`[rows,64,T]` copy), and the normalize divides the `[rows,64,512]` output not the
  T-wide `w`. Nets **three fewer passes / two fewer T-wide allocations** per score call.
  **Reassociation-level vs control** (real-shape max |Δ| ≈ 1.2e-6, **greedy-identical**), NOT
  bit-identical. Arms `score_lean` and `prefill_lean` (= `layer_major` + dense-experts + lean, the f32
  successor to `prefill_fast`). **SDPA is a dead end here:** `mx.fast.scaled_dot_product_attention`
  accepts head_dim 512 on CPU but has **no per-head value-0 sink** — plain SDPA differs from our
  semantics by 2.1e-3 (would change tokens); the 2×256 head-split is only a reassociated QK^T that still
  can't fold the sink into SDPA's fused softmax, so it buys nothing over `lean`.
- **Probe (W50):** `_sparse_attend*` now emit `stage_attn` sub-brackets (`attn.<mode>.score.{qk_matmul,
  cast,scale_mask_sink,softmax,online_softmax,pv_matmul,combine,out_proj}`) reported under a new
  `attn_breakdown` (kept OUT of the flat sum, never double-counting `attn.<mode>.score`; no-op / byte-
  identical off the prefill probe) so the next 16K window attributes the 153 s reuse-score term.
  **GPU gate remains KG-g.** See `W50_PREFILL_SCORE_PRECISION.md`.

---

### K26 — Prefill dequantize-once / dense-bf16 experts (`MTPLX_DSV41_PREFILL_DENSE_EXPERTS`) — **Rank (W51, the ALU-bound 16K prefill switch)**
- **Mechanism:** W47's 16K layer-major stage timing put `moe.routed_switch` at **105 s / 40
  layer-major calls** (~2.6 s/layer for ~98,304 routed rows = 16,384 tokens x top-6), of which the
  `switch_breakdown` attributes ~95 s to the `gather_qmm` compute itself (miss_submit 6.2 s, route_plan
  4.0 s) and ~22 s to the bank read (269 GiB once under layer-major @ 13 GB/s). The arithmetic:
  98,304 rows x 3 projections (gate/up 5120->2304, down 2304->5120) = **6.96 TFLOP/layer x 40 = 278
  TFLOP in 95 s ≈ 2.9 TFLOPS** — the mxfp4 gs32 `gather_qmm` at large M is **ALU/dequant-bound**
  ([[metal-sub4bit-alu-bound]]: dequant ALU/occupancy binds before bandwidth), while a dense bf16
  matmul on this box runs ~15-25 TFLOPS. K26 (env, default OFF) groups a prefill wave's rows by expert
  (the host-side `binding.buffer.bank_index`, **no device sync**); for every expert with ≥
  `MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS` rows (default 128) it **dequantizes gate/up/down from mxfp4 gs32
  to bf16 ONCE** (`mx.dequantize(weight, scales, group_size=32, bits=4, mode="mxfp4")` — E8M0 scales,
  no bias) and runs three **dense bf16 matmuls** over that expert's rows (ClampedSwiGLU unchanged),
  scattering back into router order; experts below the threshold keep `gather_qmm`. **Where:** prefill
  only — the switch gates it to `RoutingPhase.PREFILL` + `codec=="mxfp4"`, so the decode M=1 path is
  structurally excluded and byte-identical.
- **Cost model (16K layer-major, 384 experts, ~256 rows/expert avg so ~all clear 128):**
  - *dequant bf16 written/layer* = experts x 3 x 2304 x 5120 x 2 B ≈ 384 x **70.8 MB ≈ 27 GB/layer**;
    x40 ≈ **1.09 TB** across the prefill (the whole routed bank expanded from ~0.53 B/weight mxfp4 to
    2 B/weight bf16). DRAM round-trip @ 614 GB/s ≈ **1.8 s write + 1.8 s read ≈ 3.5 s total** (the 269
    GiB packed read is unchanged, paid by both paths).
  - *matmul FLOPs unchanged* at 278 TFLOP; at dense **15-25 TFLOPS → 11-19 s** (vs the measured 2.9
    TFLOPS / **95 s** gather).
  - *expected:* switch compute **95 s → ~15-22 s** (dense matmul + dequant DRAM), so `moe.routed_switch`
    **105 s → ~25-35 s** (bank read 22 s now co-binds) — **≈ 70-80 s off the 16K prefill**, i.e. the
    layer-major TTFT **370 s → ~295 s (44 → ~55 tok/s)**, contingent on the GPU-window measurement.
  - *transient peak* bounded by processing experts in batches of `MTPLX_DSV41_PREFILL_DENSE_BATCH`
    (default 8) with `mx.eval` between batches: ≤ 8 x 70.8 MB ≈ **0.57 GB** live, none kept across
    layers (well under the 60 GiB plan / 100 GiB knob, [[never-exceed-the-memory-knob]]).
- **Exactness:** NOT bit-identical to `gather_qmm` (fp32 matmul accumulation order differs; the dequant
  itself is lossless — FP4 code x 2^E8M0 lands exactly in bf16). CPU test on random mxfp4 gs32 weights
  (`tests/models/test_deepseek_v41_prefill_dense_experts.py`, hidden 256 / inter 128, 436 rows / 5
  experts) vs a **float64 reference**: `gather_qmm` max|Δ|=**9.40e-4** (rel 2.00e-2), dense bf16
  max|Δ|=**2.81e-4** (rel 5.96e-3) — the dense path is *closer* to fp64 (lossless dequant + fp32-
  accumulated dense matmul); dense-vs-gather max|Δ|=**1.22e-3** (rel 2.56e-2 to peak), the bf16
  accumulation-order divergence, David's documented FP class (cf. #171 vk_k split-K, the K16/K24
  compile divergences). Decode (M=1) and every below-threshold / small-M wave are a **byte-identical
  `gather_qmm` fall-through** (`mx.array_equal`, same test). This is the opposite of the "hand dequant
  kernel" dead entry below (§6): it uses **stock `mx.dequantize` + stock dense `mx.matmul`**, no hand
  kernel, and only at the large-M regime where dense wins — the W17 microbench (`window-17/
  gather-qmm-microbench.json`) confirms dense **LOSES** at M≤4 (bf16_dense 3197 µs vs mxfp4 982 µs @
  M=4, memory-bound reading 3.8x the bytes), which is exactly why the threshold + prefill gate exist.
- **STATUS (W51, `feat/deepseek-v41-w51`):** IMPLEMENTED + CPU-proven, default OFF.
  `expert_mlx.py`: `_run_component_bank_dense_prefill` + `_dequantize_mxfp4_slot`, threaded through
  `HotExpertSwitchGLU._dispatch_component_bank(dense_prefill=...)` from the PREFILL split wave only.
  Arm `prefill_dense_experts` (layer-major + dense on) in `scripts/deepseek_v41/ab_decode_env_levers.py`
  (dry-run extended).
- **MEASURED (GPU window 20, integration 7a789731d, 16K, layer-major, 60 GiB plan, chunk 1024):**
  `prefill_dense_experts` TTFT **332.2 s vs layer_major 352.8 s (−20 s, +6% prefill tok/s)**,
  token-sha identical, **same 76.6 GB peak** — far short of the modelled −70-80 s. The model assumed
  dense bf16 runs at 15-25 TFLOPS; the shortfall says it does not. Two suspects, and W20 rules out the
  first: **(a) threshold coverage** — from W24's routing census (`routing_census_1024.json`:
  `prefill_distinct_experts` mean 261/384 per layer at 1024 tok, so ~256 rows/expert avg at 16K) an
  occupancy estimate puts **~90-97% of rows already in ≥128-row experts** (equal-prob core ~0% below
  128; a skew-aware bracket with distinct@16K→280-384 gives 2.5-10% below 128, <32 captures >96%, <16
  >98%). So the threshold is NOT why the win is small — lowering it recovers a few % of rows at most.
  **(b) the bf16 dense matmul kernel** — W50 measured bf16 score matmuls **34% slower than f32** at 16K
  on this box; the dense gate/up/down likely pay the same slow bf16 path, eating the ALU win. The
  dequant is *not* the leak: `mx.dequantize(mode="mxfp4")` returns bf16 directly (no f32 intermediate),
  and K26 now dequantizes straight to the compute dtype (no cast, no doubled write).
- **W51 FOLLOW-UP (this branch):** (1) nested prefill brackets `switch.dense.{group_rows,dequant,
  matmul,scatter}` + `switch.gather_qmm_fallback` and per-layer tallies (`dense.rows_dense/rows_gather/
  experts_dense/experts_under_threshold`) exported into the W47 stage-timing receipt
  (`switch_breakdown` + new `switch_tallies`), so the next 16K window attributes the residual and
  confirms the coverage estimate directly. (2) tunable arms `dense_min32` (threshold 32), `dense_batch16`
  (batch 16), `dense_f32` (`MTPLX_DSV41_PREFILL_DENSE_MATMUL_DTYPE=f32` — f32 dequant + matmuls, the
  W50 hypothesis test). **Open GPU-window question (KG-k):** does `dense_f32` recover the modelled
  win, and do the counters confirm >90% dense coverage at threshold 128/32. See
  `W51_PREFILL_DENSE_EXPERTS.md`.

### K27 — Routed-gather row-sort → fused `gather_qmm_rhs` + shape/tiling audit (`MTPLX_DSV41_LAYOUT_FIX`) — **W56 (the shape defect the switch cost hides)**
- **Full audit:** `W56_SHAPE_TILING_AUDIT.md` (every hot op's operand shape/dtype/stride,
  the kernel mlx 0.32.2 selects on **`applegpu_g17s`** — arch gen 17, size `s`, NAX
  available — and whether N/K/M hit the aligned steel/gather/qmm tiles, all cited to
  `backend/metal/{matmul,quantized}.cpp` @ v0.32.2). Companion to K25 (score precision)
  and K26 (dense experts): those price precision/dequant; this prices **which kernel the
  shape selects and whether its tiles are ragged**.
- **The defect (F1):** `expert_mlx.py:_gather_component_bank` calls
  `mx.gather_qmm(x[rows,1,1,5120], w, s, rhs_indices=slot, transpose=True)` with **rows in
  router (token×top_k) order — unsorted by expert — and `sorted_indices` unset**. In
  `GatherQMM::eval_gpu` the fused **`gather_qmm_rhs`** kernel (streams each expert weight
  ONCE over its contiguous row block) fires ONLY when `M==1 && B≥16 && right_sorted_ &&
  B/E≥4`, and `right_sorted_ = sorted_indices && lhs_indices is None`
  (`quantized.cpp:1905`, `ops.cpp:5632`). With the flag unset, M=1 < the qmv batch limit 13
  routes to per-row **`gather_qmv`** — each of ~98 304 rows re-reads its expert's full mxfp4
  weight with no cross-row reuse. **This is the W47 "2.9 TFLOPS / 105 s" prefill switch.**
- **The fix (implemented, default OFF):** when `MTPLX_DSV41_LAYOUT_FIX=1` AND the wave has
  ≥ `MTPLX_DSV41_LAYOUT_FIX_MIN_ROWS` rows (default 2048 → decode 6 / verify 24 / small
  waves keep the **exact shipped unsorted call**), the gather sorts rows by bank slot on
  device (`mx.argsort`), runs gate/up/down with `sorted_indices=True` (→ `gather_qmm_rhs_nax`,
  `bm 32/64, bn/bk 64`, aligned since N,K %64==0), and unsorts the output. Tunable
  `MTPLX_DSV41_GATHER_ROWS_PER_CALL` (0=one call) chunks the sorted wave. A permutation +
  its inverse over an M-independent per-row matmul is **BYTE-IDENTICAL on CPU** (one
  gather_qmm impl); on Metal it swaps `gather_qmv → gather_qmm_rhs_nax`, a kernel
  reassociation in the **same documented FP class as K26** (measured in a window, not
  bit-identical there). CPU tests: `tests/models/test_deepseek_v41_w56_layout_fix.py`
  (15 cases, decode/verify/chunk-major/layer-major × mxfp4+affine, flag on==off,
  max|Δ|=0.0; rows-per-call 512/1000/4096 identical; default threshold leaves small waves
  untouched).
- **Estimate:** prefill switch **105 s → ~25–45 s** (fused streams the 269 GiB bank once
  vs per-row thrash); the exact figure is the GPU-window A/B. Biggest K27 lever.
- **Second-order shape facts (documented, microbench-armed, NOT implemented):**
  - **F2** — the expert **down**-proj K=inter=**2304** misses the mxfp4 fast `gather_qmv`
    (needs K%512==0; 2304%512=256), so ⅓ of the decode switch runs the slow ragged-K
    kernel; gate/up (K=5120) hit fast. Fix = pad K→2560 (zeros exact) **at bank-build
    time** — outside the mechanical-reorder allowlist; `down_align` microbench arm prices
    it before any bank change.
  - **F5** — verify M=4 dense projections want **bf16** (→ `gemv_wide`, weight streamed
    once for the 4 rows); f32 re-streams 4× (gemv_wide is fp16/bf16-only, `matmul.cpp:1332`).
    Keep verify acts bf16 (already K21).
- **Refuted (proven byte-identical no-ops on CPU, `mx.array_equal` max|Δ|=0.0):** F3 score
  QK^T/PV `mx.einsum` and F4 grouped o-LoRA einsum **already lower to the optimal,
  fully-tile-aligned batched GEMM** — an explicit-matmul rewrite is a no-op. The score cost
  is the 4.3 GB `[65536,T]` **output transient** (why bf16 *loses*: a cast pass, no FLOP
  relief at K=512), addressed by the W50 `lean` pass-cut / K6, not by re-tiling. Do not
  chase an einsum→matmul rewrite. F6: the K26 dense-prefill matmul is tile-aligned; its
  W51 shortfall is the W50 bf16-slow-vs-f32 kernel, not raggedness.
- **Microbench:** `scripts/deepseek_v41/shape_tiling_microbench.py` (CPU-safe
  `--help`/`--dry-run`; GPU in-window). Window command in `W56_SHAPE_TILING_AUDIT.md §5`;
  run it in window-22 with the flock held / qwen unloaded.
- **STATUS (W56, `feat/deepseek-v41-w56`):** IMPLEMENTED + CPU-proven, default OFF.
  Open GPU-window question (**KG-l**): does `MTPLX_DSV41_LAYOUT_FIX=1` route the 16K
  layer-major switch to `gather_qmm_rhs_nax` and cut TTFT, decode byte-identical.

#### K27 F2 — down-proj K pad 2304→2560 (`MTPLX_DSV41_DOWN_K_PAD`) — W56 follow-up
- **Defect:** expert down-proj K = `moe_intermediate_size` = 2304; `2304 % 512 = 256`,
  so the mxfp4 **fast** `gather_qmv` (needs `K % qmv_fast_k_alignment(4) = 512`,
  `quantized.cpp:1337,147-148`) is DISABLED for the down gather — gate/up (K=hidden 5120,
  %512==0) hit it. Down is ~⅓ of the ~74 ms/tok decode switch.
- **Fix (byte-identical):** lay the down slot out K=2560 with a **zero** tail (256 packed
  cols + 8 E8M0 scale bytes) and zero-pad the down-gather activation to 2560. A zeroed
  mxfp4 gs32 group dequantizes to **exactly 0.0** for scale byte 0/127 (2^e·0 = 0; only
  0xFF/NaN is unsafe, and a `mx.zeros` bank tail is scale byte 0) — **verified on CPU**
  (`test_deepseek_v41_w56_down_k_pad.py`, `mx.dequantize` tail all `== 0.0`, no NaN), and
  the padded gather is byte-identical (max|Δ|=0.0) at M=1 / M=4 / prefill.
- **Implemented (contained, in expert_mlx.py, default OFF):** `pad_mxfp4_down_component`
  (exact layout transform, packed+E8M0 zero tail), `down_k_pad_slot_bytes` (slot
  arithmetic), and `_gather_component_bank` pads the SwiGLU activation to the bank's
  actual down-K when `MTPLX_DSV41_DOWN_K_PAD=1` (a **no-op when the bank is unpadded** →
  byte-identical; the fast kernel engages only once the down bank itself is 2560-wide).
- **Slot arithmetic (computed, mxfp4 gs32, hidden 5120 / inter 2304):** down component
  `6,266,880 → 6,963,200 B` = **+696,320 B = +0.664 MiB, +11.1% on down** (2560/2304).
  Full 3-projection record `18,800,640 → 19,496,960 B` = **+3.70%/record**. (David's
  "+1.8 MB/record" overshoots — only the down K is sub-512; gate/up K=5120 already align.)
  Planner effect: `slots_per_layer` scales by `18800640/19496960 = 0.9643` → **−3.57%
  slots** at any per-layer bank budget (so ~−3.6% at both the 82 GiB and 60 GiB plans);
  the concrete per-plan counts come from `memory_plan.py` once the admission path lands.
- **BLOCKER (real streamed path — needs a decision, NOT in the w56 allowlist):** admission
  is a **zero-copy `os.preadv`** into per-segment slot memoryviews, and `expert_io.py:890`
  asserts `sum(component_views) == record.logical_bytes` (the on-disk 2304 width). A
  per-row-K-padded 2560 slot is **not byte-contiguous** with the 2304 record, so preadv
  cannot fill it — the padded down bank requires a **staging buffer + strided copy (+ a
  post-fill commit hook)** in `expert_io.py`/`expert_runtime.py` (shared by hy3/glm/a3b/
  laguna/qwen) OR an **offline bank re-pack** (pad the dense down cols to 2560 before
  quantize — exact, no runtime change). The plan/manifest sizing (`memory_plan.py`,
  `expert_manifest.py`) threads through whichever path is chosen. The contained core here
  (transform + activation pad + arithmetic + tests) is design-stable for either.
- **Arms:** `down_k_pad`, `layout_fix`, `k27_stack` in `ab_decode_env_levers.py` (pin all
  19 keys). **STATUS:** contained core IMPLEMENTED + CPU-proven, default OFF; real-path
  admission awaiting the shared-infra decision above.

### K28 — Fused mask + attention-sink softmax Metal kernel (`MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL`) — **W58 (prefill, sibling of K25/K6)**
- **Mechanism:** the 16K score path materialises the `[rows=1024,64,T]` f32 transient (up to
  ~6 GiB, T≈24576) and the eager softmax walks it repeatedly (window-22 reuse: `score.softmax`
  **84.5 s** one-shot / **~53 s** lean, `scale_mask_sink` 14.7 s). K28 fuses **mask + per-head
  value-0 sink + f32 softmax** into ONE `mx.fast.metal_kernel`: one threadgroup per `(row,head)`,
  `TG=256` lanes scan T strided, an **online (max,denom)** per-lane accumulation combines up a
  threadgroup **tree** (`M=max; D=d1·exp(m1-M)+d2·exp(m2-M)`; finite `NEG=-3e38f` sentinel so an
  all-masked lane is a true identity, never `-inf−(−inf)`), the per-head sink folds in once
  (`m=max(max(scores),sink)`), then a second pass writes `p=exp(s-m)/denom` (masked → 0). Reads the
  transient twice, writes once, **zero T-wide intermediates** (no `masked_scores`/`ex`/concat/slice).
  Prefill + one-shot only (composes with the lean path — scale folded into q, kernel `scale=1.0` —
  and plain one-shot; the split-K/chunked path is NOT routed through it, though a `normalize=False,
  return_stats=True` mode exposes per-`(row,head)` `(m,denom)` for a future split-K merge).
- **Byte / pass arithmetic (per chunk-layer, `S` = the transient, up to ~6 GiB):** eager one-shot
  mask+softmax ≈ **10 S** (scale·, where, concat, softmax, slice), eager lean ≈ **6 S** (where; max,
  exp, sum); **K28 = 3 S** (2R+1W). vs lean **6 S→3 S (−50 %)**; vs one-shot **10 S→3 S (−70 %)**;
  and the `masked_scores`/`ex` (up to ~2 S) allocations vanish (peak-GB relief for the 100 GiB knob).
  **Cost:** ~**3 `exp`/element** (online rescale ×2 + write ×1) vs eager's 1 — a memory-vs-ALU
  trade; net win requires the stage to be memory/pass-bound (**window-20's score-path finding**).
  Fallback if `exp`-ALU-binds: the **3-pass variant** (max, sum, write: 2 `exp`/element, 4 S) — a
  one-flag change to `_build_source`.
- **Est. seconds saved at 16K (GPU-window-gated, NOT measured):** `prefill_lean_k28` vs
  `prefill_lean` **−25…−31 s** (TTFT 265 → ~235–240 s, −10…−11 %) if memory-bound; `softmax_kernel`
  vs `control` up to ~−50…−65 s (capped by the 3× `exp`). Same roofline-class caveat as K25 (whose
  FLOP roofline window-20 overturned) — this is an estimate, the paired A/B is the gate.
- **Exactness: reassociation-level, NOT byte-identical** (tree reorders the sums; K28 writes the
  normalised `p` so PV is `p·V` vs lean's `(ex·V)/denom`). Same class as `score_chunked`/`score_lean`.
  **CPU-proven:** an f32 numpy simulation of the exact kernel algorithm vs the eager `mx` reference
  gives **max|Δ| = 1.86e-9, argmax exact**; fully-masked rows finite + all-zero (reference
  "all-invalid → zero output"). GPU parity (`test_fused_softmax_parity_gpu`, gated `MTPLX_GPU_PARITY=1`,
  receipt to `MTPLX_PARITY_RECEIPT`) confirms the Metal execution on `[64,64,4096]` + `[8,64,16384]`:
  **pass if max|Δ| ≤ 1e-6 and argmax mismatch 0**.
- **STATUS (W58, `feat/deepseek-v41-w58`):** IMPLEMENTED + CPU-proven (algorithm + wrapper plumbing
  + model dispatch + byte-identical CPU fallback), default OFF, GPU-only. Kernel
  `mtplx/kernels/dsv41_fused_softmax.py`; integration `_sparse_attend_oneshot`; tests
  `tests/models/test_deepseek_v41_fused_softmax.py` (19 CPU + 1 GPU-gated), peak RSS < 0.2 GB. Arms
  `softmax_kernel`, `prefill_lean_k28` in `ab_decode_env_levers.py` (pin all 20 keys). Report
  `W58_FUSED_SOFTMAX.md`. Numeric-throughput A/B pending a GPU window.

### K29 — Fused decode / verify MLA attention Metal kernel (`MTPLX_DSV41_DECODE_ATTN_KERNEL`) — **W60 (decode, sibling of K22/K24; the SDPA tail a tape cannot fold)**
- **Mechanism:** decode at 1K is dispatch-bound at ~160 ms/token, attention ~60 ms (`attn.reuse` 50 ms /
  30 layers ≈ 1.5–1.9 ms per M=1 layer) with ~110 Metal primitives/layer after K22/K24, while the M=1
  math is tiny (64 heads × head_dim 512 vs ~1K–16K keys). K22 folded the prep chains (qkv 33 + out 7 =
  40 prim/call) and K24 the window mask (~429 prim/tok @ 40 layers); **W45 proved the SDPA proper
  (`_sparse_attend`: two einsums + softmax + mask + sink-concat + slice) is an irreducible ~22-primitive
  reduction a shapeless `mx.compile` CANNOT fold** (the dynamic sink-slice `softmax(full)[...,:KV]`
  raises `Slice cannot infer output shapes`; verdict *"do not spend a GPU window on a shapeless SDPA
  tape"*). K29 is the viable alternative: ONE `mx.fast.metal_kernel` per layer does QK^T score + CSA/
  causal mask + per-head value-0 sink + f32 softmax + PV, **online-softmax over `TG`-wide key tiles**
  (one threadgroup per `(row,head)`, `TG=128`, one key/lane/tile) so the `[64,T]` per-head score row
  never materialises (T=16K+ streams through ~5.6 KiB threadgroup memory). Runtime `H`/`T`/`S`/`scale`
  scalars (one kernel for every length/batch/mode); `TG`/`HD=512` compile-time. Finite `NEG=-3e38f`
  sentinel → all-masked row is finite 0, never NaN. MLA: `k_cache`≡`v_cache` (one latent, RoPE
  pre-baked). MLX fused SDPA unusable (head_dim 512 unsupported; sink differs, W50 2.1e-3).
- **Dispatch (per layer, per call):** eager SDPA **~22 primitives** (W45) → K29 **~2–4** (1 kernel +
  bool→additive-mask `where` [+ q/KV contiguity, usually a decode no-op]). Mode-invariant (all four CSA
  modes route through the identical `_sparse_attend`). At 40 layers ≈ **−760 primitives/token** on the
  attention SDPA — larger than K22's whole-token −368 and K24's −429, and on the chain a tape can't
  touch. Decode-only lever: the M=1 score row is `[1,64,T]` (~4 MB @16K), so **no material peak-GB
  relief** (unlike K28's prefill ~6 GiB transient) — the win is dispatch count.
- **Est. ms/token saved at 1K (GPU-window-gated, NOT measured):** SDPA ≈ 14% of the ~162-prim reuse
  attention call (W45 micro-census) → ~8 ms/token dispatch-uniform ceiling; central estimate
  **−6…−12 ms/token** (160 → ~148–154, ~4–8%), minus the kernel's own single dispatch + ~70M-MAC/layer
  compute at T=1088. **Same roofline caveat as K22/K25/K28** (window-20 overturned K25's FLOP roofline)
  — an estimate; the paired in-window A/B is the gate. At 16K the reuse-layer kernel compute grows ~15×
  (still ONE dispatch): the dispatch win holds, the per-kernel compute rises, net is the window's.
- **Exactness: reassociation-level, NOT byte-identical** (online tile reduction reorders the max/denom/
  value sums; sink folded into the denominator vs the shipped `softmax(concat)+slice`). Same class as
  K25/K28. **CPU-proven:** the pure-MLX references (`decode_attention_reference` one-shot +
  `decode_attention_reference_tiled` — the EXACT online-tile algorithm) vs the model eager
  `_sparse_attend_oneshot`, over decode M=1 / verify M=4 × T∈{300,1088,4096} × every CSA mode, give
  **max|Δ| ≤ 4.3e-7, argmax exact**; fully-masked row finite + all-zero. GPU parity
  (`test_decode_attn_parity_gpu`, gated `MTPLX_GPU_PARITY=1`, receipt to `MTPLX_PARITY_RECEIPT`) confirms
  the Metal kernel on random cache states at **T∈{1088,4096,16384} × each mode** (decode M=1 + verify
  M=4) + 32 real-model decode steps if the artifact is present: **pass if max|Δ| ≤ 1e-6 and argmax
  mismatch 0**.
- **STATUS (W60, `feat/deepseek-v41-w60`):** IMPLEMENTED + CPU-proven (tiled-reference algorithm +
  wrapper plumbing + model dispatch for decode & verify across all four modes + unsupported-mask
  fallback + byte-identical CPU fallback), default OFF, GPU-only. Kernel + refs
  `mtplx/models/deepseek_v41_attn_kernels.py`; integration `_sparse_attend` → `_decode_attn_kernel`;
  tests `tests/models/test_deepseek_v41_decode_attn_kernel.py` (39 CPU + 1 GPU-gated), peak RSS < 0.2 GB.
  Arm `decode_attn_kernel` in `ab_decode_env_levers.py` (pins all 21 keys); **LEFT OUT of `stack_a`
  until the parity window is clean** (KG-m below). Report `W60_FUSED_DECODE_ATTENTION.md`.
  Numeric-throughput A/B pending a GPU window.

---

## 6. Dead-here (GPU-side; do not re-propose)

| Lever | Why dead |
|---|---|
| **Hand affine / IQ sub-4-bit dequant kernel** | loses to stock `gather_qmm`: [[iq2xxs-kernel-loses-to-stock]] 1.74× slower, 80 % decode ALU; [[metal-sub4bit-alu-bound]]. Use stock native mxfp4 gather (K7 measures it), never a hand kernel. |
| **Whole-forward `mx.compile`** | [[hy3-decode-roofline]] DEAD — async_eval already overlaps the graph rebuild; [[moe-exec-fusion-25-26-27]] fused-MoE slower than stock. (Per-*layer*-step compile is the open K5, not this.) |
| **Hand MLA fused-attention kernel** | V4: NEUTRAL-to-NEGATIVE — attention is not the binding term at latent 512, and its fp32 logits tipped near-ties (accept 2.72→2.64, **hurt K3**), [[deepseek-v4-kernel-verdicts]]. Keep default fused SDPA (K4). **DSV4.1 K29 is a DIFFERENT regime, not a re-propose:** V4's verdict was the *experts-resident* regime where attention was not the binding term; DSV4.1 decode is streaming/**dispatch-bound** (§0 reframe), so a hand kernel that removes ~22 SDPA dispatches/layer targets the actual bind. The V4 near-tie caution transfers to K29's VERIFY path (a ≤1e-6 reassociation could tip a spec-decode acceptance near-tie) — that is exactly what KG-m gates: argmax parity + the served A/B's acceptance/byte-identity record before `stack_a`. |
| **Dispatch-fusion as the *decode* lever _today_** | behind the SSD wall now (OPT_LEDGER R6). **Not dead after streaming** — that reframe is this file's §0; it is dead only as a *today* decode lever. |
| **Verify WIDTH / tree / K>3** | widens the expert union → more bytes AND more barriers; V4 K>3 dead. Cross-ref OPT_LEDGER §4. |
| **Top-k / sparse head to cut head compute** | greedy argmax needs the full 129280 logits; only the *byte* cut (K9) is valid, not a compute/candidate cut. |
| **Native FP4 KV as a decode kernel** | KV is 14–52 MB at 16K; quantizing frees <40 MB and risks CSA2 shared-cache argmax; OPT_LEDGER §4 / R10. |

---

## 7. Execution order (each gets a measured A/B at David's shape, one lever/arm, `mx.eval` counted)

All A/Bs use [[dsv41-standard-benchmark-shape]] (1K + 16K, greedy, single prompt), paired in-window
control ([[island-placement-beats-tuning]] window-drift law), GPU windows through the flock with qwen
unloaded + memory-guarded ([[hy3-benchmark-panic-protocol]], [[box-110gb-hard-limit]]), no CPU-heavy
worker in-window ([[cpu-heavy-work-voids-flock-windows]]). Report prefill/decode tok/s, TTFT, peak GB,
wall, byte-identical? These GPU gates **sequence *after* OPT_LEDGER Gates 0/1/D** (which decide the
SSD world) — GPU levers only pay once SSD time is exposed, except K16 (a today 16K-TTFT lever) and the
two microbenches (K7/K8), which are CPU/queued and can run now.

| Order | Lever(s) | Gate (pass condition) | When | Cost |
|---|---|---|---|---|
| **KG-a** | K7 mxfp4-vs-affine + K8 mxfp8-vs-affine-q8 | queued microbench, x `[rows,1,K]`, `mx.eval` counted; report M in {1, 4, 1271} ratios. Decides whether the native codecs carry a compute tax before any decode A/B trusts parity. | now (CPU/queued) | queued microbench |
| **KG-b** | K16 prefill expert-major read-once | 16K one-shot bank read A/B: chunked (~13×) vs layer-major (~1×). **Pass if 16K TTFT −≥40 %**, outputs byte-identical, peak under knob. | now (16K TTFT is a today problem) | 1 window |
| **KG-c** | K3 Sinkhorn kernel (+K12 fused mix) | *after* streaming exposes GPU time: AR + K3 vs AR. **Pass if decode +≥15 % AND argmax parity.** | after R1–R4 | 1 window |
| **KG-d** | K1 barrier overlap (+K14 buffer sweep) | K3-on baseline vs +overlap-fill; **`mx.eval` counted per arm** (a3b caution: sync-delete alone was −3.27 %). **Pass if decode +≥15 %.** | after KG-c | 1 window |
| **KG-e** | K10 verify batched M=4 (needs W23) + K11/R2 dedup | MTP+dedup with per-layer M=4 dispatch vs per-position M=1. **Pass if outputs byte-identical AND barrier count 40 (not 160)/cycle AND decode ≥ 2.0× AR.** | after W23 + OPT Gate 2 | 1 window |
| **KG-f** | K4 HC-compile + fused CSA attn | carry from V4; **argmax parity + decode +** (expect the smaller residual after K3). | after KG-c | folded |
| **KG-i** | K22 attention-chain compile (+ gate-prefix / combine folds) | `attn_compile` vs control (and folded into the K4 `all_levers` stack): **argmax parity (byte-identical decode/verify) + decode +**. CPU census −368 prim/tok (−40/attention call every mode); realized GPU decode delta is the open question. | after KG-c/KG-f | folded |
| **KG-j** | K24 window-mask memo | `attn_win_memo` vs control, and `stack_a` (head_bf16 + sinkhorn_metal + attn_compile + win_memo) vs stack without it: **byte-identical decode/verify + decode +**. CPU census −77 attn prim/tok on 8 layers (~11 × (n_layers−1); ~429/tok at 40 layers). | after KG-i | folded |
| **KG-m** | K29 fused decode/verify attention kernel | (1) parity `test_decode_attn_parity_gpu` (`MTPLX_GPU_PARITY=1`): **max\|Δ\| ≤ 1e-6 + argmax mismatch 0** on T∈{1088,4096,16384}×each mode (decode M=1 + verify M=4); THEN (2) `decode_attn_kernel` vs control 1K decode A/B: **decode + AND argmax parity** (record token-id sha256 — reassociation-level, so NOT byte-identical) AND, for the verify path, **no spec-decode acceptance regression** (the V4 near-tie caution). CPU census: eager SDPA ~22 prim/layer → ~2–4 (~−760/tok @40 layers). Add to `stack_a` only after this window is clean. | after KG-i/KG-j | 1 window |
| **KG-g** | K6 D512 two-pass SDPA + K17/K18 prefill tiling/chunk + K19 head-last-row | 16K prefill ± two-pass split-K + tiling + head-last-row on the K16 base. **Pass if TTFT −≥15 % beyond K16 AND peak −≥8 GB (K19 drops the 8.47 GB logits), parity on long-prompt A/B.** | after KG-b | 1 window |
| **KG-h** | K9 head byte cut + K13 in-place verify-KV | q8/mxfp8 head vs bf16 head: **ship only if argmax parity + HumanEval(164) ≥ exact−noise**; K13 footprint-guard folded into first MTP window. | after KG-e | folded |

**Sequencing rationale:** KG-a/KG-b run **now** (CPU/queued microbenches + the today 16K-TTFT
restructure). The decode kernel/fusion gates (KG-c…KG-f) only pay once OPT_LEDGER R1–R4 have exposed
GPU time — but §0 shows they are then **on the critical path to 20 tok/s** (the 40 ms barrier + the
6.4 k Sinkhorn dispatches cap AR near 20 by themselves), so they are scheduled immediately after the
SSD stack clears its central estimate, not deferred as refinements. K3 before K1 because the Sinkhorn
kernel both removes the biggest dispatch source and shrinks the GPU-idle window K1 then fills.

---

## 8. Open unknowns that gate these numbers (GPU side)

- **mxfp4 native gather speed vs affine at M in {1, 4, 1271}** — K7/KG-a. Unmeasured ([[mlx-native-float-quant-modes]]).
- **mxfp8 resident GEMV speed vs affine-q8 at M=1** — K8/KG-a. W18 has a precision note, no speed note.
- **Real per-layer barrier p50 under the served path** — cited ~1 ms from `expert_mlx.py:1959`; confirm
  in-window and whether the shared-hoist / async_eval overlap actually recovers it (a3b backpressure
  caution).
- **Sinkhorn kernel decode delta in the SSD-exposed regime** — V4 measured +29.3 % with experts
  resident; confirm it transfers once DSV4.1 decode is dispatch-bound (it should — same regime).
- **K16 realized bank-read multiplier at 16K** — is it the full ~13× (cache holds ≪384/layer) or less
  (routing skew leaves some experts cold)? Depends on the OPT_LEDGER Gate-1 concentration census.
- **D512 two-pass SDPA parity** — split-K reassociation vs the greedy argmax gate (near-tie risk).
