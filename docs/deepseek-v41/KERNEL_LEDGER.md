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
- **No whole-forward `mx.eval`** in `_forward_span` (decode path); the only forced evals are the
  per-layer streamed-switch barrier and (in chunked prefill) `_eval_cache_state` per chunk.

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

### K4 — HC-compile + fused CSA attention (V4 dispatch stack) — **Rank 5**
- **Mechanism:** V4's HC-tape collapse + fused CSA attention: **AR +31.3 % (17.37→22.80), K3 +17.4 %,
  −26.1 % dispatches** ([[deepseek-v4-kernel-verdicts]]). **Where:** decode + prefill. **After:** the
  same post-streaming dispatch-bound regime → material; **now:** ~0. **Exactness:** parity-gated (V4
  bit-identical). **Effort:** low (V4 reuse). **⚠** keep the *default fused* SDPA — do **not** chase a
  hand MLA kernel (K-dead below). **Precedent:** measured on V4.

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

---

## 6. Dead-here (GPU-side; do not re-propose)

| Lever | Why dead |
|---|---|
| **Hand affine / IQ sub-4-bit dequant kernel** | loses to stock `gather_qmm`: [[iq2xxs-kernel-loses-to-stock]] 1.74× slower, 80 % decode ALU; [[metal-sub4bit-alu-bound]]. Use stock native mxfp4 gather (K7 measures it), never a hand kernel. |
| **Whole-forward `mx.compile`** | [[hy3-decode-roofline]] DEAD — async_eval already overlaps the graph rebuild; [[moe-exec-fusion-25-26-27]] fused-MoE slower than stock. (Per-*layer*-step compile is the open K5, not this.) |
| **Hand MLA fused-attention kernel** | V4: NEUTRAL-to-NEGATIVE — attention is not the binding term at latent 512, and its fp32 logits tipped near-ties (accept 2.72→2.64, **hurt K3**), [[deepseek-v4-kernel-verdicts]]. Keep default fused SDPA (K4). |
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
