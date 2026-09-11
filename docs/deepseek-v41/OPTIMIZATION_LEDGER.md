# DeepSeek-V4.1-Flash streaming — Optimization Ledger

Status: living document (worker feat/deepseek-v41-w25). Analysis only; no code lands from this file.
Author: Opus 4.8 worker. Every candidate is priced against the DSV4.1 cost model in §1. Ranked in §3.
Dead-here levers (do not re-propose) in §4. New model-specific ideas in §5. 48-hour execution order in §6.

David's directive (verbatim): "Optimize this until you are completely out of ideas, then come up with
new ones. Use all our previous optimizations for qwen 3.8 125b and other models as inspiration."

Goal: decode **> 20 tok/s** at the 1,024- and 16,384-token prompt shapes
([[dsv41-standard-benchmark-shape]]: mtplx.prefill_bench Python programming ladder, single prompt,
greedy, no batching), MTP on (DSpark 3-stage, ported by W23), KV minimal, expert cache maximal, under
the 100 GB box / 100 GiB wired knob.

Today (mxfp4 bank, AR): prefill 37 tok/s, TTFT 27 s, decode 4.8 tok/s.

---

## 1. Cost model (the physics everything is priced against)

### 1.1 Weights and formats (from [[dsv41-native-mxfp4-bank]], [[deepseek-v41-streaming-artifact]], PORT_PLAN §0/§3)

| Item | Value |
|---|---|
| Layers (text) | 40 (20 causal encoder + 20 decoder); + 3 DSpark MTP stages |
| Routed experts | 384 per layer, **top-6** (+1 resident shared expert) |
| Routed bank format | **native mxfp4 gs32**, bit-exact repack of source E2M1+E8M0 (cos 1.000000, top-6 agreement 31/31) |
| Routed record | **18.80 MB/expert-weight-set** (17.93 MiB); bank 40×384 = **269 GiB** |
| Residents (text) | dense q8 gs64 backbone + router/norms/hc/attn_sink/compressor/indexer ≈ **10.65 GiB** |
| MTP residents (when on) | DSpark 3×128 experts q8 gs32 + heads ≈ **6.7 GiB** |
| Engram tables | L1+L14, FP8 rows, 272 B/row, 24 rows/token/layer, **~189–195 GiB on SSD** (disk-backed, LRU) |
| KV (bf16 global, phase-1) | ~52 MB at 16 K + fixed 5 MiB/layer SWA window (native FP4 KV = 890 B/tok, later) |
| SSD read bandwidth | **12.5 GiB/s = 13.42 GB/s** (M5 Max, [[test-machines-bandwidth-file]]) |

### 1.2 Memory budget (coordinator-calibrated, this is the binding constraint)

Box ceiling **100 GB total**, shared by all workers + resident agent + the GPU step. During a GPU
window the resident Qwen agent is booted out, leaving the **GPU planner budget ≈ 82 GiB**
([[box-110gb-hard-limit]], [[never-exceed-the-memory-knob]]). Allocation:

```
82 GiB  planner budget (agent booted out)
 −10.65 text-only residents (q8 backbone)
 − 6.70 MTP residents (DSpark, when MTP on)
 − 0.05 KV bf16 @16K + SWA windows
 − ~7    runtime reserve (transient slots, IO staging, workspace)
 = ~57–65 GiB  expert cache   →  ≈ 3,400–3,600 mxfp4 slots  ≈  85–90 experts/layer of 384
```

- **MTP-on residency ≈ 85–90 / 384 = 22–23 %.** AR-only frees the 6.7 GiB MTP residents → ~100/layer ≈ 26–27 % (this is "today's ~30 % cached").
- **The MTP residents cost ~15 experts/layer of cache.** MTP must pay for that shrink and then some.
- Islands (all 40 layers pinned) need 269 GiB — impossible; streaming is mandatory.

### 1.3 The decode bottleneck: SSD bytes/token (100 % bandwidth-bound)

Per decoded token, AR: `40 layers × 6 experts × 18.80 MB = 4.51 GB` of routed records.
At ~23–30 % resident hit rate, **~3.0–3.5 GB/token stream off SSD**:

```
decode_tok_s ≈ SSD_BW / bytes_off_ssd_per_token = 13.42 GB/s / 3.0 GB = 4.5 tok/s   (matches today's 4.8)
```

This is **not** dispatch-bound (unlike Qwen3.8 Flash-Next, which was 5,415 disp/cycle, 80 % zero-byte
— [[qwen38-flash-next-80tps-program]]). DSV4.1 decode is squarely **SSD-bandwidth-bound**: the expert
records are 40× larger per activation and must cross the SSD, not just VRAM. This inverts which levers
matter (dispatch fusion is nearly worthless here; bytes-off-SSD is everything).

### 1.4 The 20 tok/s target, decomposed

20 tok/s = 50 ms/token → SSD budget = `50 ms × 13.42 GB/s = 0.67 GB/token` off SSD.
From 3.0 GB → 0.67 GB is a **~4.5× cut in SSD bytes per accepted token.** No single lever delivers
4.5×. It is the **product** of:

| Factor | Mechanism | Plausible range | Source precedent |
|---|---|---|---|
| **A. Acceptance (MTP)** | DSpark drafts 3, verify accepts L≈1+α+α²+α³ tokens/cycle | ×2.5–3.0 tokens/cycle @ α 0.7–0.8 | [[spec-decode-cycle-anatomy]], [[deepseek-v4-mtplx-port]] (K3 = 5.7× on V4) |
| **B. Verify-row record dedup** | 4 verify positions route top-6 each; read the **union** once, not 4× | union u: 24→~6–10/layer via routing overlap | Qwen3.8: expert dedup across 4 MTP rows = **−0.9…−1.3 s/cycle** ([[qwen38-flash-next-80tps-program]]) |
| **C. Routing-census residency** | pin the most-frequently-routed experts (not uniform) | hit 23 %→55–65 % if routing is power-law | [[island-placement-beats-tuning]] (cost per-miss-SERVICE), [[glm52-q1t-lane-state]] freq-alloc |
| **D. Prefetch overlap** | draft tokens known before verify → prefetch their expert rows, pipeline SSD under compute | recovers the compute fraction (~+10–20 %) only if SSD is not already saturated | [[a3b-decode-roundtrip-is-the-lever]] (prefetch KILLED at M1 when bound), [[hy3-c5-dense-islands]] |

Worked point (the 20 tok/s existence proof): MTP L≈2.73 (α 0.75), dedup union u≈6/layer (near-perfect
overlap), census hit ≈0.60:
```
SSD records/cycle = 40 × 6 × (1 − 0.60) = 96 records → 96 × 18.80 MB = 1.80 GB
time = 1.80 / 13.42 = 0.134 s ;  throughput = 2.73 / 0.134 = 20.3 tok/s
```
Sensitivity: at u≈8 and hit 0.55 → 13.5 tok/s; at u≈6 and hit 0.65 → 23 tok/s. **The three levers
must land together;** any one alone falls short. This is why the execution order (§6) measures A, B, C
independently but ships them stacked.

**Corollary — MTP is not free here.** Its 6.7 GiB of residents lower hit rate ~4 pts (~−0.6 GB
headroom → ~15 fewer experts/layer). MTP wins only because (A×B) on the accepted tokens outruns the
residency it costs. This tension must be measured, not assumed (§6 Gate 1).

---

## 2. How to read the ledger

Each candidate: **Name · Source · What it did there + measured gain · Applies to DSV4.1? (why/why not)
· Expected gain here (from §1 cost model) · Effort · Exactness risk · Dependencies · Rank.**
"Exactness" = whether it perturbs the streamed==resident argmax gate (PORT_PLAN P1.7). Ranks are
1 = do first. Dead-here levers are quarantined in §4 so nobody re-proposes them.

---

## 3. Ranked optimization ledger (live candidates)

### R1 — DSpark 3-stage MTP (acceptance amortization) — Factor A

- **Source:** [[spec-decode-cycle-anatomy]]; [[deepseek-v4-mtplx-port]] (V4 DSpark K3 = 25.86 tok/s,
  **5.7×** over AR, "bf16-acts carries it"); [[gemma4-dflash-cycle-program]]; native DSpark head
  (model.py DSpark classes), ported by W23.
- **There:** on a memory-bandwidth-bound decode, drafting K and verifying in one forward amortizes the
  per-token weight read across all accepted tokens. V4: K3 = 5.7× AR (bandwidth-bound identical shape).
  K>3 was dead on V4 (acceptance decay).
- **Applies:** YES — this is the primary lever. DSV4.1 decode is *even more* bandwidth-bound (SSD, not
  VRAM), so amortizing the 3 GB/token read across ~2.7 accepted tokens is the largest single win.
- **Expected here:** ×2.5–3.0 tokens/cycle (L≈1+α+α²+α³). **Caveat:** the verify forward reads experts
  for K+1 positions; without dedup (R2) that inflates the read to ~2× per cycle, netting only ~1.4×.
  MTP's value is realized only paired with R2 + R3.
- **Effort:** high (W23 is porting; this ledger prices it). **Exactness:** lossless if greedy verify
  (verified tokens == pure AR argmax; PORT_PLAN P3.0 done-when). **Deps:** MTP residents (6.7 GiB),
  greedy verify, R2, R3.
- **Rank: 1** (necessary; not sufficient alone).

### R2 — Expert-record dedup across MTP verify rows — Factor B

- **Source:** [[qwen38-flash-next-80tps-program]]: "Expert dedup across the 4 rows ≈ **−0.9 to −1.3 s**
  (needs routing-overlap census)"; MoE routed = 4.45 s (31.5 %) of the Qwen3.8 verify cycle.
- **There:** Qwen3.8's verify processes 4 rows (anchor + 3 MTP); each row routes independently, so
  experts are read up to 4×. Reading the **union** once cut 0.9–1.3 s off a 14 s cycle (~7–9 %).
- **Applies:** YES and **much bigger here.** On Qwen3.8 the experts are resident (VRAM), so dedup saved
  compute/dispatch. On DSV4.1 the experts cross the **SSD** at 18.80 MB each, so dedup saves *SSD
  bytes* — the binding resource. A distinct record fetched once serves every verify position that
  routes to it. This is the multiplier that turns MTP from ~1.4× into ~2.7×.
- **Expected here:** collapses per-layer distinct reads from 4×top-6 = up-to-24 down to the union u.
  Governed by routing overlap of consecutive draft tokens (they share local context → high overlap).
  If u≈6–8 (vs 24 naive), that is a **2.5–4×** reduction in verify-cycle SSD reads. Needs a
  routing-overlap **census** on real prompts to fix u.
- **Effort:** medium (dedup the per-cycle expert gather set before issuing preadv; the streaming
  `expert_io` layer already keys by record id — dedup is a set-union at the gather call site).
  **Exactness:** none (same records, read once). **Deps:** R1 (only exists with MTP), routing census.
- **Rank: 2** (the lever that makes MTP pay on a streaming bank).

### R3 — Routing-census-driven residency (hot-expert pinning) — Factor C

- **Source:** [[island-placement-beats-tuning]] ("30 tok/s @ 96 GiB by swapping *which* layers are
  islands; cost is per-miss-SERVICE"); [[glm52-q1t-lane-state]] frequency-allocation A/B (incomplete
  but the mechanism is proven direction); [[dwarfstar4-borrowable-techniques]] (DS4 validates
  freq-alloc; MTLResidencySet).
- **There:** which experts/layers you keep resident matters more than any tuning; residency should be
  allocated by *service frequency* (routing mass), not uniformly. GLM/DwarfStar freq-alloc pins the
  hot rows.
- **Applies:** YES — directly. The cache holds only ~85–90/384 experts/layer (§1.2). If routing is
  power-law (typical for DeepSeekMoE top-6 with noaux_tc bias), pinning the top ~23 % by census
  captures far more than 23 % of activations.
- **Expected here:** if the resident 23 % of experts capture 55–65 % of routing mass, hit rate
  23 %→55–65 %, i.e. **miss 77 %→35–45 %** ⇒ SSD bytes/token 3.0 GB → 1.6–2.0 GB (×1.6–1.9 alone; the
  decisive multiplier when stacked with R1×R2). **Must measure the routing histogram** — if routing is
  near-uniform (as Hy3 was, [[hy3-c5-dense-islands]]), this lever collapses to the uniform 23 %.
- **Effort:** medium (census pass over the standard prompts → per-(layer,expert) counts → static pin
  list → feed as `island_pin_order` / resident overlay). **Exactness:** none (cache policy only).
  **Deps:** a routing census on David's 1K/16K shapes; interacts with R2 (census also fixes u).
- **Rank: 3.**

### R4 — Prefill gather batching / dedup across chunks (TTFT lever)

- **Source:** [[deepseek-v4-longcontext-prefill]] (O(N²)→OOM root-caused to dense-attn + CSA indexer
  materialising full scores); [[qwen38-longctx-prefill-memory]]; the streaming `expert_io` preadv path.
- **There:** long-context prefill is dominated by expert reads and score materialization; chunking is
  required. Batching the expert gather across chunk positions dedups the record reads.
- **Applies:** YES — TTFT is 27 s at 16K. Prefill routes experts for all 16,384 positions; the union of
  distinct (layer,expert) records is bounded by 40×384 = 15,360 records = the whole 269 GiB bank read
  **once at most**. 269 GiB / 13.42 GB/s = 20.5 s floor if every expert is touched once; today's 27 s
  ⇒ we are already near the read-once floor but pay a resident/overlap tax. Batching the gather so each
  record is read exactly once across all chunks (not once per chunk) is the lever.
- **Expected here:** TTFT floor ≈ (fraction of bank touched) × 269 GiB / 13.42 GB/s. If 16K tokens
  touch ~all experts, floor ~20 s; the win is eliminating re-reads of the same record across chunks and
  overlapping SSD with attention compute. Target TTFT ~18–22 s (from 27 s), ~**1.2–1.5×**.
- **Effort:** medium. **Exactness:** none. **Deps:** prefill chunking already required; R3 residency
  reduces the touched-off-SSD fraction.
- **Rank: 4** (prefill, not decode — but TTFT is a David-visible metric).

### R5 — Engram/draft-token expert prefetch (latency-hiding) — Factor D

- **Source:** [[a3b-decode-roundtrip-is-the-lever]] (prefetch KILLED at M1 when already bandwidth-bound;
  works only where a read window exists); [[hy3-q4-overlap-lever-dead]] (~15 ms dispatchable in an
  ~89 ms read window, but bytes/depth remain); Engram n-gram reader ([[deepseek-v41-streaming-artifact]]).
- **There:** prefetch/overlap wins only when compute can hide under an SSD read window and SSD is not
  saturated. On A3B/Hy3 it was dead because the lane was already bandwidth-bound.
- **Applies:** PARTIAL. DSpark proposes the draft tokens *before* the verify forward, so the exact
  expert set the verify needs is knowable one step early → issue preadv for those records during the
  draft/attention compute. This hides SSD latency under compute, but since decode is bandwidth-bound it
  only recovers the *compute fraction* of the cycle (attention + engram + router), not the read time.
- **Expected here:** modest **+10–20 %** on top of R1–R3, and it de-risks the tail (ensures the union is
  read once, contiguously). Do **not** overclaim: if the SSD is saturated by R1×R2×R3 reads, prefetch
  adds nothing (a3b lesson). Value is pipelining, not byte reduction.
- **Effort:** medium. **Exactness:** none. **Deps:** R1 (draft tokens), R2 (dedup set), preadv path.
- **Rank: 5.**

### R6 — Sinkhorn hyper-connection Metal kernel (from V4)

- **Source:** [[deepseek-v4-kernel-verdicts]]: Sinkhorn kernel **SHIPS, +29.3 % AR**, best K3 32.5;
  MLA neutral-to-negative. PORT_PLAN §2 carries `_sinkhorn_metal_kernel` (1206), `hc_split_sinkhorn`,
  `HyperConnection` as-is (V4.1 hc math identical: hc_mult 4, iters 20, eps 1e-6).
- **There:** the Sinkhorn normalizer for hyper-connections was a large AR win on V4 (+29.3 %) because
  the naive Sinkhorn iteration was a dispatch/latency hog; the fused Metal kernel removed it.
- **Applies:** YES for the **compute** fraction of the cycle (hyper-connections run every layer,
  resident). It does **not** touch SSD bytes, so on a bandwidth-bound decode its ceiling is the compute
  fraction only — but that fraction is exactly what R5 tries to hide under the read window, so removing
  it shrinks the cycle floor. On prefill (compute-heavier) it helps more.
- **Expected here:** on the bandwidth-bound decode, gain is bounded by the hc compute share of the
  non-SSD time (small, maybe +2–5 % decode); larger on prefill/TTFT. Carry it because it is a proven,
  low-risk V4 kernel already in the skeleton, and every ms of compute floor removed is a ms R5 need not
  hide.
- **Effort:** low (port existing V4 kernel). **Exactness:** kernel is a normalizer — gate argmax
  parity. **Deps:** V4 code reuse (PORT_PLAN §2).
- **Rank: 6.**

### R7 — In-place verify-KV update (avoid the copy) — from Qwen3.8 256K fix

- **Source:** [[qwen38-verify-band-sdpa-dead-band]] / qwen38-256k-fix: verify KV write via
  `mx.slice_update` copied 267 MB × k,v × 12 layers (6.4 GB) per verify — the 256K OOM root cause; the
  in-place fix removed it.
- **There:** the MTP verify band re-materialized KV instead of writing in place; huge transient copies.
- **Applies:** YES as a **correctness/footprint** guard for the DSV4.1 MTP verify. DSV4.1's CSA2 shares
  KV across 36 layers (only 4 Full layers own KV), so a naive verify-KV copy touches those 4 caches —
  smaller than Qwen3.8's 12, but the SWA windows (5 MiB/layer × 40) and the compressed caches are still
  copy-prone. Every transient GB spent copying is a GB stolen from the expert cache (§1.2).
- **Expected here:** not a tok/s lever per se; a **footprint** lever that protects the 60–65 GiB expert
  cache from verify-time transient spikes (which would collapse throughput ~4× if they breach the knob,
  [[never-exceed-the-memory-knob]]). Prevents a regression rather than adding speed.
- **Effort:** low–medium. **Exactness:** must preserve KV bytes exactly. **Deps:** MTP verify path,
  CSA2 shared cache.
- **Rank: 7** (guard rail; do before the first MTP GPU window).

### R8 — Native FP4 global KV (size, for long context only)

- **Source:** PORT_PLAN §3b/P2.0: native FP4 global KV = 890 B/tok vs bf16 3200 B/tok; David's stated
  KV preference is bf16 (avoids the FP4 KV kernel).
- **Applies:** marginally. At 16K, global KV is ~14.6 MB (FP4) vs ~52 MB (bf16) — a 37 MB delta against
  a 60–65 GiB expert cache = **<0.06 %** more experts resident. See §4 (KV quant is dead as a *decode*
  lever). Only matters at 256K+ context.
- **Expected here:** ~0 decode. **Rank: 12** (deprioritized; not for the 1K/16K shapes).

*(Rows R9–R11 and the DeepSeek-V4 dispatch levers, Laguna/Gemma kernel verdicts, and moe-exec-fusion
verdicts are being priced in the next commit; placeholder so the ranking spine is committed first.)*

---

## 4. Dead-here (do NOT re-propose) — with the reason

| Lever | Why it's dead for DSV4.1 |
|---|---|
| **Q2 gs64 routed bank** | Quality-broken: cos 0.912 vs FP4 source, router top-6 agreement 11/31 at L2, scattered junk output (" potentially"/"Kasipak"), torch-reference confirmed ([[dsv41-native-mxfp4-bank]], [[deepseek-v41-streaming-artifact]]). Q2-from-FP4 is a *second* lossy step from a 4-bit source (PORT_PLAN §7). It is 158 GiB (1.7× less SSD/token) but it does not work. Superseded by native mxfp4 gs32 (bit-exact). Do not resurrect for the byte win. |
| **Islands on the mxfp4 bank** | Full-island (pin all 40 layers) = 269 GiB, cannot fit under 82 GiB. Partial islanding *is* R3 (routing-census residency) — use that framing. And a hand sub-4-bit dequant kernel loses to stock `gather_qmm` ([[metal-sub4bit-alu-bound]], [[iq2xxs-kernel-loses-to-stock]] 1.74× slower), so there is no compute win from "islanding" 4-bit experts either. |
| **KV quantization (as a decode lever)** | At 16K the entire global KV is ~14–52 MB; quantizing frees <40 MB of a 60–65 GiB cache = unmeasurable decode gain, while risking the CSA2 shared-cache argmax exactness (4 caches feed 36 layers). Keep bf16 KV for the 1K/16K shapes; native FP4 is a *256K-context size* item only (R8). |
| **mxfp4 bank lossless compression** | [[hy3-lossless-compression-c7]]: Q2 experts compress 1.29× (order-0 saturated), **Q4 near-incompressible**. mxfp4 E2M1 codes are high-entropy 4-bit → expect ~1.0× (no meaningful shrink). The C7 bank-compression lever is dead on this format. (Re-confirm the actual entropy on a bank sample before fully closing — cheap check.) |
| **Resident n-gram/expert arena (static prefetch table)** | [[qwen38-flash-next-80tps-program]]: decode rows were 85–93 % novel → a resident n-gram arena was *falsified*. Engram here is disk-backed by design; do not try to make it resident to "prefetch." |
| **Prefetch when SSD-saturated** | [[a3b-decode-roundtrip-is-the-lever]]: prefetch KILLED at M1 once bandwidth-bound; [[hy3-c5-dense-islands]]: near-uniform routing → prefetch dead. Prefetch (R5) only recovers the *compute* fraction; it does not add bandwidth. Do not price it as a byte reduction. |
| **Dispatch-fusion / compile-the-forward as the primary decode lever** | [[hy3-decode-roofline]]: compile-the-forward DEAD; [[moe-exec-fusion-25-26-27]]: fused MoE *slower* than stock, #25's 1.85× was ~75 % an accounting artifact (count `mx.eval` per arm). DSV4.1 decode is SSD-bandwidth-bound, not dispatch-bound (§1.3) — the Qwen3.8 dispatch story does **not** transfer. Fusion helps only the small compute floor. |

---

## 5. NEW ideas (specific to DSV4.1 structure)

> These exploit structure that Qwen3.8/Hy3/V4 did not have. Each needs a census/measurement gate before
> it is trusted; none is assumed.

1. **Verify-row routing-overlap census → fixed union `u`.** (Underpins R2.) Measure, on David's 1K/16K
   prompts with DSpark drafts, the distribution of the per-layer union of top-6 sets across the 4 verify
   positions. If consecutive draft tokens route to near-identical experts (local-context hypothesis),
   `u`→6–8 and MTP pays. If they diverge (`u`→18–24), MTP on a streaming bank is a trap. **This single
   census decides whether >20 tok/s is reachable at all** — run it first (§6 Gate 0).

2. **Engram-hash-driven expert prefetch.** Engram is an n-gram memory keyed by a rolling hash of recent
   tokens. DSpark proposes the draft tokens before verify → hash them → (a) prefetch the **engram rows**
   the verify will need (12 KiB/token, latency-bound random reads — pure latency win) and (b) as a
   *predictor* of the routed-expert set for those positions, warm the expert preadv. Novel: use the
   n-gram memory as a routing predictor for prefetch, not just as a residual contributor.

3. **Routing-census static residency by service frequency (R3), refreshed per-prompt-class.** DeepSeek
   noaux_tc gate has a learned per-expert bias → routing is expected power-law. Pin the top-k by census.
   Extension: David's shape is a *programming* prompt class; a class-specific pin list may beat a global
   one (different domains light up different experts).

4. **CSA2 shared-KV attention is nearly free — spend the saved budget on expert cache.** 36 of 40 layers
   reuse 4 owned caches (PORT_PLAN §0). Attention KV traffic is therefore tiny vs a normal 40-layer
   model. Confirm the attention compute/latency is negligible so the whole non-SSD budget and the read
   window (R5) can be dedicated to expert streaming.

5. **DSpark drafts from the final hidden state → the draft's own routing is a free expert-prefetch
   oracle.** The 3 DSpark stages produce token proposals *and* run their own (resident q8) experts. The
   routing those resident MTP experts choose correlates with the routing the full model will choose for
   the same positions → use the MTP routing as the prefetch key for the big-bank verify reads (cheaper
   than a separate predictor).

6. **mxfp4 record de-duplication at the bank level (cross-layer expert clustering).** 384×40 = 15,360
   expert weight-sets; if some experts are near-duplicates across layers (or within), a content-hash
   pass could share physical records (read once, alias many). Cheap to test (hash the 15,360 records);
   likely low hit rate but free if any duplicates exist. Distinct from R2 (which dedups within a cycle);
   this dedups the *bank storage* and raises effective cache residency.

7. **Bounded SWA "replay" to shrink resident window footprint** (PORT_PLAN P2.1): reconstruct the
   128-wide SWA window from the long-lived global KV instead of storing 5 MiB/layer × 40 = 200 MiB
   resident — small, but every MB is an expert slot. Low priority.

8. **Prefill: read the bank exactly once, streamed in routing order.** (Underpins R4.) At 16K prefill,
   order the chunk processing so each of the 15,360 records is fetched once, in a schedule that overlaps
   SSD reads with attention/CSA2 compute → TTFT approaches the 269 GiB/13.42 GB/s ≈ 20 s read-once floor.

---

## 6. Proposed 48-hour execution order (one A/B per lever, David's shape, receipts)

All gates use [[dsv41-standard-benchmark-shape]] (1K + 16K prompt, greedy, single prompt); every arm is
seed/window-suffixed and append-only ([[never-overwrite-a-measurement]]); one A/B per lever; report
prefill tok/s, decode tok/s, TTFT, peak GB, wall, and whether outputs are identical. GPU windows go
through the flock ([[gpu-work-always-through-flock]]); no CPU-heavy worker during a window
([[cpu-heavy-work-voids-flock-windows]]).

| Order | Lever | Measurement gate (pass condition) | Cost |
|---|---|---|---|
| **Gate 0** | Verify-row overlap census (New #1) | CPU-only census: per-layer union `u` across 4 verify positions on 1K/16K. **Pass if median u ≤ 10.** If u > 14, MTP-on-streaming is re-scoped. | CPU, no GPU |
| **Gate 1** | R3 routing-census residency | CPU census → static pin list; then 1 GPU A/B: uniform-cache vs census-pinned. **Pass if decode +≥40 %** (hit 23→≥45 %). Confirms routing is power-law, not uniform. | 1 GPU window |
| **Gate 2** | R1×R2 MTP + dedup (needs W23) | AR control vs MTP+dedup, greedy verify. **Pass if outputs identical AND decode ≥ 2.2× AR.** If dedup off shows <1.5×, R2 is the gap. | 1 GPU window |
| **Gate 3** | R1×R2×R3 stacked | census-pinned + MTP+dedup vs AR baseline. **Target decode ≥ 20 tok/s @ 1K and 16K, outputs identical to AR.** | 1 GPU window |
| **Gate 4** | R5 prefetch overlap | Gate-3 stack ± draft-token expert prefetch. **Pass if +≥10 %** and no exactness change; drop if ≤ noise (a3b lesson). | 1 GPU window |
| **Gate 5** | R4 prefill batching | ± cross-chunk gather dedup at 16K. **Pass if TTFT −≥15 %** (27 s → ≤ 23 s). | 1 GPU window |
| **Gate 6** | R6 Sinkhorn kernel, R7 in-place verify-KV | carry from V4; verify argmax parity + footprint stays under knob. Ship if parity holds. | folded into above windows |

Sequencing rationale: Gate 0 is CPU-only and decides feasibility before any GPU time is spent. R3
(Gate 1) is measurable *without* MTP (W23 not required) and de-risks the residency assumption early.
Gates 2–3 need W23's DSpark port. Prefetch/prefill (4–5) are refinements once the stack clears 20.

---

## 7. Open unknowns that gate the numbers

- **Routing skew** (power-law vs uniform) — decides R3. Unknown until the census. Hy3 was near-uniform
  ([[hy3-c5-dense-islands]]); DeepSeek noaux_tc bias suggests skew, but unmeasured.
- **Verify-row overlap `u`** — decides R2/MTP viability. Unknown until Gate 0.
- **DSpark acceptance α** on David's programming shape — decides L. V4 got 5.7× at K3; DSV4.1 α
  unmeasured. Owned by W23.
- **mxfp4 native gather kernel speed vs affine** — [[dsv41-native-mxfp4-bank]] flags it UNMEASURED;
  sub-8-bit gathers are ALU-bound ([[metal-sub4bit-alu-bound]]). Affects the compute floor, not SSD.
- **Effective SSD bandwidth under the real gather pattern** — 12.5 GiB/s is the sequential ceiling;
  random preadv over 18.80 MB records may under-deliver ([[mmap-willneed-unwired]]: SSD is a
  threshold). Coalesced/ordered reads (New #8) matter.
