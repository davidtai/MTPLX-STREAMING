# DeepSeek-V4.1-Flash streaming — Optimization Ledger

Status: living document (worker feat/deepseek-v41-w25). Analysis only; no code lands from this file.
Author: Opus 4.8 worker. Every candidate is priced against the DSV4.1 cost model in §1. Live candidates
ranked in §3; dead-here levers (do not re-propose) in §4; new model-specific ideas in §5; 48-hour
execution order in §6; measurement-method guardrails (why past deltas were wrong) in §7.

David's directive (verbatim): "Optimize this until you are completely out of ideas, then come up with
new ones. Use all our previous optimizations for qwen 3.8 125b and other models as inspiration."

Goal: decode **> 20 tok/s** at the 1,024- and 16,384-token prompt shapes
([[dsv41-standard-benchmark-shape]]: `mtplx.prefill_bench` Python programming ladder, single prompt,
greedy, no batching), MTP on (DSpark 3-stage, ported by W23), KV minimal, expert cache maximal, under
the 100 GB box / 100 GiB wired knob.

Today (native mxfp4 bank, AR): prefill 37 tok/s, TTFT 27 s, **decode 4.8 tok/s**.

---

## 1. Cost model (the physics everything is priced against)

### 1.1 Weights and formats (from W16/W18/W19 reports, [[dsv41-native-mxfp4-bank]], PORT_PLAN §0/§3)

| Item | Value |
|---|---|
| Layers (text) | 40 (20 causal encoder + 20 decoder); + 3 DSpark MTP stages |
| Routed experts | 384 per layer, **top-6** (+1 resident shared expert) |
| Routed bank | **native mxfp4 gs32**, lossless repack of source E2M1+E8M0 (cos 1.000000, top-6 agreement 31/31); W16 COMPLETE, 288,777,830,400 B = **269 GiB**, admitted & served |
| Routed record | **18.80 MB / expert** (all 3 weights w1/w2/w3; = 288.78 GB ÷ 40 ÷ 384) |
| Residents (text, W18 mxfp8 exact repack) | dense backbone + router/norms/hc/attn_sink/compressor/indexer ≈ **10.65 GiB** |
| MTP residents (when on, W18) | DSpark dense mxfp8 + 3×128 experts mxfp4 gs32 + heads ≈ **6.7 GiB** |
| Engram tables | L1+L14 mxfp8 (W19), **264 B/row**, 24 rows/token/layer, 768 M rows, ~189 GiB on SSD (disk-backed LRU, `NGramRowCache`) |
| KV (bf16 global, phase-1) | ~52 MB at 16 K + fixed 5 MiB/layer SWA window (native FP4 KV = 890 B/tok, later) |
| SSD read bandwidth | device ceiling **12.5 GiB/s = 13.42 GB/s**; single-stream *realized* can be far lower (§1.5) |

### 1.2 Memory budget (coordinator-calibrated — the binding constraint)

Box ceiling **100 GB total**, shared by all workers + resident agent + the GPU step. During a GPU
window the resident Qwen agent is booted out → **GPU planner budget ≈ 82 GiB**
([[box-110gb-hard-limit]], [[never-exceed-the-memory-knob]]):

```
82 GiB  planner budget (agent booted out)
 −10.65 text-only residents (mxfp8 backbone)
 − 6.70 MTP residents (DSpark, when MTP on)
 − 0.05 KV bf16 @16K + SWA windows
 − ~7    runtime reserve (transient slots, IO staging, workspace)
 = ~57–65 GiB  expert cache  →  ≈ 3,400–3,600 mxfp4 slots  ≈  85–90 experts/layer of 384
```

- **MTP-on residency ≈ 85–90 / 384 = 22–23 %.** AR-only frees the 6.7 GiB MTP residents → ~100/layer ≈ 26–27 % (≈ "today's 30 % cached").
- **MTP residents cost ~15 experts/layer of cache** (~4 pts of hit rate). MTP must earn that back.
- Islands (all 40 layers pinned) = 269 GiB — impossible under 82 GiB. Streaming is mandatory.

### 1.3 The decode bottleneck: SSD bytes per **accepted** token

AR, per token: `40 layers × 6 experts × 18.80 MB = 4.51 GB` of routed records. At ~23–30 % resident
hit rate, **~3.0–3.5 GB/token stream off SSD**:
```
decode_tok_s ≈ SSD_BW / bytes_off_ssd_per_token = 13.42 GB/s / 3.0 GB ≈ 4.5 tok/s   (matches today's 4.8)
```
This is **SSD-bandwidth/service-bound**, unlike Qwen3.8 Flash-Next which was *dispatch*-bound (5,415
disp/cycle, 80 % zero-byte — [[qwen38-flash-next-80tps-program]]). The expert records are ~40× larger
per activation and cross the SSD, not just VRAM. **This inverts which levers matter**: dispatch fusion
is nearly worthless on DSV4.1 decode; bytes-off-SSD and service-count are everything. The closest
prior analog is **GLM-5.2 q1t streaming** (SSD-bound at 12.47 GiB/s, [[glm52-q1t-lane-state]]) — its
measured numbers, not Qwen3.8's, transfer here.

### 1.4 The 20 tok/s target, decomposed

20 tok/s = 50 ms/token → SSD budget = `50 ms × 13.42 GB/s = 0.67 GB` off SSD **per accepted token**.
From ~3.0 GB → 0.67 GB is a **~4.5× cut in SSD bytes per accepted token.** No single lever delivers
4.5×. It is the **product** of four measured mechanisms — plus one hard warning:

| Factor | Mechanism | Plausible range here | Precedent (measured) |
|---|---|---|---|
| **A. Acceptance (MTP)** | DSpark drafts 3; accept L≈1+α+α²+α³ tokens/cycle | ×2.5–3.0 tokens/cycle @ α 0.7–0.8 | V4 K3 = 5.7× AR ([[deepseek-v4-mtplx-port]]); [[spec-decode-cycle-anatomy]] |
| **B. Verify-row record dedup** | 4 verify positions route top-6 each; read the **union** once | union u/layer: 24→~6–10 by routing overlap | Qwen3.8 dedup across 4 MTP rows = **−0.9…−1.3 s/cycle** ([[qwen38-flash-next-80tps-program]]) |
| **C. Frequency residency** | pin hot experts within each layer | miss cut **~20–28 %** (Belady ceiling on hy3), more only if DSV4.1 routing is more concentrated | held-out Belady beat LRU 20–28 % @80 GiB; "8×" was oracle-leak, deployable ≈0 % on hy3 ([[mmap-willneed-unwired]], [[glm52-q1t-lane-state]]) |
| **D. Realized-BW / queue depth** | issue the whole per-layer / union read set concurrently + `io_read_fanout` | realized 47 %→70 %+ of ceiling if under-saturated | GLM d3 realized 8.8 vs AR 5.9 GiB/s; pread 8-thread = 12.9 ([[mmap-willneed-unwired]]) |

**⚠ MTP is not automatically a win on a streaming bank.** GLM measured it going *backwards on bytes*:
AR 135 misses/tok (I/O ceiling 11.2 tps) → **d3 widened the union to 180 misses/tok (ceiling dropped
to 8.4 tps)**. d3 still netted +11 % *only* because it raised realized SSD utilization (Factor D),
not because it cut bytes. On DSV4.1, **MTP pays only if Factor B (dedup) collapses the union back
toward AR's per-token read while Factor A multiplies the accepted tokens.** Without dedup, expect a
GLM-shaped small win or a regression. The whole program hinges on the verify-row overlap census (§5 #1,
§6 Gate 0).

Worked existence-proof for >20 (all four stacked): α 0.78 → L≈2.85; dedup union u≈6/layer (near-perfect
overlap); frequency residency lifts hit 23 %→~40 % (within Belady's reach) → miss 0.60; realized BW at
~90 % of ceiling (12 GB/s):
```
SSD records/cycle = 40 × 6 × 0.60 = 144 → 144 × 18.80 MB = 2.71 GB
time = 2.71 / 12.0 = 0.226 s ;  throughput = 2.85 / 0.226 = 12.6 tok/s
```
That is **short of 20** at these (conservative) assumptions. To clear 20 needs either **u→~4–5**
(exceptional overlap — implies consecutive draft tokens route almost identically) **and/or** hit→~55 %
(routing far more concentrated than hy3) **and/or** a quality-neutral acceptance rule pushing α→~0.9
(R-acc, L→~3.4). **Honest read: 20 tok/s is at the optimistic edge of the stacked model; ~10–14 is the
central estimate.** The census gates (Gate 0/1) decide which world we are in before any GPU burn.

### 1.5 Realized bandwidth caveat (do not assume 12.5)

12.5 GiB/s is the *device* ceiling. Single-stream decode is serially layer-dependent (layer N's route
needs layer N−1's output), so if reads are issued naively the drive sees **~1 outstanding request →
~5.2 GiB/s realized** ([[mmap-willneed-unwired]]: one 10 MiB request draws ~5.2 GiB/s; SSD plateaus
12.2–12.5 only at queue depth ≥64, regresses at 384). Within a layer the 6 (AR) or up-to-24 (verify
union) experts *are* known simultaneously → issue them concurrently; split each 18.8 MB record with
`io_read_fanout`. **Measure today's realized decode BW first** (§6 Gate D): if it is <8 GiB/s, Factor D
alone is worth ~1.5×; if already ~12, only byte-cutting (A×B×C) moves decode.

---

## 2. How to read the ledger

Each candidate: **Name · Source · What it did there + measured gain · Applies to DSV4.1? (why/why not)
· Expected gain here (from §1) · Effort · Exactness risk · Dependencies · Rank.** "Exactness" = whether
it perturbs the streamed==resident argmax gate (PORT_PLAN P1.7). Rank 1 = do first. Two lever *classes*:
**SSD-byte/service levers** (move the binding constraint — R1–R5) and **compute-floor levers** (R6+;
help prefill/TTFT and lower the floor that Factor D hides under, but are *behind the SSD wall* on
decode). Do not confuse the two — that is the mistake the Qwen3.8 dispatch story invites.

---

## 3. Ranked optimization ledger (live candidates)

### R1 — DSpark 3-stage MTP (acceptance amortization) — Factor A

- **Source:** [[deepseek-v4-mtplx-port]] (V4 DSpark K3 = 25.86 tok/s, **5.7×** AR; "bf16-acts carries
  it"; **K>3 dead**, d5 accept 0/86); [[spec-decode-cycle-anatomy]] (verify = ~1.2× a 1-token forward
  for *resident* weights — but "K+1 rows can hit up to (K+1)·top_k experts", the MoE caveat that on a
  *streamed* bank is the whole cost); native DSpark head ported by W23.
- **There:** on a bandwidth-bound decode, drafting K and verifying in one forward amortizes the weight
  read across accepted tokens. V4 (experts resident): 5.7× at K3.
- **Applies:** YES — the primary lever, but **conditional** (see §1.4 warning). V4's experts were
  resident, so verify's extra rows were nearly free; here each extra distinct expert is an 18.8 MB SSD
  read. The union-widening (GLM: 135→180 misses/tok) can eat the acceptance gain unless R2 fires.
- **Expected here:** ×2.5–3.0 accepted tokens/cycle **iff** R2 holds u small; otherwise a GLM-shaped
  +10–15 % or a net loss. Keep K=3 (V4 proved K>3 dead; depth widens the union — doubly bad here).
- **Effort:** high (W23 owns the port; this ledger prices it). **Exactness:** lossless with greedy
  verify (verified == pure-AR argmax; PORT_PLAN P3.0). **Deps:** R2 (mandatory pairing), R4/D, MTP
  residents (−6.7 GiB cache).
- **Rank: 1** (necessary; not sufficient; dangerous without R2).
- **W57 (DSpark-DIRECT lane, code landed):** a lean self-contained loop
  (`mtplx/models/deepseek_v41_dspark_decode.py`, served via `--generation-mode dspark` /
  `MTPLX_DSV41_DSPARK_DIRECT=1`) drives W23's drafter through draft → K+1-row verify → greedy/spec
  accept → `trim_verified_window_to_prefix` (no re-forward; V4.1 cache all-trimmable), **bypassing** the
  generic native-MTP machinery that window-21 measured *net-negative* (served MTP 2.45–2.94 vs AR
  2.2–5.0). Greedy == AR byte-for-byte over 256 tokens; sampled K=0 == `generate_ar` exactly (14 CPU
  tests). Reuses the V4 K3 loop's accept/verify/rollback structure + `mtplx.cache_state` primitives (not
  the Metal Mia K5 engine). **Also fixed a blocking W23 drafter regression** (`DSparkAttention` missing
  `self.mode` broke every draft-block forward → the W23 greedy gate was failing on this branch). Cost
  model in `W57_DSPARK_DIRECT.md`: this lane removes the generic machinery's *overhead* but NOT the
  §1.4/R2 bytes wall — the dispatch-bound `T_{K+1} ≈ T1` win still needs a resident bank or R2 dedup, and
  the tok/s number is gated by α (still unmeasured on the box, §8 lever L).
- **W57 window-24 (integration eff795f75, 1K/256):** byte-identical, α superb (3.78 tokens/cycle, 189/195
  accepted, 94/98/98% by depth), but **1.53 tok/s** — root-caused to TWO issues (W57_DSPARK_DIRECT.md §6b):
  (1) the K+1-row verify defaulted to `RoutingPhase.PREFILL` (`current_expert_routing_phase`: token_count>1)
  on the bench's direct `model(...)` call, paying the prefill wave every cycle (~2.47 s/cycle) — FIXED by
  wrapping the verify in `attention_phase("decode_verify")` + `expert_routing_phase(DECODE)` (byte-identical;
  default on; `MTPLX_DSV41_DSPARK_VERIFY_DECODE_PHASE=0` to A/B); (2) the AR reference itself was 1.7 tok/s
  (vs 6.24 baseline) because `with_mtp=True` + the harness reprice shrink the resident expert cache ~14 GiB
  → far more per-miss SERVICE — this is the head's intrinsic cache cost (R2/R3 territory), quantified by the
  new `--with-mtp` AR A/B. Per-cycle cost table (draft/verify/accept/commit ms) + the W37 verify-internal
  switch census now land in the receipt (`--decode-mode dspark [--stage-timing]`).
- **W57 window-25 (integration f622f91f6, 1K/256; W57_DSPARK_DIRECT.md §6c):** DECODE-phase verify still
  1.68 s/cycle. **attn ~830 ms/verify** = the rows>1 verify took the prefill attention path because K29
  (fused decode/verify attn, b·s≤8) + K30 (selected keys) were OFF — FIXED: the lane now arms both for the
  whole dspark run (`arm_dspark_decode_kernels`; harness `setdefault`, served wraps its cycles; AR ref armed
  too so the greedy-identical kernels stay consistent; `MTPLX_DSV41_DSPARK_DECODE_KERNELS=0` opts out).
  **switch ~630 ms/verify** = `route_stage` census pins it to host-sync barriers (`hot.eval_indices` ~10 ms
  each per-layer device→host routing-index sync, `hot.allhit_fence_eval` ~3.6 ms) — the M=4 all-hit/split
  path evaluates indices per split not once per 4-row wave; PROPOSED (streamed-runtime, GPU-only, not shipped
  from a CPU worker): keep verify routing indices on-device / batch the M=4 index eval to one barrier/layer.
  **AR-with-head 2.8×**: main_hidden capture identical, MTP experts resident (not streamed) — so it is the
  reprice shrinking the COLD expert cache ~7 GiB (peak 76.3≈75.9, i.e. slots traded for MTP residents);
  `--with-mtp --no-reprice` added so window 26 separates budget/slots from a code path.
- **W57 window-27 (integration b582b73a3; W57_DSPARK_DIRECT.md §6d):** verify still 1.88 s/cycle even with
  K29 engaged, and `verify_stage_timing` was EMPTY — the W37 probe recorded only s==1 forwards, so the
  4-row verify was invisible. FIXED: `enter_forward` (decode) now records 1≤s≤8 (the K+1 verify), and the
  route-stage probe + W61 engagement counter (`hot.verify_single_barrier`) are surfaced in the receipt.
  Full audit table of every rows>1 branch in the verify (attn score-path vs K29, streamed switch M=4
  host-sync barriers vs W61 all-hit, prefill-softmax kernel, HC/attn compile caps, one-shot vs chunked,
  K19, engram) in §6d. Draft 261 ms explained: 3 SHALLOW stages (not 3×40) over 4 resident rows + markov
  loop. Added `MTPLX_DSV41_DSPARK_VERIFY_K29=0` (K30 on, K29 off) so window 29 isolates the fused kernel.

### R2 — Expert-record dedup across MTP verify rows — Factor B

- **Source:** [[qwen38-flash-next-80tps-program]] "Expert dedup across the 4 rows ≈ **−0.9 to −1.3 s**"
  (MoE routed = 4.45 s / 31.5 % of the 14 s Qwen3.8 verify cycle); [[spec-decode-cycle-anatomy]] MoE
  caveat; [[glm52-q1t-lane-state]] (d3 union widened 135→180 misses/tok *because dedup was not applied*).
- **There:** Qwen3.8's 4-row verify read experts up to 4×; the union-once saved 0.9–1.3 s/cycle (~7–9 %).
- **Applies:** YES and **decisively bigger here.** On Qwen3.8/GLM the experts were VRAM-resident, so
  dedup saved compute/dispatch; on DSV4.1 dedup saves **SSD bytes** — the binding resource — and is what
  keeps MTP from regressing (§1.4). A distinct record fetched once serves every verify position that
  routes to it, and also cuts *service count* (the per-miss overhead lever, [[island-placement-beats-tuning]]).
- **Expected here:** collapses per-layer distinct reads from ≤24 toward the union u; if u≈6–8 (vs 24
  naive) that is a **2.5–4× cut** in verify-cycle SSD reads — the multiplier that converts R1 from
  ~1.4× into ~2.7×. Ceiling set by measured routing overlap of consecutive draft tokens.
- **Effort:** medium (union the per-cycle expert-id set before issuing preadv; `expert_io` already keys
  by record id — this is a set-union at the gather call site + a shared transient slot). **Exactness:**
  none (identical records, read once). **Deps:** R1; the overlap census (§5 #1).
- **Rank: 2** (the lever that makes MTP pay on a streaming bank).

### R3 — Frequency-census residency (hot-expert pinning within a layer) — Factor C

- **Source:** [[mmap-willneed-unwired]] (oracle freq-alloc 86.5 vs LRU 685 MiB/tok = 8×, **but** the
  held-out gate REFUTED it: `trained_dynamic_quota_lru` 70.78 ≈ `uniform` 70.72; deployable ≈ 0 %;
  cross-layer coverage stdev 0.027; held-out Belady beats LRU only **20–28 %** @80 GiB);
  [[island-placement-beats-tuning]] (+2.8 % paired by swapping *which* layers island, zero memory;
  cost is per-miss-*service*); [[glm52-q1t-lane-state]] (frequency slot-allocation still unmeasured on
  GLM's own traffic; run the held-out gate first).
- **There:** the *oracle* 8× was evaluation leakage; the honest deployable ceiling is Belady's ~20–28 %
  miss cut, and *per-layer slot allocation* captured ~0 % on hy3 because every layer concentrated
  equally. The surviving axis is *within-layer* admission (pin the hot experts, don't just LRU-evict).
- **Applies:** MAYBE — must be measured, not assumed. DeepSeekMoE's `noaux_tc` learned per-expert bias
  suggests skew, but the analog (hy3) failed its held-out gate. The realistic gain is a within-layer
  frequency-admission policy reaching toward Belady.
- **Expected here:** if a CPU held-out census on David's 1K/16K prompts shows real, stable concentration,
  **~20–28 % miss reduction** (miss 77 %→~55–60 %; bytes 3.0→~2.2–2.4 GB; ×~1.25–1.35 decode). If
  concentration ≈ hy3's (0.2 %), **~0 %** — do not ship. **This is a hypothesis with a failing analog;
  gate it on CPU before GPU.**
- **Effort:** medium (census pass → per-(layer,expert) counts → held-out train/eval split → frequency
  admission policy in the cache). **Exactness:** none (cache policy). **Deps:** CPU held-out census.
- **Rank: 3** (measure first; may be dead).
- **R3-pin update (W64, `feat/deepseek-v41-w64`, CPU-verified):** the within-prompt hit-rate arm stays
  dead (W24: static top-N pin ≈ LRU within one prompt). But the *within-layer pinned working set* is now
  landed for a **different** payoff — it is the **slot-stability fix W44 needs**. Behind
  `MTPLX_DSV41_PIN_WORKING_SET` (default off, **byte-identical when off**), after prefill it ranks each
  layer's resident experts by prefill routing frequency and pins the top-K (or all, the `pin_ws` arm),
  marking those slots **never-recyclable on normal decode admission** (a memory-forced capacity eviction
  may still evict, then unpins). Pinning is pure cache policy → **byte-identical on/off** (served expert ==
  requested expert regardless of slot; CPU test models the physical bank and proves it). Telemetry
  (`pin_working_set` in snapshot / A/B receipt / served event): pinned count per layer + **all-pinned-hit
  rate** = the fraction of decode layer-routes a barrier-free device route could take race-free. **This
  does not raise the all-hit rate; it makes the existing all-hit layers SAFE for W44's barrier-free gather**
  — 0 % today (W44 window-19 garbage on any churning layer) → up to the census all-hit fraction (~0.61
  all-hit layer-calls cold, → 0.835 warm) with `pin_ws`. Device route NOT re-enabled here; §6 of
  `W64_PINNED_WORKING_SET.md` specifies how it must consume `route_all_pinned`/`pinned_static`
  (out-of-band pin at the prefill→decode boundary + device-side pinned-mask gate). Whether removing the
  barrier on that fraction nets tok/s is the GPU A/B (`pin_ws` + pinning-guarded `device_route`).

### R4 — Realized-BW / queue-depth saturation (`io_read_fanout`, concurrent issue) — Factor D

- **Source:** [[mmap-willneed-unwired]] (single-stream decode ≈ 1 outstanding request → 5.15 of 12.5
  GiB/s; pread saturates 12.9 at 8 threads; plateau qd 64, regress at 384; the only depth available is
  *intra-record fanout* since next-layer experts are unknowable); GLM realized 47 %→70 % under d3.
- **There:** the gap between realized and ceiling BW is a queue-depth problem, not per-read efficiency;
  `io_read_fanout` splits one record across concurrent chunk readers; issuing all known reads at once
  raises depth.
- **Applies:** YES if DSV4.1 decode is under-saturating the SSD (likely for AR's serial layer chain;
  MTP verify's up-to-24 concurrent expert reads per layer naturally raise depth — part of why GLM d3
  realized more BW). Bytes unchanged; utilization up.
- **Expected here:** if today's realized decode BW is ~5–8 GiB/s, **up to ~1.5×** toward 12.5 for free;
  if already ~12, ~0. **Measure realized BW first (Gate D).** Pairs naturally with R1/R2 (the verify
  union is a ready-made concurrent read set).
- **Effort:** low–medium (`io_read_fanout` knob + issue-order change; no new kernels). **Exactness:**
  none. **Deps:** measured realized BW; interacts with R2 (dedup'd union = the concurrent batch).
- **Rank: 4.**

### R5 — Prefill gather batching / read-bank-once (TTFT lever)

- **Source:** [[deepseek-v4-longcontext-prefill]] (query-chunk BOTH O(N²) ops + **per-block `mx.eval`**
  is load-bearing — chunking without eval builds one graph and doesn't bound peak; **block-shared
  top-512 gather = +11 % prefill**, byte-exact >1024; MLA `head_dim=512` is NOT a flash-SDPA shape so
  attention/indexer fall back to full score materialization); [[qwen38-longctx-prefill-memory]].
- **There:** long-context prefill blows up on dense-attn + indexer score materialization (V4: 143 GB
  single alloc at 32k); chunk + per-block eval bounds it; block-shared gather cut re-reads (+11 %).
- **Applies:** YES — TTFT is 27 s at 16K. Prefill touches (up to) all 15,360 records once = 269 GiB /
  13.42 = **~20 s read-once floor**; today's 27 s pays a re-read + overlap tax. Batch the expert gather
  so each record is read **once across all chunks**, ordered to overlap SSD with attention/CSA2 compute.
- **Expected here:** TTFT toward the ~20 s floor + overlap → target **~18–22 s (from 27), ~1.2–1.5×**.
  Also inherits V4's chunk+eval memory discipline (protects the knob at 16K).
- **Effort:** medium. **Exactness:** `_attend` chunk is bit-exact; the top-K/indexer chunk is a
  quality-gated near-tie (validate long-prompt ON/OFF A/B, [[deepseek-v4-longcontext-prefill]]).
  **Deps:** prefill chunking (already required); R3 residency lowers the off-SSD fraction.
- **Rank: 5** (prefill/TTFT, not decode — but David-visible).

### R6 — V4 dispatch-reduction stack (HC-compile + fused attn + Sinkhorn kernel) — compute floor

- **Source:** [[deepseek-v4-kernel-verdicts]] / docs/perf/deepseek-v4-dispatch-levers.md: HC-tape
  collapse + fused CSA attn = **AR +31.3 % (17.37→22.80), K3 +17.4 %**, −26.1 % dispatches; Sinkhorn
  4×4/20-iter recurrence as one Metal dispatch (6,794→86 dispatches) = **AR +29.3 % (→28.86, clears 27),
  K3 →32.50**. Host encode was exposed (~2.9 µs/dispatch, 56–59 ms vs 32–35 ms GPU). hc math identical
  in V4.1 (hc_mult 4, iters 20, eps 1e-6; PORT_PLAN §2 carries the classes as-is).
- **There:** on V4 (experts *resident*, so decode was host-encode/dispatch-bound) these were the top
  decode wins. **MLA fused-attention kernel was NEUTRAL-to-NEGATIVE** (attention is not the binding term
  at 512 latent; its fp32 logits tipped near-ties, acceptance 2.72→2.64 → hurt K3) — keep default fused,
  do not chase an MLA kernel.
- **Applies:** partially. DSV4.1 decode is SSD-bound, so the ~30–40 ms compute floor these attack is
  *hidden under* the 3 GB/token read — decode gain is only the sliver of compute not overlapped by SSD
  (small, maybe +2–5 %). **But** they help **prefill/TTFT** (compute-heavier) materially and lower the
  floor that Factor D (R4) hides under. Low-risk carry-overs already in the V4 skeleton.
- **Expected here:** decode +2–5 %; prefill/TTFT larger. `sdpa` still won't fuse (MLA latent 512 ∉ MLX
  fused head-dims 64/96/128/192/256 — same as V4). **Exactness:** Sinkhorn is a normalizer; gate argmax
  parity (V4: bit-identical bf16, argmax exact). **Deps:** V4 code reuse.
- **Rank: 6.**

### R7 — In-place verify-KV update (footprint guard) — from Qwen3.8 256K fix

- **Source:** [[qwen38-verify-band-sdpa-dead-band]] / qwen38-256k-fix: verify KV via `mx.slice_update`
  copied 267 MB × k,v × 12 layers (6.4 GB) per verify (256K OOM root cause); in-place fix removed it.
- **Applies:** YES as a **footprint guard** for DSV4.1 MTP verify. CSA2 shares KV across 36 layers (only
  4 Full layers own KV), so fewer caches than Qwen3.8, but the SWA windows (5 MiB × 40) and compressed
  caches are copy-prone. Every transient GB spent copying is a GB stolen from the 57–65 GiB expert cache
  — and a spike over the knob collapses throughput ~4× ([[never-exceed-the-memory-knob]]).
- **Expected here:** not a tok/s add; **prevents a regression** and protects the cache budget during
  verify. **Effort:** low–medium. **Exactness:** must preserve KV bytes. **Deps:** MTP verify, CSA2.
- **Rank: 7** (do before the first MTP GPU window).

### R8 — Quality-neutral acceptance rule (TokenV3 / typical) — raises α at zero SSD cost

- **Source:** [[qwen38-cascade-acceptance]]: **TokenV3 (Eq 15, `--cascade-rule tokenv3`) α=0.95 =
  107.10 tok/s (+29.3 % vs exact), HumanEval 0.9695 — quality-neutral**; OPT (Eq 10) lossy at every α
  (HE 0.9634→0.7073). Typical acceptance (#478) held 0.9634 at +26 %. David: cascade mode stays
  **off by default** (α unset = exact); TokenV3 is the recommended non-exact rule.
- **There:** a token-specific deferral rule raised acceptance ~0.67→0.72+ at unchanged quality; OPT's
  max-prob rule accepted confident-wrong drafts and craters HE.
- **Applies:** YES and **pure throughput here** — higher α raises L (accepted tokens/cycle) **without
  changing the verify union** (same experts read), so it multiplies decode at zero SSD cost. This is
  rarer than on VRAM models, where higher α also costs more verify compute.
- **Expected here:** α 0.78→~0.9 lifts L≈2.85→~3.4 (**+~17 %** decode) if a HumanEval cell shows no
  regression at David's sampler. Opt-in only (David's ruling); default stays exact/greedy.
- **Effort:** medium (port the rule into DSpark verify; likely reuses W23's verify path). **Exactness:**
  **lossy by construction** — gate with one HumanEval(164) cell ([[humaneval-one-seed]],
  [[task-evals-decide-bank-verdicts]]); ship only if quality-neutral. **Deps:** R1 (MTP verify).
- **Rank: 8** (high EV but quality-gated and opt-in).

### R9 — Engram row LRU + mxfp8 shrink (already largely landed; keep tight)

- **Source:** W19 (engram → mxfp8, **272→264 B/row**, bit-exact); [[deepseek-v41-streaming-artifact]]
  (`NGramRowCache`, preadv + LRU, 24 rows/token/layer).
- **Applies:** the per-token engram read is `24 × 2 layers × 264 B ≈ 12.4 KiB` — **~5 orders below** the
  2.7 GB expert read. Not a bandwidth lever; it is a **latency** item (random small reads) and a
  correctness/footprint item. Keep the LRU bounded; do not requantize further (PORT_PLAN §4: FP8 is
  lossless-to-source; a second lossy step on a gate-damped conditional memory buys nothing).
- **Expected here:** ~0 decode tok/s; matters only if the random engram reads serialize badly (hide
  under compute / prefetch by n-gram hash — see §5 #2). **Rank: 11.**

### R10 — Native FP4 global KV (size, long-context only)

- **Source:** PORT_PLAN §3b/P2.0 (890 vs 3200 B/tok). At 16K the whole global KV is ~14.6 MB (FP4) vs
  ~52 MB (bf16) — a 37 MB delta against a 57–65 GiB cache = **<0.06 %** more experts. See §4 (KV quant
  is dead as a *decode* lever). Only matters at 256K+. **Rank: 12** (deprioritized for 1K/16K shapes).

---

## 4. Dead-here (do NOT re-propose) — with the reason

| Lever | Why it's dead for DSV4.1 |
|---|---|
| **Q2 gs64 routed bank** | Quality-broken: cos 0.912 vs FP4 source, router top-6 agreement 11/31 at L2, junk output, torch-reference confirmed (W9, [[dsv41-native-mxfp4-bank]]). Q2-from-FP4 is a *second* lossy step from a 4-bit source (PORT_PLAN §7). It is 158 GiB (1.7× less SSD/tok) but does not work; superseded by bit-exact native mxfp4. Do not resurrect for the byte win. |
| **Islands on the mxfp4 bank** | Full-island = 269 GiB, cannot fit under 82 GiB. Partial islanding *is* R3 (use that framing). A hand sub-4-bit dequant kernel loses to stock `gather_qmm` ([[metal-sub4bit-alu-bound]], [[iq2xxs-kernel-loses-to-stock]] 1.74× slower, 80 % decode ALU) — no compute win from "islanding" 4-bit experts either. |
| **Verify WIDTH / tree / K>3** | On a streaming bank, wider verify = **bigger expert union = more SSD bytes/output-token**. GLM: d3 already widened 135→180 misses/tok; depth ≥4 "widens the verify union → MORE misses" ([[glm52-q1t-lane-state]]); V4 K>3 slower, tree-width NO-GO ([[deepseek-v4-kernel-verdicts]], [[acceptance-rate-is-a-primary-bottleneck]]). Keep K=3, single-candidate. The opposite of V4's "next lever" — because V4's experts were resident. |
| **KV quantization (as a decode lever)** | At 16K the entire global KV is ~14–52 MB; quantizing frees <40 MB of a 57–65 GiB cache = unmeasurable decode gain, while risking CSA2 shared-cache argmax exactness (4 caches feed 36 layers). Keep bf16 KV for 1K/16K; native FP4 is a 256K-context *size* item (R10). |
| **mxfp4 bank lossless compression** | [[hy3-lossless-compression-c7]]: Q2 experts compress 1.29× (order-0 saturated), **Q4 near-incompressible**. mxfp4 E2M1 codes are high-entropy 4-bit → expect ~1.0×. rANS also "shrinks bytes/service, removes none → capped ~+4 %, negative if decode serializes behind the read" ([[island-placement-beats-tuning]]). Dead on this format. (One cheap entropy check on a bank sample to fully close.) |
| **mmap / MADV_WILLNEED as the primary read path** | [[mmap-willneed-unwired]]: demand-fault flat ~1.4 GiB/s at any thread count; even fully-fixed mmap tops 6.7–10.8 vs **pread 12.9**; mmap can only approximate the *bad* LRU-uniform policy (kernel can't see routing) and drops further under a resident server. Keep pread into fixed slots; the lever is slot allocation, not the mapping. |
| **Resident n-gram / expert arena (static prefetch table)** | [[qwen38-flash-next-80tps-program]]: decode rows 85–93 % novel → resident n-gram arena *falsified*. Engram is disk-backed by design; do not make it resident to "prefetch." |
| **Prefetch as a byte reduction** | [[island-placement-beats-tuning]] (prefetch −11 % at 96 GiB despite +6–12 pts hit rate — "raising hit rate is not the objective"); [[glm52-q1t-lane-state]] ("prefetch can't cut miss bytes — NO-GO"); [[hy3-c5-dense-islands]] (history-based prefetch DEAD — 24 % prev-wave route overlap). Also [[a3b-decode-roundtrip-is-the-lever]]: sync-removal / branch-predict prefetch was KILLED (+3.27 % *slower*) not by bandwidth but because MLX's async submission backpressure (10-buffer / 50-op force-commit) conserves the blocking regardless of the Python sync. Prefetch/overlap (R4) *moves* bytes in time (queue depth), it does not delete them, and compute-side prefetch can be swallowed by the scheduler. Price it as utilization, never as fewer bytes. |
| **Dispatch-fusion / compile-the-forward as the primary DECODE lever** | [[hy3-decode-roofline]] compile-the-forward DEAD (async_eval already overlaps the graph rebuild); [[moe-exec-fusion-25-26-27]] fused MoE *slower* than stock, #25's 1.85× ~75 % accounting artifact. DSV4.1 decode is SSD-bound, so the Qwen3.8 dispatch story does not transfer to decode (it helps prefill only — R6). |

---

## 5. NEW ideas (specific to DSV4.1 structure)

> These exploit structure Qwen3.8/Hy3/GLM/V4 did not have. Each needs a census/measurement gate; none is
> assumed. Ordered by expected leverage.

1. **Verify-row routing-overlap census → fixed union `u`** (underpins R2, decides the whole program).
   Measure, on David's 1K/16K prompts with DSpark drafts, the per-layer union of top-6 sets across the 4
   verify positions. Local-context hypothesis: consecutive draft tokens route near-identically →
   `u`→6–8 → MTP pays. If they diverge (`u`→18–24), MTP-on-streaming is a GLM-shaped trap. **CPU-only;
   run first.**

2. **Engram-hash-driven expert prefetch (dual use of the n-gram memory).** Engram is keyed by a rolling
   hash of recent tokens. DSpark proposes the draft tokens *before* verify → hash them to (a) prefetch
   the engram rows the verify needs (12 KiB/token, latency-bound random reads — a pure latency win the
   engram path can use even though it is not a bandwidth lever), and (b) use the n-gram memory as a
   *routing predictor* to warm the expert preadv queue. Novel: the memory doubles as a prefetch oracle.

3. **DSpark's own routing is a free expert-prefetch oracle.** The 3 DSpark stages run their *resident*
   mxfp8/mxfp4 MTP experts (top-6 of 128) and produce token proposals. The MTP experts' routing
   correlates with the big model's routing for the same positions → use the (already-computed, resident)
   MTP routing as the prefetch key / dedup pre-seed for the big-bank verify reads, cheaper than a separate
   predictor. This directly feeds R2's union and R4's concurrent issue. **Corroboration:** every
   *history*-based prefetch died on hy3 (24 % prev-wave overlap, [[hy3-c5-dense-islands]]), but that note
   explicitly leaves **"draft-route prefetch (model-signal)"** open as the one surviving prefetch idea —
   DSpark gives DSV4.1 exactly that model signal for free.

4. **CSA2 shared-KV frees the budget for expert cache.** 36 of 40 layers reuse 4 owned caches
   (PORT_PLAN §0). Confirm attention KV traffic/latency is negligible so the *entire* non-SSD budget and
   the Factor-D read window go to expert streaming; then a wider expert cache (fewer streamed slots
   stolen by KV) is justified.

5. **Cross-layer expert de-duplication at the bank level.** Content-hash the 15,360 mxfp4 records; if any
   are byte-identical (or near, within the gate) across/within layers, alias them to one physical record
   (read once, serve many) → raises effective residency for free. Cheap CPU test; likely low hit rate but
   zero-risk if any duplicates exist. Distinct from R2 (within-cycle) — this dedups *storage*.

6. **Programming-class-specific residency pin list.** David's shape is a *programming* prompt class; the
   `noaux_tc` gate may light up a class-specific expert subset. A pin list censused on the programming
   ladder may beat a global one (extension of R3, same held-out gate).

7. **Bounded SWA "replay"** (PORT_PLAN P2.1): reconstruct the 128-wide window from the long-lived global
   KV instead of storing 5 MiB/layer × 40 = 200 MiB resident. Small, but every MB is an expert slot.

---

## 6. Proposed 48-hour execution order (one A/B per lever, David's shape, receipts)

All gates use [[dsv41-standard-benchmark-shape]] (1K + 16K, greedy, single prompt); every arm is
seed/window-suffixed and append-only ([[never-overwrite-a-measurement]]); **one A/B per lever with the
paired in-window control** ([[verify-attributed-reds-on-main]], §7 window-drift law). Report prefill
tok/s, decode tok/s, TTFT, peak GB, wall, and whether outputs are byte-identical. GPU windows go through
the flock ([[gpu-work-always-through-flock]]); **qwen unloaded + memory-guarded during every window**
([[glm52-q1t-lane-state]] panic, [[hy3-benchmark-panic-protocol]], [[box-110gb-hard-limit]]); no CPU-heavy
worker during a window ([[cpu-heavy-work-voids-flock-windows]]).

| Order | Lever | Measurement gate (pass condition) | Cost |
|---|---|---|---|
| **Gate 0** | Verify-row overlap census (New #1) | CPU-only: median per-layer union `u` across 4 verify positions on 1K/16K. **Pass if median u ≤ 10.** u > 14 ⇒ re-scope MTP-on-streaming. | CPU |
| **Gate D** | Realized decode BW today | Instrument current AR decode SSD throughput. If **<8 GiB/s**, R4 (`io_read_fanout` + concurrent issue) is worth ~1.5× — land it before MTP. | CPU probe + 1 short window |
| **Gate 1** | R3 frequency residency | CPU **held-out** census (chronological train/eval split, [[mmap-willneed-unwired]] gate) → concentration test. Only if cross-layer/expert variance ≫ hy3's 0.027 → 1 GPU A/B uniform vs freq-pinned. **Pass if decode +≥20 %.** Else mark R3 dead-here. | CPU gate → maybe 1 window |
| **Gate 2** | R1×R2 MTP + dedup (needs W23) | AR control vs MTP+dedup, greedy verify. **Pass if outputs byte-identical AND decode ≥ 2.0× AR.** If dedup-off shows <1.4×, R2 is the gap (expected per §1.4). | 1 window |
| **Gate 3** | R1×R2×R3×R4 stacked | census-pinned + MTP+dedup + fanout vs AR baseline. **Target decode → 20 tok/s @ 1K and 16K, outputs byte-identical.** Central estimate 10–14; record honestly. | 1 window |
| **Gate 4** | R8 acceptance rule (opt-in) | Gate-3 stack, exact-greedy vs TokenV3 α-sweep + **one HumanEval(164) cell**. **Pass only if quality-neutral** (HE ≥ exact −noise) at +α speed. | 1 window + offline HE |
| **Gate 5** | R5 prefill batching | ± cross-chunk gather dedup + chunk/eval at 16K. **Pass if TTFT −≥15 %** (27 → ≤23 s), peak under knob. | 1 window |
| **Gate 6** | R6 dispatch stack, R7 in-place verify-KV | carry from V4; verify argmax parity + footprint under knob. Ship if parity holds; expect decode +2–5 %, larger prefill. | folded in |

Sequencing rationale: **Gate 0 and Gate 1 are CPU-only and decide feasibility before any GPU time** —
they answer the two unknowns (union `u`, routing concentration) the whole 20 tok/s claim rests on. Gate D
is cheap and may hand ~1.5× for free. Gates 2–3 need W23's DSpark port. Acceptance-rule and prefill are
refinements once the stack clears its central estimate.

---

## 7. Measurement-method guardrails (why past deltas were wrong — apply to every gate)

- **Paired in-window control, always.** A hy3 "island" win read +5.0 % against a 30-min-old baseline;
  the paired in-window control put it at **+2.8 %** — half was the documented **2.4 % window drift**.
  Never credit a delta without the control in the same lock hold ([[island-placement-beats-tuning]]).
- **Count `mx.eval` per arm.** MoE-exec-fusion #25's "1.85×" was ~75 % an accounting artifact from
  uneven eval counts ([[moe-exec-fusion-25-26-27]]).
- **Queued vs eager Metal microbench can invert the verdict** (>10× host-sync difference); decide µs
  promotions on the queued lane ([[queued-vs-eager-metal-microbench]]). Component roofline tables inflate
  absolute ms (sum-of-parts exceeded the whole by 42–71 % on hy3) — never rank targets with them
  ([[hy3-decode-roofline]]).
- **`gather_qmm` calling-convention trap:** x must be `[rows,1,K]→[rows,1,N]`; the wrong shape silently
  does 8× work and **fakes a win** ([[gather-qmm-calling-convention-trap]]). Verify the streamed expert
  gather shape before trusting any MoE A/B.
- **Held-out gate for any "learned allocation."** The 8× frequency-allocation number was oracle leakage;
  the deployable number was ~0 % ([[mmap-willneed-unwired]]). Split train/eval chronologically before
  believing a residency policy.
- **e2e over component.** MLA fused-attention won its microbench 1.4–3.4× and recovered *zero* exposed AR
  end-to-end, and *hurt* K3 via near-tie tipping ([[deepseek-v4-kernel-verdicts]]). Rank on measured e2e,
  not component or roofline.
- **Spot-check outputs per rounding-class window** ([[spot-check-outputs-per-result]]); truncation from an
  output cap is not a quality failure ([[eval-truncation-is-not-failure]]).
- **IOV_MAX / macOS preadv trap** ([[hy3-c5-dense-islands]]): scatter `preadv` with >1024 iovecs returns
  EINVAL on macOS (a full hy3 layer was 1,728 views; a DSV4.1 verify union across 40 layers issued as one
  gather can far exceed 1024). The reader must slice per syscall — enforce this when R2/R4 issue the
  concurrent union read set, or the whole gather silently fails.

---

## 8. Open unknowns that gate the numbers

- **Verify-row overlap `u`** — decides R2/MTP viability (Gate 0). Unknown.
- **Routing concentration** (per-expert, held-out-stable) — decides R3 (Gate 1). hy3's analog failed
  (0.2 %); DeepSeek noaux_tc bias *suggests* skew but is unmeasured.
- **Realized decode SSD BW today** — decides R4 (Gate D). 12.5 is the ceiling; single-stream may realize ~5.
- **DSpark acceptance α** on the programming shape — decides L. V4 got 5.7× at K3; DSV4.1 α unmeasured (W23).
- **mxfp4 native gather kernel speed vs affine** — [[dsv41-native-mxfp4-bank]] UNMEASURED; sub-8-bit
  gathers ALU-bound ([[metal-sub4bit-alu-bound]]). Compute floor only, behind the SSD wall on decode.
- **W18 mxfp8 GEMM-precision finding** — flagged in W18_REPORT (resident mxfp8 dense GEMM precision);
  confirm it does not perturb the streamed==resident argmax gate before trusting decode parity.
