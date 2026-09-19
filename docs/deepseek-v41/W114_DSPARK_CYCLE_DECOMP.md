# W114 — DSpark cycle decomposition on the valid window-43 receipt

**2026-09-13 correction:** the receipt below remains historical evidence, but its
barrier timings do not establish an attention-only cost or a model-math throughput
ceiling. `eval_indices` fences a lazy graph and can include preceding layer work.
The new [real-shape census](receipts/attention-census-20260913/README.md) does not
reproduce a sixfold M=6 attention penalty. Scheduling and exact-arithmetic kernel
improvements remain open; a changed model or shorter context is not justified by
this decomposition. Its original prompt and memory budget also differ from the
current 110 GB Python acceptance workload.

Worker `w114/dspark-cycle-decomp`, Opus 4.8, **read-only receipt analysis + code
audit** (no production code changed). Base `7ca25d61a` (`int/w97f-lanes`, the W110
merge). Authored CPU-only, MLX pinned to CPU, no model loaded, no Metal touched (a
GPU benchmark window holds the Metal lock).

**Anchor:** `receipts/gpu-windows/window-43/dspark-d5-ring-v2-draft-attn.json`
(`cell16k_ring_v2_draft_attn`, DSpark depth 5 + draft levers + attention stack).
Cross-arm: `window-43/ar-ring-v2-attn.json` (AR with the same attention stack).
Prior audits folded in, not re-derived: W108 (cycle sync anatomy), W109 (what bounds
the verify — sha256 + prefetch-window, reads already concurrent per record).

**Two provenance corrections that change every number vs W108/W109:**

1. **Window-43 is the first receipt on the corrected prompt.** Windows 39–42 ran the
   wrong prompt (W113 / memory), so their absolute miss/byte figures (W108's 6.95
   GB/cycle from window-39, W109's 5.42 GB/verify from window-41) do **not** transfer.
   Everything below is re-derived from window-43.

2. **The DSpark headline counters live in the nested `dspark.serve_stream_counters`,
   not the top-level block.** The top-level `serve_stream_counters` /`stage_timing` in
   both window-43 files is an **AR (T=1) stage-timing pass** stamped alongside the
   DSpark headline (`route_calls=10240=256×40`, `moe.routed_switch count=10240`,
   `attn.reuse count=7680=256×30`). The DSpark verify's own counters
   (`route_calls=3480=87×40`, `expert_requests=125280=87×40×6`) are under `dspark.*`.
   Reading the top-level block as "the DSpark pass" (as an easy misstep) understates
   verify bytes by ~1.7× (it reports the AR pass's 0.99 GB/tok, not the DSpark
   headline's 1.64 GB/tok).

---

## 0. Headline

> **The historical verify takes 826 ms/cycle, with ~408 ms charged to routing
> barriers.** The boundaries identify waits in this implementation, not the
> individual kernels that cause them. The ~5.9 tok/s projection assumes all other
> costs remain unchanged when SSD waits disappear; it is not an architectural
> ceiling or evidence that reaching 20 tok/s requires changing model math.
>
> **Depth 5 is not the optimal depth.** The acceptance chain (p = 0.75, 0.68, 0.64,
> 0.64, 0.78) reproduces the measured 2.94 tok/cycle at K=5, and a cost model anchored
> on the receipt puts the optimum at **K=2–3 (~3.79 tok/s modeled, +11% over K=5)** —
> and K=5 is beaten under **every** compute-scaling assumption tested (K≤4 always
> wins). This is the cheapest lever on the board: one arm flag, byte-identical, and it
> should be measured first.

Cross-check (§1): `verify_ms × cycles = 826.47 × 87 = 71.903 s = phase_time_s.verify`
(71.903) — exact.

---

## 1. The 826 ms verify, decomposed (window-43 step 5)

Source: `dspark` block — `per_cycle_ms`, `phase_time_s`, `cycles=87`,
`tokens_per_cycle=2.954`, `accept_rate=0.698`; and `dspark.serve_stream_counters`
(the headline pass). All per-cycle.

| term | ms/cycle | source | what it is |
|---|---:|---|---|
| **verify (total)** | **826.5** | `per_cycle_ms.verify_ms` | one target forward, M=K+1=6 rows, 40 layers |
| — covering routing-fence time (`hot.eval_indices`) | **407.9** | `route_probe_sums_ns.hot.eval_indices` 35.486 s ÷ 87 | fence over current indices and unresolved graph ancestors; kernel attribution unavailable |
| — route planning (`begin_split_route`) | **19.3** | `route_probe_sums_ns.hot.begin_split_route` 1.683 s ÷ 87 | W81 split-route union planning (positive & sane here; it was a negative counter artifact in the AR pass) |
| — **residual** | **399.2** | verify − eval_indices − begin_split | wall time outside the two instrumented scopes; individual work and overlap unresolved |
| &nbsp;&nbsp;· conditional SSD duration | 360.7 | 4.833 GB/cyc ÷ 13.4 GB/s | assumes the quoted bandwidth; not a measured overlap or hardware bound |
| &nbsp;&nbsp;· conditional remainder | ~38.6 | residual − conditional SSD duration | arithmetic estimate, not measured kernel attribution |
| accept | 0.40 | `per_cycle_ms.accept_ms` | greedy compare (argmax already read) |
| commit | 2.43 | `per_cycle_ms.commit_ms` | trim to `accepted+1`, seed MTP windows (no full-KV copy) |
| draft | 32.3 | `per_cycle_ms.draft_ms` | 3 MTP stages + rollout, 100% resident (0 SSD) |

**Bytes / routing (headline, per cycle):** `bytes_read` 420.46 GB ÷ 87 = **4.833
GB/cycle**; `records_streamed` 257/cyc × 18.80 MB; `expert_misses` 306/cyc;
`expert_requests` 1 440/cyc = 40 layers × 6 rows × top-6 (confirms M=6);
`unique_expert_requests` 916/cyc → **union ≈ 22.9 experts/layer** (vs **6.0** for AR
M=1 — the union tax); `hit_rate` 0.788; `route_calls` 3 480 = 87×40 (one routing
barrier per layer per cycle). `w61_engagement`: `verify_engaged_pct=100`,
`verify_single_barrier_split=3467`, `verify_single_barrier_batched=0` (every layer's
union fit one wave under the 48-slot transient capacity → **no extra multi-wave
fences**), `all_hit=13`, `allhit_fence_eval=0`.

**Bandwidth arithmetic and its limits.** 4.833 GB / 826 ms is **5.85 GB/s over
the whole verify**. Dividing the same bytes by the 399 ms residual gives **12.1
GB/s**, but that denominator is not a measured read-active interval. The original
claim that reads occupy only the residual and leave the drive idle throughout
the routing fences was an inference, not established overlap evidence. Neither
the 361 ms conditional SSD duration nor the 408 ms covering-fence time supplies
a measured recoverable I/O budget. Timestamped I/O and GPU intervals are needed.

**What the receipt cannot separate, and the counter that would.** Inside the 399 ms
residual it cannot split **exposed SSD wait** from **`gather_qmm` compute** from
**per-record sha256 io-thread time**. W109 measured sha256 on this box at 3.3 GB/s
single-thread / 31 GB/s at 11-way, so hashing 4.83 GB/cycle costs ~155–200 ms of
io-thread time — but how much is *exposed* vs *hidden under the next read* is not in
this receipt. `verify_record_hashes` defaults **True** and is not overridden by the
cell (W109 §1.b), so it is ON here. To measure the split: arm
`MTPLX_DSV41_RESOURCE_TELEMETRY=1` and read `reader_pool.active_work_peak` /
`active_work_histogram_ns` (realized read concurrency, W109 §1.d) plus a
`verify_record_hashes=False` A/B for the exposed-hash fraction. Note also
`gate_prefetch_armed=False`/`gate_prefetch_k=0` in the receipt is the known reporting
bug (W109 §1.c) — the runtime auto-arms k=6 on RUNNER=v2, and `prefetch_issued=16969`
/ `prefetch_committed=16203` prove it ran.

---

## 2. The 335 ms AR token, decomposed (window-43 `ar-ring-v2-attn`)

Two passes in one receipt: the **untimed headline** (`decode_wall_s` 85.66 s ÷ 256 =
**334.6 ms/tok**, 2.988 tok/s — the "335 ms" figure) and a **`--stage-timing` pass**
(`frame_wall_ms_per_token` **412.6**, `stage_sum_ms_per_token` 408.7). The stage
breakdown exists **only for the timed pass, which is ~78 ms/tok slower because
`--stage-timing` forces eager**: `attn_core_compile_engagement compiled=0/eager=10880`
(K4 off) and `small_stages_engagement engaged=False` (K35 off) — exactly the
[[stage-timing-disables-compile-levers]] regime. So the per-stage ms below are the
eager regime; the headline is ~74 ms/tok faster, that saving landing on
attention + projections (the K4/K35 targets).

Eager stage breakdown (ms/tok, sums to `stage_sum` 408.7):

| group | ms/tok | stages |
|---|---:|---|
| **routed_switch (MoE gather + SSD)** | **160.2** | `moe.routed_switch` |
| &nbsp;· SSD read floor | 63.4 | 849 MB/tok ÷ 13.4 GB/s (`bytes_read_per_token`) |
| &nbsp;· gather/dispatch | ~96.8 | remainder of routed_switch |
| **attention** | **67.3** | `attn.reuse` 63.6 (30 layers) + `attn.swa_only` 3.7 (2) |
| **KV / compressor stages** | **36.1** | `attn.full` 19.8 + `attn.reindex` 16.3 (the 4 kv-source layers) |
| hc | 76.6 | `hc.premix_sinkhorn` 50.5 + `hc.combine` 26.1 |
| gate_topk | 29.8 | `moe.gate_topk` |
| shared_expert | 19.8 | `moe.shared_expert` |
| moe.combine | 10.5 | `moe.combine` |
| small (head/embed/engram/norm/sample) | 9.3 | head 3.1, embed 1.1, engram 3.4, final_norm 0.4, sample 0.4 |

So after the ~64 ms compiled-headline attention, the AR token is: **routed_switch
(SSD ~63 + gather ~97), KV ~36, hc ~77, gate/shared/combine ~60, small ~9**, with the
~74 ms eager penalty removed by K4/K35 in the headline. **AR is compute-bound:**
`utilization.gpu_active_ratio` mean **0.958**; the SSD floor (63 ms) is only 19% of
the 335 ms token, and armed prefetch on AR moved tok/s ~0 in earlier windows (W108
§2.1) because the reads already hide under the compute. AR misses/tok 45.2,
hit 0.812, union 6.0/layer.

---

## 3. Acceptance economics — and why depth 5 is the wrong depth

From `accept_rate_by_depth`, `drafted_by_depth`, `accepted_by_depth` (window-43 step
5). The drafts form a **sequential chain**: depth d is only proposed when d−1 was
accepted, so `drafted[d] = accepted[d−1]`, verified exactly:
drafted [87, 65, 44, 28, 18] → accepted [65, 44, 28, 18, 14] → conditional accept
**p = [0.747, 0.677, 0.636, 0.643, 0.778]** (overall 0.698).

**Expected tokens/cycle** = 1 (primary, always emitted) + Σ cumulative-accept:

| max depth K | Π accept to K | E[tokens/cycle] |
|---|---:|---:|
| 1 | 0.747 | 1.747 |
| 2 | 0.506 | 2.253 |
| 3 | 0.322 | 2.575 |
| 4 | 0.207 | 2.782 |
| **5** | 0.161 | **2.942** ✓ (measured 2.954) |

**Cost model** (assumptions stated; anchored to reproduce the K=5 receipt exactly):
verify in-barrier compute = 55% fixed (KV read) + 45% row-linear in M=K+1 (anchor
407.9 ms at M=6); residual SSD scales with union(M), union = 6 at M=1 → 22.9 at M=6;
draft ≈ 15% fixed + linear rollout (32.3 ms at K=5); begin_split + non-SSD residual +
commit + accept held. This rebuilds verify_ms=826 and tok/s=3.42 at K=5 (matching the
measured 3.426), so it is calibrated at the anchor:

| K | E[tok/cyc] | draft | eval_idx | SSD floor | verify_est | cycle ms | **tok/s** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.747 | 11.3 | 285.5 | 147.8 | 491 | 505 | **3.46** |
| **2** | 2.253 | 16.5 | 316.1 | 201.0 | 575 | 594 | **3.79** |
| 3 | 2.575 | 21.8 | 346.7 | 254.2 | 659 | 684 | **3.77** |
| 4 | 2.782 | 27.0 | 377.3 | 307.4 | 743 | 773 | **3.60** |
| 5 | 2.942 | 32.3 | 407.9 | 360.7 | 826 | 862 | **3.42** |

**Optimal depth on real text at accept ≈ 0.70 is K = 2–3** (~3.79 tok/s), **+11% over
the shipped K=5.** The marginal accept collapses (0.68, 0.64 at depths 2–3), so each
extra depth widens the union, adds a verify row and ~53 ms of SSD faster than it adds
accepted tokens.

**This is robust.** Sweeping the one soft assumption (how strongly the barrier scales
with M) from fully fixed to fully row-linear, the optimum moves only within K = 1…4 —
**K = 5 is never optimal**:

| eval_indices fixed-fraction | argmax depth |
|---|---|
| 0.00 (fully row-linear) | K=1 |
| 0.35 | K=2 |
| 0.55 (most defensible) | K=2 |
| 0.75 | K=3 |
| 1.00 (fully KV-bound) | K=4 |

Caveat: depths 1–4 are **modeled** — window-43 has only depth-5 receipts. This is a
prediction to measure (§5 T-A), the cheapest lever on the board.

**What higher acceptance buys** (uniform p, holding the K=5 cycle cost — accept does
not change verify cost, only tokens/cycle):

| accept | E[tok/cycle] | tok/s @ 862 ms cycle |
|---|---:|---:|
| 0.698 (today) | 2.93 | 3.40 |
| 0.80 | 3.69 | **4.28** (+26%) |
| 0.90 | 4.69 | **5.44** (+60%) |

Under this unchanged-cost model, acceptance 0.90 projects ~5.4 tok/s. That
projection does not bound implementations with different scheduling or kernels.

---

## 4. Historical lever estimates and their limits

### Conditional unchanged-cost projection

If SSD waits vanished while all remaining measured costs stayed unchanged, two
calculations from the same receipt give:

- verify − SSD floor = 826.5 − 360.7 = **465.8 ms** → cycle 501 ms → **5.90 tok/s**
- verify × (1 − exposed-wait share 0.428, from `overlap_telemetry`) = **473 ms** →
  cycle 508 ms → **5.81 tok/s**

The per-depth projection peaks at **K=3 ≈ 6.0 tok/s**. These are conditional
estimates, not independent hardware bounds. The 408 ms fence includes its graph
ancestors; changing scheduling or exact-arithmetic kernels can change that cost.
At 2.94 tokens/cycle, 20 tok/s allows 148 ms/cycle. No result here demonstrates
that target, or proves it impossible with the original model math.

### Ranked levers (ms/cycle saved on the verify path)

| # | lever | mechanism | ms/cycle | class | build |
|---|---|---|---:|---|---|
| 1 | **Lower max depth 5 → 2/3** | fewer verify rows + narrower union; marginal accept doesn't pay past depth 2–3 (§3) | **~230–270** (826→575–659) | byte-identical (greedy verify authoritative; output = AR) | zero — `--dspark-depth` arm exists |
| 2 | **Drop per-record sha256 on decode** (`verify_record_hashes=False`) | integrity is covered by the admission receipt at open; removes 4.83 GB/cyc of io-thread hashing (W109 lever 1) | ~100–200 (exposed fraction TBD) | byte-identical (bytes/outputs unchanged) | one flag / small arm |
| 3 | **Overlap the per-layer read burst under compute + deepen the gate oracle** | historical lookahead hypothesis; actual read/GPU overlap and graph ancestry need measurement | unmeasured (old 100–250 estimate was conditional) | byte-identical only after cache ownership and token/state parity gates | medium |
| 4 | **Prefetch the verify union during the draft + fix `skipped_lock_held`** | SSD idles the whole draft; seed from draft routes + last cycle's union (W108 I2/I5, W109 lever 2) | ~30–80 (**now small** — the attention stack cut draft 251→32 ms, shrinking the idle window) | byte-identical | medium |
| 5 | **Raise verify hit rate / residency** | 306 misses/cyc at hit 0.788 is the bytes lever; fewer misses = fewer reads AND less hash (W109 lever 5) | scales SSD floor (each +0.05 hit ≈ −25 ms) | byte-identical (cache policy) | separate residency lane |
| 6 | **Device-LUT verify — drop the 40 routing barriers** (W95 phase 2 / W108 I6) | removes the per-layer drain so the host issues reads earlier; **enabler** for #3, not a standalone win | ~50–150 (enabler) | byte-identical by epoch construction | large |
| 7 | **Draft head bf16 + settle draft compile** (W108 I3/I4) | draft is now only 32 ms (attention stack already collapsed it), so the fp32-head trap is a small share | ~10–20 | rounding-class (draft only) | zero/near-zero |
| 8 | **Reprofile attention and preceding graph work** | isolate the kernels behind the routing fence before selecting an optimization | unmeasured | preserve arithmetic and exact workload; parity required | diagnostic first |

**Top 3:** (1) depth 5→2/3 — biggest, free, byte-identical, dispatch first;
(2) drop verify sha256 — cheap, byte-identical; (3) overlap reads under compute /
draft-window prefetch. These historical estimates need fresh measurements under
the current workload and budget; they do not establish a combined throughput cap.

---

## 5. Two next tasks to dispatch

### T-A — Measure DSpark depth 2 and 3 (+ arm read-concurrency telemetry)
- **Class/runner:** GPU A/B (sonnet driver, once the Metal lock is free; one arm per
  process under the box flock).
- **Do:** run `ab_decode_env_levers.py` arms `cell16k_ring_v2_draft_attn` at
  `--dspark-depth 2` and `--dspark-depth 3` on the window-43 cell (16 384 ctx / 256
  decode / max-kv 17 408 / `--memory-limit-gib 60`), plus one diagnostic run with
  `MTPLX_DSV41_RESOURCE_TELEMETRY=1` at depth 5.
- **Receipt to read:** `dspark.per_cycle_ms.verify_ms`, `dspark.tokens_per_cycle`,
  `dspark.decode_tok_s`, `dspark.serve_stream_counters` (bytes/misses/union), and
  `reader_pool.active_work_histogram_ns` from the telemetry run.
- **Done-when:** `token_ids_sha256` == the AR reference (byte-identical), and the
  depth-2/3 tok/s vs depth-5 confirms or refutes the §3 optimum (model predicts
  depth 2 ≈ 3.79, +11%). The telemetry run resolves §1's SSD-vs-hash-vs-gather split.

### T-B — Add a `verify_record_hashes=False` decode arm + receipt telemetry lift
- **Class/runner:** CPU-only build (opus48-worker, `nice -n 19`, MLX on CPU, ≤1.5 GB
  RSS, no GPU).
- **Files:** `mtplx/models/deepseek_v41_loader.py` (`build_streaming_config`: gate
  `verify_record_hashes` off for the decode arm), `scripts/deepseek_v41/
  ab_decode_env_levers.py` (~~new `cell16k_ring_v2_nohash` arm~~ SUPERSEDED by W110 —
  landed as `cell16k_ring_v2_hash`, which turns hashing ON to MEASURE its io-thread
  cost; the `_nohash`/`_draft_nohash` arms were never created + lift `reader_pool`/`io`
  into `_runner_receipt_blocks` — W109 §1.d groundwork), and a CPU-pinned
  fake-runtime byte-identity test asserting hashing on/off yields identical slot bytes
  (hashing is integrity-only, byte-neutral).
- **Done-when:** the arm exists, `pytest` (no `-n auto`, `nice -n 19`) byte-identity
  test green, and the receipt now carries `verify_record_hashes` + `reader_pool` — a
  one-process GPU A/B (verify_ms, `token_ids_sha256` unchanged) is then ready to hand
  the driver.

---

## Reported numbers (for the orchestrator)

**§1 verify table (window-43 step 5, per cycle):** verify **826.5 ms** =
covering routing fence **407.9** + begin_split **19.3** + residual **399.2**;
the old 360.7/38.6 residual split was conditional, not measured. Bytes
**4.833 GB/cyc**, union **22.9/layer**, whole-verify bandwidth 5.85 GB/s.
Cross-check 826.5 × 87 = 71.903 s =
phase_time_s.verify ✓.

**§3 optimal depth:** **K = 2–3** (~3.79 tok/s modeled, +11% over K=5); K=5 is
suboptimal under every compute-scaling assumption (K≤4 always wins). Modeled — measure
via T-A.

**§4 top 3 levers:** (1) depth 5→2/3 (~230–270 ms/cyc, byte-identical, free);
(2) drop verify sha256 (~100–200 ms/cyc, byte-identical); (3) overlap reads under
compute / draft-window prefetch (~100–250 ms/cyc, byte-identical).

**Conditional projection:** ≈5.8–5.9 tok/s with non-I/O costs held fixed, peaking
~6.0 at K=3. The original claim of a model-math ceiling is withdrawn: the barrier
attribution and these historical projections do not establish one.

---

## Provenance

Base `7ca25d61a`. Receipts (read-only): `window-43/dspark-d5-ring-v2-draft-attn.json`
(anchor), `window-43/ar-ring-v2-attn.json` (AR); `window-41`/`window-42` inspected only
to confirm the wrong-prompt caveat (their absolute miss/byte figures do not transfer).
Prior audits: W95 (AR to-be), W96 (AR as-is), W100 (verify prefetch path), W103/W104
(draft), W108 (sync anatomy), W109 (sha256 + prefetch window; reads already concurrent
per record). All arithmetic in `nice -n 19` CPU Python; no Metal, no model load.
