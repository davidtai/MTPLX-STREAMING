# F7 — score-aware expert-cache retention screen (DeepSeek-V4.1 M6 decode)

CPU-only screen. No MLX, no GPU, no service touched: `score_aware_cache_sim.py`
installs a `NoMLX` meta-path finder that raises on any `mlx` import, uses numpy
only, and runs single-process under `nice -n 19`. This is a retrospective
cache-policy screen, **not** a throughput result and **not** a production-code
proposal. A slot bank is a pure cache — it changes only which records are read,
never a routed expert's computed output — so every policy here is **bit-exact**.

## Question

Q4 decode on one SSD is read-bound: ~31,573 expert-record reads (17.7 MB each,
~1.372 ms of critical-path read time) per 1,024-token run at 105–111 slots/layer,
so every 1% of reads is ~0.43 s. The shipped **transition-window** retention
policy (`mtplx/expert_streaming.py` `LayerExpertSlotBank`) scores each expert with
three **history-only** terms:

```
score(e) = w_pred·P(e | previous route)      (online 1st-order transition matrix)
         + w_freq·windowed_frequency(e)/max   (last WINDOW routes)
         + w_rec ·1/(1+epoch−last_used(e))     (recency)
```

At each route the top-scoring of {evictable residents + this route's misses} are
retained; a rejected miss is served through transient scratch. This receipt tests
the one causal signal history cannot see: the **router's own gate scores for the
experts it did not select** — a fourth term `w_gate·g(expert)`.

### Real-vs-proxy statement (applies to every Task-B/C/D number)

The 64-cycle capture stores, for target layer *T* at cycle *c*, the gate applied
to layer *T−1*'s **post-attention** input — a **1-layer-ahead PREDICTION** of
layer *T*'s scores (top-6 recall ~72% vs the true route), **not** layer *T*'s true
gate scores at cycle *c*. `g` is built from that predicted tensor, so it is a
**noisy PROXY**; a true-score signal (layer *T*'s own `[384]` max-over-rows scores)
can only do better, and every Task-B gain below is a **lower bound**.

## Commands (from the worktree root, under `nice -n 19`)

```
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f7-policy
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python   # py3.12, numpy 2.4.4

# full study (Tasks A–D); writes results.json next to this README, prints a summary:
nice -n 19 $PY scripts/deepseek_v41/score_aware_cache_sim.py \
    --out docs/deepseek-v41/receipts/f7-cache-policy-20260919/results.json

# Tasks B/C/D only (no 206-cycle trace needed; ~90 CPU s vs ~60 s for the full run):
nice -n 19 $PY scripts/deepseek_v41/score_aware_cache_sim.py --skip-a --out /tmp/f7-bcd.json
```

The full run is ~60 CPU s. `results.json` carries every number below plus
provenance and the whole variant sweep.

## Inputs (read-only; sha256)

| Role | Path | sha256 |
| --- | --- | --- |
| Router-score capture (NPZ, 45.6 MB; 64 cycles @ **105** slots) | `.worktrees/deepseek-v41/.benchmark-artifacts/deepseek-v41/router-feature-20260918/full-router-feature-20260918-v2.router-capture.npz` | `5d8dd85c…edca3e` |
| Verify route trace (206 cycles × 40 layers @ **73** slots) | `docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz` | `07b4b720…66d0da` |
| Cache implementation (real production policy) | `mtplx/expert_streaming.py` (`LayerExpertSlotBank`) | `de177192…0571c39` |
| Cache-policy restore helper | `docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py` | `7be34759…03fdccc` |
| Clairvoyant helper (Belady bypass) | `scripts/deepseek_v41/analyze_route_cache.py` | `92f71538…16de71f3` |
| Re-scorer (ranked predictor / capture loader) | `scripts/deepseek_v41/rescore_router_capture.py` | `aefdedd0…bcb251af` |
| Overlap sim (trace loader, `make_bank`, timing) | `scripts/deepseek_v41/overlap_schedule_sim.py` | `5e0cc019…13a0e695` |
| **This module (new)** | `scripts/deepseek_v41/score_aware_cache_sim.py` | `249b72ec…2cc6701a` |

The capture and trace sha256 are asserted at load. Capture NPZ keys: `scores`
(2,64,36,6,384) f32 — feature 1 = post-attention router; `actual` (64,40,6,6)
native routes; `persistent`/`physical` (64,36,384) bool residents; `reads`
(64,40,384) in {0,1} the actual physical reads.

## Method — the replay is validated against the captured reads

For Tasks B–D each of the 36 target layers (4–39) is replayed independently
through the **real** `LayerExpertSlotBank` (single pool, 384 experts, 105
persistent + 48 transient), seeded at cycle 0 from the captured `persistent[0]`
resident set (84 experts/layer; the pool fills to 105 within a few cycles). A
`ScoreAwareBank` subclass changes **only** the retention score — `plan()`,
admission slot assignment, transient overflow and every route output stay
byte-identical. The captured run had **no effective prefetch**: `reads[c,L]` ==
`unique(actual[c,L]) − physical[c]` for **2,275 / 2,304** target cells (the 29
exceptions are READY-transient/ring carryover), so misses are pure resident-misses.

**Two facts anchor the whole screen.** (1) The captured demand stream reproduces
the receipt totals exactly: 10,623 reads over 64 cycles (held-out 32–63 = 4,878,
train 0–31 = 5,745; layer 3 = 279; layers 0–2 = 0). (2) The 206-cycle trace at
111+48 reproduces the f1-overlap control **31,636 records** exactly (Task A).

---

## Task B — score-aware retention (105+48; fit on cycles 0–31, reported on 32–63)

### Control fidelity (shipped transition-window replayed from `persistent[0]`)

Shipped weights read from `LayerExpertSlotBank.__init__`:
`transition-window` (default) = **w_pred 0.7 / w_freq 0.2 / w_rec 0.1, window 16**;
`transition-window-tuned` = 0.8 / 0.1 / 0.1, window 32.

| Reads | captured | control replay (default) | control replay (tuned) |
| --- | ---: | ---: | ---: |
| all 64 | 10,623 | 10,730 (**+1.01%**) | 10,714 (+0.86%) |
| held-out 32–63 | 4,878 | 4,875 (**−0.06%**) | 4,859 (−0.39%) |
| train 0–31 | 5,745 | 5,855 (+1.91%) | 5,855 |

The control replay reproduces the captured held-out reads to **−0.06%** (per-cell
exact miss-count match **1,137 / 1,152 = 98.7%**). The +1.01% on all-64 is the
train half's cold start (the replay begins with empty transition/recency/window
state that the captured run had warmed over 16 K prefill tokens; the window limit
is 16, so state is re-warmed well before cycle 32). The `default` replay is closer
to the capture than `tuned`, confirming the **captured run used the default
weights**. The control vs variant **delta** shares this cold-start bias and the
identical seed/machinery, so it is clean regardless.

### Sweep — fourth term `w_gate·g`

`g` variants (all normalized to [0,1], like the shipped `window_frequency/max`
term); each also as an EMA over cycles, decay ∈ {0.5,0.7,0.9}. `w_gate` on a log
grid {0.05,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8}. Control: **5,855** train / **4,875**
held-out reads. Reciprocal-rank-of-max-over-rows is the strongest `g`; the whole
w_gate curve (train vs held-out) is the important part:

| w_gate (recip_rank, ema 0.7) | 0.4 | 0.8 | **1.6** | 3.2 | 6.4 | 12.8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **train** % (fit set) | −2.03 | −2.12 | −2.39 | −2.85 | **−3.33** | −2.99 |
| **held-out** % (test set) | −0.59 | −1.25 | **−1.85** | −1.48 | −0.10 | +1.48 |
| layers improved / p | 19, .05 | 23, 9e-4 | **28, 1.9e-5** | 25, .009 | 21, .11 | 14, .49 |

**The train objective keeps improving with `w_gate` (down to −3.33% at w=6.4) but
the held-out gain peaks at moderate w≈1.6–3.2 and collapses past it** (as
`w_gate → ∞` a fourth-term ranking degenerates to the pure-score policy, below).

### Honest fit-vs-report (params fit on 0–31, reported on 32–63)

| Selection | config | train % | **held-out %** | layers | p |
| --- | --- | ---: | ---: | ---: | ---: |
| **fit `w_gate` to train grid-min** (literal, OVERFITS) | recip_rank ema0.7 w=6.4 | −3.33 | **−0.10** | 21/36 | 0.11 (n.s.) |
| **moderate w, train+test agree** (regularized) | recip_rank ema0.7 **w=1.6** | −2.39 | **−1.85** | 28/36 | 1.9e-5 |
| best on held-out (selected on test, UPPER ref) | recip_rank ema0.5 w=3.2 | −2.85 | **−1.97** | 26/36 | 5.4e-4 |

Naively fitting `w_gate` to the 32-cycle **train** minimum overshoots to w=6.4 and
**does not generalize** (−0.10% held-out, n.s.): 32 cycles are too few to fit
`w_gate`. But the signal is real — at a moderate, cross-half-**stable** weight
(w=1.6, which train −2.39% and held-out −1.85% both endorse) the gain is **−1.85%
held-out, 28/36 layers improved, p = 1.9e-5** (bootstrap 95% CI [−3.53, −1.50]
reads/layer). Across the full grid, **47 / 264 variants are held-out-significant
improving**; the best per non-degenerate g-kind: recip_rank −1.85%, ind48 −1.60%,
max −1.46%, mean −0.92% (ind12/ind24 not significant).

### Pure score-EMA and victim-only — history is essential

| Variant | best held-out % | layers improved |
| --- | ---: | ---: |
| **pure** score-EMA (no history terms) | +6.44% (worst +97.6%) | 0 / 36 |
| **victim-only** (history admits, g picks victim) | +6.44% | 0–1 / 36 |

Ranking retention or the eviction victim by the gate-score proxy **alone** is far
worse than control at every g-kind: the proxy scores the *current* route but,
standing alone, discards the multi-cycle hot set the history terms capture. The
score helps only as a **fourth term added to** the shipped history score.

**Task B verdict:** at a moderate, train+test-stable weight (transition-window +
`w_gate`·reciprocal-rank-of-predicted-gate-score, EMA 0.7, **w=1.6**) the causal
fourth term removes **1.85% of reads held-out (28 of 36 layers improved, sign-test
p = 1.9e-5)** ≈ **0.80 s per 1,024-token run** (extrapolating the held-out fraction
to ~31.6 K records × 1.372 ms; **PROXY 1-layer-ahead scores**). Fitting `w_gate` to
the 32-cycle train minimum overfits to −0.10% (n.s.); the test-selected optimum is
−1.97%. All below the 3% bar.

---

## Task C — Belady-style clairvoyant lower bound (105+48)

Same clairvoyant family as mtp-verify-routes-20260913. **bypass** =
`analyze_route_cache.optimal_batch_misses` (may drop a just-used one-use expert;
this is the "29,812 floor" family, there at 73 slots). **mandatory admission** =
force-keep every current-batch expert, evict farthest-next-use among the rest (the
runtime's older mandatory-admission oracle). Seeded from `persistent[0]` (all-64) /
`physical[32]` (held-out); per-layer, summed over layers 4–39.

| | control (shipped replay) | clairvoyant mandatory | clairvoyant bypass |
| --- | ---: | ---: | ---: |
| held-out 32–63 | 4,875 | 3,291 | 3,242 |
| all 64 | 10,730 | 6,629 | 6,500 |

The available held-out gap is **control − mandatory = 1,584 reads (32.5%)**
(bypass 1,633 / 33.5%). The stable Task-B variant's 90-read held-out gain is
**5.7% of the mandatory gap** (5.5% of the bypass gap); even the test-selected
optimum (96 reads) is only 6.1%: the proxy-gate fourth term captures a small
slice of the clairvoyant headroom.

---

## Task D — host cost per layer call (best variant vs shipped scoring)

Tight loop, 4,000 iterations, warmed 105-resident/window-16 bank, ~4 misses.

| Path | µs / layer call |
| --- | ---: |
| shipped scoring (`_transition_window_scores`: iterates 384 `_ExpertHistory` for `last_used`; `_transition_window_admissions`: sorts ~115 candidates with 4-tuple keys) | **46.1** |
| best variant scoring (shipped + one `[384]` multiply-add of `w_gate·g`) | 47.5 (**+1.3**) |
| g-vector build alone (max-over-rows of `[6,384]` + normalize + EMA) | 2.4 |

The fourth term adds **~1.3 µs (~3%)** to the ~46 µs shipped scoring path. In
production the device already computes the gate scores, so it hands the host a
precomputed `[384]` vector; the extra ~2.4 µs g-build is only incurred if the host
recomputes it. Cheap relative to the ~1.372 ms of a single record read.

---

## Task A — transient + warm start (206-cycle trace @ 111+48, transition-window)

Baseline (actual boundary snapshot, real policy replay) = **31,636 records**
(exactly the f1-overlap control). Steady-state = **134.8 misses/cycle** (cycles
100–205). **Caveat:** the trace is a 73-slot capture grown to 111, so the boundary
leaves 38/111 slots artificially empty — this inflates the early transient and the
warm-start gains; a clean number needs a native 111-slot boundary snapshot.

### Misses per cycle (decade means)

| cyc | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 | 80–89 | 90–99 | steady |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| misses | 246.9 | 194.7 | 167.5 | 136.0 | 139.8 | 203.7 | 186.0 | 170.7 | 126.8 | 162.4 | 134.8 |

An initial warm-up (cyc 0–39, 247 → 136) **plus** a mid-run re-warming spike
(cyc 50–69). Cumulative excess over steady state = **5,695 records (7.81 s** at
1.372 ms); of that, only the initial cyc 0–39 portion (~**2,058 records / 2.82 s**)
is a boundary transient — the rest is mid-run workload shift a static warm start
cannot fix.

### Warm-start removal

| Warm start | initial residents = | total records | removed vs baseline |
| --- | --- | ---: | ---: |
| baseline | actual snapshot (73 + 38 empty) | 31,636 | — |
| **oracle** N=10 | 111 most-used in decode cyc 0–10 | 29,579 | **2,057** |
| oracle N=30 | 111 most-used in decode cyc 0–30 | 30,472 | 1,164 |
| oracle N=all | 111 most-used over all cycles | 31,264 | 372 |
| **causal** | top-111 by `_prefill_route_freq` | 31,410 | **226** |

Oracle N=10 removes 2,057 records ≈ the entire initial cyc 0–39 excess (2,058):
a perfect early-hot-set warm start eliminates the boundary transient but not the
mid-run spikes. The **causal** warm start (`_prefill_route_freq` **is** carried in
the trace) removes only **226** — ~11% of the oracle's best — so the prompt's
prefill frequency is a weak proxy for the decode hot set. (`N=all` removes least:
optimizing residents for the run average, not the cold start, barely dents the
early transient.)

---

## Task E — production change (conditional; NOT triggered)

Task E is specified only if Task B shows a held-out gain ≥ 3%. The stable causal
variant reaches **1.85%** held-out (test-selected optimum 1.97%; proxy scores),
**below the 3% bar**, so **no production change is specified**. The signal is real
(28 / 36 layers improved, p = 1.9e-5) but small, and history-only terms dominate.
Because these are the
1-layer-ahead **proxy** scores (top-6 recall ~72%), a true-score signal could
clear the bar; the capture that would settle it must record, for **all 40 layers
× all cycles**, layer *T*'s **own** `[384]` max-over-rows gate scores at cycle *c*
(the device tensor evaluated with `indices` inside the gate, riding the existing
routing-index sync), rather than the *T−1* prediction used here.

## Verdict

**The best stable causal variant removes 1.85% of reads held-out (28 of 36 layers
improved, sign-test p = 1.9e-5) ≈ 0.80 s per 1,024-token run; PROXY (1-layer-ahead)
scores, so a lower bound.** It is real but small — ~5.7% of the 32.5% clairvoyant
gap — and costs ~+1.3 µs/call. Fitting `w_gate` to the 32-cycle train minimum
overfits (held-out −0.10%, n.s.); the test-selected optimum is −1.97%. The gate
score helps **only** as a fourth term added to the history score; ranking
retention or the eviction victim by the score alone (pure / victim-only) is far
worse than the shipped policy. Separately, a boundary warm start is worth up to
~2,057 records / ~2.8 s (oracle), but the only causal signal available at the
boundary (`_prefill_route_freq`) captures just 226.

## Limits

- Screen only. No production code changed, no GPU/service touched, no throughput
  proof. The measured GPU runtime still rejects prefetch under this policy.
- Task-B/C `g` is the **1-layer-ahead predicted** gate tensor (a proxy), not the
  true per-layer scores; every gain is a lower bound.
- Task-B replay seeds the resident **set** from `persistent[0]` but starts the
  transition/recency/window/history state cold (unknown from the capture);
  validated by the −0.06% held-out control match.
- Task A uses the 73-slot trace grown to 111 (38 boundary slots artificially
  empty); warm-start magnitudes are upper-biased and mixed with mid-run shifts.
- Oracle warm starts and the clairvoyant bounds are unachievable references, not
  deployable policies.
