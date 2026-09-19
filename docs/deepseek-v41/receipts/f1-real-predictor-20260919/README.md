# F1 real-predictor: full coverage/precision curve + overlap replay

CPU-only screen. No MLX, no GPU, no service touched: both scripts install a
`NoMLX` meta-path finder that raises on any `mlx` import, use numpy only, and run
single-process under `nice -n 19`. This is a scheduling/quality screen, **not** a
throughput result and **not** a production-code proposal.

## Question

f1-overlap-sim-20260919 left one unknown: the **real** predictor quality on the
D5/M6 verify workload. Its oracle-1-ahead ceiling was 19.25 TPS, a synthetic
0.74/0.62 predictor 15.8-17.2 TPS, and the only *real* predictor tested (the
cross-layer causal transition matrix) was net-harmful at precision 0.09. Codex's
router-feature-20260918 capture recorded, for the first 64 verify cycles of the
exact 16,384/1,024 run and the 36 target layers 4-39, the next layer's gate
applied to (i) the mean residual **before** attention and (ii) the native router
input **after** attention of the current layer — a 1-layer-ahead prediction —
plus native routes, READY physical owners, and the actual physical record reads.
Codex evaluated it at only one operating point (>=85% precision, per-layer
width/margin fitted on cycles 0-31 → coverage 4.9% / 15.6%). This receipt draws
the **whole** curve with a parameter-free ranked predictor and replays it through
the f1 overlap model.

## Commands (from the worktree root, under `nice -n 19`)

```
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f1-overlap-sim
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python   # py3.12, numpy 2.4.4

# coverage/precision curve over both inputs x {max,sum} merge x k in {1,2,3,4,6,8,12}
# (writes curve.json next to this README, prints the held-out table):
nice -n 19 $PY scripts/deepseek_v41/rescore_router_capture.py \
    --out docs/deepseek-v41/receipts/f1-real-predictor-20260919/curve.json

# REAL-predictor overlap arm: replay the 64 captured cycles, control vs real vs
# oracle-1-ahead at rings 16/32/64 (writes capture-sim.json, prints the table):
nice -n 19 $PY scripts/deepseek_v41/overlap_schedule_sim.py --capture-only \
    --capture-out docs/deepseek-v41/receipts/f1-real-predictor-20260919/capture-sim.json

# unit tests (no MLX/mtplx/trace; capture-backed tests skip if the NPZ is absent):
nice -n 19 $PY -m pytest tests/test_dsv41_overlap_schedule_sim.py -q
```

Each script runs in <2 CPU s. `--capture-only` needs no mtplx and no 206-cycle
trace: the capture stores the real demand reads directly.

## Inputs (read-only; sha256)

| Role | Path | sha256 |
| --- | --- | --- |
| Router-score capture (NPZ, 45.6 MB) | `.worktrees/deepseek-v41/.benchmark-artifacts/deepseek-v41/router-feature-20260918/full-router-feature-20260918-v2.router-capture.npz` | `5d8dd85c…edca3e` |
| Capture summary (Codex's fitted point) | `.worktrees/deepseek-v41/docs/deepseek-v41/receipts/router-feature-20260918/full-v2/full-router-feature-20260918-v2.router-capture.json` | `dc813d4d…4a576f89` |
| Re-scorer (new) | `scripts/deepseek_v41/rescore_router_capture.py` | `aefdedd0…bcb251af` |
| Overlap sim (real arm added) | `scripts/deepseek_v41/overlap_schedule_sim.py` | `9962da4c…93c7ff5ce` |

The NPZ sha256 is asserted at load. It holds: `scores` (2,64,36,6,384) f32 —
feature 0 = pre-attention mean, 1 = post-attention router; the score index
`target_idx` maps to layer `target_idx+4`, computed at layer `target_idx+3`.
`actual` (64,40,6,6) native routes; `physical` (64,36,384) READY owners at
prediction time; `reads` (64,40,384) in {0,1} the actual physical reads (no
duplicate reads). Captured **decode_slots_per_layer = 105** (not 111).

## Captured demand stream (facts, not model)

- 36 target layers (4-39) perform **10,623** physical reads over the 64 cycles;
  held-out cycles 32-63 = **4,878** misses (train 0-31 = 5,745). No duplicate
  reads, so read==miss. All 6 verify rows present in every captured target-layer
  cycle. Layer 3 has 279 reads (used only as the source of layer-4 predictions);
  layers 0-2 were **not** instrumented (0 reads recorded).
- Coverage ceiling (fraction of misses whose expert is NOT a READY resident at
  prediction time, the honest exclusion): held-out **99.69%** (only 15/4,878
  misses are evicted residents), all-64 99.70%. So the exclusion barely caps
  coverage; the binding limit is predictor rank quality, then the compute window.

## Merge rule

Per-expert merged score = **max** (resp. **sum**) of the per-row predicted gate
score over the 6 verify rows; experts ranked by merged score descending; READY
physical residents excluded; top-k issued. This is the rank-ordered union of the
per-row candidate experts. **Max-over-rows beats sum-over-rows at every budget
and both inputs** (sum dilutes: an expert strongly favored by one verify row
outranks one weakly favored by several). No precision floor; the same global k is
applied to every layer (no per-layer selection, no eval-half fitting).

## Coverage / precision curve — held-out cycles 32-63 (denominator 4,878)

| k | (i) pre-attn mean · max | | (ii) post-attn router · max | | (ii) post-attn router · sum | |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | prec | cov | prec | cov | prec | cov |
| 1 | 53.12% | 12.55% | **60.07%** | 14.19% | 35.24% | 8.32% |
| 2 | 45.49% | 21.48% | **50.43%** | 23.82% | 28.52% | 13.47% |
| 3 | 39.50% | 27.98% | **43.81%** | 31.04% | 24.19% | 17.14% |
| 4 | 34.98% | 33.05% | **38.85%** | 36.70% | 21.66% | 20.46% |
| 6 | 28.34% | 40.16% | **31.52%** | 44.67% | 18.61% | 26.36% |
| 8 | 24.00% | 45.35% | **26.96%** | 50.94% | 16.29% | 30.77% |
| 12 | 18.51% | 52.46% | **20.72%** | 58.71% | 13.27% | 37.62% |

Best input+rule = **(ii) post-attention router, max-over-rows** at every budget.
All-64 (denominator 10,623) tracks the held-out half within ~1 pt, e.g. post·max
k=3 precision 45.83% / coverage 29.82%, k=8 28.37% / 49.22%, k=12 21.89% /
56.98% — the predictor is calibration-free and stable across halves. (The
pre-attn·sum column is omitted; it is the weakest, e.g. k=1 precision 34.64%.)

Codex's fitted >=85%-precision point sits at the far-left, low-budget end of this
curve (coverage 15.6% at precision 85% on input (ii) with per-layer widths).
Dropping the precision floor buys large coverage: 44.7% at k=6 (precision 31.5%),
58.7% at k=12 (precision 20.7%).

## Per-layer breakdown — post-attn router · max · k=3 · held-out

Precision spread 24%-60% across the 36 layers (total 43.81% / coverage 31.04%,
issued 3,456, useful 1,514). Strongest: L33 60%, L13/L30 58%, L12/L27 55%.
Weakest: L15 25%, L6 24%, L16 27%, L28 28%. No layer is degenerate; the signal is
broad, not carried by a few layers. (Full per-layer arrays at k∈{1,2,3,4,8} are
in `curve.json → per_layer_best_heldout`.)

## Overlap replay — 64 captured cycles @ 105 slots, rate 12.9 GB/s, record 17.7 MB

Predictor = post-attention router / max, ranked width 8. The compute window
(c_layer 3.252 ms) admits **2.371 records** (1.372 ms each), so the sim issues ~3
predictions/call and budget k>~3 is inert — matching the k=3 curve point
(precision 0.46). Growth is excluded (these are mid-stream cycles). Layers 4-39
predicted (windows 3-38); layers 0-3 unpredicted. Oracle-1-ahead is scoped to the
same targets 4-39 for an apples-to-apples ceiling.

| Ring | Arm | Total s | Read wait s | Hidden % (useful/demand) | Issued | Useful | Wasted | Precision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | control | 23.988 | 14.954 | 0.0% | 0 | 0 | 0 | — |
| 32 | **real** | 21.618 | 12.585 | 29.1% | 6,912 | 3,177 | 3,735 | 0.46 |
| 32 | oracle-1-ahead | 17.229 | 8.195 | 55.2% | 6,021 | 6,021 | 0 | 1.00 |
| 16 | real | 21.629 | 12.596 | 29.1% | 6,912 | 3,169 | 3,743 | 0.46 |
| 64 | real | 21.491 | 12.457 | 30.0% | 6,912 | 3,270 | 3,642 | 0.47 |
| 16/64 | oracle-1-ahead | 17.229 | 8.195 | 55.2% | 6,021 | 6,021 | 0 | 1.00 |

The real arm hides 29% of demand reads but cuts read wait only ~16%: at precision
0.46 roughly half the issued reads are wrong, and under the no-preempt rule an
in-flight wasted read delays demand, eating about half the gross coverage. Ring
size is a weak lever for this predictor (29.1% → 30.0% from 16 to 64) — better
than the useless causal arm (ring-insensitive at 6.4%) but far from the good
synthetic predictor's ring sensitivity. Oracle-1-ahead here hides 55.2%, below
the full-run oracle's 59.4%, because these first-64 cycles at 105 slots are
miss-heavier (4.60 vs 3.84 misses/layer-call), so the fixed ~2.37-record window
covers a smaller fraction.

## Extrapolation to the full 1,024-token run

Apply the captured read-wait-reduction fraction to the full-run control (74.667 s
total / 43.395 s read wait, from f1-overlap-sim-20260919 @ 111+48, rate 12.9).

| Arm | Ring | Read-wait reduction | Seconds removed | Implied full total | Implied TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | — | — | — | 74.667 | 13.71 |
| **real** | 32 | 15.8% | **6.88 s** | 67.79 | **15.11** |
| **real** | 64 | 16.7% | **7.25 s** | 67.42 | **15.19** |
| oracle-1-ahead | 32/64 | 45.2% | 19.61 s | 55.05 | 18.60 |

This is an **extrapolation**: it assumes the first-64-cycle hidden fraction holds
across the run. Two biases run opposite ways and roughly cancel: (a) the captured
cycles are miss-heavier than steady state (4.60 vs 3.84 misses/layer-call), so the
fixed ~2.37-record window covers a *smaller* fraction there — this **understates**
what the predictor achieves at steady state; (b) the reduction fraction is
measured on layers 3-39 but applied to the full-run read wait including layers 0-2
(which are unpredicted, so get zero reduction) — this **overstates** the removed
seconds by roughly the layers-0-2 read-wait share (a few tenths of a second).
Treat ~7 s / ~15.1 TPS as an indicative central estimate, not a throughput proof;
the real predictor's precision at 111 slots is unmeasured.

## Two-layers-ahead

**Not evaluable.** The capture stores, at layer L, only the L+1 gate applied to
layer L's features (one 1-ahead score tensor per target). No L+2 score tensor
exists at layer L, so two-layers-ahead predictor quality cannot be measured from
this capture. (The f1 sim's oracle-2-ahead uses the *true* future misses, not a
predictor, so it is unaffected.)

## Verdict

**At the best operating point (post-attention router, max-over-rows, ring 64,
window-limited to ~2.37 records/window) the real 1-ahead predictor removes ≈7.3 s
of the 74.7 s control (ring 32: 6.88 s) — about 16% of the 43.4 s read wait —
lifting extrapolated full-run decode from 13.71 to ~15.1 TPS; the oracle-1-ahead
ceiling on the same scoped cycles removes 19.6 s (18.6 TPS).** The real predictor
is genuinely useful — precision ~0.46 / coverage ~30% at the window budget, in
the class of the synthetic 0.74/0.62 reference and far above the net-harmful
causal transition matrix — but it captures roughly a third of the oracle's gain
because precision ~0.46 means half its speculative reads are wasted and delay
demand under no-preempt. It does not reach 20 TPS.

## Limits

- Screen only. No production code changed, no GPU/service touched, no latency or
  throughput proof. The measured GPU runtime still rejects prefetch under the
  transition-window policy.
- Prefetch lives outside the slot pool, so demand misses are predictor-independent
  — taken here directly from the captured `reads` (no bank replay).
- Layers 0-2 were uninstrumented (0 demand modelled); layer 3's 279 reads are
  exposed in every arm. Layers 0-3 are unpredicted and identical across arms, so
  they do not bias the control-vs-arm delta, but the 64-cycle absolute totals omit
  layer-0-2 demand.
- Compute is uniform over layer calls (c_layer from the 111-slot 198-cycle
  extension-bank run); the capture is 105 slots / first 64 cycles, so the timing
  model is transferred, not native to the capture. Same no-preempt / FIFO-ring
  assumptions as the parent f1 sim.
