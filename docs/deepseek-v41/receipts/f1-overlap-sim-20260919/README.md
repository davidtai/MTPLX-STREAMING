# F1: overlap-schedule discrete-event simulation of M6 verify decode

CPU-only screen. No MLX, no GPU, no service touched: `overlap_schedule_sim.py`
installs a `NoMLX` meta-path finder that raises on any `mlx` import, uses numpy
only, and runs single-process. The receipt is a scheduling screen, **not** a
throughput result and **not** a production-code proposal.

## Question

The native DeepSeek-V4.1 decode runner is serial per layer call: route -> wait
for demand expert-record reads (main thread blocks) -> GPU compute. The SSD is
therefore idle for the compute fraction of every layer (~41% of decode in the
best full run). F1 tests whether issuing layer L+1's *predicted* record misses
onto the SSD while layer L computes hides a large share of the exposed read wait.
Earlier GPU screens (lookahead-io-20260918, lookahead-adjacent-20260918,
ridge-prefetch-20260919) replayed only 3 layers with attention omitted, so they
had no compute window and are scheduler-cost evidence only; this sim supplies the
missing compute window from measured attribution.

## Commands (run from the worktree root, under `nice -n 19`)

```
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f1-overlap-sim
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python   # py3.12, numpy 2.4.4

# validate the cache replay against two published receipts (exact), then stop:
nice -n 19 $PY scripts/deepseek_v41/overlap_schedule_sim.py --anchors-only

# full simulation (writes results.json next to this README, prints the arm table):
nice -n 19 $PY scripts/deepseek_v41/overlap_schedule_sim.py

# tiny unit tests of the DES core (no MLX, no mtplx, no trace):
nice -n 19 $PY -m pytest tests/test_dsv41_overlap_schedule_sim.py -q
```

Full simulation runs in ~5 CPU s; `results.json` (same directory) carries every
number below plus provenance.

## Inputs (read-only; hashes asserted at run time)

| Role | Path | sha256 | Real / synthetic |
| --- | --- | --- | --- |
| Route trace (206 cycles x 40 layers) | `docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz` | `07b4b720…66d0da` | REAL native M6 verify capture |
| Cache-policy replay helper | `docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py` | `7be34759…03fdccc` | REAL (installs the NoMLX guard) |
| Cache implementation | `mtplx/expert_streaming.py` (`LayerExpertSlotBank`) @ `2026906b0` | `de177192…571c39` | REAL production cache |
| Timing decomposition | `docs/deepseek-v41/receipts/extension-bank-20260919/summary.json` (`full`) | (curated receipt) | REAL best full run |
| Causal predictor method | `docs/deepseek-v41/receipts/causal-prefetch-screen-20260917/screen.py` | (curated receipt) | REAL, CPU-reproduced |
| Predictor arm (e) | none (coverage 0.74 / precision 0.62) | — | **SYNTHETIC**, labelled |

A 64-sample real router-score capture exists
(`.benchmark-artifacts/.../full-router-feature-20260918-v2.router-capture.npz`)
but is a smaller, differently-shaped capture not aligned 1:1 with this 206-cycle
trace, so it is not used.

## Cache replay validated exactly before any timing claim

`--anchors-only` reproduces two published miss counts with the real
`LayerExpertSlotBank`:

| Anchor receipt | Config | Expected | Reproduced |
| --- | --- | ---: | ---: |
| prefix-readiness-20260918 | cap 102 (73+29), transition-window, `plan()` | 35,164 misses / 8,240 calls | **35,164 / 8,240** exact |
| mtp-verify-routes-20260913 | cap 73+48, frequency, resident-first probe | 53,999 records | **53,999** exact |

## Control reconciliation

The control (no predictor) config is the exact 111+48 slot config of the best
full run, using the production **transition-window** decode policy:

- Replaying the saved 206-cycle trace at 111 persistent + 48 transient yields
  **31,636 demand records** — **+0.20%** vs the extension-bank measured 31,573.
- The saved trace is 206 cycles at 73 slots (commit `bd542a39`); the measured
  31,573 / 43.399 s best run is a **different** 198-cycle trajectory at 111 slots
  (commit `d5f15e7a`). Replaying the 206-cycle trace at 111+48 is the closest
  faithful control; the residual is the acceptance/cycle-count difference.

Timing model (from extension-bank-20260919 `full`, the 111-slot best run):
`verify 69.15495 s`, `read union 43.39900 s` -> **non-read compute 25.75596 s**
over 40x198 = 7,920 layer calls -> **c_layer = 3.25201 ms** (distributed
uniformly; per-layer entrypoint arrays exist only in cpu-attribution-20260918 at
a different capacity, 84->108 slots / 77 s, so they are not mixed in).
`t_draft = 9.9538 ms/cycle`, `t_accept = 0.3437 ms/cycle`,
`t_commit = 0.7712 ms/cycle`, one-time `growth = 2.1957 s`.
SSD = one server; record = **17,694,720 B** (extension-bank
`decode_weight_record_bytes`, "17.7 MB"; trace `source_record_bytes` 18,800,640
is the older capture). Primary rate **12.9 GB/s** (decimal); measured window
12.873 GB/s.

At 12.9 GB/s the control reproduces **read wait 43.395 s** (measured union
43.399 s, -0.01%) and **total 74.667 s** (measured 73.762 s, +1.23%). At the
measured 12.873 GB/s: read wait 43.486 s (+0.20%), total 74.758 s (+1.35%).
Both within a few percent.

## Arm table (primary: 12.9 GB/s, ring 32 slots, record 17.7 MB, 1,024 tokens)

| Arm | Total s | TPS | Read wait s | Read wait hidden | Spec issued | useful | wasted | Realized precision | Extra bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) control (no predictor) | 74.667 | 13.71 | 43.395 | 0.0% | 0 | 0 | 0 | — | 0 |
| (b) oracle 1-ahead | 53.192 | 19.25 | 21.919 | 59.4% | 18,780 | 18,780 | 0 | 1.00 | 0 |
| (c) oracle 2-ahead | 51.197 | **20.00** | 19.925 | **66.6%** | 21,061 | 21,061 | 0 | 1.00 | 0 |
| (d) real causal, width 6/10/12/16 | 78.349 | 13.07 | 47.076 | 6.4% | 22,472 | 2,023 | 20,449 | 0.09 | 362 GB |
| (e) **synthetic** 0.74/0.62 | 64.961 | 15.76 | 33.688 | 32.2% | 19,786 | 10,188 | 9,598 | 0.51 | 170 GB |
| (f) cross-cycle oracle (L0-3) | 72.618 | 14.10 | 41.346 | 5.2% | 1,646 | 1,646 | 0 | 1.00 | 0 |

Real-causal widths 6/10/12/16 are byte-for-byte identical (see finding 2), so
they are collapsed to one row.

Oracle arms are lossless (precision 1.00) because they prefetch the true future
misses. The **real causal** arm (d) is CPU-reproduced from
causal-prefetch-screen-20260917 (online cross-layer transition matrix, causal,
decay 0.98); (e) is a **synthetic** reference at gate-oracle strength.

## Findings

1. **Decode is read-bound, and the overlap ceiling is ~20 TPS, not the SSD-bound
   23.6 TPS.** Perfect prefetch can hide at most 66.6% of read wait (oracle
   2-ahead), giving 20.00 TPS. It does not reach the theoretical SSD-busy floor
   (all 31,636 reads back-to-back = 43.4 s = 23.6 TPS) because the compute window
   is too small to prefetch far enough ahead (see finding 2).
2. **The compute window admits only ~2.37 record reads.** c_layer = 3.252 ms;
   one record at 12.9 GB/s = 1.372 ms, so ~2.37 reads fit per layer's compute
   window against ~3.84 demand misses/layer. This is why oracle 1-ahead hides
   only 59.4% (bandwidth-per-window limited, not lead-time limited) and why the
   causal **prefetch width is degenerate**: the ranked lists average 14.8 experts
   but only ~2.37 are ever issued per window, so widths 6/10/12/16 are identical.
3. **Ring size is a lever only for a good-but-imperfect predictor.** Oracle arms
   at ring 16 / 32 / 64 are byte-for-byte identical (the ring never fills; useful
   prefetches are consumed at the next layer). The real causal arm is also
   ring-insensitive (6.4% at every size) -- its predictions are too wrong to
   benefit. But the synthetic 0.74/0.62 predictor improves with ring size,
   32.1% -> 42.8% hidden (15.74 -> 17.18 TPS) from 16 to 64, because a larger ring
   buffers correct prefetches against FIFO eviction and lets fill() skip
   re-issuing persistent wrong predictions instead of churning bandwidth.
4. **The real causal predictor is net-harmful here.** It hides 6.4% of reads but
   raises total time to 78.3 s (vs control 74.7 s): at precision 0.09 it wastes
   20,449 speculative reads (+362 GB), and under the no-preempt rule an in-flight
   wasted read delays demand more than the 2,023 hidden reads save. This confirms
   causal-prefetch-screen-20260917's rejection (its optimistic held-out
   cross-layer top-1 was 13.69% precision, 3.54% hidden; contention makes it
   worse).
5. **A strong predictor would help but not reach the oracle ceiling.** The
   synthetic 0.74-coverage / 0.62-precision predictor (realized precision 0.51
   after window truncation and eviction) hides 32.2% -> 15.76 TPS at ring 32
   (42.8% -> 17.18 TPS at ring 64), at a cost of 170 GB extra reads. It sits
   between the useless real predictor and the perfect oracle.
6. **The cross-cycle term is small.** Prefetching layers 0-3 during the
   ~9.95 ms/cycle draft gap (oracle) hides 5.2% -> 14.10 TPS; the draft gap is a
   minor overlap opportunity next to the verify compute windows.

## Sensitivity

Prefetch-ring size (rate 12.9; total s / TPS / hidden):

| Ring | oracle 2-ahead | synthetic 0.74/0.62 | causal w10 |
| ---: | --- | --- | --- |
| 16 | 51.197 / 20.00 / 66.6% | 65.039 / 15.74 / 32.1% | 78.349 / 13.07 / 6.4% |
| 32 | 51.197 / 20.00 / 66.6% | 64.961 / 15.76 / 32.2% | 78.349 / 13.07 / 6.4% |
| 64 | 51.197 / 20.00 / 66.6% | 59.591 / 17.18 / 42.8% | 78.344 / 13.07 / 6.4% |

SSD rate (control + strongest arms, ring 32):

| Rate GB/s | control | oracle 2-ahead | causal w10 | synthetic |
| ---: | --- | --- | --- | --- |
| 11.0 | 82.162 s / 12.46 TPS | 58.166 / 17.60 | 90.682 / 11.29 | 73.558 / 13.92 |
| 12.873 (measured) | 74.758 s / 13.70 TPS | 51.282 / 19.97 | 78.498 / 13.04 | 65.065 / 15.74 |
| 12.9 | 74.667 s / 13.71 TPS | 51.197 / 20.00 | 78.349 / 13.07 | 64.961 / 15.76 |

Cache capacity (rate 12.9, ring 32):

| Capacity | Records | Control total / read wait | Oracle 2-ahead total |
| --- | ---: | --- | ---: |
| 111 + 48 | 31,636 | 74.667 s / 43.395 s | 51.197 s |
| 115 + 48 | 30,235 | 72.745 s / 41.473 s | 49.687 s |
| 119 + 48 | 28,886 | 70.895 s / 39.622 s | 48.269 s |

Note that +8 persistent slots (111->119) removes 2,750 reads and 3.77 s of control
time on its own — more than the real causal predictor delivers.

## Limits

- Screen only. No production code changes, no GPU/service touched, no latency or
  throughput proof. The measured GPU runtime currently rejects prefetch under the
  transition-window policy.
- Prefetch is modeled as living **outside** the slot pool (as in the runtime), so
  it does not alter the cache policy; demand misses are therefore deterministic
  and predictor-independent, precomputed once by the exact replay.
- Compute is distributed uniformly over layer calls (per-layer arrays exist only
  for a different-capacity 77 s instrumented run). Prefetch reads are assumed
  submitted at each compute window's start and to proceed on the SSD independent
  of GPU/host compute (the hypothesis under test).
- The control is the 206-cycle trace at 111+48; the measured best run is a
  different 198-cycle trajectory (+0.20% records). Oracle arms are unachievable
  upper bounds; arm (e) is synthetic and not a claim that any predictor reaches
  0.74/0.62 on this workload.
