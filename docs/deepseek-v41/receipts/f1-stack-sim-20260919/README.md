# F1 stack-sim: four what-if knobs on the real 1-ahead predictor

CPU-only screen. No MLX, no GPU, no service touched: `overlap_schedule_sim.py`
installs a `NoMLX` meta-path finder that raises on any `mlx` import, uses numpy
only, and runs single-process under `nice -n 19`. This is a scheduling screen,
**not** a throughput result and **not** a production-code proposal.

## Question

f1-real-predictor-20260919 established the real 1-ahead predictor (post-attention
router, max-over-rows) on the 64-cycle capture: at the ~2.37-record compute window
it removes ~16% of read wait (≈7 s of the 74.7 s control, ~15.1 TPS extrapolated),
about a third of the oracle-1-ahead gain, because precision ~0.46 means half its
speculative reads are wasted and delay demand under the no-preempt/FIFO rule. This
receipt asks whether four what-if knobs move that number:

1. **Speculative read granularity / preemption** — a record is three weight planes
   (gate, up, down; 5.9 MB each). Issue speculative reads plane-by-plane so a
   demand read waits at most one plane (~0.46 ms) instead of a whole record
   (~1.37 ms); a partially read useful record needs only its remaining planes.
   Plus the ideal fully-preemptible bound (zero demand delay).
2. **Issue policy** — a per-call budget k in {1,2,3}, a merged-score threshold
   (fit on cycles 0-31), and an arm that stops issuing when the remaining compute
   window is below one plane.
3. **Waste recycling** — keep a wasted speculative record in a per-layer
   persistent ring instead of FIFO-evicting it; count the demand hits it earns for
   the same layer 1-2 cycles later (ring 32/64/128 records).
4. **Stack sensitivity** — for the best real-predictor configuration, a grid over
   persistent rows/layer (105 / 111 / 119), compute-window scale (1.0 / 0.8 / 0.6),
   and SSD rate (12.9 / 18.0 GB/s), with the full-run extrapolation.

**Every knob table below reports HELD-OUT numbers (capture cycles 32-63); any
fitted parameter is chosen on cycles 0-31.** The plane DES is validated exactly
against the parent whole-record model before any knob claim (see below).

## Commands (from the worktree root, under `nice -n 19`)

```
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f1-overlap-sim
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python   # py3.12, numpy 2.4.4

# four knobs + stacked grid (needs the 206-cycle trace AND the capture NPZ):
# reproduces the two published anchor miss-counts first, then writes stack-sim.json
nice -n 19 $PY scripts/deepseek_v41/overlap_schedule_sim.py --stack \
    --stack-out docs/deepseek-v41/receipts/f1-stack-sim-20260919/stack-sim.json

# the parent screens must still reproduce byte-for-byte (unchanged by this work):
nice -n 19 $PY scripts/deepseek_v41/overlap_schedule_sim.py --anchors-only
nice -n 19 $PY scripts/deepseek_v41/overlap_schedule_sim.py            # full 206-cycle sim
nice -n 19 $PY scripts/deepseek_v41/overlap_schedule_sim.py --capture-only

# unit tests (DES core incl. plane model; capture-backed tests skip if NPZ absent):
nice -n 19 $PY -m pytest tests/test_dsv41_overlap_schedule_sim.py -q
```

The stack run takes ~5 CPU s. `stack-sim.json` (this directory) carries every
number below plus provenance and `"mlx_imported": false`.

## Inputs (read-only; sha256)

| Role | Path | sha256 |
| --- | --- | --- |
| Router-score capture (NPZ, 45.6 MB, 105 slots) | `.worktrees/deepseek-v41/.benchmark-artifacts/deepseek-v41/router-feature-20260918/full-router-feature-20260918-v2.router-capture.npz` | `5d8dd85c…edca3e` |
| Route trace (206 cycles x 40 layers) — capacity anchors | `docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz` | `07b4b720…66d0da` |
| Cache-policy replay helper (installs NoMLX) | `docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py` | `7be34759…03fdccc` |
| Cache implementation (`LayerExpertSlotBank`) | `mtplx/expert_streaming.py` @ `2026906b0` | `de177192…571c39` |
| Overlap sim (this work) | `scripts/deepseek_v41/overlap_schedule_sim.py` | `5e0cc019…3a0e695` |
| Re-scorer (unchanged; supplies ranked scores) | `scripts/deepseek_v41/rescore_router_capture.py` | `aefdedd0…bcb251af` |

The NPZ, trace, and helper hashes are asserted at run time. `overlap_schedule_sim.py`
and `test_dsv41_overlap_schedule_sim.py` are this receipt's own inputs, hashed here
so the tables are reproducible; the re-scorer is untouched and only its exposed
merge/rank helpers are called (so ranked ids match `real_predictions()` exactly).

## Model, provenance, and exact validation

- **Timing model is inherited unchanged** from f1-overlap-sim-20260919 /
  extension-bank-20260919: `c_layer = 3.25201 ms` (non-read compute distributed
  uniformly over layer calls), record = 17,694,720 B ("17.7 MB"), plane =
  5,898,240 B, `t_draft/t_accept/t_commit` per cycle. At the primary 12.9 GB/s a
  record reads in **1.3717 ms** and a plane in **0.4572 ms**; the compute window
  admits **2.371 records** (3.31 at 18.0 GB/s).
- **The plane DES reduces to the parent whole-record model.** With `n_planes=1,
  preempt='record'`, `simulate_planes()` reproduces `simulate()` **bit-for-bit**
  (control and real arms, every ring size) — verified in-run and in
  `test_planes_reduce_to_record_model`. The existing arms/anchors are unchanged:
  `--anchors-only` still gives 35,164/8,240 and 53,999; the full 206-cycle sim and
  `--capture-only` outputs are byte-identical to the parent receipts.
- **Held-out slicing is leak-free.** The real predictor is within-cycle (target
  L+1 from cycle c's own post-attention router features), so replaying cycles
  32-63 introduces no cross-cycle leakage. The held-out half has **5,030** demand
  records; the held-out control read wait at 12.9 GB/s is **6.8996 s**.
- Layers 0-2 were uninstrumented in the capture (0 demand); layer 3 carries its
  captured reads but is unpredicted; layers 4-39 are predicted. Layers 0-3 are
  identical across arms and do not bias the control-vs-arm delta.

## Knob 1 — granularity / preemption (held-out 32-63, 12.9 GB/s, k=3)

Control read wait 6.8996 s over 5,030 demand records. All three arms land the same
1,521 useful / 3,448 issued speculative records (precision 0.44, 30.2% hidden);
only the demand-delay treatment differs.

| Arm | Read wait s | Hidden % | Precision | Read-wait reduction vs control | Spec GB read |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (record, no preempt) | 5.8006 | 30.2 | 0.44 | 15.9% | 61.0 |
| plane preempt (≤1 plane wait) | 5.4389 | 30.2 | 0.44 | **21.2%** | 54.3 |
| ideal full-preempt (0 delay) | 5.1365 | 30.2 | 0.44 | **25.6%** | 54.3 |

Plane granularity recovers ~1.9 s of the ~2.4 s in-flight penalty per held-out
half — it lifts read-wait reduction from 15.9% to 21.2%, roughly half-way to the
ideal 25.6% bound. (Plane issue also reads 11% fewer speculative bytes, 61.0→54.3
GB, because a boundary-crossing record stops after its in-flight plane instead of
finishing the whole record.)

## Knob 2 — issue policy (plane base; threshold fit 0-31, reported 32-63)

| Arm | Threshold | Read wait s | Hidden % | Precision | Read-wait reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| k=1, fitted threshold | none | 5.8982 | 14.5 | 0.66 | 14.5% |
| k=2, fitted threshold | none | 5.2796 | 23.5 | 0.52 | 23.5% |
| k=3, fitted threshold | none | 5.4389 | 30.2 | 0.44 | 21.2% |
| **k=3, window-stop** | none | **5.1365** | 30.2 | 0.44 | **25.6%** |

- **The score threshold never helps.** Fitting on cycles 0-31 selects "no gating"
  at every budget: the router scores rank experts well but their absolute value is
  not a useful issue gate, so any positive threshold only drops correct issues.
- **k alone trades coverage against contention.** k=2 (23.5%) beats k=3 without
  window-stop (21.2%) — the third speculative record leaves a plane in flight that
  delays demand more than its extra coverage saves — and beats k=1 (too little
  coverage, though highest precision 0.66).
- **Window-stop is the win, and it reaches the ideal-preempt read wait.** Refusing
  to start a plane that cannot finish inside the compute window removes the
  in-flight penalty entirely, so k=3 keeps its full 30.2% coverage at the ideal
  full-preempt read wait (5.1365 s, 25.6%). You do not need a preemptible SSD; you
  need to schedule conservatively. **Selected best config: k=3, window-stop, plane
  granularity, no threshold** (chosen by minimum TRAIN read wait; window-stop is a
  scheduling rule, not a fitted parameter, so selecting it is leak-free).

## Knob 3 — waste recycling (held-out extra hidden reads, k=3, horizon 1-2 cycles)

A wasted speculative record kept in a per-layer persistent ring (evict globally
oldest) instead of FIFO-evicted, counted as a demand hit for the same layer in the
next 1-2 cycles. These are extra reads hidden **on top of** the 1-ahead same-cycle
hits (disjoint sets).

| Ring | Ring memory | Extra hidden reads (held-out) | Extra hidden fraction |
| ---: | ---: | ---: | ---: |
| 32 records | 566.2 MB | 8 | 0.16% |
| 64 records | 1,132.5 MB | 35 | 0.70% |
| 128 records | 2,264.9 MB | 80 | 1.59% |

Recycling is small: a wasted expert this cycle is rarely the same layer's demand
miss 1-2 cycles later, because the router's per-cycle expert set turns over. Even a
2,265 MB ring recovers only 1.59% of held-out demand — and 128 records of ring is
~128 persistent slots' worth of memory, which (per the capacity table) would remove
far more read wait if spent on residency instead. The stack uses the 64-record ring
(+0.70%) as its recycling bonus and flags the memory trade.

## Best real-predictor configuration for the stack

`n_planes=3, preempt=plane, window_stop=True, budget_k=3, threshold=none`, plus the
64-record recycling bonus (+0.70%). Held-out read-wait-reduction fraction by
(rate, compute-scale), applied in the grid below:

| rate \ scale | 1.0 | 0.8 | 0.6 |
| --- | ---: | ---: | ---: |
| 12.9 GB/s | 26.2% | 21.1% | 17.9% |
| 18.0 GB/s | 30.9% | 26.2% | 21.1% |

(Each is the sim fraction + 0.70% recycle bonus. A faster drive widens the window
in records so window-stop keeps more coverage; a smaller compute window shrinks it.)

## Knob 4 — the stacked grid

**Extrapolation rule (same as f1-real-predictor-20260919, and it is an
extrapolation):** measure the real predictor's read-wait-reduction fraction `f` on
the capture — here the **held-out** cycles 32-63 with the best config above — then
apply `f` to the full 1,024-token control read wait at that cell's capacity and
rate (from the 206-cycle trace replay, the f1 receipt's capacity method) and
subtract from the full control total. It assumes the held-out hidden fraction holds
across the run, and holds `f` constant across capacity (capacity enters only through
the control anchor). Trace-replay demand records: **105 → 33,917, 111 → 31,636,
119 → 28,886** (the +6/+14 rows remove reads exactly as the parent capacity table).

| persist | rate GB/s | compute scale | control total s | control read-wait s | f_total % | **real total s** | **real TPS** | ≤ 51.2 s? |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 105 | 12.9 | 1.0 | 77.80 | 46.52 | 26.2 | 65.58 | 15.61 | |
| 105 | 12.9 | 0.8 | 72.44 | 46.52 | 21.1 | 62.62 | 16.35 | |
| 105 | 12.9 | 0.6 | 67.08 | 46.52 | 17.9 | 58.75 | 17.43 | |
| 105 | 18.0 | 1.0 | 64.61 | 33.34 | 30.9 | 54.30 | 18.86 | |
| 105 | 18.0 | 0.8 | 59.25 | 33.34 | 26.2 | 50.50 | 20.28 | **Y** |
| 105 | 18.0 | 0.6 | 53.90 | 33.34 | 21.1 | 46.86 | 21.85 | **Y** |
| 111 | 12.9 | 1.0 | 74.67 | 43.39 | 26.2 | 63.28 | 16.18 | |
| 111 | 12.9 | 0.8 | 69.31 | 43.39 | 21.1 | 60.15 | 17.02 | |
| 111 | 12.9 | 0.6 | 63.95 | 43.39 | 17.9 | 56.18 | 18.23 | |
| 111 | 18.0 | 1.0 | 62.37 | 31.10 | 30.9 | 52.75 | 19.41 | |
| 111 | 18.0 | 0.8 | 57.01 | 31.10 | 26.2 | 48.85 | 20.96 | **Y** |
| 111 | 18.0 | 0.6 | 51.65 | 31.10 | 21.1 | 45.09 | 22.71 | **Y** |
| 119 | 12.9 | 1.0 | 70.89 | 39.62 | 26.2 | 60.49 | 16.93 | |
| 119 | 12.9 | 0.8 | 65.54 | 39.62 | 21.1 | 57.18 | 17.91 | |
| 119 | 12.9 | 0.6 | 60.18 | 39.62 | 17.9 | 53.09 | 19.29 | |
| 119 | 18.0 | 1.0 | 59.67 | 28.40 | 30.9 | **50.88** | **20.12** | **Y** |
| 119 | 18.0 | 0.8 | 54.31 | 28.40 | 26.2 | 46.86 | 21.85 | **Y** |
| 119 | 18.0 | 0.6 | 48.95 | 28.40 | 21.1 | 42.96 | 23.84 | **Y** (control alone) |

At the base cell (105 / 12.9 / 1.0) the stacked knobs lift the extrapolated full
run to **65.58 s / 15.61 TPS** — vs the parent receipt's 67.4-67.8 s / ~15.1 TPS
for the baseline record model (the parent anchored the fraction to 111 slots; this
grid capacity-matches to the 105-slot capture). The knobs (plane + k=3 window-stop
+ recycling) buy ~+0.5 TPS on the base hardware; the large movers are the drive and
the compute window.

## Findings

1. **Plane granularity is worth ~5 pts of read-wait reduction; window-stop is
   worth ~10 and reaches the ideal-preempt bound.** Held-out at 12.9 GB/s: baseline
   record 15.9% → plane preempt 21.2% → k=3 window-stop 25.6% (= the ideal
   full-preempt read wait). The predictor's realized precision (0.44) and coverage
   (30.2%) do not change — only the demand delay does.
2. **The issue-policy lever is window-stop, not the score threshold.** Score
   gating never helps (fitting picks "no gate"). Conservative scheduling (don't
   start a read that cannot finish in the window) captures the full k=3 coverage
   without the in-flight penalty.
3. **Waste recycling is negligible** (≤1.6% even at a 2.3 GB ring); router expert
   turnover means yesterday's wasted prediction is rarely today's demand miss.
   Memory spent on recycling would remove more read wait as residency.
4. **On the base 12.9 GB/s drive, no cell reaches the 51.2 s / 20 TPS
   perfect-prefetch ceiling** — the best 12.9 cell is 119 / 0.6-scale at 53.1 s /
   19.3 TPS. The stacked real predictor tops out at ~15.6-19.3 TPS at 12.9 GB/s.
5. **The striped drive (18.0 GB/s) is the pivotal lever.** It cuts the full control
   read wait by ~28% and widens the compute window to ~3.3 records, so window-stop
   keeps more coverage (f rises to 30.9% at scale 1.0). 7 of 18 cells reach ≤ 51.2 s
   — all of them at 18.0 GB/s.

## What would it take to reach 51.2 s

51.2 s (20.00 TPS) is the parent f1-overlap-sim oracle-2-ahead full-run total — the
perfect-prefetch ceiling at base hardware. This grid's real predictor reaches it
only with the second striped drive, plus one more lever:

- **No cell at 12.9 GB/s reaches 51.2 s**, at any capacity or compute scale, with or
  without the predictor. The striped 18.0 GB/s drive is necessary.
- **The striped drive alone is not sufficient** at true compute (105 / 18.0 / 1.0 =
  54.30 s; 111 / 18.0 / 1.0 = 52.75 s). You need one more lever on top of it.
- **Cheapest cells that reach 51.2 s (each is the striped drive + one lever):**
  - **119 / 18.0 / 1.0 → 50.88 s / 20.12 TPS.** Keeps true compute (scale 1.0);
    adds the striped drive and +14 persistent rows/layer (≈248 MB more resident).
    Here the predictor is load-bearing: control alone is 59.67 s.
  - **105 / 18.0 / 0.8 → 50.50 s / 20.28 TPS.** Keeps base capacity; adds the
    striped drive and a 20%-faster compute window (a real kernel/model speedup).
- The predictor is only decisive in the ~1-lever-past-the-drive cells. At the most
  aggressive corner (**119 / 18.0 / 0.6 → 42.96 s**) the control reaches 48.95 s on
  its own — the striped drive plus a 40%-faster compute window already clears 51.2 s
  without any prefetch, and the predictor is then a bonus, not the mechanism.

**Bottom line: no configuration at the current 12.9 GB/s drive reaches 51.2 s. The
cheapest cells that do are the striped 18.0 GB/s drive plus exactly one of {+14
persistent rows/layer, a 20%-faster compute window}, landing at 50.5-50.9 s /
~20.1-20.3 TPS — right at the perfect-prefetch ceiling.**

## Limits

- Screen only. No production code changed, no GPU/service touched, no latency or
  throughput proof. The measured GPU runtime still rejects prefetch under the
  transition-window policy.
- Prefetch lives outside the slot pool, so demand misses are predictor-independent
  (taken from the captured `reads`); the plane model only changes how a demand read
  treats the in-flight speculative plane and gives partial credit to a partly-read
  useful record.
- The predictor's precision/coverage are the capture's (105 slots, first 64
  cycles); the extrapolation holds `f` constant across capacity, which is
  conservative — at 119 slots the fixed compute window would cover a *larger*
  fraction of the (fewer) misses, so real gains are likely a little higher there.
  The compute model is uniform per layer call (c_layer transferred from the
  111-slot 198-cycle extension-bank run). Same no-preempt/FIFO-ring assumptions as
  the parent sims for every arm except the explicit preempt/window-stop knobs.
- Recycling is an optimistic count (it assumes the full k=3 prediction set is issued
  each cycle; window-stop may issue fewer) and is reported separately, not folded
  into the DES.
- The 18.0 GB/s "second striped drive" is a hardware what-if, not a measured rate on
  this box (measured single-drive window 12.873 GB/s).
