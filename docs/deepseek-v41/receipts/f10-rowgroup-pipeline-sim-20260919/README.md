# F10: row-group pipeline discrete-event simulation of M6 verify decode

CPU-only screen. No MLX, no GPU, no service touched: `rowgroup_pipeline_sim.py`
installs a `NoMLX` meta-path finder that raises on any `mlx` import, uses numpy
only, and runs single-process under `nice -n 19`. This is a **scheduling screen,
not a throughput result and not a production-code proposal.** `"mlx_imported":
false` in `results.json`.

## Idea under test (David's; simulated, not judged by intuition)

Each routed verify layer is serial today: GPU stage **G** (hyper-connections +
attention + gate, ends in the routing-index host sync) -> host route prep -> SSD
reads of the missing experts (GPU idle but for hit experts) -> miss-expert compute
**E** + the next layer's G. "Row-group pipelining" splits the M=6 verify rows into
two causal groups (A = rows 0-2, B = rows 3-5) and runs them as a two-stage
pipeline **staggered by one layer** on a single host thread, so one group's GPU
stage always overlaps the other group's SSD reads. Every read stays an exact demand
read; routing, experts and arithmetic are row-independent and unchanged -- only M
per kernel call changes. The fixed host program (A leads B by one layer) is in the
brief and in `simulate()`.

## Commands (from the worktree root, under `nice -n 19`)

```
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f10-pipeline-sim
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python   # py3.12, numpy 2.4.4

# reproduce the two published cache anchors exactly, then stop:
nice -n 19 $PY scripts/deepseek_v41/rowgroup_pipeline_sim.py --anchors-only

# full simulation (writes results.json next to this README, prints every table):
nice -n 19 $PY scripts/deepseek_v41/rowgroup_pipeline_sim.py

# unit tests (DES core on hand-built streams; cache-replay tests skip if trace absent):
nice -n 19 $PY -m pytest tests/test_dsv41_rowgroup_pipeline_sim.py -q
```

Full simulation runs in ~5 CPU s; `results.json` carries every number below plus
provenance. The cache replay reuses the f1 sim's validated helpers unchanged
(`overlap_schedule_sim.make_bank/load_trace/anchor_checks`); the f1 sim itself is
imported, not modified.

## Inputs (read-only; sha256 asserted at run time where noted)

| Role | Path | sha256 |
| --- | --- | --- |
| Route trace (206 cycles x 40 layers, 36 ids = 6 rows x 6 experts) | `docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz` | `07b4b720…66d0da` (asserted) |
| Cache-policy replay helper (installs NoMLX) | `docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py` | `7be34759…03fdccc` |
| Cache implementation (`LayerExpertSlotBank`) @ `631fc856f` | `mtplx/expert_streaming.py` | `de177192…571c39` |
| Sibling f1 sim (imported for constants + helpers) | `scripts/deepseek_v41/overlap_schedule_sim.py` | `5e0cc019…3a0e695` |
| This sim | `scripts/deepseek_v41/rowgroup_pipeline_sim.py` | `2ca4c2cf…2b64a0` |
| This test | `tests/test_dsv41_rowgroup_pipeline_sim.py` | `0c801a17…72ca41` |

## Model, decomposition, and calibration

The f1 receipt lumps all non-read compute into `C_LAYER = 3.25201 ms/layer` (=
(verify 69.155 s - read-union 43.399 s) / (40x198 layer calls)) and reads serially
after it. F10 **decomposes** that lump into a discrete-event schedule over three
resources -- one **host thread**, one FIFO **GPU queue**, one FIFO **SSD server** --
so a group's GPU stage can overlap the other group's reads:

- **G** = `G_full` 2.05 ms (control) or `G_half` in {1.2,1.5,1.8,2.05} ms (pipeline;
  a half batch is *not* half the cost -- small-op chains are launch-bound). Ends in
  the routing-index sync (the barrier).
- **host_pre** 0.35 ms (route prep, after each barrier) and **host_post** 0.701 ms
  (post-read host + graph build, before each E_miss). Both paid **per group** in the
  pipeline (doubled per layer).
- **Expert compute is per-assignment** (coordinator addendum): an assignment is one
  (row,expert) pair, 6/row. `E_hit = hit_assignments x c_assign` runs on the GPU
  right after `submit_reads` (while that group's records read); `E_miss =
  miss_assignments x c_assign` runs on the GPU after `finish`. `c_assign` in
  {0.025,0.035,0.045} ms (central 0.035; 36 x 0.035 = 1.26 ms/layer of expert
  compute, matching Codex's ~1.3-1.5 ms layer-34 replay for ~37 assignments). A
  group's **next G depends on that group's E_hit and E_miss completing** (FIFO
  submission order enforces it; the MoE output feeds the next layer).
- Per-cycle fixed = the f1 skeleton verbatim: draft 9.954 + accept 0.344 + commit
  0.771 ms/cycle, one-time growth 2.196 s. **Head is inside VERIFY_S -> C_LAYER**, so
  there is no separate head term (avoids double-count; David's "25 ms/cycle incl.
  head" folds growth-amortized ~11 ms + draft/accept/commit ~11 ms + the head that
  lives in verify). Pipeline adds **+3 ms/cycle** for split/concat + the second syncs.
- 206-cycle trace replayed with per-198-cycle rates, **exactly as the f1 sim**;
  `TPS = 1024/total`.

**host_post is the one calibrated constant.** It is set once, at c_assign=0.035, so
the mean per-layer non-read critical path (`G_full + host_pre + host_post +
mean_miss_assign x c_assign`, with mean miss-assign 4.318) equals C_LAYER exactly ->
**host_post = 0.7009 ms** (G_full and host_pre are the brief's givens). In the
pipeline it is paid per group, which conservatively assumes the C_LAYER residual is
per-group host/build work.

### Cache replay validated exactly (before any timing claim)

`--anchors-only` reproduces the two published miss-counts with the real
`LayerExpertSlotBank` (via the f1 helpers): prefix-readiness-20260918
**35,164 misses / 8,240 calls @ cap102**, and mtp-verify-routes-20260913
**53,999 records @ cap73+48** -- both exact. The single-route control at 111+48,
transition-window reproduces the **31,636 demand records** of the f1 control exactly.

### Control reconciliation vs the measured run

The control is one group / full 36-id route / `G_full`, no extra syncs.

| Line | total s | TPS | verify ms/cyc | read wait s | records |
| --- | ---: | ---: | ---: | ---: | ---: |
| Measured retained run (extension-bank) | 73.762 | 13.88 | 349.3 | 43.399 | 31,573 |
| f1 control anchor (206c @ 111+48, 12.9) | 74.667 | 13.71 | 349.3 | 43.395 | 31,636 |
| **F10 control, c_assign=0.035** | **74.992** | **13.65** | 349.3 | **43.39** | **31,636** |
| F10 control, c_assign=0.025 / 0.045 | 74.427 / 75.558 | 13.76 / 13.55 | 349.3 | 43.39 | 31,636 |

F10's control is **+0.43% vs the f1 anchor, +1.67% vs the measured run** (both within
a few %). The blocked-in-finish time equals the f1 read wait to the ms (43.39 s). The
+0.43% is real and expected: **E_hit does not hide on the 7.1% of layer-calls with
zero demand misses** (no read to overlap under), spilling ~0.3 s at c=0.035; it is
the only place the addendum's "E_hit stays hidden under the read wait" fails.

## The split changes what the cache sees (two route observations/layer)

The pipeline replays each layer as **two** `plan()` calls (group A's 18 ids, then
group B's 18 ids) on the same bank, so the transition-window policy makes two
observations/layer and B sees A's just-admitted experts as residents (an expert
needed by both groups is read once, by A). This is not free:

| Capacity | single-route records | two-route records | delta | zero-miss A / B / either | (single zero-miss) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 111 + 48 | 31,636 | **32,559** | **+2.92%** | 21.9% / 20.3% / 35.4% | 7.1% |
| 115 + 48 | 30,235 | **31,053** | **+2.71%** | 23.2% / 22.0% / 37.7% | 8.0% |

The split adds ~2.9% demand records (~1.3 s of SSD work the pipeline must still pay)
and triples the zero-miss rate: **35.4% of layer-calls have at least one group with
no misses**.

## Main grid: G_half x SSD rate x capacity (c_assign = 0.035, +48 transient)

Pipeline seconds/run, TPS, seconds removed vs the same-capacity control, SSD/GPU busy
fraction, the binding resource, host blocked-in-finish vs blocked-in-barrier, and the
per-layer period p10/p50/p90.

| persist | rate | G_half | ctrl s | **pipe s** | **removed** | pipe TPS | SSD% | GPU% | bound | blkFin s | blkBar s | period p10/p50/p90 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| 111 | 12.9 | 1.2 | 74.99 | 62.31 | **+12.68** | 16.43 | 72 | 48 | SSD | 18.8 | 21.0 | 4.5/5.5/11.0 |
| 111 | 12.9 | 1.5 | 74.99 | 64.72 | **+10.28** | 15.82 | 69 | 54 | SSD | 16.3 | 26.0 | 5.1/5.5/11.0 |
| 111 | 12.9 | 1.8 | 74.99 | 67.72 | +7.27 | 15.12 | 66 | 59 | SSD | 14.3 | 30.9 | 5.7/5.8/11.1 |
| 111 | 12.9 | 2.05 | 74.99 | 70.59 | +4.40 | 14.51 | 63 | 63 | SSD | 13.1 | 35.1 | 6.2/6.3/11.1 |
| 111 | 14.0 | 1.2 | 71.58 | 59.27 | +12.31 | 17.28 | 69 | 51 | SSD | 15.8 | 21.0 | 4.5/5.1/10.1 |
| 111 | 14.0 | 1.5 | 71.58 | 61.99 | +9.60 | 16.52 | 66 | 57 | SSD | 13.6 | 26.0 | 5.1/5.2/10.2 |
| 111 | 14.0 | 1.8 | 71.58 | 65.43 | +6.15 | 15.65 | 63 | 61 | SSD | 12.1 | 30.9 | 5.7/5.8/10.2 |
| 111 | 14.0 | 2.05 | 71.58 | 68.31 | +3.28 | 14.99 | 60 | 65 | **GPU** | 10.8 | 35.1 | 6.2/6.3/10.2 |
| 115 | 12.9 | 1.2 | 73.06 | 60.93 | +12.13 | 16.81 | 70 | 49 | SSD | 17.5 | 21.0 | 4.5/5.1/10.7 |
| 115 | 12.9 | 1.5 | 73.06 | 63.45 | +9.61 | 16.14 | 67 | 55 | SSD | 15.1 | 25.9 | 5.1/5.4/11.0 |
| 115 | 12.9 | 1.8 | 73.06 | 66.56 | +6.50 | 15.39 | 64 | 60 | SSD | 13.2 | 30.9 | 5.7/5.8/11.0 |
| 115 | 12.9 | 2.05 | 73.06 | 69.51 | +3.54 | 14.73 | 61 | 64 | **GPU** | 12.1 | 35.0 | 6.2/6.3/11.0 |
| 115 | 14.0 | 1.2 | 69.80 | 58.07 | +11.73 | 17.64 | 68 | 52 | SSD | 14.6 | 21.0 | 4.5/4.9/10.0 |
| 115 | 14.0 | 1.5 | 69.80 | 60.89 | +8.91 | 16.82 | 64 | 58 | SSD | 12.5 | 25.9 | 5.1/5.2/10.1 |
| 115 | 14.0 | 1.8 | 69.80 | 64.43 | +5.36 | 15.89 | 61 | 62 | **GPU** | 11.1 | 30.9 | 5.7/5.8/10.1 |
| 115 | 14.0 | 2.05 | 69.80 | 67.39 | +2.41 | 15.19 | 58 | 66 | **GPU** | 10.0 | 35.0 | 6.2/6.3/10.1 |

The pipeline collapses the p50 per-layer period from the control's 7.32 ms to
5.1-6.3 ms (the read wait moves off the critical path); the p90 stays ~10-11 ms (the
heavy-read layers that still stall). `blkFin` (exposed read wait) falls as G_half
grows because more GPU work overlaps reads; `blkBar` rises because the host waits
longer on the bigger G stage.

## c_assign sensitivity (primary cell: G_half 1.5, 12.9 GB/s, cap 111)

| c_assign ms | ctrl s | pipe s | removed | GPU% | SSD% | bound |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.025 | 74.43 | 64.56 | +9.86 | 50 | 69 | SSD |
| 0.035 | 74.99 | 64.72 | +10.28 | 54 | 69 | SSD |
| 0.045 | 75.56 | 65.26 | +10.30 | 58 | 68 | SSD |

Expert-compute cost barely moves the primary cell's removed seconds (9.9-10.3 s): at
G_half=1.5 the pipeline is SSD-bound across the whole c_assign range, so raising
c_assign lifts the control and pipeline together.

## Imbalance analysis

Splitting 3.84 misses/layer across two groups of ~1.9 leaves **35.4% of layer-calls
with a zero-miss group** (vs 7.1% single-route). Those layers are actually a little
*faster* (primary-cell mean period 6.40 ms on imbalanced layers vs 7.52 ms balanced)
because they carry fewer reads -- the split loses no wall-time there directly. What it
loses is **overlap efficiency**: a zero-miss group still runs G_half + E_hit on the
shared GPU with no same-group read to hide under, so that GPU work is unhidden and
pushes the cell toward GPU-bound. This, plus the +2.9% records and the doubled G +
extra syncs, is why the pipeline tops out at ~10 s removed rather than the ~24 s that
the two-route SSD floor (32,559 x 1.372 ms = 44.7 s + fixed ~5 s ~= 50 s) would allow
under perfect overlap: at G_half=1.5 the pipeline is 64.7 s, ~15 s above that floor
(blkFin 16.3 s exposed read wait + blkBar 26.0 s GPU-stage waits + host 17.3 s).

## Byte-lever sensitivity (synthetic record thinning; cap 111, 12.9, c 0.035)

The two byte levers from earlier receipts are modelled as **synthetic random thinning
of demand records** (-5/-10/-20%, seed 20260919; SSD load only, miss-assignment
compute unchanged) -- label: SYNTHETIC. This shows when the **GPU**, not the SSD, sets
the period.

| G_half | thinning | ctrl s | pipe s | removed | records | GPU% | SSD% | bound |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1.5 | 0% | 74.99 | 64.72 | +10.28 | 32,559 | 54 | 69 | SSD |
| 1.5 | -5% | 72.89 | 63.25 | +9.64 | 30,950 | 55 | 67 | SSD |
| 1.5 | -10% | 70.80 | 61.86 | +8.94 | 29,350 | 57 | 65 | SSD |
| 1.5 | -20% | 66.55 | 59.17 | +7.38 | 26,097 | 59 | 60 | ~crossover |
| 1.8 | 0% | 74.99 | 67.72 | +7.27 | 32,559 | 59 | 66 | SSD |
| 1.8 | -5% | 72.89 | 66.36 | +6.53 | 30,950 | 60 | 64 | SSD |
| 1.8 | -10% | 70.80 | 65.08 | +5.73 | 29,350 | 62 | 62 | ~crossover |
| 1.8 | **-20%** | 66.55 | 62.61 | **+3.94** | 26,097 | **64** | 57 | **GPU** |

The byte levers and the pipeline are **partly redundant** -- both attack read wait --
so the removed seconds *shrink* as records are thinned (the thinning helps the fully
exposed control more than the already-overlapped pipeline). At G_half=1.5 the primary
cell reaches the SSD/GPU crossover only at -20%; at G_half=1.8 the -20% cell is
clearly **GPU-bound** (GPU 64% > SSD 57%). Beyond that the two G-halves + 36
assignments of expert compute per layer set the floor, and more bytes saved buy
nothing.

## Findings

1. **The pipeline removes real time and it is read-overlap, not compute.** At the base
   cell (111 / 12.9 / c 0.035): G_half=1.2 -> +12.7 s (16.4 TPS); **G_half=1.5 ->
   +10.3 s (15.8 TPS)**; G_half=1.8 -> +7.3 s; G_half=2.05 (no half-batch speedup at
   all) -> +4.4 s. Even the pure launch-bound case (G_half=G_full) removes 4.4 s
   because staggering by one layer lets each group's read wait hide behind the other
   group's GPU stage.
2. **The split is not free: +2.9% demand records and 35% zero-miss layers.** Two route
   observations/layer make the transition-window policy admit slightly worse (+923
   records at cap 111), and half-width routes triple the zero-miss rate; both cap the
   gain well short of the two-route SSD floor (~50 s / ~20 TPS).
3. **On 12.9 GB/s the pipeline stays SSD-bound except at G_half=2.05.** GPU busy
   crosses SSD busy only when G is not sped up at all, or with the 14.0 GB/s drive, or
   under >=~10-20% record thinning at G_half>=1.8. This is exactly where the addendum's
   "GPU is now far busier (two G stages/layer)" bites: the second G stage plus the
   full 36-assignment expert compute is what the pipeline cannot overlap.
4. **c_assign barely matters at the primary cell** (removed 9.9-10.3 s across
   0.025-0.045) because it is SSD-bound there; it matters at the GPU-bound corners,
   where a higher c_assign deepens GPU-boundedness.
5. **A faster drive helps the control more than the pipeline** (14.0 vs 12.9 removes
   ~2 s of the *gap*), and record thinning shrinks the gap outright -- the pipeline and
   the byte levers are substitute levers for the same read wait, not additive ones.

## Verdict

**At G_half = 1.5 ms the two-group row-group pipeline removes 10.3 s of the 75.0 s
control (74.99 -> 64.72 s; 13.65 -> 15.82 TPS) at 111+48 slots / 12.9 GB/s /
c_assign 0.035** -- bracketed by +12.7 s (16.4 TPS) at G_half 1.2 and +4.4 s (14.5
TPS) at the pure launch-bound G_half 2.05; the gain is read-overlap and it is capped
by the +2.9% split records, the 35% zero-miss layers, and the doubled GPU stage,
which turns the cell GPU-bound under a faster drive, a higher c_assign, or >=20%
byte-lever thinning.

## Assumptions (every one is an assumption)

- **A1.** Timing skeleton (C_LAYER, record bytes, growth, draft/accept/commit,
  206-cycle-with-198-rates scaling, TPS=1024/total) is inherited verbatim from f1 /
  extension-bank-20260919; the control reproduces the f1 anchor to +0.43%.
- **A2.** `G_full`=2.05, `host_pre`=0.35 are the brief's givens; `host_post`=0.7009
  is calibrated once (A6) and paid per group in the pipeline.
- **A3.** Expert compute = assignments x c_assign (E_hit on the GPU during reads,
  E_miss after; a group's next G waits for both via FIFO). c_assign swept
  {0.025,0.035,0.045} ms, central 0.035.
- **A4.** The head is inside VERIFY_S -> C_LAYER (no separate head term); the pipeline
  pays +3 ms/cycle for split/concat + the second syncs.
- **A5.** One FIFO GPU queue, one FIFO SSD server (rate rd/record, a group's layer
  misses submitted as one contiguous batch, no preemption -- the f1 convention), one
  host thread that blocks at each barrier (on the group's G) and each finish (on the
  group's read batch). At each cycle end the host waits for the last MoE output before
  head/accept/commit.
- **A6.** host_post calibrated so mean(G_full + host_pre + host_post +
  mean_miss_assign x c_assign) = C_LAYER at c_assign=0.035, using the single-route
  mean miss-assignment 4.318.
- **A7.** Two route observations/layer (A then B on the same bank) is the exact
  transition-window policy; an expert needed by both groups is read once (by A) and is
  a hit for B, because A's read batch for layer L is submitted before B's on the FIFO
  SSD, so it completes before B's finish(L) -- no extra wait modelled.
- **A8.** A leads B by one layer is the only order that keeps a single open decode
  route/layer (the runtime's limit). Records come from replaying the 206-cycle trace;
  compute rates are the 198-cycle measured totals (as f1).
- **A9.** Byte levers = synthetic random record thinning (seed 20260919), SSD load
  only, compute unchanged -- **labelled synthetic**, not a claim any lever achieves it.
- **A10.** Prefetch/prediction is absent by design: every read is an exact demand read.

## Limits and variants skipped

- **Screen only.** No production code changed, no GPU/service touched, no latency or
  throughput proof. It shows the *scheduling* upper bound of demand-only row-group
  pipelining under the measured component costs.
- **Half-step variant (B.barrier(L) before A.finish(L))**: opens B's route on layer L
  before A's route on L closes -> **>1 open decode route/layer, which the runtime
  forbids**. Not simulated (marked infeasible, per the brief).
- **3-group variant**: not a trivial generalisation -- three groups touching layer L
  per iteration need >1 open route (same runtime limit) and an unmeasured `G_third`
  launch-bound cost. **Skipped.**
- The +0.43% control excess (E_hit unhidden on zero-miss layers) and the per-group
  doubling of host_post are conservative against the pipeline. The 14.0 GB/s drive is
  a hardware what-if (measured single-drive window 12.873 GB/s).
- Pre-existing on the base branch (f1/overlap-sim @ 631fc856f), unrelated to this work:
  `tests/test_dsv41_scripts_no_undefined_names.py` fails on 3 F821 `Undefined name
  'sys'` in `ab_decode_env_levers.py`. This sim and its test are F821- and
  full-ruff-clean.

---

# Round 2: better schedules, splits, policy fix, host cost, lower bound

Same inputs/rules (CPU-only, numpy, `nice -n 19`, no MLX/GPU). Round 1 is
unchanged and still reproduces. A NEW event-driven engine (`simulate2`) replaces
the fixed alternation with a single host thread, one FIFO GPU queue, and a
plane-granular SSD (plane = 1/3 record). Run: same command as Round 1 (the full
sim now also prints and stores the `round2` block; ~16 CPU s).

## Correction (carried in)

The row-group lever is **NOT bit-identical**. The routed per-expert gather is
M-invariant, but the gate (`xf @ weight.T`) and the shared expert are **not
invariant to the row batch size M** (`mtplx/models/deepseek_v41_moe.py`
`combine_routed` docstring L446-449: "NOT invariant to the row (M) batch size, so
batching them ... reassociates their fp32 reductions and flips greedy argmax";
`deepseek_v41.py` `_layer_major_moe`). Splitting M=6 into 3+3 changes M per gate/
shared-expert call, so the lever is **rounding-class (tie-flip audit required)** —
greedy argmax can flip. Round 1's "outputs identical by construction" is withdrawn.

## Engine validation

`simulate2` in fixed/FIFO mode reproduces the control **74.992 s exactly** and the
Round-1 fixed pipeline to **<0.05%** (G1.5 64.690 vs 64.72; the 0.03 s is corrected
empty-batch semantics — an empty read batch now completes immediately instead of
waiting for the SSD to drain). Every arm is asserted `>=` its analytic lower bound.

## Schedules at the primary cell (G_half 1.5, 12.9 GB/s, cap 111, c_assign 0.035; control 74.99 s, LB 49.75 s, binding SSD)

| Schedule | sec/run | removed | TPS | SSD% | GPU% | host% | period ms | slack vs LB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 fixed / FIFO (= Round 1) | 64.69 | **+10.30** | 15.83 | 69 | 54 | 27 | 7.23 | 14.94 |
| S1 fixed / leader-priority | 69.50 | +5.50 | 14.73 | 64 | 51 | 25 | 7.82 | 19.74 |
| S2 wc / FIFO / blocking | 66.83 | +8.16 | 15.32 | 67 | 53 | 26 | 7.49 | 17.08 |
| **S2i wc / FIFO / interruptible** | **62.73** | **+12.26** | **16.32** | 71 | 56 | 28 | 6.99 | 12.98 |
| S3 wc / leader / blocking | 69.97 | +5.02 | 14.63 | 64 | 50 | 25 | 7.87 | 20.21 |
| S3i wc / leader / interruptible | 66.83 | +8.16 | 15.32 | 67 | 53 | 26 | 7.49 | 17.08 |
| (bonus) wc / trailer / blocking | 71.79 | +3.20 | 14.26 | 62 | 49 | 24 | 8.09 | 22.03 |
| (bonus) wc / trailer / interruptible | 69.71 | +5.28 | 14.69 | 64 | 50 | 25 | 7.84 | 19.95 |

Binding is analytic (schedule-independent): at G1.5 the components are SSD 44.66,
chain-B 44.50, chain-A 43.42, GPU 35.10, host 17.32 s — **SSD-bound, but the two
group chains are within 0.3% of it** (the 3:3 split is near-perfectly balanced).

## Schedule totals across G_half (removed vs 74.99 s in parens)

| Schedule | G_half 1.2 (LB 49.75, SSD) | 1.5 (LB 49.75, SSD) | 1.8 (LB 52.07, chain-B) |
| --- | ---: | ---: | ---: |
| S0 fixed / FIFO | 62.29 (+12.70) | 64.69 (+10.30) | 67.69 (+7.30) |
| S1 fixed / leader | 67.79 (+7.20) | 69.50 (+5.50) | 72.81 (+2.19) |
| S2 wc / FIFO / block | 63.81 (+11.18) | 66.83 (+8.16) | 70.16 (+4.84) |
| **S2i wc / FIFO / intr** | **60.41 (+14.59)** | **62.73 (+12.26)** | **65.93 (+9.06)** |
| S3i wc / leader / intr | 64.35 (+10.65) | 66.83 (+8.16) | 70.20 (+4.79) |

As G_half grows, the binding shifts from the SSD (44.66 s, fixed) to group B's
dependency chain (G+route+read+E_miss+host), which passes the SSD at G1.8 — more
GPU per layer lengthens each group's serial path.

## S4 — row splits (best schedule wc/FIFO/interruptible @ G1.5)

| Split (A:B rows) | records (vs 31,636) | miss share A/B | zero-miss A/B | either-zero | best sched s (removed) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 : 4 | 32,536 (+2.84%) | 0.33 / 0.67 | 0.34 / 0.14 | 0.41 | 64.67 (+10.32) |
| **3 : 3** | 32,559 (+2.92%) | 0.49 / 0.51 | 0.22 / 0.20 | **0.35** | **62.73 (+12.26)** |
| 4 : 2 | 32,581 (+2.99%) | 0.65 / 0.35 | 0.15 / 0.31 | 0.39 | 63.73 (+11.26) |

The balanced 3:3 split minimises either-group-zero (0.35) and is fastest; skewed
splits create a zero-miss-heavy small group and a read-heavy large one — but
because the SSD is shared, the imbalance does not buy overlap (priority does not
help either group, below).

## S5 — policy-observation fix (feed one union route observation/layer)

Keeping per-group admission but feeding the transition-window **one union
observation per layer** (instead of two): records **31,672 — +0.11% vs the 31,636
single-route control** (naive two-route 32,559, +2.92%). It recovers ~97% of the
split's read penalty. Schedule effect at G1.5: S0 64.69 -> 63.87 s; S2i 62.73 ->
61.68 s (~0.8-1.0 s). The residual +36 records is the doubled decode-epoch/recency
touches, which this fix leaves per-group by design.

## S6 — host-cost sensitivity (host_pre + host_post at 100% / 50%, G1.5)

| host cost | S0 fixed/FIFO | S2i wc/FIFO/intr |
| --- | ---: | ---: |
| 100% (0.35 / 0.701 ms) | 64.69 s (+10.30) | 62.73 s (+12.26) |
| 50% | 61.62 s (+13.37) | 60.50 s (+14.49) |

Halving the per-group host cost buys ~2-3 s (host busy 27% -> 14%) — a real lever
because the host work is doubled by the split and sits partly on each group's chain.

## Analytic lower bound and remaining slack

The schedule-independent lower bound is `per-cycle fixed + max(SSD, GPU, host,
chain-A, chain-B)`. At G1.5 = **49.75 s** (SSD-bound). The best realistic schedule
(64.69 s) leaves **14.9 s of slack**; the best idealised (62.73 s) leaves 13.0 s.
That slack is **SSD under-utilisation** (SSD busy only 69-71%): the demand-read
dependency (G -> routing sync -> read -> E_miss) keeps the SSD from saturating, and
**no host schedule or SSD priority closes it** — only prefetch (the rejected f1
lane) or an interruptible barrier can. Best-idealised (S2i) sensitivity: capacity
115 -> 61.25 s; byte-thin -10% -> 59.45 s; -20% -> 56.29 s.

## Can 3 groups beat 2? No.

Three groups (2:2:2, G_third = G_half = 1.5 ms — an optimistic UNDER-estimate;
a third batch is even more launch-bound): wc/FIFO/blocking **81.79 s (-6.80, worse
than the control)**; wc/FIFO/interruptible 66.08 s (+8.91) — both lose to the
2-group best (62.73 s). Three G stages/layer plus a deeper A->B->C dependency chain
and more zero-miss groups outweigh the finer read overlap. Records rise to 32,835.

## Round 2 findings

1. **Leader-priority SSD (S1) HURTS at every G_half** (G1.5: +5.50 vs S0 +10.30).
   A already leads, so the trailing group B is the critical path; serving A first
   starves the bottleneck. **Trailer-priority is worse still** (+3.20), and with
   the balanced 3:3 split neither chain has slack to trade, so **FIFO is optimal**.
2. **Work-conserving with blocking barriers (S2) is worse than the fixed schedule**
   (66.83 vs 64.69) under the specified preference (A.finish > A.barrier > B.finish):
   the host commits to A's blocking barrier when it could run B. A finish-first
   preference recovers only ~0.7 s. **The fixed Round-1 schedule is the best
   realistic (blocking-barrier) schedule.**
3. **The one real lever is the interruptible barrier** (S2i): +12.26 s / 16.32 TPS,
   ~+2 s over the fixed schedule. That is a **runtime change** (make the
   routing-index sync non-blocking so the host can react to reads during a G stage),
   not a schedule change — it is the "non-blocking eval" bound.
4. **The 3:3 split is near-optimally balanced** (chains within 0.3% of the SSD),
   which is why priority and skewed splits do not help; the +2.9% read penalty is
   almost fully removed by the S5 union-observation fix (+0.11% vs control).
5. **~13-15 s of slack above the LB is unrecoverable by scheduling** — it is SSD
   under-utilisation from the demand-read dependency chain; the byte levers and host
   trim chip at it, but only prefetch or an interruptible barrier reaches the SSD floor.

## Round 2 verdict

**At G_half = 1.5 ms the best REALISTIC schedule is still the fixed Round-1
alternation (S0 fixed/FIFO): 64.69 s / 15.83 TPS (+10.30 s). Leader-priority (the
S1 hypothesis) and work-conserving-with-blocking both do worse; the only schedule
lever that beats it is an INTERRUPTIBLE routing-sync barrier (S2i wc/FIFO/interruptible:
62.73 s / 16.32 TPS, +12.26 s), a runtime change. All schedules sit 13-15 s above
the 49.75 s analytic lower bound (SSD-bound, chains balanced within 0.3%); that slack
is SSD under-utilisation from the demand-read dependency and is not closed by host
scheduling or SSD priority. 3 groups lose to 2 (66.08 s best). The row-group lever
is rounding-class (tie-flip audit required), not bit-identical.**

## Round 2 assumptions (additional)

- **R1.** `simulate2` resources: one host thread; one FIFO GPU queue (stage starts at
  max(queue-free, submit, deps); G(g,L) after G(g-1,L) by submission order = KV
  order); one plane-granular SSD (plane = rd/3). Empty read batch completes
  immediately (corrected vs Round 1's drain-wait; control unaffected, pipeline <0.05%).
- **R2.** Work-conserving host picks the highest-preference ENABLED action
  (A.finish, A.barrier, B.finish, B.barrier); a group opens its route on L only after
  the previous group closed L (its barrier on L+1, or forward-end at L=39). A blocking
  barrier commits the host until that G stage ends (it cannot react to reads meanwhile);
  the interruptible variant enables a barrier only once its G stage is already done.
- **R3.** leader/trailer-priority preempt only at plane boundaries (an in-flight plane
  is never cut); FIFO never preempts. trailer-priority is a bonus corrective, not in
  the brief.
- **R4.** S5 suppresses the per-group transition-window OBSERVATION and feeds one union
  observation/layer; all other per-group state (decode epoch, touch, recency, admission)
  is unchanged — hence the +0.11% residual, not exactly 0.
- **R5.** 3-group uses G_third = G_half = 1.5 ms, a lower bound on its true (more
  launch-bound) cost; the 3-group verdict therefore holds a fortiori.
- **R6.** The analytic lower bound assumes E_hit fully hidden (optimistic) and treats
  per-cycle fixed as non-overlappable; it is a valid lower bound on the two-group total.
