# F2 next-layer expert prefetch — build (CPU-verified, GPU window pending)

A next-layer expert prefetch lane for the D5/M6 verify decode of the Q4 (native
mxfp4) runner, composed onto the retained best run (extension bank 84+27 rows,
predictable projection expansion, native KV16, D5 + two lookup tokens). This is a
**build + CPU verification**: no GPU/Metal was touched (another session holds the
lock), no service was touched, and **no throughput/latency claim is made**. The
GPU window that would measure it is written (`scripts/deepseek_v41/
run_f2_prefetch_window.sh`) but not run. `20 TPS remains unmet`; the retained best
run is 13.8688167379 TPS / 73.7626013330 s (extension-bank-20260919).

## Predictor and schedule (exactly the f1 winners)

* **Predictor** = f1-real-predictor-20260919: at source layer L, apply layer
  **L+1**'s gate (the model's native FP32 router projection + score function, no
  separately-fitted correction) to L's **post-attention** router input — the
  tensor L's own gate consumes — merge the per-verify-row scores by **MAX**, rank
  descending, **exclude experts that are READY residents of L+1**, keep the top
  **k = 3**. Targets 4..39 are predicted from sources 3..38; layers 0..3 are
  unpredicted. (Post-attention · max is f1-real-predictor's best input+rule at
  every budget; k=3 matches the ~2.37-record compute window; the score threshold
  is not used because fitting always picks "no gate".)
* **Schedule** = f1-stack-sim-20260919's selected config `n_planes=3,
  preempt=plane, window_stop=True, budget_k=3, threshold=none`: issue **after** L's
  demand reads and resident/shared GPU work are submitted; speculative reads go
  **plane-by-plane** (gate/up/down) behind demand priority; **window-stop** = do
  not start a plane once L's demand reads have all completed and the remaining
  compute window is shorter than one plane read. A speculative record that becomes
  demanded is **promoted** (only its unread planes read). Ring default **32
  records**, memory charged in admission (17,694,720 B each).
* Results depend only on the true route: outputs are computed from `indices`; a
  mispredict only wastes a speculative read.

## Files

New, all on branch `f2/next-layer-prefetch` (nothing else changed — the mature
`expert_runtime.py` / `deepseek_v41.py` demand hot path is untouched, so the
retained best run stays byte-identical with the lane off):

| File | Role | MLX? | CPU-tested |
| --- | --- | :--: | :--: |
| `scripts/deepseek_v41/f2_predictor.py` | ranking/merge/exclusion, window-stop arithmetic, plane-granular ring scheduler (lease/promotion/eviction/failure-drain), ring budget charge | no | yes |
| `tests/test_dsv41_f2_prefetch.py` | 12 CPU tests (below) | no (MLX hard-blocked) | — |
| `docs/deepseek-v41/receipts/f2-prefetch-build-20260919/f2_prefetch_lane.py` | GPU install glue: next-layer predictor riding the routing sync + plane reader binding + frozen-config guard + admission charge (staged like lookahead-io `plane_lane.py`) | yes | **no (window)** |
| `docs/deepseek-v41/receipts/f2-prefetch-build-20260919/f2_window_arms.py` | A/B/A arm driver inside the window (drives the real extension-bank `launch_full.py` per arm, digest gate) | no | compiles only |
| `scripts/deepseek_v41/run_f2_prefetch_window.sh` | the guarded window (write, do NOT run) | — | `bash -n` only |

The pure logic that the GPU lane executes (predictor algebra, window-stop, ring
scheduling, budget) lives in `f2_predictor.py` and is imported by both the GPU
lane and the CPU tests, so the tests pin the *exact* selection and schedule the
lane will issue.

## Barrier count per layer — before vs after (equal)

* **Before (stock verify decode):** one device→host barrier per source layer —
  the routing-index sync `mx.eval(indices)` (then `indices.tolist()` reads the
  already-realized array). This is the sync the streamed switch performs anyway.
* **After (candidate):** the predictor's [rows, 384] next-gate score block is
  computed **on device**, reduced on device (max over rows → [384], READY
  residents → −inf, top-k → a [3] id array), and that [3]-id array is evaluated in
  the **same** `mx.eval(indices, predicted_ids)` — one barrier, now covering both.
  The [rows, 384] block is never brought to host (avoiding the naive per-layer
  [rows, 384] sync); only the [3] ids and the layer's own indices cross. The
  predicted ids are a pure read of `x` and the frozen L+1 gate weights, never an
  ancestor of L's output, so they ride the existing eval.

**Barriers/layer: 1 → 1. Equal.** (`f2_prefetch_lane._make_run.run`.)

## Memory charge

The speculative ring is resident, so admission charges it: `ring_charge_bytes(32)
= 32 × 17,694,720 = 566,231,040 B`. `f2_predictor.admits_with_ring` adds this on
top of the retained 111-slot **launch physical estimate** (109,745,344,620 B,
extension-bank README) and re-checks the **110,000,000,000 B** whole-machine
ceiling. 109,745,344,620 + 566,231,040 = 110,311,575,660 B **> 110e9**, so with
the retained launch estimate a 32-record ring does **not** fit; the window's CPU
preflight (`run_f2_prefetch_window.sh` phase 0) and `f2_prefetch_lane.install`
both **refuse before model load** in that case. A ring fits only when the base
launch leaves ≥ 566,231,040 B of headroom (e.g. a smaller `RING_RECORDS`, or the
predictable-expansion memory saving that dropped MLX peak ~1.17 GB). The 100 GiB
wired ceiling is unchanged and never raised.

## How the AGENTS.md hot-path rules are met

* **No eligible-or-stock / try-then-fallback / lane-disabled-then-stock branch in
  the enabled hot path.** The lane is a construction-time route: `install`
  replaces the source layers' `switch._run` at the quiescent post-prefill
  boundary. Stock behaviour is the *un-installed* path (byte-identical to the
  retained run). The enabled run wrapper always predicts + issues; it never tests
  "am I eligible" or falls back to stock per token.
* **Validate once, at the installation boundary.** `_validate_frozen_config`
  checks model geometry, slot layout, cache scope, `decode_miss_records_per_part`,
  the transition-window policy family and the fanout reader **once** at install and
  fails loudly otherwise (fail once, before measured generation). The hot path
  re-checks none of it.
* **No per-token/per-layer/per-dispatch proof counters.** Engagement is derived
  from existing execution statistics: the shipped `prefetch_issued_verify` /
  `prefetch_committed_verify` verify-phase slice, plus the ring's own
  per-victim-layer waste and the scheduler's aggregate `SchedulerCounters`
  (issued/useful/wasted/promoted planes), collected outside the device path and
  reported once. No counter is added merely to prove the lane ran.
* **Correct by design.** Record-slot assignment (round-robin, target-1
  protection, re-eviction embargo, per-victim waste) is delegated **unchanged** to
  the shipped `GlobalPrefetchRing`; the lane only layers plane state on top. A
  failed plane read drains the record's ring assignment and never flips a health
  flag.

## CPU verification (`pytest`, MLX hard-blocked, `nice -n 19`, no `-n auto`)

```
12 passed in 0.49s
```

* **Predictor parity vs the offline scorer** — `merge_rank_exclude` equals
  `rescore_router_capture.py`'s `real_predictions` (ranked order) **and**
  `issued_mask` (set) for **every** one of the 64 × 36 = 2,304 target-layer calls
  in the real router-feature-20260918 capture (sha256 asserted at load), at
  **k=3, feature = post-attention router, merge = max**. (Ran, not skipped — the
  45.6 MB NPZ is present.) Plus a held-out check that max beats sum at k=3.
* **Window-stop arithmetic** — the construction-time constants reproduce the
  published f1 numbers (record read 1.3717 ms, plane 0.4572 ms at 12.9 GB/s,
  window admits 2.371 records) and the window-stop predicate (inert while demand
  is outstanding; starts a plane iff it fits; stops otherwise). An issue that
  window-stops mid-record issues only the fitting planes.
* **Ring lease / promotion / eviction / failure drain** — promotion re-reads only
  the unread planes and keeps the ring lease; full read commits and consumption
  credits useful planes; a failed plane read invalidates the ring assignment
  without raising or flipping health; an evicted committed-but-unconsumed tenant
  is counted as wasted against its own layer.
* **Budget** — `ring_charge_bytes(32) = 566,231,040`; admission refuses when the
  111-slot launch estimate + ring exceeds 110e9, and admits with exactly the ring
  charge of headroom.
* **Tiny synthetic end-to-end** — a two-layer streamed decode driven through the
  scheduler produces **identical** output with prefetch on vs off (output computed
  only from the true route), while the on-run issues fewer demand loads for the
  correct prediction and wastes the mispredict.

## What is unverified until the GPU window

* Everything in `f2_prefetch_lane.py`: the MLX gate evaluation (`_biased_scores`,
  `next_layer_topk_ids`), the reader plane reads (`bind_plane_issue` via
  `reader._readv_range_into` with the plane offsets `(0, 6,266,880, 12,533,760)`),
  the real slot transactions, and that the prediction genuinely rides the switch's
  existing `mx.eval(indices)` in the live runner.
* The runtime seams the lane calls (`runtime.model_layers`, `runtime.bank`,
  `runtime.expert_routing_phase`, `runtime.remaining_compute_window_ns`,
  `runtime.demand_reads_outstanding`, `runtime.prefetch_destination`) — named to
  match the shipped runtime shape, but bound/verified only in the window.
* The one integration edit `run_full.py` needs (call `f2_prefetch_lane.install`
  at the post-prefill boundary when `MTPLX_DSV41_F2_PREFETCH=1`), applied at
  window time in the extension-bank stage-edit style (`stage_full.py`).
* Any throughput/latency/peak-memory result. Prior GPU verify-prefetch screens
  (lookahead-io optimistic-only, lookahead-adjacent all rejected, ridge-prefetch
  flat/slower) were not beaten; f1 extrapolates the winner config to ~15.6 TPS on
  this drive (still short of 20). The window will produce the actual A/B/A.

## Launch (do NOT run here — needs the exclusive GPU lock)

```
bash scripts/deepseek_v41/run_f2_prefetch_window.sh
```

Runs a CPU admission preflight, then ONE guarded window arming control /
candidate / control on the exact 16,384-in / 1,024-out workload (same prompt-ids,
budgets, D5 + two lookup, native KV16), fresh receipt dir per arm under
`/tmp/dsv41-f2-prefetch-20260919`, full output ids stored, sha256 gated against
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`. If three full
arms exceed the window's time/thermal envelope, re-run
`F2_ARMS='candidate control' bash scripts/deepseek_v41/run_f2_prefetch_window.sh`.
