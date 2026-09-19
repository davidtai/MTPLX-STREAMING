# F2 next-layer expert prefetch — build receipt (2026-09-19)

Next-layer expert prefetch for the DeepSeek-V4.1 Q4 D5/M≤8 verify decode, composed
onto the exact configuration of the retained 13.87 TPS packed run. This receipt is the
design + seam table + findings; the code lives in `scripts/deepseek_v41/f2/` (not here),
and the CPU proofs are `tests/test_dsv41_f2_prefetch_lane.py` (new) and
`tests/test_dsv41_f2_prefetch.py` (the retained offline-scorer / window-arithmetic core).

Branch `f2/next-layer-prefetch`. No GPU was touched building this; another session holds
`/tmp/mtplx-gpu-exclusive.lock`.

## Design

The candidate is the retained packed decode lane
(`.../extension-bank-20260919/full/sources/packed/plane_lane.py` `PackedDecode`) with a
next-layer speculative prefetch composed on. It reuses the GPU-proven ridge-prefetch v2
lane (`.../ridge-prefetch-20260919/v2/plane_lane.py`) with its three flat-screen defects
fixed:

- **fix #1 — read with ≥ the control's parallelism.** v2 replaced the native 15-worker
  fanout pool with a fixed **four**-worker `PriorityReads`, so its candidate arms read
  with *less* parallelism than their controls. `f2/priority_reads.py` takes `workers` as
  a constructor argument; `install` defaults it to the native pool's worker count read at
  install (`reader._fanout_pool_workers`; fanout 4 → `max(4, (1+4)·3)` = 15,
  `mtplx/expert_io.py:419-429`) and refuses an override below it. The control arm never
  installs this reader — it stays on the untouched native `ThreadPoolExecutor`.
- **fix #2 — keep the barrier count at 1 routing + 1 miss-drain.** v2 did the numpy
  ranking inside `Issue.prepare`, on the critical path *before* the demand reads were
  submitted. `f2/issue.py` splits it: `prepare` builds ONLY the device prediction and
  evaluates it on the source route's existing indices barrier
  (`mx.eval(indices, merged)`); all host ranking / READY filtering / issue happen in
  `Issue.__call__`, which the runner calls *after* `begin_split_route` has submitted the
  demand reads.
- **fix #3 — parameter-free predictor.** v2 replayed saved scores through a learned ridge
  adapter and issued only above an 85 %-precision margin (5–16 % coverage). `f2/issue.py`
  is parameter-free: the device computes `merged = max over rows of the next layer's
  native biased gate score` → `[384]` f32 (the model's own `_gate_prefix` /
  `_gate_prefix_impl` on the source layer's post-attention router input — exactly the
  transform `Gate.__call__` ranks, `deepseek_v41_moe.py:296/111`), the host takes the top
  `k=3` experts that are not READY owners of the target layer (persistent, transient or
  committed ring), and issues them via `runtime.prefetch_experts(target, ids,
  verify=True)`. Target layers 4..39 from sources 3..38; layers 0..3 unpredicted, the
  final layer has no successor (`run_full_install.select_prefetch_sources`).

The speculative reads are warmed entirely through the **shipped** runtime path —
`prefetch_experts` → the shared `GlobalPrefetchRing` → `slots.load_speculative` — and the
true route always gathers from the TRUE `indices`, so a mispredict only wastes a
speculative read and can never change a logit. There is **no** eligible-or-stock /
try-then-fallback branch in the enabled hot path: the lane is chosen once, at install, by
which `switch._run` is bound (control = retained `plane_lane.install`; candidate =
`f2.plane_lane_prefetch.install`).

### Package
```
scripts/deepseek_v41/f2/
  priority_reads.py       N-worker demand-priority reader (fix #1); MLX-free
  issue.py                live predictor (fix #2/#3): device biased-gate max, host top-3
  plane_lane_prefetch.py  PrefetchDecode + install (mirrors plane_lane.install; requires prefetch_slots==R)
  full_config.py          FullPrefetchConfig: transition-window + ring R (16/32); ring_reserve_bytes
  run_full_install.py     stage-edit entrypoint install_f2_growth + select_prefetch_sources
  window_preflight.py     CPU preflight: resolves every seam on the real classes + deps
  gpu_smoke.py            WRITE-ONLY bounded 3-layer Metal parity smoke (guarded)
```

## Seam table (every runtime/reader/slot/ring/gate name the lane touches)

Zero unresolved names: `window_preflight._seam_checks()` resolves every method/field
below on the real classes on CPU (`test_window_preflight_resolves_every_seam`). "CPU" =
executed on the real runtime in a CPU test; "smoke" = executed only in the guarded GPU
smoke (the `PrefetchDecode.run` body needs the plane-split reader + Metal gather — see
"What stays unverified"); the name itself is still CPU-resolved by the preflight.

| Seam | Defining file:line | Lane use | Covered by |
|---|---|---|---|
| `ExpertStreamingRuntime.begin_split_route` | expert_runtime.py:4127 | open the demand route; runs the reconcile at :4168 | CPU (route) + smoke |
| `ExpertStreamingRuntime.observe_route` | expert_runtime.py:4378 | `run()` route accounting | smoke; preflight |
| `ExpertStreamingRuntime.flush_deferred_slot_releases` | expert_runtime.py:3601 | `run()` drain deferred releases | CPU (called) + smoke |
| `ExpertStreamingRuntime.defer_slot_release` | expert_runtime.py:3586 | `run()` deferred split close | smoke; preflight |
| `ExpertStreamingRuntime.prefetch_experts` | expert_runtime.py:5076 | `Issue.__call__` issues top-k | CPU `test_issue_ranking…`, `…_speculative_reads`, `…_wasted`, `…_failure_drain` |
| `ExpertStreamingRuntime._reconcile_prefetch_for_route` | expert_runtime.py:4995 | await in-flight → commit hit (await :5044, commit :5052) | CPU `…_demanded_inflight_tenant_is_awaited_not_reread` |
| `ExpertStreamingRuntime._run_speculative_load` | expert_runtime.py:5274 | ticket recycle-skip (:5297-5303); the read | CPU (via prefetch+settle); priority_reads cites it |
| `ExpertStreamingRuntime._apply_prefetch_completions` | expert_runtime.py:5346 | publish/commit or invalidate settled reads | CPU `…_wasted`, `…_failure_drain` |
| `runtime.slots / reader / spec / config / plan` | expert_runtime.py:2512-2513 etc. | install gate + wiring | CPU `test_install_accepts_and_wires` |
| `runtime._split_executor / _prefetch_executor / _pipeline_ledger / _single_slot_pool` | expert_runtime.py:2743/2826/2516/2573 | install wraps / gate | CPU `test_install_accepts_and_wires` |
| `PendingSplitRoute.iter_ready_misses` | expert_runtime.py:1235 | `run()` completion loop | smoke; preflight |
| `PendingSplitRoute.hit_ready / abort / close` | expert_runtime.py:926 / 902-body | `run()` hit path + failure drain | smoke; preflight (`__init__` sig) |
| `ExpertSlotPool.load_speculative` | expert_slots.py:1868 | ring read into a slot | CPU `…_demanded_inflight`, `…_failure_drain`, `…_speculative_reads` |
| `ExpertSlotPool.ensure_route_part` | expert_slots.py:1810 | `run()` demand miss part | smoke; preflight |
| `ExpertSlotPool._persistent / _transient / _prefetch` | expert_slots.py:776/847/852 | `Issue` READY-owner snapshot | CPU `…_issue_ranking`, `…_speculative_reads` |
| `ExpertSlotPool._executor` | expert_slots.py:888 | install wraps (ReaderExecutor) | CPU `test_install_accepts_and_wires` |
| `PositionalExpertReader._readv_range_into` | expert_io.py:1049 | plane read (bind_priority_reader) | smoke; preflight |
| `PositionalExpertReader._fanout_executor` | expert_io.py:430 | replaced by `PriorityReads` | CPU `test_install_accepts_and_wires` |
| `PositionalExpertReader._fanout_pool_workers` | expert_io.py:419 | fix #1 default worker count | CPU `test_install_accepts_and_wires` (==15) |
| `reader.read_record_into / read_component_records_into` | expert_io.py (rebound) | plane-split reads; `_fill` calls (expert_slots.py:1424/1485) | smoke; preflight |
| `LayerExpertSlotBank.plan` | expert_streaming.py:1528 | resolves committed ring hits in place (:1553-1589) | CPU `…_ring_hit_excluded_from_pool…` |
| `LayerExpertSlotBank.plan_prefetch` | expert_streaming.py:854 | ring slot assignment | CPU `…_speculative_reads`, `…_wasted` |
| `LayerExpertSlotBank.prefetch_ticket / commit_prefetch / invalidate_prefetch` | expert_streaming.py:870/877/891 | ticketed publish/forget | CPU `…_demanded_inflight`, `…_failure_drain` |
| `LayerExpertSlotBank.published_experts / _prefetch_expert_to_slot / resident_experts` | expert_streaming.py:757/918/(bank) | hit/tenancy introspection | CPU `…_ring_hit_…`, `…_speculative_reads`, `…_failure_drain` |
| `GlobalPrefetchRing.plan_prefetch` | expert_streaming.py:341 | round-robin, target-1, embargo | CPU `…_wasted` |
| `GlobalPrefetchRing.published / first_consumption / mark_used / note_decode` | expert_streaming.py:476/494/486/334 | in-place hit resolution | CPU `…_ring_hit_…` |
| `GlobalPrefetchRing.commit_prefetch / invalidate_prefetch / consume_wasted_by_layer` | expert_streaming.py:436/452/523 | publish / forget / waste drain | CPU `…_ring_hit`, `…_failure_drain`, `…_wasted` |
| `deepseek_v41_moe._gate_prefix / _gate_prefix_impl / _attn_compile_gate` | deepseek_v41_moe.py:114/101/85 | predictor scoring (biased) | CPU `test_gate_predictor_merged…` |
| `Gate.weight / e_score_correction_bias / gate_temp / score_func` | deepseek_v41_moe.py:266/269/261/260 | predictor gate | CPU `test_gate_predictor_merged…` |
| `biased = scores + e_score_correction_bias` | deepseek_v41_moe.py:296 (`_gate_prefix_impl` :111) | the "native biased gate score" | CPU `test_gate_predictor_merged…` |
| `ExpertSlotState.READY` | expert_slots.py:71 | READY-owner exclusion | CPU `…_issue_ranking…` |
| `expert_mlx._clamped_swiglu / _DeferredSplitClose` | expert_mlx.py:1603 / 318 | PackedOps swiglu / deferred close | smoke; preflight |
| `HotExpertSwitchGLU._run` | expert_mlx.py:2533 | `switch._run` rebind target | CPU `test_install_accepts_and_wires` |
| `CacheCounters.prefetch_{issued,committed,awaited_inflight,wasted,bytes,hit_on_true_route,first_consumption_hits}` | expert_streaming.py:156/157/182/181/183/180/191 | window receipt counters (read once) | CPU `…_speculative_reads`, `…_demanded_inflight`, `…_wasted`, `…_ring_hit`; window post-run |
| `ExpertStreamingConfig.{prefetch_slots,transient_slots,slot_layout,cache_scope,cache_policy,decode_miss_records_per_part,split_route_release,overlap_miss_reads,resource_telemetry,io_read_fanout}` | expert_runtime.py:250/176/197/200/199/244/233/238/202/(field) | install gate + FullPrefetchConfig | CPU `test_full_config_*`, `test_install_refuses_*` |
| `plan.prefetch_ring_slots / transient_slots` | expert_runtime.py:756 / 176 | install gate (ring==R, ≥48) | CPU `test_install_*` |

## Ring tenancy finding (does prefetch turn cache hits back into reads?)

**A hit ring tenant stays a ring tenant — it is read in place, NOT promoted into the
persistent pool.** When a true route at layer L+1 needs a committed ring entry,
`LayerExpertSlotBank.plan` (`expert_streaming.py:1553-1589`) resolves it via
`GlobalPrefetchRing.published(...)` (`:476`) and maps the expert to its **ring slot** —
the comment at `:1556-1558` is explicit: *"resolve as hits reading the ring slot in place
(no re-read, no copy)."* The expert is added to `hit_set` (`:1589`) for the gather but is
**not** in `self._expert_to_slot`, so the pool promotion block (`:1603-1615`, which only
touches `expert in self._expert_to_slot`) never promotes it, and it is excluded from
`miss_order` / transition-window admission (`:1569-1573`, `:1597-1602`). `mark_used`
(`:1588`) records the consumption so a later recycle is not miscounted as wasted.

Consequences, decided by test `test_ring_hit_excluded_from_pool_and_stays_a_ring_tenant`
(asserts the hit expert remains in `bank._prefetch_expert_to_slot` and stays out of
`bank.resident_experts`):

- **Within a token:** a ring hit is never re-read (resolved in place, excluded from pool
  loads). No cache hit becomes a read.
- **Across tokens:** the shared ring (R slots across all 40 layers) can recycle a
  *consumed* tenant on a later `plan_prefetch` round-robin (`:414-422`), subject only to
  target-1 protection (layer L-1) and the 2-epoch re-eviction embargo. If it is recycled
  before the next token routes that layer, the expert is re-predicted and re-issued as a
  **speculative** read during the previous layer's forward (off the demand path), then
  re-commits and re-hits. It degrades to a demand read only when the ring is too pressured
  to hold/re-commit it in time — and even then the demand route *awaits* the in-flight
  speculative read (`_reconcile_prefetch_for_route`) rather than issuing a duplicate.
- **So prefetch does not silently turn cache hits back into demand reads.** The ring is a
  one-step lookahead reserve, not the cache; the persistent pool remains the stable
  residency for the hottest experts. A reliably ring-hit expert is *shadowed* from pool
  admission (it never appears in `miss_order`), so it lives only in the ring and is
  re-prefetched each token.

**Smallest change if that cross-token re-prefetch churn proves costly** (not implemented
— it changes the retained pool's admission arithmetic and needs its own measurement): on a
first-consumption ring hit (`GlobalPrefetchRing.first_consumption`, `:494`, already
computed at `plan` :1585), admit the expert into the transition-window pool via a slot
exchange, so a persistently-hot lookahead expert graduates to stable pool residency and
stops churning the ring. The window's `prefetch_first_consumption_hits / prefetch_issued`
ratio (both existing counters) measures whether this churn is worth the change.

## Memory arithmetic (derived through the retained admission code)

The ring is a resident reserve of R weight records
(`packed_admission.py WEIGHTS = 17,694,720` = `f2_predictor.EXPERT_RECORD_BYTES`;
verified equal in `test_ring_reserve_bytes`):

- R=32 ring = 32 × 17,694,720 = **566,231,040 B**; R=16 = 283,115,520 B.
- Retained 111-row launch estimate = **109,745,344,620 B**.
- 111 rows + R=32 = **110,311,575,660 B > 110e9** (overflow 311,575,660) → does NOT fit.
- 110 rows frees 40 × 17,694,720 = 707,788,800 B, so 110 rows + R=32 =
  **109,603,786,860 B ≤ 110e9** (headroom 396,213,140) → fits. R=16 also needs 110 rows
  (111 + R16 = 110,028,460,140 > 110e9).

So the candidate is **110 persistent rows + R=32** (R=16 supported). This is derived
through the SAME admission code the retained run uses, not a parallel formula: the ring
reserve `full_config.ring_reserve_bytes(R)` is charged into `packed_admission.resolve_admission`'s
capacity search at `sources/packed/packed_admission.py:126-131` (add it to `active` /
`physical` / the `wired + … + 1 GiB ≤ 100 GiB` checks), and the loop `range(112,
old_capacity, -1)` then breaks at 110 instead of 111. `f2.full_config.ring_reserve_bytes`
is the reserve only; the row count comes from that admission loop, and the window
preflight re-checks the constants (`test_ring_reserve_bytes`,
`f2_predictor.admits_with_ring` cross-check in `tests/test_dsv41_f2_prefetch.py`).

## Barrier count before / after

Per source layer-call, unchanged at **1 routing barrier + 1 miss-drain**:

- Control `PackedDecode.run`: `mx.eval(indices)` (routing) + the completion loop
  (`part.gate_up_ready.result()` / `next(iter_ready_misses)` — miss-drain).
- Candidate `PrefetchDecode.run`: `Issue.prepare` replaces `mx.eval(indices)` with
  `mx.eval(indices, merged)` — the SAME single barrier, now also forcing the device
  prediction (fix #2). `Issue.__call__` reads the settled prediction with `.tolist()` (no
  new barrier) and calls `prefetch_experts`, which submits async reads and adds no
  `mx.eval` / `mx.synchronize`. `_reconcile_prefetch_for_route`'s bounded await is an I/O
  join on a prefetch worker's future, not a Metal barrier. CPU proof:
  `test_issue_prepare_adds_no_host_sync` (exactly one main-thread `mx.eval` in `prepare`);
  the full-lane per-call parity is a GPU-smoke assertion.

## AGENTS.md compliance

- Construction-time install, fixed-shape entrypoints: `install` validates the geometry /
  codec contract against the ops object (`PackedOps.contract` = 5120/2304/6/mxfp4/32/4/10.0,
  pinned where the Metal kernels are built) and every config/plan/runtime invariant once,
  then binds one runner per layer. Invalid states cannot reach execution.
- No eligible-or-stock / try-then-fallback in the hot path: control vs candidate is a
  construction-time route (which `switch._run` is bound); a mispredict only wastes a
  speculative read.
- No per-token/per-layer proof counters: engagement is read ONCE after decode from the
  runtime's existing `prefetch_*` statistics (`CacheCounters`); the CPU tests read the
  same counters.
- Fail once, clearly, before measured generation: `window_preflight` resolves every seam +
  dependency on CPU before the service is unloaded; `install` refuses loudly on any
  violated invariant; `FullPrefetchConfig` refuses any off-geometry field.
- Preserved the retained path's arithmetic / ownership / tiling / layout: the gather math,
  plane offsets and swiglu clamp are the retained lane's, unchanged; the prefetch reuses
  the shipped ring/reader/slot state machine rather than a parallel one.

## What stays unverified until the guarded window

The CPU tests exercise the whole runtime/slot/ring/reconcile machinery, the live
predictor, the priority reader, the config and the install wiring on the real classes. The
following execute only on Metal and are asserted by `f2/gpu_smoke.py` (write-only; run
under the guard before any full window):

- `PrefetchDecode.run`'s execution body with the plane-split `bind_priority_reader`
  (reads three weight planes of the production `experts.bin` at offsets 0 / 6,266,880 /
  12,533,760 — out of bounds for a 6,912-byte tiny record) and the Metal `PackedOps`
  gather (`paired_kernels.make_projection` is `mx.fast.metal_kernel`, geometry-locked to
  2304/5120). No tiny CPU artifact can drive these; materialising production-geometry
  records is impossible without Metal and CPU/IO-heavy enough to perturb the concurrent
  measurement window.
- The end-to-end output identity through the packed body, the aggregate prefetch counter
  deltas on the real experts.bin, and throughput/latency (prefill tok/s, decode tok/s,
  peak GB, wall, TTFT) — all window measurements.

The GPU smoke asserts, on 3 routed layers with attention at the M≤8 verify shape: (1)
byte-identical routed output lane on vs off, (2) the prefetch counters move, (3) the
generation-thread `mx.eval` count per source layer-call is unchanged.

## Exact launch command (guarded window — orchestrator runs, not the author)

```
bash /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f2-prefetch/scripts/deepseek_v41/run_f2_prefetch_window.sh
```

It runs the CPU preflight, then arms `control_a` (111 rows, native) / `candidate` (110
rows + R=32, F2 lane) / `control_b` (111 rows, native) — and, with
`F2_INCLUDE_CONTROL110=1`, `control_110` (110 rows, native, to isolate the row cut) —
each in its own guarded window with a fresh receipt dir, the full 1,024 token ids stored,
the token-id sha256 gated against `0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`,
and the candidate receipt's once-read prefetch counters. `F2_RING_RECORDS=16` selects the
R=16 ring. The candidate arm requires the F2 stage edits applied to the staged runner tree
(`STAGE_CANDIDATE`): the single `install_plane_lane(...)` call in `packed_phase.py` →
`run_full_install.install_f2_growth(...)`, the ring charge in `packed_admission.py:126-131`,
and `FullPrefetchConfig` + the counters row in `run_full.py` `record_pass`.
