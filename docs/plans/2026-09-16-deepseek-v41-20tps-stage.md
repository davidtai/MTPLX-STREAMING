# DeepSeek V4.1 20 TPS Stage Implementation Plan

> **For agentic workers:** Execute inline because the three changes share one
> memory plan and one verification route. Do not dispatch subagents.

**Goal:** Install separately measurable memory-capacity, cache-admission, and
verification-overlap candidates for the exact DeepSeek V4.1 16K/1K workload.

**Architecture:** Correct the construction-time cache lifetime first, add one
bounded per-layer causal admission policy, then move existing shared GPU work
ahead of the M6 miss wait. Keep every performance lane construction-selected
and fail before generation if its invariants are unavailable.

**Tech Stack:** Python, NumPy, MLX/Metal, pytest, macOS guarded runner.

**Assumptions:** The cap-80 receipt represents an isolated baseline; live
admission remains authoritative. The trace has six router rows per verification
cycle; this plan will not apply the policy to prefill or non-DeepSeek profiles.

---

## Files

- `mtplx/models/deepseek_v41_loader.py`: target projection-cache lifetime.
- `tests/test_deepseek_v41_projection_cache_lifetime.py`: existing accounting
  assertion, updated after the formula changes.
- `mtplx/expert_runtime.py`: construction-time policy validation and binding.
- `mtplx/expert_streaming.py`: bounded causal policy state and admission.
- `mtplx/models/expert_mlx.py`: M6 shared-work submission ordering.
- `mtplx/models/deepseek_v41_dspark_decode.py`: staged target verification.
- `mtplx/serve_stream_counters.py`: model-owned Engram counter collection.
- `scripts/deepseek_v41/ab_decode_env_levers.py`: pass the benchmark model to
  the shared counter collector and expose the cache policy arm.
- `scripts/deepseek_v41/bench_standard_shape.py`: keep profile cache-policy
  resolution and scheduling receipt stamping aligned with the A/B runner.
- `docs/deepseek-v41/receipts/`: winning benchmark and validation evidence only.

### Task 0: Correct benchmark memory and Engram reporting

**Files:**
- Modify: `mtplx/serve_stream_counters.py`
- Modify: `scripts/deepseek_v41/ab_decode_env_levers.py`
- Modify: `tests/test_deepseek_v41_single_slot_pool.py`

**Security flag:** none

- [x] Keep allocator peak, process `phys_footprint`, and whole-machine physical
  use as separate measures; never add process footprint to machine use.
- [x] Pass the benchmark model separately from its bare expert runtime so
  model-owned Engram cache totals appear in `serve_stream_counters`.
- [x] Run the existing counter test and a direct no-MLX runner plumbing check.

### Task 1: Correct projection-cache lifetime accounting

**Files:**
- Modify: `mtplx/models/deepseek_v41_loader.py`
- Modify after correction: `tests/test_deepseek_v41_projection_cache_lifetime.py`

**Security flag:** none

**Does NOT cover:** Direct packed decode, dense-only decode, non-layer-major
prefill, or MTP-stage fp32 caches; their formulas remain unchanged.

- [x] Replace the non-direct, dense-plus-fused, layer-major target reserve with
  `max(N*fused, dense + (N-1)*fused)`.
- [x] Preserve `N*max(dense, fused)` for every other non-direct combination.
- [x] Update the existing parameterized arithmetic expectation for this exact
  combination; do not add a new test module.
- [x] Run
  `PYTHONPATH=. ../deepseek-v4-adaptive-d1-8/.venv/bin/python -m pytest -q tests/test_deepseek_v41_projection_cache_lifetime.py`.
  The checkout-local virtualenv lacked pytest, so the existing sibling project
  virtualenv was used. The broader CPU-only batch passed 57 checks without a
  model load; the memory-reporting subset was rerun after runner wiring and
  passed 17 checks.
- [x] Recompute the corrected BF16-head plan. The unchanged reserve admits cap
  83. Cap 91 is the highest allocator-safe baseline candidate; cap 92 exceeds
  that allocator limit by 540,558,640 bytes even though it fits the 110 GB
  whole-machine projection.
- [x] Add a receipt-stamped `--runtime-reserve-gib` runner choice: 7 GiB remains
  the cap-83 control, 3 GiB selects cap 89, and 2 GiB selects cap 91. Refuse
  values below 2 GiB; the outer measured transient band remains unchanged.

### Task 2: Add the causal transition-window admission candidate

**Files:**
- Modify: `mtplx/expert_runtime.py`
- Modify: `mtplx/expert_streaming.py`

**Security flag:** none

**Does NOT cover:** Prefill admission, global-scope banks, non-single-pool
profiles, speculative prefetch, or an automatic fallback to LRU/2Q.

- [x] Extend construction validation with one explicit policy name and bind it
  only for layer-scoped single-pool DeepSeek banks.
- [x] Expose the construction choice as runner `--cache-policy`, inherit the
  served profile when it is omitted, and stamp the installed policy/pool in the
  resolved-plan receipt so an A/B cannot silently run control twice.
- [x] Allocate fixed per-layer state for 384 experts: float32 transition counts
  and denominators, a 16-route deque/frequency vector, and the prior route.
- [x] At each decode plan, update the observed prior-to-current transition and
  compute the fixed `0.7/0.2/0.1` score before persistent admission.
- [x] Route low-score misses through existing transient slots; never change the
  true expert list, gather order, output positions, or slot bound.
- [x] Add reset and request-boundary clearing for the new bounded state.
- [x] Run only syntax/import validation before GPU evidence:
  `.venv/bin/python -m py_compile mtplx/expert_streaming.py mtplx/expert_runtime.py`.
  Result: exit 0 and no `mlx` import. The actual bank replay saved 9,718 of
  49,298 physical reads at cap 92; GPU wall-time evidence remains open.

### Task 3: Submit M6 shared work before miss waits

**Files:**
- Modify: `mtplx/models/expert_mlx.py`

**Security flag:** none

**Does NOT cover:** Prefill, routes without pending misses, all-hit routes, or
the generic bounded-wave path that already submits shared work early.

- [x] In the verification single-barrier path, after hit-gather submission and
  before `iter_ready_misses()`, claim the existing shared pipeline item.
- [x] Call `shared_work()` once, submit the result with callable
  `mx.async_eval`, close the claim in `finally`, and clear the work item.
- [x] Preserve the existing post-route `shared_work()` call as the all-hit and
  non-decode path; `shared is None` remains the once-only guard.
- [x] Default the lane off, expose it as `--verify-shared-overlap`, bind its
  submit callable once during switch construction, fail construction when
  `mx.async_eval` is unavailable, and stamp the installed value in both runner
  receipts so the unchanged control remains measurable.
- [x] Run syntax validation only before GPU evidence:
  `.venv/bin/python -m py_compile mtplx/models/expert_mlx.py`.
  Result: exit 0.

### Task 3b: Expose bounded decode miss completion

**Files:**
- Modify: `mtplx/expert_runtime.py`
- Modify: `scripts/deepseek_v41/ab_decode_env_levers.py`
- Modify: `scripts/deepseek_v41/bench_standard_shape.py`

**Security flag:** none

**Does NOT cover:** Prefill, compressed streamed records, non-DeepSeek models,
an automatic part-size choice, or a fallback from the enabled candidate.

- [x] Add immutable `decode_miss_records_per_part`; require the DeepSeek-V4.1
  component-bank overlap path and raw sidecar placement at construction.
- [x] Submit every bounded part before resident work, sort by physical sidecar
  placement, and prefer part boundaries between noncontiguous records.
- [x] Preserve assignment order, disjoint slot ownership, transaction rollback,
  completion-order consumption, and final policy publication.
- [x] Expose `--decode-miss-records-per-part` in both runners and stamp the
  installed value in receipts and the runtime snapshot.
- [x] Run syntax, no-MLX construction/plumbing, and whitespace validation only;
  defer regression tests until a matched GPU arm wins.
- [ ] Screen three records per part against the unchanged layer-wide batch.
  Screen two records only if the three-record candidate improves wall time.

### Task 3c: Stop target verification after an early rejected chunk

**Files:**
- Modify: `mtplx/models/deepseek_v41_dspark_decode.py`
- Modify: `scripts/deepseek_v41/ab_decode_env_levers.py`
- Modify: `scripts/deepseek_v41/bench_standard_shape.py`

**Security flag:** none

**Does NOT cover:** Automatic schedule selection, a hot fallback to full verify,
or promotion before matched GPU wall-time evidence.

- [x] Add a construction-time verify-row partition whose default is the unchanged
  single `K+1` forward and whose values must sum exactly to `K+1`.
- [x] Evaluate acceptance after each chunk and submit the next target forward only
  when every draft covered by the current chunk was accepted.
- [x] Preserve target-cache rollback, committed DSpark hidden seeding, sampled RNG
  order, stop handling, and first-divergence logits capture across chunks.
- [x] Count `cycles` as logical speculative cycles and `verify_calls` as actual
  target forwards; stamp the installed chunk schedule in both runner receipts.
- [x] Expose `--dspark-verify-chunks` in both runners and reject an invalid
  partition before model work.
- [x] Run syntax and whitespace validation only; do not add optimization tests
  before a matched arm wins.
- [ ] Screen `3,3` against the unchanged six-row verify at depth five. Screen
  `2,2,2` only if `3,3` improves decode wall time.

### Task 3d: Stage conditional cap 92 and cap 93 capacity

**Files:**
- Conditional benchmark wrappers and commands under `/tmp/dsv41-110-stage/`.
- Promote only winning reusable behavior into tracked runner code.

**Security flag:** none

**Does NOT cover:** General-serving compact MTP residency, assumed q8 parity,
static admission of cap 93, or a compressed expert artifact.

- [x] Price the authenticated prepacked affine-q8 head before target-cache
  allocation. It saves 620,544,000 resident bytes and improves the six-row head
  microbenchmark, but its 63/64 sampled argmax result still requires measured
  output/tie classification.
- [x] Construct the exact trace-covered MTP resident inventory and verify its
  source manifest, selected expert sets `(93, 58, 32)`, tensor geometry, and
  3,778,928,640-byte saving before allocation.
- [x] Stage cap 92 with 45 transient slots and a 19,489,831,752-byte fixed
  footprint. Require a measured cap-91 q8 win before this command is eligible.
- [x] Bound Engram at 119,537,664 bytes per bank. The 128-token screen requires
  at most 414,720 of 452,794 rows; the recorded 1,023-step control requires at
  most 447,432 rows. Exact payload, metadata, and 1 GiB other-host reserve total
  1,544,647,680 bytes.
- [x] Stage cap 93 with 44 transient slots and a 19,471,031,112-byte fixed
  footprint. Require matching cap-92 receipt, bounds, and OS samples, then
  derive MLX, process-footprint, and whole-machine admission from those measured
  peaks plus the exact one-slot/one-transient delta.
- [x] Confirm the reference projection: 98,702,754,032 active bytes against a
  98,708,752,320-byte allocator limit, leaving 5,998,288 bytes. This is a
  refusal-sensitive reference, not sufficient admission evidence.
- [x] Replay transition-window plus `3,3` at cap 93 across all eight
  within-chunk acceptance orders. Median 36,991.5 reads, 369 forwards, and 1,107
  rows imply a 19.34-tok/s raw-I/O ceiling, so 20 TPS remains open.
- [x] Measure one exact whole-record rANS size on CPU without importing MLX:
  17,561,059 of 18,800,640 bytes. Its 20.70-tok/s I/O-only ceiling warrants a
  later direct-decoder gate; the current serial decode-and-copy path is not an
  arm.
- [x] Sample scale bytes from four experts in every routed layer with uncached
  reads. A lossless three-bit-plus-escape layout projects 3.6765% record savings
  but only 0.20 seconds of I/O margin at 20 TPS, so do not build the sidecar yet.

### Task 3e: Stage transition-bound cap 93 and MTP-direct cap 94

**Files:**
- Conditional benchmark wrappers and commands under `/tmp/dsv41-110-stage/`.
- Promote only a measured winner into tracked loader or runner code.

**Security flag:** none

**Does NOT cover:** A direct target projection, hot-path eligibility checks,
silent fallback, or promotion from replay alone.

- [x] Prove the `3,3` target route needs at most eighteen transient expert
  slots per forward. Confirm the exact trace reaches eighteen and retain
  `route_waves` plus `batch_admission_slots` for broader prefill routes.
- [x] Stage cap 93 with eighteen transient slots, an 18,982,214,472-byte fixed
  footprint, and a 503,422,976-byte plan remainder inside the fixed engine
  budget. Derive allocator, process, and physical bounds from a matching
  successful cap-93 frequency/full-verify predecessor.
- [x] Keep the packed MXFP8 direct route rejected for target attention because
  its exact screen has non-tie divergences. Restrict the conditional route to
  the three MTP proposal stages, whose candidates remain target-verified.
- [x] Remove only the three dense draft `wo_a` caches from resident pricing,
  saving 402,653,184 bytes. Stage cap 94 with an 18,579,561,288-byte fixed
  footprint and a 154,050,560-byte `plan_remainder_bytes` field. Derive live
  allocator and whole-machine headroom from the measured predecessor.
- [x] Validate all three draft projection contracts once after strict model
  load, install fixed compiled direct callables, leave target attention
  untouched, and fail construction instead of retaining an enabled fallback.
- [x] Replay uniform cap 94 through cap 97. Cap-94 `3,3` reaches a 19.57-tok/s
  raw-I/O ceiling; cap 96 first crosses 20 TPS but would prune 90 more MTP
  residents and affect 157 of 206 cycles, so reject that funding path.
- [x] Solve a trace-shaped nonuniform cap-93 allocation. Its eight-order mean
  is 35,750 reads but its median and maximum miss the 35,769.77-read threshold;
  reject it before GPU work because it has no wall-time margin.
- [x] Run syntax, static arithmetic, and no-MLX checks only. Add focused tests
  only after a matched GPU arm wins.

### Task 4: Measure and promote only winners

**Files:**
- Create only after a win: `docs/deepseek-v41/receipts/<dated-stage>/`
- Add focused tests only for winning lanes.

**Security flag:** none

**Does NOT cover:** Any GPU execution while another job owns the lane or codec
implementation without a separate direct-decoder gate.

- [ ] Wait until the operator says the GPU lane is available; do not probe or
  queue the lock while other jobs run.
- [ ] Use `scripts/deepseek_v41/gpu_window.sh` directly. It acquires the shared
  lock before bootout, captures the exact Qwen identity, reclaims the stopped
  service and candidate file caches, enforces the live 110,000,000,000-byte
  whole-machine ceiling, restores Qwen, and releases the lock last. Do not nest
  it under `bench/laguna/run_guarded.py`; both guards own the same lock.
- [ ] Run one short matched batch with separately selectable corrected cap-83
  control, three-record miss parts, causal-policy, shared-overlap, and `3,3`
  staged-verify arms. Remove losing candidates; screen finer partitions only
  after the corresponding coarse candidate wins.
- [ ] Screen the winning stack at cap 89. Attempt cap 91 only after cap 89
  confirms allocator and whole-machine headroom.
- [ ] If the q8 head wins with an allowed output classification, run conditional
  prepacked cap 92. Run bounded-Engram cap 93 only after the matching cap-92
  receipt establishes its measured predecessor bounds. Run the cap-93 `3,3`
  transition arm with eighteen transient slots only after the matching cap-93
  frequency/full-verify arm succeeds. Run conditional cap-94 MTP-direct only
  after the matching cap-93 transition receipt succeeds. Combine other winners
  only after each unchanged predecessor succeeds.
- [ ] Run the exact 16,384-input/1,024-output Python workload for the winning
  stack and require at least 20 decode tok/s.
- [ ] Add focused regression tests only for measured winners, then run those
  tests once.
- [ ] Record guard exit, service health/model identity, lock release, token
  digest or allowed tie classification, physical reads, wall time, and all three
  memory measures.

## Self-review

- Every design requirement maps to a task.
- Experimental tests are deferred until measurement; the existing planner test
  changes because memory admission correctness is already established.
- Policy names, fixed weights, cache scope, and excluded paths are explicit.
- Component compression remains a separate stage because it needs an artifact
  and decoder design plus independent performance evidence.
