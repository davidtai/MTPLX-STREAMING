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
- [x] Recompute the corrected plan. The unchanged reserve admits cap 83. Cap 91
  is the highest allocator-safe staged candidate; cap 92 exceeds the allocator
  limit by 540,558,640 bytes and is excluded even though it fits the 110 GB
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

### Task 4: Measure and promote only winners

**Files:**
- Create only after a win: `docs/deepseek-v41/receipts/<dated-stage>/`
- Add focused tests only for winning lanes.

**Security flag:** none

**Does NOT cover:** Any GPU execution while another job owns the lane, cap 93,
or component-rANS implementation without a separate decoder gate.

- [ ] Wait until the operator says the GPU lane is available; do not probe or
  queue the lock while other jobs run.
- [ ] Use `bench/laguna/run_guarded.py`, capture current baseline and exact Qwen
  identity, and refuse any arm whose admitted peak exceeds 110,000,000,000 bytes.
- [ ] Run one short matched batch with separately selectable corrected cap-83
  control, three-record miss parts, causal-policy, and shared-overlap arms.
  Remove any losing candidate; screen two-record parts only after a chunking win.
- [ ] Screen the winning stack at cap 89. Attempt cap 91 only after cap 89
  confirms allocator and whole-machine headroom; never arm cap 92 under the
  current allocator limit.
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
