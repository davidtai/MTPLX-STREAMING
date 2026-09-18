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

## Current update, 2026-09-18

Task 4 remains open. The native D5 plus two-token causal lookup run reaches
**13.4141517619 TPS / 76.2627423750 s**, preserving all 1,024 native IDs.
It uses 198 target calls rather than 206, at 84->110 slots and native KV16.
Machine peak 109,238,927,360 B fits the 109,631,928,540-byte bound, including
1,421,996,032 B host reserve. This saves 0.5417692920 s versus input-row caching,
a best single result rather than an isolated or repeated speedup. Expert reads
increase by 25; this is not a material I/O reduction. 20 TPS remains unmet.

Two host regressions pass after the full win. Longer extensions receive only
an independent-boundary CPU screen; added target work rejects a GPU follow-up.
Both guards exit 0; exact Qwen restoration/warmup/free lock are independently
verified at 10:39:43 UTC. See
`../deepseek-v41/receipts/hybrid-lookup-20260918/README.md`.
The current stack includes the strict allocator and exact input-row cache.
Later bounded screens at source19ea3ac reject early-hit submission (1.21845%
slower) and source-matched read alignment (flat within control spread). A true
80/40/24 draft subset removes733,224,960B payload and retains198 target calls in
head-only replay, with different boundaries. Its111-slot full candidate refuses
loading at the live11,137,220,608B background because its bound is110,308,317,404B.
The ceiling stays110GB. Larger draft cuts need206 or200 calls and do not justify
full runs. There is no new TPS/output result or test suite. All guards restore
exactQwen/warmup and release; final independent check11:43:05UTC succeeds.
See prelaunch-hits-20260918, read-alignment-20260918 and draft-surrogates-20260918.
Prior phase-lifetime audits are complete; do not repeat them. Next work needs
material expert-I/O or verification reduction. Fixed Q8 and full 256K prefill
remain secondary. Work inline, no agents, with minimal checks.

## Historical cache-budget update, 2026-09-18

Task 4 remains open. The cache-budget candidate completes at 12.6712544 TPS /
80.7339166 s with exact 1,024 native output IDs and 84->103 slots. It keeps
1 GiB allocator cache during prefill and sets 256 MiB at the existing quiescent
growth boundary. Two fixed 64 MiB Engram arenas have zero full-run evictions.
Pricing recovers 1,011,844,096 B while retaining all existing overshoot, compile,
copy, KV and wired margins. At the actual 11,964,268,544 B baseline, the old
configuration admits 102 slots and this candidate admits 103. Its different
capacity prevents an isolated speed comparison with the retained 104-slot run.
The physical bound is 109,677,955,304 B; independent machine peak is
107,034,165,248 B. Headline cache reporting follows the installed phase.

Three bounded operators precede that one full run: expert zero-cache is flat;
attention zero-cache loses; the stable 256 MiB attention comparison is flat with
exact output/state. No broad suite or further optimization test is justified.
All guards are terminal exit0 and exact Qwen restoration, warmup, health and lock
release are verified. See
`../deepseek-v41/receipts/cache-budget-20260918/README.md`.

The native grouped BF16 projection audit is also complete. One bounded fused
decode/transpose operator has exact weight and output parity across five real
layers, but is 45.71% slower than cached BF16 at M6. Its projected 1.98-second
full-workload overhead and unchanged growth-copy limit reject it without a
full-model run or new test. Guard96675 is terminal0 with restoration verified.
See `../deepseek-v41/receipts/woa-fused-transpose-20260918/README.md`.

The subsequent strict-allocator stage completes the separate growth, seed and
steady-decode lifetime audit. The historical next-step proposal was:
Do not discount the original overshoot allowance from endpoint readings. A
growth-only zero cache setting may be considered separately from the rejected
zero-cache attention route; steady-phase pricing still requires an allocation
inventory and a proved boundary after seed. No new full-model run is staged.

## Earlier update, 2026-09-18

Task4 remains open. The memory-composed full native run reaches12.8091055TPS
at84->104 slots; all output IDs match. The subsequent one-record miss candidate
reaches12.826718TPS /79.755398s at the same capacity, under a109,591,219,432B
bound with107,051,008,000B sampled machine peak. Its0.1375% point difference is
not a reliable full-model improvement; the default remains unchanged.
Two bounded one-layer comparisons win before two focused CPU lifetime checks
and that single full run. Full6 attention remains: prefix operators add too
much cost to justify the proposed split decoder. See
`../deepseek-v41/receipts/miss-batches-20260918/README.md` and
`../deepseek-v41/receipts/prefix-operators-20260918/README.md`.
That phase-specific cache investigation is completed by the current update
above. Exact Qwen restoration, warmup and lock release were verified.

## Earlier measured update, 2026-09-17

Task 4 has full-workload results; **20 TPS remains unmet**. The latest single
complete candidate is 12.6731624 TPS at depth 5/cap 84->102, with
106,288,578,560B maximum sampled whole-machine usage during the DeepSeek run.
See `../deepseek-v41/receipts/plane-overlap-20260917/README.md` for exact output,
bounded integration and the distinct post-restoration Qwen memory finding.
The earlier native growth stage below reached 12.1146645 TPS at cap93->100;
see `../deepseek-v41/receipts/post-prefill-cache-growth-20260917/README.md`.
The initial cap-94 result below was 11.7203483 TPS.

The accepted prefill change evaluates MTP hidden captures at the existing
chunk fence, freeing earlier Hyper-Connection graphs. After exact slot-byte
normalization it saves 3,011,286,868B of allocator peak; cap 94 safely uses that
space. The promoted module AST matches the measured candidate, and both
existing tiny quantized prefill cases pass. There is no additional hot-path
validation, fallback, instrumentation, or synchronization.

The full workload selects depth 5 even though the short prefix favored depth 3.
`3,3` verification loses, so Tasks 3e-3h remain historical staging only: their
conditional q8/eighteen-transient/tuned-capacity ladder is superseded and must
not run unchanged. Full-M6 replay also rejects the tuned policy. Nonuniform
allocation is only a roughly 3% read-reduction projection, without GPU proof.
The target head stays native BF16 and the winning route retains 48 transients.

The DSpark decode-start memory field now samples after prefill (e589c1e4b).
The Bash 3.2 guard handles empty auxiliary arrays (c9f090573), and automatic
Qwen cache reclamation/restoration has been observed in every completed window.
AR-reference reuse records null public AR measurements and explicit provenance;
a hashed diagnostic row avoids repeated AR replay while MTP logits stay fresh.

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
- [x] Preserve allocator-limit readback, derived allocator GC threshold, and
  peak-over-limit with authoritative byte fields plus decimal-GB and binary-GiB
  renderings. Do not relabel the raw limit readback as the GC threshold.
- [x] Print allocator peak, process `phys_footprint` peak, and whole-machine
  physical-used peak by their full names in the standard runner and retain all
  three in its fastest-of summary; keep the legacy `peak_gb_highest` only as an
  allocator-peak compatibility alias.
- [x] Add a single-arm DSpark acceptance gate that permits byte identity or an
  index-matched `tie_flip`, but returns nonzero for genuine or suspect
  divergences after preserving the diagnostic receipt.
- [x] Reject unclassified control/candidate token differences even when a
  rounding lever changed. Each arm's DSpark-vs-AR classification applies only
  to that arm; it cannot establish parity between different target heads.
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
- [x] Fix the runner/model seam so `--verify-shared-overlap` actually hands the
  shared branch to every streamed target layer. Bind the routed/shared callable
  once after switch installation, fail if any routed layer cannot install it,
  and report the installed layer count instead of trusting the requested flag.
- [x] Close the optional shared-work attribution span on the all-hit and
  completed split-verify early returns. These routes already ran the shared
  branch once, but previously left resource telemetry reporting it as open.
- [x] Stage cap-95 M6 as a separate conditional 128-token arm. Require a
  same-commit cap-95 control, the unchanged 89,424,018,248-byte engine geometry,
  and exactly forty bound overlap routes. Permit the 1,023-token command only
  after the M6 screen beats cap-95 control by decode wall time.
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
- [ ] Before promoting q8, compare its AR output with the matched BF16-head
  control on the same prompt and output length. Matching Q8 AR and Q8 DSpark
  outputs alone do not pass this gate. A cross-head mismatch requires paired
  logits at the first divergence on the same prefix; classify it outside the
  timed pass only after the speed screen wins. Unclassified differences remain
  ineligible for promotion.
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

### Task 3f: Stage tuned admission and nonuniform cap 94

**Files:**
- Modify: `mtplx/expert_streaming_models.py`
- Modify: `mtplx/expert_streaming.py`
- Modify: `mtplx/expert_runtime.py`
- Modify: `mtplx/expert_slots.py`
- Modify: `mtplx/models/expert_mlx.py`
- Modify: `scripts/deepseek_v41/ab_decode_env_levers.py`
- Modify: `scripts/deepseek_v41/bench_standard_shape.py`
- Conditional wrappers and commands under `/tmp/dsv41-110-stage/`.

**Security flag:** none

**Does NOT cover:** Automatic capacity learning, per-request policy changes,
prefetch, islands, a hot fallback, or promotion from replay alone.

- [x] Preserve `transition-window` as the unchanged 0.7/0.2/0.1, 16-route
  control. Add `transition-window-tuned` as a separate construction-selected
  0.8/0.1/0.1, 32-route arm.
- [x] Reject the `4,2` verify schedule: its median adds 196 reads to `3,3` in
  exchange for only fourteen fewer target forwards. Reject the stage-conditioned
  cache ranker because every coarse candidate regresses the tuned control.
- [x] Replay tuned uniform cap 94 over all eight acceptance orders: median
  36,061 reads versus 36,547.5 for the prior policy, with no memory change.
- [x] Add an explicit 40-layer capacity vector to the immutable plan. Require
  DeepSeek-V4.1 component banks, layer scope, no islands, and no prefetch; reject
  vectors whose exact component bytes exceed the resolved cache budget.
- [x] Allocate and resolve every physical persistent slot from its layer's
  construction-time capacity. Keep the enabled route free of metadata checks,
  environment reads, counters, and fallbacks.
- [x] Report `slots_per_layer=null`, the full layer-to-capacity map, and the
  uniform-equivalent scalar for nonuniform plans. Keep all three memory measures
  separate in benchmark receipts.
- [x] Solve the full cap-94 vector: median 34,829 reads at the same 3,760 slots.
  Constrain the staged arm to five bank shapes `(73, 88, 96, 112, 128)` to bound
  graph-shape variation; its median is 34,896.5 reads, only 67.5 above the
  unconstrained optimum and 1,164.5 below tuned uniform cap 94.
- [x] Confirm without importing MLX that the plan contains 3,760 slots,
  70,690,406,400 persistent bytes, and the exact 40-layer vector. Run syntax and
  whitespace validation only; do not add regression tests before a measured win.
- [x] Stage the 128-token tuned-policy arm after uniform cap 94, then the
  five-shape arm after tuned uniform. Each 1,023-token wrapper additionally
  requires its own 128-token arm to beat the matching predecessor.

### Task 3g: Exchange low-presence draft experts for nonuniform cap 95

**Files:**
- Conditional benchmark wrappers and commands under `/tmp/dsv41-110-stage/`.
- Promote only a measured winner into tracked model-loading code.

**Security flag:** none

**Does NOT cover:** General-serving MTP pruning, unverified draft-token commit,
automatic expert selection, or promotion from route replay alone.

- [x] Derive the retained draft inventory deterministically from the pinned
  route trace. Rank each stage by distinct cycle appearances with expert id as
  the tie break, retain `(67, 49, 27)`, and stack tensors in ascending expert-id
  order so the compact lookup and physical expert axis agree.
- [x] Stream three true subset safetensors from the existing authenticated
  `(93, 58, 32)` artifacts with `F_NOCACHE`, verify every source payload digest,
  and retain only the 143 selected records. Do not rely on
  `ResidentShardReader.retained_names`, because macOS `mx.load` still
  materializes every tensor in its input file.
- [x] Remove forty additional draft experts, exactly 752,025,600 bytes, and
  assign that band to one target-cache slot in every routed layer. The fixed
  footprint becomes 17,827,535,688 bytes; cap 95 plus the 154,050,560-byte
  remainder reproduces the unchanged 89,424,018,248-byte engine budget.
- [x] Solve a five-shape cap-95 vector with 3,800 slots. Its eight-order target
  replay median is 34,453 reads versus 34,896.5 for five-shape cap 94. The draft
  subset replaces 59 distinct stage-cycle expert incidences across 40 of 206
  recorded cycles, so replay is only a screen and not a throughput claim.
- [x] Fix the cap-94 nonuniform wrapper's construction check so it does not
  reference the CLI-derived target-slot value before that value is bound.
- [x] Gate cap 95 on matching five-shape cap-94 evidence. Require its own
  128-token arm to beat cap 94 before allowing the 1,023-token run.
- [x] Before GPU measurement, improve only the equal-frequency selection tie.
  Rank all stage/expert residents by distinct-cycle presence globally; eleven
  residents tie at the capacity boundary and nine fit. Among those 55 subsets,
  retain the one touching the fewest trace cycles, then use lexicographic order.
  The resulting `(69, 48, 26)` split keeps the same 143 residents, same
  2,688,491,520 payload bytes, and same 59 omitted stage-cycle incidences while
  reducing affected cycles from 40 to 35. Stage it separately so the earlier
  artifact and its receipt remain auditable.
- [x] Run syntax, shell-parse, arithmetic, artifact-subset, and no-MLX static
  checks only. Add focused tests only if the measured arm wins.
- [x] Reclaim Qwen's clean model pages immediately after its captured process
  tree exits, before testing the post-stop availability threshold. Bind every
  staged prepacked command to separate receipt-covered compact-MTP and q8-head
  auxiliary roots; the original compact-MTP receipt did not name the head file.
- [x] Restore the guard's documented 100 GiB child-footprint default. Its live
  clamp now supplies the real secondary bound (`110 GB - measured baseline`)
  instead of the stale 93 GiB cap rejecting an otherwise compliant plan.

### Task 4: Measure and promote only winners

**Files:**
- Create only after a win: `docs/deepseek-v41/receipts/<dated-stage>/`
- Add focused tests only for winning lanes.

**Security flag:** none

**Does NOT cover:** Any GPU execution while another job owns the lane or codec
implementation without a separate direct-decoder gate.

- [x] User authorized retry. Acquire the exclusive lock before MLX; wait for
  other owners and never steal their lane.
- [x] Use `scripts/deepseek_v41/gpu_window.sh` directly. It acquires the shared
  lock before bootout, captures the exact Qwen identity, reclaims the stopped
  service and candidate file caches, enforces the live 110,000,000,000-byte
  whole-machine ceiling, restores Qwen, and releases the lock last. Do not nest
  it under `bench/laguna/run_guarded.py`; both guards own the same lock.
- [x] Run one short matched batch with separately selectable corrected cap-83
  control, three-record miss parts, causal-policy, shared-overlap, and `3,3`
  staged-verify arms. Remove losing candidates; screen finer partitions only
  after the corresponding coarse candidate wins.
- [x] Screen the winning stack at cap 89. Attempt cap 91 only after cap 89
  confirms allocator and whole-machine headroom.
- [x] Reject the dependent `3,3` ladder after its measured loss. Keep native
  BF16 target arithmetic, full verification, and48 transients. The hidden-capture
  lifetime win funds cap94 through separately measured whole-workflow bounds.
- [x] Run the complete workload at depth3 and depth5 with matched cap94.
  Depth5 wins11.7203483 vs11.1612548 TPS; full output digest is unchanged.
- [ ] Run the exact 16,384-input/1,024-output Python workload for the winning
  stack and require at least 20 decode tok/s.
- [x] After the measured lifetime win, run the two existing tiny quantized
  prefill checks once. Reuse the18 passing CPU memory-reporting checks; no new
  optimization test module or broad GPU suite.
- [x] Record guard exit, service health/model identity, lock release, token
  digest or allowed tie classification, physical reads, wall time, and all three
  memory measures.

## Self-review

- Every design requirement maps to a task.
- Experimental tests are deferred until measurement; the existing planner test
  changes because memory admission correctness is already established.
- Policy names, fixed weights, cache scope, and excluded paths are explicit.
- Component compression remains a separate stage because it needs an artifact
  and decoder design plus independent performance evidence.

## Additional Task4 result, 2026-09-17 13:43 UTC

The post-MoE HC combine now has a separately measured prefill-only compiled
callable. Existing evaluation fences and diagnostic timing hooks remain.
The full1024-token MTP digest and indexed AR tie classification are unchanged.
Atcap93 the complete-run allocator peak is95,208,121,956B; the separately
measured seed+decode peak is88,755,252,592B. Maximum sampled machine usage is
105,458,794,496B. The historical capacity-normalized allocator difference is
about1.186GB; actual old import-bound globals were not recorded, so do not
present it as a fresh identical-flags paired causal measurement.

The existing strict layer-major/chunk-major regression passed under guard.
The final inactive timing hook has CPU-only equivalence and registration
checks. No broad suite or new test module was added. See
[the receipts](../deepseek-v41/receipts/hc-post-prefill-20260917/README.md).

The latest full decode result is11.6513TPS at93slots, not a throughput winner.
Task4 stays open. A larger post-prefill cache needs a coherent single-bank
resize implementation and copy/ownership/memory bounds; the measured decode
peak is evidence for that design, not permission to reuse a prefill slot bound.

## Additional Task4 result, 2026-09-17 14:29 UTC

One-request post-prefill cache growth93->100 improves the complete workload to
12.1146645TPS from a fresh93-slot control at11.6574573TPS (+3.922%). The1.9541s
resize is charged to decode, and35,880 expert records replace39,093. Full output
is identical. Maximum sampled machine usage105,642,098,688B remains under110GB;
the allocator full-run peak stays95,208,120,648B because prefill still dominates.
The phase controller preserves bank identity and allocator-plan ownership,
synchronizes each old backing release, and rejects another prefill. This is
measured benchmark code; general-serving shrink/reload is not implemented.

The initial candidate phase alias was stale; its recorded DSpark slot_plan
already contains the correct100 slots. The immutable original, derived phase
correction and CPU check of the actual reporting assignments are archived.
See[the receipts](../deepseek-v41/receipts/post-prefill-cache-growth-20260917/README.md).
Task4's20TPS acceptance remains unchecked.


## Additional Task4 result, 2026-09-17 15:08 UTC

Fanout8 is not promoted. Its fixed93/100 full-model attempt was refused before
loading because a12.209GB baseline plus bounded allocations and added worker
headroom exceeds110GB. A small native component-scatter CPU A/B/A then gives
only0.351% bandwidth gain while doubling preadv calls. It does not measure GPU
overlap or complete decode throughput. Main layers have384 experts, versus128
in MTP; the initial probe's wrong main-inventory assumption failed before reads.
The corrected probe verifies the manifest inventory and landed byte hashes.
All owned windows restored exact Qwen identity, health and warmup before release.
See[the receipts](../deepseek-v41/receipts/native-scatter-fanout-20260917/README.md).
Task4 remains open at12.1146645TPS. Do not repeat this fanout screen or weaken
admission to fit the current baseline.

## Additional Task4 result, 2026-09-17 16:24 UTC

Native D7/M8 is not promoted. Exact target-state replay screened it at176 versus
206 cycles; the native cap16 allocation probe then matched129 output tokens.
The full16K/1024 run at91 prefill/98 decode slots gives11.4315253TPS,176cycles
and40,607 physical expert reads. Full output is identical to retained MTP.
The lower cache capacity differs from the retained93/100 best, so this is not
a matched estimate of width alone. It does not beat12.1146645TPS; no additional
GPU control or new optimization test module was run. Maximum external sampled
machine usage105,610,772,480B is below the109,314,275,908B admission bound and
110GB ceiling. Exact Qwen restoration/warmup/lock release completed16:15:08UTC.

The A/B summary now selects the requested measured pass for throughput, memory,
cross-arm token identity and decode budget. Previously a reused AR reference
printed unavailable measurements and could hide a different DSpark output.
The24-case CPU memory/reporting suite passes with real MLX imports forbidden;
the previous source fails the new regression. Both benchmark growth wrappers
also abort through runtime cleanup when a critical resize callback fails,
instead of allowing the telemetry callback to swallow that failure. A focused
CPU control-flow check covers failure and unchanged success.

Teacher tensors are preserved locally with hashes so this capture need not be
repeated. Measured wrappers and raw receipts remain immutable; corrected
wrappers are archived separately. See
[the receipts](../deepseek-v41/receipts/native-draft-width-20260917/README.md).
Task4's20TPS acceptance remains unchecked.

## Additional Task4 result, 2026-09-17 process-reader stage

The process memory reader now caches only Mach/ctypes bindings, reducing its
CPU A/B/A median from 24.167/22.1875 us controls to 1.125 us. A fresh 16 MiB allocation
is observed immediately. Successful truncated replies now stay unknown instead
of reporting zero footprint. 27 focused CPU reporting cases pass with real MLX
imports blocked; the installed SDK confirms the ABI. This is sampling overhead
reduction, not a measured decode throughput gain.

Direct host_statistics64 was rejected despite a 700x microbenchmark: kernel rate
limiting returned stale system counters while platform vm_stat observed 18.53 MB
of growth. The production system reader remains unchanged. Eight online cache
policies and post-service admission also fail CPU selection and are not installed.

A guarded full D5/M6 diagnostic at 91->100 slots preserves the complete output.
Its cProfile thread attribution is inconsistent and reproduced on CPU, so its
function timings and 10.7873 TPS are not optimization evidence. Sampled internal
machine peak 106.117 GB is below the 109.431 GB conservative bound. Exact Qwen
restoration/warmup/lock release completed 17:01:33 UTC. No new GPU work follows.
See [the evidence](../deepseek-v41/receipts/memory-reader-20260917/README.md).
Task4 remains open at the retained 12.1146645 TPS.

The subsequent native small-M HC screen is not promoted:12.2191TPS has204 vs206
cycles and an additional unclassified control/candidate token difference at480.
Native M6 synthetic outputs are not bit-exact despite the old row-cap comment;
comments are corrected without changing the executable AST. Retained best remains
12.1146645TPS. See the HC follow-up in the same receipt folder.

## Additional Task4 result, 2026-09-17 explicit read attribution

CPU-validated main-thread boundary timing attributes52.271397s to expert-read
waiting/completion and19.940574s to Metal evaluation/encoding, out of84.671768s.
Only0.350480s is expert graph construction. Native D5/M6 at84->98 matches the
entire output, but its instrumented11.79836TPS is not a throughput control.
Independent machine peak105,505,128,448B fits the109,126,510,060B bound.

Causal prompt lookup loses the CPU screen. Native confidence0.5 reduces head-only
verification rows, but full84->99 yields12.12157TPS,217cycles and a new output
trajectory. Its additional AR divergence376 lacks a reference logit row, so no
tie proof exists. The candidate is not promoted and receives no optimization
tests. Exact Qwen restore/warmup/lock release completes17:56:43UTC.

Missing diagnostic logits now report unclassified instead of asserting a non-tie
difference; acceptance remains strict.33 focused CPU reporting checks pass with
real MLX imports blocked; no generation arithmetic changed. Evidence:
[decode-read-attribution-20260917](../deepseek-v41/receipts/decode-read-attribution-20260917/README.md).
Task4 stays open at the retained12.1146645TPS.

## Additional Task4 result, 2026-09-17 18:37 UTC

A15-record CPU entropy screen rejects cross-expert XOR/reference coding.
Sorting existing native decode banks also cannot select MLX0.32.2's sorted
reuse route at36 assignment rows over98-100 persistent or48 transient slots.
A corrected native down specialization preserves all sampled output bytes,
but the complete MLP gain is7.67% at18rows/3experts and only0.82-0.94% for
larger cases. Keep it as a prototype; no production kernel or full-model run.

The short probes exposed guard-summary ambiguity. It now prints sampled
peaks with exact bytes, observation count and cadence; no complete child sample
means n/a. Admission, readers, cadence and restoration are unchanged. One
focused CPU regression covers missing, measured-zero and nonzero cases;
19 existing hermetic memory/abort checks pass. No broad GPU tests were added.
Qwen restored and warmed before lock release18:31:39UTC, independently checked
18:37:02UTC. See[the evidence](../deepseek-v41/receipts/read-kernel-screen-20260917/README.md).
The retained12.1146645TPS and Task4's open20TPS criterion are unchanged.

## Additional Task4 result, 2026-09-17 resident packed scales

The exact full Python workload reaches 12.4439935 TPS / 82.2083358s with all
1,024 output IDs unchanged. A complete lossless 3,086,136,060-byte scale inventory
covers all 15,360 target experts. The one-request phase replaces raw scales,
grows weights once from91 to102 slots per layer, and reads only three native
weight planes on a miss. Its3.4511s installation is charged to decode. Weight
reads total620,341,493,760B plus3,086,136,060B of scale installation reads.

Sampled machine peak106,215,473,152B is below the109,483,268,328B bound and110GB
ceiling. The historical native best remains12.1146645TPS at93->100; the new result
is not a fresh paired comparison. A native91->100 D5 control is staged, pending
an idle Qwen service and the shared GPU lock.20TPS remains unmet.

The CPU exporter exceeded its proposed incremental-memory bound because source
file cache accumulated despite F_NOCACHE; actual physical used stayed below110GB.
Its Qwen restore timed out, then verified-source reclamation restored exact Qwen
health/warmup. Do not rerun that exporter unchanged. The full runtime candidate
uses direct Metal destinations and automatically reclaims its source/packed file
cache in the guarded finally. Its guard exits0 and restores Qwen before releasing
the lock at19:39:28UTC. Another job subsequently acquired the lane.

The candidate exposed stale source-sized slot/transient reporting. Both runner
formatters now report current record storage, preserve source_expert_record_bytes,
and use active-plan transient bytes. Four new CPU cases fail before the fix;
all37 focused reporting cases pass afterward with real MLX imports forbidden.
Only reporting ASTs change; original receipts remain immutable and a derived
metadata correction retains their complete timings, memory readings and tokens.
See[the evidence](../deepseek-v41/receipts/resident-packed-scales-20260917/README.md).

### Native control retry, 2026-09-17 20:08 UTC

The complete native D5 cap91->100 control was refused before model loading: the
post-Qwen baseline was 28.353 GB versus the unchanged admission maximum 9.672 GB.
Qwen identity, health, warmup, and lock release were independently verified.
Do not retry with unchanged headroom or overwrite this attempt's prefix.
Task 4 remains open; see
`../deepseek-v41/receipts/resident-packed-scales-control-refusal-20260917/README.md`.

### Completed cap84 native/packed comparison, 2026-09-17

Using the already-measured 84-slot prefill envelope resolved fixed 91 admission
failures without weakening any memory margin. One guarded full 16K/1024 batch
measured native 84->98 at 11.7141789 TPS and packed 84->99 at 12.2253796 TPS: an
observed 4.364% gain and 3.6517s less decode time. Both produce all 1,024 identical
output IDs. External machine peaks are 105.847 GB/105.400 GB; both fit 110 GB.

The pair includes different live baselines and maximum-admitted decode
capacities; it does not isolate the kernel contribution or prove repeatability.
Current source/current-storage reporting is correct in both real receipts.
Qwen identity, health, warmup and lock release were independently verified.
Task 4 and the 20 TPS goal remain open. See
`../deepseek-v41/receipts/resident-packed-scales-pair-20260917/README.md`.

### User-requested 110 GB and fixed Q8 KV, 2026-09-17

The user authorized fixed Q8 KV and asked that RAM use be standardized at110GB.
This changes KV precision explicitly; it does not change the earlier q8 output
head decision. The new cache is separate from the old disabled bounded-KV lane.

- [x] Unify decimal/GiB defaults and staged admission on110,000,000,000B.
  The recorded baseline admits100 packed slots at109,708,761,320B; a one-byte
  excess at the110GB boundary reduces capacity. Other allowances remain.
- [x] Add fixed packed target window/compressed/index and draft KV; keep native
  compressor arithmetic in fixed rolling stores. Resolve geometry and reserve
  all backing/copy/view allowances before expert allocation.
- [x] Expose `--kv-cache-bits 8 --kv-max-append 953` for max KV17664 and report
  storage/reserve geometry in both real benchmark receipts.
- [x] Verify112,503,168B constant backings, packed snapshot/rollback, draft seed,
  native detach and tiny real-model forward/trim. Fix owner reference cycles.
  Allocator peak274,186,240B;975,688B active after teardown.45 focused CPU cases
  and the legacy budget assertion pass with real MLX imports blocked.
- [x] Verify automatic Qwen reclamation and exact healthy/warmed restoration.
  No abandoned DeepSeek process was found; active other jobs were preserved.
- [x] Establish a fresh complete full-model Q8 memory envelope and Q8 reference,
  then measure the exact workload and throughput. The cache lifetime probe
  bounds retained prefill views; the full envelope adds the complete Q8 reserve
  without discounting native allowances. Keep native default/control.

Evidence: `../deepseek-v41/receipts/fixed-q8-budget110-20260917/README.md`.
The independent packed-geometry screen rejected all three candidates; see
`../deepseek-v41/receipts/packed-geometry-screen-20260917/README.md`.
Task4 and the20TPS goal remain open; best full result remains12.4439935TPS.

### Full fixed Q8 result and priority update, 2026-09-17

The complete Q8 candidate yields10.7541904TPS/95.1257102s, including3.3733s
packed installation, at84->101 slots/layer. All1024 output tokens are present;
the first difference at53 passes the existing index-matched tie gate against a
new Q8 AR reference. Complete FP32 reference rows are archived by hash for all
1024 positions. This is workload evidence, not a broad model-quality assessment.

Calculated physical bound109,817,256,168B and guard peak106,263,920,640B both
fit110GB. Final Qwen restoration, health/warmup, no owned child and free lock
were independently verified. No additional optimization tests or full rerun
are justified for this slower candidate. See
`../deepseek-v41/receipts/fixed-q8-full-20260917/README.md`.

The user requires at least256K KV support, explicitly secondary to20TPS.
- [x] Account for262144-token fixed Q8 storage:552,567,168B with an additional
  2,952,790,016B reserve. The existing factory accepts that configuration.
- [ ] Reach20TPS on the unchanged16K-input/1024-output Python workload.
- [ ] Establish and verify the complete256K prefill/rollover envelope, including
  retained chunk views and hidden states. The16K full bound does not cover it.

### CPU causal prefetch screen, 2026-09-17

Four causal route predictors were screened on the full206-cycle M6 trace,
using the actual transition-window bank at102 slots and chronological103/103
training/evaluation halves. The best nontrivial precision is13.69%: cross-layer
top1 could move3.54% of physical reads earlier while adding22.30% traffic, even
with unlimited lead time. No layers pass the training-half80% precision gate.
Reject this family without GPU execution, production changes or new tests.
The1.535-second CPU screen imports no MLX and uses47.3MB of predictor arrays.
Evidence: `../deepseek-v41/receipts/causal-prefetch-screen-20260917/README.md`.

### Strict allocator cache, 2026-09-18

Matched MLX0.32.2 host builds retain byte-identical Metal shaders. Strict free
and limit reduction enforce the configured inactive-cache capacity. Bounded
native attention and206-route expert comparisons preserve exact outputs/state
and show no material stable-case regression. The loaded library is attested
before crediting the2,258,155,644B overshoot allowance; all other reserves stay.

One complete16K/1024 run reaches13.1509467TPS /77.7890768s at84->109 slots,
with all1024 native IDs identical. At the actual10.240868352GB baseline, stock
accounting admits105 and strict109. Bound109.979515100GB; machine peak
109.671972864GB. Expert reads fall to571.822571520GB. This is the best single
result, not a matched-capacity/repeatability claim.20TPS remains unmet.

Only after that improvement,3 targeted cache regressions were added; all fail
on stock and pass on strict. Every guard is terminal0; exactQwen health/warmup
and free lock were independently checked07:24:19UTC. No production package was
replaced. Strict admission credit requires the pinned strict binary; stock
keeps its overshoot allowance. No broad suite or unchanged full rerun.

Task4 remains open. See
`../deepseek-v41/receipts/strict-cache-20260918/README.md`. Preserve this result
while reducing expert traffic or exposed verification time.256K KV prefill
verification stays secondary to20TPS.

### Packed operators and causal router diagnostic, 2026-09-18

Two bounded expert operators are rejected: grouping rows by physical slot is
1.72% slower, and compiling the complete clamped activation is flat. All206
outputs/reads match. No full-model runs or regression tests follow those arms.
See `../deepseek-v41/receipts/packed-operators-20260918/README.md`.

A CPU screen on the historical W35 hidden trace identifies a better causal
feature: apply the next gate to the current native router input after attention.
That trace uses a different16K prompt and256AR rows. One exact16K/1024 diagnostic
then captures64 nativeM6 cycles without prefetching or changing cache policy.
All1024 output IDs and2368 observed layer routes match the saved native trace.
Diagnostic fences invalidate TPS comparisons; headline timing fields are null.

With configurations chosen on the first32cycles and evaluated on the next32,
the later feature predicts763/4878 physical misses(15.64%) while adding134reads
(2.75%), at85.06% precision. The existing feature covers4.86%. Configurations
vary by layer; no tested global setting meets the training precision threshold.
This is an unlimited-lead-time estimate. A bounded paired-layer I/O experiment
is justified before a full prefetch implementation or performance claim.

The diagnostic reserves384MiBhost plus64MiBMetal and grows84->105slots. Its
107.986800748GB bound covers the107.152556032GB machine peak. The corrected
variant checks the real CLI budget resolver; the first launch refused an
omitted CLI host reserve before model allocation. Guard77187 exits0; exact
Qwen restoration/warmup and free lock are verified. The retained winner stays
13.1509467TPS. Task4 and20TPS remain open;256K prefill remains secondary.
See `../deepseek-v41/receipts/router-feature-20260918/README.md`.

### Paired prefetch I/O screen, 2026-09-18

Two-layer105/48/16 replay shows5.11% lower summed held-out pair latency with
.93% control spread and exact128 outputs. This optimistic screen excludes live
router computation/attention and leaves hashing/terminal drain outside timing.
It does not establish full TPS or shared-ring behavior across adjacent sources.
11GiB incremental bound; MLXpeak5.305GB, final8B. Guard40813 exits0; exactQwen
restoration/warmup/free lock independently verified. Next is a continuous
three-layer cost screen.20TPS stays open; retained full result13.1509467TPS.
See `docs/deepseek-v41/receipts/lookahead-io-20260918/README.md`.

### Adjacent prefetch and mixed-row rejection, 2026-09-18

Three continuous layer30/31/32 screens pay native gate-shaped cost on synthetic
inputs while replaying exact-workload prediction scores. Last-GU issue loses
2.27%; first-GU/demand-priority and NumPy ranking are flat. All192 outputs match
per arm, memory fits14GiB incremental, final Metal16B. No full-prefetch/default
or tests are promoted. See the lookahead-adjacent-20260918 receipt.

A C109 census motivates one mixed pair/single kernel, preserving native dot
order and removing the earlier pair/single dispatch split. All206 native layer34
outputs/reads match, but ratio1.0038815 is within.6348% control spread. Reject;
no full run or new tests.9GiB incremental bound; final Metal8B. See the
mixed-row-pairing-20260918 receipt. Guards48399/99230/41743/58512 all exit0,
reclaim source pages and restore exactQwen/warmup/free lock; checks are fresh.

Best remains13.1509467TPS;20TPS is open. The directpreadv path already avoids
a Python payload copy, and fanout8 is previously rejected. Next work should
establish reliable coarse CPU/wait attribution, avoiding the invalid cProfile
data and repeated unchanged full runs.256K prefill remains secondary.


### CPU attribution and exact input-row cache, 2026-09-18

A fixed coarse clock diagnostic preserves all 1,024 native IDs and seven native
MLX calls, adding no GPU fences. Decode cycles take 73.4290 s elapsed versus
13.5526 s main-thread CPU and 39.6364 s process CPU. Expert entrypoints dominate
elapsed time. Public TPS fields are null, and elapsed-minus-CPU is not GPU idle.
See the cpu-attribution-20260918 receipt; the older read-attribution receipt
remains valid. No need to repeat this diagnostic.

An exact BF16 input-row cache then frees 1,323,827,200 Metal bytes after prefill.
The fixed 16 MiB arena receives 32 MiB host allowance; actual retirement and
clean source-page reclamation precede expert-bank growth. No prefill credit.
One successful full candidate reaches 13.3195300 TPS / 76.8045117 s, all 1,024
native IDs identical, 84->110 slots, native KV16. Machine peak 109.255884800 GB
fits its 109.849655516 GB bound. Expert reads fall by 380 records / 6.724 GB.
It is 0.9845652 s faster than retained strict, but background/capacity differ;
claim only a new best single result. 20 TPS and full 256K prefill remain open.

The first full attempt stops before growth on two str-versus-Path reclamation
calls. Both are fixed; failed evidence is retained. Final guard 26008 exits 0,
restores exact Qwen and warmup, releases 10:11:42 UTC; independent healthy/free
check passes 10:11:58 UTC. Two host-only resource regressions are added after
the win and pass. No broad suite or unchanged full rerun. Runtime source is
575c3c8b3beb0420d16fc03c727f3a27c0f36edd. See embedding-rows-20260918 receipt.
Task 4 stays open. Next work needs material expert-I/O/verification improvement,
not repeated cache or prefetch families already rejected. Work inline, no agents.

### Native MTP plus causal lookup, 2026-09-18

The complete candidate retains all native D5 proposals and adds up to two
past-text continuations, verified by the unchanged native target path. The
head-only replay predicts 198 rather than 206 cycles; one full run confirms
198 calls, exact 1,024 IDs and 13.4141517619 TPS. The 109.238927360 GB machine
peak remains under the 110 GB ceiling. This is a small single-run improvement;
20 TPS and complete 256K prefill are still open. Two focused host regressions
pass after the win. Longer extensions do not justify another GPU run based
on added target work in a CPU-only screen. All lifecycle checks pass. See the
hybrid-lookup-20260918 receipt for exact source, accounting and limitations.

### Reader executor screen, 2026-09-18

The reader-hop screen at source85c7a33fd9f171c3847d7e59f8566b8b72de7225
is not promoted. Batching each native miss part and running its fill on the
existing miss worker preserves all206 layer34 outputs/reads at110/48 slots.
Median ratio0.9871322 is within1.4375% control spread; no full run or new tests.
The9GiB bound covers3,047,281,161B MLX peak; final Metal8B. Guard33095 exits0,
restores exactQwen/warmup/releases10:55:23UTC; independent healthy/idle/warmed/
free check10:55:55UTC. Receipt:reader-hop-20260918. No owned child remains.
Retained full result13.4141518TPS; Task4 and20TPS stay open.
