# Current Goal

Resume DeepSeek V4.1 in MTPLX-STREAMING: correct memory reporting first, then
runner bugs, under **110 decimal GB including the machine baseline**. The active
performance goal is **20 decode TPS on the Python workload with 16K prefill**.
The pinned task requests at most 1,024 output tokens; the benchmark uses 16,384
input tokens and 1,023 decode steps plus the first token emitted by prefill.

# Decisions

- User permits parity differences caused by tie breakers (2026-09-13).
  Record and diagnose divergences; this does not authorize larger numerical
  errors or changes to the 110 GB memory limit.
- Never execute MLX without the parent-held `/tmp/mtplx-gpu-exclusive.lock`.
  Use the guarded runner, restore the exact service, verify health/warmup and
  lock release. A free lock alone does not prove memory headroom.
- Budget Python capacity at 2 GiB, including two 256 MiB Engram arenas and
  metadata. Metal gets the remainder after the measured system baseline.
  Its 10 GiB transient reserve includes a 2 GiB retained allocator cache.
- Expert, Engram and resident file reads use uncached descriptors on macOS.
  Physical used memory includes active/inactive file cache; speculative pages
  are free. Report bytes, decimal GB and binary GiB separately.
- Current slot layouts retain full backing arrays. Reserve configured maximum
  KV before allocating expert slots; logical eviction cannot fund physical KV.
- Keep the unvalidated bounded-KV lane out of full-model execution. Its
  underlying Metal numerical cause is unresolved; isolated cache tests remain
  available. No silent fallback and no new hot-path validation.
- Preserve Claude's separate dirty W126/W127/W128 worktrees. W126 is unpromoted
  and has no full GPU performance proof; its async lane is inert with pf0.

# Plan Status

## Current MTP measurement and next allocation fixes, 2026-09-13 12:30 UTC

- New best full workload: **8.461786462 TPS**, source `4b1e5d369`, native
  DSpark depth 5, pinned 16K Python input, all 1024 output IDs. 120.896457 s decode,
  108.907734 s prefill, 67 slots/layer, 14 GiB band, 2 GiB retained allocator cache,
  2 GiB Python, fanout 4, pf0. **20 TPS remains open.**
- Equal-capacity AR reference 5.764061 TPS still matches all prior AR IDs.
  MTP is 29.46% above the previous 6.535996 TPS 83-slot AR best, 46.80% above this
  reference. First difference at 297 is an actual AR tie: both tokens 33.75,
  DSpark 33.75/34.0, classified tie_flip. No broader parity relaxation.
- Full workflow physical peak 104,911,437,824B (includes divergence replay),
  DSpark-pass physical peak 102,210,043,904B, true MTP active peak 91,544,599,528B.
  2648 external 250 ms samples, swapouts 4399765 unchanged. Measured baseline
  10.3739 GB after automatic reclamation freed 36.233 GB. Reviewed bound 108.031 GB.
- Benchmark receipt completed, but guard exited 8 on an exit-state classification
  error. Exact Qwen/health/warmup/lock release verified after 12:24:06 UTC restore.
  No OOM. No GPU child remains. Raw receipt/OS/bounds/wrapper/output/summary:
  `receipts/dspark-python-16k-20260913/`. Its `dspark_end` phase includes replay.
- Draft real-weight isolated probe before the full run: compiled 9.357708 ms,
  eager 10.304375 ms, identical draft IDs; 16K seeding 154.706 ms. Stage-only resident
  plus embed/head 10,597,621,640B; load peak 15,415,845,136B. Committed under
  `receipts/dspark-head-peak-20260913/`; not a target throughput result.
- Full MTP 206 cycles, 4.97 outputs/cycle; draft 12.30 ms, verify 571.15 ms,
  accept 0.92 ms, commit 1.73 ms. Decode SSD 1.101 TB, 1.076 GB/output, 57.25 records/output,
  measured I/O-window 13.2378 GB/s. This is not a hardware ceiling or additive
  critical-path split; verification/SSD traffic dominate the next optimization.
- Subsequent source fixes: all five AR benchmark/profiling/replay prefills now
  use logits_keep=1 (8.472 GB output becomes 517 KB). Model.__call__ skips discarded
  target hidden captures for return_hidden=False, with correct conditional
  unpacking. DSpark return_hidden=True and hc_hidden retain their contracts.
  Five focused CPU allocation checks plus eight selected EOS/warm cases pass
  with real MLX imports blocked. No new full peak/TPS claim for these fixes yet.
- Guard bug reproduced with a 64 MiB CPU process: Darwin ps returned `?E`, rc 0,
  kill0 still alive. Guard now accepts its exact exiting-state grammar as LIVE,
  keeps all memory checks, and waits for gone/zombie normally. Bare ? and reader
  failures still fail closed. Two new red/green mocked-service cases plus the
  existing unreadable-live case pass (3 total). Scoped review found no issues.
  Receipts: `receipts/ar-prefill-and-guard-exit-20260913/`.
- **Next:** measure actual I/O improvements or reduce verify traffic before
  another large run. Memory increments must use the measured MTP peak plus
  exact storage delta and graph margin. A newly lower AR peak no longer includes
  retained target HC inputs: add those separately if deriving MTP from AR.
  Keep bounded-KV and unvalidated W126 disabled. Do not infer 20 TPS from the
  draft-only probe or top-level AR receipt field.

## DSpark memory batch, 2026-09-13 11:55 UTC

- Full-workload best remains **6.535995651 TPS**; no new full-model MTP run.
- Fixed direct DSpark prefill's all-row vocabulary head (8.472 GB at 16K) and
  full-prompt tensor lifetime in direct/served paths. `mx.take` gathers small
  independent last-hidden/window buffers; verify rows and custom forward API
  are unchanged. `deepcopy` was experimentally shallow on installed MLX.
- Final bounded memory checks: 17,920 bytes retained in all three routes,
  versus original ~34-51 MB on synthetic <64 MiB prompt arrays. Four guarded
  checks (three memory cases + tiny actual-model accept/reject parity) passed;
  two pure completion-cache lifetime cases passed. Synchronize/GC only in the
  measurement, not production decode. Receipts: `memory-budget-110/DSPARK_MEMORY.md`.
- Standalone loader now resolves with_mtp/config before expert allocation,
  sets spec.mtp_included, and reserves actual stage-count fp32 wo_a + windows.
  Component allocator and runtime share one additional-resident value.
  Canonical serving now charges the same SWA/wo_a/stage caches (its MTP raw
  manifest pricing already existed). Separate construction rejects an unpriced
  MTP head. Explicit flags beat env; custom manifest_path preserved.
- Removed benchmark approximate 7.4 GiB double deduction. Legacy reprice flags
  cannot disable loader accounting. Five focused CPU admission/selection cases
  passed after red evidence, plus five guarded existing compatibility cases.
- Real-artifact static plan at last full run's 86,369,798,112-byte engine budget:
  AR 83 slots/layer, MTP 72; additional MTP fixed bytes **8,353,408,392**.
  AR resident/fixed = 15,103,104,448 / 23,578,252,736; MTP = 23,456,512,840 /
  31,931,661,128. Component and runtime plans equal; allocation backend mocked,
  all MLX imports blocked. This proves planning, **not MTP peak safety**.
- Last GPU guard exited 0 and restored/warmed exact Qwen at 11:52:49 UTC;
  independent health/warmup/lock-free check followed. No GPU process remains.
  Swapouts unchanged (4,399,765). First expected-failure window additionally
  returned 8 (`live step state unreadable` during child exit); service restored.
  Root cause not yet reproduced; do not label that an OOM or weaken guard.
- Independent scoped review found no remaining source correctness issues.
- **Next:** establish bounded draft + verify temporary peak with small/real-shape
  accounting before full MTP. Candidate starts from pf0 + existing draft compile
  / bf16 draft head; preserve native artifact, user allows ties only. Historical
  window48 DSpark 3.56 TPS used pf24 + unvalidated bounded KV, so is not current
  110 GB evidence. Its ~2.95 accepted output/cycle motivates current batching.
  Keep unvalidated bounded-KV disabled. No per-token eligibility/proof counters.

Continuing the runner program in `docs/deepseek-v41/W95_RUNNER_DESIGN.md`.
Memory reporting and the revised default allocation have full-model evidence.
The current batch fixes fixed-slot KV admission, cross-request frequency state,
mixed-bank zero-ring pricing, and benchmark parity/plan comparison reporting.
CPU regressions and the combined guarded integration bundle (313 tests) pass.
Independent construction-guard review passed; no remaining scoped findings.
Device-fixture review added guaranteed restoration even if synchronization
raises, verified with a fake device and no MLX import.

# Evidence

- Integration branch: `feat/deepseek-v41-streaming`; historical source
  `f743d1bdc` produced the corrected-run diagnostic. Source `1fbe425ca`
  produced the clean default-budget run; see git log for later receipt commits.
- `docs/deepseek-v41/receipts/memory-budget-110/` contains raw receipts, OS
  traces, guard logs, target plans and summaries. JSONL files require explicit
  `git add -f`; do not edit historical raw observations.
- Default run: 35 expert slots/layer, **4.201764739 TPS**, 135.005 s prefill,
  243.469 s decode, **106,349,838,336 B** sampled physical peak (250 ms),
  **56,845,381,092 B** true MLX active peak. No added swapouts.
- Automatic reclamation, current default reserve, source `995522859`: **84
  slots/layer, 6.281322159 TPS**, 134.470 s prefill, 162.864 s decode. All 1,024
  IDs equal the prior controls; **106,591,666,176 B** sampled physical peak,
  **93,693,612,196 B** MLX active peak, no added swapouts. This is a 49.49%
  throughput gain over 35 slots, still below 20 TPS. Guard completed at 11:09:18
  UTC; after the session handle vanished across continuation, OS/log checks
  confirmed normal exit 0, no remaining child, restored healthy Qwen and free
  lock at 11:15 UTC. No OOM/crash occurred.
- All 1,024 output IDs match the 27-slot control:
  `2bd0ad017b9580c8fec340e297696a0bd81a7759b6c5dfe7c5d64de6d40c1090`.
  The output is a capped Python diff, not validated generated code.
- Prompt SHA:
  `38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2`.
- Current best, source `d25825554`, after decode pool transition repair:
  **6.535995651 TPS**, 83 slots/layer, 136.356 s prefill, 156.518 s decode,
  106,063,642,624 B external physical peak, 92,947,785,768 B MLX active peak.
  All 1,024 IDs equal the controls; no added swapouts. Guard exited 0 at 11:35:02
  UTC, restored Qwen/warmup and released the lock by 11:35:19, independently
  verified afterward. No GPU child remains. Full receipt prefix:
  python-16k-1024-pool-transition. Gain is 55.55% over35 slots and 4.05% over84,
  but this is not an isolated same-capacity A/B. The goal remains unmet.
- Prior guarded bundle: 332 tests plus 7 subtests. New fixed-storage/request
  fixes: 315 CPU tests, actual MLX imports blocked; 15 new cases independently
  rechecked. Construction/parity reporting: 32 focused regressions; combined integration
  bundle: 313 passed. An initial mixed-bank failure came from CPU device
  contamination; test-scoped Metal selection fixes it without tolerance changes.
- Corrected-run diagnostic: 106,163,322,880 B sampled physical peak,
  56,845,378,900 B MLX active peak, all 1,024 output IDs equal, no new swapouts.
  Admitted 17,407 KV tokens and verified release. Guard exited 0; exact service
  healthy with warmup done and lock free, independently verified after 07:16 UTC.
- Diagnostic launch mistakenly set unused MTPLX_EXPERT_IO_FANOUT=4; actual
  MTPLX_DSV41_IO_READ_FANOUT was unset (fanout 1). Its 3.783595 TPS is not
  comparable to clean fanout-4 performance. New reports stamp installed fanout.
  Three red/green fanout regressions plus prior reporting/cache cases: 38 CPU
  tests pass with actual MLX blocked; independent reviewer has no findings.

# Open Issues

- **20 TPS is not met.** Do not mark the active goal complete.
- Completed `run_python_16k_1024_oracle.py` capture is archived with raw routes,
  OS trace, token output, memory plan and guard log in the receipt directory.
  Offline replay must use try_plan_all_hits before plan to reproduce the exact
  88,637 misses; the direct-plan-only replay differed by four. Twelve synthetic
  warm-state checks match complete plans and final slots.
- Offline screen: best existing frequency decay 0.97 gives 84.282 misses/token
  versus 86.644 control (2.7% reduction), with the same captured warm contents.
  Not a full prefill comparison and not promoted. Clairvoyant bypass floor:
  51.740 misses/token per layer, 49.091 with a global 1,400-slot relaxation.
  At 20 TPS those still need 19.455/18.459 GB/s; observed clean I/O-window
  bandwidth is 12.657 GB/s, not a hardware ceiling. Eviction tuning alone does
  not support the target. Next investigate MTP/verify scheduling only after
  deriving its separate resident, KV, compile/graph peak bound on small shapes.
- Use MTPLX_DSV41_IO_READ_FANOUT=4 explicitly for the next controlled run.
  Keep unchanged controls and exact workload/token receipts; do not promote
  W126 or bounded KV without their missing validation.
- New real-shape attention census (receipts/attention-census-20260913): isolated
  eager pipeline proxy M1=24.355 ms, M6=33.664 ms. No sixfold premium reproduced;
  full-mode layer 20 is not covered and op counts are unavailable. W114/W116's
  model-math ceiling and attention-only barrier attribution are withdrawn.
- Historical window-48 DSpark receipt: nested DSpark TPS is 3.560338615;
  top-level 5.347034636 is AR. Old prompt, 61 slots, no current 110 GB proof;
  output differs from AR. Do not cite it as a 5.347 TPS DSpark result.
- Guarded clean-file-cache experiment: DeepSeek cached pages were zero. Stopped
  Qwen service artifacts held 33.771 GB cached pages; read-only msync invalidation
  reduced actual physical used from 46.877 to 13.127 GB. File identities/sizes/
  mtimes unchanged, no added swapouts; service restored and lock release checked.
  Receipts and scripts: receipts/file-cache-reclaim-20260913. Reclaim before a
  fresh guard baseline, never subtract an estimated cache count. Larger expert
  residency still requires a bounded peak and an exact-workload parity run.
- User requested automatic reclamation on Qwen shutdown, then minimum testing
  and focus on optimizations. Normal gpu_window.sh now captures the actual model
  path/process tree before stop, waits for descendants, reclaims read-only model
  cache before the new baseline, and refuses workload on helper failure. Scope is
  this runner's guarded shutdown. Six helper and 29 guard CPU tests plus existing
  shell guard suites (51 cases) passed. Review found restoration could overlap a
  surviving service descendant; a new targeted red/green regression fixes this
  with a post-bootout survivor check before restore. Do not repeat broad tests
  without a concrete new concern.
- Completed first larger-cache run used 16 GiB transient reserve, 2 GiB retained Metal
  cache, 2 GiB Python capacity and the new lower measured baseline. Reviewed
  same-graph bound adds exact persistent-storage delta plus two complete bank
  images beyond the measured 35-slot active peak. At baseline 13.127 GB it
  admitted 72 slots/layer at actual baseline 12.9284 GB; 5.834645856 TPS and
  100.216 GB sampled physical peak, all output IDs equal.
  Temporary wrapper: /tmp/dsv41-110-preflight/run_python_16k_1024_reclaimed.py.
  The normal 10 GiB reserve then admitted 84 slots at baseline 10.2463 GB and
  reached the result above. One intermediate diagnostic wrapper rejected an
  11.8336 GB baseline before model load due to an unnecessary 12 GB lower bound;
  it restored normally, and the v2 wrapper corrected that diagnostic interval.
  All raw receipts/scripts/logs are in receipts/memory-budget-110, with prefixes
  python-16k-1024-reclaimed and python-16k-1024-reclaimed-defaults-v2.
- Current decode at 84 slots: 43.840 misses/token, 0.824 GB/token, I/O windows
  67.554 ms/token at 12.201 GB/s. Those windows are not an additive critical-path
  split. Next priority is a short existing decode-timeline profile on the same
  16K Python input, plus capturing initial warm bank state at the boundary for
  causal policy screening at the larger capacity. No new hot-path counters or
  broad test runs. Use explicit fanout 4 and retain parity/memory gates.
- Short timeline completed on source 0e57cc0ff: 82 slots, 128 decode steps,
  all 129 IDs and ordered routes equal the controls' prefix. 105.513 GB physical
  peak, no added swapouts, guard exit 0 and Qwen/warmup/lock verified at 11:25 UTC.
  SSD wait 75.060 ms/token; routing barriers 46.162; gather fences 36.153. Barrier
  and fence durations include GPU and sync. Estimated probe overhead 0.057 ms.
  Full receipts are python-16k-128-timeline-largecache.* in memory-budget-110.
- The larger-cache replay exposed a prefill-to-decode policy defect: all seeded
  experts remain protected beyond the 80% decode cap. Boundary-only demotion
  gives 6281 vs 7012 misses in the measured prefix; all slots, pins and recency
  remain intact. Other fractions are worse. Hypothetical full-route continuation
  projects 42926 vs 45883, not a full measured throughput claim.
  _begin_pool_decode now trims once on the first decode route (normal/all-hit);
  transaction rollback and next-request reopening preserve prior semantics.
  Five targeted regressions plus 37 existing CPU policy tests pass with real
  MLX imports blocked. Independent scoped review found no issues. The clean full
  run above confirms useful throughput with all output IDs equal. Decode misses
  are 42368 (41.415/token), versus44848 (43.840/token) in the84-slot control.
  I/O windows63.9998ms/token at12.1662GB/s, not an additive critical-path split.
  The old storage-delta active projection underpredicted active peak by5,175,876B;
  full10GiB transient/cache reserve still covered active overhang plus2GiBcache.
  Future full-model capacity bounds must allow graph-workspace variation explicitly.
  Historical raw bounds stay unchanged. CLI help now names the actual target-mode
  defaults:110GB total,2GiB allocator cache,10GiB transient band,1.45GiB overshoot.
- Historical work already checked: window-16 switch_fastpath_b was only
  4.1339 vs 4.0191 TPS on its older 1K prompt; W127b saw <=2.5% at16K.
  Window-17 native mxfp4 full-MLP microbench was 417.083 us at M1, convention
  variant395.167 us. Do not rebuild these as presumed large wins or rerun merely
  because old prose still calls their GPU measurements pending.
- `scripts/deepseek_v41/analyze_route_cache.py` computes a tested clairvoyant
  per-layer lower bound with optional admission and temporary service storage.
  It is diagnostic, not a deployable policy or promotion throughput.
- Guarded integration uses the shared venv at
  `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python`.
- Artifact remains
  `/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4`.
  Do not substitute other quantizations or the separate MiaAI artifact.
# Projection cache ownership, 2026-09-13 12:47 UTC

- Best full workload remains 8.461786462 TPS; 20 TPS unmet.
- Uncached real-record I/O probe: fanout 1/4/8 all about 13.1 GB/s, so keep 4.
  Eight records per batch, interleaved controls, final-batch hashes exact;
  no MLX/model load. Physical peak 13.197 GB, process 0.388 GB, no new swapouts.
  Receipt: `receipts/verify-io-uncached-20260913/`.
- Fixed target wo_a duplicate ownership: cold cache builders release the other
  representation, including later prefills. Fused reload keys include scales and
  biases. Existing cache-hit paths remain direct; no added phase/env checks.
- Keep full 5 GiB target prefill reserve when fp32 caching is on; price 2.5 GiB
  in fused-only configurations. Native MTP fp32 reserves unchanged.
- Nine focused CPU cases plus five related MTP budget checks pass with real MLX
  blocked. One native real-shape Metal projection confirms exact weights and
  active bytes 168,820,752 (fp32) -> 101,711,888 (bf16T), repeated twice. Peak
  403,701,908 B including comparisons. Receipt includes source diff/SHA.
- Both guards exited 0; exact Qwen restored healthy/warm, lock free, no new swap.
- Next full-run bound must use the prior measured MTP peak plus exact slot delta
  and graph margin. Do not yet credit the full 5 GiB decode-storage saving to
  full-workload peak: prefill can still own the larger fp32 representation.
