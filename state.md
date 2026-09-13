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

## 78-slot exact workload and MTP locality, 2026-09-13 23:11 UTC

- **Best full workload is now 9.498634789 decode TPS; 20 TPS remains unmet.**
  Source `5c7661db48a1bed22cef753e336f9aba911add24` has unchanged production
  model code from the prior full run. Exact 16,384-input/1,024-output Python
  benchmark, native MTP block5, fanout4, pf0, original MXFP4 artifact.
  Scoped benchmark transient planning band 11->6 GiB admits 78 target slots/layer
  rather than72. Keep the general runner's conservative reserve for other shapes;
  this candidate is bounded for the pinned workload, not a global default.
- Full MTP decode107.699688s / 9.498635 TPS versus114.444762s / 8.938810 TPS,
  **+6.262854%**. I/O 50,552 records/950,409,953,280B, down4,142 records
  and77,872,250,880B. AR6.289236 TPS. Both complete AR and MTP ID streams equal
  the prior full run; same tie_flip at297,206 MTP cycles,91.6388% acceptance.
- MTP MLX peak91,897,033,560B. Physical full-workflow peak103,392,870,400B
  (250ms samples,2,516 samples) versus108,632,200,780B static admission bound
  and110e9 hard ceiling. Baseline10.2927GB,2GiB Python,2GiB allocator cache,
  2GiB graph margin beyond exact slot delta; zero prefill saving credited.
  Swapouts unchanged4,399,765; guard0, exact Qwen restored/warm and GPU lock free
  independently checked. Receipt:
  `docs/deepseek-v41/receipts/target-slot-band6-20260913/`.
- Diagnostic full-length native MTP gate capture at71 target slots recorded
  206 cycles /618 stage route arrays, with all1,024 IDs equal to the prior MTP
  run. MTP stages used93/58/32 distinct experts; a hypothetical cold LRU bank
  of72/48/24 slots would free exactly6 target slots/layer but add213 draft
  record reads/4.0045GB plus three new host routing barriers/cycle. No MTP
  streaming implementation or speed claim; bounded source reads and native
  ownership remain required. Receipt:
  `docs/deepseek-v41/receipts/mtp-route-capture-20260913/`.
- Next: profile target verify's non-I/O work and screen a deployable cache policy
  against actual MTP verification routes; AR-only clairvoyant curves are
  opportunity bounds, not MTP proof. Avoid another full run for minor slot gains;
  78 slots already uses the conservative physical admission band.

## Block-6 screen, 2026-09-13 22:44 UTC

- **Best full workload remains 8.938810168 TPS; 20 TPS remains unmet.**
  Source `8a8803ad86108c3448331a8e880fe2a0b891e809` contains only the prior
  full-run receipt commit above the measured source. No production block-6 lane
  was installed. Paired short diagnostic receipt:
  `docs/deepseek-v41/receipts/dspark-block6-short-pair-20260913/`.
- Same pinned16K Python input/65-output prefix, pf0/fanout4,71 target slots/layer,
  48 shared transients, native MTP head. Block5/6 took14 cycles each, averaging
  4.642857 outputs/cycle; both streams equal each other and the archived AR
  prefix. Decode wall9.242825s vs10.111764s, 6.924290 vs6.329262 TPS:
  **block6 loses8.593351% on this slice.** Decode SSD bytes83,850,854,400 vs
  94,210,007,040 (+12.354260%); 551 more records (4,460 vs 5,011).
  Acceptance87.9310% vs87.5000%. Reject block6 for this prompt slice;
  do not spend a full1,024-token GPU run on it without a new signal.
- Native real-head T6 probe loaded the exact10,597,621,640B head/embed/output
  payload, with15,415,845,136B MLX load peak inside25,249,119,226B bound;
  eager/compiled draft IDs matched, first-five proposals changed versusT5.
  Tiny CPU-pinned target M1/4/6/7 logits+KV were exact with matched prefill
  routes. Real-dimension one-layer M6/M7 attention peak1,132,996,826B under
  8GiB; no new swapouts. Receipt: `receipts/dspark-block6-probes-20260913/`.
  The first exploratory tiny check had a mismatched compiled prefill; fixed
  route setup made it exact. The attention wrapper's raw `complete:false`
  reflects its mishandling of a normal CLI `SystemExit(0)`; a separate
  validation record proves complete output/guard/restoration. Raw saved.
- Block5/6 static physical bounds105,169,989,004/105,494,689,004B against
  110e9; measured250ms peaks99,251,191,808/98,581,299,200B. No added swaps.
  Both guard0, exact Qwen restored/warm, GPU lock free, independently rechecked
  after the final guard. No GPU child remains.
- Next speed work: screen ordered native MTP expert-route locality before
  selective MTP residency; no stream lane from topology alone. Alternatively
  use current measured peak to reprice target expert slots with explicit
  physical/wired headroom, then benchmark the exact full workload. Keep full
  default MTP block5 as the control. Do not infer20TPS from short diagnostics.

## Current full measurement, 2026-09-13 13:22 UTC

- **Best full workload: 8.938810168 TPS; 20 TPS remains unmet.** Source
  `1f3b9bca7ae5aab6a4bf30eb969b1f6dd0368711`, native MTP depth5, pf0, fanout4,
  72 expert slots/layer, 48 shared transient slots. Full receipt:
  `docs/deepseek-v41/receipts/dspark-layer-prefill-20260913/`.
- Layer-major prefill releases each layer's fp32 wo_a cache after its existing
  `mx.eval(hs)` fence. Reuses the weight across that layer's chunks; preserves
  <=8-row custom-chunk verification, MTP stage caches and nonfused decode rebuild.
  Generic fixed reserve remains unchanged. Two guarded quantized tiny-model
  checks passed after meaningful red evidence, with exact logits/hidden/KV,
  within-layer reuse and two requests. Receipt:
  `receipts/layer-major-projection-lifetime-20260913/`.
- Exact pinned 16K/1,024-output workload. Both complete AR/MTP ID streams equal
  the prior run; the full AR/MTP tie_flip record at297 is unchanged. AR6.055910
  TPS, MTP114.444762s decode/108.412733s prefill,206 cycles,91.6388% acceptance.
  Gain4.470918% versus8.556267; combined release and four-slot capacity change,
  not isolated speed attribution. Current MTP I/O54,694 records/1,028,282,204,160B,
  down5.349139%;77.737013s active I/O window,13.2277GB/s.
- MTP MLX peak **87,384,896,236 B**, down2,226,389,720 B despite3,008,102,400 B
  additional expert storage. Decode-end active76,587,741,124 B/cache2,121,803,959 B.
  Whole-workflow physical peak **98,520,252,416 B** (250ms,2,575 samples), including
  replay; process footprint peak88,413,815,240 B overlaps it. Swapouts unchanged.
- Baseline9.4325GB, engine86,608,856,288 B,11GiB transient band,2GiB Python and
  2GiB allocator cache. Admission credited zero release savings; physical bound
  107,420,597,476 B, active bound93,693,130,180 B. Prior replay crosscheck
  102,616,514,976 B. Guard0, exact Qwen healthy/warm and lock free at13:22:16 UTC.
  Session75934 is terminal0; no GPU child remains. No run needs restarting.
- Static next-lead notes: `receipts/mtp-expert-residency-20260913/`. All three
  MTP expert banks occupy7,219,445,760 B; partial residency needs bounded direct
  source reads and separate exact-layout/pricing/ownership work. Actual ordered
  MTP routes are unmeasured; capture diagnostic-only existing index arrays and
  serialize after normal fences before selecting a capacity. No streaming lane
  was implemented or promoted.
- The then-next block6/M7 screen completed and lost on the actual-context short
  pair above. M8 crosses a documented numerical boundary and has no workload
  acceptance or peak proof. Keep the native block5 control.
- Full benchmark wrapper from this run still performs AR, MTP and divergence
  replay. The separate diagnostic MTP-only window now reuses its loader/guard,
  intercepts `_generate` to run native `_generate_dspark`, writes scoped results
  and exits through `_run_arm`'s existing finally. It did not fabricate AR
  results or promote diagnostic throughput. Recompute all new bounds from
  measured87.384896GB at72slots plus exact
  storage/shape deltas and the whole-workflow crosscheck. Do not run a new full
  model geometry before its peak is bounded.

## Prior full measurement, 2026-09-13 13:03 UTC

- **Best full workload: 8.556266508 TPS; 20 TPS remains unmet.** Source
  `3a8b17284ef3ed6fa06cd130a0a56700e926ba62`, native MTP depth5, pf0, fanout4,
  68 expert slots/layer, 48 shared transient slots. Receipt:
  `docs/deepseek-v41/receipts/dspark-cache-owned-20260913/`.
- Exact pinned 16,384 input IDs and 1,023 decode steps plus prefill token;
  both 1,024-ID AR/MTP streams separately match the previous run. Their first
  difference remains the verified tie_flip at297, AR margin0 and MTP margin0.25.
  AR5.790674 TPS; MTP119.561493 s decode,108.434765 s prefill,206 cycles,
 91.6388% acceptance. Gain1.11655% vs previous MTP8.461786 TPS.
- Combined AR-prefill allocation fixes and mutually exclusive projection caches
  lower whole-workflow physical peak to **99,469,770,752 B**, including replay:
  5,441,667,072 B below prior full run despite one extra expert slot/layer.
  Current MTP active peak **89,611,285,956 B at68slots**; decode-end active
  73,579,638,724 B and allocator cache2,122,337,529 B. Process peak
  88,283,806,280 B is separate from physical used; never add the two.
- Baseline10.3676 GB; engine83,526,272,640 B; 2 GiB Python, 2 GiB allocator
  cache,13 GiB transient band. Admission bound108,032,934,248 B credited no
  reclamation saving. Prior whole-workflow/replay cross-check106,730,905,248 B.
  Guard0; exact Qwen restored healthy/warm, lock free. Swapouts unchanged at
  4,399,765. No GPU/guard child remains.
- MTP I/O:57,785 records,1,086,394,982,400 B,82.187483 s active I/O window,
  13.2185 GB/s, QD14.601. Uncached CPU fanout screen showed no useful gain
  above current4. Receipt: `receipts/verify-io-uncached-20260913/`.
- Real MXFP4 entropy screen completed,12 records/225.6MB: zlib1 saves9.1506%,
  whole-record order0 ideal6.7587%, separate-component ideal10.7525%.
  Exact source hashes/roundtrips, no artifact changes, no MLX, guard0/Qwen restored.
  Existing streamed rANS adds host copies and serial per-record decode; enabling
  it is not a next-step speed win. Receipt: `receipts/mxfp4-entropy-20260913/`.
  Ledger's blanket K>3 veto was withdrawn; native depth5 has actual evidence.
- **Next optimization:** avoid accumulating all40 fp32 attention caches during
  layer-major prefill. Existing per-layer `mx.eval(hs)` settles its consumers;
  releasing that layer's dense cache afterward preserves within-layer reuse.
  Preserve nonfused decode and custom tiny-chunk verify behavior, plus native
  MTP stage caches. Generic planner still reserves full fp32 lifetime. One
  quantized tiny-model gate should cover chunk reuse, release before next layer,
  later decode rebuild, hidden/KV parity, then measure the exact full workload.
- New allocation bounds must start from current measured MTP active peak at68
  slots plus exact storage delta and graph margin, and cross-check whole-workflow
  replay. Do not spend an unmeasured prefill-saving estimate. Keep bounded-KV and
  unvalidated W126 disabled; no per-token eligibility checks or proof counters.
- An external AR-only historical35slot hit-order replay was not promising:
  current2Q88371 misses vs route-ordered hits88439. No source promotion and no
  MTP/TPS claim. Frequency decay is not a useful knob for 2Q's recency victim path.

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
