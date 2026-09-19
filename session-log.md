## 2026-07-13 06:47 [saved]
Goal: Make benchmark bottleneck diagnosis resource-based and agent-readable.
Decisions:
- Correlate uncached reader throughput with queue, worker, byte, fence, CPU, and GPU evidence on one clock.
- Keep telemetry opt-in and label enabled token rates diagnostic until reproduced without instrumentation.
- Report absent GPU and DRAM measurements as unavailable; emit candidates only when their required evidence exists.
- Treat cached reader bytes as logical demand; require F_NOCACHE for SSD-ceiling attribution.
Rejected:
- Infer serialization from elapsed time or low SSD use alone.
- Treat pending Metal fences as GPU utilization or routed bytes as DRAM traffic.
Open: Capture authorized process GPU samples when attribution remains incomplete.

## 2026-09-17 15:08 UTC [saved]
Goal: Screen native I/O fanout without exceeding110GB or disturbing other jobs.
Decisions:
- Keep fanout4. Bounded CPU component-scatter A/B/A gives only0.351% bandwidth
  gain for fanout8 while doubling preadv calls; no full-model speedup is claimed.
- Full-model fanout8 was refused before loading:12.209GB baseline produces
  conservative111.672GB prefill/112.517GB decode bounds, including512MiB for
  added worker stacks and kernel I/O state. Qwen file-cache reclamation succeeded.
- Native main inventory is40x384 experts;128 belongs to MTP. An initial probe
  setup error caught this before reads; v2 uses the manifest-checked inventory.
- Archive exact scripts, raw metrics, setup/refusal logs and separate admission
  bounds. No regression tests or production runtime change for this small gain.
Lifecycle:
- All owned windows terminal; final exact Qwen restore, warmup and lock release
  at15:06:03UTC, live verified15:07:54. No unrelated process signaled.
Open:
- Best complete workload remains12.1146645TPS;20TPS remains unmet.
- Reduce expert bytes or exposed verification cost. New full-model attempts need
  a fresh baseline and bounded allocation/host/cache/compile headroom.

## 2026-09-17 [saved]
Goal: Reduce DeepSeek V4.1 memory and exact-workload decode cost within 110 GB.
Decisions:
- Evaluate captured hidden means at the existing prefill fence; lazy source graphs otherwise retain earlier layer states.
- Retain depth five for the complete Python workload; the short prefix favored depth three.
- Reuse validated AR references with null current-run measurements and hashed diagnostic logits; candidate logits remain fresh.
Rejected:
- Staged 3+3 verification and its dependent allocation ladder.
- Tuned transition policy on full six-row route replay.
- Early projection reclamation with no measured benefit.
Open:
- Reach 20 TPS by reducing I/O and verification cost.

## 2026-09-17 12:55 UTC [saved]
Goal: Improve DeepSeek memory accuracy and reduce exact-workload decode cost.
Decisions:
- Preserve unavailable MLX peaks as null; headline and detailed memory values share one observation.
- Report effective DSpark depth separately from requested depth; requesting six does not extend the native five-token head.
- Keep cache policy and capacity changes conditional on measured full-workload allocation and throughput evidence.
Rejected:
- Serial rANS decoding costs more than its saved I/O time.
- Native HC chain compilation without verified numerical compatibility.
- More prefill fences without a measured whole-run peak reduction.
Open:
- Isolate prefill combine allocations when the exclusive GPU lane is available.
- Reach 20 TPS within the 110 GB whole-machine ceiling.

## 2026-09-17 13:43 UTC [saved]
Goal: Reduce DeepSeek V4.1 memory before using more cache to approach20TPS.
Decisions:
- Bind import-time HC/attention/window flags before model construction; record
  actual booleans. Historical arm_env alone did not prove route engagement.
- Compile only post-MoE HC during layer-major prefill, preserving existing fences.
  Full1024-token MTP digest and the existing indexed AR tie classification hold.
- Preserve full-run allocator peak while measuring seed+decode separately:
  95.208GB versus88.755GB atcap93. Whole-machine sampled peak105.459GB.
- Archive measured installations, memory samples and hashed references; run one
  existing strict prefill regression after the memory win. It passed under guard.
Rejected:
- Treat the1.186GB capacity-normalized historical difference as a fresh paired
  identical-flags causal estimate; old window-memo state is unrecorded.
- Add a second persistent bank for cache growth; existing gather and device-LUT
  paths require one bank identity per layer.
Open:
- Bound and implement cache resize at drained phase boundaries, including all
  exported views, allocator closure plans, logical slots and future requests.
-20TPS remains unmet; latest11.6513TPS at93slots is a memory result, not a new
  throughput winner. Qwen restored and lock released13:39:41; another job owns it.

## 2026-09-17 14:29 UTC [saved]
Goal: Reach20TPS on exact16K/1K DeepSeek under110GB.
Decisions:
- Preserve post-prefill cache growth as a one-request benchmark; subsequent prefill needs physical shrink or reload.
- Charge resize time to decode and retain one bank per layer with stable row indices.
- Synchronize copied components before spending released-buffer headroom; account for allocation-page padding.
- Report pass-specific plans; the initial-loaded-plan header predates DSpark and can be stale for decode.
Rejected:
- Inferring freed Metal backing solely from mx.eval and releasing a memoryview.
- Promoting benchmark-only growth as general-serving memory management.
Open:
- Assess native I/O scheduling and a tighter cache-retention bound.
-20TPS remains open; exact receipts are preserved in the worktree.

## 2026-09-17 14:43 UTC [saved]
Goal: Preserve the validated cache-growth gain while pursuing20TPS.
Decisions:
- Keep HC compilation disabled: additional token divergence is unclassified and per-cycle verification does not show a useful improvement.
- Treat the HC row cap as a workload bound, not native bit-identity proof.
Rejected:
- Using the AR tie at297 to classify candidate/control divergence at480.
- Promoting an isolated HC kernel gain as end-to-end decode improvement.
Open:
- Native I/O scheduling and cache-retention accounting remain optimization candidates.

## 2026-09-17 16:24 UTC [saved]
Goal: Correct DSpark reporting and screen wider native drafts under110GB.
Decisions:
- Report and compare the requested measured pass; reused AR references cannot supply current DSpark memory, throughput, tokens or phase budgets.
- Keep D5. Full D7 uses fewer cycles but misses the retained throughput; differing cache capacities prevent an isolated width-effect claim.
- Preserve native FP32 prefill and BF16 decode teacher tensors; oracle replay screens acceptance without proving target throughput.
- Abort critical benchmark resize failures through runtime cleanup; ordinary telemetry exception handling otherwise swallows them.
Rejected:
- Promoting D7 from oracle cycle counts or the short native allocation probe.
- Comparing matching AR references to establish cross-arm DSpark parity.
- Repeating the full teacher capture when its hashed tensors remain available.
Open:
- Reach20TPS by reducing physical expert reads or exposed verification cost.
- General serving needs shrink or reload before a second prefill.
# 2026-09-17 process memory reader [saved]

Cache Mach function bindings, never measurements, for the process-footprint
reader. CPU A/B/A24.167/1.125/22.1875us; immediate16MiB growth observed. Reject
short TASK_VM_INFO replies as unknown instead of zero.27 focused CPU checks
pass with real MLX forbidden. Keep platform vm_stat: the rejected native host
reader hides18.53MB of fresh growth behind kernel rate limiting despite its
fast microbenchmark. Eight causal online policies and post-service pin relaxation
also fail their CPU screens. Full D5/M6 profiling matches1024tokens/206cycles
within109.431GB bound, but cProfile thread attribution is corrupt and its timing
is unusable. Exact Qwen restored/warmed and lock released17:01:33UTC; no owned
GPU child remains. Evidence:docs/deepseek-v41/receipts/memory-reader-20260917.
Retained12.1146645TPS;20TPS open. Next profile must validate thread attribution
before using a full-model window. No agents; no global memory writes.

## 2026-09-17 explicit decode attribution [saved]

Valid explicit timing replaces cProfile:52.271s expert-read wait/completion and
19.941s eval/encoding out of84.672s; expert graph build only0.350s. Full native
D5 output unchanged,84->98cap,109.127GB bound,105.505GB sampled machine peak.
Prompt lookup loses CPU selection. Confidence0.5 screen looks promising, but
full84->99 gives12.12157TPS/217cycles and unclassified extra divergence376.
No promotion; no repeated GPU control or optimization tests. Qwen restored,
warmup done and lock released17:56:43UTC; independent checks pass17:57:08UTC.
Missing/empty logits now report unclassified/rows_consistent=null and retain
replay errors; the tie gate remains strict.33 focused CPU checks pass without
real MLX imports. Generation arithmetic unchanged.47 hashed raw artifacts at
docs/deepseek-v41/receipts/decode-read-attribution-20260917. Retained12.1146645TPS,
20TPS still open. Next improvement must materially reduce read bytes or expose
less I/O wait; do not repeat rejected width,confidence,lookup or policy screens.

## 2026-09-17 18:37 UTC [saved]
Goal: Reduce expert transfer/compute costs while preserving accurate memory reporting.
Decisions:
- Label guard peaks as sampled, include exact bytes and sample counts, and report n/a without child observations.
- Retain down specialization as a prototype; its whole-MLP gain shrinks to about1% on larger cases.
- Check dispatch against installed MLX0.32.2; the local mlx-fork is0.31.2.
Rejected:
- Cross-expert XOR/reference coding; tested entropy bounds lose to independent component bytes.
- Sorting existing decode banks to activate native reuse; B/E is too small.
- Promoting a down-only timing gain as a complete decode improvement.
Open:
- Reach20TPS through materially fewer expert reads or less exposed I/O wait.
- Preserve exact arithmetic and bounded memory before another full-model run.

## Checkpoint, 2026-09-17 resident packed scales

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

## 2026-09-17 15:14 CDT [saved]

Goal: Resume the exact DeepSeek comparison within the 110 GB machine limit.

Decisions:

- Keep the native control pending: its current baseline cannot meet unchanged admission.
- Preserve the refusal receipt and use a new output prefix for any future retry.
- Count Qwen reclamation and DeepSeek cache residency separately from machine physical use.

Rejected:

- Retrying unchanged admission or purging unidentified caches to manufacture headroom.

Open:

- Fresh native comparison when machine headroom permits.
- DeepSeek 20 TPS target remains unmet.

## 2026-09-17 15:44 CDT [saved]

Goal: Continue the exact DeepSeek 20 TPS workload within 110 GB.

Decisions:

- Use the measured 84-slot prefill envelope when fixed 91 admission fails; preserve every existing margin.
- Compare native and packed scales in one guarded sequential batch, reclaiming owned files between arms.
- Treat the single-pair gain as combined storage/capacity evidence, not isolated kernel or repeatability proof.

Rejected:

- Repeating fixed 91 admission against changing baselines or loosening its safety margins.

Open:

- Reduce expert-read or verification cost toward 20 TPS.
- General serving requires an explicit subsequent-prefill storage lifecycle.

## 2026-09-17 16:47 CDT [saved]

Goal: Standardize whole-machine RAM at 110 decimal GB and add fixed Q8 KV.

Decisions:

- Use one 110,000,000,000-byte default across decimal/GiB interfaces and staged
  capacity checks. Preserve host, allocator cache, copy and wired-memory bounds.
- Install Q8 explicitly with `--kv-cache-bits 8 --kv-max-append 953`, max KV17664.
  Fixed target/draft backings total112,503,168B; reserve503,316,480B before experts.
- Keep compressor working arithmetic native and its history bounded. Avoid
  owner cycles so cache teardown releases Metal without waiting for cyclic GC.
- Preserve active Colima containers and editors; no abandoned DeepSeek process
  was found. Guard-managed Qwen shutdown automatically reclaimed36.54GB pages.

Verified:

- Final small guarded probe: fixed16K storage, native compressor reference,
  packed snapshots/rollback, draft seed/detach, tiny real-model verify/trim.
- Peak274,186,240 allocator bytes;975,688 active after teardown; no new swapouts.
- 45 focused CPU cases plus the legacy110GB default assertion, without real MLX.
- Exact Qwen restoration, health/warmup and lock release; no remaining child.

Rejected:

- Three exact packed-geometry candidates were flat/slower; no full rerun.
- Reusing native cache bounds or the native AR digest as Q8 full-model proof.

Open:

- Derive a new complete full-model Q8 envelope and measure Q8 quality/throughput.
- Old full-run source proofs are stale after these implementation changes.
- 20 TPS remains unmet; best complete result remains12.4439935TPS.

## 2026-09-17 17:34 CDT [saved]

Goal: Complete fixed Q8 workload evidence; preserve20TPS as the primary target.

Decisions:

- User requires at least256K KV support, explicitly secondary to20TPS. Keep the
  exact16K-input/1024-output performance workload unchanged.
- Full Q8 admission retains native envelopes and adds503,316,480B, supported by
  a fresh cache lifetime probe with per-chunk shared views held through prefill.
- Use a new full Q8 AR reference with all1024 FP32 logit rows; preserve its
  529,530,880-byte binary as an ignored artifact with whole-file and row hashes.
- Keep Q8 explicit. Its10.7541904TPS result is slower than retained12.4439935TPS;
  different output trajectories and capacities prevent an isolated comparison.

Verified:

- Lifetime Native/Q8/Native peaks669,729,292/373,304,920/669,729,292B; each arm
  releases to8 active bytes. No full model was needed for this lifetime probe.
- Q8 MTP completes1024 tokens in229 cycles at95.1257102s, charging3.3733s growth.
  Index53 divergence against Q8 AR is an index-matched accepted tie.
- Q8 MTP bound109,817,256,168B; guard peak106,263,920,640B. No new swapouts;
  independent candidate samples observe8 swapins.
- Both full guards exit0 and restore exact Qwen before releasing the lock.
  Final release22:26:03UTC; independent22:26:42 healthy/idle/warmed/free, no child.
- CPU-only256K store pricing:552,567,168B fixed,2,952,790,016B additional reserve.

Open:

- Reduce expert-read or target-verification cost materially toward20TPS; do not
  repeat unchanged Q8 throughput or add tests for its losing throughput result.
- Secondary256K full-prefill bound/verification: retained per-chunk decoded
  views, hidden states, attention and draft seeding are not covered by the16K bound.
- Receipt: docs/deepseek-v41/receipts/fixed-q8-full-20260917/README.md.

## 2026-09-17 17:41 CDT [saved]

Goal: Screen a new CPU-only causal read predictor toward20TPS.

Result: Rejected previous-route, same-layer, cross-layer and blended predictors
on the complete206-cycle M6 trace, with chronological103/103 train/eval halves
and the actual transition-window bank at102 slots. Best nontrivial precision
is13.69%;3.54% optimistic early-read coverage costs22.30% added traffic.
No layer meets the training-half80% precision gate. The1.535-second screen
uses47.3MB predictor arrays, imports no MLX, and leaves Qwen running.

No production implementation, GPU benchmark, or optimization tests are justified
for this candidate. Preserve the existing prefetch/policy construction guard.
Evidence: docs/deepseek-v41/receipts/causal-prefetch-screen-20260917/README.md.
Primary20TPS and secondary256K prefill verification remain open.
# 2026-09-18 04:18 UTC: Full native projection-owner memory win

Exact 16,384/1,024 native KV16 D5/M6 run at source2488ebf648 completes in
82.1580336s /12.4516125TPS,206cycles,84->101slots. All output IDs match the
best control. Capacity-normalized final MLX saving is exactly1,384,120,320B;
peak saving790,843,596B, now limited by native full MTP seeding. No speed win.
Live baseline10,881,843,200B; bound109,600,670,040B; external measured physical
peak105,996,976,128B. Original prefill/growth/seed bounds and all allowances
remain; projection retirement is credited only in steady decode.

Guard38104 exit0;225samples;no compressor growth;source/packed cache ends0.
Qwen restored healthy with exact identity and warmup before04:17:52 lock
release. Independent04:18:49 check:healthy,idle,warmed,free,no owned child.
Receipt:docs/deepseek-v41/receipts/woa-owner-20260917/README.md.
Next:compose measured tail2048 and projection retirement with bounded Engram
host storage. Best stays12.6731624TPS;20TPS remains unmet. No new agents,
unrelated process termination, broad suites or general-serving defaults.
# 2026-09-18 04:34 UTC: Composed memory candidate reaches12.8091055TPS

One full16,384/1,024 run at source5ab1776598:79.8650616660s,206cycles,all native
IDs identical.84->104 slots at10,467,377,152B baseline,109,900,696,808B bound,
107,546,591,232B guard physical peak. Measured MLX94,075,778,388B stays within
94,523,220,076B active bound. Compared with previous best,1.0727% faster and
14,739,701,760 fewer expert-read bytes. Different live baseline/capacity;
single best, not repeatability or isolated-kernel proof.20TPS remains unmet.

Tail2048 phase saving, native projection-source retirement and truly bounded
119,537,664B Engram arenas are composed without double-crediting phases.
Zero Engram evictions. Corrected nominal BF16 hidden-byte reporting using the
observed FP32 `[1,2048,15360]` tensor:125,829,120B. Historical receipts preserved.
Guard54606 exit0,221samples,no compressor growth,cache ends0. Qwen healthy with
exact model/warmup before04:31:30UTC release; independent04:34:07 healthy/free.
Receipt:docs/deepseek-v41/receipts/memory-compose-20260918/README.md.

CPU-only prefix diagnostic reproduces all35,164 native policy misses. First
row needs no reads50.59% of layer calls, but row splitting adds48.49% expert
compute requests. Partial-row native block cost/cache/ownership must be bounded
before any asynchronous inter-layer implementation or full GPU run. CPU import
path refusal fixed explicitly; no MLX execution or production changes.
Receipt:docs/deepseek-v41/receipts/prefix-readiness-20260918/README.md.


## [saved] 2026-09-18 — packed operator rejections and later router feature

Measured source013ba48db733659d287d40e7304e1683a0d7e179. Retained best stays
13.1509467TPS;20TPS remains unmet. Row grouping is1.72% slower; compiled clamps
are flat. Both exact206-route operators are archived, without full runs/tests.

Recovered W35 hidden trace under .benchmark-artifacts; its prompt differs from
the acceptance workload. CPU screen selects the current post-attention router
input for the next gate. One exact16K/1024 diagnostic captures64M6 cycles,
then removes hooks. All1024 IDs and2368 layer routes match; no prefetch reads.
Held-out later-feature coverage15.64% of4878physical misses at85.06% precision,
with2.75% extra traffic; existing feature covers4.86%. Configurations are chosen
on first32cycles; no tested global setting meets85% precision. This is an
unlimited-lead-time screen, not a TPS result. Next is a bounded paired-layer
I/O replay before any full prefetch lane. Native packed readers require a
thread-local PlanePart witness that speculative workers currently lack.

Diagnostic105slots adds384MiBhost+64MiBMetal; bound107.986800748GB,
measuredmachine107.152556032GB. First setup refused an omitted CLI host reserve
before allocation; v2 validates the CLI resolver. Guard77187 terminal0,
221samples,zero compressor growth. Qwen restored/warmed/released08:21:52UTC;
independent08:23:58 health/idle/warmup/free-lock check. No owned child.
Receipts:packed-operators-20260918 and router-feature-20260918. NPZ45.6MB is
pinned under .benchmark-artifacts/deepseek-v41/router-feature-20260918; /tmp
path remains a symlink. All runtime source hashes are unchanged. Work inline;
no new regression tests before a measured win;256K Q8 remains secondary.

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

## 2026-09-18 10:48 UTC [saved]
Goal: Reach 20 TPS on exact 16K/1024 Python within 110 decimal GB.
Result:
- Native five-token MTP plus at most two causal lookup tokens: 13.4141517619
  TPS / 76.2627423750 s, 198 cycles, all 1,024 native IDs exact. This is 0.7104%
  above input-row caching as a single result; no isolated/repeated claim.
- Reads increase 25 records. Machine peak 109,238,927,360 B fits bound
  109,631,928,540 B, with 1,421,996,032 B host reserve. 84->110 slots, KV16.
- Two host regressions added after the win pass. Longer extensions only receive
  a CPU opportunity screen; their added target work does not justify GPU work.
Lifecycle:
- Guards 7091/39139 terminal0; source cache cleanup and exact Qwen restoration
  succeed. Final release 10:38:22 UTC; independent health/warmup/free 10:39:43.
Open:
- 20 TPS remains unmet by about 25.11 s decode wall. Full 256K prefill remains
  secondary. No agents, broad tests, unchanged full controls or default change.
- Receipt: docs/deepseek-v41/receipts/hybrid-lookup-20260918/README.md.

## 2026-09-18 10:56 UTC [saved]
The reader-hop screen at source85c7a33fd9f171c3847d7e59f8566b8b72de7225
is not promoted. Batching each native miss part and running its fill on the
existing miss worker preserves all206 layer34 outputs/reads at110/48 slots.
Median ratio0.9871322 is within1.4375% control spread; no full run or new tests.
The9GiB bound covers3,047,281,161B MLX peak; final Metal8B. Guard33095 exits0,
restores exactQwen/warmup/releases10:55:23UTC; independent healthy/idle/warmed/
free check10:55:55UTC. Receipt:reader-hop-20260918. No owned child remains.

## 2026-09-18 11:43 UTC [saved]

Measured source:19ea3ac2f888edf5035e3a43bc314bea64f3cb6f. Three bounded receipts
are archived; retained full performance remains13.4141517619TPS,20TPS open.

- Cached-expert prelaunch is1.21845% slower in the exact206-call layer34 replay;
  source-matched CPU reader alignment is within control spread. No promotion,
  full runs or regression tests follow either screen.
- Draft router-nearest aliases use only first-half training routes. The original
  hybrid control reproduces198 boundaries;80/40/24 also takes198 calls but its
  boundaries differ.64/28/12 takes206; global52/46/26 takes200. These are teacher
  acceptance screens, not target parity or throughput results.
- Actual80/40/24 subset files remove733,224,960B payload. MLX-blocked construction
  checks admit111 target slots at the reference baseline. Guard85153 refuses
  before model loading at11,137,220,608B live background: the111-slot bound would
  total110,308,317,404B. Budget and allowances stay unchanged; no OOM occurs.
- The initial head staging path error is retained, fixed with ordered path
  replacement and AST path checks, then the head screen completes. Both head
  screens remain within49GiB incremental; MLX peak21,299,586,448B. No broader
  tests or new regressions are added because no full optimization is promoted.
- All guards reach terminal status, restore exactQwen/warmup and release. Final
  guard27488 exits0/releases11:42:08UTC; independent healthy/idle/warmed/free
  verification passes11:43:05UTC. No child or GPU job is pending.

Receipts: prelaunch-hits-20260918, read-alignment-20260918,
draft-surrogates-20260918. Do not rerun unchanged rejected screens or the
conditional111-slot candidate without a fresh source audit and fitting live
bound. Task4 stays open; fixedQ8/full256K prefill remain secondary.

## 2026-09-18 13:07 UTC [saved]

Goal: Reduce DeepSeek expert I/O and verification cost under 110 GB.
Decisions:
- Retain the 13.414 TPS full candidate; later head/CPU screens do not establish a throughput improvement.
- Keep compact learned routing conditional: transferred prediction improves coverage, but finite lead time, contention and predictor cost remain unmeasured.
- Assess Vontra 2-bit separately; its resident runtime exceeds budget, and changing target weights exceeds tie-breaker scope.
Rejected:
- Reader-pool reduction and R2 GU fusion: no robust integrated gain.
- Long conditioned suffixes and current-route features: extra work outweighs limited opportunity.
- Full-hidden raw-residual predictor: worse miss precision and coverage, much larger parameters.
Open:
- Reach 20 TPS; verify complete 256K prefill second.
- Measure compact learned prefetch economics before considering production integration.

## 2026-09-18 Q2 draft follow-up [saved]

Goal: Apply the requested 2-bit strategy toward DeepSeek's 20 TPS target.

Decisions:

- Screen draft weights first; target-Q2 scope remains unanswered and would change outputs beyond tie breakers.
- Keep KV-building projections native in teacher replay; requantizing them requires rebuilding the initial cache.
- Keep expert-only Q2 conditional: 1.42 GB payload saving comes with extra verification calls and slower head execution.

Rejected:

- Broader query/output/shared-FFN Q2: 233 calls versus 198 control.
- Treating capacity-proxy traffic reduction as full-model throughput evidence.

Open:

- Reach 20 TPS; target quantization scope remains pending.
- Measure compact learned prediction overlap; complete 256K prefill remains secondary.

## 2026-09-18 Q4 shelved; Q2 next [saved] [superseded by 2026-09-19]

Goal: Preserve Q4 work, publish its checkpoint PR, then attempt the Q2 main model.

Decisions:

- Shelve Q4 optimization and preserve every DeepSeek V4.1 worktree, including unfinished worker changes, without resetting their indexes.
- Open a draft PR on davidtai/MTPLX-STREAMING because the 20 TPS and complete 256K requirements remain unfinished.
- Test Q2 target output separately; draft-only acceptance checks cannot establish main-model quality.

Rejected:

- Describing Q2 draft screens as a Q2 main-model benchmark.
- Deleting worker workspaces or rewriting their history during checkpointing.

Open:

- Attempt Q2 main-model generation and inspect Python output.
- Reach 20 TPS; verify complete 256K prefill second.

## 2026-09-19 11:30 UTC [saved]
Goal: Resume Q4 and evaluate predictable dense prefetch within 110 GB.
Decisions:
- Stop Q2; preserve its branch and unverified final correction separately.
- Start with fixed-order Q4 query projections and bounded rotating buffers; preserve quantized values and native matmuls.
- Measure overlap against expert-read contention; reclaimed capacity alone does not prove a throughput gain.
Rejected:
- Moving Metal arrays into Python memory as a whole-machine RAM saving.
- Double-counting embedding retirement or retired packed output projections.
Open:
- Safe buffer retirement, actual overlap and complete admission.
- Reach 20 TPS on the unchanged 16K/1K workload.

## 2026-09-19 11:51 UTC [saved]
Goal: Measure predictable Q4 dense prefetch against real expert-read contention.
Decisions:
- Retain resident query weights; two-buffer streaming loses despite the two additional expert slots.
- Preserve the bounded prototype and output digests so later schedules can reuse its native arithmetic and retirement boundary.
- Treat demand-first submission as ordering, not a guarantee of SSD priority.
Rejected:
- Promoting predictable scheduling or reduced resident bytes as a throughput win.
- Extrapolating the component's omitted attention into a universal offload verdict.
Open:
- Materially reduce expert traffic or target work within110GB.
- Reach20TPS on the exact16K/1K workload.

## 2026-09-19 consensus suffix [saved]
Goal: Reduce expensive Q4 target calls without changing target arithmetic.
Decisions:
- Keep the original hybrid default; consensus reduces calls but has no absolute full-run speed win.
- Charge the additional consensus index 32 MiB in every phase; live admission remains authoritative.
- Preserve the guarded full result and causal proposal construction for future composition.
Rejected:
- Treating 109-versus-110-slot runs as an isolated speed regression.
- Adding regression tests or promoting a candidate without a measured win.
Open:
- Materially reduce expert I/O or verified target work.
- Reach 20 TPS while retaining the 110 GB whole-machine ceiling.
## 2026-09-19 compact ridge prefetch [saved]
Goal: Reach 20 TPS with Q4 inside the fixed 110 GB budget.
Decisions:
- Retain the existing full runner; finite-lead-time ridge prefetch provides no material component gain.
- Preserve raw receipts and publish corrected accounting when copied metadata contains stale descriptions.
- Keep source-layer protection and physical pin waits when moving speculative issue earlier.
Rejected:
- Promoting unlimited-lead-time prediction coverage as measured throughput.
- Repeating either one-layer-ahead schedule unchanged.
Open:
- Material expert traffic or verified target-work reduction.

## 2026-09-19 predictable projection expansion [saved]
Goal: Reclaim predictable non-MoE storage within the Q4 110 GB budget.
Decisions:
- Keep packed projection weights resident and expand ahead; recurring dense SSD traffic was too expensive.
- Add independent expert rows after native seed; second full-bank copies erase the component benefit.
- Preserve the memory-saving candidate without claiming an isolated throughput win; source, outputs and every memory phase stay attributable.
Rejected:
- Applying late projection savings to unchanged prefill or first-growth peaks.
- Treating compact storage alone as evidence of 20 TPS.
Open:
- Measure whether avoiding the first bank copy pays for extra grouping.
- Reach 20 TPS on the exact 16K/1K workload.
