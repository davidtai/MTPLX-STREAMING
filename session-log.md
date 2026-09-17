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
