# Current Goal

DeepSeek V4.1: correct memory reporting and runner bugs, then reach20 decode
TPS on exact16,384-input/1,024-output Python under110 decimal GB.
**Retained best12.1146645TPS;20TPS remains unmet.**

# Decisions

- Keep allocator, process phys_footprint and machine physical usage separate.
  Missing measurements stay null; limits are policies, not measured usage.
- Ceiling110,000,000,000B covers baseline, Python, Metal, caches and graph peaks.
  Reserve2GiB host; price allocator cache overshoot and temporary copies.
- Hold /tmp/mtplx-gpu-exclusive.lock before MLX or Qwen shutdown. Run
  scripts/deepseek_v41/gpu_window.sh directly, never nested. Reclaim stopped
  model file caches; restore exact Qwen identity, health and warmup before release.
- Other GPU jobs retain their lane. Work inline; minimal tests, only after wins.
  Tie breakers are allowed; arbitrary output changes are not.
- Preserve Claude W126/W127/W128 worktrees; keep unvalidated bounded KV off.

# Plan Status

Executing docs/plans/2026-09-16-deepseek-v41-20tps-stage.md; Task4 remains open.
Memory/runner fixes include the measured DSpark headline/comparison and phase
budget. This stage caches process-reader function bindings and rejects a short
Mach response instead of reporting its untouched footprint field as zero.

# Retained Evidence

Best source d4051aecc: D5/M6, native BF16 target head, compactMTP93/58/32,
cap93 prefill->100 decode,48 shared transients,pf0,transition-window,fanout4,
missparts3,sharedoverlap,maxKV17664. Actual HC/attention/window compile flags
false; the prefill post-MoE HC combine stays compiled.12.1146644762TPS /
84.4431145420s,206cycles,35,880 physical records /674,566,963,200B.
Resize1.954109s is charged to decode. Freshcap93 control11.6574573TPS.
Full allocator95,208,120,648B; externalprocess95,784,385,944B and
machine105,642,098,688B. Growth remains one-request benchmark code.
Receipts:docs/deepseek-v41/receipts/post-prefill-cache-growth-20260917/README.md.
Full MTP digest0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.

# Latest Evidence

Process-reader CPU A/B/A medians24.167/1.125/22.1875us; a touched16MiB allocation
immediately adds16,809,984B of measured footprint. Bindings only are cached.
The old reader reports0 for a successful reply missing phys_footprint; the new
reader reports unknown.27 focused CPU cases pass, real MLX imports forbidden.

Keep platform vm_stat for system usage. Direct host_statistics64 was rejected:
kernel rate limiting returns cached counters while vm_stat sees18,530,304B
of fresh growth. Its700x microbenchmark and29 schema tests do NOT justify use.

Eight causal online logistic cache policies fail the100-cycle selection window;
post-service pin relaxation yields0 promotions and0 read savings. No runtime
cache-policy change. Baseline replay35,981reads starts73+27empty slots, not the
retained93->100 prefill state. Do not claim this as exact current-run replay.

Source8f30 full D5/M6 cProfile diagnostic at91->100 matches1024tokens/206cycles,
but thread attribution is corrupt (self time exceeds cumulative time; CPU-only
reproduction confirms).10.7873TPS is instrumented, not a throughput candidate.
Sampled machine106,117,201,920B versus109,431,449,068B bound; extra profiler
host reserve128MiB. Do not use raw cProfile call counts or timings as evidence.
All evidence:docs/deepseek-v41/receipts/memory-reader-20260917/README.md.
Previous native D7/M8 loses at11.4315TPS; teacher tensors remain preserved.
See docs/deepseek-v41/receipts/native-draft-width-20260917/README.md.

# Lifecycle

All owned GPU children are terminal. Latest diagnostic child exit0; exact Qwen
restore, health, warmup and lock release17:01:33UTC. Independently verified
health/model identity/free lock afterward. Other jobs may acquire at any time.
No unrelated process was signaled. Subsequent work is CPU-only.

# Open Work

-20TPS remains open. Reduce physical expert bytes or exposed verification cost;
  fewer draft cycles alone increased total reads in the measured D7 screen.
- A fresh profile needs independently verified thread attribution or explicit
  timing boundaries. Do not repeat this environment's corrupt cProfile capture.
- Reject repeated HC-compile,fanout8,D7-full,staged3+3,D3-full,retirement-only,
  serial-rANS and target-head non-tie-divergence screens. Do not repeat them.
- One-request growth preserves one bank/layer and owns/synchronizes old backing.
  General serving still needs physical shrink/reload before another prefill.
- /tmp/dsv41-cache-growth-20260917 holds retained growth; /tmp/dsv41-depth7-full-20260917
  holds D7 evidence and the fixed wrapper. Rebuild installation source/HEAD hashes
  after commits before another GPU run. Future comparisons must match actual caps.
- Teacher scripts/proofs: /tmp/dsv41-depth-replay-20260917. Tensor files are also
  preserved under ignored benchmarks/raw/deepseek-v41-depth-teacher/20260917,
  with hashes in the new receipt; avoid another full-model teacher capture.
