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
budget, cached process-reader bindings and rejection of truncated Mach replies.
Missing diagnostic logits now stay unclassified; they still fail the tie gate.

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

Source7953020cd explicit main-thread boundaries replace corrupt cProfile.
Native D5/M6 at84->98 slots preserves the entire output. Loop84.671768s:
missing-expert wait/completion52.271397s, mx.eval encode/wait19.940574s,
expert graph construction0.350480s. Exclusive times sum exactly to root.
36,783 records/691,543,941,120B; read union53.198113s. Charged11.79836TPS is
instrumented, not a throughput control. External machine105,505,128,448B
versus109,126,510,060B bound; no sampled swap-in/out growth.

CPU prompt lookup loses: best575 versus206 native cycles. Native confidence0.5
head-only screen suggests209cycles/1142rows versus206/1236, but full target
run84->99 gives217cycles/1188rows and12.12157TPS, essentially unchanged.
Output differs from retained MTP at297 and AR at376; AR logits376 unavailable.
Reject candidate, not a proven tie or non-tie. External machine105,654,009,856B
versus108,774,057,452B bound;0 swapouts and32 swapins. No optimization tests added.

Runner now labels missing/empty logits unclassified with rows_consistent=null,
names unavailable sides and preserves the AR replay error. Acceptance unchanged;
33 focused CPU reporting checks pass with real MLX imports blocked. Generation
arithmetic AST unchanged. Evidence and scripts:
docs/deepseek-v41/receipts/decode-read-attribution-20260917/README.md.

Keep platform vm_stat: direct host_statistics64 was rejected for stale counters.
Process reader caches bindings only;24.167/1.125/22.1875us CPU A/B/A and fresh
16MiB growth verified. Earlier reader/policy rejections:memory-reader-20260917.

# Lifecycle

All owned GPU children are terminal. Confidence child exits4 on its output gate;
exact Qwen restore/health/warmup and lock release17:56:43UTC, independently
verified17:57:08UTC. No unrelated process signaled. Later work CPU-only.

# Open Work

-20TPS remains open. Reduce physical expert bytes or exposed verification cost;
  fewer draft cycles alone increased total reads in the measured D7 screen.
- Explicit timing now identifies the bottleneck; do not recapture this profile
  or use this environment's corrupt cProfile timings.
- Reject repeated HC-compile,fanout8,D7-full,staged3+3,D3-full,retirement-only,
  serial-rANS,online-policy,prompt-lookup,confidence0.5 and target-head screens.
- One-request growth preserves one bank/layer and owns/synchronizes old backing.
  General serving still needs physical shrink/reload before another prefill.
- /tmp/dsv41-cache-growth-20260917 holds retained growth; /tmp/dsv41-depth7-full-20260917
  holds D7 evidence. Refresh live installation source/HEAD hashes after commits.
  Keep measured wrappers immutable. Future comparisons must match actual caps.
- /tmp/dsv41-explicit-profile-20260917 and /tmp/dsv41-confidence-20260917 retain
  this stage. phase_memory_control_sha256 is the predecessor RECEIPT hash, never
  the wrapper hash. The initial confidence wrapper failed this check before load.
- Teacher scripts/proofs: /tmp/dsv41-depth-replay-20260917. Tensor files are also
  preserved under ignored benchmarks/raw/deepseek-v41-depth-teacher/20260917,
  with hashes in the new receipt; avoid another full-model teacher capture.
