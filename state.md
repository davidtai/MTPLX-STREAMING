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
Memory/runner fixes are committed through the prior stages. This checkpoint
also fixes the A/B headline and cross-arm comparison selecting the AR reference
instead of the actual DSpark measurements/tokens/phase budget.

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

Source7537ead7 full native teacher atcap84 matches1024tokens/206cycles.
Preserve FP32 prefill state and raw BF16 committed hidden bits. Draft-only
replay validates D5 boundaries and screens D7 at176cycles. Compact/full D7
individual boundaries differ despite equal totals; D9/D13 lose the screen.
Native M8 target probe atcap16 matches129tokens before full-shape admission.

Full D7/M8 atcap91->98 gives11.4315252563TPS /89.4893705840s,176cycles,
40,607 physical reads /763,437,588,480B. Exact output digest preserved.
Allocator93,704,070,260B; seed/decode92,532,465,178B; externalprocess
94,506,621,600B and machine105,610,772,480B. Admission109,314,275,908B.
No throughput promotion or additional matched GPU control: it does not beat
retained best, and91/98 vs93/100 confounds an isolated width-effect claim.
Receipts:docs/deepseek-v41/receipts/native-draft-width-20260917/README.md.

A/B summary now reports actual DSpark throughput/memory, compares DSpark output
hashes and uses measured decode budgets.24 CPU reporting cases pass with real
MLX imports blocked. Previous source fails the new regression.
The benchmark resize observer now aborts through runtime cleanup instead of
letting telemetry swallow a resize failure; both live wrappers are fixed.
Measured originals are preserved. CPU checks cover failure and success paths.

# Lifecycle

All owned GPU children are terminal. Latest full D7 child exit0; exact Qwen
restore, health, warmup and lock release16:15:08UTC. Independently verified
health/model identity/free lock afterward. Other jobs may acquire at any time.
No unrelated process was signaled. No GPU work after that window in this stage.

# Open Work

-20TPS remains open. Reduce physical expert bytes or exposed verification cost;
  fewer draft cycles alone increased total reads in the measured D7 screen.
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
