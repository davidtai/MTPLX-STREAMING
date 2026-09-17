# Current Goal

DeepSeek V4.1: correct memory reporting and runner bugs, then reach20 decode
TPS on exact16,384-input/1,024-output Python under110 decimal GB.
**Retained best12.1146645TPS;20TPS remains unmet.**

# Decisions

- Keep allocator, process phys_footprint and machine physical usage separate.
  Missing measurements stay null/n/a; limits are policies, not measured usage.
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
Memory/runner fixes include measured DSpark headline/comparison and phase
budget, cached process-reader bindings and rejection of truncated Mach replies.
Missing diagnostic logits stay unclassified and still fail the tie gate.
The guard now labels sampled peaks, prints exact bytes/count/cadence, and
reports n/a if no complete child observation exists. Enforcement is unchanged.

# Retained Evidence

Best source d4051aecc: D5/M6, native BF16 target head, compactMTP93/58/32,
cap93 prefill->100 decode,48 shared transients,pf0,transition-window,fanout4,
missparts3,sharedoverlap,maxKV17664. Actual HC/attention/window compile flags
false; prefill post-MoE HC combine stays compiled.12.1146644762TPS /
84.4431145420s,206cycles,35,880 physical records /674,566,963,200B.
Resize1.954109s is charged to decode. Freshcap93 control11.6574573TPS.
Full allocator95,208,120,648B; externalprocess95,784,385,944B and
machine105,642,098,688B. Growth remains one-request benchmark code.
Receipts:docs/deepseek-v41/receipts/post-prefill-cache-growth-20260917/README.md.
Full MTP digest0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.

# Attribution and Rejections

Source7953020cd explicit main-thread boundaries replace corrupt cProfile.
Native D5/M6 at84->98 slots preserves all output. Loop84.671768s includes
52.271397s missing-expert wait/completion,19.940574s mx.eval encode/wait and
0.350480s expert graph build.36,783 records/691,543,941,120B,read union53.198113s.
Exclusive times sum exactly to root; do not recapture this profile or use cProfile.
External machine105,505,128,448B versus109,126,510,060B bound;no swap growth.

Prompt lookup loses. Confidence0.5 full84->99 gives217cycles/1188rows,
12.12157TPS and extra unclassified divergence376; cached AR logits only297.
Missing-logit reporting fix has33 CPU checks; generation arithmetic unchanged.
Receipts:docs/deepseek-v41/receipts/decode-read-attribution-20260917/README.md.
Keep platform vm_stat: direct host_statistics64 hides fresh growth.
Prior process-reader and policy evidence:receipts/memory-reader-20260917.

# Latest Evidence

Source8f6c84d394: cross-expert XOR/reference coding loses a15-record CPU entropy
screen. Independent component bytes ideal89.215%, XOR94.776%, conditional89.549%.
These are entropy bounds, not actual codec sizes; no new codec installed.
Sorting existing decode banks cannot select MLX0.32.2's B/E>=4 reuse route:
36 assignment rows versus98-100 persistent or48 transient bank slots.
Local mlx-fork is0.31.2; source geometry was checked against upstream0.32.2.

A specialized native down-projection prototype avoids2304->2560 padding.
Initial ushort strides were wrong; corrected fixed/unrolled loops match every
sampled output byte. Whole-MLP comparison:unroll9 saves7.67% for18rows/3experts,
but0.94% for36/12 and0.82% for36/36. Real layer20 weights, synthetic BF16 inputs;
not full-model parity or TPS. No production kernel or full-model run this stage.
Bound6GiB; allocator peak678,724,030B; final process1,628,718,832B and machine
11,758,698,496B are boundary observations, not continuous peaks.

Short windows exposed the guard's misleading zero/rounded peak summary.
The focused reporting regression fails old source and passes the correction;
19 existing hermetic memory/abort checks pass. The stale test expectation93GiB
was updated to the already-existing100GiB child policy, not vice versa.
Evidence:docs/deepseek-v41/receipts/read-kernel-screen-20260917/README.md.

# Lifecycle and Open Work

All owned GPU children are terminal. Exact Qwen restore/health/warmup and lock
release18:31:39UTC; independent check18:37:02UTC confirms active0,warmupdone,
expected model and free lock. No unrelated process signaled. Later work CPU-only.

-20TPS remains open. Materially reduce physical expert bytes or exposed I/O wait.
- Do not repeat rejected fanout8,D7-full,staged3+3,D3-full,serial-rANS,policy,
  prompt-lookup,confidence0.5,HC-compile or target-head screens unchanged.
- Down specialization remains a limited prototype; larger MLP cases barely gain.
- One-request growth preserves bank ownership and synchronized old-buffer release;
  general serving needs physical shrink/reload before another prefill.
- Live growth helpers:/tmp/dsv41-cache-growth-20260917 and depth7-full-20260917.
  Refresh installation HEAD/source hashes after commits; measured archives immutable.
- phase_memory_control_sha256 identifies the predecessor RECEIPT, never wrapper.
- Cached teacher:/tmp/dsv41-depth-replay-20260917; hashed tensors also under ignored
  benchmarks/raw/deepseek-v41-depth-teacher/20260917. Do not recapture.
