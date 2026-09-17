# Current Goal

DeepSeek V4.1: correct memory reporting and runner bugs, then reach 20 decode
TPS on the exact 16,384-input / 1,024-output Python workload under 110 decimal GB.
**Best measured: 12.1146645 TPS. The 20 TPS goal remains open.**

# Decisions

- Keep allocator, process phys_footprint and machine physical usage separate.
  Missing readings stay null; limits are policies, not usage measurements.
- Reserve 2 GiB for Python/host state. Price Metal cache overshoot, live baseline,
  temporary copies and compile/graph peaks before loading. Retention requests
  are not hard instantaneous cache bounds. Ceiling:110,000,000,000 bytes.
- Hold /tmp/mtplx-gpu-exclusive.lock before MLX or Qwen shutdown. Use
  scripts/deepseek_v41/gpu_window.sh directly; never nest guards or disturb
  another owner. Reclaim stopped model file caches, restore exact Qwen identity,
  verify health/warmup and release the lock last.
- Minimal testing; tests only after successful optimizations. Work inline.
  Tie breakers are allowed; preserve exact output or indexed tie evidence.
- Preserve Claude's W126/W127/W128 worktrees. Keep unvalidated bounded KV off.

# Plan Status

Executing docs/plans/2026-09-16-deepseek-v41-20tps-stage.md, Task4 remains open.
Memory/runner fixes are already committed. ca207c3d7 binds actual import-time
model levers before loading. d4051aecc compiles only the layer-major prefill
post-MoE HC combine, preserving decode arithmetic and diagnostic timing hooks.

# Evidence

New receipts: docs/deepseek-v41/receipts/post-prefill-cache-growth-20260917/README.md.
Source d4051aecc. One full candidate and one fresh control, actual HC compile,
attention compile and window memo all false. Existing post-only compiled
prefill remains active. Native BF16 target head, compact MTP93/58/32, D5/M6,
48 shared transients, pf0, transition-window, fanout4, three-record miss parts,
shared overlap, maxKV17664. Only cache capacity changes after prefill.

- Fresh control cap93:11.6574572920TPS /87.7549858750s;39,093 expert records,
  734,973,419,520 bytes read. Previous same-control run:11.6513006TPS.
- Candidate cap93 prefill ->100 decode:12.1146644762TPS /84.4431145420s;
  35,880 records /674,566,963,200 bytes. +3.922% TPS, -8.219% records.
  The1.954109s resize is charged to decode. No per-layer/token instrumentation.
- Both allocator full-run peaks95,208,120,648B. Candidate seed+decode peak
  94,018,544,659B; external sampled process95,784,385,944B and machine
  105,642,098,688B. Candidate whole-machine admission bound108,757,001,708B.
- Candidate engine budget90,194,844,488 ->95,459,023,688B;
  allocator limit stays98,866,695,168B. Growth5,264,179,200 raw bytes; actual
  active growth5,263,196,160B after scale-page padding changes.
- Candidate and control full token digest
  0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.
  AR reference unchanged; indexed tie_flip297 remains valid. AR timing/memory
  are null in current-run receipts; hashed AR logits are reused independently.
- Tiny native byte-copy/ownership probe passes after adding synchronization
  before reusing allocation headroom. Allocation accounting includes16KiB
  page padding. Close leaves8B active; probe allocator cap512MiB.
- Candidate initial phase alias copied the pre-DSpark plan. The correct100-slot
  plan is already recorded under dspark.serve_stream_counters.slot_plan. Keep
  the raw receipt immutable; archived correction and actual assignment-AST
  check provide phase plans without changing usage samples or wall time.

# Lifecycle

All owned GPU jobs are terminal. The fanout8 full-model attempt was refused by
admission before model loading; Qwen restored/warmed/released14:59:08 UTC.
The small I/O probe setup failure restored15:04:20; its corrected v2 completed
and restored/warmed/released15:06:03 UTC. Live health, exact identity and lock
release verified15:07:54. Last swap2589.38MiB, below the prior2693.44MiB.
Another owner may acquire at any time; never infer ownership from this note.

# Open Work

- Cache growth is a benchmark installation, not general serving. It preserves
  one bank/layer and row indices, drains all owners, grows component by component,
  synchronizes old backing release, rebuilds the allocator with its new captured
  plan, and publishes pool/policy/config/LUT updates once. Failure aborts/cleans
  up. Another prefill is rejected; serving requires physical shrink or reload.
- Current sources/results are archived and under /tmp/dsv41-cache-growth-20260917.
  Source-pinned wrappers must be regenerated after commits; preserve measured
  wrapper versions. Future comparisons must fix actual flags and decode cap100.
- HC compile was screened at fixed93/100 and stays disabled. Native M6 probe
  is not bit-exact; comments overstating row-cap parity are corrected with an
  unchanged module AST. Full screen12.2191TPS changes tokens first at480 against
  retained MTP control; that difference is unclassified. Its204 vs206 cycles
  and nearly unchanged verify time/cycle do not support a useful HC speed claim.
  Follow-up evidence is included with the growth receipt; original full sources
  and logs remain under /tmp/dsv41-hc-decode-20260917.
- Fanout8 is screened and not promoted. Native component scatter uses Python
  preadv, not the scalar native extension. A bounded CPU I/O A/B/A gives only
  +0.351% bandwidth, doubles read calls and increases thread stack capacity.
  Main layers have384 experts, MTP layers128. The initial128-main-expert probe
  assumption failed before reads; the corrected exact inventory is40*384.
  Evidence:docs/deepseek-v41/receipts/native-scatter-fanout-20260917/README.md.
- The latest full-model attempt saw12.208914432GB baseline. With512MiB extra
  fanout headroom, prefill/decode physical bounds were111.672/112.517GB and the
  run was correctly refused before loading. Do not silently lower cache capacity
  for an allegedly matched comparison or weaken bounds to force admission.
- Next reduce expert bytes per committed token or exposed verification cost,
  using retained full counters. Inspect cache overshoot ownership before pricing
  more capacity. A depth>5 path must change actual block geometry and establish
  native-shape allocation bounds. Do not repeat HC/fanout8 screens or equate an
  isolated operation gain with full decode throughput.
- Reject prior staged3+3, D3 on the full workload, retirement-only tiny wins,
  per-chunk fences, serial rANS, tuned full-M6 policy, and target-head non-tie drift.
  A depth>5 candidate must change actual DSpark block geometry, not only CLI depth.
