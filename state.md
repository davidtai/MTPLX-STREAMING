# Current Goal

Q4 resumed at the user's request; Q2 is stopped and preserved separately.
Target: 20 decode TPS on 16,384 Python prompt / 1,024 output tokens under 110,000,000,000 whole-machine bytes. The target remains unmet.
Best single full result: 13.8688167379 TPS / 73.7626013330 s, 198 target calls,
84 original rows plus27 extension rows/layer, all1,024 native IDs exact.
This is2.5001410420s shorter than the historical13.4141517619TPS result;
background and capacity differ, so it is not an isolated repeated win.
Predictable packed-projection expansion is composed with extension banks.
Q4 only. The dense SSD and ridge-prefetch schedules remain rejected.

# Decisions

- Report MLX allocator, Darwin phys_footprint, and whole-machine physical use
  separately. Never substitute RSS, sum overlapping metrics, or use zero for
  missing measurements. Budget includes background, Python, Metal, KV, file
  cache, allocator cache, I/O, copies, and compile/graph peaks.
- Preserve the 100 GiB wired ceiling and 110 decimal GB physical ceiling.
  Best-run host reserve is 1,438,773,248 B; consensus prices 1,455,550,464 B.
  Only the attested strict allocator removes its proved 2,258,155,644 B
  overshoot allowance. Stock MLX retains that allowance.
- Before any MLX import/load/compile/run, use the parent-held exclusive
  /tmp/mtplx-gpu-exclusive.lock via scripts/deepseek_v41/gpu_window.sh directly.
  Never nest guards, steal the lock, or stop unrelated jobs. A free lock alone
  does not establish memory safety. Use fresh whole-machine/wired admission.
- Acquire before Qwen shutdown, reclaim its clean model pages automatically,
  and restore exact identity, health, and warmup before releasing. Reap only
  owned children. Distinguish subsequent foreign windows from restoration
  failures when independent health checks race another owner.
- Work inline, no agents. Minimal checks; add optimization regressions only
  after a measured win. Tie breakers are permitted; unclassified drift is not.
  Validate invariant routes at construction; no enabled-lane silent fallback,
  repeated metadata/env checks, or proof counters in measured execution.
- Q2 is stopped at its separate 2c9393fec checkpoint, including an unverified
  final prefill correction. Preserve Claude W126/W127/W128 and all worktrees.
  Q8 and complete 256K prefill verification remain secondary. A 256K allocation
  is not a verified 256K prefill; the full fixed-Q8 candidate is 10.7541904 TPS.

# Plan Status

Executing docs/plans/2026-09-16-deepseek-v41-20tps-stage.md; Task 4 is incomplete.
Memory reporting and automatic shutdown reclamation are retained. New work is
an experimental receipt checkpoint; production defaults remain unchanged.
Draft PR: https://github.com/davidtai/MTPLX-STREAMING/pull/4
All 140 worktrees and seven dirty snapshots remain preserved; inventory:
docs/deepseek-v41/checkpoints/20260918-worktrees.json. Large artifacts stay put.

# Evidence

- Extension receipt: docs/deepseek-v41/receipts/extension-bank-20260919.
  Scratch /tmp/dsv41-extension-bank-20260919; measured source d5f15e7a0.
  Keep84 old banks, install packed scales, seed native MTP, add27 rows/layer.
  No expert-weight resize; original owners survive. Growth2.1956928751s,
  1.4780189590s shorter than previous. Reads31,573 /558,675,394,560B.
  Baseline10,983,129,088B; launch estimate109,745,344,620B;
  sampled machine109,852,753,920B, process97,565,150,648B, MLX96,548,916,582B.
  Machine peak exceeds the launch estimate107,409,300B but stays below110GB.
  Do not hide this variation or treat sampling as an unsampled hard bound.
  Equal110-slot component A/B/A gains2.93676% with allocation charged;
  steady replay approximately flat,206 exact outputs/arm. Two CPU admission
  regressions pass after the full win. Output sanity unchanged/truncated.
  Guard85235 exits0, restores exactQwen/warmup, releases15:06:03UTC.
  Independent15:07:32check finds foreign60643 and no owned child/waiter.
- Predictable projection receipt: docs/deepseek-v41/receipts/predictable-expansion-20260919.
  Scratch /tmp/dsv41-predictable-expansion-20260919; measured source d569990aa.
  Native packed wo_a remains resident; next exact BF16 transpose is issued
  during current expert demand. Two buffers plus a priced replacement save
  1,098,907,648 B conservatively. One extra expert row/layer costs707,788,800 B.
  Full84→109→110: baseline11,483,348,992 B; bound109,965,277,404 B;
  guard machine peak108,678,955,008 B, process96,860,719,488 B, MLX95,840,954,166 B.
  198 calls /31,961 reads /565,540,945,920 B; all1024 native IDs match.
  40 overflow rows install in54.731209ms without moving existing owners.
  An unset expert-cache limit is preserved in the full plan/config.
  Output inspected: coherent Python diff opening, truncated at1024 tokens.
  Guard5388 exits0, restores exactQwen/warmup, releases14:33:10UTC; later
  independent check finds foreign owner28587 and no owned child or waiter.
  Component A/B/A gains1.65%; separate-row variant gains1.41% with allocation
  charged. A native second resize projects1.798s and is rejected.
- Earlier ridge, consensus and dense schedules are preserved in their dated
  20260919 receipts. Ridge is slower/flat; consensus13.3997126089TPS at109 slots
  is not an isolated comparison; query SSD offload is50.07% slower. No repeats.
- Previous full winner: docs/deepseek-v41/receipts/hybrid-lookup-20260918.
  Native D5 plus up to two lookup tokens, strict allocator, exact embedding-row
  cache, packed expert scales, native KV16. Scratch full-v1 under
  /tmp/dsv41-hybrid-lookup-20260918. Baseline 10,447,192,064 B;
  bound 109,631,928,540 B; machine peak 109,238,927,360 B; reads 31,961 /
  565,540,945,920 B. Output SHA256:
  0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.
- Earlier rejected/conditional experiments remain in the dated receipts and
  the implementation plan: cache policies, fanout8, GU gaps, HC compile,
  D7/D9/D13, prefix1+5, alignment, smaller readers, R2 GU, row pairing,
  completed-layer features, raw-hidden router prediction, Q2 draft subsets,
  and conditioned neural suffixes. Do not repeat unchanged candidates.

# Open Issues and Next Work

- Q4 needs22.6126s less decode time to reach20TPS. Avoiding the initial bank
  copy is now measured. Next work needs material I/O/verification reduction.
  Future full admission must account for observed background variation;
  do not simply reuse the historical launch estimate or147MB sampled headroom.
- GU read coalescing with a368,640B scale gap was already rejected in
  gu-combined-read-20260917 (3.37% slower at3records). Do not repeat it.
- The 80/40/24 draft subset saves 733,224,960 B but its 111-slot full bound
  refused at 11.137 GB background. Do not retry without fresh fitting admission.
- Strict library: /tmp/dsv41-strict-cache-20260918/strict-lib/libmlx.dylib;
  SHA256 32f8c0e361d6f35251c9e05aeba05563f94ae54cc1f5f8e4bcb5ec9e6c42fba9.
  Packages remain unchanged. Refresh new helper pins after commits; never
  rewrite measured receipts. Native AR diagnostic logits cover index 297.
