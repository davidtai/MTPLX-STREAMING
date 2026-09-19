# Current Goal

Q4 only; Q2 is stopped and preserved. Reach 20 decode TPS on the exact
16,384-input / 1,024-output Python workload within 110,000,000,000 machine bytes.
Best single full result: 13.8688167379 TPS / 73.7626013330 s, 198 target calls,
111 expert rows/layer, all 1,024 native IDs exact. The target remains unmet.

# Decisions

- Report MLX allocator, Darwin phys_footprint and whole-machine physical use
  separately. Never substitute RSS, sum overlapping metrics, or hide missing data.
- Keep 110 decimal GB physical and 100 GiB wired ceilings. Include Python, file
  cache, KV, allocator cache, I/O, graph/compile temporaries and background.
- Before any MLX import/load/compile/run, use scripts/deepseek_v41/gpu_window.sh
  directly, with its parent-held /tmp/mtplx-gpu-exclusive.lock. Never nest guards,
  steal the lock or stop unrelated jobs; a free lock is not a memory bound.
- Acquire before Qwen shutdown, reclaim its clean pages, restore exact identity,
  health and completed warmup before release. Reap only owned children. Later
  foreign windows do not contradict a completed guard restoration.
- Work inline, no agents. Minimal checks; add regressions after measured wins.
  Permit classified tie breakers, not unexplained drift. Validate invariant
  routes at construction; no enabled-path fallback or engagement counters.
- Native KV16 remains retained. Q8 and complete 256K prefill are secondary;
  fixed allocation alone does not verify 256K support. Full Q8 reached10.7542TPS.
- Preserve Q2 checkpoint2c9393fec, Claude W126/W127/W128 and all worktrees.

# Plan Status

Executing docs/plans/2026-09-16-deepseek-v41-20tps-stage.md; Task4 remains open.
Memory reporting and automatic reclamation are retained. Production defaults
remain unchanged. Draft PR: https://github.com/davidtai/MTPLX-STREAMING/pull/4.
All140 worktrees and seven dirty snapshots are preserved in the checkpoint
inventory at docs/deepseek-v41/checkpoints/20260918-worktrees.json.
No owned GPU child or waiter remains after the latest admission refusal.

# Evidence

- Latest composition: receipts/packed-draft-composition-20260919 under
  docs/deepseek-v41. Native93/58/32 draft, packed FP32 output; target projection
  primes at84 rows before extension. CPU-only phase/CLI checks pass.403MB credit
  applies only to steady decode;128MiB GPU,16MiB host and256MiB background are
  explicit. Host+background1,723,985,920B. Fresh12,160,188,416B background refuses
  minimum111 rows before model load. Even with zero wired pressure,111 prices
 111,073,252,600B. No full result or unchanged retry. Guard exits1, restores exact
  Qwen/warmup/releases17:04:14UTC. Later check sees foreign60832/61115, no owned job.
- Packed smaller draft: receipts/packed-draft-subset-20260919. Saved80/40/24 alias
  table plus packed output yields199 calls/1248 verify rows versus198/1242 in both
  native controls. Head wall1.867674041/1.756225542/1.869658708s. Physical expert
  owners remain93/58/32 in this screen. No target execution or full promotion.
- Indexed expert input: receipts/indexed-expert-input-20260919. Same110 rows,
 206 exact outputs/arm and759 reads. Warmed1.363446458/1.347701750/1.351989792s;
  nominal0.7377% below0.8438% control spread. Not promoted. Original /tmp alias
  pin failure preserved; canonical v2 verifies33 pins. Both guards restore.
- Retained draft component: receipts/draft-packed-projection-20260919. Actual
 402,653,184B retirement; native/candidate/native1.867688667/1.748210542/
 1.866354375s.198 equal boundaries,1242 rows. One rejected depth5 proposal differs
  at position315. Target never runs; head-only saving is not full parity/TPS.
- Best full: receipts/extension-bank-20260919, source d5f15e7a0. Keep84 original
  expert rows, add27 without copying weights. Growth2.195692875s;31,573 reads /
 558,675,394,560B. Baseline10,983,129,088B; estimate109,745,344,620B; sampled
  machine109,852,753,920B, process97,565,150,648B, MLX96,548,916,582B. Measured
  machine exceeds estimate107,409,300B, yet stays below110GB. Sampling is not an
  unsampled peak bound. Future admission includes256MiB background variation.
- Target schedule: receipts/predictable-expansion-20260919. Retain1,384,120,320B
  packed output weights; two BF16 buffers, price three during replacement.
  Conservative credit1,098,907,648B, no recurring dense SSD reads. Native target
  arithmetic, KV16 and D5+two causal lookup tokens stay unchanged.
- Full output SHA256:0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.
  Output is a coherent Python diff opening, truncated at1024; not a validated patch.
- Prior smaller-draft112-row composition refuses at11.561GB background. Dense
  query SSD streaming is50.07% slower. Compiled clamped activation, row sorting,
  tested ridge/lookahead, GU-read coalescing and related rejected candidates are
  recorded in dated receipts and the plan; do not repeat unchanged versions.

# Open Issues and Next Work

- Remove22.6126s to reach20TPS. Need material expert-I/O or verification savings;
  memory savings and predictable scheduling alone do not establish throughput.
- Packed draft full output, real phase peaks and wall time remain unverified.
  The latest111-row composition needs another1.073GB at its observed background;
  do not weaken admission or repeat unchanged hoping background falls.
- The caller drops125,829,120B prompt hidden only after the seed hook returns.
  Current extension runs inside that hook: no credit for this later release.
- One complete extra expert row/layer costs707,788,800B before transient charges.
  Keep48 transient slots forM8. Smaller-band proofs do not cover this workload.
- Use only attested strict library /tmp/dsv41-strict-cache-20260918/strict-lib/libmlx.dylib,
  SHA32f8c0e361d6f35251c9e05aeba05563f94ae54cc1f5f8e4bcb5ec9e6c42fba9.
  Packages are unchanged; stock MLX retains its overshoot allowance. Refresh
  new helper pins after commits; never rewrite measured receipts.
