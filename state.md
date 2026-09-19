# Current Goal

Q4 resumed at the user's request; Q2 is stopped and preserved separately.
Target: 20 decode TPS on 16,384 Python prompt / 1,024 output tokens under 110,000,000,000 whole-machine bytes. The target remains unmet.
Best single full result: 13.4141517619 TPS / 76.2627423750 s, 198 target calls,
84→110 slots, all 1,024 native IDs exact. This is not an isolated repeated win.
Latest consensus candidate: 13.3997126089 TPS / 76.3449209590 s, 195 calls,
84→109 slots, all native IDs exact. Different capacity prevents attribution;
it is not promoted. Dense-query streaming is 50.07% slower; compact ridge
prefetch is slower or flat. Neither component justifies full integration.

# Decisions

- Report MLX allocator, Darwin phys_footprint, and whole-machine physical use
  separately. Never substitute RSS, sum overlapping metrics, or use zero for
  missing measurements. Budget includes background, Python, Metal, KV, file
  cache, allocator cache, I/O, copies, and compile/graph peaks.
- Preserve the 100 GiB wired ceiling and 110 decimal GB physical ceiling.
  Best-run host reserve is 1,421,996,032 B; consensus prices 1,455,550,464 B.
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

- Ridge receipt: docs/deepseek-v41/receipts/ridge-prefetch-20260919.
  Scratch /tmp/dsv41-ridge-prefetch-20260919; source 3dcc16054. Three real layers,
  synthetic predictor inputs, saved routes, no attention.
  First-GU issue loses 1.0988%; earlier issue gains 0.3085% held-out but its
  whole replay loses 0.0236%. All outputs exact. Late reads fall 128→117/209;
  traffic stays 1250 versus 1214 control records. Neither is promoted.
  Actual 379-slot geometry fits 14 GiB incremental (10 Metal + 4 host).
  Summary explicitly corrects stale copied descriptions, leaving raw intact.
  Both guards exit 0 and restore exact Qwen/warmup before release; latest is
  13:26:40 UTC. Later independent check finds foreign guard 55556 and no owned
  child/waiter. Local q4-ridge-prefetch-inputs.tar.gz preserves 61 exact files.
- Consensus receipt: docs/deepseek-v41/receipts/suffix-consensus-20260919.
  Scratch: /tmp/dsv41-q4-suffix-consensus-20260919; measured source 4b5591e82.
  Native D5 and lookup plus unanimous repeated suffix, at confidence ≥0.9;
  32 MiB extra host reserve. Full baseline 11,047,714,816 B admits 109 slots;
  bound 109,546,420,444 B; guard peak 109,272,825,856 B. Reads 32,357 /
  572,548,055,040 B. Same coherent Python diff beginning, truncated at 1,024.
  Guard restores exact Qwen/warmup and releases at 12:45:47 UTC.
- Dense receipt: docs/deepseek-v41/receipts/q4-dense-prefetch-20260919.
  Real query weights plus expert I/O, synthetic query inputs, attention omitted.
  A/B/A/B/A outputs are exact. Resident110 median 1.279569542 s; streamed112
  1.920236479 s. Savings: 0.336 GB expert reads; added query reads: 8.910 GB.
  Two buffers free 1.64364288 GB payload. Reject this schedule, not all offload.
  Guard 37741 restores Qwen; independent healthy/free check passes 11:50:33 UTC.
- Retained full winner: docs/deepseek-v41/receipts/hybrid-lookup-20260918.
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

- Q4 needs 25.1127 s less decode time to reach 20 TPS. Neither current candidate
  supplies that reduction. Seek material expert-I/O or verified-target savings.
  Do not promote consensus or call its 109/110-slot comparison a regression.
- Compact correction's unlimited-lead-time quality does not produce a material
  gain in either measured one-layer-ahead schedule. Do not repeat unchanged.
- The 80/40/24 draft subset saves 733,224,960 B but its 111-slot full bound
  refused at 11.137 GB background. Do not retry without fresh fitting admission.
- Strict library: /tmp/dsv41-strict-cache-20260918/strict-lib/libmlx.dylib;
  SHA256 32f8c0e361d6f35251c9e05aeba05563f94ae54cc1f5f8e4bcb5ec9e6c42fba9.
  Packages remain unchanged. Refresh new helper pins after commits; never
  rewrite measured receipts. Native AR diagnostic logits cover index 297.
