# Current Goal

DeepSeek V4.1: correct memory reporting and runner bugs, then reach 20 decode
TPS on exact 16,384-input / 1,024-output Python under 110 decimal GB.
Best complete candidate: **12.4439935 TPS; 20 TPS remains unmet.**
Fresh pair: native 11.7141789 TPS, packed scales 12.2253796 TPS (+4.364%).

# Decisions

- Keep allocator, process phys_footprint and machine physical usage separate.
  Missing measurements stay null/n/a. Limits and plans are not measured usage.
- Ceiling 110,000,000,000B includes baseline, Python, Metal, caches and peaks.
  Keep 2 GiB host reserve and bounded allocator overshoot / copy / graph space.
- Use scripts/deepseek_v41/gpu_window.sh directly, never nested; acquire the
  exclusive /tmp/mtplx-gpu-exclusive.lock before MLX or Qwen shutdown.
  Never steal another job's lane or shut down Qwen while requests are active.
- Qwen shutdown reclaims its clean model-file cache automatically. Restore exact
  identity, health and warmup before release; verify recovery after failed runs.
- Work inline, no agents; minimal checks, optimization tests after wins.
  Tie breakers are allowed; arbitrary or unclassified output drift is not.
- Preserve Claude W126/W127/W128 worktrees; keep unvalidated bounded KV off.
- User explicitly requested fixed Q8 KV. The new fixed Q8 factory is separate
  from the old disabled bounded-KV lane; native 16-bit storage remains the control.

# Plan Status

Executing docs/plans/2026-09-16-deepseek-v41-20tps-stage.md; Task 4 remains open.
Reporting fixes cover actual measured passes and phases, fresh Mach footprint
reads, short-reply rejection, missing diagnostic logits, sampled guard peaks,
and current slot/transient storage versus source_expert_record_bytes.
The 37 CPU reporting cases and eight loader/budget cases pass with real MLX
imports blocked. Both latest full arms exercise correct source/current record
sizes and shared transient allocation bytes.

# Latest Fixed Q8 and Budget Work

All default budget interfaces and the staged native/packed admission helpers
now use 110,000,000,000 bytes. On the recorded 11.2549 GB baseline, native-KV
packed admission selects 100 slots/layer at bound109,708,761,320B. Exactly110GB
admits; one byte over reduces slots. Existing host/cache/copy margins remain.

Explicit settings: --box-target-gb 110 --max-kv 17664 --kv-cache-bits 8
--kv-max-append 953. Target window/compressed/index and draft storage is Q8,
group64 with FP32 metadata; compressor working rows remain native FP32.
Fixed backings total112,503,168B; loader reserves503,316,480B for all Q8
backings/copies/views before expert allocation, keeping old native allowances.
Packed snapshots/rollback preserve bytes; no cache owner bound-method cycles.

Final bounded probe passes 16K real-width storage/reference/rollback/restore,
native draft detach, and a tiny real model's prefill plus12 verify/trim cycles.
Storage stays112.5MB; MLX peak274,186,240B; after teardown975,688B active.
No full-model Q8 quality or throughput result. Native full-run growth wrappers
are now source-stale and their native cache bounds are invalid for Q8; do not
refresh hashes alone or reuse the native AR digest as a Q8 reference.
Receipt: docs/deepseek-v41/receipts/fixed-q8-budget110-20260917/README.md.

Latest guard child/guard exit0. Exact Qwen restore/warmup and lock release at
21:37:12UTC; independent21:37:40UTC healthy/idle/warmed/free, no owned child.
Automatic shutdown reclamation removed36.54GB cached pages (36.47GB measured
physical reduction). No abandoned DeepSeek process was found. Active Colima
containers ndh-runner/qwen36-webstatus and VS Code were preserved.

Completed packed geometry screen: R8/SG2, R4/SG4 and FP4 float-bit conversion
all exact but flat/slower across real shapes; no installation or full rerun.
Receipt: docs/deepseek-v41/receipts/packed-geometry-screen-20260917/README.md.

# Latest Full Workload Pair

Measured source 170a576dc; one sequential native/packed full-workload batch.
Both use native BF16 target arithmetic, compact MTP 93/58/32, D5/M6, 48 shared
transients, pf0, transition-window, miss parts 3, shared overlap, max KV 17664.
The measured 84-slot prefill replaces fixed 91; all original margins remain.
Native grows 84->98; packed scales grow 84->99 at separate live baselines.

Native: 11.7141788902 TPS / 87.3300646670s; baseline 10,974,773,248B;
whole-machine bound 109,298,427,372B; external machine peak 105,846,996,992B.
Packed: 12.2253796176 TPS / 83.6783831670s; baseline 11,254,906,880B;
bound 109,000,972,520B; external machine peak 105,399,713,792B.
Both produce 1,024 identical IDs in 206 cycles, SHA
0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.

Native reads 691,543,941,120B /36,783 records; packed reads 643,326,935,040B /
36,357 records plus 3,086,136,060B of scale installation. Read unions 53.9567s
and 49.5391s are not GPU-idle measurements. Charged phase costs 1.8932s/3.3879s.
Packed reports 17,694,720-byte weight records and 849,346,560 shared transient
bytes; the original source record remains 18,800,640B. Scale owners are separate.
This single pair includes capacity/baseline differences, not isolated kernel or
repeatability proof. No new optimization tests or production defaults were added.
Receipt: docs/deepseek-v41/receipts/resident-packed-scales-pair-20260917/README.md.

# Lifecycle and Artifacts

Guard child/guard exit 0; 462 complete samples; no compressor growth or new swapouts.
Both model children are terminal. Each parent cleanup removed 12,416,466,944
cached-page bytes to zero, separate from physical use. Exact Qwen restoration,
health and warmup preceded lock release 20:42:03UTC; independent verification
at 20:42:43UTC found healthy/idle/warmed Qwen, no owned children and a free lock.
Earlier cap91 refusals at 28.35/10.71/11.13GB are retained; do not repeat them.

Live pair helpers (now source-stale): /tmp/dsv41-prefill84-pair-20260917/{native,packed} plus
run_pair.py; used output prefixes must not be overwritten. Helpers are one-request
benchmarks and explicitly reject a second prefill. Update live source proofs
only after verifying unchanged runtime hashes when committing documentation.
Packed payload: ignored benchmarks/raw/deepseek-v41-resident-scales/20260917;
360 files /3,086,136,060B, complete source SHA coverage and exact reconstruction.
Manifest SHA b8aebeabdb0dc7c9362f644e4460771b6e0cb0ef84dfc332189733e2149c0e16.

# Retained Context and Next Work

- Earlier best packed91->102 is12.4439935TPS at 106,215,473,152B sampled physical
  use. See receipts/resident-packed-scales-20260917; do not rerun unchanged.
- Latest sources /tmp/dsv41-resident-scales-20260917 remain cap91-specific.
  Cap84 pair helpers solve current prefill admission without weakening budgets.
- Valid native attribution: receipts/decode-read-attribution-20260917. Its loop
  includes 52.271397s missing-read wait and 19.940574s Metal eval/encode. Existing
  miss handling already overlaps hits/shared work and dispatches ready parts.
- Further progress must reduce expert-read or verification cost materially.
  Small policy cleanup cannot bridge the remaining gap to 20TPS.
- Do not repeat unchanged rejected fanout8, D7-full, staged3+3, D3-full, rANS,
  XOR/reference coding, cache-policy, prompt-lookup, confidence0.5, HC or q8-head
  candidates. Down-only specialization gives small larger-case MLP gains.
- Teacher: /tmp/dsv41-depth-replay-20260917 and ignored
  benchmarks/raw/deepseek-v41-depth-teacher/20260917. Do not recapture.
- AR cached logits cover only divergence 297; never reuse them for 376 or 480.
- The CPU scale exporter accumulated source-file cache despite F_NOCACHE and
  caused a restore timeout; recovery was verified. Do not rerun it unchanged.
