# Current Goal

DeepSeek V4.1: correct memory reporting and runner bugs, then reach 20 decode
TPS on exact 16,384-input / 1,024-output Python under 110 decimal GB.
Best complete candidate: **12.4439935 TPS; 20 TPS remains unmet.**
Latest full Q8 candidate: 10.7541904 TPS; keep native KV for the fastest route.
At least 256K KV support is also required, secondary to reaching 20 TPS.

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
- The 16K/1K throughput workload stays unchanged. A 256K store configuration
  does not establish the memory envelope for a complete 256K prefill.

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

The fresh Native/Q8/Native lifetime probe preserves every per-chunk shared
compressed/index view as layer-major prefill does. Native peak669,729,292B
in both arms; Q8 peak373,304,920B; all release to8 active bytes. Thus the full
Q8 envelope retains native bounds and adds503,316,480B without native discounts.

Source67c0906cbc658de6219d384cf94bbdbc0484efe2 full Q8 runs:
- AR reference84->99, complete1024 tokens, full FP32 logits at every index.
  Physical bound109,853,513,960B; internal machine peak104,161,181,696B.
  AR7.005TPS includes logit capture and is diagnostic only.
- MTP84->101:10.7541904TPS/95.1257102s, including3.3733s packed installation.
  229 cycles; complete1024 tokens. First Q8-AR difference at53 passes the
  index-matched tie gate (0.125 contested margins/deltas, band0.375).
  Physical bound109,817,256,168B; guard physical peak106,263,920,640B.
  Reads726,739,845,120B;56.658s read union;88.869s verification,2.405s draft.
  Different trajectories/capacities prevent isolating Q8 overhead. Not a winner.
Receipt: docs/deepseek-v41/receipts/fixed-q8-full-20260917/README.md.

All guard children terminal with exit0. Final exact Qwen restore/warmup and
lock release22:26:03UTC; independent22:26:42 healthy/idle/warmed/free, no child.
Candidate shutdown reclamation removed35.83GB cached pages to zero. No new
swapouts;8 swapins during candidate samples. Active other jobs were preserved.

Live source-pinned Q8 harness: /tmp/dsv41-q8-full-20260917. Never overwrite used
output prefixes. Logits also retained at ignored
benchmarks/raw/deepseek-v41-fixed-q8-reference/20260917/ar-logits.f32,
SHA c0cccefd38d50628259f2e72f671f1f4f78951244f24d1fb565a6470ff8ecc3e.
The Q8 reference is specific to Q8; it cannot classify native-KV candidates.

256K CPU geometry: max_kv262144, max_append953, fixed552,567,168B and additional
reserve2,952,790,016B. Full256K prefill is unverified: retained per-chunk views,
hidden states, attention and draft seeding need a complete bound first. This
secondary task must not delay the20TPS work.

Completed packed geometry screen: R8/SG2, R4/SG4 and FP4 float-bit conversion
all exact but flat/slower across real shapes; no installation or full rerun.
Receipt: docs/deepseek-v41/receipts/packed-geometry-screen-20260917/README.md.

# Retained Native/Packed Full Workload Pair

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

# Retained Pair Lifecycle and Artifacts

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
  That restriction is for native KV. The new Q8 reference covers all1024 rows.
- The CPU scale exporter accumulated source-file cache despite F_NOCACHE and
  caused a restore timeout; recovery was verified. Do not rerun it unchanged.
