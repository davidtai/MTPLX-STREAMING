# Current Goal

DeepSeek V4.1: correct memory reporting and runner bugs, then reach 20 decode
TPS on exact 16,384-input / 1,024-output Python under 110 decimal GB.
Best single complete candidate: **12.6731624 TPS; 20 TPS remains unmet.**
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

# Latest Adaptive Draft Stage

20 TPS remains open; no newer full throughput result than12.6731624TPS.
New draft-only policy: startD5, thenD7 after full acceptance orD3 otherwise.
Exact teacher replay reduces206->195 cycles; verify rows1236->1242. D5 control
reproduces all native commit boundaries. A three-way3/5/7 policy also takes195
cycles but1282 rows, so retain the two-way policy for the full screen.
Incremental bound52,613,349,376B; allocator peak21,299,586,448B; guard process
15,376,608,560B and physical31,218,286,592B. Guard26808 is terminal exit0;
Qwen restored23:49:22UTC, final independent health/warmup/free-lock check passes.

Full staging: /tmp/dsv41-adaptive-depth-20260917/full, parameter-sharing views
in adaptive_lane.py. No full-target execution yet. The existing nativeM8 bound
is retained plus16MiB host metadata; native search floor96 is removed, allowing
85..100 while preserving every memory inequality. CPU packed plans:102slots
at9.955GB baseline;93 at15.745GB;92 at16.5GB.

Fullv1 guard36864 exited1 before model loading: post-shutdown baseline28.6GB,
including19.9GB file-backed. Fullv2 guard88353 exited5 before GPU child launch:
optional OS disk-cache purge required an administrator password. Neither OOMed.
Both restored exactQwen health/warmup and logged lock release (23:58:23 and
00:05:37UTC). No live child remains. Read-only scan:experts.bin cache0B;
Qwenngram22,790,144B; these do not identify the remaining file-backed memory.

Guard07998bbd6 waits for zero active/queued work before bootout. Guard233e344e3
adds opt-in GPU_WINDOW_PURGE_DISK_CACHE=1 before fresh admission. The subsequent
fix checks cached admin authentication before lock/service changes. Three
focused CPU guard regressions pass. No unrelated application was terminated.
User async request is pending:run sudo /usr/sbin/purge in their own terminal;
never collect the password. After RAM reclamation, repin source/helper hashes
and use a freshv3 prefix with optional purge OFF. Do not rerun against the same
oversized baseline or claim the draft-only cycle gain is a TPS improvement.

The next continuation revalidated the same RAM/admin-auth blocker. Read-only
mincore audit found0 resident bytes across68,403 saved-session blobs (10.425GB)
and289 task-owned artifact files (13.576GB). A2MiB private-file positive control
returned2MiB before and0 after invalidation, so no session/artifact cleanup is
justified. Qwen remained healthy/idle/warmed; machine use133.107GB while serving,
including37.497GB file-backed. sudo -n -v still requires a password. No GPU run
or service restart occurred. A third consecutive goal turn revalidated the same
condition:37.528GB file-backed RAM while Qwen was healthy/idle/warmed, no owned
candidate, and sudo -n -v still requiring a password. The goal is now marked
BLOCKED pending external RAM reclamation;20TPS remains unmet. After the user
resumes, start a fresh blocked audit. Evidence:the receipt's ram-cache-audit
subdirectory plus the final live tool observations. The stagedv3 source hashes
are current and no additional GPU run was started.

Receipt:docs/deepseek-v41/receipts/adaptive-draft-20260917/README.md.
CPU ARC/S3-FIFO screens lost (heldout17411/17193 vs15544 demand misses), so no
GPU tests followed. Raw CPU screens:/tmp/dsv41-cache-replacement-20260917.

# Latest Packed Plane Overlap

Source ee72b77e0: full native-KV D5/M6 candidate84->102 completes at
12.6731624TPS/80.7217623s, including3.4263s phase installation. Same1024 native
IDs and206 cycles, SHA0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.
Fresh candidate vs cached nativeAR297 passes the same index-matched tie gate.
This is1.84% above the old12.4439935TPS single result; prefill84 vs91 and
background differ, so no isolated/repeatable full-model gain is claimed.

Gate/up computation starts after those planes finish; original full ReadyRoute
publication, policy, leases and deferred releases still wait for down. The
one-time post-prefill lane carries context through both split and I/O executors.
One-layer real-runtime interleaving is exact for168 outputs,3.34%/2.05% lower
M1/M6 medians; the original three-expert coupled probe was5.4-5.6% faster.
Two focused CPU success/down-failure cases verify writer-view lifetime. First
integration failure (missing thread context) is archived; no full load failed.

Bound109,891,934,440B at baseline9,955,393,536B; original allowances plus64MiB
for early roots. Allocator peak94,044,288,356B; internal process95,804,685,552B;
internal physical106,288,578,560B; guard physical106,287,955,968B (223 samples).
Full reads620,943,114,240B/35,092 records; read union47.9346s, verify74.8540s,
draft2.0636s. No new swapouts;2,372 swapins in the wider sampled phase.
Receipt: docs/deepseek-v41/receipts/plane-overlap-20260917/README.md.
Live harness: /tmp/dsv41-plane-overlap-20260917/full; pinned to measuredHEAD.
Lane is experimental and single-request; not a general serving default.

Full guard exit0/restored23:19:34UTC, independently healthy/free23:19:54.
A post-restore sample later reached132.913GB. An additional cleanup-only guard
found ZERO residual DeepSeek resident/aux pages, reclaimed22.130GB Qwen cache,
and measured10.303GB after shutdown. Restored exactQwen again23:25:05;
23:25:58 healthy/idle/warmed/free, but machine use131.652GB. Thus110GB is
verified for the DeepSeek run, not Qwen normal service. Do NOT infer a DeepSeek
cleanup defect from this growth. No Qwen configuration or other job was changed.
No live child remains (8306 full and30594 cleanup guard both terminal).

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
- New CPU causal prefetch screen is rejected: cross-layer top1 has13.69%
  precision, only3.54% optimistic earlier reads and22.30% extra traffic.
  Other previous-route/same-layer/blended predictors also lose. No GPU run,
  production change or new test;47.3MB predictor arrays,1.535s screen.
  Receipt: receipts/causal-prefetch-screen-20260917. Do not rerun unchanged.
- Do not repeat unchanged rejected fanout8, D7-full, staged3+3, D3-full, rANS,
  XOR/reference coding, cache-policy, prompt-lookup, confidence0.5, HC or q8-head
  candidates. Down-only specialization gives small larger-case MLP gains.
- Teacher: /tmp/dsv41-depth-replay-20260917 and ignored
  benchmarks/raw/deepseek-v41-depth-teacher/20260917. Do not recapture.
- AR cached logits cover only divergence 297; never reuse them for 376 or 480.
  That restriction is for native KV. The new Q8 reference covers all1024 rows.
- The CPU scale exporter accumulated source-file cache despite F_NOCACHE and
  caused a restore timeout; recovery was verified. Do not rerun it unchanged.
