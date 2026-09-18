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

# Current Draft-Width Screen

CPU position-weighted retention is rejected: native held-out misses15,544;
weighted frequency15,546; weighted predictor+frequency15,587. Unit weights
reproduce every native per-cycle count. No GPU execution or tests followed.
Receipt:docs/deepseek-v41/receipts/row-weighted-cache-20260917/README.md.

Completed bounded screen:/tmp/dsv41-even-depth-20260917. FixedD4, fixedD6 and aD7
head with a constant six-proposal cut, versus exactD5 teacher control. The
existing receipts contain noD4/D6 measurement. Four views share the native
compact weights. Bound41GiB active+4GiB cache+4GiB host; current baseline and
wired usage must fit before MLX import. The target trunk never runs. The
stdlib controller waits for actual child exit, then reclaims source-file
pages before the original guard restores Qwen. No new full model is staged.
Teacher replay is acceptance evidence only; it cannot establish changed target
arithmetic, physical reads or TPS. NativeD5 reproduces every saved boundary.
D4:238cycles/1190rows;D6:191/1337;D7cut6:189/1323 versus native206/1236.
No full target run is justified by these cycle/row tradeoffs. Allocator peak
21,299,586,448B; guard process15,197,695,592B; machine26,291,191,808B. Controller
reclaimed15,348,088,832B of source cache to0 after child exit. Guard43111 is
terminal exit0; exactQwen restored/warmed and lock released02:11:01UTC;
independent02:11:48 healthy/idle/warmed/free check found no owned child.
Receipt:docs/deepseek-v41/receipts/even-draft-width-20260917/README.md.

Completed operator screen:/tmp/dsv41-fused-gu-20260917. NativeV16/R4/SG2 gate and
up reductions share the input load and one launch; each produces the original
BF16 output, then uses unchanged clamp/SwiGLU and nativeV8 down. No weight
layout or arithmetic change. Four real whole-MLP shapes, nonidentity slots,
48-slot bank and6GiB incremental bound. Exact outputs, but whole-MLP changes
are-16.62%,+0.45%,-5.54%,+2.70% latency reduction at6/6,18/3,36/12,36/36.
Reject without runtime installation, full load or extra tests. Allocator peak
928,978,000B; active after close8B; guard process1,363,969,656B; machine
11,578,392,576B. Guard87930 terminal exit0, exactQwen restored/warmed and lock
released02:17:57UTC; independent02:19:39 healthy/idle/warmed/free, no child.
Receipt:docs/deepseek-v41/receipts/fused-gate-up-20260917/README.md.

Completed diagnostic:/tmp/dsv41-transition-cost-20260917. Attribute the existing
84-to-102 one-layer transition among raw-scale release, packed-scale load/hash
and native weight copies, within8GiB. Three real old rows cover indices0/41/83.
The full transition costs3.426s, but it is not all copying. At the best saved
baseline, steady active bound exceeds resize by514,906,000B, so eliminating
the copy peak alone does not admit more slots. Do not implement extension banks
before establishing a worthwhile copy-time saving and unchanged decode cost.
One-layer costs:0.039524s scale load/hash,0.039921s weight growth;0.079530s total.
Copy-only saving scales to about1.60s, not all3.426s; no extension-bank code.
Exact old rows; allocator peak2,483,786,252B; active after close8B. Guard76769
terminal exit0; restored/warmed/released02:26:43UTC; independent02:32:16 check
healthy/idle/warmed/free, no child. Receipt:receipts/transition-cost-20260917.

Next bounded operator:/tmp/dsv41-tail-seed-20260917. DSpark seed_main projects
every prompt hidden row, but its three caches retain only the last128 rows.
Seed-only attention is pointwise projection/norm/RoPE before append; there is
no compressed MTP history. Compare native full16K seeding to final128 rows with
absolute offsets16256, using only the12 native dense/norm seed tensors and a
synthetic16K sequence tiled from authentic saved target hiddens. Bound8GiB;
controller reclaims source pages after actual child exit. No target generation.
If useful, narrowing captures would also remove the8.053GB full main-hidden
tensor at256K. The generic MTP history API still requires all rows, so any
eventual change must be explicit to DSpark prefill. No production code changed.
First seed attempt refused at the standard eager loader's64MiB unselected
tensor cap before any comparison. Guard49374 terminal exit1; Qwen restored,
warmed and lock released02:34:21UTC; independent02:34:38 check healthy/free
with no owned child. Do not relax that loader cap. Variant2 under v2/ reads
only12 validated native tensor ranges (89,224,192B) into final MLX owners with
F_NOCACHE/preadv, bounded8MiB views and source identity checks. Same8GiB bound.
Receipt:docs/deepseek-v41/receipts/tail-seed-20260917/README.md.
Variant2 completed: native16K max allocator peak2,279,214,764B versus tail128
142,877,484B; seed medians0.0711085s/0.0017979s. Final main row and all offsets
match, but all three window byte sequences differ. Do not install as an exact
replacement. Guard38266 terminal0, cleanup8 active bytes and74,366,976B file
cache invalidated to0; exactQwen restored/released02:37:42UTC. An intervening
foreign Bonsai guard51816/51826 owned the lane; no signal or shared-code edit
occurred. After that job completed,02:42:32 check found Qwen healthy/idle/warmed
and lock free. Next candidate v3/ retains2048rows to screen batch arithmetic
while keeping most of the saving. It is staged only; no full model is ready.
FixedQ8DraftCache._seed currently resets offset from supplied row count, so an
eventual partial seed must explicitly preserve its absolute start too. Do not
apply the native offset adjustment blindly to Q8 caches.
Variant3 completed at source8b4b44c622bf69ba189c430022c3c884f4cd548e. Retaining
2048rows matches every final-row/window byte and all offsets in two interleaved
controls. Peak416,511,276B vs2,279,214,764B saves1,862,703,488B; median seed
0.0086178s vs0.0715489s. This is operator evidence only, not more slots or TPS.
Guard36042 terminal0;8 active bytes after cleanup;74,366,976B file cache to0;
exactQwen restored/warmed/released02:44:16UTC; independent02:50:06 healthy,
idle,warmed,free check found no child. Archive:tail-seed-20260917/tail2048.

Integration staged in /tmp/dsv41-tail-seed-20260917/integration. The isolated
tail-prefill body changes only target hidden captures and their concatenation.
Installer binds an explicit one-request16K native-KV backbone type and seeds
fresh native caches at14336; no model/bound-method ownership cycle. It must
not be applied to Q8 yet. Small actual eight-layer capture check is staged:
FP32/BF16, two chunk-crossing tails, exact logits/hiddens/cache/next step.
No full model staged yet. Keep old full-model memory allowances plus metadata
until fresh full evidence supports a discount;20TPS remains unmet.

Capture check v2 completes all four FP32/BF16 cases with exact logits, retained
hiddens, cache state/offsets and next decode step. First attempt was a NumPy
BF16 conversion error in the harness; v2 compares byte views. Peak1,591,702B;
28 active bytes after close. Guard32403 terminal0; exactQwen restored/warmed
and lock released02:59:14UTC; independent03:00:51 healthy/idle/warmed/free,
no owned child. Both attempts are archived under tail-seed-20260917.
Next:/tmp/dsv41-tail-seed-20260917/full, one nativeD5/M6 full16K/1K packed-plane
candidate with tail2048 capture and absolute seed offset14336. Keep all old
admission allowances plus16MiB host metadata in every phase. No operator-derived
capacity discount; native capacity search may admit85..100 instead of96..100
without relaxing any inequality. The full output digest remains mandatory.

Full tail2048 candidate completed at source dcbec19dc:12.4289961TPS/82.3075322s,
1024 exact native IDs,206cycles,84->101slots; baseline10,603,659,264B and bound
109,849,188,584B. Peak allocator93,336,409,632B, process95,127,698,216B,
machine106,151,673,856B. Prefill peak saves591,643,308B and boundary active
saves1,536,180,224B versus the best packed-plane run. After normalizing the
one-slot difference (707,788,800B), final decode active bytes are identical;
overall peak difference is only89,924B. No steady capacity discount or TPS win.
Guard17051 terminal0; Qwen restored/warmed/released03:16:02UTC; independent
03:16:54 healthy/idle/warmed/free, no child. Full-host-refusal archive retains
the earlier preallocation mismatch; v2 fixes CLI host reserve to2.015625GiB
and checks the actual CPU runtime resolver against admission with MLX blocked.
Full receipts:tail-seed-20260917/{full-host-refusal,full-tail2048,full-summary.json}.
Next small I/O screen compares native separate gate/up planes with a combined
scatter read across their368640-byte scale gap; down and slot readiness stay
separate. CPU destination buffers only, no model. Existing~13.1GB/s uncached
receipts limit the possible gain; do not launch a full candidate without a
clear end-to-end read-batch improvement. Keep the best12.6731624TPS baseline.

Combined-GU read screen completed and rejected at source db5f9dd0d. Native
PlanePart/bind_reader extracted unchanged; candidate scatters across the gate
scale gap, reading368640extra bytes/record into preallocated scratch. CPU-only,
MLX blocked,2GiB allowance,128batches perarm at1/3/6records. Exact final weights
and full native record digests; end-to-end wall changes+0.06%,+3.37%,+2.11%.
First case is within control variance; relevant batches regress. No full run,
runtime installation or added tests. Initial harness counter-reset failure is
preserved; v2 rebinds the reader after replacing metrics outside timing.
Guard50127 terminal0, source cache0 after child exit, process-tree peak
372,606,368B, machine9,923,100,672B; exactQwen restored/warmed/released03:30:33UTC.
Independent03:31:18 check healthy/idle/warmed/free, no child. Archive:
docs/deepseek-v41/receipts/gu-combined-read-20260917. Do not repeat unchanged.
No full candidate is currently staged or running. Best full remains12.6731624TPS;
20TPS is open. The full tail2048 win concerns prefill memory only, with no steady
capacity discount. Next work needs materially less expert traffic or better
overlap of verification with reads; the existing I/O path already reaches its
observed~13GB/s payload rate. Keep Q8/256K work secondary and its caveats intact.

# Current Projection Ownership Stage

Previous turn made progress: full tail2048 captures save prefill memory while
steady storage remains unchanged; combined GU reads are rejected. Current
CPU reanalysis of12 archived samples gives only5.9408percent ideal weight-only
order0 byte savings now that scales are resident; serial decode would need
220.17GB/s before framing/dispatch to break even at13.08GB/s read throughput.
This is not a universal compression bound. No codec or GPU run follows it.
Receipt:docs/deepseek-v41/receipts/weight-only-compression-bound-20260917.

Staged:/tmp/dsv41-woa-owner-20260917. All40 target fused output paths currently
retain34,603,008bytes/layer of original MXFP8 wo_a plus67,108,864bytes/layer
of BF16 transpose. Native fused engagement is8240/8240 calls in the best run.
A construction-validated first-use callable invokes the original cache builder,
then installs a BF16-only callable with identical operations and retires the
packed holder/cache tuple. Lazy materialization order is preserved; no new
steady eligibility checks, counters, parent-owner cycles or stock fallback.
One real native layer0 output-projection screen loads only4 tensors/77,856,768B
from the174,962,526B shard3 through bounded uncached ranges. Native M1/M6/M8
outputs and exact physical release are checked. Bound4GiB (2GiB MLX plus2GiB
host/cache/compiler); controller reclaims shard3 after actual child exit.
No full model staged yet; do not subtract1.384GB from every phase before its
ownership/transition envelope is established. Keep20TPS open.

Operator v2 completed at source72b0d219e. Exact M1/M6/M8 output bytes, including
cold first use. Warm active145,952,648->111,349,640B releases exactly34,603,008B;
40-layer static total1,384,120,320B. Both cold peaks213,061,512B: source remains
live during materialization, so charge one layer's34,603,008B overlap beyond
the native cold-to-warm envelope. Eight active bytes after close. Guard10513
terminal0; source cache66,387,968->0B after child exit; process-tree403,309,816B,
machine10,487,103,488B. ExactQwen restored/warmed/released03:55:19UTC; independent
03:59:04 healthy/idle/warmed/free, no child. First flat-position harness error
and success archived under receipts/woa-owner-20260917.
Next:/tmp/dsv41-woa-owner-20260917/full. Clone the best native packed-plane
candidate, preserve original prefill/copy bounds, and discount only steady
projection storage by39*34,603,008B after pricing the one-layer cold overlap.
Add16MiB host metadata consistently to CLI/admission. Keep resident plan reserve
conservative, report actual module retirement separately, and retain strict
full1024-ID digest. Do not add tail capture or a transition credit to this lane.

# Latest Adaptive Draft Stage

20 TPS remains open. Adaptive full v3 finished at12.3057465TPS/83.1318930s;
the best remains12.6731624TPS. All1024 native output IDs match. The run used
84->99slots versus the earlier best's84->102, so the policy effect is not
isolated. No throughput promotion or extra tests for this candidate.
New draft-only policy: startD5, thenD7 after full acceptance orD3 otherwise.
Exact teacher replay reduces206->195 cycles; verify rows1236->1242. D5 control
reproduces all native commit boundaries. A three-way3/5/7 policy also takes195
cycles but1282 rows, so retain the two-way policy for the full screen.
Incremental bound52,613,349,376B; allocator peak21,299,586,448B; guard process
15,376,608,560B and physical31,218,286,592B. Guard26808 is terminal exit0;
Qwen restored23:49:22UTC, final independent health/warmup/free-lock check passes.

Full staging: /tmp/dsv41-adaptive-depth-20260917/full, parameter-sharing views
in adaptive_lane.py. Full-target execution now completed. The existing nativeM8 bound
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
The user confirmed the OS file cache was purged. Source/helper hashes were
repinned and fullv3 ran with optional purge OFF. Its fresh baseline11.518GB
admitted99slots and a109,348,083,944B bound. Allocator peak91,931,939,555B;
internal process93,697,472,216B; internal machine108,799,164,416B; guard machine
108,678,807,552B. Whole-machine samples stayed below110GB. Read volume652.546GB,
36,878records;196cycles; verification77.3694s; drafting2.0935s. Installation
3.2996s is charged to decode. Same native-AR index297 tie classification.
Guard parent27410/shell27429/child27599 are terminal; child exit0, exactQwen
restored with warmup and lock released00:56:30UTC. Independent checks pass.
Full harness/result/lifecycle:receipt full-v3 directory,30 hashed files.

The next continuation revalidated the same RAM/admin-auth blocker. Read-only
mincore audit found0 resident bytes across68,403 saved-session blobs (10.425GB)
and289 task-owned artifact files (13.576GB). A2MiB private-file positive control
returned2MiB before and0 after invalidation, so no session/artifact cleanup is
justified. Qwen remained healthy/idle/warmed; machine use133.107GB while serving,
including37.497GB file-backed. sudo -n -v still requires a password. No GPU run
or service restart occurred. A third consecutive goal turn revalidated the same
condition:37.528GB file-backed RAM while Qwen was healthy/idle/warmed, no owned
candidate, and sudo -n -v still requiring a password. The goal was marked
blocked, then resumed after external reclamation. That blocker is cleared;
fullv3 above is complete and20TPS remains open. Do not repeat the blocked
status or relaunchv3. The negative RAM audit remains useful evidence against
unnecessary saved-session or task-artifact cleanup.

Receipt:docs/deepseek-v41/receipts/adaptive-draft-20260917/README.md.
CPU ARC/S3-FIFO screens lost (heldout17411/17193 vs15544 demand misses), so no
GPU tests followed. Raw CPU screens:/tmp/dsv41-cache-replacement-20260917.

Completed candidate:/tmp/dsv41-prefill-allocation-20260917/full. Prefill-only marginal
frequency allocation, fixed total slots, min84/max128 per layer, selected once
at the existing timed transition. CPU demand misses at average99 improve
36421->35631 (2.17%); second half16187->15636 (3.41%). No decode rows select the
vector; identical73-slot seed plus empty slots across CPU arms. This is not
physical-read or TPS evidence. Admission charges the max128 component copy and
16MiB metadata before model load. Layer vectors must drive plans, policy,
physical owners and reports. Full run:12.5526155TPS/81.4969600s;84->4000total
slots, actual layer capacities84..128, uniform-equivalent100. Same1024IDs and
206cycles; same native-AR index297 tie. Reads35097/621.032GB vs best35092/620.943GB
at4080slots; capacity/background differ, so no isolated throughput promotion.
Baseline11,050,860,544B; bound109,588,601,064B; internal machine106,146,136,064B.
Guard28436 terminal exit0, exactQwen restored/warmed and lock released01:19:26UTC.
Independent health/free-lock checks pass. Qwen model residency scan found36.05GB
of cache covered by automatic shutdown reclamation; no new manual purge needed.
Receipt:prefill-allocation-20260917/README.md. No further tests for this result.

Bounded operator:/tmp/dsv41-row-pairing-20260917. Exact native M6 routing
census has296640 assignments, average24.2443 distinct experts/layer/cycle;
139196 assignments (46.9242%) can pair with the same expert. Sharing FP4
conversion across pairs could remove23.4621% of repeated weight decodes before
overhead. Staged float2 dot retains native V16/V8, R4, 2-SIMD geometry and each
row's accumulation order. CPU grouping routes remaining singles to the native
operator. Three-expert MLP is bit-exact in all4 shapes: rows4 mixed loses14.15%,
rows6 paired gains3.57%, rows9 mixed loses4.08%, rows18 paired gains21.40%.
Allocator peak131,863,420B;8B afterclose. Guard9965 terminal exit0; Qwen restored
healthy/warmed and lock released01:30:45UTC, independently verified. No full-model
gain. Receipt:row-pairing-20260917/README.md. One-layer integration also completed:
native vsM6-hit-pair lane,32persistent/48transient slots,4interleaved blocks,
26experts/36assignments. All64 outputs bit-exact;134reads/arm. M6 median18.669354
->18.527844ms (0.758%), below0.972% spread between native controls. M1 unchanged
operator measures0.848% slower. No full-model run justified; do not promote the
isolated21.40% case. Guard8711 terminal exit0; allocator1,581,355,529B,8B afterclose;
guard machine12,985,729,024B. ExactQwen restored/healthy/warmed and lock released
01:37:42UTC; independent checks found no owned candidate. Both new full candidates
and both operator windows are terminal. Automatic Qwen cleanup is working;
the earlier manual-purge blocker is cleared. Best remains12.6731624TPS.

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
