# Current Goal

Q4 optimization is shelved at the user's request on 2026-09-18. Preserve all
DeepSeek V4.1 workspaces and open a draft PR on davidtai/MTPLX-STREAMING, then
attempt the Q2 main model separately. The eventual target remains 20 decode TPS
on 16,384 input / 1,024 output Python tokens under 110 decimal GB.
Best single full result: **13.4141517619 TPS / 76.2627423750 s; 20 TPS unmet.**
All 1,024 native IDs match; 1,023 timed steps, 198 target calls, 84->110 slots.
The result is not a matched/repeated speedup claim. Fixed Q8 and complete 256K
prefill verification remain secondary; the full Q8 candidate is 10.7541904 TPS.

# Decisions

- Separate MLX allocator, Darwin phys_footprint and whole-machine physical use.
  Never substitute RSS, sum overlapping metrics, or replace unknown with zero.
- Ceiling is 110,000,000,000 B including live baseline, Python, Metal, KV,
  inactive cache, I/O, compile/copy and phase peaks. Preserve 100 GiB wired limit.
  Latest full host reserve is 1,421,996,032 B. Only the attested strict allocator
  may remove its proved 2,258,155,644 B overshoot allowance; stock keeps it.
- Every MLX import/load/compile/run requires the parent-held exclusive lock
  /tmp/mtplx-gpu-exclusive.lock. Use scripts/deepseek_v41/gpu_window.sh directly,
  never nested. Wait for other jobs; never steal their lane or terminate them.
- Acquire before idle/warm Qwen shutdown; reclaim its clean model pages
  automatically. Restore exact service identity and warmup before release.
  After terminal child/guard status independently verify health and lock.
- Work inline, no agents. Minimal checks; add optimization tests only after
  a measured win. Tie breakers are permitted; unclassified target drift is not.
- Validate/install invariant routes once. No enabled-lane silent fallbacks,
  repeated hot-path metadata checks, environment reads or engagement counters.
- Preserve Claude W126/W127/W128 worktrees. New fixed Q8 is separate from the
  old disabled bounded-KV lane. A 256K allocation is not a verified 256K prefill.
- User explicitly requests Q2 main-model work after the Q4 checkpoint PR.
  Draft-only Q2 screens do not establish Q2 target output quality or throughput.

# Plan Status

Shelved docs/plans/2026-09-16-deepseek-v41-20tps-stage.md; Task 4 is incomplete.
Memory/reporting corrections and automatic shutdown reclamation are retained.
Current stage adds diagnostic receipts only; production defaults are unchanged.
Workspace inventory: docs/deepseek-v41/checkpoints/20260918-worktrees.json.
All 140 worktrees and seven dirty snapshots are preserved under checkpoint refs;
original indexes and working files remain intact. Ignored artifacts stay in place.

# Evidence

- Full winner: receipts/hybrid-lookup-20260918 under docs/deepseek-v41. Native
  D5 plus at most two past-text lookup tokens, strict allocator and exact input
  row cache. Decode source48de2aaac8c13c5d31cfbeb8ee5bc0f92f78f9b3, native KV16.
  Root /tmp/dsv41-hybrid-lookup-20260918/full-v1. Baseline10,447,192,064 B;
  bound109,631,928,540 B; machine peak109,238,927,360 B. Expert reads31,961
  records /565,540,945,920 B. Output SHA256:
  0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.
- Earlier screens measured source61b6899f27fee89f63bf7eed9f1c8aa77e73a53a.
  Reader pool reduction and R2 GU fusion fail bounded performance selection.
  Completed-layer mean improves router recall only slightly with no early lead.
  Receipts: packed-reader-pool-20260918, fused-gu-r2-20260918,
  completed-input-feature-20260918. Do not rerun unchanged.
- Conditioned suffix with confidence filter takes193 versus198 hybrid calls
  in teacher/head replay but adds~0.48 s head work. Conditional, not a full
  parity/TPS result. Longer/raw suffixes are rejected. Receipt:
  draft-conditioned-tail-20260918; keep original target verification.
- Compact prompt-trained score residual transfers to saved exact M6 capture:
  1,062/4,878 useful physical misses at84.2189% precision versus763 at85.0613%
  for direct control. Unlimited-lead-time quality only; no measured overlap.
  Adapter21,399,552 B. Capture105 slots differs from full winner110.
  Current-route features add too little and are not selected. Receipt:
  prompt-router-adapter-20260918. Exact holdout reused during research.
- Full-hidden raw-residual affine model loses its5.593 s CPU screen:
  top-six75.0778% versus75.1736% compact; proxy miss coverage20.7820% versus
  21.0863%, precision86.0202% versus87.0056%. It would need283,170,816 B.
  Reject this variant; no full run/tests. Receipt: full-hidden-router-20260918.
  Static CPU bound1 GiB; sampled process333,365,968 B, machine12,367,265,792 B,
  six samples, zero compressor growth. Endpoint footprint333,431,504 B.
- Q2 draft screens at source4f3af25aeed6a37c7297db50d8987f214cc2ee37 are in
  receipts/q2-draft-20260918. Affine2/64 compact experts save1,416,683,520 B
  payload but need201 versus198 calls and22.83% longer warmed head execution.
  Adding selected dense matrices saves1,760,354,304 B total but needs233 calls;
  reject that variant. Expert-only remains conditional; no full run or tests.
  Keep KV-building projections native: changing them invalidates saved KV.
- Capacity sensitivity exactly replays historical53,999 reads before comparing
  current110/112-slot policy on fixed native routes:2.3694% less traffic. It
  does not predict Q2 routes, current full performance or complete admission.
- Both49GiB-bound screens exit0, reclaim15,342,174,208 source-cache bytes each.
  Final guard48370 releases15:58:15UTC; exactQwen/healthy/idle/warmed/free-lock
  check15:59:42UTC passes. No owned child/queued window remains. Check live state.

# Open Issues and Next Work

- Q4 needs 25.1127 s less decode time to reach 20 TPS and is now shelved.
  Start Q2 main-model work in a separate worktree after publishing the PR.
  The old affine Q2 target failed output quality (W9_REPORT.md); compare the
  linked Vontra recipe with that evidence before building or downloading it.
  No fresh Q2 target generation or output sanity check has run in this stage.
- Preserve arithmetic/layout/ownership while reducing expert I/O or verified
  target work. Do not repeat rejected cache-policy, fanout8, GU-gap reads,
  HC, D7/D9/D13, prefix1+5, alignment or unchanged small-kernel candidates.
- Conditional80/40/24 draft saves733,224,960 B but111-slot full admission
  refused at11,137,220,608 B background. Do not retry without a fitting bound.
- Traces remain in .benchmark-artifacts/deepseek-v41; teacher replay remains in
  /tmp/dsv41-depth-replay-20260917. Preserve them for resuming the saved Q4 work.
- Strict library: /private/tmp/dsv41-strict-cache-20260918/strict-lib/libmlx.dylib.
  SHA25632f8c0e361d6f35251c9e05aeba05563f94ae54cc1f5f8e4bcb5ec9e6c42fba9.
  Production packages unchanged. Refresh live helper source pins after commits;
  never rewrite measured receipts. Native AR logits cover297, not376 or480.
