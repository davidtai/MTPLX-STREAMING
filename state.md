# Current Goal

DeepSeek V4.1: correct memory reporting and runner bugs, then reach 20 decode
TPS on the exact 16,384-input / 1,024-output Python workload under 110 decimal GB.
**Best measured: 11.7203483 TPS. The 20 TPS goal remains open.**

# Decisions

- Keep allocator, process phys_footprint, and whole-machine physical memory
  separate. Never add process usage to machine usage or substitute RSS.
- Reserve 2 GiB for Python/host caches and metadata. The measured candidate uses
  a 1 GiB allocator cache; current machine baseline determines the Metal limit.
- Hold `/tmp/mtplx-gpu-exclusive.lock` before MLX import/execution or stopping
  Qwen. Use `scripts/deepseek_v41/gpu_window.sh` directly; never nest guards.
  Reclaim stopped Qwen caches automatically, restore the exact service, verify
  health/warmup, and release the lock last. Never terminate other GPU jobs.
- User permits tie breakers only. Preserve full token digests and require an
  index-matched classification against fresh candidate logits.
- Keep the unvalidated bounded-KV lane out of full-model execution. Preserve
  Claude's separate W126/W127/W128 worktrees. Use minimal testing and add
  optimization tests only after measured wins.

# Plan Status

Executing `docs/plans/2026-09-16-deepseek-v41-20tps-stage.md`, Task 4.
Promoted lifetime fix: 9c4fac40f. Reporting fixes: c031119a1 records effective
and requested draft depth; b95f8d1a7 preserves unknown MLX peaks as null and
derives headline values from the same memory-block reading. Earlier memory
and guard fixes remain at e589c1e4b and c9f090573. The target is unmet.

The winning full run uses depth 5, 94 persistent slots/layer, 48 shared
transients, full six-row verification, transition-window admission,
three-record miss parts, shared overlap, fanout 4, pf0, compact MTP banks
(93,58,32), native BF16 target head, and max_kv 17664. The wrapper's admission
is specific to the pinned native artifact and exact 16K workload.

A captured mean is evaluated at the existing chunk fence. This frees earlier
Hyper-Connection graphs without another fence or changed arithmetic. The
slot-normalized allocator peak saving is 3,011,286,868 bytes. Four additional
expert slots per layer consume 3,008,102,400 bytes of that saving.

# Evidence

Receipts: `docs/deepseek-v41/receipts/hidden-capture-110gb-20260917/README.md`.
The archived source-pinned wrappers contain each measured installation.

- Full depth 5/cap 94: 87.2840954s decode, 11.7203483 TPS, 206 cycles,
  38,613 expert reads / 725,949,112,320 bytes; active I/O window 55.6445380s.
  Versus full depth 3/cap 94: 91.6563610s / 11.1612548 TPS. Earlier cap 80 full
  result was 9.6589676 TPS; the combined improvement is 21.34%.
- Depth5 MLX peak 97,146,257,816B; sampled process footprint 98,103,862,608B;
  maximum sampled whole-machine usage 107,453,988,864B. Admission projected
  108,645,754,112B including baseline drift, within 110,000,000,000B.
- Baseline 8,835,645,440B; host reserve 2,147,483,648B;
  allocator limit 99,016,870,912B; engine budget 90,946,870,088B.
  Retained cache and wired/graph headroom are separately checked.
- Full MTP digest remains
  `0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
  AR digest remains
  `2bd0ad017b9580c8fec340e297696a0bd81a7759b6c5dfe7c5d64de6d40c1090`.
  First difference is the accepted tie_flip at 297; AR contested margin 0.0.
- Twenty CPU reporting checks passed after b95f8d1a7, with MLX imports blocked.
  The missing-counter regression rejects the previous implementation. Existing
  guarded tiny prefill checks validate the earlier promoted lifetime change.
- On 2026-09-17 at 12:35 UTC the cap94 attempt refused its 11.962 GB live
  baseline before model loading. Qwen returned healthy/warm with its exact ID
  and the lock was released at 12:36:04. The cap90 retry timed out queued at
  12:48:12, with no child launch. At12:56:51 Qwen was healthy/warm with exact
  model ID; another benchmark still owned the lane.
- Reused AR timing, memory and counters are null. Provenance and the cached
  diagnostic row are hashed; candidate MTP logits remain fresh.

# Open Issues

- 20 TPS requires at most 51.15s decode; current active expert I/O alone is 55.64s.
  Reduce I/O and remaining verification cost before claiming the target.
- Reject staged 3+3 (slower), early projection reclaim (no gain), and the prior
  chunk256 arm (17 MB saving and changed output). The old 3+3-dependent cap 93-97
  ladder is superseded; its eighteen-transient bounds do not price full M4/M6.
- Full-M6 replay rejects tuned admission; the 35-policy sweep favors the current
  policy. Wider nonuniform allocation projects only 3% fewer reads.
- MTP seed setup does not raise the observed peak; seed truncation is not
  supported as the next peak-memory optimization. The remaining peak precedes it.
- `mlx_active_*_at_decode_start` in receipts before e589c1e4b is mislabeled;
  other peak and decode-end fields are unaffected. Reused-reference wrappers
  are tied to their recorded base source, not portable general-runner defaults.
- Retiring consumed inputs alone preserves the full digest but gives only
  11.7617 TPS and 15 MB lower peak; no promotion. Per-chunk combine evaluation
  alone raises the prefill peak to 97,208,261,708B. Their combination is unmeasured.
- The unchanged prefill peak is 97,146,256,508B at layer39's combine fence.
  This includes queued MoE reorder/reduction work; it does not isolate HC alone.
- Serial rANS decoding costs at least 1.369 ms per record, versus under 0.1 ms
  saved I/O: reject. Native-shaped HC chain compilation changes results: keep off.
- Decode-only growth to cap104 projects 34,211 reads versus 38,613 at cap94,
  but resize ownership and full decode peak remain unbounded; do not install it.
- Current-source profile wrappers still pin c031119a1 and are obsolete after
  b95f8d1a7. Regenerate the exact source compatibility proof before retrying.
- `/tmp/dsv41-110-stage/CONTINUATION.md` holds detailed negative receipts and
  job status. HC-post session81450 also timed out queued at 12:55:04 UTC;
  no owned guard or child remains. Its prepared 512MiB probe is unmeasured.
