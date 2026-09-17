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
Memory-reporting and guard fixes are committed at e589c1e4b and c9f090573.
The captured-hidden lifetime fix is promoted and verified; receipt archival
and this checkpoint accompany that change. The target performance is unmet.

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

See `docs/deepseek-v41/receipts/hidden-capture-110gb-20260917/README.md` and its manifest.
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
- Eighteen CPU memory-reporting checks passed. After the successful optimization,
  both existing guarded tiny quantized prefill checks passed. Full model AST
  equality confirms promotion matches the measured patch. No new test module.
- Every completed window exited 0 and restored Qwen healthy/warm with exact ID
  `mtplx-flash-next-optimized-speed`. Last check is in
  `promotion-service-health.json`; no owned GPU child or guard remains.
- AR-reference reuse saves repeated AR generation/replay. Its public AR timing,
  memory and counters are null, provenance is explicit, and candidate MTP
  logits are always fresh. The cached AR diagnostic row is content-hashed.

# Open Issues

- 20 TPS requires at most 51.15s decode; current active expert I/O alone is 55.64s.
  Reduce I/O and remaining verification cost before claiming the target.
- Reject staged 3+3 (slower), early projection reclaim (no gain), and the prior
  chunk256 arm (17 MB saving and changed output). The old 3+3-dependent cap 93-97
  ladder is superseded; its eighteen-transient bounds do not price full M4/M6.
- Full-M6 CPU replay rejects the tuned transition policy. Five-shape nonuniform
  allocation projects only about 3% fewer reads and has no GPU speed proof.
- MTP seed setup does not raise the observed peak; seed truncation is not
  supported as the next peak-memory optimization. The remaining peak precedes it.
- `mlx_active_*_at_decode_start` in receipts before e589c1e4b is mislabeled;
  other peak and decode-end fields are unaffected. Reused-reference wrappers
  are tied to their recorded base source, not portable general-runner defaults.

Historical observations remain in dated receipts and prior state.md revisions.
