# Current Goal

Resume DeepSeek V4.1 in MTPLX-STREAMING: correct memory reporting first, then
runner bugs, under **110 decimal GB including the machine baseline**. The active
performance goal is **20 decode TPS on the Python workload with 16K prefill**.
The pinned task requests at most 1,024 output tokens; the benchmark uses 16,384
input tokens and 1,023 decode steps plus the first token emitted by prefill.

# Decisions

- Never execute MLX without the parent-held `/tmp/mtplx-gpu-exclusive.lock`.
  Use the guarded runner, restore the exact service, verify health/warmup and
  lock release. A free lock alone does not prove memory headroom.
- Budget Python capacity at 2 GiB, including two 256 MiB Engram arenas and
  metadata. Metal gets the remainder after the measured system baseline.
  Its 10 GiB transient reserve includes a 2 GiB retained allocator cache.
- Expert, Engram and resident file reads use uncached descriptors on macOS.
  Physical used memory includes active/inactive file cache; speculative pages
  are free. Report bytes, decimal GB and binary GiB separately.
- Current slot layouts retain full backing arrays. Reserve configured maximum
  KV before allocating expert slots; logical eviction cannot fund physical KV.
- Keep the unvalidated bounded-KV lane out of full-model execution. Its
  underlying Metal numerical cause is unresolved; isolated cache tests remain
  available. No silent fallback and no new hot-path validation.
- Preserve Claude's separate dirty W126/W127/W128 worktrees. W126 is unpromoted
  and has no full GPU performance proof; its async lane is inert with pf0.

# Plan Status

Continuing the runner program in `docs/deepseek-v41/W95_RUNNER_DESIGN.md`.
Memory reporting and the revised default allocation have full-model evidence.
The current batch fixes fixed-slot KV admission, cross-request frequency state,
mixed-bank zero-ring pricing, and benchmark parity/plan comparison reporting.
CPU regressions and the combined guarded integration bundle (313 tests) pass.
Independent construction-guard review passed; no remaining scoped findings.
Device-fixture review added guaranteed restoration even if synchronization
raises, verified with a fake device and no MLX import.

# Evidence

- Integration branch: `feat/deepseek-v41-streaming`; current committed head
  `f743d1bdc` produced the corrected-run diagnostic. Source `1fbe425ca`
  produced the clean default-budget run; see git log for later receipt commits.
- `docs/deepseek-v41/receipts/memory-budget-110/` contains raw receipts, OS
  traces, guard logs, target plans and summaries. JSONL files require explicit
  `git add -f`; do not edit historical raw observations.
- Default run: 35 expert slots/layer, **4.201764739 TPS**, 135.005 s prefill,
  243.469 s decode, **106,349,838,336 B** sampled physical peak (250 ms),
  **56,845,381,092 B** true MLX active peak. No added swapouts.
- All 1,024 output IDs match the 27-slot control:
  `2bd0ad017b9580c8fec340e297696a0bd81a7759b6c5dfe7c5d64de6d40c1090`.
  The output is a capped Python diff, not validated generated code.
- Prompt SHA:
  `38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2`.
- Prior guarded bundle: 332 tests plus 7 subtests. New fixed-storage/request
  fixes: 315 CPU tests, actual MLX imports blocked; 15 new cases independently
  rechecked. Construction/parity reporting: 32 focused regressions; combined integration
  bundle: 313 passed. An initial mixed-bank failure came from CPU device
  contamination; test-scoped Metal selection fixes it without tolerance changes.
- Corrected-run diagnostic: 106,163,322,880 B sampled physical peak,
  56,845,378,900 B MLX active peak, all 1,024 output IDs equal, no new swapouts.
  Admitted 17,407 KV tokens and verified release. Guard exited 0; exact service
  healthy with warmup done and lock free, independently verified after 07:16 UTC.
- Diagnostic launch mistakenly set unused MTPLX_EXPERT_IO_FANOUT=4; actual
  MTPLX_DSV41_IO_READ_FANOUT was unset (fanout 1). Its 3.783595 TPS is not
  comparable to clean fanout-4 performance. New reports stamp installed fanout.
  Three red/green fanout regressions plus prior reporting/cache cases: 38 CPU
  tests pass with actual MLX blocked; independent reviewer has no findings.

# Open Issues

- **20 TPS is not met.** Do not mark the active goal complete.
- Completed `run_python_16k_1024_oracle.py` capture is archived with raw routes,
  OS trace, token output, memory plan and guard log in the receipt directory.
  Offline replay must use try_plan_all_hits before plan to reproduce the exact
  88,637 misses; the direct-plan-only replay differed by four. Twelve synthetic
  warm-state checks match complete plans and final slots.
- Offline screen: best existing frequency decay 0.97 gives 84.282 misses/token
  versus 86.644 control (2.7% reduction), with the same captured warm contents.
  Not a full prefill comparison and not promoted. Clairvoyant bypass floor:
  51.740 misses/token per layer, 49.091 with a global 1,400-slot relaxation.
  At 20 TPS those still need 19.455/18.459 GB/s; observed clean I/O-window
  bandwidth is 12.657 GB/s, not a hardware ceiling. Eviction tuning alone does
  not support the target. Next investigate MTP/verify scheduling only after
  deriving its separate resident, KV, compile/graph peak bound on small shapes.
- Use MTPLX_DSV41_IO_READ_FANOUT=4 explicitly for the next controlled run.
  Keep unchanged controls and exact workload/token receipts; do not promote
  W126 or bounded KV without their missing validation.
- `scripts/deepseek_v41/analyze_route_cache.py` computes a tested clairvoyant
  per-layer lower bound with optional admission and temporary service storage.
  It is diagnostic, not a deployable policy or promotion throughput.
- Guarded integration uses the shared venv at
  `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python`.
- Artifact remains
  `/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4`.
  Do not substitute other quantizations or the separate MiaAI artifact.
