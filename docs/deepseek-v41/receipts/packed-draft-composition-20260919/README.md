# Packed draft and earlier target projection priming: full admission refusal

The integration is staged and passes CPU construction and phase-budget checks,
but its full run is refused before model loading. At the fresh background of
12,160,188,416 bytes, the required minimum of 111 expert rows cannot fit under
the unchanged 110,000,000,000-byte whole-machine ceiling. No prefill, decode,
full output or new TPS result exists. The best complete Q4 result remains
13.8688167379 TPS; 20 TPS is still open.

## Scheduled allocation and ownership

The candidate preserves the native 93/58/32 draft experts, native KV16, target
arithmetic and D5 plus two causal lookup tokens. It installs the measured
packed FP32 draft output projection once at construction. Native seed returns
before this projection; no 2,048-row seed input reaches the new operation.

The target retains native packed output weights and its exact scheduled BF16
expansion. The first projection is primed after native seeding at 84 rows,
before the independent expert extension is allocated. This moves the cold
three-buffer allowance to the smaller phase; extension retains one initialized
67,108,864-byte projection, while steady replacement still prices three. All
priming and growth remain inside decode wall time. No target kernel changes.

The 402,653,184-byte draft cache credit applies only to steady decode. Prefill,
seed and extension receive zero draft credit. Add 128 MiB GPU workspace,
16 MiB host state and 256 MiB background variation, all inside 110 GB. Process
host reserve is 1,455,550,464 bytes; including background it is 1,723,985,920.
The 100 GiB wired ceiling remains unchanged. See the [design](../../../specs/2026-09-19-deepseek-v41-packed-draft-composition.md).

CPU-only preflight blocks MLX imports, verifies 30 helper pins, compiles source,
checks unchanged target operators and native seed control flow, and verifies
CLI/admission agreement for five background cases. This is static evidence,
not full GPU validation. The pinned packed Metal backend and temporary
inventory accompany the installation.

## Refusal and lifecycle

After waiting for prior jobs, the canonical guard acquires the lane, stops
Qwen and reclaims source pages. Live admission fails the minimum-111-row gate
at `run_full.py:387`. No live bounds file or result is emitted. A separate CPU
reconstruction removes the wired constraint entirely and still admits at most
109 rows at the live background. The 111-row physical estimate is
111,073,252,600 bytes, exceeding the ceiling by 1,073,252,600 bytes. This
counterfactual does not claim to recover the unrecorded live wired snapshot.

Guard and child exit 1. The guard restores exact Qwen identity, health and
completed warmup and releases at 17:04:14 UTC. The independent check observes
a subsequent foreign window and no owned process. This is distinct from the
completed restoration. No unrelated job is signaled, and there is no unchanged
full retry or new regression test.

Source is `b5f41c51d74a770069048dddd0a644f07886a999`.
`archive-sha256.json` pins the original staging, CPU proof, refusal, command and
lifecycle evidence. Large immutable artifacts remain at their recorded paths.
