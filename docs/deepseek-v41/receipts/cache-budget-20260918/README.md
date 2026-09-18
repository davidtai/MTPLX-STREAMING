# Phase-specific allocator cache and bounded Engram capacity

The full candidate completes the exact 16,384-input / 1,024-output Python
workload at **12.6712544 TPS / 80.7339166 seconds**, including charged cache
growth. All 1,024 native output IDs match, with 206 D5/M6 cycles. **20 TPS remains
unmet.** This is a memory-capacity result, not a demonstrated throughput win.

The retained part3 memory-composition result is 12.8091055 TPS at 104 slots per
layer. This candidate uses 103 slots at a larger background baseline, so their
1.08% TPS difference does not isolate the allocator-cache change. Do not promote
a speed claim or repeat either arm unchanged from this comparison.

## Actual budget change

The candidate keeps the 1 GiB allocator cache through native prefill, then sets
256 MiB at the existing quiescent post-prefill transition before its existing
cache clear. The applied-limit receipt changes with that phase. No per-token
cache setter, extra cache clear, fallback, or instrumentation is added.

Both Engram arenas have a fixed 64 MiB capacity: 254,200 rows each. The complete
run retains 180,226 and 180,228 rows with zero evictions. Pricing includes their
full payload capacity, 256 bytes of metadata per possible row, 1 GiB for other
Python allocations, and 32 MiB for helpers.

| Budget component | Retained configuration | Candidate |
| --- | ---: | ---: |
| Prefill allocator-cache setting | 1,073,741,824 B | 1,073,741,824 B |
| Growth/decode allocator-cache setting | 1,073,741,824 B | 268,435,456 B |
| Additional cache-overshoot allowance | 2,258,155,644 B | 2,258,155,644 B |
| Python plus helper reserve | 1,578,202,112 B | 1,371,664,384 B |

The configured budget saving is **1,011,844,096 bytes**. At the actual
11,964,268,544-byte baseline and 3,414,097,920-byte wired snapshot, unchanged
admission implementations calculate 102 slots for the retained configuration
and 103 for this candidate. See `full/same-baseline-admission.json`; this is a
CPU allocation calculation, not another measured full run. Earlier CPU cases
show 103–106 slots as the background baseline varies. The ceiling remains
**110,000,000,000 decimal bytes** with the existing prefill, KV, graph, compiler,
copy, allocation-margin and wired allowances.

The matching [MLX 0.32.2 allocator source](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/allocator.cpp)
checks current cache size before recycling an entire freed buffer. A positive
cache setting can therefore be exceeded. Setting a limit does not clear the
cache; allocation subsequently trims excess cached buffers. Active peak also
excludes cached buffers. Accordingly, this candidate preserves the original
overshoot allowance; it does not treat 256 MiB as a hard total-cache bound.
`allocator-audit.json` records installed package metadata, header hashes and the
scope of the upstream-source inspection, without claiming binary identity.

## Bounded operator decisions

Three guarded operators precede the full candidate. Their exact helpers,
commands, raw timing/parity receipts, reclamation and service recovery records
are preserved in the corresponding subdirectories.

- Native layer34 expert replay: disabling the allocator cache is flat
  (candidate/control 0.9992, control spread 0.90%). All routes, outputs and
  physical reads match. Its 9 GiB incremental bound covers a 2,934,477,321-byte
  MLX peak; the final active allocation is 8 bytes.
- Native attention with cache disabled: outputs, persistent state and selected
  indices match, but Full/Reuse and Full/Reindex attention slow substantially.
  Controls contain outliers, so the raw 1.74x/1.54x ratios are not precise
  general estimates. The consistent candidate slowdown rejects zero cache
  before any full-model zero-cache run.
- Native attention with 256 MiB cache: outputs and complete state match. The
  stable Full/Reindex case is flat (0.9973 ratio, 0.40% control spread).
  Full/Reuse has a noisy control block; SWA also does not prove a robust win.
  Each attention operator has an 8 GiB incremental bound, peaks at
  1,155,786,464 MLX bytes, and releases to 8 active bytes.

No broad suite or additional optimization test was added. The full candidate
itself verifies the applied phase limit, exact output digest, completed growth,
current slot storage, Engram capacity and retained admission allowances.

## Full-run memory and recovery

| Measure | Bytes |
| --- | ---: |
| Whole-machine admission bound | 109,677,955,304 |
| MLX active bound | 93,815,431,276 |
| Measured MLX allocator peak | 93,368,091,658 |
| Internal sampled process footprint peak | 94,359,404,456 |
| Internal sampled machine physical peak | 107,025,219,584 |
| Independent sampled process-tree footprint peak | 94,346,395,560 |
| Independent sampled machine physical peak | 107,034,165,248 |
| End-of-decode allocator free cache | 250,855,410 |

These measures overlap and must not be added. The 223 independent guard samples
show zero compressor growth. Decode reads 34,680 records / 613,652,889,600 bytes;
the union of read intervals is 47.4181955 seconds. Charged growth is 3.6700298
seconds. Existing accepted AR tie classification at output297 is unchanged.

Measured source is `cb85fb3d518e3a99a8b1e36b60afb9048f36e095`. Guard session12198
returns exit0. Exact Qwen identity, health and background warmup are verified
before lock release at 06:15:55 UTC on September18. Independent verification at
06:16:26 UTC confirms idle, warmed, healthy Qwen, a free lock and no owned child.
The packed file cache ends at zero. All three earlier operator guards likewise
returned exit0 and restored Qwen. No unrelated job was terminated.

The archive includes raw JSONL, OS samples, bounds, output sidecars and SHA-256
coverage. Commands and helpers preserve their measured absolute paths and source
pins; they are historical evidence, not commands to rerun after changing HEAD.
Large model and packed-scale payloads are represented by their existing
manifests rather than copied into this receipt.

Native KV16 remains the fastest retained route. Fixed Q8 and 256K backing
geometry are separate; this run does not establish a full 256K prefill bound.
