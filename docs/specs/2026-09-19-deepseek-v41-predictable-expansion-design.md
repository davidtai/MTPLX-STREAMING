# Predictable resident MXFP8 expansion during expert I/O

The retained Q4 runner owns 2,684,354,560 bytes of BF16 target output-projection
weights. SSD offload adds too much traffic. Direct packed matmul changes target
outputs beyond the accepted tie band. The existing exact fused MXFP8-to-BF16
transpose avoids that arithmetic change, but expanding weights immediately at
use adds projected latency. These are different mechanisms; none establishes
the result of scheduling the exact expansion one layer earlier.

Keep all 40 native packed output weights in unified RAM and enqueue the next
layer's exact BF16 expansion after the current layer submits expert demand
reads. Retain the native grouped BF16 matmul and packed MXFP8 wo_b projection.
This uses predictable layer order, without a learned predictor or recurring
dense SSD reads. No new quantization or target math is introduced.

First measure one bounded component comparison: resident output weights and
110 expert slots versus predictable expansion and 111 slots, in A/B/A order.
Rotate through all 40 real output projections while replaying the existing
206 M6 routes of independently selected expert layer 34. Synthetic projection
inputs and saved routes retain the previous-output and router dependency
barriers. Attention is omitted. Compare every expert and projection output
after timing, and include final GPU completion. The first four calls warm
kernels; retain both whole and warmed timings.

Packed wo_a payload is 1,384,120,320 bytes. Two live BF16 expansions require
134,217,728 bytes, saving 1,166,016,512 bytes against the retained dense cache.
Price a third BF16 array during replacement, reducing conservative credit to
1,098,907,648 bytes. This covers one 707,788,800-byte uniform packed expert slot
across 40 layers at the payload level. Full prefill/growth/seed admission is a
separate gate; this component does not establish full-model capacity or TPS.

The component bounds Metal, allocator cache and compilation at 10 GiB, host
and reader/compiler memory at 4 GiB, and all touched source-file cache at a
conservative 12 GiB. The 26 GiB incremental bound is checked against fresh
whole-machine and wired memory before MLX imports. Use the canonical guard,
110 decimal GB physical ceiling and unchanged 100 GiB wired ceiling.

The current projection is complete at the source router's covering evaluation.
The new expansion owns a distinct immutable MLX output; a retained output or
queued consumer can never race an in-place overwrite. The store holds only two
recent arrays and prices transient replacement. The next layer's matmul depends
on its queued expansion. All GPU work is synchronized before store/runtime
cleanup on success or failure. Reader threads never touch MLX.

Risks: conversion may compete with expert kernels for GPU bandwidth; deferred
graphs may keep more arrays alive than expected; a one-layer route replay may
overstate useful overlap in the full model. Retain measured memory and scope
limits. A material, exact-output win is required before a full integration or
optimization regression tests. Production defaults remain unchanged.

## Measured implementation

The initial component is 1.65% faster with exact outputs. A second native bank
resize costs a projected 1.798 s and is rejected. One separate added row per
layer avoids that copy; its component remains 1.41% faster including allocation.

The full candidate retains the original 84-row prefill and first packed growth.
It installs the projection schedule after that growth, completes native MTP
seeding, appends one independent bank row per layer, then primes layer 0.
All transitions are inside the decode timer. The packed expert runner is
construction-bound to issue the next projection after demand/resident
submission; layer 39 prepares layer 0. No runtime eligibility check is added.

Admission retains the original first-growth and seed bounds, separately prices
the appended rows, one temporary raw-scale owner, three BF16 buffers and page
padding, and adds 16 MiB host reserve in every phase. The 1,098,907,648-byte
projection credit applies to steady decode only. An unset optional expert-cache
limit remains unset in the updated plan and runtime configuration.

The complete result at 84→109→110 rows is 13.3988661661 TPS, all native IDs exact,
with 108.679 GB guard physical peak under a 109.965 GB bound. Sampled process
peak is about 1.17 GB lower than the historical 110-row best. Wall time is
0.1141% longer; no isolated speed gain or 20-TPS completion is claimed. See
`../deepseek-v41/receipts/predictable-expansion-20260919/README.md`.
