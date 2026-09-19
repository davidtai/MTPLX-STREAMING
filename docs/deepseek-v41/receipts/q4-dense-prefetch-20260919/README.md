# Q4 predictable query-weight streaming

The user's scheduling premise is correct: target query projections execute in
known layer order. A two-buffer prototype preserves every measured output while
releasing most of their persistent storage. Its extra reads outweigh the expert
cache benefit in this bounded comparison, so this version is not promoted.

## Construction and scope

All forty real `attn.wq_b` projections use native MXFP8, group size 32, with unchanged
`mx.quantized_matmul` arithmetic. The resident payload is 1,730,150,400 B;
two writable buffers require 86,507,520 B. The 1,643,642,880-byte difference
can fund two additional packed expert slots per layer before full admission.

The probe combines these projections with real Q4 expert reads and computation
for layer 34's saved 206-call M6 route sequence. It restores the same 73 initial
expert residents and adds empty slots to reach 110 or 112. Query inputs are
synthetic; saved route IDs carry a dependency on the query result, so the
existing router evaluation consumes the current query buffer. A single dense
reader prefetches the next layer into the other buffer after expert demand
submission. This ordering does not confer hardware I/O priority. The original
packed expert kernels, read fanout 4, three-record miss parts and deferred slot
releases are preserved.

The loop is continuous, with the previous expert output feeding the next query
dependency. Output hashing is outside timing. The first four calls warm kernels;
reported steady timing covers the remaining 202 calls. This component omits
attention and the complete model, so it is not a full-workload TPS estimate or
a conclusive rejection of every possible dense offload schedule.

## Results

One guarded A/B/A/B/A batch at source `70331ffc5b7aec6b41cff6d173766811d6ef787b`:

| Measurement | Resident query weights, 110 slots | Streamed query weights, 112 slots |
|---|---:|---:|
| Median steady wall time | 1.279569542 s | 1.920236479 s |
| Expert records read, 206 calls | 759 | 740 |
| Expert read bytes | 13,430,292,480 | 13,094,092,800 |
| Dense read bytes during execution | 0 | 8,910,274,560 |
| Query backing bytes | 1,730,150,400 | 86,507,520 |
| Active MLX bytes after evaluated outputs | 4,764,666,592 | 3,156,413,048 |

The streamed arm takes **50.0689% longer**, with 2.3706% control spread.
Both candidate repetitions preserve all 206 query and expert-output digests,
including buffer reuse through five complete layer-order wraps. Every output
is finite. The candidates save 336,199,680 expert-read bytes while adding
8,910,274,560 query-read bytes. Do not treat predictable read scheduling as a
bandwidth saving or promote this component as a speed optimization.

## Memory and lifecycle

The incremental static bound is 24 GiB: 8 GiB for Metal/cache/compile, 4 GiB for
host/compiler state and 12 GiB for conservative touched-source file cache.
The guard retains the 110,000,000,000-byte whole-machine ceiling and 100 GiB
wired limit. Sampled peak whole-machine physical usage is 21,430,992,896 B;
sampled child-tree footprint is 5,198,615,040 B. These measurements overlap.
MLX allocator peak is 4,765,497,916 B; only 8 B remains after owner cleanup.

Guard 37741 and its child exit 0. Source reclamation removes 1,479,802,880 cached
bytes after the child exits. Exact Qwen identity, health and warmup are restored
before the lock releases at 11:49:16 UTC. An independent check at 11:50:33 UTC
confirms healthy/idle/warmed Qwen and a free lock. No unrelated job is signaled.

The original CPU inventory and exact 53,999-read historical policy replay remain
in this directory. `probe-v1/` contains the new source, bounded installation,
all output digests, measurements, reclamation and lifecycle evidence. Its
`archive-sha256.json` pins the archived bytes. Large immutable packed-scale
artifacts remain at the installation's recorded local path.

The retained full Q4 result remains 13.4141517619 TPS on 16,384 input / 1,024 output
tokens. 20 TPS remains unmet. This losing component does not justify a full-model
run or additional optimization regressions.
