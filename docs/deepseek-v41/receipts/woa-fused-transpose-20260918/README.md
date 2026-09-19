# Fused MXFP8 decode/transpose: rejected full-model candidate

One fused conversion kernel preserves the native BF16 weight layout and all
tested projection outputs, but remains slower than the retained cached-BF16
route. No full-model run, default change, or new optimization test follows.
The 20 TPS goal remains open.

This differs from the previously rejected direct `gather_qmm` route. The kernel
decodes MXFP8 group32 weights into a contiguous `[8,4096,1024]` BF16 array, using
a 32x32 transpose tile. The original grouped `mx.matmul`, BF16 output and native
MXFP8 `wo_b` operation remain unchanged. Five real target layers—0,2,3,20,24—are
cycled to cover more than a single repeated projection.

All five complete converted weight arrays match byte-for-byte. All final
projection outputs also match at M1 and M6 for both the native fresh-conversion
control and the fused candidate.

| Five-layer operator median | M1 | M6 |
| --- | ---: | ---: |
| Retained cached BF16 | 1.944042 ms | 2.628500 ms |
| Native fresh decode plus transpose | 4.749208 ms | 5.382522 ms |
| Fused fresh decode plus transpose | 3.041375 ms | 3.830083 ms |

At M6, fusion reduces the fresh-conversion path's time by 28.84%, but costs
45.71% more than retaining the BF16 weights. The extra cost is 0.2403166 ms per
layer. Multiplying by the exact workload's 40 layers and 206 cycles gives
1.9802 seconds of projected overhead; this is not a measured full-model result.
The M6 cached-control block spread is 1.87%; M1 is noisier at 7.98%. Raw samples
are retained in `probe.json`, with two untimed warmups per block and nine timed
samples in each of seven interleaved blocks.

The memory tradeoff also fails on its own. At the latest full run's baseline,
the preceding bank-growth copy envelope would put 104 slots at
110,363,144,536 bytes. A change to later projection storage leaves that growth
envelope unchanged. Thus this candidate alone admits no extra slot under
110,000,000,000 bytes. See `summary.json` for the calculation and its source.

The operator statically admits 8 GiB: 4 GiB for Metal/cache/intermediates and
4 GiB for Python, bounded I/O and compilation. It reads 389,283,840 bytes of
selected weights from five authenticated-header shards. MLX peaks at
861,863,936 bytes and returns to zero active bytes. The two complete guard
samples observe a 789,448,240-byte process-tree footprint peak and a
12,046,860,288-byte machine physical peak, with zero compressor growth. These
sampled figures do not claim to capture every short transient.

Measured source: `9835305c1f2a66b7ad4ef98fd877db81156cd3ae`. Guard96675 is
terminal exit0. The owned controller reclaims 331,923,456 source-file bytes to
zero after child exit. Exact Qwen identity, health and warmup are verified
before lock release at 06:33:42 UTC on September18. Independent verification at
06:36:13 UTC confirms healthy/idle/warmed Qwen, a free lock and no owned child.

The source audit consulted MLX v0.32.2's
[FP8 conversion definitions](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/kernels/fp8.h)
and [floating-point dequantization kernel](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/kernels/fp_quantized.h).
Those exact files, their hashes and the upstream MIT license accompany the
receipt. Parity above comes from the installed runtime, not from assuming the
source tag proves binary identity. Historical command/source pins must be
refreshed before any changed candidate; do not rerun this losing arm unchanged.
