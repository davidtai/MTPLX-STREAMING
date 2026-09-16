# Direct packed MXFP8 `wo_a` real-layer screen

This receipt is the promotion gate for using the artifact's packed MXFP8
`wo_a` directly with `mx.gather_qmm`. It measures one real target-attention
layer from the pinned DeepSeek V4.1 artifact with the exact non-contiguous
grouped layout produced by the fused output path. Both arms include the same
packed `wo_b` projection. This is a component result, not a model-throughput
claim.

| Rows | Cached BF16 control | Direct MXFP8 | Speedup | Time reduction |
|---:|---:|---:|---:|---:|
| 1 | 0.397832 ms | 0.327344 ms | 1.2153x | 17.72% |
| 6 | 0.525986 ms | 0.442868 ms | 1.1877x | 15.80% |

The isolated `wo_a` projection improved 1.2688x at one row and 1.5437x at six
rows. Full-output relative L2 error was `4.80e-7` at one row and `1.40e-5` at
six rows; maximum absolute error was `0.00012207` and `0.0078125`,
respectively. The alternate reduction order is a rounding-class change.

The direct route avoids the 67,108,864-byte BF16 transposed cache in each
target layer, or 2,684,354,560 bytes across 40 layers. The screen's MLX active
and peak allocations were 212,860,972 and 214,237,256 bytes.

The command ran as the sole child of `scripts/deepseek_v41/gpu_window.sh` with
`GPU_WINDOW_CHILD_RSS_CAP_BYTES=4294967296` and the 110,000,000,000-byte whole
machine ceiling. The guard reported a 12.2683 GB post-reclamation baseline,
no increase in its sampled physical or compressor peaks, child exit zero, and
restored `mtplx-flash-next-optimized-speed` with healthy completed background
warmup before releasing the GPU lock. Clean Qwen model-file cache reclamation
ran automatically after shutdown.

Source checkout HEAD was `cea0b40a3`. The benchmark and result SHA-256 values
are `74e03fa7cb713919a00bb0129bfbedf853cad07fac11c92cda57f4daf847d3b8`
and `4f9f55a4d18825c1e38dd24890a7b68047f2853d407098349f7102647e4154da`.
The exact 16K-input/1K-output model run remains the acceptance gate.
