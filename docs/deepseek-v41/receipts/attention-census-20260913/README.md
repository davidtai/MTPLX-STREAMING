# Real-shape attention census, 2026-09-13

Source `f37d3ea074d0f1aa323b5323e162e1140984c9ca`. The guarded wrapper measures
one random-weight attention layer at the exact DeepSeek artifact dimensions and
mxfp8 projection codec: 16,384 cached positions, M=1/2/4/6, nine repeats, three
warmups and four queued copies. No full model, experts or MTP weights were loaded.
The wrapper, raw sub-op medians, memory samples and guard log are retained here.

| Weighted isolated sub-op proxy | M=1 ms | M=6 ms |
|---|---:|---:|
| Eager, pipelined | 24.355 | 33.664 |
| K29, pipelined | 26.816 | 48.885 |
| SDPA, pipelined | 24.681 | 44.169 |
| Eager, fenced | 46.781 | 60.133 |

These sums use 2 SWA, 4 full, 4 reindex and 30 reuse layers. They are neither
end-to-end measurements nor lower/upper bounds: the full-mode sample is layer 2
and omits layer-20-specific candidate-source operations. They do not establish
numerical parity for alternative kernels. Every exported op count is zero, so
graph counts are unavailable rather than evidence of zero work.

The measurements do not reproduce W114/W116's inferred sixfold attention premium.
A routing fence can include preceding graph work; it cannot by itself attribute
408 ms to attention or establish a six-TPS model-math ceiling. The shared-tile
kernel proposal needs new critical-path evidence before implementation.

Static scope was bounded below 8 GiB, with a 512 MiB allocator cache. Observed
MLX active peak was 1,106,782,172 B and sampled physical-used peak was
48,148,496,384 B. The guard exited successfully; the exact Qwen service was
restored healthy with warmup complete, and lock release was independently checked.
