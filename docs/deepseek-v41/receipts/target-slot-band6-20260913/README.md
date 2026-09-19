# Exact 110 GB workload: 78 target expert slots per layer

The pinned Python programming prompt used 16,384 input IDs and produced 1,024
output IDs (the prefill token plus 1,023 decode steps). Native MTP block 5,
prefill layer-major, fanout 4, original MXFP4 artifact and production model code
were unchanged. This experiment reduced only the benchmark's transient planning
band from 11 to 6 GiB, retaining separate 2 GiB allowances for the Python
process and MLX allocator cache. The measured machine baseline of 10.2927 GB
admitted **78** target expert slots per layer, up from 72.

| Complete workload | Prior 72-slot run | 78-slot run |
|---|---:|---:|
| MTP decode TPS | 8.938810 | **9.498635** |
| MTP decode seconds | 114.444762 | **107.699688** |
| MTP target expert records read | 54,694 | **50,552** |
| MTP target read bytes | 1,028,282,204,160 | **950,409,953,280** |
| MTP MLX peak bytes | 87,384,896,236 | **91,897,033,560** |
| Whole-workflow sampled physical peak bytes | 98,520,252,416 | **103,392,870,400** |

Decode improved **6.263%** while reading 4,142 fewer expert records
(77,872,250,880 fewer bytes). Both 1,024-ID AR and MTP output streams match
the prior full run exactly. The existing AR/MTP difference at token 297 remains
the previously classified BF16 tie flip; native depth and acceptance are also
unchanged (206 cycles, 91.6388%). This is one complete candidate against the
prior complete control, not a repeated isolated A/B. The 20 TPS goal is unmet.

The admission bound charged the previous complete MTP MLX peak, all six extra
40-layer slot images, and 2 GiB of graph/workspace variation, with no credit
for cache reclamation or reduced prefill allocations. Its physical bound was
108,632,200,780 B against the hard 110,000,000,000 B ceiling. The guard's
250 ms samples peaked at 103,392,870,400 B across startup, AR, MTP and post-pass
work; swapouts remained at 4,399,765. The guard exited zero, automatically
reclaimed clean Qwen model file cache before the baseline, then restored the
exact Qwen model with completed warmup and released the GPU lock. A separate
health and nonblocking lock check passed afterward.

Raw benchmark JSONL, pass digests, 250 ms OS trace, admission calculation,
guard log, exact wrapper and machine-readable summary are archived here.
