# Rejected exact workload: direct packed MXFP8 `wo_a`

This receipt is the exact 16,384-input-ID, 1,024-output-ID promotion gate for
the direct packed-MXFP8 target-attention `wo_a` route. The route improved both
AR and native depth-5 DSpark decode time, but it changed the greedy stream
outside the allowed BF16 tie band. It is therefore **rejected** and remains an
explicit opt-in experiment. The 20 TPS goal remains unmet.

| Complete workload | 78-slot control | Direct packed MXFP8 | Change |
|---|---:|---:|---:|
| AR decode TPS | 6.289236 | 6.468328 | +2.848% |
| AR decode seconds | 162.658872 | 158.155241 | -4.503631 s |
| MTP decode TPS | 9.498635 | 9.619617 | +1.274% |
| MTP decode seconds | 107.699688 | 106.345186 | -1.354502 s |
| MTP cycles | 206 | 208 | +2 |
| MTP acceptance | 91.6388% | 91.3870% | -0.2518 points |
| MTP target records read | 50,552 | 50,318 | -234 |
| MTP target read bytes | 950,409,953,280 | 946,010,603,520 | -4,399,349,760 |
| MTP MLX peak bytes | 91,897,033,560 | 91,897,036,372 | +2,812 |

All 40 target layers reported `direct_mxfp8_gather_qmm`, so the result is an
engagement measurement rather than a silent stock fallback. The full run did
not realize a lower MLX allocator peak even though the route omits the
2,684,354,560-byte persistent BF16 target `wo_a` cache.

The candidate AR and MTP streams first disagree at output 273: AR selected
token 27344 and MTP selected token 54921. The contested margins were 0.125 and
0.25, while the deltas at those tokens were 0.625 and 0.25. The maximum logit
delta was 1.125. With a BF16 ULP of 0.125 at the contested peak, the configured
three-ULP band was 0.375; the 0.625 delta exceeds it. The runner classified the
result as `divergent`, not `tie_flip`. The candidate AR stream also diverges
from the exact control at output 273, and the candidate MTP stream diverges
from the exact control at output 481.

The measured post-reclamation baseline was 12,981,400,000 B. The admission
bound was 108,636,529,944 B against the hard 110,000,000,000 B limit. Across
the complete guarded workflow, the 250 ms OS trace peaked at
106,764,632,064 B of physical use, 93,418,999,528 B of process
`phys_footprint`, and 96,201,916,416 B wired. Compressor use did not grow and
swapouts remained at 7,204. The exact Qwen service was restored healthy with
completed warmup before the GPU lock was released.

Two earlier admission attempts exposed a runner bug before model execution.
The original bank scan populated the file cache and crossed the guard ceiling;
`F_NOCACHE` prevented ordinary cached pages but left speculative pages behind.
The final scanner uses bounded mappings and invalidates each window before it
publishes the receipt. A later high-baseline attempt safely refused the run
before model loading because it could not retain 78 slots. Candidate-model
cache reclamation then found zero cached candidate pages, while automatic Qwen
cache reclamation removed 33,951,039,488 B of physical use before the accepted
run. These incidents are archived here to distinguish the runner fixes from
the rejected math optimization.

Raw benchmark JSONL, pass digests, 250 ms OS traces, admission calculations,
guard logs, output sidecars, the exact wrapper, and a machine-readable summary
are archived in this directory.
