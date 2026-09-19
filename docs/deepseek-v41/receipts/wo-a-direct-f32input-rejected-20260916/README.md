# Rejected exact workload: FP32 input to packed MXFP8 `wo_a`

This receipt tests one follow-up to the rejected direct packed-MXFP8 target
attention route: cast the grouped BF16 activation to FP32 before
`mx.gather_qmm`, then cast the projected result back to BF16. The production
tree was not changed. The candidate was installed only by the archived exact
workload wrapper.

The one-layer screen did not establish a robust improvement over the existing
packed route. At one row it measured 0.324057 ms versus the earlier packed
route's 0.327344 ms, within cross-window noise. At six rows it measured
0.455091 ms versus 0.442868 ms, 2.76% slower. The transient dense alternatives
were exact but 2.0-2.4x slower than the cached BF16 path and were dropped.

| Complete workload | Accepted 78-slot control | Prior packed route, 78 slots | FP32-input route, 79 slots |
|---|---:|---:|---:|
| AR decode TPS | 6.289236 | 6.468328 | 6.404937 |
| MTP decode TPS | 9.498635 | 9.619617 | 9.693073 |
| MTP decode seconds | 107.699688 | 106.345186 | 105.539287 |
| MTP target records read | 50,552 | 50,318 | 49,684 |
| MTP target bytes read | 950,409,953,280 | 946,010,603,520 | 934,090,997,760 |

The apparent MTP gain is not an isolated arithmetic win: the lower measured
baseline admitted 79 expert slots instead of 78. AR regressed 0.98% against
the prior packed route. Both candidate token hashes are identical to the prior
rejected route.

The first AR/MTP disagreement remains output 273, token 27344 versus 54921.
The delta at the AR token is 0.625 while the three-BF16-ULP tie band is 0.375;
the runner therefore classifies it as `divergent`, not `tie_flip`. This
candidate is rejected. No production code or tests were added.

Memory is reported with separate semantics. The MTP pass's MLX allocator peak
was 92,650,042,192 B. The full guarded workflow's 250 ms trace measured a
91,983,368,344 B process `phys_footprint` peak, 94,603,411,456 B wired peak,
and 109,847,248,896 B whole-machine physical-use peak. Compressor use did not
grow and swapouts stayed at 7,436. The whole-machine peak left only
152,751,104 B below the 110,000,000,000 B limit.

The static admission estimate was 108,600,255,544 B, 1,246,993,352 B below
the observed workflow peak. Future full-model admissions must retain more
headroom rather than treating this estimate as a hard upper bound.

The guarded run exited zero and verified the exact Qwen model healthy with
completed background warmup before releasing the GPU lock. Raw benchmark,
pass, OS-trace, admission, guard, wrapper, and component-screen artifacts are
archived beside this file.
