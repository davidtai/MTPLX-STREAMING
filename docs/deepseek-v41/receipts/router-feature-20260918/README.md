# Causal router feature diagnostic

Source: `013ba48db733659d287d40e7304e1683a0d7e179`. The later router input
improves predicted physical-miss coverage in this diagnostic. No prefetch was
issued, no performance default changed, and **20 TPS remains unmet**.
The retained performance result is 13.1509467 TPS.

The current predictor applies the next gate to the mean residual entering the
current layer, before attention. The candidate applies that same next gate to
the current layer's native router input, after attention. Both inputs are
available before the current layer reads its routed experts.

## Cheap historical screen

The retained W35 trace was recovered from
`.benchmark-artifacts/deepseek-v41/route-traces-w35`. It contains a different
16K prompt and 256 AR rows, so it is only a preliminary screen. NumPy on CPU,
with MLX imports blocked, self-aligns with 99.9476% of native top-six IDs.
For target layers 4–39, top-one precision improves from 92.7734% to 95.5512%;
top-six recall improves from 64.7371% to 72.0775%. Adjusting normalization gains
only 0.092 percentage points at six, so the exact-workload capture omits it.
The screen takes 0.770 seconds and finishes at 95,830,568 process-footprint
bytes. Its static incremental envelope is 512 MiB.

The machine initially used 115.1 GB with Qwen running, so the screen used the
normal exclusive shutdown/reclamation/restoration guard. A launcher floor
refusal and a NumPy reader API failure are retained. The corrected screen's
guard 82331 exited 0; restoration/health/warmup/free-lock checks succeeded.

## Exact-workload capture

One full 16,384-input / 1,024-output Python run observes only its first 64 native
D5/M6 verification cycles. It records both predicted score tensors, native
routes, READY physical owners and actual physical record reads. The hooks are
installed after prefill/growth and removed after cycle 64. They issue no reads,
change no cache policy or route, and return the original native results.
Diagnostic fences invalidate throughput comparisons; headline decode TPS and
wall time are null, with observed instrumented values explicitly labelled.

All 1,024 output IDs match the retained native digest:
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
All 2,368 captured layer routes match the saved native trace. The 36 target
layers perform 10,623 physical reads during capture, with no duplicate reads.

Each layer's width, confidence margin and maximum issue count are chosen using
only cycles 0–31, requiring at least 85% physical-miss precision and eight
issued records. The settings are evaluated unchanged on cycles 32–63.
Predictions for READY physical records are excluded. The denominator is 4,878
actual physical misses across all 36 target layers in the held-out half.

| Input | Selected layers | Useful predicted misses | Extra reads | Precision | Miss coverage | Added traffic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing mean residual before attention | 14 | 237 | 52 | 82.01% | 4.86% | 1.07% |
| Native router input after attention | 28 | 763 | 134 | 85.06% | 15.64% | 2.75% |

No tested global setting meets the training precision threshold. These figures
are unlimited-lead-time estimates: they do not include predictor cost, finite
prefetch capacity, demand contention, owner replacement or time available
before the next layer. A bounded paired-layer I/O experiment is justified;
a full prefetch optimization or TPS claim is not yet justified.

## Complete memory and lifecycle accounting

Native prefill remains at 84 slots; the diagnostic grows to at most 105. It
retains every native reserve and adds 384 MiB host plus 64 MiB Metal. Captured
array payload is 45,591,040 bytes. Total host reserve is 1,774,317,568 bytes,
including the diagnostic component. The first full launch failed before model
allocation because the CLI host argument omitted that component. Variant 2
checks the real CLI budget resolver on CPU before launch; both accounting and
allocator limit include the reserve.

At the actual 10,645,929,984-byte baseline, the physical bound is
107,986,800,748 bytes. Measured MLX peak is 94,788,388,470; process footprint
95,828,935,456; whole-machine physical peak 107,152,556,032. The guard observes
the same machine peak, 221 samples and zero compressor growth. Limits remain
separate from usage; no RSS substitution or double-counting is used.

Full guard 77187 exited 0; exact Qwen identity, health and warmup were restored
before lock release at 08:21:52 UTC. The independent 08:23:58 UTC check confirms
healthy/idle/warmed service and a free lock. Source/packed clean-file cache is
zero after reclamation. No owned GPU child remains at this checkpoint.

`full-v2/` retains measured helpers and raw receipts; `admission-refusal-v1/`
retains the failed setup. `capture-artifact.json` pins the 45.6 MB NPZ stored
under the ignored artifact directory; the original /tmp path is a symlink.
Large model/packed-scale payloads and the isolated MLX library are not copied.
`SHA256SUMS.json` covers every other archived file. No new regression tests or
unchanged full benchmark repeats were added.
