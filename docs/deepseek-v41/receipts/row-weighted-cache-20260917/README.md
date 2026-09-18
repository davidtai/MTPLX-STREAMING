# Position-weighted expert retention: rejected on CPU

Earlier verification rows are more likely to be committed. This screen weights
their past demand by 0.9 per row, while still serving every requested expert.
Neither candidate reduces misses, so neither warrants GPU execution or tests.

| Policy | First 103 cycles | Held-out 103 cycles | Total demand misses |
|---|---:|---:|---:|
| Native transition-window | 19,620 | 15,544 | 35,164 |
| Rewritten unit-weight control | 19,620 | 15,544 | 35,164 |
| Weighted frequency | 19,633 | 15,546 | 35,179 |
| Weighted predictor and frequency | 19,635 | 15,587 | 35,222 |

The unit-weight control matches every native per-layer, per-cycle miss count.
Each arm starts from the same captured 73 residents plus 29 empty slots, uses
the native planning/pinning path, and sees only current and earlier requests.
Acceptance lengths, target logits and future routes do not select retention.
The two decay designs were fixed before the replay. Each takes under 0.7 CPU
seconds; MLX imports are blocked by the existing replay helper.

These are policy demand misses, not physical reads or throughput measurements.
No production code changed. `sha256.json` binds the script and full results
measured at source `123b9021066d365028e09ed4f0154092bb19a6c0`.
