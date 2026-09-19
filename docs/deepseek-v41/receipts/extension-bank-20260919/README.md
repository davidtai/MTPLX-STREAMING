# Preserve expert banks while growing the decode cache

The exact Q4 16,384-input / 1,024-output Python workload reaches
**13.8688167379 TPS / 73.7626013330 seconds**, with 198 target calls and all
1,024 native output IDs unchanged. This is the best single complete result,
2.5001410420 seconds shorter than the historical 13.4141517619 TPS run.
The historical comparison has different background memory and 110 versus
111 slots per layer. It is not a repeated, isolated full-model speedup.
The 20 TPS target remains open: another 22.6126 seconds must be removed.
Production defaults remain unchanged; this is a measured experimental lane.

## Mechanism and ownership

Keep the 84 prefill expert rows and their backing arrays. At the decode
boundary, retire raw scales and install packed scales without resizing any
expert weights. Complete the native MTP seed, then allocate a separate
extension bank. This run adds 27 rows per layer, giving 111 final slots.
Original physical row objects and indices remain stable. The allocator owns
each extension before subsequent operations can fail, so ordinary cleanup
covers both banks. Cache policy, capacity and physical memory plans update
once at the quiescent boundary.

The previous predictable non-MoE schedule is unchanged: retain native packed
output projections and expand the next layer while current expert reads run.
Two BF16 buffers replace the full BF16 cache; admission prices a third buffer
for replacement. Projection arithmetic, expert kernels, native KV16 and the
D5 plus two-token causal lookup verification path are unchanged. The measured
hot path gains no new eligibility branch, fallback or proof counter.

## Equal-capacity component

An A/B/A comparison at 110 final slots replays the same 206 M6 routes from
layer 34 with all 40 packed output projections and synthetic inputs.
Attention is omitted. Allocation is charged to replay; every expert and
projection output matches exactly in all three arms.

| Arm | Allocation, ms | Warm replay, ms | Charged replay, ms |
| --- | ---: | ---: | ---: |
| Resize 84→109, append one row | 46.998 | 1338.284 | 1385.282 |
| Keep 84, append 26 rows | 9.354 | 1333.952 | 1343.306 |
| Resize 84→109, append one row | 49.058 | 1333.558 | 1382.616 |

The candidate is 2.93676% faster than the controls' median charged time;
control spread is 0.19264%. Steady replay is approximately flat. All arms read
759 expert records / 13,430,292,480 bytes and no dense weights from SSD.
Only the candidate preserves the original backing arrays. This is component
evidence, not an extrapolated full-model TPS result.

The 26 GiB incremental bound prices 10 GiB Metal/cache/compile, 4 GiB host and
12 GiB source cache. Explicit Metal inventory is 10,104,602,624 bytes.
Cumulative MLX peak is 6,612,029,964 bytes, with 8 bytes active after cleanup.

## Complete result and memory

| Measurement | Result |
| --- | ---: |
| Fresh background baseline | 10,983,129,088 B |
| Launch physical estimate | 109,745,344,620 B |
| Internal sampled machine peak | 109,852,753,920 B |
| Guard sampled machine peak | 109,798,211,584 B |
| Process phys_footprint peak | 97,565,150,648 B |
| MLX peak | 96,548,916,582 B |
| Packed-scale installation plus extension allocation | 2.1956928751 s |
| Expert records / weight bytes | 31,573 / 558,675,394,560 B |
| Union of read intervals | 43.398996549 s |

The measured machine peak remains below **110,000,000,000 bytes**, with
147,246,080 bytes of sampled headroom. It exceeds the launch estimate by
107,409,300 bytes; do not report the launch estimate as an observed bound.
Whole-machine values include file cache. Allocator, process and machine
metrics overlap and must not be added together. The current observations
cannot assign background variation to a particular application, and sampling
can miss shorter peaks. A future full run needs fresh admission accounting
for this variation. The 100 GiB wired ceiling remains unchanged; peak sampled
wired memory is 100,985,913,344 bytes. No compressor growth is observed.

Growth is 1.4780189590 seconds shorter than the preceding packed-projection
run. The extra slot reduces expert reads by 388 records / 6,865,551,360 bytes
against the historical best. All transition work, native seed and projection
priming remain inside the decode clock. Output is the same coherent Python
diff opening for generation statistics; it is truncated at 1,024 tokens and
is not a complete validated patch.

## Verification and lifecycle

After the full win, two CPU regressions pass with MLX imports blocked:
admission at the exact decimal ceiling versus one byte above it, and refusal
under excessive wired pressure. Existing component/full checks cover physical
row ownership, backing arrays, byte accounting, output identity and cleanup.
No broad GPU test suite or extra full repeat was run.

Both owned guards exit zero, restore exact Qwen identity and completed warmup,
then release at 14:53:30 UTC (component) and 15:06:03 UTC (full). An independent
15:07:32 observation finds foreign owner 60643, the API stopped for that later
window, and no owned child or waiter. No unrelated process is signaled.

Measured sources are pinned to `d5f15e7a02d225115debf6ce317e26dc4a935b60`.
`summary.json` retains separate bounds, measurements, comparison limits and
raw-file hashes. `sha256.json` covers this curated archive. Original absolute
paths identify the exact temporary installation; the separate local input
checkpoint retains raw OS samples and all helper inputs. Large model/packed
artifacts and the attested strict library remain at their pinned local paths.
