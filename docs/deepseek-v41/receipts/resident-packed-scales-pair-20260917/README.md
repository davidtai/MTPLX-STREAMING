# Packed-scale comparison with bounded prefill, 2026-09-17

One sequential native/packed comparison covers the exact 16,384-input /
1,024-output Python workload. Packed scales produce **12.2254 TPS** versus
**11.7142 TPS** native: an observed **4.364% throughput increase**, or
**3.6517 seconds / 4.181% less decode wall time**. All 1,024 output IDs match
between arms and the retained native stream. The 20 TPS goal remains unmet;
the best previous complete candidate remains 12.4440 TPS at 91->102 capacity.

| Measurement | Native | Packed scales |
| --- | ---: | ---: |
| Prefill / decode slots per layer | 84 / 98 | 84 / 99 |
| Decode seconds, including phase installation | 87.330065 | 83.678383 |
| Decode TPS, 1,023 timed steps | 11.714179 | 12.225380 |
| Live baseline, bytes | 10,974,773,248 | 11,254,906,880 |
| Whole-machine admission bound, bytes | 109,298,427,372 | 109,000,972,520 |
| MLX allocator peak, bytes | 92,514,520,730 | 91,920,930,862 |
| Sampled process footprint peak, bytes | 94,272,993,632 | 93,715,871,712 |
| Sampled whole-machine peak, bytes | 105,846,996,992 | 105,399,713,792 |
| Physical expert records read | 36,783 | 36,357 |
| Expert read bytes | 691,543,941,120 | 643,326,935,040 |
| Read interval union, seconds | 53.956705 | 49.539127 |
| Charged phase installation, seconds | 1.893162 | 3.387902 |

Process and machine peaks in the table use the external sampler. They overlap
with allocator usage and must not be added. Neither arm reports new swapouts.
Packed-scale installation reads another 3,086,136,060 bytes, included in
`summary.json`'s combined read-byte comparison. Read interval union is not a
measurement of GPU idle time or host blocking time.

Both arms use native BF16 target arithmetic, compact MTP 93/58/32, D5/M6,
48 shared transients, no prefetch, transition-window policy, three-record miss
parts and shared-work overlap. Their decode capacity is the largest admitted
by each exact storage layout and live baseline under the same 110,000,000,000
byte ceiling. This is one pair with differing live baselines and capacities;
it does not isolate the kernel contribution or establish a repeated speedup.

The attempted 91-slot prefill could not fit the changed baseline. The new
construction helpers use the previously measured 84-slot prefill and its
unchanged native allocation envelope. They retain the 109.5 GB admission target,
2 GiB host reserve, 1 GiB allocator-cache policy, measured cache overshoot,
complete logical KV allowance, copy bounds, page padding and compile margins.
CPU admission checks cover lower and higher baselines and still refuse the
28.35 GB case. No expert kernel, reader, route or acceptance logic changed.
`construction-proof.json` binds the prior full cap84 receipt and changed files.

The current reporting fix is exercised by both real arms: native records are
18,800,640 bytes with 902,430,720 transient bytes; packed weight records are
17,694,720 bytes with 849,346,560 transient bytes. Both preserve the original
source record size, and the transient allocation is shared across all layers.
The packed phase separately owns 3,086,136,060 bytes of resident scales.

One guard owned the complete native/packed batch. The parent imported no MLX,
ran the model children sequentially, and reclaimed only the owned model,
compact-resident and expert-file cache between arms and after the batch.
Each cleanup removed 12,416,466,944 cached-page bytes to zero; cached pages
include speculative/free pages and are distinct from physical-used memory.
The packed child also reclaimed its packed payload. Child exit was 0, with
462 complete guard samples and zero compressor growth. Qwen served requests
during restoration; background warmup yielded to them. Exact model identity,
health and warmup were restored before lock release at 20:42:03 UTC. The
independent 20:42:43 UTC check found no owned children and a free lock.

Source revision: `170a576dcb22183c5e5d1366b4241a1f03e67de3`. `command.txt` and
`pair-config.json` preserve the exact launch and per-arm arguments. Helpers
retain their measured temporary paths. `raw-manifest.json` records the original
hashes of gzip-compressed raw receipts, samples, passes and output sidecars.
The 3.086 GB packed payload is reused from the prior durable ignored artifact;
it is not duplicated here. These remain one-request benchmark helpers, with
an explicit rejection of a second prefill, rather than a general-serving lane.
