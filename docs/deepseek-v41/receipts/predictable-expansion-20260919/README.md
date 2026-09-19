# Predictable packed output-projection expansion

Q4 only, measured at `d569990aa606bb86c877645f8d10f1d0d1450c58`.
The exact 16,384-input / 1,024-output Python goal remains 20 TPS. The retained
complete result remains 13.4141517619 TPS. This candidate completes at
13.3988661661 TPS and saves about 1.17 GB of sampled process memory.

The target's layer order is known. Keep its native packed MXFP8 `wo_a` weights
resident, then expand the next layer into the exact native BF16 transposed
layout while current-layer expert reads are in flight. Keep grouped BF16
matmul, RoPE, packed `wo_b`, and expert arithmetic unchanged. No recurring
dense SSD reads or new quantization is introduced.

## Component evidence

Both comparisons rotate through all 40 real output projections with synthetic
inputs and replay 206 saved M6 routes for expert layer 34. Attention is omitted.
The first four calls warm the kernels. Both complete and warmed timing are
retained; output comparison runs after timing. Every expert and projection
output matches, and each process releases Metal ownership to 8 bytes.

| Comparison | Control median | Candidate | Change |
| --- | ---: | ---: | ---: |
| Resident BF16 / 110 rows vs packed expansion / 111 rows | 1.333788 s | 1.311799 s | 1.65% faster |
| Same, with one separate extra-row bank and allocation charged | 1.336260 s | 1.317415 s | 1.41% faster |

Both use A/B/A order. Control spread is 0.7202% and 0.6202%, respectively.
Each candidate reads 748 expert records versus 759, or 13,235,650,560 versus
13,430,292,480 bytes. The separate extra bank takes 2,188,916 ns to install;
the existing row owners do not move. Existing packed kernels group physical
rows by backing bank, so the additional bank's dispatch cost is included.

A separate native 110→111 resize probe measures a median 44,943,833 ns per
layer, projecting 1.79775332 seconds across 40 layers. Existing payloads are
exact and the new row is zeroed, but copying would erase the component gain.
That second-resize strategy is rejected. Its raw evidence is under `resize/`.

## Memory and scope

The old full BF16 projection cache is 2,684,354,560 bytes. The candidate retains
1,384,120,320 packed bytes and two 67,108,864-byte BF16 buffers. Pricing a third
buffer during replacement gives a conservative 1,098,907,648-byte credit.
One additional packed expert row in every target layer requires 707,788,800
bytes. Packed `wo_b` is unchanged. This is a payload calculation, separate
from the full prefill, growth, seed, KV, allocator, Python and machine budget.

The component's bound is 26 GiB incremental: 10 GiB Metal/cache/compilation,
4 GiB host and reader allowance, and 12 GiB source-file-cache allowance. Every
guard retains the 110,000,000,000-byte machine ceiling and 100 GiB wired ceiling.
The per-arm `mlx_peak_bytes` field is cumulative within the process; it is not
an isolated candidate peak. `active_after_eval_bytes` is a per-arm snapshot.
The raw construction plan describes initial native banks; the stripped-weight
inventory and extra-row ownership are reported by the measured arm.

All three guards complete successfully and restore exact Qwen identity,
health, and completed background warmup before release, at 13:49:18, 13:59:49,
and 14:08:20 UTC. Later independent checks encounter foreign GPU windows;
they do not imply failure of those completed restorations. Raw guard and
child receipts preserve the lifecycle evidence.

## Full integration status

The full candidate preserves 84-row prefill and the original first packed
growth. After the native MTP seed, it appends one separate row per layer and
primes layer 0. Every target expert runner issues the following layer's exact
expansion at the measured demand/resident boundary. Layer 39 prepares layer 0
for the next target call. All allocation and priming costs are charged.

An extra 16 MiB host reserve is priced in every phase. CPU checks compare the
native projection body, expert schedule, kernels and hybrid proposal path, and
check separate prefill, first-growth, seed, append/prime and steady bounds.
The full plan preserves an unset optional expert-cache limit; the component's
explicit limit had hidden that integration case.

## Complete 16K/1K result

| Metric | Historical best | New packed-expansion candidate |
| --- | ---: | ---: |
| Decode TPS | 13.4141517619 | 13.3988661661 |
| Decode wall | 76.2627423750 s | 76.3497438750 s |
| Initial → final decode capacity | 110 → 110 | 109 → 110 |
| Background | 10.447 GB | 11.483 GB |
| MLX peak | 97.007 GB | 95.841 GB |
| Sampled process footprint peak | 98.028 GB | 96.861 GB |
| Guard whole-machine peak | 109.239 GB | 108.679 GB |

All 1,024 output IDs match the retained native-MTP control. Both runs use
198 target calls and read 31,961 expert records / 565,540,945,920 bytes.
The output is the same coherent Python diff opening, cut off at the requested
1,024-token limit; it is not a complete validated patch. The existing AR
tie-flip classification remains unchanged. No new output drift is accepted.

The candidate's first growth takes 3.618981 s, and appending 40 separate rows
takes 0.054731 s. Existing row owners remain unchanged. Completion verifies
1,384,120,320 packed projection bytes plus 134,217,728 BF16 buffer bytes, with
no full per-layer dense caches. Peak process footprint drops 1,166,851,880
bytes and MLX peak drops 1,165,892,772 bytes in the historical comparison.

The fresh admitted bound is 109,965,277,404 bytes, including 1,438,773,248 bytes
of host reserve. Guard physical peak is 108,678,955,008 bytes with no compressor
growth. Guard and child exit zero; exact Qwen identity, health and completed
warmup are restored before release at 14:33:10 UTC. The later independent
check sees a foreign GPU owner and no owned child or waiter.

This is a verified memory-saving candidate, with wall time 0.0870 s / 0.1141%
longer than the historical best. It is not an isolated or repeated throughput
gain and is not promoted. The 20-TPS target remains open. No broad regression
suite was run. CPU construction and admission checks and the exact full output
are preserved under `full/`. Scratch inputs and launch files remain at
`/tmp/dsv41-predictable-expansion-20260919`.
