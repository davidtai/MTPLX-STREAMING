# Current native DSpark workload, 2026-09-13

Source `4b1e5d369c2d5be35927e09caa3d7d11c3357406`, native local mxfp4
artifact, pinned Python prompt (16,384 IDs), 1,023 decode steps plus the prefill
token. Both streams reached the 1,024-token cap. The **20 TPS goal is not met**.

| Lane | Decode TPS | Decode seconds | Prefill seconds | MLX active peak GB |
| --- | ---: | ---: | ---: | ---: |
| AR reference, MTP weights resident | 5.764061 | 177.479050 | 135.091648 | 88.860239 |
| DSpark, depth 5 | **8.461786** | 120.896457 | 108.907734 | 91.544600 |

DSpark is 29.46% faster than the previous best 6.535996 TPS AR run (83 slots),
and 46.80% faster than this equal-capacity reference (67 slots). This is one
full run, not a repeatability claim. The first divergence, at output 297, is an
AR tie: candidates 11067 and 62596 both score 33.75. DSpark scores 33.75 and 34.0;
the classifier reports `tie_flip`, consistent with the user's tie allowance.
The reference's entire stream still matches the earlier AR digest.

The fresh machine baseline was 10.3739 decimal GB after automatic Qwen cache
reclamation freed 36.232937472 GB of physical usage. The plan charges MTP
weights and caches before selecting 67 expert slots/layer, with 48 shared
transient records, no prefetch ring, fanout 4, 2 GiB Python, 2 GiB retained Metal
allocator cache and a 14 GiB temporary band. The reviewed active bound was
93,362,425,264 B and physical bound 108,031,292,560 B.

The 250 ms sampler observed:

- AR pass physical peak 102,048,481,280 B.
- DSpark pass physical peak 102,210,043,904 B.
- Entire workflow physical peak**104,911,437,824 B**, including divergence
  replay. Process footprint peaked 94,498,371,568 B; it is not added to physical
  used memory, which already includes the process.
- 2,648 samples; swapouts remained 4,399,765 pages.

The receipt completed before the guard returned 8 on `live step state
unreadable` during process exit. Thus this is a completed benchmark measurement
with a failed guard exit classification, **not a clean guard-exit receipt**.
Exact Qwen service/warmup and lock release were verified after restoration at
12:24:06 UTC. No OOM occurred. A later 64 MiB CPU reproduction observed Darwin
`ps` state `?E`; the subsequent guard fix is documented separately.

DSpark produced about 4.97 outputs/cycle across 206 cycles. Per-cycle timings:
draft 12.30 ms, verify 571.15 ms, accept 0.92 ms, commit 1.73 ms. Decode read
1,101,134,684,160 B from SSD:1.076GB/output, 57.25 records/output. The measured
I/O-window bandwidth was 13.2378 GB/s; it is not a hardware ceiling or an additive
critical-path timing. Verification and SSD traffic remain the optimization
priority. The draft-only 9.36 ms probe is in `../dspark-head-peak-20260913/`.

Raw JSONL, 250 ms OS trace, guard log, target plan, preflight bounds, output texts,
wrapper and a compact `summary.json` are retained here. The external
`dspark_end` phase also covers divergence replay. No old receipt was rewritten.
