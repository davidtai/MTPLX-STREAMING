# Per-layer prefill projection release: full workload

Source `1f3b9bca7ae5aab6a4bf30eb969b1f6dd0368711`, native MXFP4 artifact,
16,384 pinned Python input IDs and 1,023 decode steps plus the prefill token.
Native MTP depth 5, pf0, I/O fanout 4, 72 expert slots per layer and 48 shared
transient slots. **8.938810168 decode TPS; the 20 TPS goal remains unmet.**

The layer-major prefill now releases each layer's fp32 output-projection cache
after its existing `mx.eval(hs)` completion fence. It preserves reuse within that
layer's chunks, small verification calls, and subsequent nonfused decode cache
rebuilding. The generic loader still reserves the full fp32 cache lifetime.
[Focused quantized checks](../layer-major-projection-lifetime-20260913/README.md)
cover exact logits, hidden/KV state, ownership and repeated requests.

| Measurement | Previous, 68 slots/layer | Current, 72 slots/layer |
|---|---:|---:|
| AR decode TPS | 5.790674136 | 6.055909762 |
| MTP decode TPS | 8.556266508 | 8.938810168 |
| MTP decode seconds | 119.561493 | 114.444762 |
| MTP prefill seconds | 108.434765 | 108.412733 |
| MTP MLX active peak, bytes | 89,611,285,956 | 87,384,896,236 |
| Whole-workflow physical peak, bytes | 99,469,770,752 | 98,520,252,416 |
| MTP expert records read | 57,785 | 54,694 |
| MTP SSD bytes read | 1,086,394,982,400 | 1,028,282,204,160 |

This is one combined release/capacity comparison, not isolated speed attribution.
MTP throughput increased 4.470918%; expert reads fell 5.349139%. The extra four
slots/layer retain exactly 3,008,102,400 additional bytes. Current MTP I/O-window
bandwidth is 13.2277 GB/s over 77.737013 seconds. All 206 cycles and the 91.6388%
acceptance rate match the prior run. AR prefill took 134.879929 seconds.

All 1,024 AR IDs and all 1,024 MTP IDs separately equal their prior streams.
The AR/MTP difference remains at token 297: AR is exactly tied between the two
contested tokens at 33.75, while MTP gives 33.75/34.0. The full divergence record
is unchanged (`tie_flip`, maximum logit delta 0.75 within the recorded three-bf16-
ULP band). This is the user-authorized tie case, not general numerical leniency.

The admission calculation credited **zero unmeasured release savings**: measured
baseline 9,432,500,000 B, engine budget 86,608,856,288 B, 11 GiB transient band,
2 GiB Python allowance and 2 GiB retained allocator cache. Previous measured MTP
peak plus exact slot delta and 1 GiB graph margin gave an active bound of
93,693,130,180 B and physical bound of 107,420,597,476 B. The independent prior
whole-workflow/replay projection was 102,616,514,976 B. Hardware wired limit was
unchanged; the separate child cap was 100 decimal GB.

Physical memory was sampled every 250 ms, including the divergence replay:
2,575 samples, peak **98.520252416 decimal GB**. Process footprint peak was
88,413,815,240 B; it overlaps physical memory and must not be added to it. MTP
decode-end active/cache bytes were 76,587,741,124 / 2,121,803,959. Swapouts stayed
at 4,399,765. These are sampled host peaks, not guarantees about subinterval peaks.

Guard exited 0, restored the exact Qwen server, completed background warmup and
released the exclusive GPU lock at 13:21:54 UTC. Independent health, model,
warmup and nonblocking lock checks passed at 13:22:16 UTC. No OOM occurred.

`summary.json`, raw JSONL, external OS samples, target plan, bounds, wrapper,
pass summaries, output sidecars and guard log are archived here. The pass-summary
hook runs after generation timing freezes; it adds no token-path diagnostics.
