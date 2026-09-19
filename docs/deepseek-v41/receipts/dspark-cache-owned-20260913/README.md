# Full Python workload after projection cache reclamation

Source `3a8b17284ef3ed6fa06cd130a0a56700e926ba62`; exact native artifact and
pinned 16,384-token Python prompt, 1,023 decode steps plus prefill token.
Depth5, pf0, fanout4, 48 shared transient slots, 68 persistent experts/layer.
110 decimal GB target; measured baseline 10.3676 GB; 2 GiB Python, 2 GiB MLX
allocator cache, 13 GiB transient band. Engine budget 83,526,272,640 B.

| Pass | Decode TPS | Decode seconds | Prefill seconds | MLX active peak GB |
| --- | ---: | ---: | ---: | ---: |
| AR reference | 5.790674 | 176.663369 | 134.332306 | 85.599756 |
| Native MTP | **8.556267** | 119.561493 | 108.434765 | 89.611286 |

MTP improves 1.11655% over the previous 8.461786 TPS result. This comparison
combines AR-prefill allocation fixes, projection-cache reclamation, and one
additional expert slot per layer. It does not isolate their speed effects.
**20 TPS remains unmet.** All 1,024 AR IDs and all 1,024 MTP IDs separately match
the previous run exactly. AR/MTP first differ at 297: an AR tie at 33.75 and MTP
margin 0.25; the same verified tie_flip as before. 206 cycles, 91.6388% acceptance.

250 ms OS sampling measured a whole-workflow physical peak of
**99,469,770,752 B**, including the later divergence replay: 5,441,667,072 B
below the previous full run. Process footprint peak 88,283,806,280 B is a
separate view, not an amount to add to physical usage. MTP decode-end MLX
active 73,579,638,724 B; allocator cache 2,122,337,529 B. No new swapouts
(4,399,765 pages before and after).

Admission credited no projection-memory saving: previous MTP active peak plus
exact slot delta and 1 GiB graph margin, then 2 GiB Python and 2 GiB allocator
cache, bounded physical usage at108,032,934,248 B. The prior whole-workflow
peak including replay, adjusted by the same baseline/slot delta and margin,
projected 106,730,905,248 B, also below admission. Keep that cross-check because
the MTP-pass active peak excludes later replay.

MTP read 57,785 records / 1,086,394,982,400 B. The I/O window was 82.187483 s,
13.2185 GB/s, realized queue depth 14.601. Guard exit 0; Qwen restored with exact
identity, health and background warmup verified, then lock released. An
independent health/warmup/lock check followed before the next guarded probe.
