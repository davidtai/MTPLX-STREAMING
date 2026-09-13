# Native MTP expert-route diagnostic

The exact pinned 16,384-ID Python prompt produced 1,024 output IDs in 206 native
block-5 MTP cycles. All output IDs match the earlier full MTP run. The wrapper
retained the three stages' existing `[5, 3]` gate-index arrays and converted
them only after the normal draft/generation fences; no production lane or
per-stage host synchronization was added. This is a locality diagnostic, not a
comparable TPS receipt.

| Stage | Distinct experts across 206 cycles | Mean distinct per cycle | Maximum |
|---|---:|---:|---:|
| 0 | 93 | 8.058 | 13 |
| 1 | 58 | 8.155 | 12 |
| 2 | 32 | 7.233 | 11 |

The current MTP head holds all 128 experts per stage: 7,219,445,760 B in total.
On these observed routes, a hypothetical cold LRU cache with `(72, 48, 24)`
stage slots would free 4,512,153,600 B, exactly six 40-layer target expert slots
per layer, but would add 213 draft record reads (4,004,536,320 B). These are
**offline** counts. Actual partial residency would also need bounded direct
reads from the source safetensors, correct native MXFP4 slot ownership, and
three new draft routing barriers per cycle. The current eager loader touches
whole expert-only shards, so merely filtering its loaded tensors is not a
bounded-memory implementation. Keep the draft fully resident until a paired
wall-time result proves the extra work pays for itself.

The safer next experiment is to allocate more target expert slots from the
measured 110 GB headroom by tightening only this workload's transient planning
band. That retains the existing barrier-free MTP head and source artifact.

The run used 71 target expert slots per layer at a measured 10.542 GB machine
baseline. Its conservative physical bound was 104,369,347,180 B; the 250 ms
sampled physical peak was 97,928,101,888 B, and MLX peak was 86,633,869,112 B.
No new swapouts occurred. The guard exited zero, restored the exact Qwen model
with completed warmup, and released the GPU lock; a separate health and lock
probe confirmed that state. The exact wrapper, route arrays, admission bound,
OS samples, guard log, and derived offline profiles are archived here.
