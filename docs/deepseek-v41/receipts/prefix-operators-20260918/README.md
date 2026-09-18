# Native prefix operator costs

Keep the full six-row verifier. These bounded measurements do not justify an
inter-layer split decoder; no production implementation or new test suite is
added. The retained full-workload result is **12.8091055 TPS** and the **20 TPS
goal remains open**.

All three operators use source `53277a85ceacfc999503cfbf343f68e79dc7c66c`,
unchanged native arithmetic and actual model weights. The inputs/history are
synthetic. They measure costs and state, not model throughput or token quality.
The 110,000,000,000-byte whole-machine ceiling remains enforced.

## Packed MLP

Two real layer-20 routes have 21 and 19 unique experts, each with six token rows
and 36 expert assignments. Native packed execution already computes every row
assignment; splitting does not add the previously conjectured 48.49% arithmetic
amplification. It adds dispatch and synchronization costs and changes reuse.

| Captured cycle | Full6 median | 1+5 total | 3+3 total |
| --- | ---: | ---: | ---: |
| 3 | 1.152208 ms | 1.374563 ms | 1.385125 ms |
| 198 | 1.152333 ms | 1.333438 ms | 1.368313 ms |

All sampled outputs match native kernel bytes. The first row is available about
0.765 ms earlier, and the first three about 0.438 ms earlier. Total costs rise
15.7–19.3% for 1+5 and 18.7–20.2% for 3+3. Control spread is 2.3–4.9%.
This is not a scheduling win: useful future reads would have to repay that cost.

The 6 GiB incremental envelope contains 48 native expert slots, one packed-scale
layer, scratch, cache, reader and compiler space. MLX peaks at 928,535,386 bytes;
the guard samples 1,374,373,952 bytes of process-tree footprint and
12,303,302,656 bytes of whole-machine physical use. Teardown leaves eight active
MLX bytes. Source cache is zero after the owned child exits. Independent health,
identity, warmup, no-child and free-lock verification completes at 04:50:51 UTC.

## Attention

Five real Attention objects cover SWA-only, Full/Reuse and Full/Reindex routes.
Each source precedes its consumer within the same chronological partition. The
logical position is 16,896, past the first compressed-bank growth boundary;
fresh native caches are seeded outside timing. Output projections use the
already-proven packed-owner retirement helper. No full backbone, HC or MoE runs.

| Layer group | 1+5 / full6 cost | 3+3 / full6 cost |
| --- | ---: | ---: |
| 0, SWA | 1.579 | unstable control; no reliable ratio |
| 2 + 3, Full/Reuse | 1.800 | 1.965 |
| 20 + 24, Full/Reindex | 1.569 | 1.676 |

The 3+3 SWA control varies 43%; do not use its point estimate. The other 3+3
controls vary 0.4–2.9%, and the 1+5 controls vary 0.8–1.7%. No repeat is needed
to reject the current split-only proposal.

All selected indices match in these cases. The 1+5 output and some persistent
cache values differ in floating point. The 3+3 outputs and canonical final 134
window rows match exactly; layer 2's index-key cache still differs. This does
not establish exact full-model state or classify a future token divergence.

Native window retention depends on compaction: full6 retains 144 rows and 1+5
retains 149. This is not evidence of a ring bug. The native minimum retention
already covers verification rollback. `attention1/probe.json` compares full
retained windows; `attention3/probe.json` additionally compares the canonical
window-plus-verification suffix. Neither probe changes cache behavior.

Each attention screen is bounded to 8 GiB incremental memory. MLX peaks are
1,240,336,688 and 1,140,246,192 bytes, respectively; both leave eight active bytes.
Guard process-tree peaks are 2,236,467,792 and 1,574,029,520 bytes; sampled machine
peaks are 13,663,518,720 and 12,136,349,696 bytes. Each controller reclaims
577,667,072 bytes of source file cache to zero after child exit. Both guards
exit zero and restore exact Qwen health and warmup before release. Independent
checks complete at 05:10:47 and 05:18:54 UTC, with no owned child and a free lock.

## Decision and provenance

The historical frequency-policy cap-83 screen already compared full6 with
staged 3+3: 15.427058 versus 15.787703 seconds for 128 timed steps. Reads fell
from 7,467 to 6,562, but wall time increased 2.34%. It matched that short output.
Its read-interval union overlaps compute, so subtracting it from wall time
does not measure exclusive compute. The original result is in
`../hidden-capture-110gb-20260917/screen-summary-20260917.json`.

The current operator results add missing cost/state evidence. They do not
prove that every possible asynchronous architecture loses, but they remove
the justification for building one from the prefix-hit histogram alone.
The next bounded candidate is smaller read-completion batches in the existing
plane-overlap runtime, preserving full6 attention and its cache semantics.

The directories retain exact commands, construction bounds, helper sources,
outputs and lifecycle receipts. `sha256.json` records all archived file hashes.
The existing packed-scale artifact is reused; its 3.09 GB payload is not copied.
