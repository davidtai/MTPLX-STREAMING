# Fixed Q8 full-workload result

Fixed Q8 completes the exact 16,384-input / 1,024-output Python workload at
**10.7541904 decode TPS / 95.1257102 seconds**, including the 3.3733-second
packed-scale installation and growth from 84 to 101 expert slots per layer.
The retained native-KV result remains 12.4439935 TPS. Q8 is available as an
explicit memory/precision choice; it is not a throughput winner. **20 TPS is
still the primary open goal.**

Measured runtime source: `67c0906cbc658de6219d384cf94bbdbc0484efe2`.
Settings: `--box-target-gb 110 --kv-cache-bits 8 --max-kv 17664
--kv-max-append 953`, native BF16 target/head arithmetic, compact draft experts
93/58/32, D5/M6, 48 shared transients, transition-window policy, packed scales,
three-record miss parts, shared-work overlap, and no prefetch.

## Memory and lifetime evidence

A bounded cache-only Native/Q8/Native probe reproduced the layer-major prefill
lifetimes: per-chunk compressed/index views remain alive while later chunks
append, then are replaced by the next source layer. It covered all 40 target
layers, three draft caches, real widths, native compressor dtypes, 16K prefill,
and verification rollback. Native peak was 669,729,292 bytes in both arms;
Q8 peak was 373,304,920 bytes. Each arm released to eight active bytes.
These timings are not model throughput evidence.

The full admission retains the established native M6/M8 envelope and adds the
entire **503,316,480-byte Q8 reserve**. No native KV allowance was discounted.
The reference additionally reserves 640 MiB for complete FP32 logit capture.
Both use the same 110,000,000,000-byte whole-machine ceiling.

| Measure, decimal GB | Q8 AR reference | Q8 MTP candidate |
| --- | ---: | ---: |
| Background baseline | 10.9330 | 10.1523 |
| Calculated physical bound | 109.8535 | 109.8173 |
| MLX allocator peak | 91.1407 | 92.9709 |
| Internal process footprint peak | 92.9414 | 94.7931 |
| Internal whole-machine peak | 104.1612 | 106.2490 |
| Independent 250 ms whole-machine peak | 104.1523 | 106.2636 |

The guard's separate one-second candidate peak was 106,263,920,640 bytes.
Sampling windows differ. These overlapping measurements are never added.
Fixed target/draft backing storage is 112,503,168 bytes; temporary decoded
views and allocator cache are reported separately. Neither run had new swapouts;
the candidate's independent sampler observed eight swapins.

## Output and performance interpretation

The new Q8 AR reference contains all 1,024 tokens and complete FP32 logits for
every output position. All row hashes were verified. It includes logit-capture
overhead, so its 7.005 TPS is diagnostic and is not a candidate comparison.
The 529,530,880-byte binary remains in ignored local artifacts; its full hash,
location and row inventory are retained in `logits-artifact.json` and the receipt.

The candidate produced all 1,024 tokens in 229 cycles. Its first difference
from Q8 AR is at index 53: each contested margin is 0.125 and both contested
logit deltas are 0.125, within the existing 0.375 BF16 tie band. The exact
index-matched rows pass the user's authorized tie-break gate. This validates
that gate on this workload, not general model quality or native/Q8 equivalence.

Candidate output SHA256:
`3a4e1e1dabc16180e0f7d5488697ca8104e181d95a514d61927449f0d15f5a91`.
Q8 AR SHA256:
`fafa084b75946cc5fd0abb713a91fc70f40e2fdfb2fa0be458ef09a9e4d60eb2`.

The candidate reads 726,739,845,120 expert bytes with 56.658 seconds of union
read intervals. Verification takes 88.869 seconds; draft generation takes
2.405 seconds. The native retained run has a different output trajectory and
206 cycles, so this does not isolate Q8 kernel overhead. No repeat or new
optimization tests were added for this losing throughput candidate.

## Cleanup and next work

Both child and guard exit zero. Qwen shutdown automatically reclaims 36.09 GB
and 35.83 GB of cached pages to zero; measured physical reductions are recorded
separately. Exact Qwen identity, health and warmup precede each lock release.
The final release is 22:26:03 UTC; independent verification at 22:26:42 confirms
healthy idle Qwen, warmup complete, no owned child, and a free lock.

The user also requires at least 256K KV capacity, secondary to 20 TPS. The
existing configurable Q8 store prices 262,144 tokens at **552,567,168 backing
bytes**, with a **2,952,790,016-byte additional reserve**. This CPU accounting
is not a complete 256K prefill bound: layer-major retained views, hidden states,
attention temporaries and draft seeding still need an explicit envelope.
No full 256K model run is authorized by the 16K envelope.

`archive-manifest.json` preserves raw receipts and exact harness/helper sources.
The next primary work must materially reduce expert-read or target-verification
cost; do not repeat this unchanged Q8 throughput run.
