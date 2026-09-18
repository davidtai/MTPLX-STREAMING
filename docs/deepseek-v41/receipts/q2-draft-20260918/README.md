# Affine 2-bit draft screens

The user asked to use the strategy from Vontra's 2-bit DeepSeek artifact.
These bounded screens apply ordinary affine 2-bit/group64 requantization
to the existing compact draft. Target-weight conversion remains a separate
scope question because it changes outputs beyond the earlier tie allowance.
No target weights, production defaults or full-model benchmark are changed.

The expert-only variant remains a conditional memory prototype. It saves
1,416,683,520 B of weight payload but increases teacher-trajectory verification
calls and warmed head time. The broader mixed-dense variant is rejected for
its larger acceptance loss. Neither establishes an end-to-end speedup or
justifies a full run on this evidence. Best retained full TPS is 13.4141517619;
20 TPS and complete 256K prefill remain unmet.

## Exact scope

All 183 experts in the existing 93/58/32 compact draft retain their source
identity and route lookup. Each projection is dequantized from native
MXFP4/group32 to BF16 and requantized with MLX affine2/group64 at construction.
Only one expert projection is materialized densely at a time. The resulting
U32 codes plus BF16 scales/biases occupy 2,023,833,600 B instead of
3,440,517,120 B. No weights are exported or downloaded.

The pre-existing compact lookup maps nonresident draft IDs to slot0; this
table is unchanged in both arms. This is an approximate compact draft, not
the full128-expert/stage DSpark head. The native target remains the proposed
verifier, but these screens do not execute that target.

`dense/` additionally converts 21 query/output and shared-FFN projections,
saving another 343,670,784 B. Main projection and KV-building weights remain
identical, so the saved initial KV is valid and all committed-KV seeding uses
the native arithmetic. Routers, norms, Markov/confidence heads and shared
token embedding/output head also remain native. This mixed variant does not
reproduce Vontra's complete checkpoint quantization.

## Teacher-state results

The exact saved16K/1024 native target states drive independent draft
trajectories. Future output IDs only score completed proposals; committed
target states seed the next cycle. Native hybrid control reproduces all198
saved boundaries. Every arm has a fresh cache; repeated codec trajectories
are identical. No target parity or full TPS follows from teacher acceptance.

| Draft | Calls | Verification rows | Warm head time, s |
| --- | ---: | ---: | ---: |
| Native hybrid control | 198 | 1,242 | 1.889598 / 1.894284 |
| Q2 experts | 201 | 1,266 | 2.323908 |
| Q2 experts plus selected dense | 233 | 1,456 | 2.634387 |

The expert-only head remains about22.83% slower than the mean warmed controls;
the difference is not just first-use compilation. Each repeated candidate
has one warmed timing, so no broad statistical or full-runtime claim is made.

## Capacity sensitivity

A2.495-second CPU replay blocks MLX imports and reproduces all53,999 recorded
reads at the original73-slot frequency policy. The first attempt wrongly used
transition-window for this identity gate; the second omitted the native
resident-hit fast path, whose ordered touches affect recency. Both failures
are retained; the corrected helper follows the established replay sequence.

On the same fixed206-cycle native route trace, the current transition-window
policy uses32,033 reads at110 slots,31,274 at112 and30,921 at113. Two extra
packed slots per layer cost1,415,577,600 B and reduce this proxy's traffic by
759 records /13,430,292,480 B /2.3694%. This is not a full admission proof:
remaining allowance is small, and the actual Q2 trajectory, complete prefill,
live background, kernel peaks and runtime descriptors are not modeled.

The proxy uses actual73-slot initial residents expanded with empty capacity.
It is neither the198-cycle full hybrid nor the201-cycle Q2 route trace. It
cannot establish a net throughput result. Combined with the added head work
and calls, it provides weak grounds for another full-model run.

## Memory and lifecycle

Measured source: `4f3af25aeed6a37c7297db50d8987f214cc2ee37`. Both head screens
retain the complete49GiB incremental bound:41GiB active,4GiB cache and4GiB host.
The unused7,219,445,760-byte full native expert bank is retired before Q2
conversion. Native plus Q2 compact banks, stack/copy and one-matrix conversion
buffers remain within the earlier stacked-bank allowance. Only one draft
owner executes at a time. Native target trunk and streaming banks never load.

| Screen / terminal guard | Sampled process peak, B | Sampled machine peak, B | Samples |
| --- | ---: | ---: | ---: |
| Experts /16023, exit0 | 11,902,248,248 | 26,196,803,584 | 10 |
| Mixed dense /48370, exit0 | 14,412,975,920 | 26,195,804,160 | 20 |

MLX peak is21,299,586,448 B in both screens; compressor growth is zero.
Metrics overlap and must not be added. Both codecs coexist for comparison,
so payload savings are not a measured reduction in whole-machine peak.

Each guard reclaims15,342,174,208 B of source pages to zero, restores exact
Qwen identity and warmup, and releases its lock. First release15:50:50UTC;
an immediate independent check encounters an active service request, which
is left alone. Healthy/idle/warmed/free verification succeeds15:54:56UTC.
Final release15:58:15UTC; independent verification succeeds15:59:42UTC.
No owned child or queued window remains. No new regression tests follow these
unpromoted candidates.
