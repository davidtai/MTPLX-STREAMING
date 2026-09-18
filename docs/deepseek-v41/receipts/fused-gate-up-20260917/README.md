# Shared-input gate/up projection: rejected

Combining gate and up preserves all four whole-MLP output byte sequences but
does not yield a useful general improvement. The common six-row case is
16.62% slower. No runtime installation, full-model run or new tests follow.

| Rows / unique experts | Packed control, ms | Combined gate/up, ms | Latency reduction |
|---|---:|---:|---:|
| 6 / 6 | 0.417730 | 0.487167 | -16.62% |
| 18 / 3 | 0.591875 | 0.589209 | +0.45% |
| 36 / 12 | 1.230646 | 1.298792 | -5.54% |
| 36 / 36 | 1.329688 | 1.293833 | +2.70% |

The candidate shares each input-vector load between gate and up. Both retain
native V16/R4/two-SIMD-group reductions, independent packed scales and BF16
projection outputs. Clamp/SwiGLU and native V8 down are unchanged. The probe
uses 36 real layer-20 experts in nonidentity physical slots, synthetic BF16
inputs and interleaved controls. Reference bytes come from the native raw-scale
MLP before releasing its scale backings. These are whole-MLP operator timings,
not full-model throughput or GPU-kernel-only timings.

The 6 GiB incremental bound covers the 48-slot bank, one packed-scale layer,
small tensors and host/compiler/cache space. Allocator peak is 928,978,000
bytes and explicit cleanup leaves eight active bytes. The guard samples
1,363,969,656 bytes of process footprint and 11,578,392,576 bytes of machine
physical usage. Known candidate source/scale pages are zero after cleanup.

Guard 87930 is terminal exit 0. Exact Qwen identity, health and warmup returned
before lock release at 02:17:57 UTC on September 18; independent verification
at 02:19:39 UTC found it healthy, idle and warmed, with the lock free and no
owned child. Source: `57fb1b2ac110e8deebbe02a19b659671165eac81`.
`sha256.json` binds the raw results, exact scripts, bounds and lifecycle.
