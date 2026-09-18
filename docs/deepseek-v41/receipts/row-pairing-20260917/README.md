# Pairing expert rows to share FP4 conversion

The bounded operator is bit-exact in all four measured MLP cases. Pairing
helps the fully paired 18-row case by 21.40%, while small mixed groups lose.
This is an operator result, not a full-model speedup. **20 TPS remains unmet.**

| Assignments | Rows per expert | Native median, ms | Candidate median, ms | Latency reduction |
|---:|---|---:|---:|---:|
| 4 | 1, 1, 2 | 0.752250 | 0.858667 | -14.15% |
| 6 | 2, 2, 2 | 0.372709 | 0.359417 | 3.57% |
| 9 | 3, 3, 3 | 0.430458 | 0.448000 | -4.08% |
| 18 | 6, 6, 6 | 1.035625 | 0.814000 | 21.40% |

Each case uses the same three real layer-20 experts, synthetic BF16 rows,
and the retained exact native packed-scale operator. Timing includes CPU
grouping, input gathers, odd singleton execution and output-order restoration.
Three alternating blocks per arm contain five calls each after compilation.
Medians are compared within each shape; differing cache/clock conditions
make the absolute times unsuitable for cross-shape scaling claims.

The new kernel keeps native V16/V8, four output rows per SIMD group and two
SIMD groups per threadgroup. A `float2` dot shares each FP4 conversion across
two rows using the same physical slot and scale expert ID, preserving each
row's accumulation order. Leftover singletons use the explicit native route.
It changes neither model precision nor physical SSD record bytes.

The saved native M6 trace contains 296,640 assignments, of which 139,196
(46.9242%) can pair. Ideal repeated weight decodes would fall by 23.4621%
before scheduling and register costs. This census does not predict throughput.

The incremental bound was 8 GiB. Allocator peak was 131,863,420 bytes, with
8 bytes active after explicit close. The guard separately sampled a process
peak of 336,479,168 bytes and machine peak of 10,855,268,352 bytes. Guard
9965 exited 0; exact Qwen identity, health and warmup returned and the lock
was released at 01:30:45 UTC on September 18. Independent checks passed.

The next staged check is `.../row-pairing-20260917/integration`: one real
runtime/slot-pool layer, with 32 persistent and 48 transient slots. It pairs
only M6 cache-hit work; misses and the other explicit M routes remain native.
Its stride-four route has 26 experts across 36 assignments, near the captured
trace's 24.24-expert average. An unchanged native lane and the candidate will
run in four interleaved blocks. No full-model run is justified yet.

`sha256.json` binds the 12 operator, census, command and lifecycle files.
