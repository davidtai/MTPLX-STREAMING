# DeepSeek V4.1 decode read attribution

The explicit boundary profile identifies expert-read waiting as the largest
exposed cost. This is diagnostic evidence, not a promoted optimization.
The retained uninstrumented result remains **12.1146645 TPS**; **20 TPS is unmet**.

## Exact workload and timing

Source `7953020cda67ccda296a729d01a03adb9cd1ae4b`, native D5/M6 target arithmetic,
16,384 Python input tokens, 1,024 output tokens (1,023 timed decode steps).
Compact MTP residents remain 93/58/32. This run uses 84 prefill slots per layer,
charged growth to 98 decode slots, 48 shared transients, transition-window
admission, fanout 4, three-record miss parts and shared-work overlap.
HC/attention/window compile flags are false; prefill post-MoE HC stays compiled.

| Main-thread boundary | Exclusive seconds |
| --- | ---: |
| Wait for missing experts and finish their completion bookkeeping | 52.271397293 |
| `mx.eval`: graph encoding and GPU waiting combined | 19.940573891 |
| Target forward outside the separately timed child boundaries | 3.247913215 |
| Streamed switch outside its separately timed child boundaries | 1.781569689 |
| `mx.async_eval` submission | 1.750893300 |
| Cache policy transaction | 1.549629408 |
| Route preparation/read submission outside cache policy | 1.215634304 |
| All other measured boundaries and root remainder | 2.914156900 |
| **Complete decode-loop root** | **84.671768000** |

Expert graph construction accounts for only 0.350480397 seconds within the
last row. The switch's `mx.eval` calls account for 16.911346441 seconds of the
total evaluation time. Evaluation timings are not GPU-kernel-only timings.
Shared MLP callbacks were bound before instrumentation, so their work remains
inside the containing switch/MoE timings; absence of a separate row does not
mean that overlap was inactive.

The runtime read 36,783 records, or 691,543,941,120 bytes. Its union of read
intervals is 53.198112937 seconds, about 13.0 GB/s. The software interval
concurrency metric is not hardware queue depth. Read waiting alone exceeds
the 51.15-second whole-decode budget needed for 20 TPS.

Full charged decode is 86.706967042 seconds / 11.798359865 TPS, including
1.890526041 seconds of cache growth and MTP seeding. The full output hash is
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`, identical
to the retained MTP result. Different cache capacities and instrumentation
prevent treating this as an unchanged throughput control.

## Attribution and memory checks

The timer accepts only calls from its owning main thread. Nested scopes subtract
child duration, and generator timing covers each `next()` call rather than work
between yields. Exceptions balance the stack. The CPU proof excludes a running
foreign thread, separates consumer time, and measures 323.14459 ns per flat call.
Exclusive durations sum exactly to the root duration, with no negative values.
This replaces the previously rejected cProfile attribution.

| Memory measure | Bytes |
| --- | ---: |
| Post-Qwen whole-machine baseline | 10,668,638,208 |
| Conservative physical bound, including extra 128 MiB profiler reserve | 109,126,510,060 |
| Full-run MLX allocator peak | 92,514,505,646 |
| Internal sampled process `phys_footprint` peak | 94,214,126,064 |
| Internal sampled whole-machine physical-used peak | 105,473,327,104 |
| Independent sampled process `phys_footprint` peak | 94,286,199,280 |
| Independent sampled whole-machine physical-used peak | 105,505,128,448 |

The independent 882 samples report zero swap-in/out counter growth. Machine
usage includes file cache; none of these overlapping measures are added together.
A preceding cap-91 attempt refused before model loading at its higher baseline.
Both guards restored the exact Qwen model, health and background warmup before
release; the successful profile released its lock at 17:35:36 UTC.

## Candidate screens

The CPU-only prompt-lookup screen selects proposals from the prompt and already
committed output, using the earliest or latest match of the longest 2–16-token
suffix. Future teacher tokens score acceptance only. Even its best standalone
configuration needs 575 cycles versus native D5's 206; it is rejected without
target execution. This does not measure a hybrid lookup/MTP decoder.

One bounded draft-head-only window then replays the saved exact target states
using the existing native confidence function. The untrimmed control reproduces
all 206 recorded commit boundaries.

| Confidence threshold | Cycles | Target verification rows proposed |
| --- | ---: | ---: |
| Disabled | 206 | 1,236 |
| 0.25 | 206 | 1,221 |
| 0.5 | 209 | 1,142 |
| 0.75 | 229 | 1,071 |
| 0.9 | 263 | 1,054 |

The 0.5 candidate removes 7.61% of verification rows with 1.46% more cycles.
These are acceptance estimates on saved target states, not target execution,
physical-read savings or throughput evidence. The draft-only window's MLX peak
is 21,299,586,448 bytes against a conservative 41 GiB active plus 8 GiB host/cache
bound. Qwen restoration and lock release completed at 17:48:02 UTC; independent
health, exact model identity, background warmup and free-lock checks succeeded.

## Full confidence candidate: rejected

The 0.5 threshold completes the same input/output lengths at 84 prefill and 99
decode slots, yielding 12.121569555 TPS / 84.395011333 seconds. It performs 217
cycles and 1,188 verification rows, rather than the head-only screen's 209 and
1,142. Its output differs from retained MTP at 297 and from AR at 376. The
reference cache has no AR logits at 376, so that additional difference is
unclassified. No tie proof from index 297 applies at index 376.

This is not a throughput winner: the raw rate is only 0.057% above the retained
best, with different cache capacities and an unclassified output. Read traffic
is 34,783 records / 653,942,661,120 bytes, with 50.448693034 seconds of union read
intervals. A changed output trajectory prevents attributing those read savings
solely to confidence trimming. The candidate is not installed, and no further
GPU control or new optimization test is justified by this result.

The physical bound is 108,774,057,452 bytes at a 9,698,377,728-byte baseline.
MLX peak is 93,267,045,850 bytes. Independent samples peak at 95,028,247,792 bytes
of process footprint and 105,654,009,856 bytes of whole-machine usage. There are
zero new swapouts and 32 swapins; do not describe this as zero swap activity.
The guard exits 4 on the output gate, restores Qwen and completes warmup, then
releases the lock at 17:56:43 UTC. The independent post-run health receipt
confirms the exact model, zero active requests, warmup done and a free lock.

An initial attempt failed before model loading because the staged installation
mistakenly stored the wrapper hash in `phase_memory_control_sha256`. The original
measured-receipt hash was restored and three CPU admission calculations checked
before retry. The failed metadata and guard log remain archived.

## Runner diagnostic correction

The rejected run exposed a reporting error: absent AR logits became
`rows_consistent=false`, followed by the assertion that the row did not produce
its credited token and the difference was not a tie. Missing evidence proves
neither assertion. Missing/empty rows now report `class=unclassified`,
`rows_consistent=null`, and the unavailable side. The runner also retains the AR
replay error in its receipt. The strict tie acceptance gate is unchanged.

The existing focused CPU reporting suite passes 33 cases with real MLX imports
blocked, covering missing AR, missing DSpark, empty and both-missing rows, a
proven tie and a proven non-tie. A source-based CPU reproduction shows the old
incorrect label and new unknown label. Matching expectations in two model test
files were updated but those modules were not executed. An AST audit confirms
all generation arithmetic is unchanged; only the diagnostic classifier and
post-run reporting functions differ. No further GPU window was used for this fix.

`artifact-manifest.json` identifies immutable raw receipts, scripts and hashes.
Large JSON/JSONL files are losslessly gzip-compressed with their original hashes
and sizes retained. The retained best remains 12.1146645 TPS.
