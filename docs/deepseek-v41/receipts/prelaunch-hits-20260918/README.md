# Early cached-expert submission screen

Rejected: submitting cached-expert computation before CPU route planning was
1.21845% slower in this bounded layer replay. No full-model run or regression
tests follow from this result. Retain the native runner.

Measured source: `19ea3ac2f888edf5035e3a43bc314bea64f3cb6f`.
The candidate submits the existing indices first, then native masked cached
gate/up/down work using a construction-owned expert-to-slot lookup. CPU route
planning waits only on the original indices event. Miss kernels, protection of
requested resident slots, publication, pinning and failure cleanup remain native.
The masked kernel only adds a signed slot and uniform negative-slot early return
before the unchanged native dot body. Native MLX 0.32.2 event behavior was checked
in its local source. No predictor, prefetch ring, new worker or extra I/O exists.

Five interleaved arms use actual layer34 routes from all206 native cycles,
110 persistent slots and48 shared transients. The captured initial state has73
residents and37 empty slots. Inputs are synthetic BF16; router IDs are already
evaluated outside the timer. Every per-call output and physical-read count
matches. Each warm arm reads734 records.

| Arm | Sum of warm call times, ns |
| --- | ---: |
| Native before | 1,230,981,091 |
| Prelaunch 1 | 1,240,504,410 |
| Native middle | 1,223,549,415 |
| Prelaunch 2 | 1,248,092,133 |
| Native after | 1,229,319,623 |

Median candidate/control ratio:1.0121845029. Control spread:0.60454%.
Hashing gaps are outside the timer. These sums are neither continuous decode
latency nor full-model throughput.

The9GiB incremental bound includes5GiB Metal/cache/compile and4GiB host/reader/
compiler space. MLX peak is3,047,281,161B; final active allocation is8B. The guard
samples3,539,306,968B peak process footprint and15,222,833,152B peak machine
physical usage across14 samples, with zero compressor growth. These distinct
measures are not added together.

Guard36067 exits0, source clean pages are reclaimed to0, exact Qwen identity and
warmup are restored, and the lock releases at11:11:46UTC. Independent health,
idle, warmup and free-lock verification passes at11:13:13UTC on2026-09-18.
The command, source pins, construction audit and raw outputs are retained here.
