# Combined gate/up reads: rejected

Combining two native weight-plane reads into one scatter call does not improve
the relevant read batches. The candidate also reads the intervening 368,640
bytes of gate scales into separate, preallocated scratch. Down reads remain
separate, and the native gate/up readiness and complete-record ownership rules
are preserved. No model or Metal allocator is loaded, and no runtime change,
full-model run or additional optimization tests follow this loss.

The control is an unchanged extraction of `PlanePart` and `bind_reader` from
the best packed-plane installation. Each arm uses the native positional reader,
F_NOCACHE, fanout four with 15 worker capacity, and 128 batches after warmup.
The final weight hashes agree across arms and with complete manifest-verified
native records. Native source identity is checked before and after reading.

| Records per batch | Control before, s | Combined, s | Control after, s | Combined wall change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.205073 | 0.205638 | 0.205951 | +0.06% |
| 3 | 0.543424 | 0.562321 | 0.544537 | +3.37% |
| 6 | 1.039005 | 1.060848 | 1.038917 | +2.11% |

Positive change is slower. The one-record difference is smaller than the
0.43% control spread; the other regressions exceed their control spreads.
The six-record control delivers 13.08 GB/s of useful weight payload, compared
with 12.81 GB/s for the combined candidate. These are CPU-buffer read-batch
measurements, not model throughput or evidence about GPU overlap.

The screen has a 2 GiB incremental allowance beneath the 110,000,000,000-byte
whole-machine ceiling. Destination and validation buffers total at most
127,180,800 bytes; the allowance also covers Python, native manifest metadata,
thread stacks and I/O state. Sampled process-tree peak is 372,606,368 bytes,
and sampled whole-machine peak is 9,923,100,672 bytes. MLX imports are blocked.

The first attempt fails its accounting check because replacing the metric
object after warmup leaves the installed record-reader closure holding the old
object. The retry rebinds after resetting metrics, outside timing, so record
and range counters share an owner. This changes the harness only. The refusal
is retained in `counter-refusal/`; its guard exits 1 and restores/releases at
03:28:15 UTC on September 18. Independent 03:29:21 verification is healthy,
idle, warmed and free, with no owned child.

The completed screen is pinned to source `db5f9dd0d`. Its controller observes
child exit 0, then confirms zero retained source-file cache. Guard session
50127 exits 0, restores exact Qwen identity, health and warmup, and releases at
03:30:33 UTC. Independent verification at 03:31:18 UTC confirms healthy, idle,
warmed Qwen, a free lock and no owned child. Each archive has a SHA-256 manifest.

The best complete 16K/1K result remains 12.6731624 TPS. The 20 TPS goal remains
unmet; another full-model load needs a different, measured improvement.
