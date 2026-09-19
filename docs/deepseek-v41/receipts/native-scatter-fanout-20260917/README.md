# Native component-scatter read fanout screen

Fanout eight is not promoted. The retained complete 16K-input/1K-output result
remains **12.1146645 decode TPS** with fanout four. The 20 TPS goal remains open.
No production runtime code changes in this screen.

The full-model candidate was refused before model loading. Its measured
baseline was 12,208,914,432 bytes, versus 8,985,821,184 in the retained run.
Admission included 512 MiB of extra host/kernel allowance for 20 additional
fanout workers; their measured aggregate stack capacity is 335,790,080 bytes.
The resulting conservative whole-machine bounds were 111,672,157,440 bytes
for prefill and 112,516,965,868 bytes for decode, above the 110,000,000,000-byte
ceiling. The allocator limit was also below the admitted prefill requirement.
No decode result exists for this refused arm. These are bounds, not usage peaks.

## Bounded I/O comparison

One control/candidate/control screen used the production component-scatter
`os.preadv` path, F_NOCACHE, three concurrent records and the exact six native
component lengths. It allocated only 56,401,920 bytes of CPU destination buffers
under a 2 GiB child limit. MLX was not imported. Each arm read 1,536 records,
28,877,783,040 bytes, after four warmup batches. Final landed record hashes
match the manifest; existing error counters are zero.

| Arm | Fanout | Wall seconds | GB/s | Python preadv calls |
|---|---:|---:|---:|---:|
| Control before | 4 | 2.3341569580 | 12.3718257 | 6,144 |
| Candidate | 8 | 2.3243794170 | 12.4238680 | 12,288 |
| Control after | 4 | 2.3309262910 | 12.3889731 | 6,144 |

The candidate gains 0.351% bandwidth against the controls' mean wall time;
the controls differ by 0.139%. That small screening gain, doubled read calls
and larger thread allowance do not justify another full-model run. This is
an I/O-only screen, not a decode speedup: CPU bytearrays do not reproduce Metal
page ownership, model-cache pressure, inter-layer scheduling or GPU overlap.
Read-interval concurrency is software instrumentation, not hardware SSD queue
depth. The native scalar extension is unused by this scatter path.

Maximum sampled process phys_footprint is 290,063,512 bytes; maximum sampled
machine physical use is 11,695,079,424 bytes. Samples can miss shorter peaks.
The 2 GiB probe allowance and 110 GB machine guard are separate limits.

## Provenance and lifecycle

Sources are pinned to `769122f7e04fe2f53c36ae727547546ec392ed9e`. The full-model
wrapper preserves the validated arithmetic and fixes cache capacity at
93 prefill/100 decode, all three bound compile/memo flags false. The current
model executable AST is identical to the measured d405 predecessor. Original
absolute paths in archived scripts identify the exact temporary installation.

The initial I/O probe incorrectly assumed 128 main-layer experts and exited
before expert reads. The native manifest has 40 x 384 = 15,360 main records;
128 applies to the MTP layers. The corrected v2 inventory was checked directly
against the manifest. Both sources and the failed setup log are retained.

All three owned windows terminated. Qwen was restored with exact model ID,
health and completed background warmup, then the lock was released at
14:59:08 UTC (full-model refusal), 15:04:20 UTC (probe setup failure), and
15:06:03 UTC (completed I/O screen). Automatic stopped-service file-cache
reclamation ran in every window. No unrelated process was signaled.

`summary.json` separates the refused arm from the completed screen;
`cpu-scatter-results-v2.json` contains raw metrics and memory snapshots.
`manifest.json` hashes every archived evidence file. No regression test suite
was added for this unpromoted experiment.
