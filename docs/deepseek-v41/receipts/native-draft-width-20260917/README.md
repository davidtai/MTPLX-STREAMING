# Native draft-width screen and runner corrections

The exact 16,384-input / 1,024-output Python workload reached **11.4315253 decode
TPS** with D7/M8. All output tokens match the retained native MTP digest, but
this screen does not beat the retained **12.1146645 TPS** result. D7 is not
promoted; the 20 TPS goal remains open. No additional GPU comparison was run.

| Full workload | Prefill/decode cache slots per layer | Cycles | Decode seconds | Decode TPS | Physical expert records read |
|---|---:|---:|---:|---:|---:|
| Retained D5/M6 | 93 / 100 | 206 | 84.443115 | 12.114664 | 35,880 |
| Screened D7/M8 | 91 / 98 | 176 | 89.489371 | 11.431525 | 40,607 |

The live baseline was 10,776,903,680 bytes. Admission selected 98 decode slots
under the unchanged 110,000,000,000-byte whole-machine ceiling. Capacities and
prefill states differ between the rows, so this is not a matched estimate of
the draft-width effect. The candidate still fails the absolute throughput
screen. It reads 763,437,588,480 physical bytes and spends 84.985017 seconds in
verification, versus 674,566,963,200 bytes and 79.972021 seconds for the retained
run. Fewer draft cycles alone did not produce a faster full workload.

The candidate uses native BF16 target arithmetic, compact MTP banks 93/58/32,
48 shared transient slots, no prefetch, transition-window policy, fanout4,
three-record miss parts, and shared overlap. HC compile, attention compile and
window memo are explicitly false; the existing compiled prefill post-MoE HC
combine is unchanged. Source: `7537ead7ab6ae53b0cbec8fb78a225c65b37c4bc`.

Full output SHA256:
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
The separately indexed AR tie at token 297 remains accepted. Reused AR timing
and memory measurements are null; current DSpark measurements are independent.

## Memory and lifecycle

| Measurement | Bytes | Decimal GB |
|---|---:|---:|
| Full-run MLX allocator peak | 93,704,070,260 | 93.704070 |
| MLX peak after prefill, including resize | 92,532,465,178 | 92.532465 |
| Internal sampled process phys_footprint peak | 94,293,064,008 | 94.293064 |
| Internal sampled machine physical peak | 105,582,788,608 | 105.582789 |
| External 250 ms sampled process phys_footprint peak | 94,506,621,600 | 94.506622 |
| External 250 ms sampled machine physical peak | 105,610,772,480 | 105.610772 |

These are distinct measurements. Do not add process usage to machine usage or
call the allocator peak total RAM. Machine physical usage includes file cache.
The admission bound was 109,314,275,908 bytes, with 2 GiB reserved for host state,
separate allocator-cache overshoot, page padding, a 256 MiB allocation margin,
and an additional entire 17,664-token logical compressed-KV allowance. Detailed
bounds and phase plans are in [results.json](results.json).

Cache growth completed from 91 to 98 slots. Its 2.000506 seconds are included
in decode time. Engine budgets were 88,690,793,288 and 93,954,972,488 bytes;
the allocator policy limit remained 97,075,612,672 bytes. These limits are not
usage measurements.

All GPU children ran through the direct exclusive guard. Qwen shutdown and
clean-file-cache reclamation were automatic. Every window restored exact model
identity, health and background warmup before releasing the lock. The final
child exited 0; restoration and release completed at 16:15:08 UTC. Health,
model identity and free lock were then independently verified. No unrelated
process was signaled.

## Screening evidence

- `teacher/`: full native D5/M6 teacher capture at cap84. It matches all 1,024
  tokens and 206 cycles. The initial admission/host-reserve mismatch and the
  later NumPy BF16 bridge error are preserved as setup failures; neither was an
  OOM. Exact native BF16 bits are stored as uint16, while prefill state stays
  FP32. The 32,428,672 bytes of tensor files are preserved locally under ignored
  `benchmarks/raw/deepseek-v41-depth-teacher/20260917/`; their hashes and original
  dtypes are archived, rather than committing the binary tensors.
- `draft-replay/`: draft-only replay against the captured target trajectory.
  Compact D5 reproduces every reference commit boundary. Compact/full widths
  5, 7, 9, 13 take respectively 206/206, 176/176, 178/179, 218/232 cycles.
  Compact/full D7 have equal totals but different individual boundaries. This
  replay screens acceptance; it is not target execution or a throughput result.
- `m8-probe/`: native target verification at cap16 with the exact 16K prompt
  and 129 output tokens. Prefix identity, depth7 and verify chunk8 are confirmed.
  Allocator peaks are 38,444,125,800 bytes during prefill and 30,847,218,871 bytes
  after prefill. This probe bounds native M8 allocations before the full run.
- `full-depth7/`: immutable measured wrapper, helpers, installation hashes,
  command and raw receipts. The actual block size and all three draft stages
  are constructed at seven; changing only requested depth would still cap at5.

## Runner fixes after measurement

The A/B runner printed the top-level AR reference even in DSpark mode. With AR
reuse, this displayed `decode_tok_s=None` and unavailable memory despite valid
DSpark measurements. Its cross-arm comparison could likewise compare matching
AR references while overlooking different DSpark outputs. The runner now uses
the requested measured pass for throughput, memory, token identity and actual
decode-phase budget. It keeps AR reference measurements null. The current
24-case CPU memory/reporting suite passes with real MLX imports forbidden; the
previous source fails the new regression. The measured receipt itself is not
rewritten. Its corrected display is in `runner-fixes/corrected-headline.txt`.

The experimental resize callback also sat inside the model's intentionally
nonfatal telemetry callback. A resize exception could therefore be swallowed.
Both retained-growth and width-screen wrappers now propagate such failures as
a fatal exit through the existing runtime cleanup. They also reject receipts
whose growth phase or actual decode capacity differs from admission. A focused
CPU check uses the actual callback control flow: the old failure returns, the
fixed failure aborts with its cause, and success is unchanged. No new GPU run
was needed for these reporting and failure-path changes.

`runner-fixes/` contains the installed corrected wrappers and CPU evidence.
The immutable measured wrappers remain under `full-depth7/` and the prior
growth receipt directory. Rebuild installation commit/source hashes before
future GPU execution; the wrappers intentionally refuse stale provenance.
Growth remains a one-request benchmark installation. Another prefill requires
physical shrink or reload before it can be used in general serving.
