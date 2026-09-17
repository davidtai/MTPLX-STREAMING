# Post-prefill cache growth: 12.1147 TPS under 110 GB

The exact 16,384-input / 1,024-output Python workload improves from 11.6575 to
12.1147 decode TPS (+3.922%) when persistent expert banks grow from 93 to 100
rows after prefill. The 1.9541-second resize is included in decode wall time.
**The 20 TPS goal remains open.** This is a one-request benchmark installation,
not a general-serving capacity controller.

| Complete workload | Fresh control | Cache-growth candidate |
|---|---:|---:|
| Prefill slots per layer | 93 | 93 |
| Decode slots per layer | 93 | 100 |
| Decode wall seconds | 87.7549858750 | 84.4431145420 |
| Decode TPS | 11.6574572920 | 12.1146644762 |
| Verification seconds | 85.1949380510 | 79.9720211650 |
| Expert records read | 39,093 | 35,880 |
| Expert bytes read | 734,973,419,520 | 674,566,963,200 |
| MLX full-run peak bytes | 95,208,120,648 | 95,208,120,648 |
| MLX post-prefill peak bytes | 88,755,206,968 | 94,018,544,659 |
| External sampled process phys_footprint peak | 95,495,444,960 | 95,784,385,944 |
| External sampled machine physical-used peak | 105,181,265,920 | 105,642,098,688 |
| Reclaimed-machine baseline bytes | 8,723,234,816 | 8,985,821,184 |

The measurements are distinct: allocator usage is not process footprint or
machine physical use. Machine use includes file cache. Sampled peaks can miss
shorter spikes; admission also prices allocation/copy peaks, host reserve,
cache retention and wired headroom. `records_read` counts expert records;
`read_operations` counts native read operations and is not interchangeable.

Both runs use source `d4051aeccc938a65a71734f1379e00f69f0d1823`, native BF16 target
head, D5/M6, 48 shared transients, no prefetch, fanout 4, three-record miss parts,
transition-window admission and shared-work overlap. HC compile, attention
compile and window memo are explicitly bound false. The promoted post-only
compiled prefill HC callable is unchanged. The fresh control agrees with the
preceding 11.6513-TPS cap93 run; this receipt contains one full candidate and one
fresh full control, not a repeated distribution of candidate timings.

All 1,024 candidate token IDs match the control digest:
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
Both retain the indexed AR `tie_flip` at 297, zero AR contested margin,
consistent captured rows and a matching capture index. The AR stream and
hashed diagnostic row are reused references; public AR timing and memory are
null. They are not current-run AR performance measurements.

## Allocation and ownership

`phase_growth.py` installs a transition after the original prefill callback
starts the decode timer. It drains pending routes, I/O completions and Metal
consumers before copying one component at a time. Every existing row and the
bank object identity survive. New slots begin empty; policy history survives.
The new allocator closes over the new immutable plan. Pool slots, policy
capacity, transient indices, configuration and device LUTs change together.
The enabled decode route receives no new checks, counters or fallback.

The growth adds 5,264,179,200 payload bytes. MLX active storage grows by
5,263,196,160 bytes because the scale components lose 983,040 bytes of page
padding across forty banks. MLX active memory is 86,497,096,948 bytes before the
transition and 91,760,293,108 afterward. The transition peak is 92,347,544,824.
The engine plan grows from 90,194,844,488 to 95,459,023,688 bytes while the
allocator limit stays 98,866,695,168 bytes. The 2 GiB host reserve remains
separate. Requested allocator cache is 1 GiB; admission conservatively allows
3,331,897,468 bytes for cache plus a large retained buffer. The candidate's
whole-machine admission bound is 108,757,001,708 bytes.

An initial tiny probe exposed that `mx.eval` plus releasing an exported view
can leave the old input live until Metal synchronization. The final native
six-component 2-to-3-row probe preserves two distinct row hashes, verifies the
new zero row and accounts for 16 KiB allocation pages. Closing leaves 8 bytes
active. Its 512 MiB allocator cap and 2 GiB child cap avoid loading the model
while establishing this storage behavior. No broad test suite was run.

A second prefill is rejected by this benchmark installation. A partial resize
failure aborts generation and closes the runtime; it cannot silently execute
with mixed capacities. General serving still needs a shrink/reload protocol
before a subsequent prefill. Do not expose this installation as an unrestricted
serving feature or reuse its bounds for different shapes or resident artifacts.

## Reporting correction and provenance

The original candidate already reports the actual 100-slot plan under
`dspark.serve_stream_counters.slot_plan`. Its top-level `resolved_plan` was
constructed before DSpark and records the initial 93-slot plan. The experimental
`phase_memory_plans.decode` alias incorrectly copied that earlier header.
`growth-v1-report-correction.json` derives the correct phase plans from the
recorded DSpark plan; the original receipt stays immutable. The revised wrapper
uses the pass-specific plan and labels the initial header. Its actual assignment
AST was applied to the successful receipt on CPU, preserving all memory samples,
limits and timing. `comparison.json` uses that corrected phase mapping.

`run_full.py.growth-v1` and `installation.json.growth-v1` are the exact measured
candidate sources. `run_full.py` is the reporting-corrected fresh-control
wrapper. The implementation difference after generation is receipt stamping;
both execute the same fixed control arithmetic. Full JSONL receipts and OS
samples are gzip-compressed without content changes. The manifest authenticates
all archived files. Commands retain their original local paths; reproduce with
the pinned source and named local model, compact-MTP and hashed AR artifacts.
The bounded probe additionally uses `/tmp/dsv41-expert-record-0.bin`, whose hash
and source manifest are recorded in `native-record.json`.

Both guards exited zero, automatically reclaimed stopped Qwen file pages and
restored the exact `mtplx-flash-next-optimized-speed` service. Candidate health,
background warmup and lock release completed at 14:21:02 UTC; fresh control at
14:27:13 UTC. Live API identity and warmup were checked afterward. No other GPU
owner was interrupted. Swap did not increase during these runs.

## HC follow-up: not promoted

The existing small-M HC compilation lane was screened separately with exactly
93 prefill/100 decode slots. A bounded native `[1,6,4,5120]` float32 probe showed
1.6269ms eager versus0.7167ms compiled median, with10.20MB versus9.22MB allocator
peak. It was not bit-exact: maximum post-output difference0.000369728. The source
comments now correctly state that the seven-row cap does not guarantee native
bit identity; an unchanged module AST proves this is only a comment correction.

The full screen reports12.2191TPS versus12.1147TPS, but its first difference
against the retained MTP control is at480 (tokens944 versus45706). The existing
AR tie at297 does not classify that additional difference. No paired logit
classification exists at480, so this is not evidence of either an allowed tie
or a non-tie error. The candidate also uses204 versus206 cycles, and verification
time per cycle is essentially unchanged. The apparent gain is insufficient to
justify more diagnostic runs; HC compile remains disabled in the retained arm.

`followup-hc-screen.json` records the exact sources, probe, rejected result and
comparison limits. The full guard exited4 on output-digest rejection, restored
and warmed the exact Qwen service, then released the lock at14:40:47 UTC. Live
health, identity and lock release were verified. The immutable rejected output
is included for audit; it is not a successful optimization receipt.
