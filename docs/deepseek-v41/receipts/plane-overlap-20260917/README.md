# Packed expert plane overlap: 12.673 TPS full candidate

The exact 16,384-input / 1,024-output Python workload completes at
**12.6731624 decode TPS / 80.7217623 seconds** (1,023 timed decode steps),
including 3.4263 seconds for the post-prefill packed-scale installation and
bank resize. **20 TPS remains unmet.** The candidate uses native KV, D5/M6,
compact native MTP residents 93/58/32, 48 shared transient slots, transition-window
admission, three-record miss parts, and fanout four.

All 1,024 token IDs match the retained native MTP stream, SHA
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`,
in the same 206 verification cycles. The first difference from the separate AR
reference is still index 297: the fresh candidate logits and cached, index-matched
AR row classify it as `tie_flip` (contested margins 0.0/0.25, tie band 0.75).
Current-run AR speed and memory remain unmeasured.

This is the fastest single complete result so far, 1.84% above the previous
12.4439935 TPS. That is a historical comparison: both have 102 decode slots,
but prefill capacity is now 84 rather than 91 and the background differs.
It does not establish an isolated or repeatable full-model speedup. The lane
remains an explicit experimental harness, with its source archived here; it
is not installed as a general serving default.

## Change and ownership

A miss can start gate/up computation after those two weight planes finish,
while the down-plane read completes. The arithmetic, packed kernels, scale
owners, true expert IDs, physical bank indices and final assignment order are
unchanged. Slot READY publication still waits for every component writer.
The existing runtime iterator owns policy commits, leases and deferred release.
Down computation starts only after that iterator yields a complete ReadyRoute.

Installation occurs once, at the quiescent post-prefill boundary. It binds a
single-wave decode path for at most eight rows and retains 48 transient slots.
There is no enabled-path eligibility fallback, worker-thread MLX execution,
new engagement counter or second weight bank. The full wrapper only permits
one request, native KV and the fixed D5/M6 workload. It rejects a later prefill.
Cleanup drains submitted GPU work and slot transactions, clears bound runners,
and then releases packed owners. This is not a multi-request serving design.

The first real-runtime probe exposed an incorrect assumption about thread-local
context: the slot pool submits reads to another executor. Its safe failure is
retained under `integration/failed-first-handoff`. The corrected lane carries
the part witness across both executors and serializes concurrent readiness
publication. No full model was loaded for the failed probe.

## Minimum bounded checks

The initial coupled reader/MLP probe uses three real layer-20 experts and exact
packed weights/scales. Early computation reduces median latency by 5.42–5.58%
across 3, 9 and 18 assignment rows. Reordering reads but waiting for all planes
is flat or slower. Every sampled output matches the original native kernel bits.

The real Runtime/SlotPool/Switch integration uses eight persistent slots and
48 transients in one layer. It compares stock, specialized full-ready and early
readiness in seven interleaved blocks, with deterministic M1/M6 routes and shared
work. All 168 outputs match; all arms read the same record counts.

| Rows | Stock median, ms | Full-ready specialization, ms | Early GU, ms | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9.354188 | 9.340698 | 9.041969 | 3.34% |
| 6 | 18.534000 | 18.571406 | 18.154031 | 2.05% |

These are medians of block medians, excluding each shape's first two calls.
The integrated probe has an 8 GiB incremental envelope, including a 2 GiB Metal
policy and its 256 MiB allocator cache. Measured Metal peak is 1,130,140,169 B;
all ownership releases to eight active bytes. The guard observes a
1,531,152,016 B child footprint and 11,759,386,624 B machine peak.

After the operator win, two focused CPU lifecycle cases check successful and
failed down reads: GU can publish early, full readiness waits for down, and all
writer views release only after terminal completion. No broad test suite or
additional full-model control was run.

## Full-run memory and I/O

The live baseline is 9,955,393,536 B. Admission selects 84 prefill / 102 decode
slots at a **109,891,934,440 B whole-machine bound**, within the unchanged
110,000,000,000 B ceiling and 100 GiB wired limit. This retains the original
Python, KV, cache, padding, copy and compile allowances, adding 64 MiB for early
plane intermediates and scheduler state. Actual extra assignment payload is
under 2 MiB; no old allowance was discounted.

| Measurement | Bytes |
| --- | ---: |
| MLX allocator peak | 94,044,288,356 |
| Internal sampled process phys_footprint peak | 95,804,685,552 |
| Guard sampled child footprint peak | 95,812,009,200 |
| Internal sampled whole-machine physical peak | 106,288,578,560 |
| Guard sampled whole-machine physical peak | 106,287,955,968 |

These overlapping measures must not be added. The guard has 223 complete
samples at a one-second cadence. The wider phase sampler observes 2,372 swapins
and no new swapouts; zero swap activity is not claimed.

Decode reads 620,943,114,240 weight bytes / 35,092 records, plus 3,086,136,060
scale-installation bytes. Read union is 47.9346 seconds, verification 74.8540
seconds and drafting 2.0636 seconds. These time scopes overlap; read union is
not a measure of GPU idle time. Further progress still requires a material
reduction in expert-read or verification cost.

## Restoration and subsequent machine memory

The full child and guard exit zero. Exact Qwen identity, health and background
warmup precede lock release at 23:19:34 UTC; an independent check at 23:19:54
finds Qwen healthy and idle, no owned candidate child and a free lock.

A later Qwen-resident machine sample reaches 132,913,250,304 B, outside the
DeepSeek window. A guarded cleanup-only follow-up ends at 23:25:05 and finds
**zero cached DeepSeek resident or compact-MTP pages**. Qwen shutdown itself
reclaims 22,129,917,952 cached bytes to zero; the stopped-service baseline is
10,303,307,776 B. After restoring Qwen, the 23:25:58 sample again reaches
131,652,091,904 B while Qwen reports healthy, warmed and idle.

This evidence does not support blaming retained candidate pages or changing
DeepSeek's cleanup based on that hypothesis. It distinguishes the enforced
110 GB DeepSeek window from the larger restored Qwen service footprint. No
Qwen configuration or unrelated job was changed, and no whole-machine 110 GB
cap is claimed for Qwen's normal service. The operational guard remains unchanged.

## Evidence and continuation

Measured source: `ee72b77e0cdbc33db4208e08a8883f3a631d3067`.
`summary.json` and `archive-manifest.json` retain results, hashes and original
paths. Raw JSONL streams, commands, source, failed integration evidence and
all restoration logs are included. The 3.09 GB packed artifact is retained in
its existing ignored location; it was not copied or exported again.

The live candidate is `/tmp/dsv41-plane-overlap-20260917/full/command.sh`.
Its source proof is intentionally pinned to the measured commit. A later run
must audit source changes and use a fresh output prefix. Do not overwrite this
receipt or rerun an unchanged full candidate merely to fill a testing matrix.
Q8 and complete 256K prefill support remain secondary to the 20 TPS objective.
