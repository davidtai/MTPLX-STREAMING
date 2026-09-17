# Exact resident packed scales, 2026-09-17

The complete 16,384-input / 1,024-output Python candidate reaches **12.4439935
decode TPS in 82.2083358 seconds**, with the same 1,024 token IDs as the retained
native run. **20 TPS remains unmet.** Source is `4a5acf9d4776bbe2a6a0519a4f1fc4b8965274dc`.
The [summary](summary.json) identifies the exact receipts and hashes.

This is one-request benchmark installation. Prefill uses the native route;
after its fence, the installer replaces raw scale storage, resizes the weight
banks once, and binds the packed-scale decode kernels and weight-only reader.
Another prefill requires a reload. Installation takes 3.4511023 seconds, all
charged to decode. Slot generations, pins and physical bank identities survive
the transition. The enabled decode callable executes directly.

| Measurement | Packed candidate | Historical native run |
|---|---:|---:|
| Decode TPS | 12.4439935 | 12.1146645 |
| Decode wall seconds | 82.2083358 | 84.4431145 |
| Prefill / decode slots per layer | 91 / 102 | 93 / 100 |
| Physical expert records read | 35,058 | 35,880 |
| Expert payload bytes read | 620,341,493,760 | 674,566,963,200 |
| Expert read union seconds | 47.8115610 | 51.6387630 |
| Charged installation / resize seconds | 3.4511023 | 1.9541090 |

The candidate also reads **3,086,136,060 bytes of packed scales** during its
charged installation. Those direct reads are separate from expert-reader
counters: combined installation and expert payload is 623,427,629,820 bytes.
The historical relative TPS difference is +2.718%; this is **not a fresh paired
comparison**, and the initial cache capacities differ. Another job owned the
GPU lock after this window, so a fresh native control was not launched.

Both runs use native BF16 target arithmetic, compact MTP banks 93/58/32,
native D5/M6, 48 shared transients, no prefetch ring, transition-window admission,
three-record miss parts and shared-work overlap. The candidate changes each
physical record read from 18,800,640 to 17,694,720 bytes by reading only its three
native weight planes. The source sidecar and FP4 codes retain their format.
Software read concurrency is 10.9616; it is not hardware queue depth.

The complete output SHA256 is
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
The existing indexed AR tie classification also passes. Text sidecar names use
the AR digest to pair streams; their primary file explicitly contains the
DSpark stream and its own complete digest in the header.

Memory remains separated by ownership and measurement:

| Quantity | Bytes |
|---|---:|
| Post-Qwen whole-machine baseline | 9,613,836,288 |
| Conservative whole-machine admission bound | 109,483,268,328 |
| Whole-machine ceiling | 110,000,000,000 |
| Python/host reserve | 2,147,483,648 |
| MLX allocator limit policy | 98,238,680,064 |
| Full-run MLX allocator peak | 94,044,462,082 |
| Internal sampled process phys_footprint peak | 95,847,005,496 |
| Internal sampled whole-machine physical-used peak | 106,215,473,152 |
| Guard sampled process peak | 95,770,262,840 |
| Guard sampled whole-machine physical-used peak | 106,198,253,568 |
| Decode physical weight-slot storage | 73,043,804,160 |
| All resident packed target scales | 3,086,136,060 |
| Raw scale backing released at transition | 4,078,632,960 |

Peaks come from different sampling schedules and must not be added together.
The guard collected 226 complete child observations at its one-second poll
interval. Its reported compressor delta stayed zero. Bounds retain the prior
native M6/M8 allocation evidence, full logical KV allowance, allocator cache
overshoot, page padding and copy margins. Only exact physical expert-storage
terms change. The measured admission's inherited `capacity_selection` text
still says 96..100; its authoritative selected capacity and actual storage are
102. The live helper's label is corrected separately after preserving this run.

The full inventory verifies every source record SHA256 and reconstructs every
E8M0 scale byte exactly: 15,360 records covering all 288,777,830,400 source bytes.
This verifies complete record coverage; it does not recompute a second whole-file
SHA256. Raw target scales total 16,986,931,200 bytes. Row descriptors select
0/1/2/4/8-bit deltas from each row's minimum exponent, with word-padded payloads
and expert offsets. The [inventory](artifact/manifest.json) hashes all 360 files.
Their 3.086 GB payload is preserved under ignored
`benchmarks/raw/deepseek-v41-resident-scales/20260917`; the live temporary path
points there through a symlink, following a same-filesystem rename.

The run exposed a runner reporting bug: both resolved-plan formatters took the
record size from the original artifact even after slot storage changed. They
now distinguish the current `expert_record_bytes` from
`source_expert_record_bytes` and take transient bytes from the active plan.
The original receipt's slot summary therefore shows stale 18,800,640-byte slots
and 902,430,720 transient bytes. Actual phase owners, allocation bounds and memory
observations are correct. The [derived receipt](full/derived-storage-corrected-receipt.json)
corrects that metadata to 17,694,720 and 849,346,560 while preserving every token,
timing and memory observation. Four new CPU cases fail before the fix;
all 37 focused reporting cases pass afterward with real MLX imports forbidden.
Only the two resolved-plan reporting functions' ASTs change.

Small evidence preceded the full model:

- The 160-record CPU sample round-trips all scale bytes. The full inventory
  replaces its size projection with an exact allocation count.
- The real-weight MLP probe matches every sampled output byte. Packed kernels
  are approximately even in the small case and 5-6% faster in larger cases.
- A three-caller CPU I/O A/B/C/B/A screen measures about 5.93% lower wall time
  with three weight-plane reads. Four split jobs add calls for little gain.
- The storage probe uses persistent-103 and transient-48 physical banks with
  nonidentity slot/expert indices and all 384 experts' packed layer-20 scales.
  Four MLP cases and every weight byte match. Explicit close leaves eight MLX
  active bytes; peak allocator usage is 2,841,412,606 bytes within an 8 GiB bound.

The CPU export exposed a separate lifecycle failure: despite `F_NOCACHE`, it
accumulated source-file cache and invalidated its proposed 20 GiB incremental
machine-memory bound. Its guard physical peak was 76,802,080,768 bytes and its
process peak only 182,256,240 bytes. Qwen restoration timed out with guard exit
10. Reclaiming the verified source invalidated 122,342,653,952 cached-page bytes,
including speculative/free pages; this is not a physical-used reduction of that
size. Exact Qwen identity, idle state, warmup and lock release were verified
after recovery. **Do not reuse this exporter unchanged for another full scan.**

The candidate loader reads directly into final Metal owners, bypasses file cache
and disables read-ahead. Its guarded `finally` reclaims verified source and
packed-artifact clean pages before Qwen restoration; this full run found zero
cached pages remaining. The full candidate's guard exits zero, restores exact
Qwen identity and completed warmup, then releases the lock at 19:39:28 UTC.

An earlier cap-93 attempt was refused before model loading when the live
baseline rose; its evidence is under `v1-admission-refused/`. No ceiling was
relaxed. The successful run uses the already-proved cap-91 prefill envelope.
