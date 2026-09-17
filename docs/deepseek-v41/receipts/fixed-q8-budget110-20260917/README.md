# Fixed Q8 KV and 110 GB budget

The decimal and legacy GiB budget interfaces now share **110,000,000,000
bytes**. The staged native/packed admission helpers use that same physical
ceiling. Baseline, Python, allocator cache, transient copies and wired-memory
limits still apply; 110 GB is the whole-machine ceiling, not a model allowance.

On the recorded 11,254,906,880-byte baseline, removing the packed helper's
additional 109.5 GB cutoff admits 100 slots/layer at a 109,708,761,320-byte
bound, instead of 99 at 109,000,972,520 bytes. At the exact 110 GB boundary it
admits 100; one byte more reduces capacity to 99. This is CPU accounting for
the previous native KV path, not a new throughput or peak-memory measurement.
The old inherited receipt ceiling field was misleading; the archived proof
separates that field from the outer helper's actual cutoff.

## Explicit Q8 configuration

Both benchmark runners and `load_deepseek_v41_streaming` accept Q8. The settings
for the active workload are:

```text
--box-target-gb 110 --max-kv 17664 --kv-cache-bits 8 --kv-max-append 953
```

`--kv-cache-bits 16` retains the native control and remains the general default.
Q8 is an explicit lossy precision choice; full-model quality and throughput have
not been measured. Existing native growth wrappers have stale source proofs
and native cache bounds. They require a new complete allocation bound and Q8
reference before a full-model Q8 run; refreshing hashes alone is insufficient.

Affine Q8 groups contain 64 values with FP32 scales and biases. Target window,
compressed and index KV and all three draft windows use fixed-capacity packed
buffers. Native compressor working rows remain FP32 in fixed rolling buffers.
All banks are allocated and evaluated when the request cache is constructed.
Rollback rewinds packed state without requantization. Factories and append
routes avoid Python reference cycles that otherwise retain Metal allocations.

For 17,664 tokens, a 953-row maximum append, window 128 and eight-row rollback:

| Fixed store | Bytes |
| --- | ---: |
| Target windows | 50,181,120 |
| Compressed KV | 25,436,160 |
| Index KV | 6,359,040 |
| Native compressor working rows | 26,763,264 |
| Three draft windows | 3,763,584 |
| Total backing storage | **112,503,168** |

The loader reserves **503,316,480 bytes before expert-bank allocation** for
backings, replacement copies, decoded views and slack. Existing native KV
allowances are not discounted. Packed capacity stays fixed; temporary decoded
attention views still vary within that reserve. This does not assert constant
allocator addresses or constant whole-process usage.

## Verification

The final bounded guard child passed in 1.0094 seconds without full-model
weights. Forty target layers and three draft caches kept exactly 112,503,168
backing bytes through 16,384-row prefill and six-row verification/rollback.
Native compressor output matched exactly; window, compressed and index Q8
matched an independent quantization reference. Packed snapshot restore,
full-prompt draft seeding and native draft detach also passed.

A real four-layer model with small dimensions and random weights completed
prefill plus 12 verify/trim cycles with finite logits and fixed storage. Its
32 prefill greedy choices matched native storage; relative logit RMS error
was 1.743%. This is an integration check, not released-model quality evidence.
Random real-width row quantization error was 0.540% relative RMS.

MLX allocator peak was 274,186,240 bytes. Active memory after teardown and
cache release was 975,688 bytes, down from 116,670,464 bytes retained by owner
cycles in the preceding storage probe. The external guard sampled a
10,815,127,552-byte machine peak; the child's final snapshot was
10,819,698,688 bytes. Sampling and allocator peaks have different scopes.
No new swapouts occurred during the probe.

The focused CPU run passed 45 cases with real MLX imports forbidden, including
Q8 reserve-before-allocation and existing memory reporting. The legacy default
budget assertion also passed in isolation. Stale test doubles were updated to
include the current construction-time attention resolvers.

The first development probe failed on a missing `slice_update` axes argument;
the second validated storage but exposed delayed cache reclamation. Their
sources, receipts and recovery evidence are retained. No OOM occurred.

## Service and machine state

The final guard automatically reclaimed 36,539,088,896 cached-page bytes from
stopped Qwen files, all to zero; measured physical-use reduction was
36,472,471,552 bytes. File cache and physical usage are reported separately.
Exact Qwen identity, health and warmup were restored before lock release at
21:37:12 UTC. Independent verification at 21:37:40 found a free lock, healthy
idle service and no remaining probe children.

The machine audit found no abandoned DeepSeek worker. The other larger users
were active Qwen, VS Code and a Colima VM running `ndh-runner` and
`qwen36-webstatus`. Those active jobs were preserved. RSS was not substituted
for process `phys_footprint` to choose termination targets.

`source-manifest.json` pins changed runtime/test files. `archive-manifest.json`
pins the raw evidence; original measurements were not rewritten. The full
20 TPS goal remains open; the best complete run is still 12.4439935 TPS.
