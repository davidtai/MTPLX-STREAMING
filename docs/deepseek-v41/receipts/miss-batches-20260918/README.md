# Smaller miss batches: operator gains, negligible full-run difference

The exact 16,384-input / 1,024-output Python workload completes at
**12.826718 TPS / 79.755398 seconds**, with all native output IDs unchanged.
The previous three-record result is 12.809105 TPS / 79.865062 seconds at the
same 84-to-104 cache capacity. The 0.1375% difference is too small to establish
a reliable full-model improvement. Keep the existing default; **20 TPS remains
unmet**. Do not rerun unchanged candidates to fill a testing matrix.

## Bounded operator selection

The native 104-slot policy is replayed on the saved 206-cycle M6 routes without
MLX. Layer 34 is selected before timing as the upper median by total policy
miss count: 819 misses, averaging 3.98 per route, with a maximum of 15. The
physical operator restores all 73 captured warm residents into their actual
slots, adds 31 empty persistent slots and retains 48 transient slots. It uses
all 206 native routes chronologically, actual native expert weights and packed
scales, synthetic BF16 inputs and the same synthetic shared-work callback.
It does not run attention, a full decoder or a native shared-expert MLP.

Only the construction-bound maximum records per miss part changes. The native
cache policy, physical assignments, gate/up and down arithmetic, output order,
full-record READY publication, slot leases and deferred releases are preserved.
The candidate installer changes its declared supported batch size; there is
no added hot-path validation, fallback or engagement instrumentation.

Each comparison runs five interleaved arms. Totals below exclude the first four
calls of each arm; output/read checks cover all 206 calls. Every candidate has
identical output bytes and identical physical reads at every call. Each warm
arm reads 794 records across its 202 timed calls.

| Comparison | Control median warm total | Candidate median warm total | Reduction | Control spread |
| --- | ---: | ---: | ---: | ---: |
| Part3 to part2 | 1.336512 s | 1.303814 s | 2.45% | 0.73% |
| Part2 to part1 | 1.304618 s | 1.290174 s | 1.11% | 0.73% |

These separate-window operator gains are not a combined full-model speedup.
Both use an 8 GiB incremental bound: 4 GiB for MLX/cache and 4 GiB for Python,
readers and compiler space. Raw banks require 2,857,697,280 bytes before scale
retirement; packed banks require 2,689,597,440 bytes. MLX peaks at 2,934,477,321
bytes for all arms and returns to eight bytes after final teardown. Controllers
reclaim source pages only after actual child exit; both end with zero cached
source pages. Guard machine peaks are 14,442,364,928 and 13,681,836,032 bytes,
with 14 complete one-second samples each and zero compressor growth.

After the operator wins, the existing two CPU reader-lifetime checks pass for
successful and failed down reads. They verify early gate/up publication,
full readiness waiting for down completion, and writer-view release after
terminal completion. No broad test suite is added.

## One full run

The full candidate composes the already-measured native KV16 memory setup,
tail2048 MTP seeding, packed output-projection owner retirement, bounded Engram
arenas and one-record early-plane reads. It retains native D5/M6, all 48 shared
transients, fanout four and transition-window policy. All 11 pinned runtime
source files match the preceding full result; only staged paths and the miss
batch construction choice change.

CPU admission agrees exactly with the preceding memory composition at two
baselines. No memory credit is taken for smaller read batches. The original
64 MiB plane-root/scheduler reserve covers at most 48 part witnesses and less
than 2 MiB of assignment payload. Native prefill, copy, cache, KV, compiler,
wired and Python allowances remain unchanged.

| Full-run measurement | Bytes |
| --- | ---: |
| Live baseline | 10,157,899,776 |
| Whole-machine admission bound | 109,591,219,432 |
| MLX active bound | 94,523,220,076 |
| MLX measured peak | 94,075,749,060 |
| Internal process-footprint peak | 95,822,742,208 |
| Guard process-tree peak | 95,862,817,472 |
| Internal machine physical peak | 107,047,895,040 |
| Guard machine physical peak | 107,051,008,000 |

These overlapping memory measures are not additive. The guard records 222
complete one-second samples and zero compressor growth. The measured machine
peak is below both the admitted bound and 110,000,000,000-byte ceiling.

All 1,024 tokens match native SHA
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`;
206 cycles execute. The separate AR comparison still classifies index 297 as
the allowed tie flip, using fresh candidate logits and the cached native AR
row. No new AR timing or memory measurement is claimed.

The full run reads the same 34,259 records / 606,203,412,480 bytes as part3.
Read-interval union rises from 46.996676 to 47.148010 seconds. Verification
falls from 74.160687 to 73.880078 seconds, while charged growth rises from
3.358249 to 3.480915 seconds. These scopes overlap and must not be summed.
Complete decode saves only 0.109664 seconds; different background and time
windows prevent attributing that small difference to the candidate reliably.

Source `ac7a15c0dac9749531cc220270243064e7a356f1`; full guard session68394 exits0.
Exact Qwen health, identity and background warmup precede lock release at
05:36:19 UTC. The independent 05:36:51 check finds healthy, idle, warmed Qwen,
no owned child and a free lock. No other job was signaled. Fixed Q8 and complete
256K prefill validation remain secondary and are not advanced by this result.

## Reproduction and next decision

`part2/` and `part1/` contain operator commands, proofs, exact outputs and
lifecycle receipts. `full/` retains the full command, source audit, CPU checks,
unchanged source snapshots and raw receipts. `sha256.json` covers every archived
file. Packed artifact manifests are included; their existing 3.09 GB payload
is reused rather than copied. Reproduction requires the named local artifacts
and guarded environment; source-pinned helpers must be audited and refreshed
before a new run. The measured wrapper's unsupported-input message still says
two or three records; its actual construction check correctly requires one.
Correct that diagnostic in any future copy, without rewriting this receipt.

The next CPU investigation is the phase-specific inactive allocator-cache
allowance. The current reserve is 3,331,897,468 bytes. Existing growth already
clears the cache, and endpoint measurements alone cannot justify discounting
that reserve. Derive a hard bound from the installed allocator and allocation
lifetimes before changing admission or loading another full model.
