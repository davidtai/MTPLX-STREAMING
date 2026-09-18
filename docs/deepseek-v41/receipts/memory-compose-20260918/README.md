# Composed memory savings: 12.8091055 TPS at 104 decode slots

The exact 16,384-token Python prompt and 1,024 output tokens complete in
**79.8650617 seconds / 12.8091055 TPS**, including 3.3582486 seconds of cache
growth. All output IDs match the native control; 206 verification cycles run.
This is 1.0727% faster than the previous best, 12.6731624 TPS / 80.7217623s.
It is one complete candidate at a different live baseline and capacity, not
an isolated kernel comparison or repeatability claim. **20 TPS remains unmet.**

The candidate composes three bounded storage choices:

- Retain only the final 2,048 prompt hidden rows for native MTP seeding. The
  existing operator and full run establish exact target outputs and native
  draft state. The full cap101 phase comparison saves 880,803,840 active bytes
  before and after growth. This credit applies to growth only; the original
  prefill envelope remains and steady decode receives no tail credit.
- Retire packed target `wo_a` owners after original first-use BF16 transpose
  construction. Its separate full run proves 1,384,120,320 fewer steady bytes.
  The steady bound retains one 34,603,008-byte cold source; prefill, growth and
  seeding receive no projection credit. The output arithmetic is unchanged.
- Bound each native Engram row arena at 119,537,664 bytes. Full payload,
  256 bytes of metadata per maximum row and 1 GiB of other Python capacity
  require 1,544,647,680 bytes. An additional 32 MiB prices the two helpers.
  Actual native arenas and row capacities are checked at construction.

The lower live baseline admitted **104**, not the initially reported 103,
decode slots per layer after 84-slot prefill. The complete receipt is the
authoritative capacity record. Admission preserves the original KV, cache,
compiler, padding and wired allowances and the 110,000,000,000-byte ceiling.

| Measurement | Bytes |
| --- | ---: |
| Measured baseline | 10,467,377,152 |
| Whole-machine bound | 109,900,696,808 |
| Active MLX bound | 94,523,220,076 |
| Measured MLX peak | 94,075,778,388 |
| Internal process-footprint peak | 95,852,314,944 |
| Internal whole-machine peak | 107,528,060,928 |
| Guard process-tree peak | 95,824,118,080 |
| Guard whole-machine peak | 107,546,591,232 |

These overlapping measurements are separate, not additive. Every observed
phase peak remains within its admitted envelope. Final MLX active memory is
93,922,915,764 bytes: after correcting for two extra slot bands, 1,384,124,416
bytes below the earlier native control. The 4,096-byte difference from the
separate projection ownership result is retained as measured variation.

Expert reads fall from 35,092 / 620,943,114,240 bytes to
34,259 / 606,203,412,480 bytes. The union of active reads falls from
47.934568s to 46.996676s. These intervals do not measure GPU-idle time.
Both Python caches finish with zero evictions: 180,226 and 180,228 resident
rows, each within its 452,794-row capacity.

The native hidden tensor is now measured directly at the existing seed
boundary: `[1,2048,15360]`, FP32, **125,829,120 bytes**. The older tail helper's
62,914,560-byte metadata assumed BF16. This candidate overwrites that nominal
field with the observed dtype and allocation. The prior admission never used
that nominal field for a discount; source and measurement archives remain
immutable.

Measured source: `5ab177659880abaed7c528e9ef1f65898ee7470b`.
Guard session54606 exits0 after221 complete memory samples, with no compressor
growth and source/packed file cache ending at zero. Exact Qwen identity, health
and background warmup precede lock release at04:31:30UTC on September18.
Independent04:34:07UTC verification finds healthy/idle/warmed Qwen, a free
lock and no owned child. The full archive retains all helpers, CPU admission
checks, raw JSONL and output receipts, cleanup and lifecycle records with
SHA-256 coverage. No additional model run, broad test suite or general-serving
default is introduced.

Native KV16 remains the throughput path. Fixed Q8 KV and the 256K geometry are
separate work; this result does not claim a 256K prefill run.
