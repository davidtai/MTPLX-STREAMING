# Exact BF16 input rows in a fixed host cache

The candidate keeps the native input table for prefill, then replaces its
1,323,827,200-byte Metal owner with a fixed 16 MiB host row arena before expert
bank growth. A 32 MiB allowance covers that arena, bounded LRU metadata, row
reads and output copies. The untied output head and Markov embedding stay on
their native paths. KV remains native 16-bit for the throughput workload.

Construction verifies the exact native `[129280,5120]` BF16 tensor, artifact
identity and sole model-tree ownership. Reclamation and row-cache cleanup are
registered before the model can change. At the completed prefill fence, the
candidate verifies the active-memory drop and removes clean source-shard pages
before spending the released capacity. Prefill receives no source-memory credit.
Resize, MTP seed and steady decode receive the exact table credit. All other
native KV/compiler/copy/wired reserves remain. Plans retain the original source
reserve and are labeled separately from actual owners and measured usage.

The cache reads only requested token rows, with uncached positional I/O on the
native shard. A returned tensor uses an independent small output copy, never a
view of an evictable arena slot. The draft path makes one shared-embedding call
per cycle for a primary token and four fixed noise tokens. The target path uses
the same row cache. No future token trace or oracle preload is used.

The bounded operator loads only the 1.324 GB table under an 8 GiB incremental
envelope: 4 GiB Metal/cache/compiler and 4 GiB host/source/I/O. M1/M5/M6 outputs
match native BF16 bytes in cold and warm arms. Native repeat medians are
201/211/210 us; file-cold medians 59/66/81 us and warm medians 53/59/61 us.
Native first-arm M1 includes compilation and is not used as the control median.
Cold and warm refer to the row arena; the source could retain older clean OS
pages during this small screen. The full candidate explicitly reclaims those
pages before decode. These are isolated costs, not a full-model speedup claim.

The operator releases exactly 1,323,827,200 active bytes and ends at zero Metal
owners. A held output stays byte-identical after more than an arena of eviction.
Capacity is 1,638 rows. MLX peak is 1,323,958,296 B. The short child is caught by
only one guard memory sample, so that sample is not a reliable physical peak;
the complete static envelope and MLX peak are retained explicitly. Guard 66617
exits 0, reclaims source pages and restores exact Qwen/warmup/free lock; the
independent check is at 09:58:58 UTC.

The first full attempt reaches prefill completion and stops before expert-bank
growth because the new reclamation call passes `str` instead of the helper's
required `Path`. The same wrong type also appears in final cleanup. It is not
an OOM. Guard 9584 exits 1 and restores exact Qwen/warmup/free lock at
10:06:13 UTC; independent verification passes at 10:06:37 UTC. Both arguments
are corrected in `full-v2`; failed `full-v1` evidence is preserved unchanged.

Measured source: `575c3c8b3beb0420d16fc03c727f3a27c0f36edd`. The strict MLX
allocator and packed expert arithmetic are the retained control's pinned builds.

## Full candidate: new best single result

The corrected `full-v2` completes 16,384 prompt tokens and 1,024 generated IDs
in 206 D5/M6 cycles. All IDs equal the retained native stream:
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
It reaches **13.3195300 TPS / 76.8045117 s**, versus the retained strict run's
13.1509467 TPS / 77.7890768 s: 0.9845652 s less wall time, 1.2819% higher TPS.
This is a new best single result. Capacity and background differ, so it does
not establish an isolated speedup or repeatability. **20 TPS remains unmet.**

The live baseline is 10,681,696,256 B. Admission selects 84 prefill and 110
decode slots per layer, with 48 shared transients. Host reserve is
1,405,218,816 B, allocator limit 97,913,084,928 B, active bound 97,494,304,988 B
and whole-machine bound 109,849,655,516 B. The native prefill envelope remains
intact; only post-prefill phases receive the table credit.

Measured MLX peak is 96,998,277,438 B, process footprint peak 98,018,134,304 B,
and machine physical peak **109,255,884,800 B**. Guard samples independently
peak at 98,017,167,648 B process and 109,230,161,920 B machine. There are 218
guard samples and zero compressor growth. Final inactive allocator cache is
256,835,618 B within its 268,435,456-byte limit.

At the transition the actual table release is exactly 1,323,827,200 active bytes,
and 1,320,714,240 clean source-page bytes are separately reclaimed to zero.
Retirement costs 0.131795 s; packed bank growth costs 3.624598 s. Both are inside
the reported decode wall time. The row arena finishes with 313 of 1,638 slots
occupied. Expert reads are 31,936 records / 565,098,577,920 B: 380 records and
6,723,993,600 B less than the retained run. Read-union time is 43.8395 s; it is
not GPU idle time. Draft time is 2.1114 s and verify time 70.6366 s.

Guard 26008 exits 0, reclaims source pages, restores exact Qwen and warmup, and
releases at 10:11:42 UTC. Independent healthy/idle/warmed/free-lock verification
passes at 10:11:58 UTC. No owned GPU child remains. Two focused host regressions
are added only after this full win: duplicate ordering/eviction/output ownership
and close, plus short-read rejection without publishing corrupt rows. Both pass
with real MLX imports blocked; actual Metal/BF16 equivalence is established by
the guarded operator and full output evidence. No broad suite or unchanged
full control was run. Fixed Q8 and full 256K prefill validation stay secondary.
