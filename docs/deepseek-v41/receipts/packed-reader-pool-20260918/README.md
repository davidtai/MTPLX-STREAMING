# Packed-plane reader pool screen

Reducing the auxiliary FIFO pool from 15 workers to 6 is not promoted. The
median latency reduction is 0.5031%, with 0.2677% control spread; the second
candidate essentially ties its neighboring controls. This does not justify a
full-model run or new regression tests.

The five warm totals, in native/candidate/native/candidate/native order, are
1,226,451,418; 1,212,232,534; 1,224,250,171; 1,223,949,127; 1,223,174,371 ns.
All 206 layer-34 outputs and physical read counts match in every arm. Each
warm segment reads 734 records. The replay uses synthetic BF16 inputs, actual
73 captured residents plus 37 empty slots, 110 persistent and 48 transient
slots. It sums call timers with hashing between calls; it is not continuous
decode or full-model TPS evidence.

Native `io_read_fanout=4` creates 15 auxiliary workers, using
`min(64,max(4,(1+4)*(4-1)))`. The installed packed lane sends three whole
5,898,240-byte plane reads per physical record. Replacing the executor after
seeding changes concurrency only. This differs from the earlier fanout-8
experiment, which also changed split-range geometry. Plane readiness, slot
publication, pins, cleanup, cache policy and kernels are unchanged.

The complete incremental allowance is 9 GiB: 5 GiB Metal/cache/compile and
4 GiB host/reader/compiler. MLX peak is 3,047,281,161 B. Per-arm cleanup retains
76,780,040 B of shared scale owners; final cleanup leaves 8 B. The guard samples
3,537,684,784 B process-tree footprint and 15,156,690,944 B machine physical
usage across 14 observations, with zero compressor growth. These overlapping
measurements remain separate; a limit is not a usage measurement.

Measured source: `61b6899f27fee89f63bf7eed9f1c8aa77e73a53a`. Guard 32996 exits 0,
reclaims source pages, restores exact Qwen identity and warmup, and releases
the lock at 11:59:30 UTC. Independent health/idle/warmup/free-lock verification
passes at 12:01:28 UTC. Helper/runtime hashes, command and raw evidence are
retained. No production default changes.
