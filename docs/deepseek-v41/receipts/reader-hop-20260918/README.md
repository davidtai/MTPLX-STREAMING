# Miss-worker reader screen: not promoted

One native miss part can use the installed noncontiguous plane batch reader
and execute its fill on the already-running miss worker at the native future
wait. This removes an executor hop without changing native slot admission,
publication, pinning, failure drain, GPU fences or expert arithmetic. The
candidate batches only the existing at-most-three-record part. It runs after
the native completion-error lock has been released.

The layer-34 screen uses the same 206 saved M6 routes and physically restores
the captured 73 residents before adding 37 empty slots. It is not the current
hybrid M6/M8 trajectory or the full prefill's 84-slot state. Source is
`85c7a33fd9f171c3847d7e59f8566b8b72de7225`. Both arms keep 110 persistent and
48 transient slots, part size three, fanout four and the attested strict
256 MiB allocator cache. Native runtime source hashes match the prior operator.

Five interleaved arms give summed warm latencies of 1.239440, 1.209235,
1.221840, 1.207971 and 1.224357 seconds. Candidate/control median ratio is
**0.9871322**, a 1.2868% reduction against **1.4375% control spread**. Every
output and physical read count matches in all 206 calls per arm. Each arm's
warm region reads 734 records. Hashing happens outside each call's timer;
these are summed operator latencies, not continuous full-model throughput.

The small gain does not justify a full-model run or a default change. No new
regression tests are added. The retained full result remains 13.4141518 TPS;
20 TPS is still unmet.

The complete incremental bound is 9 GiB: 5 GiB Metal/cache/compile and 4 GiB
host/reader/compiler, with one runtime at a time and no added GPU owners or
threads. MLX peak is 3,047,281,161 B; final active allocation is 8 B. Guard
samples peak at 3,371,060,584 B process footprint and 15,722,168,320 B machine
physical usage over 14 samples, with zero compressor growth. Those overlapping
memory measures are not added.

Guard 33095 exits 0. Candidate source pages are reclaimed before exact Qwen
restoration; model identity, health and warmup pass before release at
10:55:23 UTC. Independent model/health/idle/warmup/free-lock verification passes
at 10:55:55 UTC. The stage script and all measured helpers/results are retained;
the model artifact is referenced by hash and is not copied.
