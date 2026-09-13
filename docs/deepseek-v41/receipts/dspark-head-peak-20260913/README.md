# Real-weight draft-only measurement

Source b64ef4966, same native MXFP4 artifact. Only the three MTP stages and
shared embedding/output head were constructed: 10,597,621,640 parameter bytes.
No target layers or expert cache were loaded. The guard held the GPU lock,
reclaimed Qwen's clean file cache, admitted a 10.048 GB baseline, and restored
the exact warmed Qwen service at 12:03:47 UTC before releasing the lock.

The static loading bound was 25,249,119,226 bytes; measured MLX loading peak
was 15,415,845,136 bytes. Lazy construction used 1,190 active bytes before strict
checkpoint replacement. The touched shards contained only 42,496 discarded
bytes and the largest shard was 2,406,563,050 bytes.

Seeding all three windows from a synthetic 16K main-hidden tensor took 154.7 ms
and added 2,473,893,804 peak active bytes. Warm eager five-row draft latency was
10.30 ms; compiled latency was 9.36 ms (median of three). Cold eager allocation
added 469,763,987 bytes, including the 402,653,184-byte persistent wo_a caches.
The subsequent cold compiled call added 18,638,211 bytes and took 292 ms to
compile and execute. Eager and compiled draft IDs agreed.

Sampled physical peak was 23,646,502,912 bytes. Swapouts stayed at 4,399,765.
These are isolated draft allocation/timing results on synthetic hidden values,
not a full-model throughput or output-parity result. The measured seed/draft
increments inform the next full-workload memory admission.
