# Packed draft projection with the smaller expert alias table

The combined candidate saves head-only wall time but increases teacher replay
from 198 to 199 target calls and from 1,242 to 1,248 verification rows. It is
not promoted as a full-model speed improvement. This screen never executes
the target, and its expert storage still physically contains the native
93/58/32 rows; the smaller 80/40/24 alias table does not earn a memory credit
in this measurement.

At source `b5f41c51d74a770069048dddd0a644f07886a999`, one native/candidate/native
batch measures 1.8676740410 / 1.7562255420 / 1.8696587080 seconds. Both controls
reproduce the pinned 198 commitment boundaries. The candidate uses the saved
smaller draft table and the previously measured packed FP32 output projection.
The additional target call makes the head-only saving insufficient evidence
for promotion. No full run or new optimization regression test follows.

The screen retains its 49 GiB incremental bound and the 110,000,000,000-byte
whole-machine ceiling. Sampled physical peak is 25,811,795,968 bytes and sampled
child-tree footprint is 14,468,321,072 bytes, with no compressor growth. These
are overlapping metrics, not quantities to add. The guard and child exit zero,
restore exact Qwen identity, health and completed warmup, then release at
16:19:46 UTC. The independent 16:22:04 check sees a subsequent foreign GPU
window and no owned child; this is distinct from the completed restoration.

Original helpers, identities, output rows, reclamation and lifecycle records
are under `head/`; `archive-sha256.json` pins the copied bytes. The separate
native-expert packed-draft projection remains eligible for full integration.
The best complete Q4 result remains 13.8688167379 TPS; 20 TPS is unmet.
