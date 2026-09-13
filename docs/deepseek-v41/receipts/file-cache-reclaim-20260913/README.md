# Stopped-service file-cache experiment, 2026-09-13

Read-only mappings plus `msync(MS_SYNC | MS_INVALIDATE)` recovered
**33,750,155,264 B** of measured physical used memory after stopping the exact Qwen
service. Its model shards, MTP file and ngram table held **33,770,864,640 B** of
cached pages. Physical used fell from **46,876,852,224 B to 13,126,696,960 B**.
The DeepSeek artifact had zero cached pages and offered no reclamation benefit.

This uses the documented [Apple msync API](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/msync.2.html).
The 64 MiB scratch test first verified full SHA-256 equality after invalidation.
Actual artifact descriptors and mappings were read-only; inode, size and mtime
were unchanged for every file. The experiment did not rehash the full artifacts.
Only new, exclusively created report files were written. An independent review
found an initial report-overwrite risk; exclusive creation and canonical path
exclusion fixed it, and four CPU alias/path regression cases passed.

Each diagnostic ran under the exclusive guard with a 1 GiB child cap and the
unchanged 110 decimal GB physical ceiling. One 1 GiB virtual mapping and at most
64 KiB of page flags were live at once. No MLX was imported. Both service windows
restored the exact service, passed independent health/warmup checks and released
the lock. Swapout counts did not increase during reclamation.

`mincore` includes speculative free pages, so its count must not be reported as
physical-used reduction. `vm_stat` before/after supplies the actual reduction.
Restoring Qwen can cache those files again. A larger DeepSeek allocation requires
reclaiming after service stop and then taking a fresh guard baseline in the same
window; subtracting cached-byte estimates from the old baseline is invalid.

These are scoped experiment scripts, not installed runner behavior. No production
service configuration, scheduling, model bytes or wired-memory setting changed.
