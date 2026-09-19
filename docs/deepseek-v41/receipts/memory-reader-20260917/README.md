# Memory reader and optimization screens, 2026-09-17

The retained full-workload result remains **12.1146645 decode TPS**. The 20 TPS
goal is open. This stage improves process-memory sampling and fixes a truncated
kernel response; it does not establish an inference throughput improvement.

## Accepted process reader

The process reader now binds its Mach functions and ctypes signatures once.
Every observation still allocates a fresh result and calls `TASK_VM_INFO` for
the current task. No memory value or process port is cached. A successful short
response that omits `phys_footprint` now returns unknown instead of reporting
the untouched zero in the result buffer. RSS never substitutes for footprint.

The CPU A/B/A uses the previous reader extracted from pinned source
`8f30fc002f3af27e0fec5afd56c522d3abf1da90`, with 100 observations per arm:

| Arm | Median cost per process observation |
| --- | ---: |
| Control before | 24.167 us |
| Candidate | 1.125 us |
| Control after | 22.1875 us |

This is about 20.6 times cheaper for this small reader, not a decode speedup.
A touched 16 MiB allocation immediately increases measured process footprint
by 16,809,984 bytes. The previous reader reports zero footprint for a successful
rev0 reply; the candidate reports unknown. The installed SDK confirms the
38-word rev1 prefix and the field offsets. The focused reporting suite passes
27 cases in 0.34 seconds with real MLX imports forbidden. Three cases were
added only after the CPU optimization succeeded.

See [the reproducible CPU screen](accepted-process-reader/screen_process_reader.py),
[its result](accepted-process-reader/process-reader-production-screen.json), and
[the focused checks](accepted-process-reader/cpu-process-reader-tests.log).

## Rejected whole-machine reader

Calling `host_statistics64` directly appeared roughly 700 times cheaper than
launching `vm_stat`, and schema/ABI checks passed. Freshness checks reject it:
after a touched 16 MiB allocation the direct counters remain identical, while
platform `vm_stat` sees 18,530,304 additional physical bytes and the process
reader sees 16,809,984 additional footprint bytes.

Apple's kernel rate-limits third-party `host_statistics64` callers and can
return cached counters; platform binaries bypass that limit. See
[Apple's implementation](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/host.c#L703-L749).
The whole-machine reader therefore remains `vm_stat`. The candidate source,
patch, successful schema checks and disqualifying freshness result are retained
under `rejected-system-reader`; none of that implementation is installed.
The initial attempt to include the older memory-profile suite was stopped by
the import guard because that file imports MLX at collection. No real MLX was
initialized by the CPU checks.

## Rejected cache-policy screens

The causal CPU replay uses the captured 206-cycle native M6 route, a 100-cycle
selection window and the following 106 cycles held out. All arms start with
the captured 73-slot bank plus 27 empty slots. This older initial state is not
an exact replay of the retained 93-to-100 growth run.

The existing policy reads 35,981 records (19,471 selection, 16,510 held out).
None of eight online logistic configurations improves the selection window.
The optimistic full-history best of 35,889 reads does not qualify for promotion.
Relaxing current-hit protection after service causes zero promotions/copies and
leaves reads unchanged. Neither screen warrants runtime code or a GPU benchmark.
The scripts, complete per-layer results and summary are in `rejected-cache-policy`.

## Guarded profile and its limits

The first cap93 attempt is refused before model loading because the current
baseline does not fit its established bound. The cap91 attempt then stops before
model loading because the diagnostic wrapper's extra 128 MiB host reservation
disagrees with the runner's allocator-policy calculation. A CPU arithmetic check
catches the mismatch before the corrected launch. Both windows restore Qwen.

The corrected native D5/M6 profile uses 91 prefill slots and 100 decode slots,
16,384 input tokens and 1,024 total output tokens. It completes 206 cycles with
the retained output digest. Instrumented throughput is 10.7873 TPS and is not a
performance candidate. The static physical bound is 109,431,449,068 bytes,
including the extra profiler reservation; sampled internal machine usage peaks
at 106,117,201,920 bytes. MLX peak is 94,018,428,478 bytes and sampled process
footprint peaks at 95,730,302,360 bytes. These distinct peaks must not be added.

The cProfile attribution is unusable in this Python 3.12.13 environment: worker
calls appear despite the intended main-thread scope, with three entries whose
self time exceeds cumulative time. A CPU thread/sleep reproduction demonstrates
the inconsistency without MLX. Do not use the raw profile's thread-scope label,
call counts or function times to choose an optimization. The native generation
receipt and OS observations remain diagnostic evidence, not throughput evidence.

The child exits zero; exact Qwen identity, health and warmup are restored before
lock release at **17:01:33 UTC**. No owned GPU job remains. No unrelated process
was signaled. Guard logs, raw receipts, wrapper versions and the profile rejection
are in `diagnostic-profile`. Next profiling work needs independently validated
thread attribution or explicit timing boundaries; do not repeat this cProfile run.
