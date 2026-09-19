# Coarse CPU attribution for exact native decode

This is a diagnostic, not a throughput candidate. All 1,024 native output IDs
match the retained control. Public decode TPS and wall fields are null in both
the full receipt and pass rows. The observed instrumented run was 77.0591 s at
84-to-108 slots; it does not replace the 13.1509467 TPS performance result.

Seven fixed phase accumulators and 40 attention/40 expert entrypoint accumulators
measure elapsed time, main-thread CPU time and whole-process CPU time. Existing
native operations and seven explicit MLX calls are unchanged; no GPU fences or
tensor owners are added. Removing the timing edits recovers the exact native
function source. The exact Python 3.12.13 clock probe distinguishes a busy worker
from a blocked main thread; empty span cost averages 1.239 us. Its estimated
17,717-span cost is 0.0219 s. The earlier system-Python probe is retained and
explicitly distinguished from the runtime clock validation.

| Scope | Elapsed s | Main-thread CPU s | Process CPU s |
| --- | ---: | ---: | ---: |
| Decode cycles | 73.4290 | 13.5526 | 39.6364 |
| Draft | 2.0333 | 0.5230 | 0.8177 |
| Verify | 71.1591 | 12.9127 | 38.6647 |
| Accept | 0.0646 | 0.0254 | 0.0347 |
| Commit | 0.1559 | 0.0758 | 0.0942 |
| Target forward | 70.4559 | 12.8568 | 38.5639 |
| Final evaluation | 0.6857 | 0.0385 | 0.0743 |

Attention entrypoints total 1.5283 s elapsed and 1.4900 s main CPU; expert
entrypoints total 66.0044 s elapsed and 10.3716 s main CPU. Entrypoint time can
charge deferred work. Elapsed minus main CPU includes workers, blocking and
descheduling; it is not GPU idle time. Process CPU sums all threads, and nested
scopes must not be added to their parents. These findings favor reducing expert
I/O or exposed verification work, rather than assuming a large Python-only win.
The earlier valid decode-read-attribution-20260917 receipt already separates
read waits from evaluation; this diagnostic adds CPU clocks at current capacity.

The diagnostic adds 16 MiB host reserve and no additional Metal tensor owners.
Baseline is 10,648,535,040 B; host reserve 1,388,441,600 B; complete physical bound
109,684,373,724 B. MLX peak is 96,906,538,516 B, internal process footprint peak
97,937,115,520 B and internal machine peak 109,217,759,232 B. The guard records
218 samples and zero compressor growth. These quantities remain separate.

Source is `575c3c8b3beb0420d16fc03c727f3a27c0f36edd`. Guard 16767 exits 0,
reclaims source pages, restores exact Qwen and warmup, and releases at
09:38:41 UTC. Independent health/model/free-lock verification passes at
09:39:18 UTC. Final executed helpers are archived. The staging script was
subsequently brought into agreement with their diagnostic pass-row labeling.
