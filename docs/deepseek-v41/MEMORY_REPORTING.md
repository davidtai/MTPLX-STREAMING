# DeepSeek V4.1 memory reporting

New benchmark memory blocks use `schema_version: 2`. Bytes are authoritative;
`*_gb` is decimal (1,000,000,000 bytes), and `*_gib` is binary (1,073,741,824
bytes). Historical receipts retain their original mixed units and must not be
compared by field name alone. New top-level `peak_gb` is the allocator peak in
decimal GB. It is not the process or machine total.

| Measurement | Meaning |
| --- | --- |
| `mlx_peak_bytes` | Peak active allocations reported by MLX for this pass; excludes Python and retained allocator cache. |
| `mlx_active_bytes_at_decode_start/end` | Live allocator bytes at the decode boundaries. |
| `mlx_cache_bytes_at_decode_end` | Retained freed allocator buffers at decode end. |
| `process_footprint_peak_bytes` | Sampled `TASK_VM_INFO.phys_footprint`, including Metal and charged host memory. Never substituted with RSS. |
| `ru_maxrss_bytes` | Lifetime peak RSS, kept separately; includes earlier passes and load. |
| `system_used_peak_bytes`, `box_used_gb` | Sampled physical machine usage, including active/inactive file cache, matching `top`'s PhysMem used definition. |
| `baseline_plus_process_peak_estimate_gb` | The old pre-load baseline plus process peak calculation. An estimate, not measured system usage. |

System used is `(wired + active + inactive + physical compressor) * page_size`.
Speculative pages are free, and compressed logical pages must not replace the
physical compressor size. This follows [Apple's top implementation](https://github.com/apple-oss-distributions/top/blob/main/globalstats.c#L485-L488).
File cache is reclaimable but remains visible in the physical total; excluding it
must be labeled explicitly. Neither process footprint nor MLX bytes should be
added to this system total: unified-memory pages are already included.

The benchmark takes OS observations at the generation boundaries and every
second on a background thread that does not call MLX. Each observation has start
and end monotonic timestamps around sequential kernel reads. Receipts keep up
to 4,096 observations, a total observation count, failed-read count, and the
complete observations at the process and system peaks. These are sampled peaks;
brief transients between observations can be missed. Missing readings are null
in JSON and `n/a` in the headline.

AR and DSpark have separate sampling windows and allocator resets. AR's final
observation is collected while its cache is still alive. A generation receipt
does not claim to include model loading. The GPU guard covers that lifecycle.
For DeepSeek V4.1, `/health.memory_usage` uses the same OS reader, with bytes and
sample timing; it is sampled only when health is requested.

The 110 GB allocation target must cover Metal allocations, allocator retention,
Python caches, I/O buffers, and the separately measured system baseline. Memory
limits and planned cache capacities are configuration, not usage measurements.
The reporting change does not increase allocation limits or weaken guard trips.

Validation includes fixture arithmetic, missing-reader behavior, distinct AR and
DSpark windows, serving-health wiring, short-run boundary sampling, and a live
64 MiB allocation compared through both `task_info` and `proc_pid_rusage`.
Receipts are under `receipts/memory-reporting/`. The initial `051622Z` receipt
predates the speculative-page correction; later receipts use the corrected
definition.
