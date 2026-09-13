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
The allocation change below is separate from usage measurement.

## 110 GB allocation

The default target is **110,000,000,000 bytes**. Both benchmark entrypoints and
the DeepSeek serving constructor resolve the following allocation before load:

| Allocation | Default |
| --- | --- |
| System baseline | Measured physical used memory after the resident service stops, including file cache |
| Python capacity reserve | 2 GiB, increased when configured cache capacity requires more |
| Engram payload within Python reserve | Two 256 MiB row arenas; index/free-list storage priced separately at 256 bytes per slot |
| Other Python within reserve | 1 GiB for tokenizer, temporary rows and reader bookkeeping |
| Metal allocator and wired policy | 110 GB minus measured baseline minus Python reserve |
| Freed Metal allocator cache | 6 GiB, **inside** the Metal allocation |
| Metal prefill/decode reserve | `max(5.54, 1.45 + 6)` GiB; sizes the engine below the allocator limit |
| Serving session bank | 2 GiB retained compact owners, 2 GiB admitted copy candidate and 2 GiB scratch allowance for strided copies, additionally reserved inside Metal |

For example, a 12 GB baseline leaves 95.85 GB for Metal and 2.15 GB for Python.
The engine receives 87.85 GB for benchmarks or 81.41 GB for serving with the
default session bank. These engine totals include their existing runtime/KV
reserves; the expert slot planner derives the remaining cache capacity.

The previous 2 GiB **per layer** Engram setting implied 4 GiB of row arenas
before Python indexes, so a 0.5 GiB host reserve could not cover it. Overrides
now price full configured capacity, even before pages are touched. Explicit
host reserves below that amount fail before loading. Smaller explicit serving
engine budgets remain respected; generic memory/wired controls that conflict
with the target fail clearly. Allocator cache overrides enter the same budget.

The SSD session writer can admit an oversized entry beyond its backlog cap.
This budgeted DeepSeek serving path therefore defaults SSD session caching off
and rejects an explicit request to enable it until its host peak is bounded.
RAM session reuse uses compact, materialized snapshot owners so visible tensor
bytes match retained storage. Other model families keep their existing policy.

Missing/refused wired or allocator-cache APIs fail before model allocation.
Targets must fit below the fixed 100 GiB wired ceiling and leave at least 8 GB
of physical RAM beyond the target. Pinned benchmark plans restore Python cache
capacity and reject a current system baseline larger than the saved baseline.
Explicit legacy `--memory-limit-gib`/`--box-budget-gib` benchmark plans remain
available; they opt out of automatic target sizing and still require the guard.

`gpu_window.sh` defaults to a 110,000,000,000-byte physical-used ceiling. Its
new `GPU_WINDOW_TOTAL_MEM_CEILING_BYTES` accepts bytes; the historical
`GPU_WINDOW_TOTAL_MEM_CEILING_GB` override retains its **GiB** meaning. Supplying
both fails. Failed memory/process readers abort instead of becoming zero.
Restoration requires the owned child tree to exit and the exact previous model
identity plus healthy completed warmup to return. Failed restoration is a
nonzero `RESTORE_FAILED` result; bounded recovery failure requires operator
recovery before another GPU run.

The guard's sampled physical peak and conservative baseline-plus-child guard
estimate are reported separately. Sampling and MLX allocator policies do not
prove a hard instantaneous peak bound. The prior transient bands are workload
evidence, not certification of new shapes or a full-model 110 GB run. Validate
compile/graph and loading headroom before such a run; file-cache growth can
legitimately consume the budget and trip the guard.

Validation includes fixture arithmetic, missing-reader behavior, distinct AR and
DSpark windows, serving-health wiring, short-run boundary sampling, and a live
64 MiB allocation compared through both `task_info` and `proc_pid_rusage`.
Receipts are under `receipts/memory-reporting/`. The initial `051622Z` receipt
predates the speculative-page correction; later receipts use the corrected
definition.

The final fix validation on 2026-09-13 passed 636 guarded tests across the
memory reporters/planners, both benchmark runners, public/expert CLIs, DSV
decode, SessionBank, Engram state and warm-prefix restore. Tiny CPU and Metal
copy checks preserve signed-zero/NaN payload bits, release a 16 MiB backing,
and handle strided views. Separately, 25 hermetic guard regressions passed;
parent reruns also passed all 14 abort/orphan and 10 restore assertions.
The previous Qwen model identity returned healthy with warmup complete and
the exclusive GPU lock released. No full DeepSeek model or 110 GB benchmark
was executed as part of this validation. The snapshot bit checks cover bank
admission copies, not every existing auxiliary/MTP restore path.
