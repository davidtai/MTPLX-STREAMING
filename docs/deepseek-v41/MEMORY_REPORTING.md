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
| Freed Metal allocator cache | 2 GiB, **inside** the Metal allocation |
| Metal prefill/decode reserve | `max(10, 1.45 + 2)` GiB; sizes the engine below the allocator limit |
| Serving session bank | 2 GiB retained compact owners, 2 GiB admitted copy candidate and 2 GiB scratch allowance for strided copies, additionally reserved inside Metal |

For example, a 12 GB baseline leaves 95.85 GB for Metal and 2.15 GB for Python.
The engine receives 85.12 GB for benchmarks or 78.67 GB for serving with the
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

## File-cache accounting follow-up

The subsequent full-model calibration found two sources of unbounded file-cache
growth: the benchmark omitted the expert profile's cache-bypass setting, and
Engram row readers used buffered I/O alongside their Python LRUs. DeepSeek expert
readers now default to `F_NOCACHE` on macOS, both benchmark runners preserve the
profile's explicit setting, and Engram row readers install required cache bypass
once at construction. An explicit buffered expert control remains available.
Missing or refused required cache bypass fails before reading payloads.

New receipts stamp the actual expert `resolved_plan.io_cache_mode` and the typed
`resident_load_report.engram_io_cache_modes`. The transient pool is shared across
layers: 48 slots of 18,800,640 bytes occupy 902,430,720 bytes, not 40 times that
amount. Decode I/O counters and the final memory-profile token label use actual
generated decode tokens when EOS stops a pass early.

With expert and Engram row bypass installed, a guarded 16K/32-step calibration
at a reduced 100 GB allocation target, 16 GiB transient band and 2 GiB allocator
cache reached a sampled physical peak of 99.188 GB. All 33 output token IDs
matched the corresponding historical control prefix. This establishes that
specific calibration, not the default 110 GB plan or a 1,024-token performance
result. The original path-based resident loader still added about 9.737 GB of
file cache during load and is measured separately. See
[`receipts/memory-budget-110/README.md`](receipts/memory-budget-110/README.md) for
raw evidence, provenance, and explicit corrections to the older receipt fields.

The macOS resident path now opens uncached file handles. MLX eagerly evaluates
all tensors when passed a file object, so each load group admits every touched
header before its first `mx.load`: shards are at most 3 GiB, headers at most
1 MiB, individual tensors below the signed 2 GiB read limit, and discarded
payload at most 64 MiB in aggregate. Text shard metadata must match the manifest.
These checks cover the AR and MTP partitions without materializing excluded
expert banks or unrelated large tensors.

Engram attachment separately admits its sidecar and loads it once for both
declared layers. An explicit name allowlist prices discarded tensors and retains
only the projection fields used by those layers. Existing standalone single-layer
and non-macOS APIs retain their prior lazy path loading. These are construction
routes; no eligibility checks or fallback branches were added to generation.
Tiny guarded tests preserve BF16/F32 signed zero and NaN payloads, U8/U32 data,
shape and dtype after the uncached file handle closes.

The uncached 16K measurements also exposed an underpriced default transient
reserve. The active allocation peak exceeds fixed storage plus actual expert
slots by about 6.53 GiB; the allocator's full retained cache must be added to
that amount. The old 5.54 GiB prefill band and 6 GiB cache admit a projected
115.21 GB envelope for the measured Python workload. The revised defaults are
a 10 GiB transient band and 2 GiB cache, with a projected 107.91 GB envelope
after actual slot rounding. This regression uses the true MLX active peak,
full cache capacity, full Python reserve and measured baseline, not a sampled
OS peak alone. These are workload-specific bounds; other shapes, MTP and larger
explicit cache overrides still require their own headroom checks.

The revised defaults completed the exact 16,384-input/1,024-output Python
workload: 4.201765 decode TPS, identical output token IDs to the smaller-cache
control, 56.845 GB MLX active peak, and 106.350 GB sampled physical peak from
the 250 ms trace. The runner's 1 s trace observed 104.55 GB; both are correctly
labeled sampled observations. No additional swapouts occurred, and the exact
service returned healthy with completed warmup. See the default-run receipt
summary for raw byte counts and provenance. This validates that workload,
not the separate 20 TPS objective.
