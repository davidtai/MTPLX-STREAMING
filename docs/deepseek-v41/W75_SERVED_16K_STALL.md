# W75 — served 16K prefill stall (DeepSeek-V4.1-Flash streaming)

**Symptom.** The served streaming lane (`mtplx serve --model …-mxfp4
--expert-memory-limit <cap>`, launched by
`scripts/deepseek_v41/served_cell_bench.sh`) stalls on the standard 16,384-token
cell at BOTH an 80 GiB and a 70 GiB expert cap, while the in-process bench
(`scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384`, 80 GiB
plan) completes the same prefill in ~200 s at an 85 GB peak. Logs:
`docs/deepseek-v41/receipts/gpu-windows/window-29a/serve-16k-70g-stall.log`,
`serve-16k-80g-stall-window28b.log`. Signature: repeated
`pressure_trim level 4 … allocator_fraction 1.02–1.13 … bank_bytes_after 0`
through prefill, then `mtplx_stream_stall_break owner_frozen_s 300.2
deadline_s 300.0 streamed_tokens 0`. A 1K prompt serves fine at the same caps.

## Root cause — three compounding bugs

### (1) The pressure guard's CRITICAL denominator is the soft engine budget, not the machine ceiling

`_allocator_pressure_level` (`mtplx/server/openai.py:18625`) computes
`fraction = (active + cache) / caps["memory_limit_bytes"]` and calls
`fraction >= 1.02` CRITICAL (level 4). For the SSD-streamed lane
`caps["memory_limit_bytes"]` is the **engine budget** = `--expert-memory-limit`
minus the runtime reserve/io-staging (`apply_mlx_memory_cap` /
`reconcile_mlx_memory_cap`, `mtplx/expert_runtime.py:1644`): **63 GiB at a 70 GiB
cap, 73 GiB at 80 GiB** (the serve banner prints `engine budget 63.0G/73.0G
(Metal limit)`). That budget deliberately sits far below physical RAM so the
**reclaimable** expert cache and the **bounded, admission-reserved** per-chunk
prefill transient can use real machine headroom.

A 16K prefill legitimately peaks at ~85 GB (in-process, 80 GiB plan). Against a
73 GiB limit that reads `85/73 = 1.16`, so the guard fired CRITICAL on a healthy,
already-admitted request. Note the inconsistency: the admission gate
(`_expert_streaming_prefill_admission`, `openai.py:18241`) explicitly budgets the
transient via `_dsv41_prefill_transient_bytes()` (`openai.py:18224`, ~8 GB, the
`MTPLX_DSV41_PREFILL_CHUNK_TARGET_GB` bound) into its reservation and **admits**
the prompt — but that same transient was **not** in the guard's denominator, so
the guard fought the admission decision.

### (2) The CRITICAL trim tore the buffer pool out from under the live prefill

In `_memory_pressure_loop` the level-4 branch shed the bank (already empty →
`bank_bytes_after 0`, `bank_entries_evicted 0`) and then ran `mx.clear_cache()`
**unconditionally** (`if evicted or level >= 4:`). `clear_cache` returns the
allocator's cached free buffers to the OS; mid-prefill the growing working set is
actively reusing those buffers, so each clear forces re-allocation from the OS
and re-faults the SSD expert stream — the exact thrash the *dynamic_ceiling*
branch a few lines up already refuses while busy ("freed bank buffers return to
the allocator pool and get reused by the growing KV"). This both slowed the
prefill past its ~200 s clean time and, by oscillating the fraction back under
1.02 after each clear, **reset the sustained-critical abort streak**
(`_note_critical_pressure_tick`, `_PRESSURE_ABORT_TICKS = 3`,
`openai.py:18734`), so the graceful 507 abort never armed — leaving only the
watchdog to end the request.

### (3) The stall watchdog can't tell a long prefill from a wedged owner

`_OwnerStallProbe` (`openai.py:19326`) breaches only when the model-owner
progress heartbeat (`mtplx/progress_heartbeat.py`) is frozen for the full 300 s
deadline. The heartbeat ticks inside `generation._eval` ("every settled engine
forward") and per scheduler item — but the DSV4.1 chunk-major/layer-major prefill
loops evaluate each chunk with a **bare `mx.eval`** (`_eval_cache_state` /
`_eval_layer_transients` in `mtplx/models/deepseek_v41.py`), which never ticked.
The whole 16K prefill runs inside one `chat.stream` scheduler item, so the
heartbeat was frozen for the entire prefill. A clean 200 s prefill squeaks under
300 s; the trim thrash from (2) pushed it over, and the frozen heartbeat made the
watchdog read a legitimately long, still-progressing prefill as wedged and kill
it with 0 tokens (`owner_frozen_s 300.2`). The receipt is proof: the pressure
fraction moved every tick (memory allocating/freeing → the owner was alive, not
deadlocked) while the heartbeat stayed frozen.

### Why the in-process bench does not stall

`ab_decode_env_levers.py` is not a server: no `_memory_pressure_loop` (no
mid-prefill `clear_cache`, bug 2), no `_OwnerStallProbe` (bug 3). It sets the same
engine budget but the MLX memory limit is a soft guideline — a single op may
exceed it — so the 85 GB peak simply completes in ~200 s.

## Fixes (all CPU-tested; no model load)

**(A) Budget the prefill transient into the guard limit — `openai.py`.**
`_allocator_pressure_level(state, *, extra_limit_bytes=0)` (18625) now measures
against `limit + extra_limit_bytes` (18658). New
`_streaming_prefill_pressure_headroom_bytes(state)` (18672) returns, for a busy
expert-streaming lane, `safe_ceiling − limit` where `safe_ceiling = total_ram −
system_reserve` (112 GiB on a 128 GiB box), else the bounded transient as a
floor, else 0. The loop (18916) reads the busy signal once and passes the
headroom only while a foreground request is in flight. So a legitimate 16K
prefill (85 GB) reads `85/112 = 0.76` → NORMAL; a genuine approach to physical
RAM (>112 GiB) still escalates, and the independent macOS pressure signal is
untouched. Idle and non-streaming lanes are unchanged (`extra_limit_bytes=0`).

**(B) Count prefill chunks as liveness — `deepseek_v41.py`.** The chunk-major
fence `_eval_cache_state` (2483) and the layer-major fence
`_eval_layer_transients` (2731) now call `_owner_progress_tick()` (import at
`deepseek_v41.py:43`) after each settled `mx.eval`. A long prefill becomes a
moving heartbeat, so the watchdog never kills a live prefill (any context, incl.
64K where a clean prefill legitimately exceeds 300 s).

**(C) Level-4 trim is non-blocking for the owner — `openai.py:19036`.**
`if evicted or (level >= 4 and not critical_busy):` — a CRITICAL tick with a
request in flight (empty bank → 0 evicted) no longer clears the cache the owner
is reusing. Idle CRITICAL still reclaims; an actual eviction still clears.

A alone keeps the guard from firing on a healthy prefill; B+C keep the request
alive even if a residual spike briefly crosses a threshold. The sustained-abort
and admission gate remain the safety valves for genuine runaway.

### Tests (green)
- `tests/test_memory_pressure_guard.py` — **31 passed** (19 prior + 12 new W75:
  headroom helper cases, busy-prefill NORMAL vs pre-W75 CRITICAL, still-CRITICAL
  near the ceiling, clear_cache skipped while in-flight / run when idle).
- `tests/test_deepseek_v41_prefill_heartbeat.py` — **5 passed** (both eval fences
  tick on a settled chunk, no phantom tick with nothing to eval, monotone over
  many chunks).
- `tests/test_stream_stall_watchdog.py` — **5 passed** (unchanged).
- Parity unaffected (ticks are additive counters after an existing eval):
  `test_deepseek_v41_ab_env_levers.py` 63, `…_torchref_golden.py` 8+1skip,
  `…_engram_state.py` 6, `…_memory_profile.py` 22, `…_serve_bench_1k.py` 12.

## Served-bench cap recommendation

`DSV41_MEMORY_LIMIT_GIB=80` → `--expert-memory-limit 80GiB` → engine budget
≈ 73 GiB, **measured 16K prefill peak ≈ 85 GB**, i.e. ~15 GiB under the 100 GiB
GPU knob and well within the 110 GiB box budget (no CPU workers during a GPU
window). The peak tracks the budget (the expert cache fills it): from the two
receipts, `(60 GiB→76.6 GB)` and `(80 GiB→85 GB)`, peak ≈ `0.42·cap + 51`; to
keep peak ≤ 90 GB (10 GiB knob margin) the cap must stay ≤ ~92 GiB. 80 GiB is the
directly-measured, comfortable default; **70 GiB (peak ≈ 80 GB)** is the
conservative floor. Neither stalls once fix (A) is in: the guard tolerates the
admitted transient instead of trimming against the soft budget. The streaming
lane's wired residency is clamped to the engine budget (≤ 73 GiB), so the 100 GiB
wired knob is never approached.
