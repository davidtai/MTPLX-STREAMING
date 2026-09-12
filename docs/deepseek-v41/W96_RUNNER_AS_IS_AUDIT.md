# W96 — DeepSeek-V4.1 streaming AR decode runner: as-is audit (read-only)

Audited tree: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41` at HEAD `1e6222811`
(window-37 receipts). Nothing in the worktree was modified. One CPU-only census script was run from the
scratchpad (`w96_sync_census.py`, fake runtime, no model, no bank, `mx.set_default_device(mx.cpu)`).
While this audit ran, another worker made an uncommitted docstring-only edit to
`mtplx/models/deepseek_v41_stage_timing.py` (a "fenced ms are latency-inflated" warning); it does not change
any finding below.

## 0. Summary

**structural defects: 14; inherent syncs per token: ≈31 today (1 sampler + ≈30 layers that miss), floor ≈10
with one-layer-ahead prefetch; implementation-choice latency ≈ 220 ms of the 460 ms unfenced token.**

1. The runner pays **≈140 blocking host↔device syncs per AR token** at the 16K cell, not 40: 40 routing
   `mx.eval(indices)` + ≈10 all-hit wave fences + ≈30 split-layer hit fences + ≈59 per-miss-part fences +
   2 at the sampler (empirically counted on CPU: all-hit layer = 2 evals, split layer = 2 + m evals).
2. Every one of the 40 routing syncs is a full pipeline drain with **nothing queued behind it**; the untimed
   window-37 decode spent **≥315 ms of every 460 ms token inside `mx.eval(indices)`** (≥7.9 ms per barrier vs
   ≈2 ms of isolated kernel), and the GPU then idles through host planning, the SSD wait, and the host re-encode
   of the next layer's ≈270 dispatches (68% busy at 1.13 GHz).
3. Miss reads are issued **only after** the same layer's sync and awaited on the generation thread
   (`as_completed`), ≈2 reads in flight per layer → drive at ≈2.4 GB/s of 13.4 GB/s; ≈90–110 ms/token
   of serialized SSD wait.
4. Route planning, admission, pinning, counter snapshots, releases and 8 lock classes all run on the
   generation thread inside the drained window; the AR M=1 path is the one switch path deliberately kept
   on the legacy fenced loop (`expert_mlx.py:2897-2898`).
5. Two-tier cache: 68% of miss bytes land in transient scratch and are discarded after one use
   (window-37: `transient_loads 10,312` of `expert_misses 15,064`); bench uses `frequency`, served uses `lru`.
6. Per-token O(T) work survives the ring: the compressor frontier is `mx.concatenate`d every token on the 4
   kv-source layers (≈268 MB copied/token at 16K).
7. Token level: two sampler syncs + a host→device id upload; the host-side numpy engram hash makes the token
   id a host dependency, which silently defeats the device-sample lever (K32) on this model.
8. 55 `MTPLX_DSV41_*` env keys (37 on the decode path, ≈1,000 `os.environ.get` per token), 13 decode
   execution paths inside one 1,200-line switch method, 3 layer-forward variants × 3 cache backings,
   ≥10 distinct token loops; ≥17 keys gate levers that are dead, null or shelved yet still live in the loop.
9. The fenced stage census that drove W73→W92 measures a different code path (compile levers forced eager)
   at ≈700 MHz, so every absolute per-stage ms in the ledgers is drain latency, not compute.
10. A rebuild that keeps one lazy graph per token, device-side routing for resident experts under a
    token-scoped eviction epoch, early (predicted) reads, boundary-time policy and a lagged sampler read
    would delete ≈40 env keys, ≈11 of 14 switch paths, and ≈100 of the ≈140 syncs per token.

## 1. Evidence base

| Source | What it gives |
|---|---|
| `docs/deepseek-v41/receipts/gpu-windows/window-33/ar-16k-cell16k-ring.json` | fenced stage census, 256 tokens: frame 470.1 ms/token; `route_stage` counts: `hot.eval_indices` 10,240; `hot.allhit_fence_eval` 3,054 × 0.917 ms; `hot.begin_split_route` 7,186 × 99 µs; `hot.route_host` 10,240 × 30 µs; `hot.try_all_hit` 10,240 × 29 µs; `hot.all_hit` 3,054 / `hot.split_route` 7,186 |
| `window-37/ar-ring-ref.json` | paired reference, unfenced 2.17 tok/s (460 ms/token); utilization during the untimed decode: GPU 1,128 MHz mean (900–1,352), `gpu_usage_ratio` 0.68, 5.4 W; `serve_stream_counters`: hit 0.755, 58.8 misses/token, 1.106 GB/token, 30.6 split routes/token, `transient_loads 10,312`, `persistent_loads 4,752`, `evictions 4,752`; `switch_dispatch`: all_hit 2,404, `allhit_fence_synced` 2,404 (100%), `begin_split_route` 7,836; `route_probe_sums_ns` over the **untimed** 256-token decode: `hot.eval_indices` **80.66 s** (≥315 ms/token), `hot.allhit_fence_eval` 2.66 s, `hot.route_host` 0.36 s, `hot.try_all_hit` 0.29 s, `hot.begin_split_route` **−0.38 s** — the negative value shows these are `after − before` deltas (`serve_stream_counters.py:284-291`) taken across the probe clear at `ab_decode_env_levers.py:1469-1476`, i.e. contaminated by the prefill's sums; every value is therefore a lower bound on the decode sum |
| `window-37/ar-ring-switch.json` | fast-path arm: `allhit_fence_deferred` 2,404/2,404, 2.05 tok/s (no gain) |
| `window-36/in-model-16k-cool0.json`, `window-37/in-model-1k-*.json` | fenced in-model passes: full 700 / expert-stub 393 / attn-stub 304 ms/token at 16K; 596 / 411 / 311 at 1K; fenced census runs the GPU at 712–870 MHz, 40–52% busy |
| `docs/deepseek-v41/W91_LAYER_SMALL_STAGES.md` | dispatch census: small stages 584 dispatches/layer eager → 265 compiled → 107 with the K3 kernel; 0 host syncs |
| `docs/deepseek-v41/W92_SWITCH_DISPATCH.md` | 16K route is ≈30% all-hit; all-hit fence removal ≤2.5% |
| `~/projects/OpenSourceWTF/state.md` (last 80 lines) | program history; 460 ms ≈ 240 kernel/SSD + 220 drain/refill estimate |
| `scratchpad/w96_sync_census.py` (this audit) | CPU count of blocking `mx.eval` per streamed-switch call with a fake runtime modeled on `tests/test_multi_wave_deferred_release.py` |
| CPU microbench (this audit) | disabled `route_probe.bracket` 0.31 µs, `stime.stage` off 0.10 µs, `os.environ.get` 0.21 µs, SHA-256 over one 18.8 MB record 5.3 ms (3.55 GB/s) |

Standard cell arm (`cell16k_ring`): `SINKHORN_METAL=1 ATTN_COMPILE=1 ATTN_WIN_MEMO=1 SELECTED_KEYS=1 WINDOW_RING=1
LAYOUT_FIX=1 HEAD_MODE=bf16 PREFILL_LAYER_MAJOR=1 PREFILL_DENSE_EXPERTS=1`; `HC_COMPILE`, `SMALL_STAGES_FUSED`,
`SWITCH_FASTPATH/SUBMIT`, `DEVICE_ROUTE*`, `SHARED_OVERLAP` unset. Streaming config: `component-banks`, 48 transient
slots (one global pool), 2,440 persistent slots (61/layer), `deferred_pin_release=False` (dataclass default,
`expert_runtime.py:181`), `split_route_release="deferred"` (profile) — which alone does not defer anything
(`expert_mlx.py:3443-3448` requires both), `overlap_miss_reads=False` (default, `:211`), `io_read_fanout=1` (`:158`),
`verify_record_hashes=False` in both bench (`ab_decode_env_levers.py:1043-1046`) and served profile
(`mtplx/data/expert_profiles.json:113`) — the runtime default is `True` (`expert_runtime.py:161`), which would add
5.3 ms of SHA-256 to every miss.

Layer census (from stage counts): 30 `reuse`, 4 `full`, 4 `reindex`, 2 `swa_only`; 2 engram layers (1, 14);
top-6 of 384 experts, 18.80 MB per record.

## 2. One AR decode token, end to end

Legend: **SYNC** = blocking device→host wait (pipeline drain); **UP** = host→device array creation on the hot path;
**I** = inherent to streaming (the host must learn the route to issue an SSD read); **C** = implementation choice.
Costs are per token; "fenced" numbers are from window-33 and are latency-inflated (ratios only).

### 2.1 Token prologue (generation thread)

| # | Where | What | Sync | I/C | Cost |
|---|---|---|---|---|---|
| T1 | bench `bench_standard_shape.py:596-597` / serving `generation.py:7435` | `mx.array([[token]])` — token id uploaded from a Python int | UP | C (a lagged device id would avoid it) | ≈0.1 ms |
| T2 | `deepseek_v41.py:3862` `Model.__call__` | `_stime.active()` probe check; `_resolve_logits_keep` | – | C | µs |
| T3 | `:2848` `Backbone.__call__` | `_resolve_prefill_chunk` (env read `MTPLX_DSV41_PREFILL_CHUNK`), `_stime.set_schedule`, `_stime.chunk(0)` | – | C | µs |
| T4 | `:2929-2935` `_forward_span` | embed gather, broadcast to hc copies, `pre_mix` concat+astype (≈5 dispatches) | – | I | ≈1.0 ms fenced |
| T5 | `:2944` → `engram_v41.py:290-312` | `engram_state.advance(input_ids)`: `np.asarray(input_ids)` (host read of the id array — a hidden SYNC if the id is a lazy device array), numpy rolling hash, `np.concatenate` of the whole history (`:306`, O(T) host copy per token) | host | C (hash could run on device or off-thread; the O(T) concat is a choice) | 0.11 ms |
| T6 | `:2946` | `cache.new_shared_runtime()` | – | C | µs |
| T7 | `:2955-2969` | `_device_route_active` (2 env reads), pinned-LUT hook probe | – | C | µs |

### 2.2 Per layer (×40)

| # | Where | What | Sync | I/C | Cost |
|---|---|---|---|---|---|
| L0 | `:2972-2973` → `engram_v41.py:469-511` (layers 1, 14 only) | `current_row_ids` (numpy slice); `row_cache.dequantize` (`ngram_row_cache.py:297-343`: LRU dict walk, **synchronous `pread` on the generation thread for missed rows** `:322-339`, numpy copies, **UP** of the packed rows, `mx.dequantize`); wkv q8 GEMV + gate math (≈15 dispatches) | host I/O + UP | C (SSD rows could be prefetched; the gather is host-side) | 0.24 + 0.88 ms fenced ×2 |
| L1 | `:2654` `DecoderLayer.__call__` | `_small_stages_use(h)` → env read `MTPLX_DSV41_SMALL_STAGES_FUSED` (`:2362-2378`) → eager; global call counters | – | C | µs |
| L2 | `:2556` `attn_and_moe_input` | `_hc_use_compile` → False (`HC_COMPILE` unset) → eager `_mixes` (`:2522-2530`): astype/reshape/square/mean/rsqrt/matmul then `hc_split_sinkhorn` (`:201-229`: 3 slices, 2 affines, 2 sigmoids, +eps, reshape) → `_sinkhorn_normalise` (`:171-199`: env read `SINKHORN_METAL` + `mx.metal.is_available` + device check per call, global counter, probe count) → K3 kernel (1 dispatch); `_hc_pre` (mul/sum/astype); `_rmsnorm` | – | I (math) / C (≈12 tiny dispatches + 2 env reads) | `hc.premix_sinkhorn` 0.40 ms fenced (×2 per layer) |
| L3 | `:1328-1338` `Attention.__call__` → `_attend :1340` | `_stime.is_prefill()`; `_cos_sin` (2–3 dispatches); `_attn_use_compile` → K22 qkv tape `:1355` (1 dispatch group) | – | I | – |
| L4 | `:1381` → `deepseek_v41_cache.py:846-854` → `_WindowRing.append :452-501` | in-place `slice_update` into the ping-pong ring; `_RING_STATS` dict increments ×2; `layer_cache.window` view slice `:503-509` (a slice op per call) | – | I (append) / C (stats + view slice per layer) | `cache_append` 0.27 ms fenced |
| L5 | `:1396` | `_resolve_selected_keys()` env read (`:1667-1671`) | – | C | µs |
| L6 (kv-source, 4 layers) | `:1265` `_publish_compressed :1233-1253` → `compressor.pool` → `CompressorState.push` (`deepseek_v41_cache.py:635-663`) | **`self.raw_kv = _grow(...)`, `self.raw_score = _grow(...)` `:648-649` = `mx.concatenate` of the full fp32 `[1, T, 512]` history, twice, every token** (`_grow :99-106`); softmax pooling (≈6); RoPE + indexer keys; `append_compress`/`append_index_k` (GrowBuffer `slice_update`) | – | C (only the last `ratio` rows are needed; history is kept for trim/rollback) | `compress_append` 1.61 ms fenced ×4 = 6.5 ms; ≈268 MB of copies/token at T=16K |
| L7 (index-source, 8 layers) | `:1288-1312` | `indexer.select` over all `n_comp` rows (score GEMM + top-k mask ≈10 dispatches); `_mask_to_topk_idx :1306` = **argsort over `n_comp` per token**; `_resolve_select_fence()` env read `:1316` | – | I (select) / C (argsort per token instead of an incremental top-k) | `select` 0.73–0.88 ms fenced ×8 |
| L8 | `:1427` `_sparse_attend_selected :1105-1196` | `_window_selected_idx` (5), broadcast ×2, `_gather_rows` (3), `_selected_compress_gather :1782-1830` (env read `ATTN_SHAPE_STABLE` `:1823`, 3), concat ×2; `_decode_attn_kernel_use(q) :1163` (env read + metal check); f32 einsum, where, max/maximum, exp, sum+exp, einsum, divide (≈12) | – | I (math) / C (≈30 dispatches, 2 env reads) | inside `attn.<mode>` 6.2–8.7 ms fenced (≈2 ms isolated) |
| L9 | `:1443-1444` | K22 out tape, fed `self._o_lora_dense_weight()` (`:1455-1472`) — **`mx.dequantize(wo_a)` re-issued as a graph node every layer every token** (weight-only; size not verified here) | – | C | 40 dequant dispatches/token |
| L10 | `:2593-2606` | `hc.combine` (`_hc_post_impl`, ≈8), second `_mixes` (≈12 + env read), `_hc_pre`, `_rmsnorm` → `moe_input` | – | I/C | 0.32 + 0.40 ms fenced |
| L11 | `deepseek_v41_moe.py:356` `Gate.__call__ :205-247` | K22 gate-prefix tape; argpartition/take/argsort/take/astype/take/sum/div/mul (≈9 dispatches) | – | I | `moe.gate_topk` 0.46 ms fenced |
| L12 | `:363` | `os.environ.get("MTPLX_DSV41_SHARED_OVERLAP")` → off | – | C | µs |
| L13 | `:394` → `expert_mlx.py:2483` `HotExpertSwitchGLU._run` | shape checks; **4 env reads** (`:2527, :2543, :2575, :2576`); `current_expert_routing_phase` (contextvar); device-route gate (`:2578-2597`, off) | – | C | µs |
| **L14** | **`:2625`** | **`mx.eval(indices)` — routing barrier #1: drains every dispatch of this layer's attention/HC/gate (and any deferred gather)** | **SYNC** | **I for a layer with a miss; C for an all-hit layer** | 40/token; the fenced census shows 0.96 µs only because its stage fence already drained; the **untimed** w37 pass spent ≥80.66 s / 256 tokens = **≥315 ms/token (≥7.9 ms per barrier)** inside this call |
| L15 | `:2628-2632` | `flush_deferred_slot_releases()` (no-op in the shipped config); `indices.reshape(-1).tolist()` + `tuple(int(...))` — second host read of the same array | host read | I (the host needs the ids) / C (second copy) | `hot.route_host` 30 µs ×40 = 1.2 ms |
| L16 | `:2637-2658` | `observe_route` (early return); `_pin_hook` → env read `PIN_WORKING_SET` (`expert_runtime.py:3736-3768`) | – | C | µs |
| L17 | `:2899-2932` | `_verify_single_barrier` (env read `VERIFY_SINGLE_BARRIER`) → False at M=1; verify-shape counters | – | C | µs |
| L18 | `:3163-3172` → `expert_runtime.py:3351-3373` | `route_waves` → `partition_route_waves` (`:638-670`, tuples/sets) + `_batch_admission_slots` → 1 wave | – | C (Python planning on the generation thread) | µs |
| L19 | `:3301-3306` → `expert_runtime.py:2743-2818` | `try_all_hit_route` on **every** layer: runtime layer lock (`:2769`) → `try_plan_all_hits_transaction` (`expert_streaming.py:1210-1245`: history/set/dict snapshots + rollback closure; decode epoch++, `_touch_decode` ×6) → `slots.ensure_route` (`expert_slots.py:1683-1769`: pool `_ensure_locks[layer]` → `_ensure_route_owned :2000-2410`: `metrics.update`, per binding `_physical` + `_wait_ready` (condition lock) + `_SlotPinClaim` + `slot.pins = sum(...)` + `ExpertSlotBinding`; `ReadyRoute` with its own `Lock` + `Condition` `:313-314`) → `commit_if_healthy(publish_route)` → `_publish_route_transaction` (`expert_runtime.py:2904-2952`: `_counter_lock`, **3× `counter.__dict__.copy()`** `:2918`, observe, commit, `_mark_device_route_dirty`) | – | C | `hot.try_all_hit` 29 µs ×40 = 1.2 ms (probe-then-plan twice on the 70% split layers) |
| L20a (all-hit, ≈9–12/token) | `:3356-3360` → `_dispatch_component_bank :2322` → `_run_component_bank_q4 :1671` → `_gather_component_bank :1895` | `register_component_bank`; **UP** `mx.array([bank_index...])` `:1691-1694`; `_layout_fix_enabled` + `_layout_fix_min_rows` (2 env reads `:1918`); reshape; 3× `gather_qmm` (mxfp4) + clamp/swiglu (≈3) + `_down_k_pad_enabled` (env `:1972`) + reshape ≈ 9 dispatches | UP | I (gather) / C (upload + 3 env reads) | `hot.allhit_dispatch_build` 13.7 µs |
| **L20b** | **`:3406-3407` → `synchronous_fence :2682-2700`** | **`mx.eval(wave_output)` — a second full drain per all-hit layer, taken because `_deferred_pin_active` is False** (`:3383-3387`) | **SYNC** | **C** | `hot.allhit_fence_eval` **0.917 ms × 11.9/token ≈ 10.9 ms/token** (w33); 100% synced in the ref arm |
| L20c | `:3421-3422` → `expert_slots.py:489-554` | `ready.release(synchronize=False)`: release condition, claim cleanup under each `slot.condition`, `_route_released` under `_lifecycle` | – | C | µs |
| L21a (split, ≈28–31/token) | `:3456-3462` → `expert_runtime.py:3170-3349` `begin_split_route` | **layer lock acquired and held until `close`** (`:3192`); `_plan_route_transaction` → `LayerExpertSlotBank.plan` (`expert_streaming.py:915-1103`: epoch++, touches, hit/miss sets, **per miss a victim scan over all 61 resident slots** `_victim_slot :768-787`, frequency admission test `:1051-1054`, `SlotLoad`/`SlotEviction`, `RoutePlan`) + `plan_transaction` snapshots (`:1113-1126`) + rollback closure; `_subset_route_plan` ×2 (`:3002-3032`); `_miss_route_parts` (`:3034-3080`, one part per miss expert, O(k²) comprehensions); `slots.ensure_route(hit_plan)` (pins hits, same machinery as L19); `PendingSplitRoute(...)` (`:755-817`: 8 dicts/sets, a `Lock`, an `Event`); per miss part `_split_executor.submit(ensure_route_part, ...)` (`:3278-3306`; **thread hop 1**) | – | C (planning on the generation thread, after the drain) | `hot.begin_split_route` **99 µs** ×30.6 = 3.0 ms/token (bracketed part only) |
| L21b (reader threads) | `expert_slots.py:1796-1852` → `_ensure_route_locked` → `_ensure_route_owned :2000-2410` | `_prepare_load` (`:1244+`, slot condition, generation++), **`self._executor.submit(_fill...)` (thread hop 2)** `:2215-2252`, `future.result()` `:2268-2283`, `_wait_ready :1618-1681`, pins → `ReadyRoute`; `_fill :1397-1455` → `read_record_into` (`expert_io.py:993-1149`) → one `os.preadv` scatter per record into the 3 component-bank views (`:740-833`), `io_read_fanout=1` | SSD | I (the bytes) / C (two thread hops per part, one part per expert, issued only now) | 1.92–2.2 parts per split layer |
| **L21c** | `:3507-3513` → `evaluate_component_bindings :2759-2809` | group by bank; **UP** `mx.array(token_positions)` `:2779-2782`; `mx.take`; 9-dispatch gather of the **hit** part; `fence_bindings :2702-2757` (env read `MTPLX_EXPERT_SLOT_FENCES` `:2709`) → `force_sync=True` → **`mx.eval(hit wave)`** `:2719-2726` | **SYNC** | **C** | ≈30/token |
| L21d | `:3525-3526` | `pending.release_hits()` | – | C | µs |
| L21e | `:3533-3560` | shared-expert overlap block — **skipped** (`shared_work is None` on the non-overlap MoE path), so no GPU work is queued during the wait | – | C | – |
| **L21f** | **`:3576` → `expert_runtime.py:1088-1168` `iter_ready_misses`** | **`as_completed(futures)` — the generation thread blocks on the SSD read; the GPU has nothing queued (the hit gather was fenced at L21c)** | SSD wait | I (bytes) / **C (exposure: nothing overlaps it)** | ≈3 ms per split layer → **≈90–110 ms/token** |
| **L21g** (per miss part) | `:3597-3603` | **UP** token positions; 9-dispatch gather of one expert; `fence_bindings(force_sync=True)` → **`mx.eval(miss wave)`** per part; `release_miss :3612` (lease bookkeeping under `_state_lock`, `_validate_and_commit_policy` on the final part `:1136-1145`) | **SYNC** ×m | **C** | ≈59–62/token |
| L21h | `:3634-3643` | `pending.close()` → lock release, `_finalize_if_ready`; `mx.concatenate(outputs)` + **UP** `mx.array(output_positions)` + `mx.argsort` + `mx.take` (`:3647-3652`) to re-order the parts | UP | C | 3 dispatches + 1 upload per split layer |
| L22 | `deepseek_v41_moe.py:397` | `shared_experts(xf)` — 3 q8 GEMV + clip/min/silu/mul (≈8), **issued only after the switch returns, i.e. after the SSD wait**, although it depends only on `xf` | – | C | `moe.shared_expert` 0.31 ms fenced ×40 = 12.4 ms |
| L23 | `:399-401`, `deepseek_v41.py:2608-2620` | compiled MoE combine; `moe_combine` (hc.combine ≈8) | – | I | 0.21 + 0.32 ms fenced |

### 2.3 Token epilogue

| # | Where | What | Sync | I/C | Cost |
|---|---|---|---|---|---|
| T8 | `:2987` | `cache.advance(1)` → 40 Python method calls (`deepseek_v41_cache.py:1244-1250`) | – | C | µs |
| T9 | `:2990-2993` | final HC collapse + RMSNorm (≈6 dispatches) | – | I | 0.25 ms fenced |
| T10 | `:3876` | head GEMV (bf16, 1.3 GB weight read) | – | I | 2.7 ms fenced |
| **T11** | bench `bench_standard_shape.py:602-603` / serving `generation.py:5455` | **`mx.eval(logits)` — the token's terminal drain** | **SYNC** | I (one per token; could be lagged one step) | – |
| **T12** | bench `:599-600` / serving `:5456` | **`int(mx.argmax(logits[0,-1]).item())` — a second round trip on the same data** | **SYNC** | C | `sample` 0.32 ms |

### 2.4 Sync census per token (16K cell, shipped config)

Empirical count from `w96_sync_census.py` (fake runtime, real `HotExpertSwitchGLU._run`, CPU):

| Switch call | blocking `mx.eval` | `async_eval` | layer lock after return |
|---|---|---|---|
| all-hit, shipped | **2** (indices, wave fence) | 0 | released |
| split with 4 hits + 2 miss parts, shipped | **4** (indices, hit fence, miss fence ×2) — each miss fence after an SSD wait | 0 | released |
| split with 2 miss parts, `SWITCH_FASTPATH+SUBMIT` (w37 ring_switch) | 1 | 3 | **held across the token boundary** |
| all-hit, `SWITCH_FASTPATH+SUBMIT` | 1 | 1 | released |

Applied to the window-37 reference counts (256 tokens): 40 routing evals + 9.4 all-hit fences + 30.6 split hit
fences + 58.8 miss-part fences + 2 sampler syncs ≈ **141 blocking host syncs per token** (window-33 counts give
144). The fast-path arm removed ≈99 of them and measured 2.05 vs 2.17 tok/s — proof that the fences are hidden
behind the SSD waits and drains they sit next to, not that they are free: the exposed cost is the *drain
structure*, which the arm did not change (it still drains at L14 forty times with nothing queued).

Inherent: the host must learn the route of every layer whose experts are not all resident-or-in-flight, plus one
read at the sampler. At hit 0.755 that is ≈30.6 split layers + 1 ≈ **31 syncs/token**; all-hit layers (≈9.4)
need none with a device LUT under a token-scoped eviction epoch; with one-layer-ahead gate-oracle prefetch
(W89: missRed@10 ≈ 0.74) the split count falls to ≈8–10, i.e. a floor near **10 syncs/token**.

## 3. Where the 460 ms goes

Unfenced token (w37 ref): 460 ms. Physics (kernel + bytes) ≈ 240 ms per state.md's attribution (attention
≈80 ms of kernel at 2 ms/layer isolated; small stages + gate + shared + combine; 1.1 GB/token of SSD bytes at the
ceiling ≈ 82 ms if fully overlapped). The remaining ≈ **220 ms is implementation-choice latency**.

The one direct, unfenced measurement the receipts contain: the untimed w37 decode spent **≥ 80.66 s of its
117.8 s inside `mx.eval(indices)`** (`route_probe_sums_ns.hot.eval_indices`, a lower bound — see §1), i.e.
**≥ 315 ms of every 460 ms token, ≥ 7.9 ms per routing barrier**, while the attention kernel it waits on costs
≈2 ms/layer in isolation. The barrier wait therefore contains the layer's real GPU work (≈100–150 ms/token at the
observed 1.1 GHz) plus ≈165–215 ms of host-encode starvation, drain latency and DVFS — the GPU executes each
layer's ≈270 tiny dispatches as fast as the host encodes them into an empty queue, then stops. The rest of the
token (≤130 ms) holds the split-layer SSD waits, the ≈89 unbracketed part fences, the sampler and the prologue.
Components overlap, so they do not sum exactly:

| Component | Estimate | Basis |
|---|---|---|
| Inside the 40 routing barriers: encode-starved layer bursts + drain + DVFS (excluding ≈100–150 ms of kernel work) | 165–215 ms | ≥80.66 s / 256 tokens measured inside `mx.eval(indices)` (w37 untimed); 2 ms/layer isolated kernel; 1.13 GHz / 68% busy |
| Serialized SSD waits at L21f (issued after the sync, ≈2 in flight) | 60–110 ms | 30.6 split layers × ≈2–3.5 ms; drive at 2.4 GB/s vs 13.4 GB/s ceiling; bounded above by the ≤130 ms outside the barriers; state.md "≈93 ms serialized SSD" |
| All-hit second fence (L20b) | 10–11 ms | 0.917 ms × 11.9 |
| Split-layer per-part fences (L21c/L21g) — mostly hidden behind the SSD wait | 10–30 ms | ≈89 fences × 0.1–0.3 ms unhidden |
| Host planning/bookkeeping (L15–L19, L21a, releases) | 6–10 ms | bracketed 5.5 ms (30+29+13.7+99 µs) + unbracketed evaluate/release/close |
| Shared expert issued after the wait (L22) | 5–10 ms | 0.31 ms × 40 fenced, partially hidden |
| Sampler second sync + id upload + numpy engram (T1, T5, T12) | 1–2 ms | `sample` 0.32 ms |
| Probe/telemetry code in production (brackets, counters, stats dicts) | < 1 ms | 0.31 µs × ≈6 brackets/layer + 0.10 µs × ≈14 stages/layer + dict increments |

## 4. Ranked structural defects

**D1 — Synchronous per-layer control structure (drain with nothing behind it).** Every layer ends in
`mx.eval(indices)` (`expert_mlx.py:2625`); MLX executes one in-order stream, so the wait is a full drain, and
nothing route-independent is queued before it (the shared expert is issued after the switch, `deepseek_v41_moe.py:397`;
the next layer depends on this layer's MoE output). After the drain the host plans, waits for SSD, then re-encodes
the next layer's ≈270 dispatches into an empty queue; the GPU clock falls in the gap (w37: 1,128 MHz / 68% busy).
Evidence: w36 stub passes (full 700 → attn-stub 304 → expert-stub 393 ms fenced) and the clock-insensitive
≈8 ms/attention bracket at 1K and 16K (w37 in-model 1K vs w36 16K). Inherent only for layers with a miss;
implementation choice for the other ≈25%, and the *exposure* (no overlap) is a choice for all of them.

**D2 — Reads issued only after the sync, awaited on the generation thread, at queue depth ≈2.** `begin_split_route`
submits the miss parts (`expert_runtime.py:3278-3306`) only after `mx.eval(indices)`; `iter_ready_misses`
(`:1088-1096`) blocks the generation thread on `as_completed`; each miss is its own part (`_miss_route_parts`,
`:3034-3080`; `overlap_miss_reads=False` default `:211`) through two thread hops (`_split_executor` → pool
`_executor`, `expert_slots.py:2215-2252`) as one `os.preadv` scatter per record over its three component views
(`expert_io.py:740-833`) with `io_read_fanout=1` (`:158`). Result: 1.106 GB/token
at 2.17 tok/s = 2.4 GB/s (≈19% of the drive), ≈1.9–2.2 reads in flight per split layer, ≈30 serialized waits per
token. No prefetch exists on the shipped path (`prefetch_slots: 0`; W93 unmerged), no cross-layer read
pipelining, and the reads for layer L+1 cannot start before layer L+1's own drain.

**D3 — ≈100 extra blocking fences per token from the pin/release design.** Shipped config = `deferred_pin_release`
False → `synchronous_fence` after every all-hit gather (`:3406-3407`) and `force_sync=True` after the hit part and
after *each* miss part of a split layer (`:3507-3513`, `:3597-3603`, `fence_bindings :2719-2726`). Empirically
2 + m blocking evals per split layer. The AR M=1 path is the one switch path explicitly left on this loop:
"M=1 is excluded (kept byte-for-byte)" (`:2897-2898`) while the M=2..8 verify path got W61/W81 single-barrier +
deferred treatment (`:2933-3155`). The fast-path arm (w37) proves the fences are mostly hidden behind D1/D2, but
they are also why the slot machinery needs pins, claims, completion futures and covering-eval proofs at all.

**D4 — Planning, admission, pinning, counters and releases on the generation thread inside the drained window.**
Per route: `route_waves` + `_batch_admission_slots`; `try_all_hit_route` on every layer (lock, transaction snapshot,
`ensure_route`, `ReadyRoute` with its own `Lock`+`Condition`, 6 bindings + 6 pin claims, 3 counter `__dict__.copy()`),
then on 70% of layers a second full plan in `begin_split_route` (per-miss O(61) victim scans, subset plans,
per-expert parts, a `PendingSplitRoute` with 8 containers + `Lock` + `Event`, futures, admissions); two host→device
uploads per gather (`slot_indices :1691`, `token_positions :2779`) plus a third for part re-ordering (`:3651`).
Measured bracketed host time ≈5.5 ms/token; the structural cost is that all of it runs while the GPU is empty.

**D5 — Two-tier cache + pins + prefetch ring + single pool + device-route dirty flags fused into one policy.**
`LayerExpertSlotBank.plan` (`expert_streaming.py:915-1103`) interleaves five policies; at 16K under the shipped
`frequency` admission (`expert_runtime.py:172`) first-seen misses cannot evict (`:1051-1054`), so **68% of miss bytes
(10,312 of 15,064 loads, w37) land in the 48 global transient slots and are discarded after one use**; every
admitted miss evicts a resident (`evictions 4,752 == persistent_loads`). The served profile runs `lru`
(`expert_profiles.json:104`) — a different policy from every bench receipt. Hit rate is flat 0.73–0.76 across
60/80 GiB plans, W87's pool (0.744) and W85's simulations, because the policy is admission-limited, not capacity-limited.

**D6 — Lock design on a single-threaded critical path.** Eight lock classes touch each route: runtime `_layer_locks`
(`expert_runtime.py:1911`), pool `_ensure_locks` (`expert_slots.py:1698`), per-slot `condition`, `ReadyRoute._release_condition`
(`:313-314`), `_lifecycle`, `_counter_lock`, `_completion_error_lock` (`:2151`), `PendingSplitRoute._state_lock`. The layer
lock is non-reentrant and, on the deferred path, held across the token boundary until an unrelated layer's covering
eval flushes it (census row 3), which is what produced the W92 self-deadlock and the `_apply_derived_allowance`
flush guard (`expert_runtime.py:2472`) and the W92 LUT-builder non-blocking acquire.

**D7 — Fine-grained eager per-layer graph whose host encode is exposed by D1.** With the cell's levers the layer
is still ≈270 dispatches (W91: small stages 584 eager, 190 with the K3 kernel; attention ≈30–50 after K22; gate ≈9;
switch ≈9–20; combine ≈10) ≈ 10–11k dispatches per token, each re-encoded into an empty queue after the drain. The
fusion levers (K4 `HC_COMPILE`, K22, K35 `SMALL_STAGES_FUSED`) shrink the count but the w37 `ring_fused` arm measured
2.13 vs 2.17 — the drains, not the encodes, are on the critical path; and K35 was found non-byte-identical on Metal.

**D8 — Route-independent work serialized behind the route.** The shared expert (`deepseek_v41_moe.py:397`) and the MoE
combine are dispatched only after `switch_mlp` returns, i.e. after the SSD wait; the hoist exists behind
`SHARED_OVERLAP` (`expert_mlx.py:2611-2623`) but is off and only covers ≈0.4 ms.

**D9 — Per-token O(T) work the ring did not remove.** (a) `CompressorState.push` concatenates the whole fp32
`[1, T, 512]` `raw_kv` and `raw_score` every token on the 4 kv-source layers (`deepseek_v41_cache.py:648-649`, `_grow :99-106`)
— ≈268 MB of copies and 8 × 33 MB allocations per token at 16K (`attn.full.compress_append` 1.61 ms/layer fenced),
retained only so `trim`/`rollback` can restore any frontier; (b) `_mask_to_topk_idx` argsorts over `n_comp`
(≈8K rows at 16K) per index-source layer per token (`deepseek_v41.py:1306`); (c) the engram history is
`np.concatenate`d on the host per token (`engram_v41.py:306`); (d) `mx.dequantize(wo_a)` is re-issued per layer per
token as a K22 tape input (`:1455-1472`, size unverified).

**D10 — Token-level host dependencies.** Two sampler round trips (`mx.eval(logits)` then `argmax(...).item()`,
`bench_standard_shape.py:599-603`, `generation.py:5455-5456`) and a host→device id upload; the engram hash runs on the
host in numpy and reads the id array with `np.asarray` (`engram_v41.py:290`) inside `_forward_span` (`:2944`), so
under the device-sample lane (K32, `run_device_sample_decode` `deepseek_v41_dspark_decode.py:616-711`) the "lagged" id
is forced one step early — the one-step-lag pipeline cannot engage on this model while engram is attached (it is:
`engram.advance` count 256 in every receipt). K32 is dead by construction here, not just default-off.

**D11 — The measurement path is not the production path.** `_hc_use_compile`, `_attn_use_compile` (prefill) and
`_small_stages_use` force eager when a stage-timing session records (`deepseek_v41.py:2240-2258`, `:2362-2378`);
every bracket drains (`deepseek_v41_stage_timing.py:210-235`) and the census runs the GPU at 712–870 MHz / 40–52%
busy (w36/w37) — so every absolute per-stage ms in KERNEL_LEDGER/OPTIMIZATION_LEDGER and in W73→W90 is drain latency
(≈8 ms/attention bracket at every T, clock and arm) rather than compute. The unfenced probe sums that would have
answered the question directly are corrupted in the receipts: `route_probe_sums_ns` is an `after − before` delta
across a counter clear, giving a negative `begin_split_route` and a lower-bound-only `eval_indices` (§1).
Production-path telemetry cost itself is negligible (measured 0.31 µs/bracket, 0.10 µs/stage,
`_RING_STATS`/`_KV_STATS` dict increments, global call counters), but it is spread over 6 route brackets +
≈15–20 stage brackets + 4 counter sites per layer.

**D12 — Configuration surface and duplicated paths.** 55 `MTPLX_DSV41_*` keys in `mtplx/` (37 decode-path, 12
prefill, 4 memory, 1 fragment) plus 9 non-DSV41 keys read by the same files; ≈16 `os.environ.get` per all-hit layer
and ≈28 per split layer (read at use, `:2527-2576`, `:2709`, `:1918`, `:1972`, `:363`, `:189`, `:1396`, `:1316`,
`:1823`, `:1163`) ≈ 1,000/token (0.2 ms). `HotExpertSwitchGLU._run` (`:2483-3685`, 1,200 lines) contains 13 decode
execution paths: device-route resident, device-route pinned, shared hoist, W61 verify all-hit deferred / fenced,
W81 verify split single-wave / multi-wave, shadow-bank decode, general all-hit fenced / deferred / deferred+submit,
general split fenced-per-part / deferred, plus direct-bindings layout, dense-prefill, mixed and shadow codecs in
`_dispatch_component_bank`, gated by 9 env keys, 4 config fields and ≈20 `callable(getattr(...))` duck-type checks;
3 more switch classes exist (`MappedExpertSwitchGLU`, `DenseIslandSwitchGLU` ×2 stores, `UnboundExpertSwitch`).
Layer forward: 3 variants (eager / K4 / K35) × attention compile × selected keys × decode kernel × shape-stable ×
win-memo × Sinkhorn kernel × premix kernel; cache backing ×3 (plain / chunk-grow / ring). Token loops: ≥10
(`generate_ar` classic, `MTPLX_AR_PIPELINE` lane, DSV41 device-sample lane, `generate_mtp1`, `generate_mtpk`, dspark
`_decode_cycles` with two public entry points whose signatures diverged (W81 crash), bench `_generate` classic +
device-sample, `_stage_timing_pass`, `_sync_census`, `_warm_repeat_pass`, in-model bisect, `streamed_batch`).
Bench vs served divergence: cache policy (`frequency` vs `lru`), plan fields seeded from the profile only when flags
are unset, `verify_record_hashes` default `True` in the runtime dataclass. Dead/null/shelved levers still gating the hot
loop (≥17 keys): `DEVICE_ROUTE` (K24 shelved), `DEVICE_ROUTE_PINNED` (W71 wrong + 2.2× slower), `PIN_WORKING_SET`/
`PIN_REFRESH_TOKENS` (dead as a hit lever), `DECODE_ATTN_KERNEL` (K29 −38%), `SHARED_OVERLAP` (≈0.4 ms), `KV_CHUNK_GROW`
(superseded by the ring), `SELECT_FENCE` (attribution only), `ATTN_SHAPE_STABLE` (null, mechanism falsified),
`SWITCH_FASTPATH`/`SWITCH_SUBMIT` (null at 16K), `SMALL_STAGES_FUSED` (null, not byte-identical on GPU),
`HC_PREMIX_KERNEL` (parity pending), `SINGLE_SLOT_POOL` (no cell gain), `HC_COMPILE` (no valid GPU receipt),
`DEVICE_SAMPLE` (defeated by D10), `MTP` (generic MTP net-negative), `DRAFT_COMPILE` (draft 268 ms armed vs 184 unarmed).

**D13 — Prefill paths.** Chunk-major (`_forward_span` per chunk, `:2874-2913`) re-streams the routed bank per chunk
(≈13× at 16K) and is the default when `PREFILL_LAYER_MAJOR` is unset; layer-major (`:3142-3284`) reads the bank once
but drains per (layer, chunk) via `_eval_layer_transients` (`:3241-3242`, `:3361-3389`; 40 × 16 = 640 drains at 16K)
and per layer (`mx.eval(hs)` `:3259`), runs the same host planning per wave (sorted-unique waves, `prepare_prefill_seed`,
dense-dequant path), and carries a genuine numerical constraint (gate and shared expert are not M-invariant, `:3286-3300`).
TTFT 180–205 s at 16K against a ≈21 s bank-read floor and ≈100 s compute floor.

**D14 — DSpark lane asymmetries.** `_decode_cycles` (`deepseek_v41_dspark_decode.py:727-941`) pays 4 token-level syncs
per cycle (draft eval `:806`, verify eval `:823`, argmax `np.asarray` `:833`, commit `mx.eval` `:902`) plus a per-cycle
`snapshot_untrimmable_cache` and 40-layer `trim_verified_window_to_prefix` + `seed_main`; its verify (M=2..8) takes the
single-barrier switch path while AR (M=1) does not (D3); it arms K29/K30 for its own AR reference (`arm_dspark_decode_kernels`),
so "AR" differs between lanes; `dspark_generate` (`:967`) and `generate_dspark` (`:1071`) are two public entry points.

## 5. What the one-sync-per-token reference has that this runner lacks

Resident backend `mtplx/models/deepseek_v4.py`: `DeepseekV4MoE.__call__` (`:3316-3323`) is `gate → switch_mlp(xf, indices)
→ combine` with the routing indices staying on device (`SwitchGLU`/`gather_qmm(rhs_indices=…)`); `DeepseekV4Model.hc_hidden`
(`:3467-3489`) is a plain layer loop; no locks, no plans, no fences, no env reads per layer; the whole token is one lazy
graph and the only host read is the sampler (`generation.py:5455-5456`), or — on the `MTPLX_AR_PIPELINE` lane
(`:7250-7270`) — an `async_eval` of the next token with `tok_lazy.item()` read one step behind. The DeepSeek CUDA
reference (`inference/model.py:889-904`, `:1265-1285`) is a naive loop that does `bincount(...).tolist()` per layer
and samples on device (Gumbel-max) — one forward per token, no cache machinery on the critical path.

What the streaming runner lacks relative to that shape: (1) a token as one submitted graph with the routing decision
kept on device for resident experts; (2) any work queued ahead of a host wait; (3) reads issued from a prediction
rather than from the drained route; (4) policy work moved off the per-layer path; (5) a lagged sampler read; (6)
one code path.

## 6. What a clean streaming decode loop must contain

1. **Sync structure.** One lazy graph per token, submitted incrementally with `async_eval` at layer granularity;
   blocking host reads only at (a) layers whose route contains an expert that is neither resident nor already in
   flight, and (b) the sampler, read one step behind the next token's submission. Everything else stays on device.
   Target: ≈10–30 blocking reads/token at today's hit rate, ≈2–10 with prefetch; never a second fence per layer.
2. **Device-side routing for resident experts under a token-scoped eviction epoch.** Per layer a device `int32[384]`
   expert→slot LUT rebuilt only at token boundaries; within a token no slot that was resident at the token's start
   is recycled, so `gather_qmm(rhs_indices=lut[indices])` is exact with no pins, claims, completion futures, releases
   or covering-eval proofs. The host reads `lut[indices]` (6 ints) or a single "any miss" flag; on all-hit the layer
   never blocks. This is K24's idea made exact by construction (epoch), not by reconcile/rollback (W44/W71).
3. **Reads issued early, in one hop, at depth.** At layer L's read, also apply layer L+1's router to the residual
   entering L (W89 gate oracle, missRed@10 ≈ 0.74) and issue those reads immediately so they overlap L and L+1;
   issue all of a layer's misses as one batch from the generation thread to a reader pool (no per-part futures,
   no second executor), with fanout ≥ 4–8 per record so the drive runs at queue depth ≥ 8; a miss whose read is in
   flight from prefetch is a wait, not a new read.
4. **Eviction and policy at the token boundary, off the per-layer path.** After the sampler's eval (which covers every
   gather of the token) decide victims for the next token's admissions in one pass over the token's 40 routes
   (2Q/LRU over one resident pool — no transient scratch that discards bytes, no separate prefetch ring, no pins),
   update counters as plain ints, rebuild the dirty LUTs. Per-layer host work shrinks to: read 6 ints, bitmap test,
   append misses to the read batch. No locks on the generation thread's per-layer path (single writer within a token;
   readers only fill bytes and set a ready flag; the boundary pass is the only place slots change owner).
5. **Route-independent work ahead of the wait.** Shared expert, the next layer's route-independent prologue, and
   any prefetch reads are queued before the routing read of the current layer.
6. **Graph coarseness.** Keep the per-layer graph to ≈3 compiled segments (K35's seg1/seg2/seg3 shape) so the encode
   after an unavoidable wait is ≈10–20 dispatches; no eager Sinkhorn split, no per-token weight dequant, no per-layer
   `mx.array` uploads (slot indices come from the device LUT; token positions are `arange`).
7. **Cache lanes without O(T) per token.** Window ring as today; compressor frontier as a `ratio`-row ring plus a
   rollback journal (no history concat); compress/index preallocated to `max_kv`; index selection kept incremental
   where the indexer allows; engram row ids computed on the lagged host id with rows prefetched for the drafted/next
   token, or accepted as the one inherent per-token host dependency.
8. **One loop, one config.** One AR loop shared by bench and serving, one DSpark loop calling the same per-token
   function with M>1, the served profile as the only source of plan/policy fields, no read-at-use env gates in the
   per-layer path (levers resolved once at model open into a frozen dataclass), and an unfenced stub-attribution mode
   as the only per-token instrumentation.
9. **Exactness.** The LUT gather reads the same slots the fenced path would; boundary-only eviction removes the need
   for any recovery pass; prefetch changes only which reads happen, never a route's output.

## 7. What a rebuild deletes

- Env keys: ≈40 of the 55 `MTPLX_DSV41_*` keys (every decode-path lever gate that becomes the only code path or is
  dead: all of D12's list plus `ATTN_COMPILE`, `ATTN_WIN_MEMO`, `SINKHORN_METAL`, `SELECTED_KEYS`, `WINDOW_RING*`,
  `LAYOUT_FIX*`, `DOWN_K_PAD`, `GATHER_ROWS_PER_CALL`, `VERIFY_SINGLE_BARRIER`, the four DSpark verify keys) and the
  `MTPLX_EXPERT_SLOT_FENCES` / `MTPLX_ROUTE_STAGE_PROBE` gates; the 12 prefill keys are a separate decision.
- Switch paths: ≈11 of the 13 decode execution paths in `_run` (keep: device-LUT gather, miss-batch gather) and
  the pin/claim/completion-fence/deferred-release/pending-split machinery (`ReadyRoute`, `PendingSplitRoute`,
  `_DeferredSplitClose`, `defer_slot_release`, `flush_deferred_slot_releases`, `_ReadyRouteGroup`, `_RouteCancel`,
  `RouteIOAdmission` children, `_SlotPinClaim`).
- Loops: ≥8 of the ≥10 token loops.
- Syncs: ≈100 of the ≈141 blocking syncs per token at once (fences + sampler double read), then a further ≈20 as
  prefetch raises the in-flight hit rate.
- Surface today: 27,007 lines across the 18 DSV4.1/streaming modules (`expert_runtime.py` 4,508, `deepseek_v41.py`
  4,222, `expert_mlx.py` 3,777, `expert_slots.py` 2,588, `expert_streaming.py` 2,079), 81 test files, 34 scripts
  (13,263 lines), 78 docs.

## 8. Open questions (not verifiable from this tree)

- The exact split of the ≈220 ms between D1 (drain/encode/DVFS) and D2 (SSD exposure) needs the unfenced stub
  attribution W94 is building; the fenced receipts cannot separate them (D11).
- Size of the per-layer `wo_a` dequant (D9d) and whether MLX caches it across calls — unmeasured here.
- Whether `overlap_miss_reads=True` (one part per layer, batched scatter) was ever measured on this model; every
  receipt shows one part per miss.
- Whether the served daemon's `lru` policy changes the transient/persistent split seen in the bench receipts.
