# W47 — DeepSeek-V4.1-Flash prefill stage-timing probe

**Task class:** measurement, not a fix. W37 attributed the decode step (found the
head fp32-cast trap, +31%, and drove the lever ranking). W47 extends the same
fenced probe to the **16,384-token prefill**, the other half of David's standard
shape. Window 16 measured prefill TTFT **490 s chunk-major (33 tok/s)** and
**370 s layer-major (`MTPLX_DSV41_PREFILL_LAYER_MAJOR=1`, 44 tok/s, byte-identical
now)** against a **~20 s bank read** and a **~100 s rough compute floor** — the
370–490 s is otherwise unattributed. This probe places `mx.eval` fences at each
prefill stage boundary, tagged by chunk index, so `model.stage_timing_report()`
shows whether cost grows with chunk index (attention over growing T) or is flat
(MoE), on **both** schedules (they take different code paths:
`Backbone._forward_span` per chunk = chunk-major, `_forward_layer_major` =
layer-major).

Decode timing is untouched; probe-off is byte-identical (extended tests below).

## Contract (shared with W37)

* OFF is free and byte-identical: no session armed → `stage*()` returns a shared
  no-op context manager, nothing is fenced.
* A **prefill** session is armed by `deepseek_v41_stage_timing.begin(kind=
  "prefill")` (by `ab_decode_env_levers.py --prefill-stage-timing`);
  `Model.enter_forward(s)` records the whole prefill forward (`s ≥ 1`), and the
  chunk loop tags each chunk via `chunk(idx)`. A **decode** session (W37) still
  records only `s == 1`.
* Fences **inflate** absolute time (a host round-trip serialises every stage), so
  a prefill-timing pass's tok/s is meaningless — the **ratios** between stages,
  and their **growth with chunk index**, are the signal. The clean tok/s pass runs
  with the probe OFF.
* While a stage-timing session records, the eager Hyper-Connection (K4) and
  attention (K22) paths are forced (`_hc_use_compile` / `_attn_use_compile` return
  `False` under `_stime.recording()` / `is_prefill()`): a compiled tape is one
  opaque call the per-stage fences cannot split. At the prefill chunk (1024 rows
  ≫ the compile row caps) the eager path is the shipped path anyway.

## Prefill stages

Attention is split (the W37 decode probe kept a single `attn.<mode>` bracket; the
prefill probe replaces it with these sub-stages via `stage_prefill`, so decode is
unchanged):

| stage | region | grows with T? |
|---|---|---|
| `attn.<mode>.qkv_proj` | q/kv down+up projections + q/kv RMSNorm + RoPE | no (O(rows)) |
| `attn.<mode>.cache_append` | `append_window` of this chunk's post-RoPE KV | no (O(rows)) |
| `attn.<mode>.compress_append` | kv_source layers: pool + RoPE + indexer keys + compress/index appends | no (O(rows)) |
| `attn.<mode>.select` | indexer/candidate CSA top-k over the compressed rows | grows (O(rows·T_c)) |
| `attn.<mode>.score` | `_sparse_attend` (the `[rows,64,T]` f32 score+softmax+value) + output o-LoRA/`wo_b` | **grows (O(rows·T))** |

`<mode>` ∈ `swa_only` / `full` / `reindex` / `reuse` (the CSA2 layer role), so the
table splits per layer type. The remaining stages match decode: `embed`,
`engram.advance`, `engram.hash/row_fetch/apply` (layers 1 & 14),
`hc.premix_sinkhorn` (×2/layer), `hc.combine` (×2/layer), `moe.gate_topk` (gate +
top-k + routing-barrier fence), `moe.routed_switch`, `moe.shared_expert`,
`moe.combine`, `final_norm`, `head`.

### Streamed-switch breakdown (streamed path only)

`moe.routed_switch` is the fenced switch total. Inside the streamed switch's
prefill path (`HotExpertSwitchGLU._run`, `expert_mlx.py`) three host-side
sub-stages are recorded into `switch_breakdown` (a **nested** view kept OUT of the
flat partition sum, since it decomposes `moe.routed_switch`), via `stage_nested`
(prefill-only, no-op off / decode / resident):

* `switch.admission` — `prepare_prefill_seed` (primes the persistent hot set);
* `switch.route_plan` — `route_waves` (groups expert ids into gather waves);
* `switch.miss_submit` — `begin_split_route` (submits the SSD miss reads).

The blocking miss-I/O **wait** is not a distinct host bracket: the reads stream
asynchronously and their wait is folded into the fenced `moe.routed_switch` total
(reads overlap the gather compute). The route-stage probe's `hot.all_hit` /
`hot.split_route` counts (merged under `route_stage` when
`MTPLX_ROUTE_STAGE_PROBE=1`, which `--prefill-stage-timing` arms) give the
miss/hit ratio. On the resident (CPU test) path there is no streamed switch, so
`switch_breakdown` is `{}` and `moe.routed_switch` (the resident SwitchGLU) is the
whole switch.

## Report shape

`model.stage_timing_report()` for a prefill session adds, over the W37 decode
keys:

* `kind: "prefill"`, `schedule: chunk_major|layer_major|one_shot`, `chunks: N`;
* `by_chunk: {idx: {wall_ms, stage_sum_ms, stages: {name: {total_ms, count}}}}` —
  the per-chunk-index view that answers *does cost grow with chunk index*;
* `switch_breakdown: {name: {total_ms, count, mean_ms}}` — the nested switch view;
* `stages` (flat aggregate) and `stage_sum_ms` are the same schema as decode.

**Schedule differences the report exposes** (documented, not bugs):

* chunk-major runs the whole MoE per chunk, so `moe.*` is chunk-tagged; layer-major
  batches the streamed `switch_mlp` across chunks (bank read once), so `moe.*`
  falls outside any chunk tag and appears only in the flat `stages`. Both schedules
  produce the **same flat stage set** (embed / attn split / HC / gate / switch /
  shared / combine / final_norm) — verified by test.
* Per-chunk `wall_ms`: chunk-major times the whole chunk (all layers + the
  cache-state eval); layer-major times the attention+HC half of chunk `c`
  accumulated across all layers (the MoE is outside the chunk tag).

## Analytical roofline (16,384-token prefill, chunk 1024 → 16 chunks)

Config (released 40-layer text path): `hidden_size` d = 5120, `moe_intermediate_
size` I = 2304, `num_attention_heads` H = 64, `head_dim` = 512, `n_routed_experts`
E = 384, `num_experts_per_tok` k = 6, `window_size` = 128, routed experts mxfp4
group-size 32.

**Attention — the `[rows,64,T]` f32 score transient (per layer-chunk).** The port
computes the score over the full appended window history `T` (then masks to the
128-token window), plus the reachable compressed rows on ratio-2 layers, so
`T_c ≈ 1.5·(c+1)·1024` and the score transient is `[1024, 64, T_c]` f32 = `1024 ·
64 · T_c · 4` bytes. At the last chunk (`T ≈ 24,576`) that is **6.00 GiB** — the
single largest transient, and why chunking bounds it to one chunk's rows.
Attention FLOPs (QK + AV, contracting `head_dim`) per layer-chunk = `4 · H ·
head_dim · rows · T_c`; summed over 16 chunks it is **28.0 TFLOP/layer**, **1.12
PFLOP over 40 layers**. Because compute is `O(rows · T)` with `T` = full history
(not the 128 window), attention is **quadratic in prompt length** — the growth the
`by_chunk` `attn.*.score` numbers make visible.

**mxfp4 routed gather at M = 1024 rows.** Per expert weight matrix I·d =
11,796,480 elements; mxfp4 gs32 = 0.5 byte/elem + 1 E8M0 scale byte per 32 elems ≈
0.531 byte/elem → **6.27 MB/matrix**, three matrices → **18.80 MB/expert**. Per
layer (E = 384 experts) = **6.72 GiB**; over 40 layers = **268.9 GiB** (=
288,777,830,400 B, exactly the W16 `experts.bin`). At ~13 GB/s that is a **~22 s**
one-time bank read (layer-major reads it once; chunk-major re-reads ≈ 8.8×, per the
routing-census memo). Gather FLOPs = `k · 3 · 2 · d · I` per token = **0.42
GFLOP/token**, `× 16,384 × 40` = **0.278 PFLOP** — flat per token (O(N)), ~¼ of
attention.

**Roofline vs measured.** Total compute ≈ 1.12 + 0.28 = **1.40 PFLOP**; at ~15
TFLOP/s effective that is **~93 s** — the ~100 s floor. Bank read ~22 s (overlaps
compute on layer-major). So of the measured **370 s (layer-major) / 490 s
(chunk-major)** TTFT, roughly **~250–370 s is overhead above the roofline** —
host-dispatch / eval-barrier / graph-build, which the fenced per-stage table (and
whether `attn.*.score` tracks the quadratic curve or a flat host cost dominates) is
built to localise. Chunk-major's extra ~120 s is consistent with its ~8.8× bank
re-read plus per-chunk MoE gate/shared recompute.

## Exact 16K window command

```
MTPLX_DSV41_PREFILL_CHUNK=1024 \
scripts/deepseek_v41/ab_decode_env_levers.py \
  --context-tokens 16384 --decode-tokens 0 \
  --arms control layer_major --prefill-stage-timing \
  --memory-limit-gib 60 --out <receipt.jsonl>
```

Runs inside `scripts/deepseek_v41/gpu_window.sh` (GPU flock held, Qwen unloaded,
memory-guarded). `control` is chunk-major, `layer_major` sets
`MTPLX_DSV41_PREFILL_LAYER_MAJOR=1`; each arm's `prefill_stage_timing` block is
appended to its receipt. The fences hold the per-chunk score transient (≤ 6 GiB at
chunk 1024) well inside the 60 GiB budget. Add `--stage-timing` for the decode
table in the same run (W37).

## Tests (CPU, tiny test-double, no artifact, MLX pinned to CPU)

* `tests/models/test_deepseek_v41_stage_timing.py` — prefill: probe off↔on
  byte-identical on **both** schedules; report schema + `chunks` + `by_chunk`;
  both schedules share the flat stage set; per-chunk `stage_sum ≈ wall`; a prefill
  session does not disturb the decode report; the `stage_nested` / `chunk` tagging
  mechanism (nested → `switch_breakdown`, out of the flat sum). All the W37 decode
  gates still pass unchanged.
* `tests/test_deepseek_v41_ab_env_levers.py` — `--prefill-stage-timing` parser +
  dry-run flow (flag + schedule env into the receipt); a small real prefill forward
  through `_prefill_stage_timing_pass` on the tiny double (chunk 4 → 3 chunks);
  session teardown.
