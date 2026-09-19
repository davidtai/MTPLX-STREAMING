# W116 — DSpark verify-attention ROWS census (per-row → shared-tile)

**2026-09-13 result:** the guarded [real-shape census](receipts/attention-census-20260913/README.md)
does not reproduce the hypothesized sixfold attention premium. Its isolated eager
pipeline proxy grows from 24.355 ms at M=1 to 33.664 ms at M=6 across the layer-type
counts. Those sums are not an end-to-end model measurement. The original W117
kernel proposal below remains an unvalidated hypothesis, not an approved next
implementation. Graph-op counts are unavailable because the export parser returned
zero for every case; they cannot prove engagement or absence of work.

Branch `w116/verify-attn-rows-census` off `int/w97f-lanes` (ec9ff2979). Builds the
census that names WHICH sub-op of the K+1 verify attention scales with the row count,
so W117 can replace it with one launch per layer handling all rows with the gathers
inside. **No A/B, no kernel yet** — this window lands the instrument and its CPU-smoke
proof; the GPU cell runs in the orchestrator's lock gap.

## 0. Historical hypothesis, superseded by the census above

Window 43, step 5 (16K cell, DSpark depth 5 = verify batch `rows = K+1 = 6`, plus the
attention stack):

| quantity | value |
|---|---|
| verify | 826 ms/cycle |
| in-barrier compute | 408 ms/cycle |
| verify **attention** | ≈ 372 ms/cycle = **9.3 ms/layer** (K+1 = 6 rows, 40 layers) |
| AR M=1 in-model attention | 64 ms/token = **1.6 ms/layer** |
| ratio | **≈ 6×** |

The selected bytes are tiny — `k = sliding_window 128 + index_topk 512 = 640` keys ×
`hd 512` × 2 B × 6 rows × 40 layers ≈ **157 MB/cycle ≈ 0.3 ms** at 500 GB/s
([[test-machines-bandwidth-file]]). This byte estimate originally motivated a
per-row gather / dispatch / core hypothesis in the small-M gathered core
(`_sparse_attend_selected`, `mtplx/models/deepseek_v41.py` ~L1130-1270): each of the
K+1 rows is handed its **own** gathered `[k, hd]` operand — the K29 kernel maps
`row → its own [rows*k, hd]` slice (`q.reshape(b*s,1,H,hd)`, `KVg.reshape(b*s,k,hd)`),
so the core reads `rows × k` keys even though the rows' windows overlap by `W − 1` and
their compressed selections are nearly identical (they share the accepted prefix's
index selection). W101 already made the five **projections** row-generic (fused, rows
≤ 8; W105 priced them at ≈ 415 µs/layer mxfp8). However, the 372 ms attribution was
inferred from a covering barrier, not measured as isolated attention. Per-row
operand growth does not establish a sixfold latency increase or its critical-path
share. K29's isolated M=6 core is slower than eager in the new census.

## 1. Hypotheses

- **H1 (primary).** The attention **core** (K29 fused / eager einsum / SDPA) scales
  ~linearly in `rows`: its operand is the per-row-materialised `[rows, k, hd]` KVg, so
  score `[rows, H, k]`, softmax and PV all grow with `rows`. The **gather** output
  (`[rows, W, hd]` window + `[rows, Ck, hd]` compressed) grows with `rows` too, but
  `_gather_rows` issues it as ONE flat `mx.take` of `b*s*k` rows — one dispatch whose
  *work*, not whose dispatch count, scales. So the census should show the core's and
  gather's **fenced ms rising ~linearly with rows while their op_count stays flat** —
  i.e. more per-row work per dispatch, not more dispatches.
- **H2.** `qkv_proj` and `out_proj` are ~flat (row-generic fused kernels, W101) — the
  small-M premium is not the projections.
- **H3.** `select` scales with `rows` only on **index-source** layers (full, reindex):
  `Indexer.select` scores `[rows, H, n_comp]` and argsorts per row. **reuse** and
  **swa_only** carry no per-token select (they read the source's published selection),
  so ~½ the 40 layers pay no select. The read-amplification lever is therefore the
  core + gather, not select.
- **H4 (the W117 target).** The rows' selected keys overlap heavily: the census's
  `k_union` (union of the K+1 rows' window + compressed selections) is ≈ `k + K`, not
  `rows × k`. A shared-tile kernel that reads `k_union` once and scores all rows
  against it targets ≈ 1.2–1.5× the M=1 read, i.e. verify attention ≈ 372 → ≈ 85–110
  ms/cycle.

## 2. What the census measures

One isolated production `Attention` layer per CSA mode at the exact cell shape
(from the artifact `config.json`: 40 layers, `hidden 5120`, **64 heads × head_dim
512**, `qk_rope_head_dim 64`, `q_lora_rank 1280`, `o_lora_rank 1024`, `o_groups 8`,
`sliding_window 128`, `index_n_heads 32`, `index_head_dim 128`, `index_topk 512`,
`compress_ratios [0,0, 2×18, 1×20, 0,0,0]`, `kv_source [2,8,14,20]`, `index_source
[2,8,14,20,24,28,32,36]`, `candidate_source 20`), `T = 16,384` KV, **mxfp8 gs32**
attention projections (indexer/compressor dense bf16 — the W105 in-model resident
layout), random weights, random pre-filled cache. Reuses the W78/W102 builders
(`metal_decode_attn_bisect._build_layer` / `build_case`) verbatim, so every sub-op is
the served production method — no re-implementation.

For each `rows ∈ {1, 2, 4, 6, 8}` and each mode, the verify batch is built at
absolute positions `[T … T+rows−1]` (the K+1 rows' window KV appended once, untimed,
so the window store holds the real T+rows verify geometry), then every sub-op is timed
**fenced** (`mx.eval` per op) **and pipelined** (one closing `mx.eval` over a chain of
`chain` calls, per-call; a distinct tiny perturbation per call breaks CSE —
[[queued-vs-eager-metal-microbench]]), **median of ≥ 7** (`--repeats 9`), on static
inputs (no cache mutation in the timed loop → no length drift):

| sub-op | production path timed |
|---|---|
| `qkv_proj` | `_qkv_prep_fused` (GPU, rows ≤ 8) / `_attn_qkv_prep` tape / eager |
| `select` | `Indexer.select` + `_mask_to_topk_idx` (index-source only) |
| `gather` | `_window_selected_idx` + `_gather_rows` + `_selected_compress_gather` + concat |
| `core_k29` | (a) `deepseek_v41_attn_kernels.fused_decode_attention` (per-row S=1 batch) |
| `core_eager` | (b) `_attn_core_impl` (eager gathered f32 einsum, the shipped block) |
| `core_sdpa` | (c) `mx.fast.scaled_dot_product_attention` on the per-row `[rows,k,hd]` operand, per-row valid mask + per-head value-0 `sinks=` — a "what a batched kernel would cost" proxy for the per-row layout (numerically a proxy: MLX's softmax/sink reassociates vs the reference core) |
| `out_proj` | `_out_prep_fused` / `_attn_out_prep` tape / eager (cached `wo_a`) |

Per sub-op per rows the receipt carries `fenced_ms`, `pipelined_ms`, and `op_count`
(`mx.export_to_dot` graph-node count — the dispatch-shape proxy for H1). Derived:
each sub-op's **fenced ratio vs rows=1**, the **implied ms/cycle for 40 layers at
rows=6** (each core + the fixed sub-ops), and the **`k_union` accounting**
(`k_union`, `rows×k`, `amplification = rows×k / k_union`) that sizes H4.

**The two Metal cores (`core_k29`, `core_sdpa`) are GPU-only in the census.**
`mx.fast.metal_kernel` (K29) — and the SDPA proxy — dispatch on the GPU whenever Metal
is available, *even when the default device is CPU*, so they run ONLY on a `--gpu`
cell; on a CPU-pinned run (the `--cpu-smoke` plumbing check and the unit test) they are
recorded `present:false` with a reason and **no Metal is ever touched** — the eager
`_attn_core_impl` carries the CPU plumbing. On the GPU cell a core that raises (e.g.
SDPA unsupported at head_dim 512) is likewise recorded absent with its reason rather
than crashing the census.

## 3. The exact GPU command (orchestrator, in the lock gap)

```
GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((24*1024**3)) \
  bash scripts/deepseek_v41/gpu_window.sh nice -n 19 \
  .venv/bin/python3 scripts/deepseek_v41/verify_attn_rows_census.py \
  --gpu --out docs/deepseek-v41/receipts/w116-verify-attn-rows-census.json
```

Runs on the CPU by default (an accidental worker run never touches Metal); `--gpu`
runs the real cell through the flock ([[gpu-work-always-through-flock]]).

**Expected runtime < 5 min.** 4 modes × 5 rows × (one T=16,384 prefill build +
8 sub-ops × ~(9 fenced + 9×16 pipelined + 1 op_count) calls). Each sub-op is
sub-millisecond at these dims; the dominant cost is the 20 prefill builds
(`_publish_compressed` over 16,384 tokens), ~0.1–0.4 s each on Metal ⇒ a few seconds,
plus a few seconds of measurement.

**Computed peak GPU memory (per resident layer, one at a time):**

| resident | bytes |
|---|---|
| mxfp8 gs32 attention projections (wq_a/wq_b/wkv/wo_a/wo_b ≈ 126.6M params, ≈1.06 B/elem) | ≈ 134 MB |
| dense bf16 indexer + compressor projections | ≈ 15 MB |
| window store `[1, T, 512]` bf16 | 16.7 MB |
| compressed KV `[1, n_comp, 512]` f32 (n_comp ≤ T) + index_k `[1, n_comp, 128]` f32 + comp_state frontier | ≈ 60 MB |
| transients: KVg `[8,640,512]` f32 (10.5 MB), scores `[8,64,640]` f32 (13 MB), select score `[1,8,32,16384]` f32 (16.7 MB), prefill wkv over `[16384,5120]` | ≈ 60–120 MB |

Per-layer resident ≈ **0.3 GB**; with the allocator pool + prefill spikes the run's
GPU peak is **< 1.5 GiB** — comfortably under the 20 GiB budget (and the 24 GiB host
RSS child cap). Only one layer is built at a time (freed with `del` + `gc.collect` +
`mx.clear_cache` between modes/rows), so peak does not grow with the sweep.

## 4. CPU smoke + test

`--cpu-smoke` runs tiny dims (T=256, `k = window 16 + index_topk 16 = 32`, `hd = 32`,
4 heads, 8 layers) on the CPU — the plumbing path the unit test drives. The eager core
runs; K29 (Metal) and, where unsupported, SDPA are recorded absent with a reason.

`tests/test_deepseek_v41_w116_rows_census.py` (CPU-pinned, tiny, no model load, no
model paths, < 1.5 GB RSS): CLI flags parse; the smoke run returns the schema (k=32,
hd=32, rows keyed by count, every sub-op carries **both** `fenced_ms` and
`pipelined_ms` + `op_count`); the Metal-only K29 core is recorded absent (not a crash);
`select` is absent on a non-index-source (reuse) layer; the `k_union` accounting and
the derived scaling are present; and the script references no `/Users/davidtai/models`
path and calls no model loader.

Run one file per process under nice, MLX pinned to CPU:

```
nice -n 19 .venv/bin/python3 -m pytest tests/test_deepseek_v41_w116_rows_census.py
```

## 5. Design sketch — the batched small-M core (W117, once the census names the sub-op)

If the census confirms H1 (the core + gather fenced ms rise ~linearly in rows at flat
op_count, and `amplification ≈ rows` while `k_union ≈ k + K`), W117 replaces the per-row
core with **one launch per layer that reads the shared KV tile once and scores all
rows against it**:

- **Operand.** Build ONE `[k_union, hd]` KV tile per layer — the union of the K+1 rows'
  selected keys (window rows `∪` selected compressed rows). Because the windows overlap
  by `W − K` and the compressed selection is shared across the block, `k_union ≈ k + K`,
  not `(K+1)·k`. Queries stay `[K+1, H, hd]`.
- **One launch, all rows.** Score `[K+1, H, k_union]`, apply one `[K+1, k_union]` valid
  mask (`row → its own reachable columns`: the intra-block causal band + the per-row
  window horizon + the compressed-selection membership), sink-softmax (per-head value-0
  sink, reference `_k_sparse_attn` form: max includes the sink, normalise after PV), PV.
  Grid = `rows × heads` threadgroups tiling over `k_union` with an **online softmax** so
  the shared KV tile streams **once** (the M=1 K29 tiling generalises to M query rows
  sharing the tile).
- **What changes in `_sparse_attend_selected`.** Today it builds the per-row `KVg
  [b, s, k, hd]` (window gather ∥ compressed gather) and either hands each row to K29 as
  its own S=1 batch or runs the eager per-row einsum. W117 instead: (1) build the
  **union index list** + the `[K+1, k_union]` membership/valid mask once (host-cheap,
  from `win_idx` ∪ `comp_idx`); (2) gather the `[k_union, hd]` tile once; (3) a new
  `fused_verify_attention(q[K+1,H,hd], KV[k_union,hd], mask[K+1,k_union], sink, scale)`
  metal kernel (a K29 generalisation: `rows × heads` threadgroups, shared KV tile,
  online softmax). The M=1 decode path is `k_union = k`, `rows = 1` — the existing K29
  kernel is the degenerate case, so decode is unchanged.
- **Gather.** The union gather is one `mx.take` of `k_union` rows (vs today's
  `rows × k`), so the gather read drops by ≈ the amplification factor too.
- **Rounding class.** The kernel's tiled reduction reorders the max/denom/value sums →
  reassociation-level vs the eager f32 core (`≤ 1e-6`, greedy-argmax identical),
  **not** byte-identical — the same class as K29 today ([[dsv41-inexact-ok-if-tie-flips]],
  [[dsv41-optimizations-not-naive]]). The selection itself (which keys) is byte-identical
  to the per-row path: the union is exactly the set the per-row gathers would read; only
  the softmax reassociates. Verify stays greedy-authoritative; the review gate is
  mechanism proof + real-path engagement counter + the census baseline this window lands.

Original speculative target: 372 → 85–110 ms/cycle. The new census invalidates
using 372 ms as an established attention baseline. Reprofile full-model graph
ancestry and scheduling before assigning a savings target or building this kernel.
