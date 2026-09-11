# DeepSeek-V4.1 port — cross-worker contract

Contract changes one worker needs another to honour. Each section is owned by the
worker that wrote it; the consuming worker reads it before integrating.

## W11 — MoE submodule (`mtplx/models/deepseek_v41_moe.py`), read by W10

W11 replaces the inline MoE section of `mtplx/models/deepseek_v41.py`
(`_SharedExpert` + `DeepseekV41MoE`) with a standalone, reference-faithful module.
For W10's `DecoderLayer` to use it, the following must hold:

1. **Import + construction (arg order changes).** The classes use the reference's
   names and the reference's constructor order `(layer_id, args)` — *not* the
   current `(args, layer_id)`:

   ```python
   from mtplx.models.deepseek_v41_moe import MoE   # reference class name
   ...
   self.mlp = MoE(layer_id, args)                  # was DeepseekV41MoE(args, layer_id)
   ```

   `args` is this port's `mtplx.models.deepseek_v41.ModelArgs` (unchanged). `MoE`
   reads only: `hidden_size`, `moe_intermediate_size`, `n_routed_experts`,
   `n_shared_experts` (asserted `== 1`), `num_experts_per_tok`, `scoring_func`,
   `norm_topk_prob`, `routed_scaling_factor`, `swiglu_limit`, and optional
   `gate_temp` (defaults to 1.0). No new config fields.

2. **Forward.** `mlp(x)` — `__call__(self, x, image_mask=None)`, `image_mask`
   unused on the text path. Input and output are the same shape (`[..., hidden]`),
   output dtype `== x.dtype`. Same call site as today (`x = self.mlp(x)` inside the
   FFN sublayer). No change needed in `DecoderLayer.__call__`.

3. **Attribute / parameter names are unchanged**, so the strict 1,616-key text
   load is preserved:
   - `mlp.gate.weight` (bf16), `mlp.gate.e_score_correction_bias` (f32)
   - `mlp.shared_experts.w{1,2,3}.{weight,scales,biases}` (resident q8, gs64 affine —
     dense `nn.Linear` at construction, converted by the model's `nn.quantize`)
   - `mlp.switch_mlp.{gate,up,down}_proj.weight` — the routed seam, **not resident**
     (streamed); `_is_resident_quant_module` already skips `"switch_mlp"`, and the
     text resident dict never contains it, so the strict load is unaffected.

   `bind_streamed_switches` rebinds `layer.mlp.switch_mlp` exactly as before.

4. **Gate return order.** `Gate.__call__` returns `(weights, indices)` (the
   reference order, L828), *not* `(indices, weights)` as the imported v4 `MoEGate`
   did. This only matters if W10 calls the gate directly; `MoE.forward` already
   consumes it internally. `indices` is int32 `[n, top_k]`; `weights` is f32.

5. **No import cycle.** `deepseek_v41_moe.py` imports only `mlx` and
   `mlx_lm.models.switch_layers`. It does **not** import `deepseek_v41.py`, so W10
   can import `MoE` from it without a cycle. W10 may drop the `_SharedExpert` /
   `DeepseekV41MoE` definitions and the `ClampedSwiGLU`/`MoEGate` imports from
   `deepseek_v4` once it switches to `MoE`.

If W10 prefers to keep the name `DeepseekV41MoE` at its call site, alias at import
(`from mtplx.models.deepseek_v41_moe import MoE as DeepseekV41MoE`) **and** flip the
constructor to `(layer_id, args)` — the arg order is the load-bearing change.



Signatures one worker must call across a module boundary. Each heading is owned
by the worker that provides the surface; callers pin to what is written here so
the parallel ports integrate without reading each other's in-progress code.

---

## W13 — per-sequence attention state (`mtplx/models/deepseek_v41_cache.py`)

W13 owns all per-sequence attention **state** and its update mechanics; W10 owns
the weight-bearing `Attention` / `Compressor` / `Indexer` / `Model` and calls
into this module. Every W13 method is weight-free (it takes the already-projected
arrays W10 produces) and carries the `inference/model.py` line range it
transliterates.

### Exports

| symbol | replaces W10 inline |
| --- | --- |
| `DeepseekV41Cache(n_layers, *, window_size=128, compress_ratios=None, kv_source_layer_ids=(), engram_state=None)` | inline `DeepseekV41Cache` |
| `LayerAttentionCache(window_size=128, compress_ratio=0, is_kv_source=False)` (alias `_LayerCache`) | inline `_LayerCache` |
| `SharedAttentionRuntime()` (alias `_SharedRuntime`) | inline `_SharedRuntime` |
| `CompressorState(ratio)` | inline `comp_state` tuple |
| `make_cache(model_or_n_layers, *, window_size=None, engram_state=None)` | new factory |
| `window_topk_idxs(window_size, bsz, seqlen, start_pos)` | reference `get_window_topk_idxs` |
| `ring_view(window, window_size, length)` | new (reference ring read) |

W10 should delete the inline `_SharedRuntime` / `_LayerCache` / `DeepseekV41Cache`
and `from .deepseek_v41_cache import (...)`; the field names (`window`,
`compress_kv`, `index_k`, `comp_state`, `offset`, `engram_state`, and the shared
runtime's `compress_kv` / `index_k` / `topk_mask` / `candidates`) are kept so the
existing `Attention` / `Backbone` bodies need no other change.

### Per-forward call sequence (reference → this module)

`shared = cache.new_shared_runtime()` once at the top of `Backbone.__call__`
(reference module-global `shared_attn`); `cache.advance(seqlen)` after the layers
(reference `start_pos += seqlen`). Positions map as
`start_pos = cache.offset`, `length = cache.offset + seqlen`.

Per layer, `layer = cache.layers[layer_id]`:

* **window** — `layer.append_window(kv_win)` with `kv_win` the post-RoPE window
  KV `[B, S, head_dim]`; read the ring with `layer.ring(cache.offset + S)` and
  the attend indices with `window_topk_idxs(window_size, B, S, start_pos)`
  (reference `_window_kv` L700-720, `get_window_topk_idxs` L409-426). The append
  keeps full history; `ring(length)` reproduces the reference
  `window_kv_cache[:bsz, :min(length,W)]` slot layout exactly.
* **compressor** (kv_source, `ratio > 1`) — `pooled = layer.comp_state.push(kv, score)`
  with `kv = wkv(x)`, `score = wgate(x)` (fp32); `pooled` is `[B, g, head_dim]`
  pre-norm, pre-RoPE (reference L466-485). W10 applies `Compressor.norm`, RoPE at
  the group positions and (phase 2) quant, then `layer.append_compress(compress_new)`
  (reference `compress_kv_cache` write L761) and, for the index keys,
  `layer.append_index_k(index_new)` (reference `Indexer.k_cache` write L547).
* **compressor** (kv_source, `ratio == 1`) — `layer.comp_state is None`; W10
  projects one latent per token (reference L461-462) and calls `append_compress`
  / `append_index_k` directly, one row per token.
* **publish** (source layers) — `shared.compress_kv = layer.compress_kv`,
  `shared.index_k = layer.index_k`, `shared.topk_mask = <selection>` (aliases
  `shared.topk_idxs`), `shared.candidates = <mask>` (reference
  `SharedAttentionRuntime` L1166-1180).
* **reuse / reindex** — read `shared.compress_kv` / `shared.index_k` /
  `shared.topk_mask` / `shared.candidates`.

**Engram** — `Backbone` calls `cache.engram_state.advance(input_ids)` before the
engram layers (unchanged); `cache.engram_state` is set by `Model.make_cache`.

### Rollback seam (serve / gate / speculative verify)

* `cache.trim(n) -> int` — restore to `n` tokens earlier (window, compress_kv,
  index_k, compressor frontier, engram history, and `offset` together); returns
  the count trimmed (`mlx_lm` convention).
* `cache.mark()` / `cache.rollback(mark)` — snapshot form.
* `cache.is_trimmable() -> True`; `cache.offset: int`.

### `make_cache`

Keep `mlx_lm.models.cache.make_prompt_cache(model)` → `model.make_cache()` and
`model(ids, cache=cache)` working. W10's `Model.make_cache(self)` becomes:

```python
from .deepseek_v41_cache import make_cache

def make_cache(self):
    engram_state = self.model.engram_hash.fresh() if self.model.engram_hash is not None else None
    return make_cache(self.args, engram_state=engram_state)
```

`make_cache` accepts a `ModelArgs`-like object (`num_hidden_layers`,
`window_size`, `compress_ratios`, `kv_source_layer_ids`) or a plain `n_layers`
int. `runtime.py`'s `configure_owned_recurrent_state_cache` /
`configure_tail_owned_attention_kv_cache` are pass-throughs unless their env
flags are set, so the single-object cache flows through unchanged.

## W23 — DSpark MTP draft head (`mtplx/models/deepseek_v41_dspark.py` + the MTP surface on `mtplx/models/deepseek_v41.py`), read by the runtime MTP path (`mtplx/generation.py`) and the loader

The DSpark 3-stage speculative draft head and the model-side surface the native
MTP lane drives. Builds on **W13**'s rollback seam and **W22**'s mlx_lm per-layer
cache conformance (merged into `feat/deepseek-v41-w23`); no cache file was changed
by W23.

### Exports

| symbol | shape / contract |
|---|---|
| `DSparkHead` (`deepseek_v41_dspark`) | `layers` = 3 `DSparkBlock` stages; `draft_block(main_hidden, input_ids, caches, embed, head) -> (output_ids [b, block_size+1], logits [b, block_size, vocab], confidence [b, block_size])`; `seed_main(main_hidden, caches)` appends committed main KV |
| `DSparkStageCache` | per-stage sliding-window KV of the *main* hiddens; answers the W13 rollback seam (`trim(n)->int`, `mark()`, `rollback(mark)`, `is_trimmable()`, `offset`). W23 owns it — no MTP-stage cache was contracted before. |
| `Model.mtp_forward / mtp_update_cache / make_mtp_cache / mtp_blocks / has_mtp / hc_hidden` | the uniform `MTPLXRuntime.draft_mtp/update_mtp_cache/make_mtp_cache` surface (mirrors `deepseek_v4`); `__call__(return_hidden=True)` returns `main_hidden` = concat of the `dspark_target_layer_ids` hiddens |
| `is_deepseek_v41_mtp_config` / `inject_deepseek_v41_mtp_support` | runtime MTP dispatch (a dedicated arm in `runtime.py`; the v4 predicate/injector are not reusable) |
| `Model(mtp=True)` / loader `resolve_with_mtp` / `partition_text_residents(with_mtp=)` | opt-in head build + `mtp.*` resident keep/map; default OFF (phase-1 AR unchanged) |

### Forward / verify contract
- **Lossless bar** = greedy verify == AR argmax (NOT bit-exactness). The runtime's
  batched target verify is authoritative; the draft (block-draft served per-depth
  via a stash) only sets the acceptance rate. Proven: `generate_mtpk` == `generate_ar`
  at K=1,2,3 over 64 tokens (`tests/models/test_deepseek_v41_dspark.py`).
- **Streaming verify (R2).** The batched verify hands all K+1 rows to each backbone
  MoE layer in ONE `switch_mlp(xf, indices)` call — the precondition for record
  dedup across the routed-expert union. K>3 / tree verify are DEAD (wider union).
  Gate 0: median union u ≤ 10 at K=3 (measure on the real bank).

### Consumers must know
- The DSpark MoE experts are RESIDENT mxfp4, run through the mlx-lm SwitchGLU
  quantised path (the task's `resident SwitchGLU with the ±10 clamp`; `mx.gather_qmm`
  in mlx 0.32.2 does take `mode=`, the other option). The backbone switch binder
  walks only `model.model.layers`, so it leaves the head's experts resident.
- **Serve-path glue gap:** `--generation-mode mtp` reaches the loader today via
  `MTPLX_DSV41_MTP=1`. The one-line map from the CLI flag to `with_mtp=True` lives in
  `cli.py` / `resident_loader.py` (outside W23's allowlist) — a follow-up for whoever
  owns those.
