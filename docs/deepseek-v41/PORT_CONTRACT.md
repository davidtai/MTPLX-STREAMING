# DeepSeek-V4.1 port — cross-worker call contract

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
