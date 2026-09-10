# DeepSeek-V4.1-Flash torch-reference goldens (W9)

Float32 CPU ground truth from the **reference** `inference/model.py` + `engram.py`
(kernels replaced by pure-torch float32 equivalents; all projection weights
dequantized from the source HF shards `~/models/DeepSeek-V4.1-Flash-src`).
Produced by `scripts/deepseek_v41/torchref/ref_forward.py` on the 31-token probe:

```
ids = [0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291, 6328, 3465, 1241,
       6036, 14, 291, 3395, 361, 1354, 260, 565, 291, 6328, 3465, 21740, 6036, 14, 291, 2605]
```
(BOS id 0 + `tok.encode("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef mul(a, b):")`).

Everything is float32. The full float32 buffers referenced by each summary's
`npy` field live under `.benchmark-artifacts/deepseek-v41/w9/*.npy` (git-ignored);
regenerate with `python scripts/deepseek_v41/torchref/ref_forward.py --max-layer 2`.

## Summary block (`summarize()`)
Every tensor is recorded as a block:
`{name, shape, dtype:"float32", count, mean, std, max_abs, finite, sha256_f32,
  per_pos_feat_len, per_pos_first64 [S][64], per_pos_norm [S], first64_last_pos [64], npy}`.
`sha256_f32` is the sha256 of the C-contiguous float32 buffer — an exact-match key
across implementations. `per_pos_first64[p]` is the first 64 values of position `p`
flattened over all trailing (head/hc) axes.

## Files

| file | submodule | key tensors (shape) |
|------|-----------|---------------------|
| `torchref_golden_attn_L{0,1,2}.json`   | attention | `input_post_hc_norm` [1,31,5120] (post attn-RMSNorm x into `attn`), `output` [1,31,5120] (attn out, pre hc_post), `window_kv` [1,31,512] (RoPE'd window latent after 31 tokens). L2 also: `compressed_kv` [1,15,512], `index_k` [1,15,128], `topk_idxs` [1,31,15] int64 (per-query selected compressed-row ids, offset by window length 31; -1 = none) |
| `torchref_golden_moe_L{0,1,2}.json`    | MoE       | `input` [1,31,5120] (post ffn-RMSNorm x into MoE), `output` [1,31,5120] (routed+shared sum), `shared_expert_output` [1,31,5120], `router.topk_ids` [31,6] int, `router.topk_weights` [31,6] (gathered scores, norm_topk_prob'd, ×route_scale 1.5) |
| `torchref_golden_engram_L1.json`       | engram    | `input_pre_engram` [1,31,4,5120], `output_post_engram` [1,31,4,5120] (h + gate·value), `gate` [1,31,4], `value` [1,31,5120], `row_ids` [1,31,24] int64 (n_hash_cols = (max_ngram−1)·n_heads = 24, laid out ngram-major/head-minor) |
| `torchref_layers012.json`              | per-layer | residual-stream output of each block (`layers[i]`, [1,31,4,5120] collapsed to first64_last_pos + stats); `embed` = hc-expanded token embedding entering layer 0. Superset of the `compare_hidden_states.py` schema (`final_norm`/`argmax_token`/`logits_top8` are null — 3-layer depth cannot produce final logits) |

## Wiring notes for porters
- The reference applies the **engram at the START of an engram layer** (before the
  block), on the layer's *input* residual stream (= previous layer's output). Layer 1
  order is: `h = engram(h); h,pre_mix = block(h)`.
- `input_post_hc_norm` is `attn_norm(hc_pre(h, pre_mix))` — i.e. AFTER the hyper-connection
  collapse and the attention RMSNorm, exactly the tensor fed to `Attention.forward`.
- MoE `input` is `ffn_norm(hc_pre(h, attn_pre))`.
- The residual stream is `hc_mult=4` parallel copies; sublayer in/out tensors are the
  single collapsed copy, layer outputs are the 4-copy stream.
- Attention is MLA: one shared 512-dim KV latent, 64 query heads, per-head `attn_sink`
  (a value-0 softmax slot). Layers 0,1 are sliding-window-only (`compress_ratio=0`,
  rope_theta 10000, no YaRN); layer 2 is the first Full layer (`compress_ratio=2`,
  kv_source + index_source, YaRN at compress_rope_theta 160000).
