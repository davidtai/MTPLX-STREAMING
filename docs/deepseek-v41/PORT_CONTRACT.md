# DeepSeek-V4.1 MLX port — module split contract (W10 owns backbone)

Reference: `~/models/DeepSeek-V4.1-Flash-src/inference/model.py` (+ `engram.py`,
`kernel.py`, `config.json`). This file freezes the interfaces between the three
port modules so W10/W11/W13 build in parallel without guessing. Where a signature
deviates from the reference it is called out; deviations are the existing MTPLX
house conventions (args-first construction, `(indices, weights)` gate return,
external cache object) that the existing tests already encode.

## Ownership
- `mtplx/models/deepseek_v41.py` — **W10**: `ModelArgs`, `Model`, `DeepseekV41Backbone`,
  `DecoderLayer` (Block + hc threading), `Attention` (q/kv proj, o_lora grouped
  output, rope/yarn, `attn_sink`, SWA window call sites), `Compressor`, `Indexer`,
  `_select_candidate_blocks`, `_SharedRuntime` (reference `SharedAttentionRuntime`),
  final norm/head, `sanitize`, `attach_engram`, `make_cache`. Re-exports
  `DeepseekV41Cache`/`_LayerCache` (from W13's module) and keeps `_SharedRuntime`.
- `mtplx/models/deepseek_v41_moe.py` — **W11**: `MoE`, `MoEGate`, `Expert`/`ClampedSwiGLU`.
- `mtplx/models/deepseek_v41_cache.py` — **W13**: `DeepseekV41Cache`, `_LayerCache`.

W10 imports:
```python
from .deepseek_v41_moe import MoE
from .deepseek_v41_cache import DeepseekV41Cache, _LayerCache
```
Until W11/W13 land, W10 ships thin stubs of these two modules (faithful enough to
import + run the attention/hc/parity unit tests). On integration W10 merges
`feat/deepseek-v41-w11` / `-w13`; their files win over the stubs.

## deepseek_v41_moe.py (W11)
Mirror reference `Gate`/`Expert`/`MoE` math exactly; MLX house signatures:

- `class MoE(nn.Module)`; **`MoE(layer_id, args)`** (reference order — W11 landed this,
  superseding the earlier args-first draft). W10 constructs `self.mlp = MoE(layer_id, args)`
  in `DecoderLayer` and calls `moe(x)` with `x: [b, s, dim]` -> `[b, s, dim]`
  (`__call__(self, x, image_mask=None)`, image_mask unused on the text path).
  Required attributes (sanitize + streaming + `tests/models/test_deepseek_v41_parity.py`
  + `test_deepseek_v41_loader_contract.py` reach into these):
  - `moe.gate` : has `.weight [n_routed, dim]`, `.e_score_correction_bias [n_routed]`,
    `.topk` (int).
  - `moe.switch_mlp` : `SwitchGLU(hidden, moe_inter, n_routed,
    activation=ClampedSwiGLU(swiglu_limit))`; submodules `gate_proj`/`up_proj`/`down_proj`
    (SwitchLinear, `.weight [E, inter, dim]`). `bind_streamed_switches` REPLACES this
    attribute with a streaming switch whose `__call__(x[n,dim], indices[n,topk]) ->
    [n, topk, dim]`. The MoE forward MUST call `self.switch_mlp(xf, indices)` and get
    `[n, topk, dim]` back, then apply gate weights and sum.
  - `moe.shared_experts` : always-on clamped SwiGLU expert with `.w1`,`.w2`,`.w3`
    (nn.Linear, no bias). Clamp: up (`w3`) two-sided `[-limit,limit]`, gate (`w1`)
    upper `<= limit`; product `silu(w1)*w3` in fp32, cast back. (ref `Expert.forward`)
- `class Gate(nn.Module)`: `__call__(x_flat[n,dim]) -> (weights[n,topk], indices[n,topk])`
  — reference order `(weights, indices)` (W11's landed module; MoE.forward consumes it
  internally, so W10 never calls the gate directly). Score = `sqrt(softplus(x @ weight.T
  / gate_temp))`; select top-k of
  `scores + e_score_correction_bias`; weights gathered from the UNBIASED scores;
  if `norm_topk_prob and topk>1` divide by `sum(+1e-20)`; `*= routed_scaling_factor`.
- `class ClampedSwiGLU` : the `SwitchGLU` activation seam. `SwitchGLU` calls
  `activation(x_up, x_gate)` — first arg is UP (`w3`), second is GATE (`w1`).

### HARD W11 FINDING — the streamed clamp is dropped (leading 16/30 suspect)
The reference clamps EVERY routed expert (`MoE.__init__` passes `swiglu_limit=10.0`
to each `Expert`). W10's resident `SwitchGLU` path honours it via `ClampedSwiGLU`,
but the SERVED path (`bind_streamed_switches` -> `HotExpertSwitchGLU` /
`_gather_component_bank_mixed` / `_run_shadow_bank` / `_run_mapped_q4` in
`mtplx/models/expert_mlx.py`) runs a plain `swiglu(gate, up) = silu(gate)*up` with
NO clamp. So at serve time (the 31-token probe + the 1,024 generation, which use the
streamed q2 experts) the routed clamp is silently missing. W11 must make the clamp
reach the streamed path — either applied inside the MoE forward on the routed
gate/up before `down`, or via a seam in `expert_mlx.py` (shared infra — coordinate
with the coordinator before touching it). Confirm with the W10 ablation
(`.benchmark-artifacts/deepseek-v41/w10/ablate2*` arm `clamp_routed`).

## deepseek_v41_cache.py (W13)
External per-model cache; `Model.make_cache()` returns `DeepseekV41Cache(n_layers)`
and `mlx_lm.models.cache.make_prompt_cache(model)` returns it (it calls
`model.make_cache()`).

- `class DeepseekV41Cache`: `DeepseekV41Cache(n_layers)`; attrs `.layers`
  (list[_LayerCache], len n_layers), `.offset` (int, running token count, W10's
  backbone does `cache.offset += s`), `.engram_state` (default None; W10's
  `make_cache`/backbone set/advance it). Methods: `.mark() -> token`,
  `.rollback(mark)` (also trims `engram_state` by the decoded-token delta),
  `.trim(n)` (drop the last n decoded tokens: offset and every layer + engram).
- `class _LayerCache`: per-layer append-only KV the W10 attention reads/writes.
  Contract (attribute API — `test_deepseek_v41_parity.py` builds `_LayerCache()`
  directly and passes it to `attn(...)`):
  - `.window` : `[b, T_w, head_dim]` or None. W10 does `lc.window = _grow(lc.window, kv_new)`.
  - `.compress_kv` : `[b, n_comp, head_dim]` or None (appended at kv_source layers).
  - `.index_k` : `[b, n_comp, index_head_dim]` or None (appended at kv_source layers).
  - `.comp_state` : the compressor's partial-group tuple `(kv_acc, score_acc)` or None.
  - `.mark() -> token`, `.rollback(token)`.
  W13 MAY rewrite the internals to a bounded ring mirroring the reference
  `window_kv_cache`/`compress_kv_cache`/`k_cache` + `kv_state`/`score_state`, but MUST
  keep these attributes read/writable OR provide `append_window`/`append_compressed`
  methods AND update W10's call sites in the same merge (coordinate here first).

## Shapes / config (released 40-layer, from reference config.json)
dim 5120, n_layers 40, n_heads 64, head_dim 512, rope_head_dim 64, q_lora_rank 1280,
o_lora_rank 1024, o_groups 8, window_size 128, n_routed_experts 384, n_activated 6,
moe_inter 2304, score_func sqrtsoftplus, route_scale 1.5, swiglu_limit 10.0,
norm_eps 1e-20, compress_ratios [0,0,2×18,1×20,0,0,0][:40], kv_source [2,8,14,20],
index_source [2,8,14,20,24,28,32,36], candidate_source 20, candidate_topk_blocks 2048,
candidate_block_size 8, index_n_heads 32, index_head_dim 128, index_topk 512, hc_mult 4,
hc_sinkhorn_iters 20, hc_eps 1e-6, engram_layer_ids [1,14], compress_rope_theta 160000,
original_seq_len 65536, rope_factor 16, beta_fast 32, beta_slow 1.

## Loader contract (unchanged — W10 keeps)
`ModelArgs.from_dict(config)`; `Model(args, *, engram_bank_path=None, quantize=True)`;
`Model.sanitize(weights)`; `Model.load_weights(strict=True)` consumes exactly the
1,616 text residents; `Model.make_cache()`; `Model.attach_engram(engram_dir)`.
Engram hook contract: `engram_hook(hidden[B,L,hc_mult,dim], token_ids[B,L], cache_state)
-> hidden` (mtplx/engram_v41.py). Decode uses the same forward as prefill (one call
with a cache + start position).
