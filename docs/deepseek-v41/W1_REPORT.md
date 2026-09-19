# W1 — DeepSeek-V4.1-Flash text AR forward (P1.0–P1.3)

Branch `feat/deepseek-v41-w1` (based on fork main `3f7d6fe9` + the V4.1 converters),
merged with `feat/deepseek-v41-streaming` at `5e5ef655` (no conflicts).
File: `mtplx/models/deepseek_v41.py` (939 lines). Tests: `tests/models/test_deepseek_v41_{config,parity,loader_contract}.py`.

Status: **all W1 tests pass** on CPU under `nice -n 19`; the real 8.6 GB q8 artifact
loads strictly (1,616 kept residents, `bound_sparse_layers: 40`); attribution check clean.

## Scope delivered
- **P1.0** ModelArgs (`deepseek_v41`/`deepseek_v41_text`, nested-config aware) + the §0
  CSA2 per-layer mode table (`Full/Reindex/Reuse/SWA-only`) from `kv_source_layer_ids`,
  `index_source_layer_ids`, `compress_ratios`.
- **P1.1** Dense block: embedding, Hyper-Connection (mixes+Sinkhorn) with the reference's
  `pre_mix` threaded across sublayers, `MoEGate` (sqrtsoftplus/noaux_tc), `ClampedSwiGLU`
  (swiglu_limit), shared expert, RMSNorm, head.
- **P1.2** Attention: MLA q/kv (64 heads, head_dim 512, qk_rope 64, 1 KV head), grouped
  `wo_a`/`wo_b` o-LoRA (o_groups 8), per-head attention sink [64], sliding window 128
  (phase-1 bf16 window, no FP8 quant — realised as full-history + sliding mask, exactly
  equal to the reference ring for the positions any query can still reach).
- **P1.3** CSA2: `Compressor` at kv_source layers, `Indexer` at index_source layers,
  `select_candidate_blocks` at candidate_source layer 20, a `_SharedRuntime` publishing
  compressed KV / index K / top-k / candidates from source layers to Reuse/Reindex layers,
  a compressed+index cache, and the trim/rollback seam.
- Routed experts exposed as the `mlp.switch_mlp` seam `bind_streamed_switches` rebinds;
  resident `SwitchGLU` fallback for tests.
- Engram left as a `layer.engram_hook` slot (default None) with the merged engram's
  3-argument signature + advance/rollback threading.
- **Not in scope (untouched):** MTP/DSpark, vision, FP4/FP8 KV quant, real streaming I/O.

## Reused (imported, not copied) vs new
Imported from `mtplx/models/deepseek_v4.py` where the math is byte-identical (deepseek_v41.py:40):
| Symbol | Used for | Why identical |
|---|---|---|
| `_yarn_inv_freq` | YaRN inverse frequencies | matches reference `precompute_freqs_cis` (model.py:369) incl. the ramp |
| `_apply_interleaved_rope` | adjacent-pair complex RoPE (fwd + inverse) | matches reference `apply_rotary_emb` (model.py:392) |
| `hc_split_sinkhorn` + `_sinkhorn_ops` (via it) | mixes→(pre,post,comb) split | transcribes `hc_split_sinkhorn_kernel` (kernel.py:407) |
| `_hc_post_impl` | HC `post` expand+mix | matches reference `Block.hc_post` (model.py:962) |
| `MoEGate` | routed gate (sqrtsoftplus/noaux_tc/norm_topk/route_scale) | matches reference `Gate` (model.py:792); `num_hash_layers=0` forces the score branch |
| `ClampedSwiGLU` | routed-expert activation clamp | matches reference `Expert` clamps (model.py:841); confirmed `SwitchGLU` calls `activation(x_up, x_gate)` and `SwiGLU(x,gate)=silu(gate)*x` |

New in deepseek_v41.py (reference model.py disagrees with V4, so written fresh):
- HC threading in `DecoderLayer.__call__` (717) + `_mixes`/`_hc_pre` (701/711): the reference
  threads `pre_mix` across sublayers and returns `ffn_pre` (model.py:968–994); V4's
  `HyperConnection.pre` collapses within the same sublayer, so it could not be reused.
- `Attention` (441): sparse attention as one masked softmax over `[window ++ compressed]`
  with a per-head sink (equivalent to reference `sparse_attn`, kernel.py:392), grouped
  `_o_lora_down` (593), the ring window as full-history+mask.
- `Compressor` (265): plain gated pool (reference model.py:429), **no** `ape`/window-overlap
  (V4's compressor has both, so it was not reused); `prefill` + decode `step`.
- `Indexer` (329) + `_topk_rows` (392) + `_select_candidate_blocks` (411): the two-level
  hierarchical top-k with the candidate prefilter (reference model.py:488/583) — absent in V4.
- `_SharedRuntime` (248) + the mode dispatch in `_compressed` (531): cross-layer sharing
  (reference `SharedAttentionRuntime`, model.py:1166) — absent in V4.
- `DeepseekV41Cache`/`_LayerCache` (773/741): append-only window/compressed/index rows +
  compressor partial-group state, with a mark/rollback seam.
- ModelArgs/mode table (66/158), `_sanitize_name` (845), q8 self-quantize (863/885).

## Exact resident tensor names consumed (read from the artifact index + q8 converter)
Per text layer (×40) unless noted; q8 = affine 8-bit gs64 (weight `U32` packed + `scales`/`biases` `BF16`); norms/hc/gate/sink stay `BF16`/`F32`. `sanitize` maps each checkpoint name onto the module path shown.

| Checkpoint name | Module path | Format |
|---|---|---|
| `embed.{weight,scales,biases}` | `model.embed_tokens.*` | q8 (QuantizedEmbedding) |
| `head.{weight,scales,biases}` | `head.*` | q8 |
| `norm.weight` | `model.norm_weight` | bf16 (bare array) |
| `layers.N.{attn_norm,ffn_norm}.weight` | `model.layers.N.{attn_norm_weight,ffn_norm_weight}` | bf16 |
| `layers.N.hc_{attn,ffn}_{fn,base,scale}` | `model.layers.N.hc_*` | f32 (bare arrays) |
| `layers.N.attn.attn_sink` | `…attn.attn_sink` | f32 [64] |
| `layers.N.attn.{q_norm,kv_norm}.weight` | `…attn.{q_norm_weight,kv_norm_weight}` | bf16 |
| `layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scales,biases}` | `…attn.*` | q8 |
| `layers.N.attn.compressor.{wkv,wgate}.{weight,scales,biases}` | `…attn.compressor.*` | q8 (`wgate` only on ratio-2 kv_source L2/8/14; L20 ratio-1 has no gate) |
| `layers.N.attn.compressor.norm.weight` | `…attn.compressor.norm_weight` | bf16 (kv_source L2/8/14/20) |
| `layers.N.attn.indexer.{wq_b,weights_proj}.{weight,scales,biases}` | `…attn.indexer.*` | q8 (index_source L2/8/14/20/24/28/32/36) |
| `layers.N.attn.indexer.wk.{weight,scales,biases}` | `…attn.indexer.wk.*` | q8 (kv_source only) |
| `layers.N.attn.indexer.k_norm.weight` | `…attn.indexer.k_norm_weight` | bf16 (kv_source only) |
| `layers.N.ffn.gate.weight` | `model.layers.N.mlp.gate.weight` | bf16 (bare array, **not** quantized) |
| `layers.N.ffn.gate.bias` | `…mlp.gate.e_score_correction_bias` | f32 |
| `layers.N.ffn.gate.bias_vl` | **dropped** (text path) | — |
| `layers.N.ffn.shared_experts.w{1,2,3}.{weight,scales,biases}` | `…mlp.shared_experts.w{1,2,3}.*` | q8 |
| `layers.N.ffn.experts.E.w{1,2,3}.*` | **not resident** — streamed bank (`mlp.switch_mlp`) | fp4→Q2 |

Verified end to end: the loader's `construct_deepseek_v41_resident_model` on the real artifact
loads **1,616 kept residents / 8,673,203,648 B** strictly (the 40 `bias_vl` dropped → 1,576
model params), `strict: True`, `bound_sparse_layers: 40`.

## Interface contracts
**Switch module (streamed routed experts).** `model.model.layers[i].mlp` is a `DeepseekV41MoE`
with `.gate` (MoEGate → `(indices, weights)`), `.switch_mlp` (the seam, resident `SwitchGLU`
fallback), and `.shared_experts`. `__call__` computes `routed = switch_mlp(x, indices)` →
`(routed * weights[...,None]).sum(-2) + shared_experts(x)`. `bind_streamed_switches` replaces
`mlp.switch_mlp` with a runtime switch (verified: HotExpertSwitchGLU on all 40 layers). Constructor:
`Model(model_args, *, engram_bank_path=None, quantize=True)`; `quantize=True` self-quantizes the
resident projections to q8 gs64 (skips the bf16 router gate and the streamed `switch_mlp`); tests
pass `quantize=False` for the dense oracle. `sanitize()` performs the name mapping above.

**Engram hook.** `DecoderLayer.engram_hook` defaults to `None`. When set (by the engram worker on
layers 1 and 14), it is called at the reference's pre-attention insertion point as
`engram_hook(hidden_states, token_ids, cache_state) -> mx.array` where `hidden_states` is the
hc-expanded stream `[B, L, hc_mult, dim]` and the return replaces the stream (`h + gate*value`).
The backbone advances the engram state once per step (`cache.engram_state.advance(token_ids)`)
before any engram layer reads it; `cache.engram_state` (default None) is the `NgramHashState`-like
object, and `DeepseekV41Cache.rollback` calls `engram_state.trim(n)` in step with the KV rollback.
`Model.engram_bank_path` stores the constructor arg for the engram worker's wiring (this module
does not build engram). Verified against fakes in `test_engram_hook_wiring_and_rollback`.

## Reference-vs-plan discrepancies (reference wins, per the altitude rule)
1. **No HeadHC / hc_head.** The plan (§2) lists `HeadHC` as carry-over, but the V4.1 reference
   collapses the final hc copies with the last threaded `pre_mix` (`layer.hc_pre(h, pre_mix)`,
   model.py:1268) and `ParallelHead` has no hc_head; the artifact has no `hc_head` tensor. Ported
   without HeadHC.
2. **HC threading differs from V4.** The reference threads `pre_mix` across sublayers (attn uses the
   previous ffn's pre; ffn uses this attn's pre; returns `ffn_pre`), unlike V4's same-sublayer
   collapse — so `HyperConnection` was not reused wholesale (only its leaf math).
3. **Compressor has no `ape`/overlap.** The V4 `Compressor` adds an absolute-position embedding and
   ratio-4 window overlap; the V4.1 reference (model.py:458) is a plain gated pool. Written fresh.
4. **HC mixes eps.** The reference rsqrt-normalises the mixes with `norm_eps` (1e-20) and uses
   `hc_eps` (1e-6) only inside the Sinkhorn split (model.py:948 + kernel.py:427); matched exactly.
5. **`num_nextn_predict_layers`/MTP.** `compress_ratios` has 43 entries (40 text + 3 MTP); only the
   first 40 drive the text modes. MTP out of scope.
6. **Phase-1 drops the reference's FP4/FP8 activation-quant** on the compressed latent, index q/k and
   window KV (matches the plan's phase-1 bf16 decision; the oracle drops it too).

## Environment / method note
torch and numpy are absent from the homebrew python; the project venv
(`…/mtplx-hy3-ssd/.venv`, py3.12, mlx 0.32.2, numpy 2.4.4, mlx_lm, pytest 9) has numpy but **no
torch**, so the parity oracle is a float64 numpy transcription of `inference/model.py` (stated in
the test file). MLX's CPU GEMM rounds matmul inputs to tf32 (~8e-4 rel), so the oracle emulates
that (10-bit-mantissa matmuls) and the tests assert exact argmax parity except at genuine top-2
ties within the oracle's error band.

## Test results
```
tests/models/test_deepseek_v41_config.py::test_layer_modes PASSED
tests/models/test_deepseek_v41_config.py::test_from_dict_accepts_bare_text_config PASSED
tests/models/test_deepseek_v41_parity.py::test_attention_swa_only PASSED
tests/models/test_deepseek_v41_parity.py::test_dense_block_parity PASSED
tests/models/test_deepseek_v41_parity.py::test_csa2_modes PASSED
tests/models/test_deepseek_v41_parity.py::test_rollback PASSED
tests/models/test_deepseek_v41_loader_contract.py::test_construct_interface PASSED
tests/models/test_deepseek_v41_loader_contract.py::test_sanitize_roundtrip_and_drops PASSED
tests/models/test_deepseek_v41_loader_contract.py::test_strict_q8_load PASSED
tests/models/test_deepseek_v41_loader_contract.py::test_engram_hook_wiring_and_rollback PASSED
10 passed, 2 warnings in 0.81s
```

Merged-branch loader + spec suites (coordinator request):
```
tests/test_deepseek_v41_spec.py ..... (5 passed)
tests/test_deepseek_v41_loader.py ................ (17 passed, 1 failed)
1 failed, 17 passed in 13.81s
```
The single failure is `test_deepseek_v41_loader.py::test_construct_is_wired_and_guarded_until_w1`,
which asserts `construct_deepseek_v41_resident_model` raises "until W1 lands". Now that this W1
module lands and loads the real artifact successfully, it fails with *"DID NOT RAISE"* — the
correct signal that W1 is wired. That test is owned by W3 (`tests/test_deepseek_v41_loader.py`, not
in this task's allowlist) and should be updated to assert a successful construct. All other loader
tests (incl. the real 40-layer bind + text-only 1,616/8.67 GB resident plan) pass.
```
real strict-load probe: LOAD OK — tensor_count 1616, raw_tensor_bytes 8673203648,
                        bound_sparse_layers 40, strict True, engram_bank_path set
```

Attribution: `check_ai_attribution.py --range origin/main..HEAD` → `clean`.
