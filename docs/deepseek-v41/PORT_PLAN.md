# DeepSeek-V4.1-Flash → MTPLX SSD-streamed MoE (Q2 experts) — Port Plan

Status: design only. No production code lands from this document. Author: Opus 4.8 worker.
Scope: text-only autoregressive serving through the existing MTPLX expert-streaming lane.
Vision tower, aligner, and image tokens are out of scope.

Do not treat this file as the spec source. The concurrent worker writes `scripts/` and the
`ExpertStreamingModelSpec` entry in `mtplx/expert_streaming_models.py`. This document proposes
the spec fields (section 5) for that worker to consume; it does not edit that file.

## 0. Source facts (pinned)

| Fact | Value | Source |
|---|---|---|
| model_type | `deepseek_v41` (text: `deepseek_v41_text`) | `config.json` |
| Backbone params | 552 B | tech report |
| Engram params | 196 B (separate table) | tech report |
| Activated params / token | 8 B prefill, 16 B decode | tech report |
| Layers | 40 text = 20 causal encoder + 20 decoder; + 3 DSpark MTP | report + index |
| hidden_size | 5120 | config |
| moe_intermediate_size | 2304 | config |
| Routed experts | 384, top-6, +1 shared | config |
| Attention | MQA-shaped MLA: 64 heads, head_dim 512, qk_rope 64, 1 KV head | config |
| q_lora_rank / o_lora_rank / o_groups | 1280 / 1024 / 8 | config |
| Gate | scoring `sqrtsoftplus`, `noaux_tc`, norm_topk, route_scale 1.5 | config |
| swiglu_limit | 10.0 | config |
| Sliding window | 128 | config |
| KV cache precision | FP4 global KV + index K, FP8 window | report + `model.py` |
| Dense weight format | FP8 E4M3, UE8M0 32×32 block scales | `quantization_config` |
| Expert weight format | FP4 E2M1 (2/int8), E8M0 scales per 32 | `quantization_config` |
| Norms / hc / gate.bias / attn_sink | bf16 / f32 | `model.py` |
| Total tensors in index | 96,085 | `model.safetensors.index.json` |

Reference sources read for this plan:
- DeepSeek MIT reference `inference/{model.py,engram.py,convert.py,generate.py,config.json}` and
  `DeepSeek_V41_Tech_Report.pdf`.
- Vontra independent MLX text runtime `runtime/{runtime.py,dspark.py,generate.py}` (434-line
  `runtime.py`; the full V4.1 text forward path in MLX). Cited by line below.
- MTPLX `mtplx/expert_streaming_models.py`, `mtplx/expert_manifest.py`,
  `docs/advanced/ssd-streamed-moe.md`.
- V4 (older 284 B `deepseek_v4`) backend: `feat/deepseek-v4-flash:mtplx/models/deepseek_v4.py`
  (3,440 lines).

### CSA2 layer-mode map (drives attention dispatch and KV planning)

`compress_ratios` (43 entries = 40 text + 3 MTP), `kv_source_layer_ids=[2,8,14,20]`,
`index_source_layer_ids=[2,8,14,20,24,28,32,36]`, `candidate_source_layer_id=20`.

| Layers | ratio | Mode | Owns global KV | Owns index Q | Notes |
|---|---:|---|:--:|:--:|---|
| 0, 1 | 0 | SWA-only | — | — | encoder local; no compressed KV |
| 2 | 2 | Full | yes | yes | serves layers 2–7 |
| 3–7 | 2 | Reuse | no | no | reuse layer 2 KV + top-K |
| 8 | 2 | Full | yes | yes | serves 8–13 |
| 9–13 | 2 | Reuse | no | no | |
| 14 | 2 | Full | yes | yes | serves 14–19 |
| 15–19 | 2 | Reuse | no | no | |
| 20 | 1 | Full | yes | yes | **CED boundary**: decoder global KV projected here from final encoder states; `candidate_source` (block prefilter) |
| 21–23 | 1 | Reuse | no | no | reuse layer 20 KV + top-K + candidates |
| 24, 28, 32, 36 | 1 | Reindex | no | yes | reuse layer 20 KV; own indexer Q; masked to layer-20 candidate blocks |
| 25–27, 29–31, 33–35, 37–39 | 1 | Reuse | no | no | |

CED = the compressor at layer 20 projects the decoder's persistent global KV from the encoder's
final hidden states. Mechanically this is `Compressor` + shared-cache reuse (not a distinct
module). The Full/Reindex/Reuse split is realized by set membership in `kv_source_layer_ids` and
`index_source_layer_ids`, not by a separate encoder/decoder code path.

---

## 1. Tensor inventory → module map (text path)

114 distinct name patterns in the index. Per-layer patterns are ×40 unless noted. Shapes below are
derived from `model.py` module constructors and config; only shard 1 (small dense tensors) had
finished downloading, so quantized-tensor shapes are architectural derivations, not read headers
(flagged where load-bearing).

Placement legend: **R** = resident (wired ≤100 GiB budget), **S** = streamed (expert bank),
**D** = disk-backed (engram table, gathered per token).

| Tensor pattern | Logical shape | Source format | Consumer module | Place |
|---|---|---|---|:--:|
| `embed.weight` | [129280, 5120] | bf16 | token embedding | R |
| `head.weight` | [129280, 5120] | bf16 (fp32 at use) | `ParallelHead` | R |
| `norm.weight` | [5120] | f32 | final RMSNorm | R |
| `layers.N.attn_norm.weight` | [5120] | f32 | `Block` attn pre-norm | R |
| `layers.N.ffn_norm.weight` | [5120] | f32 | `Block` ffn pre-norm | R |
| `layers.N.attn.wq_a.{weight,scale}` | [1280, 5120] | FP8 32×32 | `Attention` q down-proj | R |
| `layers.N.attn.q_norm.weight` | [1280] | f32 | q-latent RMSNorm | R |
| `layers.N.attn.wq_b.{weight,scale}` | [32768, 1280] | FP8 32×32 | q up-proj (64×512) | R |
| `layers.N.attn.wkv.{weight,scale}` | [512, 5120] | FP8 32×32 | KV proj (1 head, dim 512) | R |
| `layers.N.attn.kv_norm.weight` | [512] | f32 | KV RMSNorm | R |
| `layers.N.attn.wo_a.{weight,scale}` | [8192, 4096] | FP8 32×32 (ref → bf16) | grouped out down-proj (block-diag over o_groups=8) | R |
| `layers.N.attn.wo_b.{weight,scale}` | [5120, 8192] | FP8 32×32 | out up-proj | R |
| `layers.N.attn.attn_sink` | [64] | f32 | per-head attention sink | R |
| `layers.N.attn.compressor.wkv.weight` | [512, 5120] | bf16 (ratio 1) / f32 (ratio>1) | `Compressor` KV pool | R (×4: L2,8,14,20) |
| `layers.N.attn.compressor.wgate.weight` | [512, 5120] | f32 | `Compressor` softmax pool gate | R (×3: L2,8,14) |
| `layers.N.attn.compressor.norm.weight` | [512] | f32 | latent RMSNorm | R (×4) |
| `layers.N.attn.indexer.wq_b.{weight,scale}` | [4096, 1280] | FP8 32×32 | `Indexer` query (32×128) | R (×8: index_source) |
| `layers.N.attn.indexer.weights_proj.weight` | [32, 5120] | bf16 | per-head index importance | R (×8) |
| `layers.N.attn.indexer.wk.weight` | [128, 512] | bf16 | index key from latent | R (×4: kv_source) |
| `layers.N.attn.indexer.k_norm.weight` | [128] | f32 | index-key RMSNorm | R (×4) |
| `layers.N.ffn.gate.weight` | [384, 5120] | bf16 | `MoEGate` router | R |
| `layers.N.ffn.gate.bias` | [384] | f32 | noaux_tc correction bias | R |
| `layers.N.ffn.gate.bias_vl` | [384] | f32 | VL routing bias | drop (text) |
| `layers.N.ffn.shared_experts.w{1,2,3}.{weight,scale}` | w1/w3 [2304,5120], w2 [5120,2304] | FP8 32×32 | shared SwiGLU expert | R |
| `layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}` | w1/w3 [2304,5120], w2 [5120,2304] | **FP4 E2M1, E8M0/32** | routed SwiGLU expert (384×40) | **S** |
| `layers.N.hc_attn_fn`,`hc_ffn_fn` | [24, 20480] | f32 | `HyperConnection` mix proj | R |
| `layers.N.hc_attn_base`,`hc_ffn_base` | [24] | f32 | mix bias | R |
| `layers.N.hc_attn_scale`,`hc_ffn_scale` | [3] | f32 | pre/post/comb scale | R |
| `layers.N.engram.embed.{weight,scale}` | [~384 M, 256] | **FP8 E4M3, E8M0/32** | `ParallelEngramEmbedding` n-gram table | **D** (L1, L14) |
| `layers.N.engram.wkv.{weight,scale}` | [25600, 6144] | FP8 32×32 | engram key/value proj | R (L1, L14) |
| `layers.N.engram.q_weight`,`k_weight` | [4, 5120] | f32 | engram gate | R (L1, L14) |
| `mtp.M.*` (3 stages) | see §2 | FP8 + FP4 | DSpark MTP head | later phase |
| `vision.*`, `aligner.*`, `image_*` | — | bf16 | vision | **excluded** |

Notes that matter for the port:
- `experts.E` count = 384 × 40 layers = 15,360 each of w1/w2/w3 = the streamed bank. Every other
  weight above is resident or disk-backed.
- `engram.embed` is two tables of `engram_num_embeddings = [384,006,168 ; 384,016,682]` rows ×
  head_dim 256. Together 768,022,850 rows = 196 B params. This is the 196 B / ~189 GiB store; it is
  **not** in the wired budget and **not** in the expert bank. It is a third artifact.
- `wo_a` is FP8 on the Hub but the reference `convert.py` dequantizes it to bf16 at pack time
  (`convert.py:157-173`) because it runs as a block-diagonal `einsum` over o_groups, not a plain
  GEMM. The MTPLX port can keep it q8 (see §2, reuse `_wo_a_grouped`).
- The raw Hub expert tensors are stored as `int8` (two FP4 nibbles per byte). `convert.py:181` does
  `view(torch.float4_e2m1fn_x2)`. FP4 code→value table: `convert.py:13-14`.

---

## 2. Reuse audit vs `feat/deepseek-v4-flash:mtplx/models/deepseek_v4.py`

The V4 backend is the right skeleton: same MLA shape, o_lora/o_groups, hyper-connections + Sinkhorn,
`sqrtsoftplus` gate, ClampedSwiGLU, Compressor/Indexer. Grep of the 3,440-line file confirms what is
present and what is absent.

**Present in V4 (grep counts): carry over or adapt**

| V4 symbol (line) | V4.1 disposition | Change |
|---|---|---|
| `HyperConnection` (1496), `HeadHC` (1583), `hc_split_sinkhorn` (1345), `_sinkhorn_metal_kernel` (1206), `_install_sinkhorn_normaliser` (1312) | carry as-is | hc_mult 4, iters 20, eps 1e-6 identical; this is exactly Vontra `hc_mix`. "Single-Pass mHC" is a kernel-fusion name (Mega-mHC), not new math. |
| `MoEGate` (2855), `_score` (2875) | carry as-is | `sqrtsoftplus` + `noaux_tc` + norm_topk + route_scale 1.5 all match config. |
| `ClampedSwiGLU` (2814), `DeepseekV4MLP` (2778) | carry as-is | swiglu_limit 10.0 (up clamped both sides, gate clamped from above). Matches `Expert.forward` in `model.py:841`. |
| `DeepseekV4MoE` (2899) | adapt | expert_count 256→384, top_k stays 6, inter 2048→2304, hidden 4096→5120. Routed experts now come from the **streamed bank**, not resident `nn.ModuleList`. |
| `DeepseekV4Attention._wo_a_grouped` (2422), `_o_lora` (2513), `_o_lora_gather_qmm` (2466) | carry as-is | o_groups 8, o_lora_rank 1024 identical. |
| `_attend` (2586), `_attn_mask` (2535), `attn_sink` handling | adapt | attn_sink present (grep 13). V4.1 concatenates a per-head sink column into softmax (Vontra `sparse_attention`, runtime.py:87-90). Verify sink is per-head float, shape [64]. |
| `Compressor` (1626), `CompressorState` (1922), `Indexer` (1809) | adapt | Config field names change: V4 derives sources from `compress_ratios`; V4.1 adds explicit `kv_source_layer_ids` / `index_source_layer_ids`. index_n_heads 64→32. |
| `DeepseekV4Cache` (2037): `update_window` (2123), `update_compressed` (2164), `update_index_compressed` (2170), `trim`/rollback (2185/1996) | adapt | Add the CSA2 cross-layer sharing (below). Keep the trim/rollback seam. |
| RoPE/YaRN `_yarn_inv_freq` (1061), `_apply_interleaved_rope` (1152); `_topk_mask` (1120) | carry as-is | interleaved rope on last 64 dims; YaRN factor 16, orig 65536, compress_rope_theta 160000. |
| `DeepseekV4MTP` (2976) | replace | V4 ships one MTP block; V4.1 needs the 3-stage DSpark head (new, §6 phase 3). |

**Absent in V4 (grep count 0): new code**

| Missing | Needed for | New from |
|---|---|---|
| `engram` / `Engram` / `ngram` / `hash_ids` (0) | Engram conditional memory | port `engram.py` `NgramHashState` + `Engram`; Vontra `EngramHash` (runtime.py:242-280) + `engram()` (308-318); `ParallelEngramEmbedding` row gather (model.py:296-326) → disk-backed reader |
| `kv_source` / `index_source` (0) | CSA2 mode dispatch | new ModelArgs fields + per-layer mode from set membership |
| `select_candidate` / `reindex` (0) | hierarchical two-level top-K | port `select_candidate_blocks` (model.py:583) + Vontra candidate path (runtime.py:377-385) |
| `window_kv` (0) | SWA ring buffer + FP8 window quant | port `_window_kv` (model.py:700) |
| a cross-layer shared-attention state object (V4 `topk_idxs` grep only 2) | Reuse/Reindex modes | port `SharedAttentionRuntime` (model.py:1166): shared `compress_kv`, `index_k`, `topk_idxs`, `candidates` published by a source layer and read by later layers |
| `dspark` / `DSpark` / `markov` / `confidence` / `noise_token` / `main_proj` (0) | MTP draft head | port DSpark classes (model.py:1032-1157) + Vontra `dspark.py`; later phase |
| `bias_vl` (0) | VL gate bias | drop for text |

**What the Vontra `runtime.py` does that `deepseek_v4.py` lacks** (line citations, the load-bearing gaps):

| V4.1 mechanism | Vontra runtime.py | What is new vs V4 |
|---|---|---|
| CED / compressed-KV projection + shared cache | `attention()` 354-365 (compressor pool 356-364, latent norm 364), 387-391 (append + concat compressed) | V4 compresses per layer; V4.1 projects at `kv_source` layers only and shares `shared['compressed']` to later layers |
| CSA2 Full/Reindex/Reuse dispatch | 355 (`layer in kv_source_layer_ids` = Full), 366-369 (`index_source` own-K = Reindex when not kv_source), 383-385 (else reuse `shared['topk']` = Reuse) | explicit 3-mode static assignment; V4 has no mode concept |
| Hierarchical indexer (candidate blocks) | 377-382 (block-max top-K at `candidate_source_layer`), 383-384 (downstream mask to candidates) | V4 has flat index top-K only; no candidate pre-filter |
| Engram hash + gated lookup | `EngramHash` 242-280 (rolling XOR hash → row ids), `engram()` 308-318 (embed → wkv → sigmoid-gated add) | absent in V4 |
| sqrtsoftplus gate | `moe()` 331 `mx.sqrt(mx.logaddexp(logits, 0))` | present in V4 (parity check only) |
| o_groups grouped output | `grouped()` 223-233, `attention()` 393-394 | present in V4 |
| attention sink | `sparse_attention()` 87-90, `attention()` 392 | present in V4 |
| SWA sliding window (bounded replay seed) | `attention()` 351 ring `[-sliding_window:]` | V4 windows exist; V4.1 report adds host-side "bounded replay" reconstruction (later optimization; §3) |
| DSpark 3-stage draft + greedy verify | `dspark.py` `propose()` 68-96, `verify_greedy()` 99-115 | V4 has 1 MTP block, no markov/confidence heads |

---

## 3. Serving memory plan at the 100 GiB wired ceiling

Ceiling = 100 GiB wired (M5 Max). Hard limit; never raised (memory-knob rule). Byte models:
- q8 gs64 affine ≈ **1.0625 B/param** (8-bit packed + fp16 scale + fp16 bias per 64).
- bf16 = 2 B/param; f32 = 4 B/param.
- Q2 gs64 expert record = 2.5 bpw (below).

### 3a. Resident dense backbone (q8 gs64)

Computed from §1 shapes. Projections quantized to q8; norms/hc/gate.bias/compressor/indexer-bf16
kept at source precision; embed + head at q8.

| Component | Formula | GiB |
|---|---|---:|
| attn wq_b (×40) | q8(32768·1280) | 1.660 |
| attn wo_b (×40) | q8(8192·5120) | 1.660 |
| shared experts (×40) | q8(3·2304·5120) | 1.401 |
| attn wo_a (×40) | q8(4096·8192) | 1.328 |
| embed | q8(129280·5120) | 0.655 |
| head | q8(129280·5120) | 0.655 |
| engram wkv (×2) | q8(25600·6144) | 0.311 |
| attn wq_a (×40) | q8(1280·5120) | 0.259 |
| gate (×40) | bf16(384·5120)+f32(384) | 0.147 |
| hc vectors (×40) | f32(24·20480·2 + …) | 0.146 |
| attn wkv (×40) | q8(512·5120) | 0.104 |
| indexer wq_b (×8) | q8(4096·1280) | 0.042 |
| compressor wgate (×3) | f32(512·5120) | 0.029 |
| compressor wkv+norm (×4) | bf16(512·5120)+f32 | 0.020 |
| indexer wproj/wk + norms/sink | small | 0.004 |
| **Resident total (embed+head q8)** | | **8.42** |
| variant: embed+head bf16 | | 9.58 |

**Resident dense backbone ≈ 8.42 GiB.** The 552 B "backbone" figure is dominated by the 543.6 B
routed-expert params, which are streamed; the dense-only remainder is ~8.4 B params.

### 3b. KV cache

Native V4.1 KV is FP4 global + FP8 window. Values per position:
- Compressed KV record: 512·0.5 (FP4) + 512/16·1 (E4M3 scale) = **288 B**.
- Index key record: 128·0.5 (FP4) + 128/32·1 (E8M0) = **68 B**.
- Window KV record: 512·1 (FP8) + 512/32·1 = **528 B**.

| Quantity | Formula | Value |
|---|---|---:|
| Global cache growth / token (native FP4) | Σ (288+68)/ratio over kv_source [L2,8,14 r2; L20 r1] = 3·178 + 356 | **890 B/token** |
| Global cache growth / token (bf16 KV, phase-1) | Σ (1024+256)/ratio = 3·640 + 1280 | 3200 B/token |
| SWA window (fixed, native FP8) | 40·128·528 | 2.58 MiB |
| SWA window (fixed, bf16) | 40·128·(512·2) | 5.00 MiB |

Global cache at 1 M tokens: 890 MB native, 3.2 GB bf16. Both fit. Reindex/Reuse layers add **no**
KV storage (they read layer-20 / their source's cache). Phase 1 keeps bf16 global KV (David's KV
preference, avoids the FP4 KV kernel); native FP4 is a later size win.

### 3c. Expert slot budget (Q2 gs64)

| Quantity | Formula | Value |
|---|---|---:|
| expert_source_parameters | 3·5120·2304 | 35,389,440 |
| packed (2-bit) | 35,389,440·2/8 | 8,847,360 B |
| scales+biases (gs64, fp16 ×2) | (35,389,440/64)·2·2 | 2,211,840 B |
| **expert_record_bytes** | packed + scale/bias | **11,059,200 B = 10.55 MiB (2.500 bpw)** |
| routed bank (40·384) | 40·384·record | 158.20 GiB |
| cold experts / token | 40·6·record | 2.472 GiB/token |
| transient scratch | 6·record | 63.3 MiB |

### 3d. Budget assembly (phase-1, bf16 KV, 4 K context)

| Line | Bytes | GiB |
|---|---:|---:|
| Resident dense backbone (q8) | — | 8.42 |
| KV: 4096·3200 + window 5 MiB | — | 0.017 |
| Runtime reserve (per MTPLX profile convention) | — | 7.00 |
| Subtotal fixed | | ~15.4 |
| **Free for expert cache / islands** | 100 − 15.4 | **≈ 84.6 GiB** |
| Streamed slots that buys (÷10.55 MiB) | | ~8,213 total ≈ 205/384 per layer |

So ~53% of routed experts stay resident-cached at any moment; the rest page from the bank. Full-
island (all 40 layers pinned) needs 158 GiB and does **not** fit — streaming is mandatory. Engram
(§4) is disk-backed and adds only I/O reserve, not wired bytes.

`plan_expert_memory` (`expert_streaming_models.py:674`) resolves the split; `kv_bytes = context ×
kv_bytes_per_token` (line 832). Choose an envelope profile like the Hy3/GLM rows in the docs.

---

## 4. Engram strategy

The 196 B engram table cannot be resident (189 GiB > 100 GiB ceiling on its own). It must be
disk-backed and gathered per token. 48 rows/token total: `(engram_max_ngram_size−1)·engram_n_heads
= 3·8 = 24` cols/layer × 2 layers (L1, L14). Each row is head_dim 256.

| Option | Row bytes | Table size | Read / token | Quality risk |
|---|---:|---:|---:|---|
| (a) keep FP8 E4M3 on SSD, gather via preadv | 256 + 8 (scale) = 264 | **188.8 GiB** | 48·264 = **12.4 KiB** | none beyond source (lossless vs Hub) |
| (b) requant 4-bit affine gs64 | 128 + 8–16 = 136–144 | 97.3–103.0 GiB | 6.4–6.8 KiB | second lossy step from an FP8 source |
| (c) requant 2-bit affine gs64 | 64 + 8–16 = 72–80 | 51.5–57.2 GiB | 3.4–3.8 KiB | aggressive; two lossy steps; engram gate is quality-sensitive |

**Recommendation: (a) keep FP8 rows on SSD.** Reasoning:
- The per-token engram read is ~12 KiB. The per-token cold-expert read is ~2.47 GiB (§3c). Engram
  I/O is ~5 orders of magnitude smaller — requantizing it buys no measurable bandwidth.
- Engram writes into the residual through a `sigmoid` match-gate (`model.py:350-368`, Vontra
  `engram()` 308-318). A conditional memory whose contribution is already damped is exactly where a
  second lossy quant step is least safe and least rewarding.
- FP8 is the Hub's native format, so (a) is bit-lossless relative to source.

Implementation cost of (a): the existing `expert_io` preadv path reads **affine** records; FP8
E4M3 + E8M0 rows are not affine. Add a thin FP8-row dequant shim (dequantize 256 E4M3 values with 8
E8M0 group scales → bf16) plus a bounded LRU row cache (Vontra bounds at 16,384 rows,
runtime.py:198-213). Reuse the preadv gather mechanics; do not reuse the affine record decoder.

Fallback: if SSD **capacity** (189 GiB) or page-cache residency becomes the binding constraint, use
(b) 4-bit gs64 (≈100 GiB, one lossy step, HumanEval-gated per §7). Do not use (c) without a
task-eval showing no regression.

---

## 5. Draft `ExpertStreamingModelSpec` — key `deepseek-v41-flash-expert-q2`

For the concurrent worker to place in `mtplx/expert_streaming_models.py`. Every field filled;
unknowns flagged `TODO`. Formulas match §3.

```python
DEEPSEEK_V41_EXPERT_Q2 = ExpertStreamingModelSpec(
    key="deepseek-v41-flash-expert-q2",
    display_name="DeepSeek-V4.1-Flash expert-only affine Q2 (gs64, q8 residents)",
    source_model="deepseek-ai/DeepSeek-V4.1-Flash",
    source_revision="TODO-pin-hub-commit",           # pin at download time
    quant_model="OpensourceWTF/DeepSeek-V4.1-Flash-Q2-MTPLX-streaming",  # TODO publish
    quant_revision="TODO-pin-after-upload",
    # total = resident dense (q8, §3a) + routed bank (§3c). Header-inventory sum after
    # conversion is authoritative; this is the derived estimate.
    total_tensor_bytes=178_912_399_808,             # 9_043_087_808 + 169_869_312_000  (≈166.6 GiB)
    total_layers=40,
    routed_layer_start=0,
    routed_layer_count=40,                           # every text layer is MoE (no dense FFN layers)
    expert_count=384,
    top_k=6,
    hidden_size=5120,
    expert_hidden_size=2304,
    quant_bits=2,
    quant_group_size=64,
    quant_parameter_bytes=2,                         # fp16 scale + fp16 bias
    router_storage="source bfloat16 with fp32 correction bias (bias_vl dropped, text-only)",
    router_matmul_dtype="float32",
    # router_bytes = routed_layers · experts · (hidden·2 + 4)  [gate.weight bf16 + gate.bias fp32]
    router_bytes=157_347_840,                        # 40·384·(5120·2 + 4)
    # kv_bytes_per_token: phase-1 bf16 global KV = Σ (5120·... ) → 3200; native FP4 = 890.
    kv_bytes_per_token=3_200,                         # bf16 global; SWA window is a fixed 5 MiB, not per-token
    mtp_layer_index=40,                               # first DSpark stage; stages 40,41,42
    mtp_included=False,                               # phase 1 = AR only
    # CSA2 index-owning layers (Full + Reindex). Confirm exact runtime consumption of this
    # field before relying on it for KV surcharge accounting.
    full_indexer_layers=(2, 8, 14, 20, 24, 28, 32, 36),  # TODO confirm semantics
    island_pin_order=(),                              # unmeasured; count-based island selection then needs an explicit layer list
    expert_codec="affine",
)
```

Derived checks (from the dataclass properties):
- `expert_record_bytes` = 8,847,360 + 2,211,840 = **11,059,200 B**.
- `routed_expert_bytes` = 40·384·11,059,200 = 169,869,312,000 B.
- `resident_bytes` = total − routed = 9,043,087,808 B (§3a; positive → passes `__post_init__`).
- `router_bytes (157 M) < resident_bytes (9.0 G)` → passes.
- `hidden_size % 64 == 0`, `expert_hidden_size % 64 == 0`, `5120·2304·2 % 8 == 0` → pass.

The **engram table is not in `total_tensor_bytes`** and not in the expert bank. It is a third
artifact (§4). The spec has no field for it; track it in the manifest `resident_tensors`/records or
a sidecar, and size it in `plan_expert_memory` as `io_staging_bytes`/reserve, not wired bytes.

### Manifest `model_key` contract (`expert_manifest.py`, `docs/advanced/ssd-streamed-moe.md`)

- `format` must be `"mtplx-expert-manifest-v1"` (`expert_manifest.py:40`).
- `manifest.model_key` **must equal** `spec.key` = `"deepseek-v41-flash-expert-q2"`; a mismatch
  fails before construction (`expert_manifest.py:2047`, `:2175`; docs "manifest model-key mismatch
  stops startup").
- For `expert_codec="affine"`, each record's logical bytes must equal `spec.expert_record_bytes`
  (11,059,200) or admission fails (`expert_manifest.py:1741`).
- The published repo ships resident shards + tokenizer/config + `expert-manifest.json` + the bank
  file named by the manifest (Hy3 uses `experts.bin`). `--download` fetches all; MTPLX writes a
  revision- and digest-bound admission receipt (docs "Install and serve").
- Record layout is component-banks (w1/w2/w3 leaves) for affine, so dense islands are possible
  later (docs "Islands versus paged streaming").

---

## 6. Port work breakdown

Ordered. Each task lists a file allowlist and a **done-when** that is a runnable check. MTP/DSpark is
phase 3, not phase 1. Phase 1 = text-only AR decode through the streaming path. Estimates are
single-worker calendar days on this box; GPU-touching gates queue behind the flock.

Concurrency note: `scripts/convert_deepseek_v41_expert_q2.py` and the spec entry are owned by the
other worker; tasks below consume their outputs, they do not write them.

| # | Task | File allowlist | Done-when (runnable) | Est. |
|---|---|---|---|---:|
| P1.0 | ModelArgs + config load for `deepseek_v41_text`; wire CSA2 layer-mode map from `kv_source`/`index_source`/`compress_ratios` | `mtplx/models/deepseek_v41.py` (new) | `pytest tests/models/test_deepseek_v41_config.py::test_layer_modes` — asserts the §0 mode table (Full/Reindex/Reuse/SWA per layer) | 1 d |
| P1.1 | Dense forward: embed, HyperConnection+Sinkhorn, MoEGate (sqrtsoftplus/noaux_tc), ClampedSwiGLU, shared expert, RMSNorm, head. Reuse V4 classes. | `mtplx/models/deepseek_v41.py` | `pytest ...::test_dense_block_parity` — one block vs Vontra `runtime.py` at fixed input, max abs err < 1e-2 (bf16) | 2 d |
| P1.2 | Attention: MLA q/kv, o_lora grouped wo_a/wo_b, attn_sink, SWA ring window + FP8/bf16 window quant | `mtplx/models/deepseek_v41.py` | `pytest ...::test_attention_swa_only` — layers 0-1 (SWA-only) match reference logits argmax on a 32-token prompt | 2 d |
| P1.3 | CSA2: `Compressor` (Full), `Indexer` (Reindex), `select_candidate_blocks`, `SharedAttentionRuntime` cross-layer reuse, compressed + index KV cache | `mtplx/models/deepseek_v41.py` | `pytest ...::test_csa2_modes` — full 40-layer forward, top-K indices at a Reuse layer equal the source layer's; argmax matches reference on a ≥256-token prompt (crosses ratio-2→1 and candidate prefilter) | 4 d |
| P1.4 | Engram: `NgramHashState` + tokenizer compressed map + `Engram` gated add; FP8-row disk reader + bounded LRU (option 4a) | `mtplx/models/deepseek_v41.py`, `mtplx/engram_io.py` (new) | `pytest ...::test_engram_hash_map` (compressed vocab == 99092, `engram.py:146`) + `::test_engram_row_dequant` (FP8 row → bf16 within 2^-6 rel) | 3 d |
| P1.5 | Streaming integration: routed experts served from the Q2 bank via existing `expert_runtime`/`expert_io`; install AR route | `mtplx/models/deepseek_v41.py`, streaming glue only (no edits to `expert_streaming_models.py` / `scripts/`) | `mtplx serve --model <local Q2 repo> --download` reaches `/health` with `generation_mode: "ar"`, correct model key, and admission receipt | 3 d |
| P1.6 | **Parity gate** vs DeepSeek reference | `tests/models/test_deepseek_v41_parity.py` (new) | `pytest ...::test_reference_argmax` — greedy next-token argmax matches `inference/generate.py` for a 5-token continuation on a fixed prompt (bounded ctx) | 1 d |
| P1.7 | **streamed == resident argmax gate** (as in the V4 port) | `tests/models/test_deepseek_v41_stream_equiv.py` (new) | `pytest ...::test_stream_equals_island` — same prompt, all-islands vs streamed cache: identical argmax sequence for 32 decode steps | 1 d |
| P1.8 | HumanEval quality cell (§7) at David's sampler | `tests/` harness only (GPU, flock) | one HumanEval(164) pass@1 cell recorded; bank verdict per memory `task-evals-decide-bank-verdicts` | 1 d (queued) |
| P2.0 | Native FP4 global KV + index-K quant kernels (size win) | `mtplx/models/deepseek_v41.py` | `::test_fp4_kv_equiv` — argmax unchanged vs bf16 KV over 32 steps; global cache 890 B/tok | 2 d |
| P2.1 | SWA bounded replay (host-memory split of long-lived global KV vs encoder SWA) | `mtplx/models/deepseek_v41.py` | `::test_swa_replay_reconstruct` — replayed window matches full window argmax | 3 d |
| P3.0 | DSpark 3-stage MTP head + greedy verify (drafts, main_proj, markov + confidence heads) | `mtplx/models/deepseek_v41.py`, `mtplx/dspark_v41.py` (new) | `::test_dspark_greedy_verify` — verified tokens equal pure-AR argmax (lossless spec) over 64 tokens; then decode tok/s A/B | 5 d |

Phase-1 critical path (P1.0–P1.8) ≈ **20 working days** single-worker, plus flock-queued GPU gates.
P1.3 (CSA2 cross-layer reuse) and P1.4 (engram) are the risk tasks. Phases 2–3 add ~15 days.

---

## 7. Quality caveat

The routed experts are **natively FP4 E2M1** on the Hub. Q2 is therefore a **second lossy step from
a 4-bit source**, not from bf16. This is unlike Hy3/GLM Q2 lanes, which quantize from higher-
precision sources. FP4→Q2 has a narrower error budget: the source already spent most of its dynamic
range, and the DeepSeekMoE top-6 routing gives each token only 6 of 384 experts, so a per-expert
error has less averaging headroom than a dense FFN.

David's rules require a task-eval verdict before this bank is trusted (memory:
`task-evals-decide-bank-verdicts`, `humaneval-one-seed`, `acceptance-rate-is-a-primary-bottleneck`):
- **One HumanEval(164) pass@1 cell** at David's sampler is the sanity gate (one seed, no greedy
  tripwire, no MBPP unless asked). Record strict pass@1; report the output cap so truncation is not
  scored as failure (memory: `eval-truncation-is-not-failure`).
- Compare the Q2 streamed bank against the **standard path** (the FP4 reference, or a q8-expert
  control if one is built) — name any regression as a bug, do not rationalize it as an expected
  tradeoff (memory: `dont-rationalize-broken-as-normal`).
- If Q2-from-FP4 craters, the honest fallbacks are: q4 routed experts (still one lossy step from
  FP4 but far smaller error), or keeping the natively-FP4 records and only paging them (no requant
  at all — the bank is then ~2× the Q2 size but bit-exact to source). The 84.6 GiB expert-cache
  headroom (§3d) does not fit an FP4-native bank fully resident, but it pages the same way.

Engram (§4 option a) is bit-lossless to source and does not enter this quality budget; option (b)/(c)
would, and would need their own HumanEval cell.
