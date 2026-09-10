# W11 — DeepSeek-V4.1-Flash MoE submodule

Owner: W11. Files: `mtplx/models/deepseek_v41_moe.py`,
`tests/models/test_deepseek_v41_moe.py`, this report, `PORT_CONTRACT.md` (W11
section). Branch: `feat/deepseek-v41-w11` off `feat/deepseek-v41-streaming`
(integration @ 4c6a8b4c). CPU-only, mlx 0.32.2.

## What shipped

A faithful, line-by-line MLX transliteration of the DeepSeek reference
`inference/model.py` MoE stack, modified so routed execution runs through the
MTPLX expert-streaming seam. Reference class names, constructor order
`(layer_id, args)`, `forward` signature and return shape all match, so W10's
`Block` calls it exactly as the reference `Block` does.

| class | reference (model.py) | notes |
|---|---|---|
| `ClampedSwiGLU(SwiGLU)` | `Expert.forward` clamp, L845–848 | `SwitchGLU` activation seam; asymmetric clamp |
| `Gate` | `Gate`, L792–828 | sqrtsoftplus / noaux_tc / norm_topk / route_scale |
| `Expert` | `Expert`, L830–851 | backs the shared expert; clamped SwiGLU |
| `MoE` | `MoE`, L854–904 | routed `experts` → streamed `switch_mlp` seam |

Every method carries its reference line range inline.

### Faithfulness decisions (what model.py actually does)

- **No group routing.** The released text config has no `n_group` / `topk_group`
  (checked `config.json`), and the reference `Gate` (L823) does a plain
  `(scores+bias).topk(topk)` over all 384 experts. There is no group-limited
  branch to transliterate. (`n_groups` in model.py L633 is the *attention*
  `o_groups`, unrelated.)
- **Bias steers selection, scores give weights.** `indices` come from
  `scores + e_score_correction_bias` (noaux_tc, L823); `weights` come from the
  *unbiased* `scores` gathered at those indices (L824). `norm_topk_prob` divides
  by `sum + 1e-20` (the training constant, L826, not `norm_eps`), then `*=1.5`
  (`route_scale`, L827).
- **Asymmetric SwiGLU clamp** (L846–847): the *up* branch (`w3`) is clipped
  two-sided `[-10, +10]`; the *gate* branch (`w1`) only has its upper tail cut at
  `+10` and keeps its full negative range. Both cuts are on the pre-activation
  projections, before `silu`. `gate_temp` defaults to 1.0 (absent from config).
- **f32 accumulation.** `MoE.forward` accumulates the weighted routed sum + shared
  expert in float32 and casts back to `x.dtype` (reference L893 `zeros(..., f32)`
  and L904 `type_as(x)`).

## Seam contract (the load-bearing part of "modify it for our MLX kernel")

`self.switch_mlp` is the routed-expert seam (hy3 convention).

- **What the switch receives:** `switch_mlp(xf, indices)` with
  `xf = [n_tokens, hidden]` and `indices = [n_tokens, top_k]` **int32** — the
  reference top-6 expert ids in `(scores+bias)`-descending order.
- **What it returns:** `[n_tokens, top_k, hidden]`, the **unweighted** per-expert
  outputs. `MoE.forward` applies the reference-normalised routing weights
  (`(routed * weights[...,None]).sum(-2)`, in f32) and adds the shared expert
  *outside* the seam — matching how the reference multiplies each expert output by
  its weight inside the dispatch loop (L900) and adds the shared expert at L903.
- **What activation it applies:** the reference's **clamped** SwiGLU
  (`swiglu_limit=10.0`).
  - *Resident / unit-test default:* `mlx_lm.models.switch_layers.SwitchGLU` built
    with `activation=ClampedSwiGLU(args.swiglu_limit)`. `SwitchGLU.__call__`
    invokes `self.activation(x_up, x_gate)` between the up/gate `SwitchLinear`s and
    `down_proj`, so `ClampedSwiGLU` applies the clamp exactly where the reference
    `Expert` does. (Arg order is `(up, gate)` — opposite of the names — handled in
    `ClampedSwiGLU.__call__`.)
  - *Streamed:* `bind_streamed_switches` (`mtplx/models/expert_mlx.py` ~L2550)
    replaces the whole `switch_mlp` module with `HotExpertSwitchGLU` /
    `MappedExpertSwitchGLU` / `DenseIslandSwitchGLU`, gathering the Q2 records from
    `experts.bin`.

### How the clamp is injected — and the current gap

I read `expert_mlx.py` end to end. **The streamed switches expose no activation
hook.** Every streamed execution path calls
`mlx_lm.models.activations.swiglu(gate, up)` directly (L1369, 1440, 1497, 1586,
1616, 1639), the runtime `spec` carries no `swiglu_limit`/clamp field, and
`bind_streamed_switches` reads *nothing* off the switch it replaces (it does not
propagate the resident `ClampedSwiGLU`). This is unlike the resident path used by
`deepseek_v4`, which does carry the clamp through `SwitchGLU(activation=
ClampedSwiGLU(...))` — but `deepseek_v4` streams through the *same* clamp-less
`swiglu`, so no existing model injects a clamp into the streamed path.

**So on the streamed path the `swiglu_limit` clamp is currently NOT applied.**

I did **not** edit `expert_mlx.py`. Rationale: it is a large shared surface
(hy3 / glm / deepseek_v4 all stream through it); a correct fix touches the spec, the
opener, and six `swiglu(...)` call sites — not the "minimal, tested change" the
allowlist permits. And it is unnecessary for this artifact:

- **The gap is numerically inert here.** At `swiglu_limit=10.0` the gate/up
  pre-activations on the real layer-0 records never reach ±10, so plain SwiGLU ==
  clamped SwiGLU bit-for-bit. Proven by
  `test_streamed_clamp_is_inert_on_real_records` (asserts `max|pre| < 10` and
  `max|plain − clamped| == 0.0` across the 4×6 selected records).

Recommended follow-up (owner: streaming runtime, not W11): add an optional
`swiglu_limit` to the expert streaming spec and clamp between the up/gate qmm and
`swiglu` in the streamed execution helpers, defaulting to off so hy3/glm/v4 are
unchanged. Until then the streamed dsv41 path is exact *only because the clamp does
not bind*; a future artifact whose activations exceed the limit would need it.

## Resident tensor names (loader parity)

Matches `mtplx/models/deepseek_v41.py::_sanitize_name`, so the strict 1,616-key
text load is preserved:

- `layers.N.ffn.gate.weight` → `...mlp.gate.weight` (bf16, not quantised)
- `layers.N.ffn.gate.bias` → `...mlp.gate.e_score_correction_bias` (f32; `bias_vl`
  dropped)
- `layers.N.ffn.shared_experts.w{1,2,3}.{weight,scales,biases}` →
  `...mlp.shared_experts.w{1,2,3}...` (resident q8 gs64 affine)
- `layers.N.ffn.experts.*` — routed, **streamed**, never resident. `switch_mlp` is
  skipped by `_is_resident_quant_module` and absent from the text resident dict.

Verified against the real artifact: layer-0 gate `weight [384,5120] BF16`, `bias
[384] F32`; shared `w1/w3.weight [2304,1280] U32`, `w2.weight [5120,576] U32`,
scales/biases BF16; Q2 record = 9 components (`{gate,up,down}_proj.{weight U32,
scales BF16, biases BF16}`), `logical_bytes 11,059,200`.

## Tests — `tests/models/test_deepseek_v41_moe.py` (CPU)

`mx.set_default_device(mx.cpu)` at import. Artifact-backed tests skip when the
artifact is absent; the artifact was present, so all ran.

| test | proves | result |
|---|---|---|
| `test_gate_parity_topk_ids_and_weights` | numpy transcription of reference `Gate` on random inputs + **real layer-0 bias** → identical top-6 id set and per-id weights, `< 1e-5` (f32) | PASS |
| `test_clamped_swiglu_activation_parity_including_beyond_limit` | `ClampedSwiGLU` vs numpy reference clamp on inputs beyond ±10; `limit≤0` == plain SwiGLU | PASS (`max|Δ| < 1e-4`) |
| `test_expert_forward_parity_including_beyond_limit` | `Expert.forward` vs numpy reference `Expert.forward`, clamp firing | PASS (`max|Δ| < 1e-3`) |
| `test_moe_seam_forward_matches_numpy_reference` | real gate + shared experts + record-backed test-double switch (6 Q2 records from `experts.bin`, clamped SwiGLU) vs from-scratch numpy MoE over the dequantised layer-0 tensors | PASS, **cos 0.999987 ≥ 0.999** |
| `test_streamed_clamp_is_inert_on_real_records` | documents the streamed clamp gap is inert at limit=10 on real records | PASS (`max|pre|<10`, `Δ==0`) |
| `test_layer0_moe_matches_w9_torch_golden` | optional golden against W9 torch receipts | SKIP (goldens absent) |

`5 passed, 1 skipped` under `nice -n 19`, CPU. Peak RSS well under the 40 GB
budget (reads only ~6 Q2 records ≈ 66 MB from the 158 GiB bank, plus the layer-0
residents).

Dequant note: the Q2 routed records are 2-bit affine, so they dequantise via
`mx.dequantize` (the exact inverse of the converter's `quantize_affine`, which is
`mx.quantize`). The converter's numpy `dequant_affine_record` is 8-bit-only, so it
does not apply to the 2-bit routed records; `mx.dequantize` is the converter's
format, used for both the q8 shared experts and the q2 routed records.

## Not done / out of scope

- No edit to `mtplx/models/deepseek_v41.py` (W10 owns it) — integration contract in
  `PORT_CONTRACT.md` under "W11".
- No edit to `expert_mlx.py` — streamed clamp injection documented, not applied
  (see the gap section).
- W9 golden test present but dormant until `torchref_*.json` land.
