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
  - *Streamed (implemented):* `bind_streamed_switches` (`mtplx/models/expert_mlx.py`)
    replaces `switch_mlp` with `HotExpertSwitchGLU` / `MappedExpertSwitchGLU` /
    `DenseIslandSwitchGLU`, gathering the Q2 records from `experts.bin`, and these
    now apply the same clamp — see "Streaming clamp".

## Streaming clamp (implemented + measured)

The follow-up: the reference clamps *every* routed expert, so the streamed path
must clamp too — not just the resident fallback. It now does.

### Implementation

`expert_mlx.py` streamed every expert through
`mlx_lm.models.activations.swiglu(gate, up)` at six sites with **no activation
hook** — the missing clamp. The fix threads the limit from the spec:

- `ExpertStreamingModelSpec.swiglu_limit: float | None = None` (new field), set to
  `10.0` on `DEEPSEEK_V41_FLASH_EXPERT_Q2`. `None` is the default, so every hy3 /
  glm / deepseek_v4 spec is byte-for-byte unchanged (drift-guarded: the pinned
  `asdict` digests in `test_expert_streaming_models.py` were updated for the
  intentional new field, values still `None`).
- One helper, `_clamped_swiglu(gate, up, swiglu_limit)`, applies the reference's
  asymmetric clamp (`inference/model.py` L846-847: up two-sided `[-L,+L]`, gate
  upper-only `+L`) before `swiglu`; `None`/`≤0` is the plain `swiglu` verbatim.
- Every streamed execution helper (`_run_q4_expert`, `_gather_component_bank`,
  `_shadow_gather_component_bank`, `_gather_component_bank_mixed`, `_run_shadow_bank`,
  `_run_mapped_q4`) takes `swiglu_limit` and calls `_clamped_swiglu`. Each streamed
  switch (`HotExpertSwitchGLU`, `MappedExpertSwitchGLU`, `DenseIslandSwitchGLU`)
  reads `self.swiglu_limit = getattr(runtime.spec, "swiglu_limit", None)` and passes
  it down. `runtime.spec` is the spec, so no opener change is needed.
- The one path that bypasses the six `swiglu` sites is the fused Metal K3 wave
  (`DenseIslandSwitchGLU.wave_call` → `hy3_q2_m4_expert_wave`, which computes SwiGLU
  inside the kernel). It is hy3-only (hy3 leaves `swiglu_limit=None`) and dsv41 never
  uses islands, but for the contract `wave_call` now returns `None` when a limit is
  set, falling back to the eager clamped dispatch. Clamping inside that kernel is a
  future Metal change, flagged in `wave_call`.

### Is the clamp load-bearing? Yes — my earlier "inert" claim was WRONG.

My first pass sampled only **layer 0 on 4 random tokens** (max pre-activation < 10)
and wrongly generalised to "inert". The 40-layer component-banks probe on the real
31-token probe A (the add/sub/mul prompt whose tail the old forward turned to junk)
shows the opposite. Measured on the **un-clamped (old) trajectory**
(`tests/models/test_deepseek_v41_clamp_probe.py`, receipt
`docs/deepseek-v41/receipts/cpu_clamp_probe.json`):

- **Global max |gate| = 72.3, max |up| = 78.0** — 7-8× the ±10 limit. Layers 0-14
  stay under 10; from **layer 15 on** many layers blow past it (L25 gate 72.3, L21
  up 78.0), the clamp cutting hundreds of activations per deep layer (L36:
  gate>10 = 1164, |up|>10 = 614).
- **7 of the 8 flagged junk positions** (11, 12, 19, 20, 21, 24, 26 — all but 6)
  have at least one expert pre-activation over ±10. (BOS position 0 and several
  non-junk positions also exceed, so "> ±10" is broad, not junk-specific.)

Causal test — teacher-forced next-token argmax on the same load, clamp OFF vs ON:

- **OFF 16/30 → ON 17/30.** OFF reproduces the receipt's baseline 16/30 exactly,
  confirming OFF == the old forward. The clamp changes predictions at positions
  {5, 7}, **fixes position 5, breaks none** (net +1).
- **But it fixes none of the 8 flagged junk positions on its own.** So the missing
  clamp is a real, load-bearing defect (net-positive and faithful to the reference),
  **but not the sole cause of the tail junk** — another defect remains (consistent
  with W8's decode-path investigation). The clamp is necessary for faithfulness, not
  sufficient for the junk.

Cost: 2 forwards + per-expert dequant on the real bank, CPU, ~650 s, peak RSS ~16 GB.

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
| `test_layer0_activations_stay_under_limit_but_deep_layers_do_not` | layer-0 pre-acts stay < ±10 (a LOCAL fact); points to the clamp probe for the deep-layer verdict | PASS (`max|pre|<10`) |
| `test_layer0_moe_matches_w9_torch_golden` | optional golden against W9 torch receipts | SKIP (goldens absent) |

`5 passed, 1 skipped` under `nice -n 19`, CPU. Peak RSS well under the 40 GB
budget (reads only ~6 Q2 records ≈ 66 MB from the 158 GiB bank, plus the layer-0
residents).

### Streaming-clamp tests — `tests/models/test_deepseek_v41_streaming_clamp.py` (CPU)

| test | proves | result |
|---|---|---|
| `test_spec_swiglu_limit_values` | dsv41 spec `swiglu_limit==10.0`; every other spec `None` | PASS |
| `test_none_limit_keeps_plain_swiglu_byte_identical` | `_clamped_swiglu(·,·,None)` and `(·,·,0)` are byte-identical to plain `swiglu`; a binding limit changes it | PASS |
| `test_hot_switch_reads_spec_swiglu_limit` | `HotExpertSwitchGLU.swiglu_limit` is 10.0 for dsv41, `None` for hy3 (spec→runtime→switch wiring) | PASS |
| `test_component_bank_helper_applies_reference_clamp` | `_gather_component_bank` (the dsv41 path) == resident `ClampedSwiGLU` **bit-exact** for limit None/3/10 on a real record; binding limit differs from None | PASS |
| `test_mapped_helper_applies_reference_clamp` | same for `_run_mapped_q4` (metal-mmap path) | PASS |

`5 passed`. Existing `tests/test_expert_streaming_models.py` (26) still passes
after the drift-guard digest update.

### Clamp probe — `tests/models/test_deepseek_v41_clamp_probe.py` (CPU, gated)

Gated behind `DSV41_RUN_CLAMP_PROBE=1` (heavy: two full 40-layer streamed forwards
+ per-expert dequant, ~650 s, peak RSS ~16 GB). Ran once; verdict above, receipt at
`docs/deepseek-v41/receipts/cpu_clamp_probe.json`.

Dequant note: the Q2 routed records are 2-bit affine, so they dequantise via
`mx.dequantize` (the exact inverse of the converter's `quantize_affine`, which is
`mx.quantize`). The converter's numpy `dequant_affine_record` is 8-bit-only, so it
does not apply to the 2-bit routed records; `mx.dequantize` is the converter's
format, used for both the q8 shared experts and the q2 routed records.

## Not done / out of scope

- No edit to `mtplx/models/deepseek_v41.py` (W10 owns it) — integration contract in
  `PORT_CONTRACT.md` under "W11".
- The streamed clamp IS now implemented in `expert_mlx.py` +
  `expert_streaming_models.py` (coordinator-directed follow-up; the earlier
  "documented, not applied" stance is superseded). The one remaining unclamped path
  is the fused Metal K3 wave (`hy3_q2_m4_expert_wave`), which is hy3-only and which
  `DenseIslandSwitchGLU.wave_call` now refuses when a limit is set — clamping inside
  that kernel is a future Metal change, out of W11's scope.
- The missing clamp is not the sole cause of probe A's tail junk (clamp fixes only
  position 5, 16→17); the residual junk is a separate defect (W8 decode-path line).
- W9 golden test present but dormant until `torchref_*.json` land.
