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
