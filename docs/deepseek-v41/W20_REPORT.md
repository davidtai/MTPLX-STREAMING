# W20 — Token-chunked prefill for DeepSeek-V4.1-Flash (streaming)

Branch: `feat/deepseek-v41-w20` off `feat/deepseek-v41-streaming` (integration @ 5d6dd8ad).
Status: IN PROGRESS (skeleton committed first per loss-recovery rule).

## Task
The 16,384-token standard-shape bench dies in prefill with
`[metal::malloc] Attempting to allocate 103,089,701,120 bytes` inside the MoE
switch (`deepseek_v41.py:670 self.mlp(x)` -> `deepseek_v41_moe.py:243
self.switch_mlp(xf, indices)` -> `expert_mlx.py` HotExpertSwitchGLU). The 1,024-token
cell works. Goal: name the allocation with arithmetic; bound prefill memory by
token-chunking the DeepSeek-V4.1 forward (attention + MoE + engram) so chunks feed
the SAME cache/shared-attention state incrementally; prove exactness on CPU vs
one-shot; bound the 16K MoE transient by test; recommend bench flags.

## Findings (WIP)
- `DeepseekV41Backbone.__call__` runs the WHOLE prompt in one forward:
  `positions = mx.arange(cache.offset, cache.offset + s)`, then loops all 40 layers
  over `[b, s, ...]`. No token chunking anywhere in this model.
- `MoE.__call__`: `xf = x.reshape(-1, dim)` = `[16385, 5120]`, `indices = [16385, 6]`;
  `switch_mlp(xf, indices)` must return `[16385, 6, 5120]`.
- Streamed switch = `HotExpertSwitchGLU` (slot_layout=component-banks). `_run`
  flattens all `16385*6 = 98,310` routed positions and groups them into route waves.
  Per-wave `_gather_component_bank` is per-row (`rows = x.shape[0]`, slot_indices
  `[rows,1]`), gate/up `[rows,1,1,2304]`, down `[rows,1,1,5120]` — bounded.
- ALLOCATION ARITHMETIC: TBD (pinning exact tensor to 103,089,701,120 bytes).
- Attention `_sparse_attend`: `scores = einsum("bshd,btd->bsht")` = `[b, s, H, T]`;
  at 16K prefill on a dense/window layer this is large — TBD arithmetic.

## Plan
1. Name the 103 GB tensor by arithmetic + a synthetic switch recording its max ask.
2. Token-chunk the forward; chunks feed the same cache incrementally (W13 cache
   advance/append; coordinate with W22's per-layer cache protocol).
3. CPU exactness: chunked == one-shot logits (<=1e-5) + equal cache state, across
   chunk sizes straddling the 128 window and ratio-2 compressor groups.
4. Memory-bound test: largest MoE transient at 16,384 with chosen chunk < 8 GB.
5. Attention/indexer 16K score check; chunk queries if a 16K x 16K tensor forms.
6. Recommended bench flags for the 16K cell.

## Peak RSS
TBD (tiny synthetic configs only; CPU; no real-artifact loads).
