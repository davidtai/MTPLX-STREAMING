# W104 — DSpark DRAFT-block resident MoE trace + sync census

Worker `w104/draft-resident-moe`. Opus 4.8, CPU-only (`mx.set_default_device(mx.cpu)`,
tiny seeded real-structure DSpark head, no artifact, no GPU, every process < 1.5 GB
RSS). Subject: the DSpark **draft** block's resident 128-expert top-3 MoE only
(`DSparkBlock.mlp(moe_input)`). NOT touched: the verify path, the expert-streaming
runtime, the backbone MoE.

## Verdict (one paragraph)

The draft MoE **already runs a barrier-free resident `mx.gather_qmm(mode="mxfp4")`**
and issues **zero host syncs per draft cycle**. `DSparkBlock.mlp` is a plain
`mtplx.models.deepseek_v41_moe.MoE` (`mtplx/models/deepseek_v41_dspark.py:623`) whose
`switch_mlp` is an mlx-lm `SwitchGLU` whose three `SwitchLinear` leaves are repacked
in place to `QuantizedSwitchLinear(mode="mxfp4", group_size=32, bits=4)` by
`Model._build_mtp_head` (`mtplx/models/deepseek_v41.py:4030-4035`). It is **not** the
streamed `HotExpertSwitchGLU`: `bind_streamed_switches` walks only
`model.model.layers[routed_layer_indices]` (the backbone), while the MTP stages live
in `model.mtp.layers` (a plain Python list), so no expert-streaming runtime, no
`mx.eval(indices)` routing barrier, no `.tolist()` route planning, no layer lock and
no deferred release ever reach the draft MoE. No per-call dequant / `astype` / repack
happens either — `gather_qmm` dequantizes on-device inside the kernel. Because Part
B's precondition ("goes through the streaming switch / barrier, or a per-call
dequant") is therefore **false**, the new `MTPLX_DSV41_DRAFT_RESIDENT_MOE` lever was
**not built** (it would only duplicate the existing path). Parts C and D are delivered
for the existing draft levers (`DRAFT_COMPILE` + the W103 `DRAFT_HEAD_BF16`). The
83 ms remainder attributed to the draft MoE in the window-39 receipt was an inference
by subtraction, not a measurement; it is **not** in the MoE — see "Next suspect".

## Corrected `gather_qmm(mode=)` fact (the module docstring is stale)

`mtplx/models/deepseek_v41_dspark.py:42-43` (and the mirror at `:39-41`) claims:

> `mx.gather_qmm` in mlx 0.32.2 has no `mode=` argument, so the resident mxfp4 path
> is the SwitchGLU quantised matmul, not a bespoke `gather_qmm(mode=)`.

This is **doubly wrong on this box** (verified 2026-09-12, `mlx` 0.32.2):

1. `mx.gather_qmm` **does** take `mode: str = 'affine'` with an optional
   `biases: array | None = None`; so does `mx.quantized_matmul`
   (`help(mx.gather_qmm)` / `help(mx.quantized_matmul)`, transcribed below).
2. The "SwitchGLU quantised matmul" **is** `mx.gather_qmm(mode="mxfp4")`:
   `mlx_lm.models.switch_layers.QuantizedSwitchLinear.__call__`
   (`.venv/.../mlx_lm/models/switch_layers.py:75-91`) is literally
   `mx.gather_qmm(x, w, scales, biases, rhs_indices=..., transpose=True,
   group_size=self.group_size, bits=self.bits, mode=self.mode, ...)`. The two are the
   **same op**, not alternatives — the resident mxfp4 path *is* a `gather_qmm(mode=)`.

```
gather_qmm(x, w, /, scales, biases=None, lhs_indices=None, rhs_indices=None,
           transpose=True, group_size=None, bits=None, mode='affine', *,
           sorted_indices=False, stream=None) -> array
```

mxfp4 gs32 runs on **both** Metal and CPU. Verified: `mx.quantize(w, group_size=32,
bits=4, mode="mxfp4")` returns `uint32` packed codes + `uint8` E8M0 scales and **no
bias leaf**; `mx.gather_qmm(..., mode="mxfp4")` evaluates on CPU (used by the
CPU-pinned test and census here).

## Exact trace of `self.mlp(moe_input)` for the 3 MTP stages

`DSparkBlock.__call__` (`mtplx/models/deepseek_v41_dspark.py:739-741`):

```
with _moe_compile_window(use):
    x = self.mlp(moe_input)
```

`self.mlp` = `MoE(args.num_hidden_layers + stage_id, _mtp_moe_args(args))`
(`deepseek_v41_dspark.py:623`), i.e. `mtplx.models.deepseek_v41_moe.MoE`
(imported `deepseek_v41_dspark.py:63`, defined `deepseek_v41_moe.py:349-474`).

`MoE.__call__` (`deepseek_v41_moe.py:415-474`) on the decode/draft path:

- **gate / top-k** — `weights, indices = self.gate(xf)` → `Gate.__call__`
  (`deepseek_v41_moe.py:271-310`): score GEMM, `sqrtsoftplus`, correction bias,
  `mx.argpartition` + `mx.argsort` + `mx.take_along_axis`, `norm_topk_prob`,
  `route_scale`. **All lazy MLX ops — no `mx.eval`, no `.tolist()`.** The
  "routing barrier (`mx.eval(indices)`)" mentioned in the `deepseek_v41_moe.py:419-425`
  comment is the *stage-timing* fence (`_st.add(weights, indices)`), which is a **no-op
  when timing is off** (`deepseek_v41_stage_timing.py:100-108` `_NullFence.add`;
  `stage()` returns the shared `_NOOP_CM` when `_ACTIVE is None or not
  p._recording_now`, `:446-456`). Default decode = timing off = no fence, no barrier.
- **routed switch** — `routed = self.switch_mlp(xf, indices)` (`deepseek_v41_moe.py:465`).
  For the MTP stages `switch_mlp` is a resident mlx-lm `SwitchGLU` (constructed
  `deepseek_v41_moe.py:371-376`) whose `SwitchLinear` leaves became
  `QuantizedSwitchLinear(mode="mxfp4", gs32)` via `nn.quantize(self.mtp, group_size=32,
  bits=4, mode="mxfp4", class_predicate=_make_mtp_expert_quant_predicate(32))`
  (`deepseek_v41.py:4030-4035`; predicate `deepseek_v41.py:3823-3838` matches
  `"switch_mlp" in path`). `SwitchGLU.__call__`
  (`mlx_lm/.../switch_layers.py:176-200`) runs `up_proj`, `gate_proj`, `down_proj`,
  each a single `mx.gather_qmm(mode="mxfp4")` (`:75-91`) over the `block_size × top_k`
  rows — three gathers, all lazy. `_gather_sort` only fires at `indices.size >= 64`
  (draft is `block_size × 3` rows, well under), and is itself pure argsort/gather.
  **No `mx.eval`, no `.tolist()`, no per-call dequant / `astype` / repack** — the
  quant→float expansion happens on-device inside the `gather_qmm` kernel.
- **shared expert + combine** — `deepseek_v41_moe.py:467-472`: dense `Expert` +
  f32 weighted sum. Pure.

The MTP stages are **not** rebound to streaming. `bind_streamed_switches`
(`mtplx/models/expert_mlx.py:3801-3877`) walks `model.model.layers[layer_index]` for
`layer_index in runtime.spec.routed_layer_indices` — the **backbone** decoder layers
only. The DSpark stages are `model.mtp.layers` (`DSparkHead.layers`, a plain list,
`deepseek_v41_dspark.py:841`; `self.mtp` attached at `deepseek_v41.py:4015`), never in
`model.model.layers`, so no `HotExpertSwitchGLU` / `ExpertStreamingRuntime` /
`route_waves` / `try_all_hit_route` / `layer_lock` / `flush_deferred_slot_releases`
touches them. The decode driver states this outright
(`mtplx/models/deepseek_v41_dspark_decode.py:788-790`):

> The 3 DSpark stages run RESIDENT mxfp4 experts (SwitchGLU), never the streamed
> switch, so drafting stays on resident weights (no phase issue). … One host sync per
> cycle (the `mx.eval` below), never per markov step — the markov argmax stays lazy.

## Sync census (fresh, CPU-pinned) — `scripts/deepseek_v41/w104_draft_moe_sync_census.py`

Wraps `mx.eval` / `mx.async_eval` / `mx.array.tolist` / `mx.array.item` with counters
that tag each call with its deepest project `file:line`, then runs one `draft_block`
(3 stages) WITHOUT a terminal eval and counts host syncs, plus an isolated,
gs32-aligned `MoE` repacked to the real mxfp4 `QuantizedSwitchLinear`:

```
### draft_block (3 stages), DRAFT_COMPILE=OFF
    host syncs DURING the call (before the terminal eval): 0
    of which originate in the MoE files: 0
### draft_block (3 stages), DRAFT_COMPILE=ON
    host syncs DURING the call (before the terminal eval): 0
    of which originate in the MoE files: 0
### isolated resident MoE, switch leaf=QuantizedSwitchLinear, out (1, 5, 64)
    host syncs DURING the call (before the terminal eval): 0
    of which originate in the MoE files: 0
```

**Draft-MoE host syncs per draft cycle: 0 (before AND after — the path is already
barrier-free).** The whole 3-stage `draft_block` is one lazy graph; the only host sync
per cycle is the caller's single terminal `mx.eval` in `deepseek_v41_dspark_decode.py`,
which drains all three stages at once and belongs to no single stage.

## Where the draft-cycle work actually is (next suspect for the 83 ms)

`scripts/deepseek_v41/dispatch_census.py --dspark-draft` (unchanged, W103) — primitives
/ non-view kernels per stage, summed over the 3 stages, per depth-5 draft cycle:

| group | primitives | non-view kernels | share of non-view |
|---|---:|---:|---:|
| attn (main_kv + qkv/out prep + sdpa) | 759 | 308 | 42% |
| hc (attn_prep + ffn_prep + moe_combine) | 542 | 261 | 36% |
| **moe (gate_topk + routed_switch + shared + combine)** | **183** | **108** | **15%** |
| head + embed + sample/markov/confidence | 121 | 53 | 7% |
| **total / cycle** | **1605** | **730** | |

The MoE's 183 / 108 is exactly the count the task cited, and it is only ~15% of the
draft cycle's kernels — dispatch-light AND sync-free. The 83 ms cannot be the MoE. The
draft cycle is dominated by the 3 stages' **sliding-window MLA attention (759 / 308)**
and **Hyper-Connection Sinkhorn prep (542 / 261)** — long chains of tiny B=`block_size`
kernels whose wall time is host-encode + GPU-DVFS-downclock bound at low M (memory:
`b1-decode-dispatch-removal-hides`, `hy3-decode-roofline`). That is precisely what
`MTPLX_DSV41_DRAFT_COMPILE` (K33/W65) folds into `mx.compile` tapes. **Recommended next
measurement:** run the existing `--dspark-draft` census under the `--stage-timing`
probe on the real 16K cell so the 251 ms `draft_ms` (window-39 `dspark.per_cycle_ms`)
is split across `dspark.attn.*` / `dspark.hc.*` / `dspark.moe.*` / `dspark.head`
*by measurement* instead of assigning the unbracketed remainder to the MoE. The
prior on the split is attention + HC ≫ MoE.

## What was built

- **B — not built (precondition false).** The draft MoE is already the barrier-free
  resident `gather_qmm(mode="mxfp4")` the lever would add; a parallel
  `MTPLX_DSV41_DRAFT_RESIDENT_MOE` would be a byte-identical duplicate of the current
  path with no dispatch or sync reduction. The stale docstring at
  `deepseek_v41_dspark.py:39-43` was corrected in place (comment-only).
- **C — ab arms** in `scripts/deepseek_v41/ab_decode_env_levers.py` (preset table
  only; `_generate_dspark` untouched): registered `MTPLX_DSV41_DRAFT_HEAD_BF16` in
  `ALL_LEVER_ENVS` + a `draft_head_bf16=` `_preset` kwarg; added
  `cell16k_ring_v2_draft` (= `cell16k_ring_v2` + `DRAFT_COMPILE=1` +
  `DRAFT_HEAD_BF16=1`) and updated `cell16k_ring_draft` (= `cell16k_ring` +
  `DRAFT_COMPILE=1` + `DRAFT_HEAD_BF16=1`). **Note:** `cell16k_ring_draft` previously
  pinned `DRAFT_COMPILE` only (W81); it now also pins `DRAFT_HEAD_BF16`. The
  `DRAFT_COMPILE`-only isolation is still available as the standalone `draft_compile`
  arm; `DRAFT_HEAD_BF16` in isolation is covered by `tests/test_deepseek_v41_w103_draft.py`.
- **D — tests** in `tests/test_deepseek_v41_w104_draft_moe.py` (CPU-pinned): the
  resident draft MoE issues no host sync (patched `mx.eval`/`tolist`/`item`/`async_eval`
  counter, over 64 draft cycles), the real mxfp4 `QuantizedSwitchLinear` path is
  sync-free, `DRAFT_COMPILE` on==off greedy `output_ids` identity over 64 cycles, the
  W103 engagement counter, and the two draft levers read at use. `test_deepseek_v41_ab_env_levers.py`
  arm assertions extended for the new key and the two arms.
