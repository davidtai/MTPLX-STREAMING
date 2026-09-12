# W91 / K35 — layer small-stages fusion (AR-decode dispatch collapse)

Kernel-ledger **K35**, env key `MTPLX_DSV41_SMALL_STAGES_FUSED` (default OFF).
Sub-lever `MTPLX_DSV41_HC_PREMIX_KERNEL` (default OFF, GPU-only). A/B arms
`small_stages_fused` and `cell16k_ring_fused` in
`scripts/deepseek_v41/ab_decode_env_levers.py`. All numbers below are the CPU
tiny real-structure model (`hidden=32, 8 layers, hc_mult=4, hc_sinkhorn_iters=20,
n_routed=8, top-2` — the same HC/Sinkhorn/MoE geometry as the 40-layer artifact);
per the worker contract the CPU is a **dispatch-count** proxy — the ms/token
number is a GPU-window measurement.

Census: `scripts/deepseek_v41/dispatch_census.py --small-stages`.
Tests: `tests/models/test_deepseek_v41_small_stages_fused.py` (12: 11 CPU + 1
GPU-gated parity). The HC scale/base/fn weights are **F32** on the artifact (read
from `model-00003.safetensors`' header, layer-0 `hc_attn_scale/base/fn` +
`hc_ffn_*`), so the fused premix kernel's f32 I/O matches — no bf16 combination is
needed in the parity test.

## The target

At M=1 the per-layer *small* stages — everything that is **not** attention and
**not** the routed-expert switch — are the top per-token dispatch source. Each is
a chain of tiny kernels whose host-encode latency (~0.02–0.05 ms/dispatch) the GPU
never notices ([[b1-decode-dispatch-removal-hides]]). K4 (`HC_COMPILE`) collapses
only the HC chains; K22 (`ATTN_COMPILE`) folds only the gate *prefix* + MoE
combine. K35 collapses the **whole** small-stage set into three compiled per-layer
graphs, separated only by the two un-fused data-dependent calls:

```
 h ─▶ seg1: input norm + attn HC premix (Sinkhorn) ─▶ attn_input
        │
        ▼  attention  (writes this layer's KV — UN-FUSED)
        │
      seg2: attn HC combine + ffn HC premix (Sinkhorn) + gate/top-k + shared expert
        │       ─▶ (xf, weights, indices, shared, carry)
        ▼  routed switch  switch_mlp(xf, indices)  (expert gather — UN-FUSED)
        │
      seg3: MoE combine (weighted routed sum + shared) + ffn HC combine ─▶ h'
```

## (1) Dispatch + host-sync count per call — before / after

Per **stage** (eager, per call), and the fused **segment** view (per layer, M=1):

| stage (eager)          | dispatches/call | host syncs/call |
|------------------------|-----------------|-----------------|
| hc.premix_sinkhorn     | 259 (198 = Sinkhorn recurrence)¹ | **0** |
| hc.combine             | 10.5            | **0** |
| moe.gate_topk          | 25              | **0**² |
| moe.shared_expert      | 14              | **0** |
| moe.combine            | 6               | **0** |
| **small-stage total/layer** | **584**    | **0** |

| fused segment | eager | `mx.compile` (fixed-shape) | + K3 Sinkhorn kernel (GPU) | Sinkhorns |
|---------------|-------|----------------------------|----------------------------|-----------|
| seg1          | 259   | 111                        | 32                         | 1 |
| seg2          | 308   | 141                        | 62                         | 1 |
| seg3          | 17    | 13                         | 13                         | 0 |
| **TOTAL/layer** | **584** | **265 (−55%)**           | **107 (−82%)**             |   |

Per token (×8 layers here; ×40 on the artifact): eager 4672 → `mx.compile` 2120 →
+K3 kernel 856. ¹ One compiled Sinkhorn (hc=4, iters=20) is **80** graph
primitives (198 eager); the K3 Metal kernel collapses it to **1** dispatch (the
`+K3 kernel` column). ² the gate stage itself is pure/lazy — see (2).

**Host syncs are the key finding.** On the tiny model, one *production* decode
step (no timing fences) is **exactly 1** `mx.eval` — the sampler. Every small-stage
pure function triggers **zero** `mx.eval` / `.item()` / `.tolist()`: they are pure
lazy graph-building. The whole token is one lazy graph, evaluated once. So the
small stages are **dispatch-bound, not sync-bound** — the fix is to cut dispatches
(fewer kernel launches / less host-encode), which K35 does.

The real model's ~40 host barriers/token are the **routing barrier** —
`mx.eval(indices)` + a `.tolist()` + a Python `int()` loop over the selected expert
ids, once per layer, inside the **streamed switch** (`mtplx/models/expert_mlx.py`
lines 2236-2237 and 2620). That is the **un-fused routed switch**, not a small
stage, and it is already removed by the `device_route` lever (expert_mlx.py:2388:
"Gathers `lut[indices]` on the device — NO `mx.eval(indices)`, no `.tolist()`, zero
host syncs on this layer"). K35 leaves it alone by design.

## (2) The dumb ones, named

1. **premix — NOT a hidden sync.** The 0.40 ms/call (window-33, K3 kernel already
   engaged) is ~11 tiny elementwise dispatches around the Sinkhorn: `astype(f32)`,
   `flatten`, `mean(square)`/`rsqrt` RMS-norm, the `fn` matmul, then the
   `hc_split_sinkhorn` **split** — three slices (`mixes[:hc] / [hc:2hc] / [2hc:]`),
   two affine transforms, two `sigmoid`s, `+eps`, a reshape — then the `_hc_pre`
   collapse (`expand`, `multiply`, `sum`) and a second RMS-norm. **Fix:** fuse
   (seg1/seg2) → `mx.compile` replays one tape; the `HC_PREMIX_KERNEL` folds the
   whole split *into* the Sinkhorn kernel (11 split dispatches + Sinkhorn → **1**).
2. **combine upcasts the 4 HC streams to f32 every call — CONFIRMED.**
   `_hc_post_impl` does `residual.astype(f32)` on the `[b, s, hc, dim]` stream (the
   4 HC copies) *and* `x.astype(f32)` every call, then an `einsum` + broadcast-mul +
   add + cast-back (hc.combine 10.5, moe.combine 6 dispatches). The upcast is
   **required** for the f32 accumulation (bit-exactness) so it cannot be removed,
   but it is fused into seg2/seg3 (no separate launch, and no host sync).
3. **gate_topk — the `.tolist()` is NOT here.** The stage itself is pure/lazy: gate
   matmul (384) + sqrtsoftplus + bias + `argpartition`(384) + `argsort` + two
   `take_along_axis` + renorm + scale = ~25 dispatches, **0 syncs**. The
   `.tolist()`/`int()` of the indices the coordinator flagged is the **streamed
   switch's** (item above); the W37 fence *attributes* the routing barrier to the
   gate_topk bracket because it evals `indices` there. **Fix:** fold the gate
   compute into seg2; the barrier stays the un-fused switch's.
4. **"Sinkhorn called twice per layer with an eval between" — FALSE in production.**
   There is no `mx.eval` between the two premixes; the fenced W37 stage timing puts
   one at each bracket (≤0.05 ms). The K3 kernel is one dispatch per call.

## (3) The fix — byte-identical, before/after

`small_stages_fused` is **exact-by-construction**: every op in seg1/seg2/seg3 is
the same fp32 op in the same order as the eager `DecoderLayer` / `MoE` bodies;
`mx.compile` only fuses adjacent elementwise runs and replays one prebuilt tape.

- **Byte-identity over the whole admitted range** (`mx.array_equal`, f32 CPU),
  flag on vs off — proven in the test suite over 64 decode steps (n=1), a K+1
  verify batch, and every admitted row count (1, `cap//2`, `cap`). The cap is
  **`_SMALL_STAGES_MAX_ROWS = 7`**, the mx.compile bit-exact regime: this tiny/real
  HC matmul + RMS/HC-mix mean reductions are `array_equal` with eager up to 7 rows
  and reassociate ~5e-7..1.2e-6 at ≥8 rows / one-shot prefill s≥8 (W33), so the cap
  keeps the lever off that band entirely. Decode (n=1) and the DSpark K+1 verify
  (≤6 rows) are fully covered. The fold of the gate top-k + shared expert + MoE
  combine adds **zero** fp delta beyond the HC collapse (seg2/seg3 outputs equal the
  real `Gate`/`Expert`/combine module outputs bitwise). This CPU byte-identity is a
  necessary condition; **n=1 GPU byte-identity must be established by the parity
  window itself** (the shared HC-compile tape family, K4, is also OFF by default and
  has no warm GPU receipt yet, so K35 cannot inherit one).
- **No per-token retrace:** exactly 3 tapes (seg1/seg2/seg3) are traced once and
  replayed for all 64 steps (`len(_SMALL_STAGES_COMPILED) == 3`, zero growth).
- **Engagement telemetry:** `_small_stages_calls()` (fused vs eager layer forwards)
  and `_hc_premix_kernel_calls()` are reset per arm and emitted as
  `small_stages_engagement` in the ab receipt (next to `sinkhorn_engagement`) — so a
  byte-identical, barely-faster arm can be told apart from "armed but forced eager".
  The flag is read **at use** (`_env_truthy`, like `_sinkhorn_metal_enabled`), so the
  server's late optimization-key stamp is honoured; both keys are in
  `openai.py::_DSV41_LEVER_ENV_KEYS`.
- **`mx.compile(shapeless=True)` is not usable on CPU:** MLX 0.32.2 raises
  `[Primitive::output_shapes] Slice cannot infer output shapes` on the
  `hc_split_sinkhorn` pre/post/comb slices and the gate top-k `[..., :topk]` slice
  under a shapeless trace — unaffected by static-width slices or `unflatten`
  restructuring (reconfirms the K4/W33 finding). Fixed-shape reaches the *same
  goal* at decode (the M=1 shape is stable → one trace, no per-token retrace). On
  the GPU the `HC_PREMIX_KERNEL` makes the premix one opaque op (declared
  `output_shapes`), which restores shapeless-compatibility there.

**Expected ms/token saving.** Applying the spec's 0.02–0.05 ms per removed
dispatch to the artifact (×40 layers): `mx.compile` removes ~319/layer = ~12.8k/tok
and +K3 kernel ~477/layer = ~19.1k/tok. That linear model is a loose **upper
bound** — most fused elementwise dispatches pipeline on the GPU; only the
host-encode of the launches on the M=1 critical path is actually cut. The
defensible ceiling is the fenced small-stage total the fusion targets
(hc.premix 32.7 + hc.combine 26.7 + gate_topk 19.4 + shared 13.3 + moe.combine 8.6
≈ **100 ms/token**, window-33). The real saving is a GPU-window A/B
(`cell16k_ring_fused` vs `cell16k_ring`); the dispatch collapse (584→107/layer,
−82%) is the deliverable.

**CPU host-wall proxy** (tiny 8-layer, decode n=1, eager prefill, 200 steps):
eager **11.28 ms/tok** → fused **7.885 ms/tok**, **−30.1%** (1.41 → 0.99 ms/layer).
This is `mx.compile`'s Python-graph-build + replay saving only (the Sinkhorn stays
the recurrence on CPU — the K3 kernel's 80→1 collapse is GPU-only), so it is a
lower bound on the GPU dispatch win.

## Reading the GPU window (counter caveats)

- **Sinkhorn `kernel_calls` under K35 is a TRACE count, not a per-token count.** The
  K3 engagement wrapper (`_sinkhorn_normalise`) runs only while `mx.compile` traces
  a segment, not on warm replay. So a fused arm's `sinkhorn_engagement.kernel_calls`
  reads ~2×(number of distinct traced shapes) — a handful — **not** the ~43k the
  eager arm shows. `engaged` (kernel>0, recurrence==0) is still the right signal;
  the magnitude is not comparable across K35 on/off. Same for `premix_kernel_calls`.
- **K1 `SHARED_OVERLAP` is a no-op under K35.** The fused path computes the shared
  expert inside seg2 (before the switch) and combines in seg3; it never calls
  `run_switch_with_shared_overlap`. seg2 already places the shared expert ahead of
  the switch's routing barrier, so the overlap is structural — do not stack the two.

## Env keys and arms

- `MTPLX_DSV41_SMALL_STAGES_FUSED=1` — the fused decode graphs (rows ≤
  `_SMALL_STAGES_MAX_ROWS = 7`, the byte-identical regime). Read at use.
- `MTPLX_DSV41_HC_PREMIX_KERNEL=1` — GPU-only; folds the HC premix split into the
  Sinkhorn kernel (one dispatch). Rounding-class (1e-6, argmax-exact), like K3;
  inert on CPU (reference `hc_split_sinkhorn` taken). Metal source in
  `deepseek_v41._hc_premix_sinkhorn_metal_kernel`. **NOT in any composite arm** and
  must not be until `test_hc_premix_kernel_parity_gpu` writes a clean GPU parity
  receipt (MTPLX_GPU_PARITY=1). **Scope:** it is routed through `_hc_mixes_split`,
  which only the compiled HC tapes call, so it is **inert unless a compiled HC tape
  (K4 `HC_COMPILE`, K33 draft, or K35 `SMALL_STAGES_FUSED`) is active** — the eager
  `DecoderLayer._mixes` path (prefill, and decode with all HC-compile off) calls
  `hc_split_sinkhorn` directly, where only the K3 Sinkhorn kernel engages. This is
  intended (the premix kernel is a decode/compiled-path optimization); it is not
  routed through eager `_mixes` to avoid changing the prefill Sinkhorn boundary.
- Arm `small_stages_fused` = `{small_stages, sinkhorn}`; `cell16k_ring_fused` =
  `cell16k_ring` + `small_stages` (neither sets `hc_premix_kernel`).
