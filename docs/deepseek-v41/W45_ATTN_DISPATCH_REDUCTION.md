# W45 — Further attention decode dispatch reduction (kernel-ledger K24)

Branch `feat/deepseek-v41-w45` off `feat/deepseek-v41-streaming` @ `5b6b8e7d2`.
Files: `mtplx/models/deepseek_v41.py` (the K24 window-mask memo),
`scripts/deepseek_v41/dispatch_census.py` (census extended for the memo + a
before→K22→K22+K24 table), `tests/models/test_deepseek_v41_attn_win_memo.py`
(new, 13 tests), `scripts/deepseek_v41/ab_decode_env_levers.py` +
`tests/test_deepseek_v41_ab_env_levers.py` (the `attn_win_memo` arm), this report,
`docs/deepseek-v41/KERNEL_LEDGER.md` (§K24 + gate KG-j). CPU-only, tiny synthetic
config, no artifact, no GPU/Metal. Peak RSS of the new test file: **0.12 GB**.

Context: window-15 (docs/deepseek-v41/receipts/gpu-windows/window-15/) showed
`stack_a` (head_bf16 + sinkhorn_metal + attn_compile) at 6.24 tok/s, +19 % over
head_bf16 alone — the K22 compiled chains help once the head trap is gone — while
the fenced stage timing still put `attn.reuse` at ~1.9 ms per M=1 layer (a
dispatch-chain cost). This task pushes the attention decode dispatch count lower.

## Verdict

Using the census to evaluate the four proposed options, **the attention SDPA is
reduction/matmul-bound and cannot be compiled-away** (option a is dead, with
evidence), while the census surfaced a *different*, larger, universally
byte-identical lever the option list did not name: the **sliding-window attend
mask is identical across all 40 backbone layers of one decode forward** and is
rebuilt from scratch every layer. K24 memoizes it on the per-forward `shared`
runtime behind `MTPLX_DSV41_ATTN_WIN_MEMO` (default OFF), computing it once and
reusing the identical array for the other `n_layers - 1` layers.

**Census (tiny 8-layer real-structure model), attention primitives/token:**

| | eager | K22 (attn tape) | K22 + K24 (memo) |
|---|---:|---:|---:|
| attention primitives/token | 1698.7 | 1378.7 | **1301.7** |
| whole-token primitives | 6630.7 | 6262.7 | **6185.7** |

K24 removes the ~11-node window mask from 7 of 8 layers (−77 attn prim/tok here);
at the real 40 layers it scales to **~11 × 39 ≈ 429 primitives/token**, larger
than K22's own 368. Flag on/off is `mx.array_equal` (f32 CPU) over decode, K+1
verify, chunked + layer-major prefill — with **and without** the K22 tapes (the
memo composes with and is independent of K22); the reduction is asserted from the
census; every existing DSV4.1 suite is green.

## 1. Options (a)–(d), evaluated with the census

### (a) shapeless tape for the whole M=1 attention step (score/softmax/value) — DEAD
The coordinator's hypothesis was that shapeless failed in K4 *only* because of the
Sinkhorn reshape, which is outside attention. Measured on CPU (mlx 0.32.2), the
attention SDPA has its **own** independent shapeless blockers, and compiling it
would not reduce dispatches anyway:

* **It does not trace shapeless.** A `shapeless=True` `mx.compile` of
  `_sparse_attend` fails: the sink-column drop `softmax(full)[..., :KV.shape[1]]`
  raises `[Primitive::output_shapes] Slice cannot infer output shapes` (dynamic
  slice), and the value einsum `bsht,btd->bshd` raises `[reshape] Cannot reshape
  array of size … ` at T ≥ 20 (its internal reshape over the dynamic T). (Isolated:
  einsum/softmax/where/matmul each trace shapeless; the **dynamic slice** is the
  hard blocker.)
* **Rewrites to avoid the slice break byte-identity or still don't reduce
  dispatches.** Folding the sink into the softmax denominator (no concat, no
  slice) diverges from the shipped `softmax(concat)+slice` **even eager**, 20/20
  seeds — the denominator sum over `T` vs `T+1` reassociates. A padded-KV variant
  (keep the exact `softmax(concat)`, append one zero KV row, einsum over `T+1`, no
  slice) *is* byte-identical at T = 8 (0/30 eager and shapeless-compiled) but (i)
  still crashes at T ≥ 20 on the einsum reshape and (ii) has **more** primitives
  compiled than eager (24 vs 22): the SDPA is two einsums + a softmax + a masked
  `where` — reductions/matmuls that `mx.compile` cannot fuse — with almost no
  elementwise chain to collapse. Compiling it saves ≈ 0 dispatches while adding a
  tape-call boundary.

**Conclusion:** the fusable elementwise dispatch in attention was already captured
by K22's prep tapes; the SDPA's remaining ~22 primitives are irreducible
reductions. Do not spend a GPU window on a shapeless SDPA tape.

### (b) fold the KV-cache append into the tape (functional update) — NOT PURSUED
The append is a slice-assign, not a dispatch a tape removes; its only value would
be to let the SDPA join a tape (feeding the post-append window in as an input) —
i.e. it only exists to enable (a), which is dead. On its own it reduces no
dispatches.

### (c) concatenate weights (wq_a ⊕ wkv) into one matmul — VIABLE, QUANTIZED-ONLY
`wq_a` and `wkv` both project the attention input `x`, so
`concat([wq_a; wkv], axis=0)` computes both in one matmul (−1 matmul/layer),
splitting the output statically. Measured bit-exactness:

* **quantized residents (serving): bit-exact** — `mx.quantized_matmul` processes
  each output row with a fixed per-group reduction, independent of the number of
  output rows, so the concatenated call equals the two separate calls to the bit
  (verified).
* **dense (the tiny test config): NOT bit-exact** — MLX's dense GEMM tiling over
  the output dimension changes the accumulation grouping, so `x@wq_a.T` vs
  `(x@concat.T)[:, :qlr]` differ ~1 ULP.

So (c) is a real but modest quantized-only lever (~1 matmul × n_layers/token). It
is **not bundled** into the W45 lever, to keep K24 byte-identical *and* testable on
the standard dense config; it is recorded here for a future quantized-gated add-on.

### (d) share one compiled callable across the identical-shape CSA layers — ALREADY DONE
K22 already builds **one** `qkv_prep` tape and **one** `out_prep` tape keyed on the
projection codec + head geometry and shared across all 40 layers (weights are tape
inputs); `tests/models/test_deepseek_v41_attn_compile.py::test_one_qkv_tape_shared_across_layers`
asserts exactly one of each. Nothing to add.

## 2. K24 — the window-mask memo (implemented)

`Attention._attend` builds the causal sliding-window attend mask
`attend = broadcast((wp <= qp) & (wp > qp - window_size), [b, s, T])` every layer.
Within one `_forward_span` the query `positions` and the per-forward `shared`
runtime are created once and handed to every layer, and every layer's window store
grows in lockstep to the same `T`, so this mask is **identical across all layers**
— the census's biggest remaining mode-invariant per-layer chunk (11 graph nodes:
`Arange`, 2 compares, `Subtract`, `BitwiseAnd`, + broadcasts/expand-dims).

`Attention._window_attend(positions, T, b, s, shared)` memoizes it on `shared`
under `MTPLX_DSV41_ATTN_WIN_MEMO`, reusing the identical array only when
`positions` **is** the same object and `(T, window_size, b, s)` match — so it is
byte-identical (the reused array is the same object), and any forward whose layers
differ (or where `shared` is absent) falls back to per-layer recompute, never
wrong. The `shared` runtime has no `__slots__`, so the memo is a plain attribute
set from `deepseek_v41.py` — **no edit to W13's cache module**. Default OFF; read
through the module global (tests/operators flip after import,
[[env-flags-read-at-use-not-import]]).

Composes with K22 (the memo feeds the eager `where`/SDPA either way) and is
independent of it (works with the tapes on or off).

## 3. Census table — primitives/token per decode stage (before → K22+K24)

Tiny real-structure model (2 swa_only, 2 full, 3 reuse, 1 reindex), CPU:

| stage | calls/tok | prim/tok before | prim/tok after (K22+K24) | Δ |
|---|---:|---:|---:|---:|
| `hc.premix_sinkhorn` | 16 | 4144.0 | 4144.0 | 0 |
| `attn.full` | 2 | 607.0 | 505.0 | 102 |
| `attn.reuse` | 3 | 486.0 | 333.0 | 153 |
| `attn.swa_only` | 2 | 321.0 | 230.0 | 91 |
| `attn.reindex` | 1 | 284.7 | 233.7 | 51 |
| `moe.routed_switch` | 8 | 216.0 | 216.0 | 0 |
| `moe.gate_topk` | 8 | 200.0 | 160.0 | 40 |
| `hc.combine` | 16 | 168.0 | 168.0 | 0 |
| `moe.shared_expert` | 8 | 112.0 | 112.0 | 0 |
| `moe.combine` | 8 | 48.0 | 40.0 | 8 |
| others (final_norm/engram/embed/head/sample) | — | 44.0 | 44.0 | 0 |
| **TOTAL** | | **6630.7** | **6185.7** | **445** |

The per-attention-call Δ is the mode-invariant K22 −40 plus the K24 −11 on every
layer past the first (`attn.reuse` −153 = 3 calls × (40 + 11); `attn.reindex` −51 =
1 × (40 + 11)). `hc.premix_sinkhorn` (K3/K4), the routed switch (W42), and the
shared expert are untouched (asserted).

## 4. Tests (`tests/models/test_deepseek_v41_attn_win_memo.py`, CPU, tiny, no artifact)

13 tests, `mx.set_default_device(mx.cpu)`, cap 7:
1–8. flag on/off `mx.array_equal` over decode / K+1 verify / chunked + layer-major
   prefill / one-shot — parametrized over K22 compile off **and** on (the memo
   composes with the tapes and stands alone).
9. `test_census_memo_reduces_attention_dispatches` — the memo REDUCES attention
   primitives/token (asserted from the census; ≥ 5 real dispatches × 7 layers),
   and the stages K24 doesn't touch are unchanged.
10. `test_census_memo_composes_with_k22` — K22 + K24 < baseline (attention + total).
11. `test_window_attend_reuses_same_object_and_recomputes_on_change` — direct memo
   semantics: same key → the SAME array object (reuse); a different `positions`
   object or `T` → fresh; flag off → never memoized but byte-identical value.
12. `test_env_default_off`.

**Verification (CPU, `nice -n 19`, no `-n auto`):**
- `test_deepseek_v41_attn_win_memo.py` — **13 passed**.
- `test_deepseek_v41_ab_env_levers.py` — **passes** (the `attn_win_memo` arm +
  independence over the 8 boolean keys).
- Full `tests/models/test_deepseek_v41_*.py` + `tests/test_deepseek_v41_*.py` —
  **346 passed, 42 skipped**, no failures. K24 default OFF, byte-identical off.

## 5. A/B arm + follow-up

`attn_win_memo` (`_preset(win_memo="1")`) added to `ARM_PRESETS`; `MTPLX_DSV41_ATTN_WIN_MEMO`
added to `ALL_LEVER_ENVS`, `all_levers`, and `stack_a` (it is byte-identical and a
pure dispatch cut, so it joins the winning stack). Dry-run verified for
`--arms control attn_win_memo stack_a all_levers`. The realized GPU decode delta is
KG-j (unmeasured); on a pass a one-line default flip lands it. Not changed:
`expert_mlx.py` (W42), the lm head (W40), the K22 tapes, the SDPA / Indexer
selection, the cache modules.
