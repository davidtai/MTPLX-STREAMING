# W60 — Fused decode / verify MLA attention Metal kernel (KERNEL_LEDGER K29)

> **VERDICT (window-27): SHELVED.** The kernel engaged on all 10,280 decode calls
> (0 fallbacks, split-K path) yet the `decode_attn_kernel` 1K/256 arm ran at
> **3.71 tok/s vs stack_a 6.01 (−38%)** — slower than the eager SDPA at M=1 even
> after the split-K occupancy fix — and GPU parity still shows **max|Δ| ~1e-3**
> (identical to window-26 to 8 digits: `precise::exp` changed nothing). The 1e-3 is
> **bf16-precision-class, not fixable** without matching the eager path's exact
> reduced-precision GPU op sequence (which defeats the fusion). Flag stays **default
> OFF**, arm is **in no stack**. See §7. This is the V4 "hand MLA fused-attention
> kernel" verdict recurring in the streaming/dispatch-bound regime (KERNEL_LEDGER
> §6 dead-here). Sections 0–6 below are the pre-verdict design record.

**Branch:** `feat/deepseek-v41-w60` (off `feat/deepseek-v41-streaming`).
**Flag:** `MTPLX_DSV41_DECODE_ATTN_KERNEL=1` (default OFF, GPU-only).
**Kernel + wrapper + references:** `mtplx/models/deepseek_v41_attn_kernels.py`.
**Integration:** `Attention._sparse_attend` (a small guarded early return) →
`Attention._decode_attn_kernel`, `mtplx/models/deepseek_v41.py`.
**Tests:** `tests/models/test_deepseek_v41_decode_attn_kernel.py` (39 CPU + 1
GPU-gated). **Arm:** `decode_attn_kernel` in
`scripts/deepseek_v41/ab_decode_env_levers.py` (pins all 21 keys; NOT in `stack_a`
until the parity window is clean).

Sibling of K22 (attention-chain prep tape) and K24 (window-mask memo): those cut
the *projection / norm / RoPE / mask-build* dispatch around the SDPA; **K29 cuts
the SDPA itself** — the score + mask + sink softmax + PV chain W45 measured as the
irreducible ~22-primitive tail that a shapeless `mx.compile` cannot fold.

---

## 0. Why (the measurement)

Windows 13–16 (`receipts/gpu-windows/window-15/stage-timing-head-bf16.json`,
`window-16/stage-timing-stack-a.json`, and W41/W45/W56) put **decode at 1,024
context at ~160 ms/token, dispatch-bound**:

- attention ≈ **60 ms/token** — `attn.reuse` 50.2 ms fenced over 30 layers ≈
  **1.5–1.9 ms per M=1 layer**, plus ~18 ms on the other 10 layers
  (`swa_only`/`full`/`reindex`);
- **~110 Metal primitives per layer** after the K22/K24 compiles;
- the actual math per layer at M=1 is tiny: 64 heads × head_dim 512 against
  ~1K–16K compressed keys (the CSA candidate mask + a per-head value-0 sink), plus
  the q-lora/norm/RoPE prep (K22) and the o-lora output (K22).

A single-row attention step costing 1.5–4.9 ms is a **dispatch-chain problem**
([[b1-decode-dispatch-removal-hides]]), not a FLOP problem. K22 collapsed the
pure prep chains (qkv 33 + out 7 = **40 prim/attention-call**, W41) and K24 the
window mask (**~429 prim/token** at 40 layers, W45). W45 then found the **SDPA
proper** (`_sparse_attend`: two einsums + softmax + mask + sink-concat + slice) is
an **irreducible ~22-primitive reduction** that a shapeless tape *cannot* remove —
the dynamic sink-column slice `softmax(full)[..., :KV.shape[1]]` raises
`Slice cannot infer output shapes`, and folding the sink diverges from the shipped
`softmax(concat)+slice` even eager. W45's verdict: *"Do not spend a GPU window on
a shapeless SDPA tape."*

**K29 is the viable alternative to that dead tape:** a hand `mx.fast.metal_kernel`
does not need shapelessness (T is a runtime scalar) and removes the concat/slice
entirely, collapsing the whole ~22-primitive SDPA into **one dispatch per layer**.

MLX's fused SDPA is not an option: head_dim 512 is unsupported by
`mx.fast.scaled_dot_product_attention`, and the value-0 sink semantics differ (W50
measured 2.1e-3 divergence vs a plain SDPA).

---

## 1. Kernel design

**One threadgroup per `(row, head)`** (`grid.x = TG · rows · H`, `rows = b·s`), `TG`
= 128 lanes (power of two; also the key-tile width — one key per lane per tile).
`H`, `T`, `S` (seq len, so `b_idx = row / S`) and `scale` are **runtime scalar
inputs**, so ONE compiled kernel serves every context length, batch and mode; `TG`
and `HD` (head_dim, statically 512) are compile-time `constexpr` (so `q_sh[HD]` /
`acc_sh[HD]` are statically sized).

**Threadgroup memory** (HD=512, TG=128): `q_sh[HD]` + `acc_sh[HD]` (query + running
value accumulator, 2·512·4 = 4 KiB) + `s_sh[TG]` (tile scores) + `p_sh[TG]` (tile
probs) + `red_sh[TG]` (tree-reduce scratch) = 3·128·4 = 1.5 KiB → **≈ 5.6 KiB**,
well under the 32 KiB limit.

**Per threadgroup:**

1. cooperatively load `q[row,head,:]` (head_dim floats) into `q_sh`; zero
   `acc_sh`;
2. **stream the T keys in TG-wide tiles** (online softmax — the `[64, T]` per-head
   score row is NEVER materialised, so T = 16K+ streams through threadgroup
   memory):
   - each lane computes its key's masked score `score = scale · Σ_d q_sh[d]·k[t,d]
     + add` (a head_dim dot; masked lane → `NEG` sentinel);
   - a threadgroup **tree** reduces the tile max `m_tile`;
   - all lanes fold the tile into the running `(m, denom, acc)`:
     `m_new = max(m_run, m_tile)`; `corr = exp(m_run − m_new)`;
     `acc *= corr`; `denom = denom·corr + Σ_tile exp(s − m_new)`;
   - the head_dim axis is split across lanes to add
     `Σ_tile p_k · v[k]` into `acc` (each lane owns dims `{lane, lane+TG, …}` — no
     atomics, no cross-lane write conflict);
3. **fold the per-head value-0 sink into the denominator once**
   (`denom += exp(sink[head] − m_run)`) and write `out[head,:] = acc / denom`
   (guarded `denom > 0` → a fully-masked row writes 0, never a NaN).

MLA: `k_cache` and `v_cache` are the **one shared latent** (RoPE already baked into
q and the cached latents upstream), passed as both inputs; the score dot and the PV
sum both run over all `head_dim` dims.

**Reductions / threads:** the tile max and tile denom are power-of-two tree
reductions over `red_sh` (`TG >> 1 … 1`), each bracketed by a threadgroup barrier;
`p_sh` is kept intact across the denom reduction (it feeds the PV pass). The `NEG =
-3e38f` **finite** sentinel keeps the online combine NaN-free: an all-masked lane's
state is `(NEG, 0)` and `NEG − NEG = 0` → `exp = 1` → `0·1 = 0`, never
`0·exp(nan)`.

### Dispatch count — before / after, per layer

| | eager `_sparse_attend` (M=1) | K29 |
|---|---:|---:|
| QK^T einsum | matmul (+reshape) | — |
| scale · | 1 | — |
| mask `where` (+attend broadcast) | ~2 | — |
| sink reshape + broadcast | ~2 | — |
| `concatenate([scores, sink])` | 1 | — |
| `softmax` (max/exp/sum/div) | ~4 | — |
| sink-column slice | 1 | — |
| PV einsum | matmul (+reshape) | — |
| **fused kernel** | — | **1** |
| additive-mask `where` (bool→{0,−inf}) | — | ~1–2 |
| q / KV contiguity (usually no-op on decode) | — | ~0–1 |
| **≈ primitives / layer / call** | **~22** (W45) | **~2–4** |

Mode-invariant (all four CSA modes — swa_only / full / reindex / reuse — route
through the identical `_sparse_attend`, so the delta applies to every layer). At 40
layers this is **≈ −760 primitives/token** on the attention SDPA — larger than
K22's whole-token −368 and K24's −429, and it lands on the chain W45 proved a tape
cannot touch.

### Expected ms/token saved (ESTIMATE — GPU-window-gated, NOT measured)

The SDPA is ~22 of the ~162 primitives in a reuse attention call (W45 micro-census;
qkv 81 + out 38 + SDPA ~22 + window/select). As a **dispatch-uniform proxy** on the
~60 ms/token attention, the SDPA chain is ~14% ≈ **~8 ms/token** at 1K; K29's
ceiling is that minus the kernel's own single dispatch + compute (~70M MACs/layer
at T=1088, sub-100 µs). Central estimate **−6…−12 ms/token at 1K** (160 → ~148–154,
~4–8%). This carries the **same roofline caveat as K22/K25/K28** — window-20
overturned K25's FLOP roofline — so it is an estimate; the paired in-window A/B is
the gate. At 16K the reuse-layer kernel compute grows ~15× (still ONE dispatch),
so the *dispatch* win holds while the per-kernel compute rises — the net at 16K is
the window's to measure.

Unlike K28 (a prefill memory/pass lever — the `[1024,64,T]` transient is up to ~6
GiB), K29 is a pure **decode dispatch** lever: the M=1 score row is only `[1,64,T]`
(~4 MB at 16K), so there is no material peak-GB relief — the win is dispatch count.

---

## 2. Integration (minimal, mode-agnostic)

`Attention._sparse_attend` gets a **guarded early return** at the top (before the
`s <= 1` decode / prefill split):

```python
if _decode_attn_kernel_use(q):          # flag on + Metal-GPU default + b*s <= 8
    out = self._decode_attn_kernel(q, KV, attend)
    if out is not None:                 # None == unsupported mask shape -> eager
        return out
# ... unchanged eager decode one-shot / prefill score path (W58/W59) ...
```

- **Small-M gate.** `_decode_attn_kernel_use` fires only for `b·s <=
  _DECODE_ATTN_KERNEL_MAX_ROWS` (8): M=1 decode and the `K+1` verify batch (MTP
  depth ≤ 7). A prefill wave (`s` large) never enters — **W58's prefill score
  path (one-shot / lean / chunked) and W59's key selection are untouched**, which
  is why the hook sits above the existing branch and is deliberately tiny.
- **Mode-agnostic.** `_decode_attn_kernel` consumes the already-assembled
  `[b,s,T]` boolean `attend` mask shared across heads — exactly what all four CSA
  modes produce here (`swa_only`'s window mask; `full`/`reindex`/`reuse`'s
  `concatenate([window_mask, comp_attend])`). It never re-derives the CSA
  candidate selection.
- **Fallback.** A per-head mask (`ndim != 3`) or any `(b, s, T)` shape mismatch
  returns `None` → the eager one-shot runs. Genuine kernel errors are left to
  propagate ([[dont-rationalize-broken-as-normal]]); the *only* silent path is the
  documented unsupported-mask fallback the task specifies.
- **GPU-only.** `_decode_attn_kernel_use` returns `False` unless a Metal GPU is the
  default device, so a CPU-pinned worker test (or a no-Metal host) runs the eager
  path, **byte-identical to control**, and dispatches no Metal
  ([[worker-tests-must-pin-mlx-cpu]]).

---

## 3. Exactness

**Reassociation-level, NOT byte-identical** vs the eager f32 path (the online tile
reduction reorders the max / denom / value sums; the sink is folded into the
denominator vs the shipped `softmax(concat)+slice`). Same float class as
K25/K28's tree softmaxes.

**CPU-proven** (no Metal): the pure-MLX references
(`decode_attention_reference`, the one-shot fold-sink math; and
`decode_attention_reference_tiled`, the EXACT online-tile algorithm the kernel
runs) vs the model's eager `Attention._sparse_attend_oneshot`, over decode M=1 /
verify M=4 × T ∈ {300, 1088, 4096} × every CSA mode:

- `max|Δ| ≤ 4.3e-7` (≪ the 1e-6 bar), **argmax mismatch 0**;
- the fully-masked row (no reachable key) is **finite and exactly 0** (matches the
  reference "all-invalid → zero output"), never a NaN.

The tiled reference matching the one-shot reference to ≤1e-6 is the on-CPU proof
that the kernel's **tiling reassociation** is greedy-safe before any GPU run.

**GPU parity** (`test_decode_attn_parity_gpu`, gated `MTPLX_GPU_PARITY=1`, receipt
to `MTPLX_PARITY_RECEIPT`): the real Metal kernel vs the eager f32 reference on
random cache states at **T ∈ {1088, 4096, 16384}** for **each CSA mode** (decode
M=1 + a verify M=4 batch) → max|Δ| + argmax parity; and, only if the streaming
artifact is present, 32 real-model decode steps (recorded — the worker box loads no
artifact; the orchestrator runs the served A/B via the `decode_attn_kernel` arm).
**Pass if `max|Δ| ≤ 1e-6` and argmax mismatch 0** for every arm.

---

## 4. Commands (orchestrator, in a GPU window)

**Parity** (proves the Metal kernel matches eager before any A/B trusts it):

```sh
MTPLX_GPU_PARITY=1 \
MTPLX_PARITY_RECEIPT=docs/deepseek-v41/receipts/gpu-windows/window-XX/W60_K29_parity.json \
PYTHONPATH=$PWD .venv/bin/python3 -m pytest -q -s \
  tests/models/test_deepseek_v41_decode_attn_kernel.py::test_decode_attn_parity_gpu
```

**Decode A/B** (inside `scripts/deepseek_v41/gpu_window.sh`, flock held, qwen
unloaded, memory-guarded):

```sh
PYTHONPATH=$PWD .venv/bin/python3 scripts/deepseek_v41/ab_decode_env_levers.py \
  --arms control decode_attn_kernel --context-tokens 1024 --decode-tokens 256 \
  --out docs/deepseek-v41/receipts/gpu-windows/window-XX/W60_decode_attn_kernel.jsonl
```

Report prefill tok/s, decode tok/s, TTFT, peak GB, wall; token-id sha256 byte-
identical? (expected **NOT** byte-identical — reassociation-level). Gate: decode
**+** and argmax parity. **Only after that window is clean**, add
`decode_attn_kernel="1"` to the `stack_a` preset (currently `head_bf16 +
sinkhorn_metal + attn_compile + win_memo`).

---

## 5. Status

IMPLEMENTED + CPU-proven (kernel algorithm via the tiled + split-K pure-MLX
references, wrapper plumbing via a spy kernel, model dispatch for decode + verify
across all four modes, unsupported-mask fallback, byte-identical CPU fallback),
default OFF, GPU-only. Peak RSS < 0.2 GB in the CPU suite. Numeric-throughput A/B
pending a GPU window (**KG-m**, see KERNEL_LEDGER K29).

---

## 6. Window-26 findings + fix (parity FAIL root cause, occupancy redesign)

Window 26 (`receipts/gpu-windows/window-26/k29-parity.{json,log}`) ran the kernel
on the real GPU: `kernel_builder_ok`, every decode arm **finite, argmax mismatch
0**, but **max|Δ| 4e-4 … 2.1e-3** (all modes/T) and **1 argmax mismatch on each
verify-M4 arm** — and the `decode_attn_kernel` 1K/256 A/B decoded at **3.25 tok/s
vs stack_a 5.98 (−45%)**, tokens byte-identical to stack_a.

**(1) Parity FAIL root cause — fast-math `metal::exp`.** The error was
~1e-3 (bf16/fast-transcendental class), NOT the ~1e-6 f32-reassociation class.
An f32 numpy simulation of the exact reduction (naive sequential 512-dim dot +
online combine) matches MLX's blocked matmul softmax to **~1e-7** — so the
reduction *order* is not the leak. `mx.fast.metal_kernel` compiles with fast math,
under which the bare `metal::exp` is the *fast* approximation (~1e-3 rel err); the
rest of the codebase already uses `metal::precise::exp` / `metal::precise::rsqrt`
for exactly this reason (`fused_norm.py`, `laguna_*`). **Fix:** every `exp` is now
`metal::precise::exp` (3 sites in the fused/split kernels, 1 in combine). Expected
to drop parity back to the f32-reassociation ~1e-6 bar (the naive-dot numpy proof
bounds the residual). Not GPU-verifiable on the worker box — re-gated in-window.

**(2) −45% root cause — M=1 occupancy (latency-bound).** The v1 kernel launched
one threadgroup per `(row,head)` = **64 threadgroups per layer at decode** on a
~40-core GPU: ~1.6 TGs/core, and the whole T-reduction (barrier-heavy) is serial
per TG. ~3.5 ms/layer added / 70M MACs ≈ ~20 GFLOP/s effective → ~99% idle →
latency-bound, not throughput. **Redesign — split-K (flash-decoding):** split the
T keys across `G` threadgroups per `(row,head)` (pass 1 → per-split `(m, denom,
acc)` partials), then a cheap pass-2 combine merges the `G` partials per
`(row,head)` with the value-0 sink. `G` (`_choose_splits`) targets `_OCC_TARGET_TG`
(512) threadgroups AND ~512 keys/split, capped at 32:

| shape | v1 TGs/layer | split-K TGs/layer (pass 1) | G |
|---|---:|---:|---:|
| decode M=1, T=1088 | 64 | **512** | 8 |
| decode M=1, T=4096 | 64 | **512** | 8 |
| decode M=1, T=16384 | 64 | **2048** | 32 |
| verify M=4, T=4096 | 256 | **2048** | 8 |

**Paper pricing:** v1 was latency-bound at ~64 TGs; split-K's ≥512 TGs let the
scheduler hide launch+barrier latency, and each pass-1 TG does only `T/G` keys
(≈2 tiles at T=1088). Pass 2 is `rows·H` TGs of a tiny `G`-way combine. Two
dispatches replace one, but both are occupancy-filled — expected to recover most
of the −45% and, since the eager SDPA it replaces is ~22 primitives, ideally beat
eager. `T ≤ 256` keeps the single-dispatch path (splitting a handful of keys is
not worth the combine). The realized decode delta is the next window's A/B.

**(3) Engagement counter.** `deepseek_v41_attn_kernels.{reset_engagement,
engagement,note_fallback}` count real dispatches (`calls`/`rows`/`split_calls`)
and armed-but-eager fallbacks; `ab_decode_env_levers.py` resets them after model
load and records `decode_attn_kernel_engagement` per arm. **`calls == 0` on a
`decode_attn_kernel` arm ⇒ the kernel never ran (all eager)**; `calls > 0` ⇒ it
ran (so a slowdown is the kernel's cost, not a no-op) — settling window-26's
"did-not-run vs slow" ambiguity next window.

**(4) Loader import fix.** The parity test's optional real-model block imported the
non-existent `deepseek_v41_loader.load`; it now imports `load_deepseek_v41_streaming`
(the served comparison + engagement counter is the `decode_attn_kernel` A/B arm,
which does the proper streamed load — a second heavy load is not reproduced in
pytest).

**Re-gate (KG-m):** parity `test_decode_attn_parity_gpu` now exercises the auto
(split-K) path for every mode × T **and** explicit `single_g1` / `split_g8` arms at
T=4096, each recording its `n_splits` + engagement. Pass = max|Δ| ≤ 1e-6 + argmax
mismatch 0 on every arm; then the `decode_attn_kernel` A/B with `calls > 0` and
decode +.

---

## 7. Window-27 re-gate → SHELVED verdict

Window 27 (integration `b582b73a3`, with the precise::exp + split-K fix
`597d6122d` merged) re-ran both gates:

**Engagement (the counter did its job).** The `decode_attn_kernel` arm ran the
kernel on **all 10,280 decode calls, 0 fallbacks, all via the split-K path** — so
the −45%/−38% is the kernel running (and being slow), NOT the kernel silently
falling back. Window-26's "did-not-run vs slow" ambiguity is resolved: it **ran,
and it is slow at M=1**.

**Throughput.** `decode_attn_kernel` decoded **3.71 tok/s vs stack_a 6.01 (−38%)**
— the split-K occupancy redesign (≥512 threadgroups/layer vs the v1 64) narrowed
the v1 −45% only to −38%. The kernel is still **slower than the eager SDPA at M=1**.
Even with good occupancy, a per-layer two-dispatch kernel doing the full
`64 heads × 512 × T` attention loses to MLX's vendor-lowered einsum SDPA at
batch-1: the eager path's ~22 primitives are cheap, well-tuned matmul/softmax
kernels, and the QK^T/PV are already tile-aligned GEMMs (W56 F3/F4). The dispatch
count K29 saves (~22 → ~3–5) does not pay when the GPU is not actually
dispatch-starved at these small kernels — the M=1 attention math is a rounding
error of the ~160 ms/token, most of which is elsewhere (MoE switch, barriers).

**Parity — the 1e-3 is bf16-class, and `precise::exp` proved it is not exp.** The
window-27 parity receipt shows **the same `max_abs_d` to 8 digits as window-26**
(e.g. `decode_full_T1088` = 0.00107455 both), so switching `metal::exp` →
`metal::precise::exp` changed the result by **zero** — on Apple silicon the two
coincide for f32, and exp precision was never the source. Ruling out the two
in-kernel hypotheses:

  * **NOT fast-math exp** — precise::exp is byte-identical (above).
  * **NOT the online/split reduction order** — the pure-MLX f32 references
    (`decode_attention_reference` one-shot, `_tiled`, and `_splitk` at every G)
    match the eager `_sparse_attend_oneshot` to **≤8.6e-7, argmax-exact** on CPU
    (strict-f32), incl. the K30 gathered shapes.

**Best explanation of the ~1e-3 (bf16-precision-class).** A CPU sensitivity check
on the exact parity input (`decode_full_T1088`, seed 101) reproduces the observed
magnitude by rounding a single operand to bf16 and back:

| what is rounded to bf16 | max\|Δ\| vs strict-f32 ref |
|---|---:|
| K + V latent (both) | 0.001912 |
| V only (PV inputs) | 0.001392 |
| scores / QK inputs only | 0.001250 |
| **observed kernel-vs-ref (window-26/27)** | **0.00107455** |

The observed 0.00107 sits squarely inside the bf16-rounding band — i.e. the
divergence is **bf16-precision-class**, an order of magnitude structure, not the
~1e-6 f32-reassociation class the parity bar demands. Its source is the
reduced-precision f32 execution path on the Apple GPU: the parity reference's
`mx.einsum` lowers to the vendor simdgroup-matrix GEMM (bf16/tf32-class multiply
accumulation), while the hand kernel accumulates its 512-dim dot and streaming
softmax differently; on the **real model** the KV latent is stored **bf16**
([[deepseek-v4-mtplx-port]] "bf16-acts"), so the eager SDPA the kernel must match
is itself bf16-class. A hand kernel cannot hit ≤1e-6 against a reference that is
not itself strict-f32 without replicating the eager path's exact GPU op sequence —
which would defeat the whole point of the fusion.

This is precisely the **V4 "hand MLA fused-attention kernel" verdict**
([[deepseek-v4-kernel-verdicts]], KERNEL_LEDGER §6 dead-here): a fused MLA
attention kernel's precision diverges from the eager path enough to tip near-ties
(V4: accept 2.72→2.64; here: 1 argmax mismatch on every verify-M4 arm), and
attention is not the binding term at latent 512. W60 tested whether the
**streaming/dispatch-bound** reframe changed that verdict — it did not: the M=1
dispatch win the reframe predicted did not materialise (kernel engaged, still
−38%), and the precision divergence is unchanged.

**Disposition.** SHELVED. `MTPLX_DSV41_DECODE_ATTN_KERNEL` stays **default OFF**;
the `decode_attn_kernel` arm is standalone and **in no stack** (`stack_a`,
`stack_b` do not set it). The code, CPU tests, and pure-MLX references remain as a
documented negative result (and the split-K + engagement-counter machinery is
reusable). No further kernel work. The DSV4.1 decode lever stays with the shipped
eager SDPA + K22/K24 dispatch cuts.
