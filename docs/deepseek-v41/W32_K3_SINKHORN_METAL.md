# W32 / K3 — Sinkhorn Metal kernel port to DSV4.1

**Branch:** `feat/deepseek-v41-w32` (off `feat/deepseek-v41-streaming`)
**Kernel ledger:** [KERNEL_LEDGER.md](KERNEL_LEDGER.md) §K3 (Rank 2)
**Status:** ported, CPU dispatch gates green, GPU numeric parity gated for a window (not yet measured on V4.1).

## 1. What shipped

DSV4.1's backbone runs the Sinkhorn alternating-normalisation loop **twice per
layer per token** — the attention Hyper-Connection mix and the ffn HC mix, via
`DecoderLayer._mixes → _hc_split_sinkhorn` (deepseek_v41.py). Until this change
that went through `deepseek_v4.hc_split_sinkhorn`, the "intentionally always
stock" reference transcription, i.e. the 20-iteration 4×4 recurrence
(`_sinkhorn_ops`) with **no fast path at all**.

This ports DeepSeek-V4's one-dispatch Sinkhorn Metal kernel to V4.1 behind
`MTPLX_DSV41_SINKHORN_METAL` (default **OFF**):

- `mtplx/models/deepseek_v41.py`
  - imports **reuse** V4's kernel source rather than copying it: `_sinkhorn_ops`
    (stock recurrence / CPU + flag-off route + parity oracle) and
    `_sinkhorn_kernel_apply` (the whole loop as one `mx.fast.metal_kernel`
    dispatch, built by `deepseek_v4._sinkhorn_metal_kernel`). The V4 and V4.1
    comb tensors are identically shaped `[..., hc, hc]`, so the kernel builder is
    shared with no fork.
  - `_sinkhorn_metal_enabled()` — reads `MTPLX_DSV41_SINKHORN_METAL` **at use**,
    never frozen at import (the serving harness stamps optimization keys after
    the module is imported — [[env-flags-read-at-use-not-import]]).
  - `_sinkhorn_use_kernel()` — the kernel is live **only** when the flag is truthy
    **and** `mx.metal.is_available()` **and** `mx.default_device() == mx.gpu`. CPU
    (every worker test pins `mx.set_default_device(mx.cpu)`) and any no-Metal
    build always take the recurrence, so the flag can never move CPU numerics.
  - `_sinkhorn_normalise(comb, hc, iters, eps)` — the dispatch point: kernel when
    `_sinkhorn_use_kernel()`, else `_sinkhorn_ops`. Both accept any leading dims
    (`rows = b*s` = any n): the kernel flattens the leading axes to one matrix
    index and reshapes back, so decode, one-shot prefill, **chunked prefill**
    (W20) and the **layer-major** path (W30/K16) all compose unchanged.
  - `_hc_split_sinkhorn(...)` — the V4.1 pre/post/comb split. The pre/post/comb
    split is byte-for-byte the split of `deepseek_v4.hc_split_sinkhorn` (origin
    named in a comment; `inference/kernel.py` `hc_split_sinkhorn_kernel`
    L371-427); only the always-stock `_sinkhorn_ops` tail is swapped for
    `_sinkhorn_normalise`. `_mixes` now calls this instead of the imported V4
    reference.
- `tests/models/test_deepseek_v41_sinkhorn_metal.py` — the CPU dispatch gates and
  the GPU numeric parity test (below).

**No `deepseek_v4.py` was modified** — the V4 kernel source is imported and reused
as-is, so V4's own suites are untouched.

## 2. Math diff — V4 vs V4.1 Sinkhorn

The arithmetic is **identical**; the differences are structural (call site,
geometry guard). The kernel is bit-identical fp32 in the same op order as the
recurrence it replaces.

| Aspect | V4 (`deepseek_v4`) | V4.1 (`deepseek_v41`) | Same? |
|---|---|---|---|
| Recurrence | softmax(axis=-1)+eps → col-norm(axis=-2) → 19×(row-norm axis=-1, col-norm axis=-2) | identical (`_sinkhorn_ops` reused verbatim) | ✅ |
| `hc_mult` / `iters` / `eps` | 4 / 20 / 1e-6 | 4 / 20 / 1e-6 | ✅ |
| eps placement | `softmax + eps`, then `/(sum + eps)` per normalise | identical | ✅ |
| exp / max-stable softmax | row-softmax over last axis `k`, max-subtracted | identical | ✅ |
| comb dtype | fp32 (`_mixes` casts to fp32) | fp32 (`_mixes` casts to fp32) | ✅ |
| comb shape | `[..., hc, hc]`; leading dims = decode/verify/prefill rows | `[b, s, hc, hc]`; leading dims = `[b, s]` | ✅ (both flatten to `nmat` in the kernel; rows = any n) |
| Kernel source | `_sinkhorn_metal_kernel(hc, iters, eps)` | **same builder, imported** (not copied) | ✅ |
| Hot-path selector | model installs a route once at construction (`_install_sinkhorn_normaliser`), `_hc_pre_impl` | per-call `_sinkhorn_normalise` inside `_hc_split_sinkhorn` (V4.1's `_mixes` calls the split directly) | ⟂ structural |
| GPU geometry guard | hard-restricted to `(4,20,1e-6)`; raises otherwise | **no restriction** — reuses the general `(hc,iters,eps)` builder, so test configs (`iters=2`) and the real (4,20,1e-6) both run | ⟂ structural |
| Flag | `MTPLX_DSV4_SINKHORN_KERNEL` | `MTPLX_DSV41_SINKHORN_METAL` (default OFF) | ⟂ name |

Numeric conclusion: **no math difference.** Same ops, order, eps, dtype, and a
shape-agnostic kernel. V4 measured this exact kernel at **AR +29.3 %** in the same
dispatch-bound regime ([[deepseek-v4-kernel-verdicts]]); V4.1's win is a GPU-window
measurement (below), not yet taken.

## 3. Dispatch count before/after (per token)

Per `_sinkhorn_ops` call at `iters=20` (the recurrence, as MLX graph primitives):

| Op | Count / call |
|---|---|
| `softmax` (axis=-1) | 1 |
| `add` (eps: 1 after softmax + 1 per normalise) | 40 |
| `reduce_sum` (1 first col-norm + 19×2 row/col) | 39 |
| `divide` (1 per normalise) | 39 |
| **Total graph primitives / call** | **119** |

The K3 ledger quotes "~80 dispatches each" — that is the `reduce_sum + divide`
core (39 + 39 = **78**), the fusable/encoded kernels; the full primitive count
including the eps-`add`s and `softmax` is 119.

**Calls per token:** `_mixes` is invoked twice per layer (attention HC + ffn HC),
so `2 × 40 layers = 80` Sinkhorn calls / token.

| | Recurrence (flag off) | Metal kernel (flag on, GPU) |
|---|---|---|
| dispatches / **call** | 119 primitives (78 = reduce_sum+divide core) | **1** |
| calls / token | 80 | 80 |
| dispatches / **token** (Sinkhorn block) | **9,520** (6,240 reduce+divide core) | **80** |

Reduction: **9,520 → 80 per token** (≈119×; ≈6,240 → 80 counting only the
reduce+divide core). The reshape in/out of `_sinkhorn_kernel_apply` is a
contiguous view (no dispatch), so each call is exactly one Metal launch. This is
the top dispatch source in the V4.1 decode step per the ledger (§K3, and the
"Dispatch count/token" note), and matches V4's whole-model `6,794 → 86` framing
scaled to V4.1's 80 pre-calls.

## 4. GPU parity test — how to run it in a window

The numeric kernel-vs-recurrence parity needs a Metal GPU and is **skipped unless
`MTPLX_GPU_PARITY=1`**. It compares one `_sinkhorn_kernel_apply` dispatch against
the 40-pass `_sinkhorn_ops` recurrence on random `[2,7,4,4]` inputs in **fp32**
(bit-identical, `max|d| ≤ 1e-6`, argmax exact) and **bf16** (argmax exact is the
hard gate; the value spread is recorded — the kernel computes in fp32 registers,
the recurrence in bf16). It also checks the armed dispatcher
(`_sinkhorn_normalise`, flag on + GPU) matches the recurrence in fp32.

Run inside the exclusive GPU flock window (it sets `mx.set_default_device(mx.gpu)`
itself and restores the device afterward), from the worktree root:

```
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w32
MTPLX_GPU_PARITY=1 PYTHONPATH=$PWD \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 \
  -m pytest tests/models/test_deepseek_v41_sinkhorn_metal.py::test_sinkhorn_kernel_parity_gpu -v -s
```

`-s` surfaces the recorded line `[W32/K3 parity] fp32 max|d|=… bf16 max|d|=…`.

## 5. Tests

- CPU dispatch gates (green, no Metal dispatched — the GPU-routing case is proven
  by monkeypatching the kernel callable to a spy):
  - flag parses (default OFF; read at use, not frozen at import)
  - `_hc_split_sinkhorn` flag-off is **bit-identical** to the stock V4 split
    (decode numerics unchanged)
  - the split routes the reshaped `[...,hc,hc]` comb through `_sinkhorn_normalise`
  - flag **off** → recurrence
  - flag **on + CPU** device → recurrence (flag inert off-GPU), bit-identical to
    `_sinkhorn_ops`
  - flag **on + GPU** device → kernel callable (spy asserted, no Metal run)
  - GPU default but flag **off** → recurrence (device alone never arms it)
- GPU parity: present, **skipped on CPU** (`MTPLX_GPU_PARITY != 1`).

Verification (CPU, `nice -n 19`, no `-n auto`):

```
tests/models/test_deepseek_v41_sinkhorn_metal.py: 8 passed, 1 skipped
core+extended V4.1 suites + this file:            115 passed, 23 skipped, 0 failed
```

## 6. Commit

`feat/deepseek-v41-w32` @ `5a3bd15 (pre-embed content commit)` (content commit; final HEAD reported in
the worker's return message). No Co-Authored-By / Claude trailers;
`scripts/check_ai_attribution.py --range feat/deepseek-v41-streaming..HEAD` clean.
