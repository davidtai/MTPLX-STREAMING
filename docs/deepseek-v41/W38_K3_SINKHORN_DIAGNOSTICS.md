# W38 / K3 — Sinkhorn kernel: root-cause, diagnostics, engagement, merge fix

**Branch:** `feat/deepseek-v41-w38` (off integration `bfd361424`)
**Follows:** W32 (kernel port) + the window-12 parity failure and the
byte-identical `sinkhorn_metal` decode arm (4.12 vs 4.06 tok/s).

## TL;DR

- **Root cause of the window-12 parity failure found:** the V4 Sinkhorn kernel
  source **fails to build for a bf16 buffer** (`assigning to 'bfloat16_t' from
  incompatible type 'float'` at `out[off+i] = c[i]`). The W32 test's bf16 arm
  called `_sinkhorn_kernel_apply(comb_bf16)` directly, so it raised a Metal
  `Unable to build metal library` RuntimeError — the failure the window truncated.
  **fp32 is bit-identical** (`max|d| = 8.9e-08 ≤ 1e-6`); fp32 was never the issue.
- **Fix:** `_sinkhorn_normalise` upcasts a non-fp32 `comb` to fp32 for the
  (fp32-validated) kernel and casts the result back, so the Metal path never sees
  a bf16 buffer. Production `comb` is fp32 (`_mixes` casts it), so this is a no-op
  on the served path.
- **Also fixed a live merge regression (6 RED tests on `bfd361424`):** W32 renamed
  the split boundary to `_hc_split_sinkhorn`, but W33's **compiled** HC path
  (`_hc_mixes_split`) and its test call `hc_split_sinkhorn` — so on the integration
  commit the compiled K4 decode path raised `NameError` **and bypassed the K3
  kernel entirely**. Renamed back to the canonical `hc_split_sinkhorn` so **both**
  the eager and compiled paths route through the kernel dispatch.
- **Engagement instrumentation** so a window can tell the kernel ran vs silently
  fell back: always-on counters + route-stage probe stages, surfaced per arm in
  the A/B env-lever receipt.
- **Self-diagnosing parity test** writing a JSON receipt to `MTPLX_PARITY_RECEIPT`.

## 1. Root cause (from the self-diagnosing receipt)

The new receipt (one arm shown) pins it:

```json
"fp32": { "gate": "kernel_vs_recurrence_fp32", "tol": 1e-06,
          "max_abs_d": 8.94e-08, "argmax_mismatch": 0, "passed": true },
"bf16": { "info_raw_bf16_kernel_ok": false,
          "info_raw_bf16_kernel_error":
            "[metal::Device] Unable to build metal library from source ...
             error: assigning to 'bfloat16_t' (aka 'bfloat') from incompatible
             type 'float'  out[off + i] = c[i];" }
```

`deepseek_v4._sinkhorn_metal_kernel` stores `out[off+i] = c[i]` with `c` a
`float[]` register array and `out` the I/O-dtyped buffer. Metal does **not**
implicitly convert `float → bfloat16_t` on assignment, so the library fails to
compile whenever the output buffer is bf16. V4 never hit this: its comb is always
fp32 and its GPU lane was hard-gated to the fp32 `(4,20,1e-6)` geometry. W32
generalised the path and the W32 parity test fed bf16 straight into the raw
kernel → build error. (A native bf16 kernel would need an explicit `T(c[i])`
store cast in `deepseek_v4.py`; out of scope and unnecessary — see the fix.)

## 2. Fix — the kernel only ever computes fp32

`_sinkhorn_normalise` (deepseek_v41.py):

```python
if _sinkhorn_use_kernel():
    _SINKHORN_KERNEL_CALLS += 1
    _route_probe.count("hc.sinkhorn_kernel")
    if comb.dtype != mx.float32:
        out = _sinkhorn_kernel_apply(comb.astype(mx.float32), hc, iters, eps)
        return out.astype(comb.dtype)
    return _sinkhorn_kernel_apply(comb, hc, iters, eps)
```

The kernel already computes the whole schedule in fp32 registers regardless of
I/O dtype; the only lossy step was the bf16 store, which also happened to be the
step Metal refuses to compile. Upcasting removes both the build failure and the
bf16-storage question in one move: the bf16 result is now
`round_bf16(fp32_sinkhorn(comb))`, the fp32 kernel's exact output rounded once.
Production (`comb` fp32) takes the unchanged branch — byte-identical to W32.

## 3. Merge regression fix (K3 × K4) — 6 RED tests → green

W32 renamed the module's Sinkhorn split boundary `hc_split_sinkhorn` →
`_hc_split_sinkhorn` (to avoid shadowing the then-imported V4 name). W33 landed the
HC-compile tapes (K4), whose pure-array `_hc_mixes_split` and its census test call
`hc_split_sinkhorn` (no underscore — the module's documented boundary at
deepseek_v41.py L809/L857/L864). The W32↔W33 merge left them inconsistent:

- `tests/models/test_deepseek_v41_hc_compile.py` — **6 tests RED** on `bfd361424`
  (`AttributeError: module ... has no attribute 'hc_split_sinkhorn'`).
- Any run with `MTPLX_DSV41_HC_COMPILE=1` (the K4 lever) hit `NameError` in the
  compiled decode path, **and** that path never reached the K3 kernel.

Fix: renamed the dispatch back to the canonical **`hc_split_sinkhorn`**; both the
eager `DecoderLayer._mixes` and the compiled `_hc_mixes_split` now call it, so the
K3 Metal kernel drops into the compiled tape as W33's design documents. The K4
suite is green (8/8) and K3 now composes with K4.

Consequence for the observed decode arm: `MTPLX_DSV41_HC_COMPILE` defaults **OFF**,
so window-12's `sinkhorn_metal` arm ran the **eager** path, where the kernel *does*
engage — the byte-identical output is the kernel being bit-exact (fp32 `max|d|`
8.9e-8, argmax-preserving), not a fallback. The new counters make this explicit
per arm rather than inferred.

## 4. Engagement instrumentation (did the kernel actually run?)

- `deepseek_v41._SINKHORN_KERNEL_CALLS` / `_SINKHORN_RECURRENCE_CALLS` — always-on,
  near-free process-cumulative counters, incremented in `_sinkhorn_normalise`;
  `_reset_sinkhorn_kernel_calls()` / `_sinkhorn_kernel_calls()` helpers.
- Route-stage probe stages `hc.sinkhorn_kernel` / `hc.sinkhorn_recurrence` emitted
  via `expert_route_probe.count(...)` (near-zero when the probe is disabled).
- `scripts/deepseek_v41/ab_decode_env_levers.py`:
  - `_run_arm` resets the counters after model load and attaches
    `receipt["sinkhorn_engagement"] = {kernel_calls, recurrence_calls, engaged,
    sinkhorn_metal_env, hc_compile_env}` — cumulative over the whole arm.
    `engaged = kernel_calls > 0 and recurrence_calls == 0`.
  - `_sync_census` adds `sinkhorn_kernel_calls_decode` /
    `sinkhorn_recurrence_calls_decode` (probe deltas around the census decode).

**Why cumulative-per-arm, not a warm-window delta:** under `MTPLX_DSV41_HC_COMPILE`
the Python `_sinkhorn_normalise` wrapper runs only during the (cold) trace, so a
warm decode window increments nothing even though the kernel runs inside the tape.
The arm-cumulative counter (which always sees the cold trace, and prefill runs
eager) reliably answers engaged-vs-fallback; the census delta is a secondary,
eager-only read.

## 5. Self-diagnosing parity test

`tests/models/test_deepseek_v41_sinkhorn_metal.py::test_sinkhorn_kernel_parity_gpu`
(skipped unless `MTPLX_GPU_PARITY=1`): builds a full `diag` — per-arm dtype,
tolerance, `max_abs_d`, argmax mismatch count, finiteness, kernel-build ok/error,
`mx.metal.is_available()`, and the engagement counters — **writes it to
`MTPLX_PARITY_RECEIPT` (JSON) and prints it BEFORE any assertion**, so a window
captures the detail even when stdout is truncated to the tail. Each measurement is
wrapped in try/except and records its own error, so the receipt is always complete.

Arms:
- **fp32** — `_sinkhorn_kernel_apply(comb)` vs `_sinkhorn_ops(comb)`, gate
  `max|d| ≤ 1e-6` + argmax exact (bit-identical; the production dtype).
- **bf16** — the production **dispatch** path (`_sinkhorn_normalise`, armed on GPU,
  upcasts) vs `_sinkhorn_ops(comb.astype(f32)).astype(bf16)` (fp32 sinkhorn rounded
  to bf16), gate `max|d| ≤ 1.6e-2` + argmax exact. **Not** vs the bf16 recurrence
  (which accumulates in bf16 and legitimately diverges — recorded as
  `info_bf16recurrence_vs_fp32_max_abs_d ≈ 9.5e-3` to document why the naive W32
  comparison failed). Records the raw-bf16 kernel build failure as
  `info_raw_bf16_kernel_ok/error`.
- **dispatch** — `_sinkhorn_normalise` armed on GPU engages the kernel
  (`kernel_calls == 1`, `recurrence_calls == 0`) and matches the recurrence in fp32.

### Command (run inside the exclusive GPU flock window)

```
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w38
MTPLX_GPU_PARITY=1 MTPLX_PARITY_RECEIPT=docs/deepseek-v41/receipts/w38_parity.json \
  PYTHONPATH=$PWD /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 \
  -m pytest tests/models/test_deepseek_v41_sinkhorn_metal.py::test_sinkhorn_kernel_parity_gpu -v -s
```

The JSON receipt lands at `MTPLX_PARITY_RECEIPT` regardless of pass/fail.

### A/B engagement check (inside a window)

```
MTPLX_ROUTE_STAGE_PROBE=1 PYTHONPATH=$PWD .venv/bin/python3 \
  scripts/deepseek_v41/ab_decode_env_levers.py --arms control sinkhorn_metal \
  --context-tokens 1024 --decode-tokens 256 --syncs 32 --out <receipt.jsonl>
```

Each arm's `sinkhorn_engagement.engaged` says whether the kernel ran; the
`sinkhorn_metal` arm should show `kernel_calls > 0, recurrence_calls == 0`, the
`control` arm the inverse.

## 6. Tests (CPU, `nice -n 19`, no `-n auto`)

- `test_deepseek_v41_sinkhorn_metal.py`: **14 passed, 1 skipped** (parity skipped
  on CPU). Adds engagement-counter gates (kernel/recurrence branch increments +
  probe stages), the bf16 upcast gate, and receipt-helper gates. No Metal is
  dispatched — every kernel-branch test monkeypatches `_sinkhorn_kernel_apply` to a
  spy.
- `test_deepseek_v41_hc_compile.py`: **8 passed** (was 6 RED on `bfd361424`).
- Broad V4.1 sweep (20 files incl. parity, chunked/layer-major prefill, hc_compile,
  ab_env_levers): **131 passed, 22 skipped, 0 failed**.

## 7. Disclosure

While building the self-diagnosing receipt I learned that `mx.fast.metal_kernel`
dispatches on the **GPU even when the default device is CPU** (it is GPU-only), so
one scratch invocation of `_sinkhorn_kernel_apply` executed a single tiny build +
56-matrix fp32 dispatch on the GPU. That one run produced the root-cause receipt
in §1. No further kernel was run; the committed CPU test suite never dispatches the
real kernel (all kernel-branch tests use a spy), and the parity test stays gated
behind `MTPLX_GPU_PARITY=1`.

## 8. Commit

`feat/deepseek-v41-w38` @ `cb1cd11 (pre-embed content commit)`. No Co-Authored-By / Claude trailers;
`scripts/check_ai_attribution.py --range bfd361424..HEAD` clean.
