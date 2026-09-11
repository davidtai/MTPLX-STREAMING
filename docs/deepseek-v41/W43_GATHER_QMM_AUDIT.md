# W43 — DSV4.1-Flash all-hit `gather_qmm` audit + microbench

Status: audit complete; GPU microbench written and CPU-verified (`--help` /
`--dry-run` mlx-free, shape-agreement test green), A/B **pending** a GPU window
(David runs it). Author: Opus 4.8 worker (`feat/deepseek-v41-w43`, from
`feat/deepseek-v41-streaming` @ `5b6b8e7d2`). CPU-only; MLX pinned to CPU; no GPU
lock taken, no `experts.bin` load. mlx 0.32.2.

## 0. Headline

The DSV4.1-Flash streamed-switch **all-hit gather is already convention-correct
and byte-minimal.** Per one routed-`moe.routed_switch` layer at M tokens it
issues **exactly three `gather_qmm` calls** (gate, up, down — one each, no
per-expert Python loop, no bank-slice pre-gather), each shaped as the canonical
`[rows,1,K]` M=1-GEMV convention (`selected = [rows,1,1,K]`, `rhs_indices =
[rows,1]`, `lhs_indices=None`, `transpose=True`), where `rows = M·top_k`. It
reads exactly the `top_k=6` routed experts' weights **once each** and does
exactly `top_k` GEMVs.

**No convention violation was found** (the "gather-qmm calling-convention trap"
bad form — `x=[rows,K]` with the M-axis dropped, 8× work — is never used). This
confirms W42's arithmetic: the gather moves `6 × 18,800,640 B = 112.8 MB`
(≈ **0.188 ms @ 600 GB/s / 0.184 ms @ 614 GB/s**), two orders below the measured
`moe.routed_switch` **2.71 ms/layer** (window-15 `stage-timing-head-bf16.json`,
count 2560) — so the stage cost is **not** the gather's FLOPs or bytes.

What is **not yet measured, and is the reason for the microbench**: whether the
native **mxfp4 gs32 M=1 gather variant actually achieves bandwidth** or is
ALU-bound (memory `metal-sub4bit-alu-bound`; the mxfp4 gather has 144 template
variants and the M=1 one may be far off the 0.19 ms floor). The microbench times
the switch's exact call against affine q4/q8 and a bf16-dense bandwidth
reference, at M=1 (AR) and M=4 (MTP verify), in the window.

## 1. Call graph (all-hit path)

`HotExpertSwitchGLU._run` all-hit branch → `_dispatch_component_bank` →
(codec `mxfp4`|`affine`) `_run_component_bank_q4` → `_gather_component_bank`.
All line numbers `mtplx/models/expert_mlx.py`.

- **`_run`** builds the wave input. Full-wave all-hit (the common decode case,
  `wave.positions == range(len(expert_ids))`), L2432–2435:
  `assignment_inputs = mx.broadcast_to(tokens[:,None,:], (M, top_k, hidden)).reshape((-1, hidden))`
  → shape **`[M·top_k, hidden]`** (one row per (token, expert) assignment). The
  non-full sub-case (L2438–2447) instead does `mx.take(tokens, token_positions)`
  → same `[rows, hidden]` shape. Dispatch at L2457.
- **`_run_component_bank_q4`** (L1474) builds `slot_indices = mx.array([…]).reshape((-1,1))`
  → **`[rows, 1]`** int32 (L1494–1497), one bank slot id per row, and requires
  `x.shape[0] == len(bindings)` (L1485) so `rows` == the assignment count.
- **`_gather_component_bank`** (L1509): `selected = x.reshape((rows,1,1,K))`
  (L1527), then three `mx.gather_qmm` calls (L1531 mxfp4 / L1543 affine),
  `gate = qmm(selected,"gate_proj")`, `up = qmm(selected,"up_proj")`,
  `output = qmm(_clamped_swiglu(gate, up, swiglu_limit),"down_proj")`
  (L1555–1557) — **one call per projection, ClampedSwiGLU applied once** (L1378
  `_clamped_swiglu`: `clip(up,±10)`, `minimum(gate,10)`, then `swiglu`).

`selected = x.reshape((rows,1,1,K))`: `x` is contiguous `[rows, K]`, so this is a
free view (no copy). `assignment_inputs` is the only materialisation on the path
(broadcast+reshape → `top_k` copies of each token: **≈ 61 KB at M=1**, ≈ 246 KB
at M=4, in bf16) — negligible vs the 112.8 MB of weights.

## 2. Per-call op table — one all-hit routed layer, `rows = M·top_k`

Geometry (canonical, `mtplx/deepseek_v41_convert.py`): `hidden=5120`,
`moe_intermediate=2304`, `top_k=6`, mxfp4 `group_size=32`, `bits=4`,
`swiglu_limit=10.0`; bank stacks `S` resident slots (≈92/layer at 82 GiB).
`transpose=True`, `lhs_indices=None` on all three. Bytes = weight(+scale) read.

| call | mode | `x`→gather (`selected`) | `w` (bank) | `scales` | `biases` | `rhs_indices` | GEMVs | FLOPs (÷rows) | bytes/expert |
|---|---|---|---|---|---|---|---|---|---|
| gate_proj | mxfp4 gs32 b4 | `[rows,1,1,5120]` | `[S,2304,640]` u32 | `[S,2304,160]` u8 | — | `[rows,1]` | rows × `[1,5120]·[5120,2304]` | `2·5120·2304` = 23.59 M | 5,898,240 + 368,640 = 6,266,880 |
| up_proj | mxfp4 gs32 b4 | `[rows,1,1,5120]` | `[S,2304,640]` u32 | `[S,2304,160]` u8 | — | `[rows,1]` | rows × `[1,5120]·[5120,2304]` | 23.59 M | 6,266,880 |
| down_proj | mxfp4 gs32 b4 | `[rows,1,1,2304]` | `[S,5120,288]` u32 | `[S,5120,72]` u8 | — | `[rows,1]` | rows × `[1,2304]·[2304,5120]` | 23.59 M | 6,266,880 |

(`affine` codec is identical in shape/rows but passes a **4th positional
`biases`** array of the scales' shape/dtype — bf16 — and `mode="affine"`; native
mxfp4 passes **no bias leaf**, scales are 1-byte E8M0.)

**Totals** (3 projections; `per-expert record = 18,800,640 B`, reproduced
bit-exact by the plan math and `tests/test_deepseek_v41_mxfp4_bank.py`):

| M | rows/GEMVs | weights+scales read | FLOPs | ideal @600 GB/s | ideal @614 GB/s |
|---:|---:|---:|---:|---:|---:|
| 1 (AR) | 6 | 112,803,840 B (112.8 MB) | 0.425 GFLOP | 0.188 ms | 0.184 ms |
| 4 (verify) | 24 | 451,215,360 B (451.2 MB) | 1.699 GFLOP | 0.752 ms | 0.735 ms |

## 3. Verdict against each audit question

| question | finding |
|---|---|
| exact `x` shape | `selected = [rows,1,1,K]` (K=5120 gate/up, 2304 down); `rows = M·top_k` |
| `indices` shape | `rhs_indices = [rows,1]`; `lhs_indices = None` (x batch consumed in order, 1:1 with rhs) |
| weight/scales/biases shapes+dtypes | see table; mxfp4 = u32 weight + u8 E8M0 scales, **no bias**; affine = u32 weight + bf16 scales + bf16 biases |
| mode / group_size / bits / transpose | `mxfp4` / 32 / 4 / `True` (the DSV4.1 lane); affine variants 64 / {4,8} / `True` |
| ClampedSwiGLU + down: one call or looped? | **one `gather_qmm` per projection** (3 total); `_clamped_swiglu` applied once between up and down; **no per-expert loop** |
| FLOPs vs minimum | **at the minimum** — exactly `top_k` (M=1) / `M·top_k` GEMVs; no broadcast waste |
| bytes vs minimum | **at the minimum** — reads the `top_k` routed experts' weights **once each**; activations negligible (<0.06%) |
| violates `[rows,1,K]` convention? | **NO.** The GEMV M-axis (`selected[-2]`) is `1` on all three calls; the trap's `[rows,K]`/8×-work form is never used. |
| re-materialises / copies bank slices per call? | **NO.** `gather_qmm` gathers the `S`-slot stacked bank internally via `rhs_indices`; no temp gather. Only copy on the path is `assignment_inputs` (≈61 KB at M=1). |

**Because the audit finds no violation, the microbench's `mxfp4_convention` arm
is an *equivalence* check, not a fix:** it re-expresses the same work in the
canonical `[M,1,K]` + `[M,top_k]`-index form (token not duplicated, `top_k`
folded into the `rhs_indices` second dim via `lhs_indices`) to test whether that
picks a different / faster mxfp4 template variant than the switch's flat
`[M·top_k,1,1,K]` + `[M·top_k,1]` form. Same FLOPs, same bytes, same expected
value.

## 4. Microbench — `scripts/deepseek_v41/gather_qmm_microbench.py`

Times one full routed-expert MLP (gate + up + ClampedSwiGLU(10.0) + down) per
arm, at M∈{1,4}, `top_k=6`, against the real geometry. `--help`/`--dry-run`
import **no mlx** and touch no GPU (pure integer geometry + the window command);
the timing path imports `mlx.core` lazily and **must** run inside the exclusive
window. N=50 iterations, 10 warmup, `mx.eval` fences, median µs/call + effective
GB/s (gathered weight+scale bytes ÷ median), per-arm `get_peak_memory`, JSON
receipt. Synthetic random bank sized to a `--memory-limit-gib` budget (default 8;
slots shrink to fit and the receipt flags any index wrap). Never overwrites an
existing `--out` (memory `never-overwrite-a-measurement`).

Arms: `mxfp4_switch` (the switch's **exact** call, via the real
`_gather_component_bank`), `mxfp4_convention` (equivalence form, §3),
`affine_q4_gs64`, `affine_q8_gs64`, `bf16_dense` (`gather_mm` over dense slices —
the bandwidth reference, no dequant ALU, reads 4× the mxfp4 bytes). The read the
window gives: **`mxfp4_switch` GB/s ÷ `bf16_dense` GB/s** at M=1 says whether the
mxfp4 gs32 M=1 gather is bandwidth-bound (≈1.0, gather is not the lever) or
ALU-bound (≪1.0, and a faster M=1 mxfp4 kernel — or batching verify M — is the
lever); `mxfp4_switch` vs `mxfp4_convention` says whether the flat shape is
leaving a variant on the table.

CPU test: `tests/test_deepseek_v41_gather_qmm_microbench.py` (16 tests, green)
drives the real `_gather_component_bank` with `mx.gather_qmm` monkeypatched to a
shape-capturing stub and asserts the production x/`rhs_indices` shapes equal the
microbench's plan (mxfp4 + affine q4/q8, M=1 and M=4), that the plan bytes
reproduce the 18,800,640 B record, and that `--dry-run`/`--help`/the
never-overwrite guard are CPU-safe (no mlx imported).

## 5. Exact window command

Run inside the guarded window (it takes the exclusive GPU lock, restores qwen,
and enforces the system memory ceiling — never launch it backgrounded with `&`;
see `scripts/deepseek_v41/README.md`). `$WT` = the worktree with this branch
checked out; bump the `window-NN` in `--out` to the live window number.

```
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w43
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
cd $WT && bash scripts/deepseek_v41/gpu_window.sh \
  env PYTHONPATH=$WT $PY scripts/deepseek_v41/gather_qmm_microbench.py \
  --arms mxfp4_switch mxfp4_convention affine_q4_gs64 affine_q8_gs64 bf16_dense \
  --m-values 1 4 --top-k 6 --hidden 5120 --inter 2304 \
  --bank-slots 92 --group-size-mxfp4 32 --iters 50 --warmup 10 \
  --memory-limit-gib 8.0 --seed 0 \
  --out $WT/docs/deepseek-v41/receipts/gpu-windows/window-16/gather-qmm-microbench-1024.json
```

`--dry-run` (prints the full shape/byte plan + this command, no mlx, no GPU):

```
PYTHONPATH=$WT $PY scripts/deepseek_v41/gather_qmm_microbench.py --dry-run
```

## 6. Scope / caveats

- **Value is A/B-pending, not asserted.** This worker proves the call is
  shape-/byte-/FLOP-minimal and builds the instrument; whether the mxfp4 M=1
  variant is bandwidth- or ALU-bound is the window's call.
- Consistent with W42: the exposed `moe.routed_switch` cost is the per-layer
  blocking `mx.eval` fence, not this gather. If the microbench shows the gather
  is already near bf16 bandwidth at M=1, the gather is confirmed off the critical
  path and the lever stays the fence/submit cadence (W42 §8).
- The switch's `DenseIslandSwitchGLU.__call__` (L743) uses the **same**
  `_gather_component_bank` with the same flat layout — the audit covers it too.
- `moe.shared_expert` is a resident dense `Expert` (q8 gs64 affine), a **separate
  stage**, not part of the routed gather — out of scope.
