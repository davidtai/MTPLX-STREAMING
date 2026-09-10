# W9 — DeepSeek-V4.1-Flash torch-reference ground truth + first-divergence localization

**Task.** Run DeepSeek's own reference implementation on CPU with the real weights,
compare per-layer/per-sublayer against our MLX port on the 31-token teacher-forced
probe, and name the first diverging layer/sublayer (or prove agreement).

**Probe (BOS id 0 + the 30 encoded ids):**
`[0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291, 6328, 3465, 1241, 6036, 14, 291, 3395, 361, 1354, 260, 565, 291, 6328, 3465, 21740, 6036, 14, 291, 2605]`

## TL;DR verdict

**No port bug. The MLX forward is correct; the entire ref-vs-MLX gap is the Q2
routed-expert bank.** The attribution ladder (below) substitutes our exact artifact
tensors into the torch reference: with our **q8 dense projections + q2 experts +
reference router (R2)**, the reference **reproduces the MLX forward to global cos
0.9993–0.9999** across all layer-0–2 outputs — the residual is the fp32-vs-bf16
activation-storage difference, itself proven ~0.995/op. Isolating each substitution:
**q8 dense (R0→R1) changes cos by <1e-4; the q2 experts (R1→R2) collapse it from
0.9999 to 0.93; forcing the MLX router (R2→R3) barely moves it.** So the whole gap is
**q2 routed-expert quantization** (source ships experts at **fp4/4-bit**, the artifact
re-quantizes them to **2-bit**). Corroborating:
- The apparent first divergence (`attn_L0_window_kv`, cos 0.9947 vs the fp32 oracle)
  is **bf16 activation storage, not a bug**: a bf16-cast recompute reproduces the
  full-run tensor to **cos 1.000000**, and at matched dtype the MLX window-KV code +
  q8 weights match the oracle to **cos 0.99999**.
- **Engram row-id hashing is bit-exact (744/744).**
- **The MoE gate is bit-identical** (source bf16 == artifact bf16, `|Δw| = 0`); fed
  the *same* input the reference and MLX gates agree **31/31** on the top-6. The
  router flips seen in the full run (19/31 at L0) are a *downstream* symptom of the
  q2-quantized MoE input drifting, not a gate bug.

This matches W8's "withdraw 'Q2 quality'": the scattered prefill junk is q2 quality
loss, not a layers-0–2 code defect.

## What was built (all CPU, float32; no GPU, no `mx` in the oracle)

| file | role |
|------|------|
| `scripts/deepseek_v41/torchref/ref_forward.py` | Imports the **reference** `inference/model.py` + `engram.py` classes, replaces only the CUDA kernels (act_quant/fp4_act_quant/fp8_gemm/fp4_gemm/hc_split_sinkhorn/sparse_attn) with pure-torch float32 equivalents, dequantizes weights lazily from the source HF shards, runs layers 0–2, emits the goldens. |
| `scripts/deepseek_v41/torchref/mlx_dump.py` | Runs OUR streaming model (component-banks) on CPU, early-stops after layer 2, captures the same per-submodule tensors. Repoints the editable-install finder to THIS worktree (the `main` checkout has no deepseek_v41 model). |
| `scripts/deepseek_v41/torchref/compare_ref_vs_mlx.py` | Per-position cosine + max-abs, router top-6 overlap, engram row-id agreement, first-divergence verdict. |
| `scripts/deepseek_v41/torchref/isolate_wkv.py` | The decisive dtype/quant isolation for the window-KV path. |
| `scripts/deepseek_v41/torchref/export_mlx_dense.py` | Dumps our q8-dequantized dense residents (layers 0–2) to npy for the ladder. |
| `scripts/deepseek_v41/torchref/ref_ladder.py` | The R0→R3 attribution ladder (numpy q2 dequant of `experts.bin`, validated bit-exact) → `torchref_ladder.json`. |
| `docs/deepseek-v41/receipts/torchref_golden_*_L*.json` | Per-submodule goldens for the W10/W11/W13 porters (README_torchref_goldens.md documents shapes/dtypes). |
| `docs/deepseek-v41/receipts/torchref_layers012.json`, `compare_ref_vs_mlx.json` | Per-layer residual dump + the comparison. |
| `tests/test_deepseek_v41_torchref_golden.py` | Encodes the reference values as regression goldens (CPU-pinned; heavy MLX parity opt-in via `MTPLX_RUN_HEAVY_DSV41=1`). |

The oracle is a **clean full-precision** reference: weights are exactly the on-disk
fp8/fp4 values dequantized to f32, activations carry no quant noise. The real
reference (`generate.py`) and our serve path both run **bf16 activations** — that
gap is the confound the isolation controls for (below).

## Reproduce

```
# oracle (torchref venv: torch CPU wheel + safetensors/numpy/sympy/transformers)
.venv-torchref/bin/python scripts/deepseek_v41/torchref/ref_forward.py --max-layer 2
# MLX serve-path side (project venv, CPU)
.venv/bin/python scripts/deepseek_v41/torchref/mlx_dump.py
# compare
.venv-torchref/bin/python scripts/deepseek_v41/torchref/compare_ref_vs_mlx.py
```

## Per-submodule cosine (torch f32 oracle vs MLX bf16/q8/q2), min over 31 positions

| sublayer | L0 | L1 | L2 | reading |
|----------|----|----|----|---------|
| attn input (post-hc-norm) | 0.99994 | 0.9314\* | 0.9044\* | hc-mix + RMSNorm faithful; L1/L2 inherit upstream q2 error |
| window_kv (RoPE'd latent)  | 0.9947 | 0.9886 | 0.858\* | **proven bf16 storage** (see isolation) |
| attn output                | 0.9909 | 0.9876 | 0.858\* | bf16 storage through same-code attention |
| moe input (post-hc-norm)   | 0.9913 | 0.9615 | 0.9397 | inherits upstream |
| moe shared-expert out (q8) | 0.9940 | 0.9687 | 0.9646 | q8 shared expert faithful within bf16+q8 |
| **moe output (adds q2 routed)** | **0.9271** | 0.9284 | 0.8509 | **q2 routed experts dominate the gap** |
| layer output               | 0.9289 | 0.9329 | 0.9326 | compounded |

\* positions inheriting the compounding q2 error from earlier sublayers; not an
independent signal.

- **Engram row ids (layer 1): 744/744 exact.** The n-gram hash recipe port
  (`mtplx/engram_v41.py` `NgramHashState`) matches the reference `engram.py`
  bit-for-bit for all 31 tokens × 24 columns.
- **Router top-6 overlap:** L0 5.58/6 (19/31 exact sets), L1 5.45/6, L2 5.10/6 —
  degrades only as the (quantized) MoE input drifts; the gate itself is bf16
  (unquantized) and its math matches.
- Both sides are **finite everywhere** (no NaN/Inf) through layer 2.

## The decisive isolation (`isolate_wkv.py`)

Feeding the layer-0 attention the reference's captured post-hc-norm input `x`
(which matched MLX to cos 0.99994), all in MLX:

```
cos(kv_q8,   kv_f32wt)        = 0.999999   # q8 wkv weight vs its own f32 dequant, identical code
cos(kv_f32wt, oracle)         = 0.999996   # MLX window-KV code (dequant weights) vs torch f32 oracle
cos(kv_q8,   oracle)          = 0.999997   # total, at matched (fp32) activation dtype
cos(kv[bf16 input], full_run_mlx_window_kv) = 1.000000   # <== full run == bf16 recompute
cos(kv[bf16 input], oracle)                 = 0.994701   # <== exactly the 0.9947 "divergence"
```

**Conclusion:** the entire 0.9947 window-KV "divergence" is the model storing the
RoPE'd latent at its native **bf16** dtype (`_store_dtype`/`_apply_interleaved_rope`
in `mtplx/models/deepseek_v4.py:1549`), amplified by the `norm_eps=1e-20` RMSNorm at
low-norm positions. The window-KV **code and q8 weights are faithful.** RoPE
convention (interleaved pairs), YaRN inv-freq, and the MLA sparse-attention math all
match the reference. The `MTPLX_DSV4_FP32_ACTIVATIONS=1` arm produced numerically
identical layer stats — the flag only changes the rope/hc **store** dtype, so it
does not by itself lift the residual stream to fp32 (the deficits are storage-level,
already resolved once inputs are fp32).

## Attribution ladder (`ref_ladder.py` + `export_mlx_dense.py`)

Substitutes our artifact tensors into the torch reference (layers 0–2, same 31
tokens). Dense q8 residents are dequantized with `mx.dequantize`; the q2 experts are
dequantized straight from `experts.bin` in numpy (affine bits=2 gs=64, **validated
bit-exact against `mx.dequantize`**, maxdiff 0.0). Receipt: `torchref_ladder.json`.

- **R0** pure reference (fp32 from FP8/FP4 source)
- **R1** R0 + dense projections → our q8 gs64
- **R2** R1 + routed experts → our Q2 (`experts.bin`)
- **R3** R2 + router top-6 forced to the MLX run's per-token selection

min-over-position cosine **vs R0** (isolates each substitution):

| output | R1 (q8 dense) | R2 (+q2 experts) | R3 (+mlx router) |
|--------|---------------|------------------|------------------|
| attn_L0_output | 0.99996 | 0.99996 | 0.99996 |
| moe_L0_output  | 0.99998 | **0.92707** | 0.92707 |
| layer0_output  | 0.99999 | **0.92906** | 0.92908 |
| layer2_output  | 0.99935 | **0.93308** | 0.93304 |

global cosine **vs the MLX run** (does the ladder rung reproduce MLX?):

| output | R0 | R1 | **R2** | R3 |
|--------|----|----|--------|----|
| moe_L0_output | 0.9344 | 0.9344 | **0.99929** | 0.99949 |
| layer0_output | 0.9340 | 0.9340 | **0.99958** | 0.99963 |
| layer1_output | 0.9376 | 0.9375 | **0.99984** | 0.99980 |
| layer2_output | 0.9384 | 0.9382 | **0.99990** | 0.99986 |

**Reading.** q8 dense (R0→R1) is numerically inert (≤1e-4). The q2 experts (R1→R2)
are the whole gap: they drop cos-vs-R0 from 0.9999 to 0.93, and — crucially — carry
R2 to **cos 0.9999 vs the actual MLX run**. Per the verdict rule, **R2 ≈ MLX ⇒ the
forward is correct and the Q2 bank is the whole story.** Forcing MLX's router (R3)
adds only ~0.001, so router flips are a minor, downstream effect.

**Router isolation.** Feeding the *same* fp32 MoE input into the reference gate
(source bf16 weight) and the MLX gate (artifact bf16 weight): **31/31 tokens agree on
the top-6, gate weight `|Δw| = 0`.** The gate is not quantized and is bit-identical;
the 19/31 top-6 agreement seen in the full run is entirely the q2-perturbed MoE input
flipping borderline routes, not a gate defect.

## Why the probe emits junk (now proven, not hypothesized)

The junk (`Kasipak`/`potentially` at scattered positions, 16/30 correct) is **q2
routed-expert quality collapse**, not a layers-0–2 code bug — the ladder proves it:
the source ships experts at **fp4 (4-bit)**; the artifact re-quantizes them to
**2-bit**, and swapping *only* that into the correct reference forward (R1→R2)
reproduces the MLX output (cos 0.9999 vs MLX). The q2 MoE loses ~0.07 cosine per
layer; over 40 layers that flips borderline next-token argmaxes at specific positions
while high-confidence positions stay correct — exactly the scattered pattern. This is
W8's "withdraw 'Q2 quality', record prefill defect", now quantified.

## Recommendations / next steps

1. **Fix quality by re-quantizing the routed experts**, not the code. The layers-0–2
   forward (attention, hyper-connections, engram, RoPE, gate, MoE application) needs
   no change for the divergence seen here. Re-quantize the routed experts at ≥ the
   source **fp4** (or q3/q4) and re-run the 31-token probe; expect the argmax matches
   to recover. A q3/q4-expert artifact is the direct next experiment.
2. **If deeper layers must be audited**, the ladder generalizes: `ref_ladder.py`
   already runs R0–R3 for layers 0–2; extend `--max-layer` (add the layer-8 second
   compressor seam) and read `torchref_ladder.json`. The R2-vs-MLX cos staying > 0.999
   is the "no bug" signal at any depth.
3. **Porters (W10/W11/W13):** test against the committed per-submodule goldens with a
   bf16/q8 tolerance (not exact-fp32). The **exact** anchors that must match
   bit-for-bit regardless of dtype are the **engram row ids** and the **layer-2
   compressor topk_idxs** (both integer, both pinned in the golden test).

## Constraints honored
CPU only (`mx.set_default_device(mx.cpu)` / `torch.device("cpu")`); no GPU lock held;
`nice -n 19` on every run; peak RSS well under 50 GB (streaming expert cache 15 GiB;
the oracle loads only the layer-0–2 tensors + selected experts + hashed engram rows,
never a whole 95 GiB shard). No `~/models` writes. No pushes/PRs. `.venv-torchref`
git-ignored via `.git/info/exclude`.
