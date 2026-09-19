# W15 — DeepSeek-V4.1-Flash native-mxfp4 routed-expert codec (converter + runtime + L0/L1 pilot)

**Task.** W9 proved the port is correct and the affine **Q2/gs64** routed-expert bank is
the sole cause of the probe's junk (`moe_L0_output` cos **0.927** vs the fp32 reference).
The lossless replacement is the **native FP4**: keep the source E2M1 codes + E8M0 per-32
scales and repack them into MLX's **mxfp4** layout (4.25 bpw, 269 GiB). This W15 delivers the
codec end to end — converter mode, manifest admission, spec entry, runtime execution, and an
L0/L1 pilot proven against the torch goldens through the real streaming path.

## TL;DR

- **`mx.quantize(mode="mxfp4", group_size=32)` on mlx 0.32.2 is a bit-exact repack** of the
  on-grid FP4 source (proven: W9 `bank_mx_probe.json` 191 experts×3; W15 spot-checks; the
  stale `test_mxfp4_is_not_bit_exact` test is corrected — it now IS exact on this box).
- **Record = 18,800,640 B** (packed FP4 17,694,720 + E8M0 scales 1,105,920, **no bias**);
  bank 40×384 = **288,777,830,400 B = 268.99 GiB**; spec total **313,941,753,752 B**.
- **Runtime codec works through the real streaming switch**: `mx.gather_qmm(mode="mxfp4")`
  and `mx.quantized_matmul(mode="mxfp4")` are both supported for gather on mlx 0.32.2 (no
  dequant fallback needed on this box), with the ±10 SwiGLU clamp preserved. affine
  (hy3/glm/deepseek_v4) stays byte-identical.
- **Pilot L0/L1 proven** (memory-safe, real streaming code path): the mxfp4 codec reproduces
  the fp32 reference expert math to **cos = 1.000000** (maxabs ≈ 6e-8) on every golden-routed
  expert; composed with the W9 candidate-bank ladder this gives **moe_L0 cos 0.99999999999999,
  moe_L1 cos 0.9999999999999988** vs the fp32 reference (the Q2 bank scored 0.9347 / 0.9283).
- **Full-bank conversion ≈ 2.6 h** single-process (608 ms/expert with per-record bit-exact
  verify); **decode 4.51 GB/token** (mxfp4) vs 2.65 GB/token (Q2) = **1.70×**.

## Record byte layout (PINNED; W16 writes the full bank with exactly this framing)

Per expert, six ordered components, **contiguous**, **no bias leaf**:

| # | component | dtype | shape | bytes | rel_offset |
|---|-----------|-------|-------|-------|-----------|
| 0 | gate_proj.weight | U32 | [2304, 640] | 5,898,240 | 0 |
| 1 | gate_proj.scales | U8 (E8M0) | [2304, 160] | 368,640 | 5,898,240 |
| 2 | up_proj.weight | U32 | [2304, 640] | 5,898,240 | 6,266,880 |
| 3 | up_proj.scales | U8 (E8M0) | [2304, 160] | 368,640 | 12,165,120 |
| 4 | down_proj.weight | U32 | [5120, 288] | 5,898,240 | 12,533,760 |
| 5 | down_proj.scales | U8 (E8M0) | [5120, 72] | 368,640 | 18,432,000 |

- **record = 18,800,640 B** = packed 17,694,720 (= 3·5120·2304·4/8) + E8M0 scales 1,105,920
  (= 3·5120·2304/32 × 1 B).  record offset = **(L·384 + e)·18,800,640** (contiguous).
- **bank (40 layers × 384 experts = 15,360 records) = 288,777,830,400 B = 268.99 GiB.**
- **pilot (2 layers × 384 = 768 records) = 14,438,891,520 B = 13.45 GiB.**
- Manifest `quantization` block = `{bits: 4, group_size: 32, mode: "mxfp4"}`; each record's
  segments list weight+scales only.  Sidecar **alignment = 8192** (largest power-of-two that
  divides 18,800,640 = 2¹³·2295, so the contiguous bank validates; the pread/F_NOCACHE
  streaming reader needs no 16 KiB alignment — only the unused zero-copy mmap-mapped store
  would, so mxfp4 uses the component-bank/pread slot layout).

## What shipped (all CPU, `nice -n 19`, no GPU lock)

| file | change |
|------|--------|
| `mtplx/deepseek_v41_convert.py` | mxfp4 primitives: `quantize_mxfp4` (reuses `bank_mx_probe.py`'s recipe verbatim), `mxfp4_dequant_equals_source`, `mxfp4_component_bytes`, `MXFP4_EXPERT_RECORD_BYTES` |
| `mtplx/expert_manifest.py` | mxfp4 mode: 6 weight+scales components, `validate_structure` (bits=4/gs=32), `_expected_component_shape` (U32 weight + U8 scales), record-order admission, `MXFP4_ALIGNMENT=8192` |
| `mtplx/expert_streaming_models.py` | `DEEPSEEK_V41_FLASH_EXPERT_MXFP4` (codec=mxfp4, bits=4, gs=32, 1-byte scale, no bias; codec-aware `expert_record_bytes`); same residents/engram/router/KV/`swiglu_limit` as Q2 |
| `mtplx/models/expert_mlx.py` | codec threaded through `_gather_component_bank` / `_run_component_bank_q4` / `_run_q4_expert` / `_run_mapped_q4` (+ `_run_mxfp4_expert`); `MappedExpertRecord` U8; slot geometry (`_component_bank_slots`) mxfp4 branch; `HotExpertSwitchGLU` / `DenseIslandSwitchGLU` / `MappedExpertSwitchGLU` carry `spec.expert_codec` |
| `scripts/convert_deepseek_v41_streamed.py` | `--expert-codec mxfp4` (streams one expert at a time, residents hardlinked from a sibling, per-record bit-exact gate, resumable) + `--from-bank` pilot builder |
| `tests/test_convert_deepseek_v41_streamed.py` | fixed stale `test_mxfp4_is_not_bit_exact` → `test_mxfp4_is_bit_exact` + mxfp4 primitive/byte tests |
| `tests/test_deepseek_v41_mxfp4.py` | mxfp4 manifest admission + spec byte math + rejection guards |
| `tests/test_expert_mlx_mxfp4.py` | mxfp4 gather == dequant-matmul (cos ≥ 0.9999), clamp applied, affine unchanged |

`expert_slots.py` needed **no** change — the slot machinery reads records generically by
segment (no bias/9-component assumption).

## Runtime codec — mlx 0.32.2 supports mxfp4 for gather

Verified on this box: `mx.gather_qmm(x, W, S, lhs_indices, rhs_indices, transpose=True,
group_size=32, bits=4, mode="mxfp4")` and `mx.quantized_matmul(..., mode="mxfp4")` both run
and return bf16.  So the streamed switch executes mxfp4 records **directly** through
`_gather_component_bank(codec="mxfp4")` — no dequantize-to-bf16 fallback is required.  The
±10 asymmetric SwiGLU clamp is applied between the gate/up qmm and the SiLU exactly as the
affine path (`_clamped_swiglu`).  Regression: `hy3/glm/deepseek_v4` (codec affine) run through
the unchanged `mode="affine"` branch; 264 codec-dispatch tests pass (2 pre-existing
`test_dense_islands` failures — `DenseIslandSwitchGLU.swiglu_limit` None-attr, an mlx
`nn.Module` issue, **confirmed identical on the base commit HEAD~2**, unrelated to mxfp4).

## Pilot — L0/L1 built and proven through the real streaming path

**Pilot artifact** `~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4-pilot/`: 768 records
(13.45 GiB) copied from W16's growing full bank (identical pinned framing — no re-quant),
residents/engram/config/index/tokenizer **hardlinked** from the mxfp4 artifact, manifest built
+ verified (`verify_expert_manifest`: **checked_records=768, sidecar_verified=true**, digest
26d9379ce6a7).  Spot-check: sampled records (L0 E0/E200/E383, w1/w2/w3) **dequantize bit-exact
vs the FP4 source**.

**Golden parity (memory-safe, `proof/w15_pilot_codec.py`, peak RSS 2.62 GB, watchdog @ 6 GB).**
The full 40-layer model load ballooned to ~100 GB (see below), incompatible with the 12 GB
cap, so the pilot is proven as a three-link chain, each link measured on the **real** pilot
bank through the **real** switch code path:

1. **Bank == source.** Pilot records dequantize **bit-exact** vs the FP4 E2M1+E8M0 source
   (spot-check + the codec harness dequant).
2. **Runtime codec exact.** `_gather_component_bank(codec="mxfp4")` (the function
   `HotExpertSwitchGLU` dispatches to) fed the real pilot-bank component bank reproduces an
   independent fp32 dequant-matmul reference (with the ±10 clamp) to **cos = 1.000000**
   (maxabs ≈ 6e-8) for **every golden-routed expert** — L0: 62 experts, L1: 63 experts.
3. **Reference tie (W9 `torchref_bank_ladder.json`, committed).** Feeding the mxfp4 (=source)
   experts into the fp32 reference forward gives **moe_L0 global cos 0.99999999999999 /
   moe_L1 0.9999999999999988** vs R0 — versus the Q2 bank's **0.9347 / 0.9283**.

Composition: the routed-expert bank is the only thing that changed vs Q2 (router weighting +
shared expert are codec-independent, unchanged, and covered by the affine regression); link 2
proves the streamed routed computation equals the fp32 reference routed computation, and link 3
proves those experts give a reference-matching MoE. Therefore **moe_L0_output / moe_L1_output
through the real streaming path match the fp32 reference at cos ≥ 0.999 (effectively 1.0)** at
the q8 floor — the deliverable's bar, cleared.

Per-record bytes (from the served manifest): **18,800,640 B/record**, 768 records.

## Memory diagnosis (why the full-model probe was killed)

The first harness (`proof/w15_pilot_golden.py`, mlx_dump-style full-model load) was killed
twice (≈100 GB, then 78 GB in 55 s). Culprits — both whole-artifact loads, not leaks:
1. `apply_memory_cap=False` + `memory_limit_bytes=100 GiB` → the expert-cache planner
   allocates a persistent bank sized toward the **full 100 GiB** budget.
2. `load_text_only_resident_arrays` materializes **all 40 layers' text residents (8.67 GB)**
   at once (manifest-driven), plus the embed table.

Even with a tight cap the resident floor (8.67 GB) exceeds the 12 GB cap's < 4 GB dry-run bar,
so the full-model end-to-end probe is not runnable under this cap. The streamed proof above
(peak 2.62 GB, one expert record at a time, watchdog self-kill) is the memory-safe substitute
and is strictly on the real bank + real switch code. A true single-model serve-path probe
should run on a box/budget that can wire the 8.67 GB residents (or once W16's full artifact
admits under the production memory plan).

## Estimates

- **Full-bank conversion wall time.** Measured 608 ms/expert (dequant_fp4 + `quantize_mxfp4`
  + per-record bit-exact verify + pack), peak RSS 1.16 GB, single process streaming one
  layer's source shard. 15,360 experts → **≈ 155.7 min ≈ 2.6 h** (`--no-verify-exact` trims
  the verify dequant; verify is the losslessness gate and is on by default).
- **Decode bytes/token (6 experts × 40 layers × record).** mxfp4 = 6·40·18,800,640 =
  **4,512,153,600 B/token (4.20 GiB)** vs Q2 6·40·11,059,200 = **2,654,208,000 B (2.47 GiB)**
  → **1.70×** the Q2 decode read. (The size is the lossless-quality cost; q4/gs64 affine, the
  next step down, is 285 GiB / 3.71 GiB/token at cos 0.995 — see W9.)

## Constraints honored

CPU only (`mx.set_default_device(mx.cpu)`); no GPU lock; `nice -n 19`; every run streamed one
expert record at a time with an in-process RSS watchdog; peak RSS ≤ 2.62 GB (pilot proof) /
1.16 GB (conversion rate) — under the 12 GB cap. No pushes/PRs. The existing Q2 artifact was
never touched (it was deleted by David mid-task; the pilot reuses W16's mxfp4 bank + the
q2-provenance resident manifest). Commits landed per piece with no attribution trailers.
