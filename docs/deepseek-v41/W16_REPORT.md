# W16 — DeepSeek-V4.1-Flash native-mxfp4 routed-expert bank (full conversion)

**Task.** Replace the broken affine-Q2 routed-expert bank (W9: `moe_L0_output`
cos 0.927 vs the fp32 reference; the whole port gap) with the routed-expert
format David picked — **native mxfp4 gs32**, a *lossless repack* of the source
FP4 (E2M1 + E8M0) experts — and convert the full 40-layer × 384-expert bank into
a streamed artifact that admits and serves.

The original W16 brief was an **affine q4 gs64 pilot** on the existing codec.
That pilot was **superseded before any conversion ran**: the W9 candidate-bank
ladder already measured q3/q4/q6 through the reference and found **native mxfp4
gs32 is bit-exact to the source (cos 1.000000) at 269 GiB — smaller than affine
q4 gs64 (285 GiB, cos 0.995)**. Native mxfp4 is therefore strictly better
(smaller *and* lossless), so the affine pilot conversion was not built. The
ladder that drove the decision is recorded below.

Status: **converter driver + bit-exact verifier + serve-probe written and
validated on synthetic FP4; artifact directory scaffolded (residents/engram
hardlinked from the q2 artifact, 0 extra bytes).** The full conversion runs as
soon as W15's record-framing + manifest-fields commit lands on
`feat/deepseek-v41-w15` (poll in progress). Conversion/verify/admission/probe
numbers are filled in as they land (marked _PENDING RUN_).

---

## Definitions (one name per quantity)

- **record bytes** — on-disk size of one expert's streamed record (gate+up+down).
  mxfp4: `3·(params·4/8 + params/32)` with `params = 3·5120·2304` per projection
  triple; **no bias leaf** (E8M0 scale only). Affine: `params·bits/8 +
  (params/gs)·2·2` (bf16 scale + bf16 bias).
- **bank GiB** — `routed_layer_count(40) · expert_count(384) · record_bytes / 1024³`.
- **bytes/token** — cold expert bytes read per decode token at the served route
  = `top_k(6) · routed_layers(40) · record_bytes`.
- **expert cos vs source** — mean per-weight cosine of the requantized expert vs
  the fp32-dequant of the FP4 source (W9 `bank_mx_probe.py`, `mx.quantize`).
- **MoE g (L2)** — global (magnitude-weighted) cosine of the layer-2 MoE output
  vs the fp32 reference R0, holding dense/engram/gate/router fixed and varying
  ONLY the expert format (W9 `bank_ladder.py`, torch reference).
- **moe_L{0,1} cos (streaming)** — cosine of the MoE-module output at layers 0/1
  through the **real streaming serve path** on the new bank vs the fp32 reference
  golden. _PENDING RUN._

## Routed-expert format ladder (W9 reference measurement; the decision)

Requant of the FP4-source experts into each candidate, reference forward with
only the expert format varying (`torchref_bank_ladder.json`, `bank_mx_probe.json`).

| format | record bytes | bank GiB (40×384) | bytes/token (6×40) | expert cos vs source | MoE g (L2) |
|--------|-------------:|------------------:|-------------------:|---------------------:|-----------:|
| affine q2 gs64 (old bank) | 11,059,200 (11.06 MB) | 158.20 | 2.47 GiB | 0.9121 | 0.9470 |
| affine q3 gs64 | 15,482,880 (15.48 MB) | 221.48 | 3.46 GiB | 0.9782 | 0.9862 |
| affine q4 gs64 (superseded pilot) | 19,906,560 (19.91 MB) | 284.79 | 4.45 GiB | 0.9953 | 0.9972 |
| affine q4 gs32 | 22,118,400 (22.12 MB) | 316.41 | 4.94 GiB | 0.9963 | 0.9978 |
| affine q6 gs64 | 28,753,920 (28.75 MB) | 411.29 | 6.43 GiB | 0.9997 | 0.9998 |
| **native mxfp4 gs32 (chosen)** | **18,800,640 (18.80 MB)** | **268.95** | **4.20 GiB** | **1.000000** | **1.000000** |

mxfp4 is **bit-exact** (`bit_exact_vs_source = true`, 191 experts × 3 weights) —
it carries a 1-byte E8M0 scale per 32 and no bias, so it is *smaller* than affine
q4 gs64 while being lossless. This is why the affine q4 pilot was dropped.

## mxfp4 record framing (mlx 0.32.2; W15 owns the canonical manifest fields)

`mx.quantize(fp32_dequant(source_fp4), group_size=32, bits=4, mode="mxfp4")` →
`(packed uint32 [out, in/8], scales uint8 E8M0 [out, in/32])`, no biases; the
dequant is bit-exact to the source (validated here on synthetic FP4, maxabs 0.0,
and per the W9 probe on real experts). Record component order (mirrors the affine
`COMPONENTS` minus biases):

```
gate_proj.weight  U32 [2304,640]   gate_proj.scales  U8 [2304,160]
up_proj.weight    U32 [2304,640]   up_proj.scales    U8 [2304,160]
down_proj.weight  U32 [5120,288]   down_proj.scales  U8 [5120,72]
```
record = 17,694,720 (packed) + 1,105,920 (scales) = **18,800,640 B**.
`record_index(layer L, expert e) = L·384 + e`; `offset = index · 18,800,640`.

## Artifact scaffold (verified)

`~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/`. Everything except the
routed bank is **hardlinked** from the q2 artifact (identical bytes; the q8 dense
residents do not depend on the routed codec):

- hardlinked (nlink=2, same inode, 0 extra bytes): `config.json`,
  `model-000{01..49}.safetensors` (residents, incl. q8 MTP dense/experts),
  `model.safetensors.index.json`, `encoding/`, `engram/` (L1+L14 banks +
  residents sidecar + manifest), `tokenizer*.json`, `LICENSE`, `README.md`,
  `route-census.json`.
- new mxfp4 files (to be written): `experts.bin`, `expert-manifest.json`
  (`model_key = deepseek-v41-flash-expert-mxfp4`; source identity =
  `OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2 @
  b64980a16283647bb213ab475335f38f516e0d9e`), `conversion-manifest.json`.
- **`island-placement.json` intentionally NOT hardlinked** — island placement is
  a function of `expert_record_bytes` (18.80 MB here vs 11.06 MB for q2), so the
  loader must recompute it for the larger record. The q2 artifact is never touched.

Verified: q2 and mxfp4 share inodes for sampled residents/engram/config; the 4
mxfp4-specific files are absent (to be generated); free space unchanged.

## Disk plan

- Free (post-reboot, 2026-09-10): **532 GiB** (`df -g`) / 572 GB (`df -H`).
- New bytes written = **experts.bin only ≈ 268.95 GiB** (residents/engram/config
  hardlinked ⇒ 0 extra). Journal + manifests < 0.1 GiB.
- After write: 532 − 269 = **≈ 263 GiB free** — above the ≥ 150 GB floor.
- The q2 artifact (~376 GiB) **stays** untouched; hardlinks share its
  residents/engram, so it is not double-counted.
- RSS: one expert streamed at a time (peak ≈ a few hundred MB); hard cap 12 GB
  for the process tree; `nice -n 19`; CPU-only (no GPU lock).

## Conversion (driver: `.benchmark-artifacts/deepseek-v41/w16/convert_mxfp4_driver.py`)

Resumable per source shard (journal OUTSIDE the artifact), deterministic pwrite
offsets into a pre-sized `experts.bin`, progress log with MB/s + ETA. Core
validated on synthetic FP4 (repack → dequant → bit-exact, maxabs 0.0).

- conversion rate (MB/s): _PENDING RUN (reported after shard 1)._
- projected full-bank wall time: _PENDING RUN._

## Verification (all _PENDING RUN_, after conversion)

1. **bit-exact sample** (`verify_mxfp4_bitexact.py`): ≥ 64 records across all 40
   layers; each projection dequantized from `experts.bin` must equal the fp32
   source dequant bit-for-bit (`np.array_equal`). _PENDING._
2. **strict spec validation** + `verify_expert_manifest` (records + sidecar +
   shard hashes). _PENDING (couples to W15 manifest)._
3. **admission** on the new artifact (`ensure_expert_admitted`). _PENDING._
4. **serve probe** (`probe_mxfp4.py`, real streaming path): teacher-forced
   31-token probe next-token **matches/30** (expect ≥ 27) and **moe_L0/L1 cos**
   vs the fp32 reference golden. _PENDING._

## Constraints honored

CPU only (`mx.set_default_device(mx.cpu)`); no GPU flock; `nice -n 19`; RSS ≤ 12
GB (conversion) / ≤ 2 GB (prep); the q2 artifact and `~/models/…-src` are never
modified; no pushes/PRs; no AI attribution trailers (checked with
`scripts/check_ai_attribution.py`).
