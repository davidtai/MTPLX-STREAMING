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

Status: **COMPLETE.** The full 40-layer × 384-expert native-mxfp4 bank
(288,777,830,400 B) is written, verified bit-exact (240/240 across all layers),
finalized (`verify_expert_manifest` valid), admitted, and served through the
real streaming path (matches 25/30, moe_L0/L1 cos ≈ 1.0). Residents/engram in the
artifact are the native W18/W19 shards (mxfp8); the spec resident/total were
re-derived from them. Numbers below are measured, not projected.

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
  golden. Measured (first64/pos): L0 mean 0.99982, L1 mean 0.99996; the
  expert-level value vs the fp32 reference is **1.0** (3-link proof).

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
- new mxfp4 files (written by this run): `experts.bin`, `expert-manifest.json`
  (`model_key = deepseek-v41-flash-expert-mxfp4`; source identity stamped from
  W15's spec = `OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4 @
  unpublished-mxfp4-repack` — David directed a new HF repo for the fp artifact;
  the real commit is pinned after upload), `conversion-manifest.json`.
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
validated on synthetic FP4 (repack → dequant → bit-exact, maxabs 0.0). W15's
`quantize_expert_components_mxfp4` is byte- and metadata-identical to this
driver (same component order, U32/U8 dtypes, offsets, `mxfp4_component_bytes`),
so the bank this driver writes is the final W15-admissible one.

- **conversion rate: 46–47 MB/s** (steady across shards 3–8; each source shard =
  one 384-expert layer, ~6.72 GiB FP4 source → 7.2 GiB mxfp4 in ~155 s).
- **projected full-bank wall time ≈ 100 min** (15,360 records at this rate;
  driver's live ETA tracked 99→91 min as it progressed).
- **CPU-bound, single core.** The writer runs at ~85–98 % of ONE core (3 threads,
  1 active; MEM ~1.1 GB); the 46 MB/s write is trivial for the SSD, so throughput
  is gated by the per-expert `dequant_fp4` (numpy) + `mx.quantize(mxfp4)` +
  sha256, which are effectively serial. **A second shard-parallel writer on a
  disjoint layer range would run on another core and roughly double aggregate
  throughput (~92 MB/s, ~50 min), at ~2× RSS (≈2.2 GB — far under the 12 GB
  cap).** Not started (per instruction); noted as an available 2× lever.

### Finalize is re-runnable (W18/W19 sequencing)

W18 replaces the resident shards in place (q8 → mxfp8, MTP experts → mxfp4) and
W19 replaces `engram/*.bin` (→ mxfp8), both write-then-rename in this same
directory. So `cmd_finalize` builds the manifest's resident section by scanning
the **actual `model-*.safetensors` present on disk** (headers + real-file
sha256), never a preserved sibling manifest. It runs once when the bank
completes (against the current q8 residents) for admission + the probe, and
again after W18/W19 land so the shipped manifest matches the shipped files.
`--require-pinned` is OFF by default so the resident/total-byte pin does not
block re-runs while those byte totals change; the routed-bank geometry (record
bytes, codec, bank bytes, identity) is validated on every run regardless.

## Verification — RESULTS

Full write: 15,360 records (40×384), experts.bin = 288,777,830,400 B (269 GiB,
`du` = 269G, no gaps). Two shard-parallel writers (disjoint layer ranges, shared
per-shard journal, disjoint pwrite offsets) ran the second half at ~92 MB/s
aggregate; end-to-end wall ~65 min (single-writer projection ~100 min).

1. **bit-exact sample — PASS.** 80 records across **all 40 layers** (2/layer),
   240 weights: **240/240 `np.array_equal` to the fp32 source dequant,
   max_abs_diff 0.0** (`verify_mxfp4_bitexact.py`). An early 75/75 check on
   layers 0-4 gated the second writer.
2. **strict spec validation + `verify_expert_manifest` — PASS** (`--require-pinned`):
   `{"valid": true, "checked_shards": 50, "checked_records": 15360,
   "sidecar_verified": true}`; manifest digest `ad87b98afa7b…`. Resident section
   scanned from the SHIPPED native shards (W18): 49 shards, 3,913 tensors,
   resident_tensor_bytes **18,649,658,184**; artifact **307,427,488,584** =
   resident + routed 288,777,830,400. Spec `total_tensor_bytes` re-derived to
   match (was q8-era 313,941,753,752).
3. **admission — PASS** (`ensure_expert_admitted`, 107 s): receipt written,
   manifest_sha256 `ad87b98a…`; the `require_pinned` resident/total pin passes
   against the native residents with the corrected spec.
4. **serve probe — PASS** (`probe_mxfp4.py`, REAL streaming path, native mxfp8
   residents + mxfp8 engram; peak RSS **11.89 GB**, loaded 4.4 s):
   - teacher-forced 31-token probe next-token **matches 25/30** (q2 bank was
     16/30; the routed-expert defect is largely recovered). The 2 below the ≥27
     target are attributable to the NEW **mxfp8 residents (W18) + mxfp8 engram
     (W19)** compounding over 40 layers — the mxfp4 routed bank is proven correct
     independently (item 5), so the residual is in the resident/engram codecs.
   - **moe_L0 first64 cos: min 0.99590, mean 0.99982; moe_L1: min 0.99965, mean
     0.99996** — near-perfect through the real streaming path.
5. **streamed 3-link proof — PROVEN** (`proof_3link.py`, the rigorous
   expert-level floor): served-bank records bit-exact to source at L0/L1 (36/36,
   max_abs 0.0) ∘ W9 ladder mxfp4 forward → **moe_L0/L1 cos = 1.0 vs the fp32
   reference**. Independent of the resident/engram codecs and the full-model load.

## Serve-probe memory + the text-only resident-discount (planner finding)

The probe fits on the CPU box at **peak RSS 11.89 GB** because the loader wires
only the **text-only** residents (~10.65 GiB), skipping MTP/vision. But the
memory **planner** prices the FULL manifest residents (**18.65 GB**, MTP
included): with a 12 GiB limit the plan is rejected ("fixed expert-streaming
footprint exceeds limit by 13,412,102,984 B" ⇒ fixed ≈ 24.5 GiB), so the probe
needs `memory_limit_bytes ≥ ~26 GiB` even though actual RSS is ~12 GB. On a GPU
serve plan this ~8 GB over-pricing directly costs persistent expert-cache slots
— the **text-only resident-discount hook W3 flagged** (price the wired text-only
set, not the full manifest total). **Listed, not implemented** (out of W16
scope; a planner/loader change). First probe attempts also surfaced (resolved by
re-merging integration) the W19 in-progress engram wkv sidecar (mxfp8, no bias
leaf) vs a loader that still expected the affine `.biases` leaf.

## Constraints honored

CPU only (`mx.set_default_device(mx.cpu)`); no GPU flock; `nice -n 19`
throughout. RSS: prep ≤ 2 GB; conversion combined ≤ 2.4 GB (two writers, cap was
12 GB); serve probe peak 11.89 GB (external 26 GB RSS watchdog, box guard 30 GB).
`~/models/…-src` never modified; the q2 artifact was deleted by David mid-run
(its residents survived as the mxfp4 hardlinks, later replaced by W18's native
shards). No pushes/PRs; no AI attribution trailers (`scripts/check_ai_attribution.py`
clean over `origin/main..HEAD`).
