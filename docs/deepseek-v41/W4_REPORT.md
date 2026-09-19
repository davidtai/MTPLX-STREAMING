# W4 — Engram resident projections sidecar (DeepSeek-V4.1-Flash)

Branch `feat/deepseek-v41-w4`. The streaming artifact
`~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2` carried the Engram row banks
(`engram/engram-L{1,14}.bin`) but **not** the small resident Engram projection tensors.
W4 re-materialises them into one sidecar next to the banks and records them in the
manifest, so `EngramV41` can be constructed straight from the artifact.

## What was added

- `scripts/convert_deepseek_v41_engram.py` — new `--mode residents` (default stays `banks`,
  byte-for-byte unchanged). Streams the four resident tensors per engram layer from the
  source shard (never mmaps the 95 GiB shard), dequantizes `wkv` per the **same FP8
  block-scale formula the streamed converter uses for every other FP8 resident**
  (`mtplx.deepseek_v41_convert.dequant_fp8_block`), re-quantizes it to affine **q8/gs64**
  via `dc.quantize_affine` (identical path to the other resident q8 tensors), keeps
  `q_weight`/`k_weight` exact, writes `engram/engram-residents.safetensors`, and adds a
  `residents` entry to `engram/engram-manifest.json` — recomputing `manifest_sha256` with the
  **same recipe** as `write_manifest` (sha256 over `json.dumps(manifest, indent=2)` of the
  manifest without the `manifest_sha256` field). `experts.bin`, the resident shards, and the
  row banks are never touched.
- `mtplx/engram_v41.py` — loader helper only: `EngramResidents` dataclass +
  `load_engram_residents(artifact_dir, layer_id)`. Wraps the affine-q8 `wkv` as an
  `mx.quantized_matmul` callable (the shape `EngramV41` expects) and returns the exact F32
  `q_weight`/`k_weight`; `EngramResidents.build_module(...)` constructs the `EngramV41` hook.
- `tests/test_engram_residents.py` — runs against the real sidecar when present, skips otherwise.
- `tests/test_convert_deepseek_v41_engram.py` — added a synthetic, always-run residents test
  (`test_convert_residents_sidecar_and_manifest`) covering the whole path end-to-end.

`mtplx/deepseek_v41_convert.py` was **not** modified — all new logic lives in the converter
script and reuses the existing `dc.*` primitives.

## Source tensors consumed (pinned rev `dba1be0a40aa45a94ad051997016db3960a90277`)

| Layer | shard | wkv.weight | wkv.scale | q_weight | k_weight |
|---|---|---|---|---|---|
| 1  | `model-00047-of-00048.safetensors` | `F8_E4M3 [25600, 6144]` | `F8_E8M0 [800, 192]` (32×32 block) | `BF16 [4, 5120]` | `BF16 [4, 5120]` |
| 14 | `model-00048-of-00048.safetensors` | `F8_E4M3 [25600, 6144]` | `F8_E8M0 [800, 192]` (32×32 block) | `BF16 [4, 5120]` | `BF16 [4, 5120]` |

**Dtype note:** the task brief listed `q_weight`/`k_weight` as `f32`; the pinned source
stores them **`BF16`**. BF16 → F32 is a lossless widening (every BF16 value is exact in F32)
and `EngramV41` upcasts them to F32 anyway, so they are stored **exact as F32** — no
information lost. Reported as `q/k exact` with `max_abs_err = 0.0`.

## Output: `engram/engram-residents.safetensors`

- Size: **334,562,304 bytes** (319.06 MiB), 10 tensors, metadata `{"format":"mlx"}`
- sha256: `5dedcbc21708d3617c41df276c218d819a4e81003db666e3f1b8dc9c4334951a`

Exact tensor names / dtypes / shapes (the q8 convention of the other resident q8 tensors,
e.g. `aligner.w1.{weight,scales,biases}`):

| tensor | dtype | shape | bytes |
|---|---|---|---|
| `layers.1.engram.wkv.weight`  | U32  | `[25600, 1536]` | 157,286,400 |
| `layers.1.engram.wkv.scales`  | BF16 | `[25600, 96]`   | 4,915,200 |
| `layers.1.engram.wkv.biases`  | BF16 | `[25600, 96]`   | 4,915,200 |
| `layers.1.engram.q_weight`    | F32  | `[4, 5120]`     | 81,920 |
| `layers.1.engram.k_weight`    | F32  | `[4, 5120]`     | 81,920 |
| `layers.14.engram.wkv.weight` | U32  | `[25600, 1536]` | 157,286,400 |
| `layers.14.engram.wkv.scales` | BF16 | `[25600, 96]`   | 4,915,200 |
| `layers.14.engram.wkv.biases` | BF16 | `[25600, 96]`   | 4,915,200 |
| `layers.14.engram.q_weight`   | F32  | `[4, 5120]`     | 81,920 |
| `layers.14.engram.k_weight`   | F32  | `[4, 5120]`     | 81,920 |

## Manifest update: `engram/engram-manifest.json`

- Size grew 7,516 → **12,903 bytes**; new top-level `residents` key inserted after `layers`.
- `residents.file = engram-residents.safetensors`, `residents.total_bytes = 334562304`,
  `residents.sha256 = 5dedcbc21708d3617c41df276c218d819a4e81003db666e3f1b8dc9c4334951a`.
- New `manifest_sha256 = 36ee1d96271b261854aebd917a79c14497aea8f8f01dbc77d7b7828996fef723`
  (recomputed with `write_manifest`'s recipe; re-derived independently in the tests).
- The `residents` entry carries `file`, per-tensor `{name, dtype, shape}`, `total_bytes`,
  `sha256`, the `quant`/`dequant` blocks, and per-layer source provenance + parity.
- Pre-update manifest backed up to
  `…/scratchpad/engram-manifest.json.bak-2026-09-10` (outside the artifact, as required).

## Verification — dequant-roundtrip parity (q8 wkv vs the FP8 source)

Cosine + max-abs error of `mx.dequantize(q8(wkv))` vs the exact `dequant_fp8_block` of the
FP8 source, over the full `[25600, 6144]` matrix:

| Layer | cos (per-row min) | cos (flattened) | max-abs err | mean-abs err | q/k exact |
|---|---|---|---|---|---|
| 1  | **0.999959** | 0.999974 | 3.9062e-03 | 2.7470e-05 | yes (max_abs_err = 0.0) |
| 14 | **0.999956** | 0.999972 | 7.8125e-03 | 3.0328e-05 | yes (max_abs_err = 0.0) |

Per-row min cosine ≥ 0.9999 on both layers (gate met). `q_weight`/`k_weight` reproduce the
BF16 source exactly (widened to F32; `max_abs_err = 0.0`).

Further checks (in `tests/test_engram_residents.py`, run against the real sidecar):
- sidecar loads with `mx.load`; `load_engram_residents` returns the expected shapes/dtypes.
- the loader's `wkv` callable == dequantize-then-matmul (flattened cosine ≥ 0.999; the tight
  max-abs equality is bf16 accumulation, not a wiring difference).
- `EngramV41` is constructed from the sidecar + manifest + on-disk row bank
  (`EngramResidents.build_module(row_cache=EngramBank.open(...).cache, …)`) and runs a finite
  forward.
- `residents.sha256` equals the on-disk file sha256, and `manifest_sha256` re-derives.

## Run facts (CPU only, `nice -n 19`, no GPU/Metal)

- Source shards read: `model-00047-of-00048.safetensors`, `model-00048-of-00048.safetensors`
  (only the 4 resident tensors per shard via `os.pread` by offset — ~157 MB + ~154 KB + 2×40 KB
  per layer; the 95 GiB shards are never mmapped/loaded whole).
- Peak memory: **maximum RSS 5,491,916,800 B ≈ 5.11 GiB**, peak footprint ≈ 5.10 GiB
  (`/usr/bin/time -l`). Well under the 10 GB ceiling.
- Wall time: ~14 s cold, ~2 s warm. Deterministic: a re-run reproduced the identical sidecar
  sha256 and `manifest_sha256` (idempotent — the `residents` entry is replaced, never duplicated).
- `mx.quantize`/`mx.dequantize` on the CPU stream (`mx.set_default_device(mx.cpu)`); no GPU
  flock held or needed.

## Tests

Command (from the worktree, venv python, `nice -n 19`, no `-n auto`):

```
PYTHONPATH=<worktree> nice -n 19 .venv/bin/python3 -m pytest \
  tests/test_engram_residents.py tests/test_convert_deepseek_v41_engram.py -v
```

Tail:

```
tests/test_engram_residents.py ........                                  [ 50%]
tests/test_convert_deepseek_v41_engram.py ........                       [100%]

============================== 16 passed in 3.65s ==============================
```

(`tests/test_engram_v41.py` is a W2 file whose import-time guard pins it to the
`.worktrees/dsv41-w2` checkout, so it fails *collection* in this worktree by design — nothing
to do with these changes, and it is outside the W4 allowlist.)

## Not done here (David's call)

The HF repo **`OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2` does NOT yet have this
sidecar** — `engram/engram-residents.safetensors` and the updated `engram-manifest.json` were
written only to the local artifact. Uploading them is David's decision; W4 pushed nothing and
opened no PR.
