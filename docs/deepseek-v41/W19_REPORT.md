# W19 — Engram row banks to MLX-native mxfp8 (exact repack)

Branch `feat/deepseek-v41-w19` off `feat/deepseek-v41-streaming`.
Scope: convert the DeepSeek-V4.1-Flash **Engram row banks** (`layers.{1,14}.engram.embed`)
from the affine-q8/gs64 record format to MLX's native **mxfp8** format, as an EXACT byte
repack of the Hub source. The affine codec stays intact; the runtime picks the codec from the
manifest.

**Status:** converter mode + runtime codec committed with tests green for both codecs; the
2,000,000-row layer-1 pilot is bit-exact through the real `NGramRowCache` path. Ready for the
full 768,022,850-row rewrite (one command below).

---

## 1. Why mxfp8 is an exact repack (and why the bank shrinks 272 → 264 B/row)

The source rows already *are* mxfp8: `embed.weight` is F8_E4M3 code bytes and `embed.scale` is
F8_E8M0, one scale per 32-column group (`layers.{1,14}.engram.embed.*`, shards 47/48). MLX's
`mxfp8` quantization mode is exactly that layout — group size 32, 8-bit E4M3 elements, one E8M0
scale per group, no bias. So the conversion keeps the source bytes verbatim:

| record | layout | bytes/row |
|---|---|---|
| affine (old) | `U32[64]` levels + `BF16[4]` scales + `BF16[4]` biases | **272** |
| mxfp8 (new)  | `U32[64]` E4M3 code bytes @0 + `U8[8]` E8M0 scale bytes @256 | **264** |

`mx.dequantize(w, scales, group_size=32, bits=8, mode="mxfp8")` on those bytes reproduces the
reference FP32 dequant `E4M3_LUT[code] * 2**(scale_byte-127)` **bit-for-bit** — verified over the
whole E4M3 byte range and on real source rows (both `bfloat16` default output and `dtype=float32`
output are exact for real data; the products are exactly representable). The affine codec had to
*requantize* the FP8-dequantized values, so it was lossy (that is why W19 exists); mxfp8 carries
the source through untouched.

### Byte math (full banks)

Rows: L1 = 384,006,168; L14 = 384,016,682; total = **768,022,850**.

| bank | affine @272 B | mxfp8 @264 B | saved (8 B/row) |
|---|---:|---:|---:|
| L1  | 104,449,677,696 | 101,377,628,352 | 3,072,049,344 |
| L14 | 104,452,537,504 | 101,380,404,048 | 3,072,133,456 |
| **total** | **208,902,215,200** (194.55 GiB) | **202,758,032,400** (188.83 GiB) | **6,144,182,800** (5.72 GiB, −2.94%) |

---

## 2. What changed

Runtime codec (both codecs load; the manifest `quant.mode` selects one):

- `mtplx/ngram_row_cache.py` — `RowGeometry` gains `mode="mxfp8"`: group_size 32, bits 8, E8M0
  (uint8) scales, **no bias**; `row_bytes` 264; `dequantize()` feeds codes + scales straight into
  `mx.dequantize(mode="mxfp8")`. `mode="affine"` unchanged.
- `mtplx/engram_bank.py` — `EngramBank` reads `quant.mode` from the manifest, parses the no-bias
  layout, and its numpy `dequantize_rows` reference is EXACT for mxfp8
  (`E4M3_LUT[code] * 2**(scale-127)` per 32-col group). `gather()` returns `(codes, e8m0_scales,
  None)` for mxfp8.
- `mtplx/engram_v41.py` — `open_engram_row_cache(engram_dir, layer)` builds the resident cache
  with the manifest's codec, so `EngramV41` needs no codec awareness of its own.

Converter (`scripts/convert_deepseek_v41_engram.py`), affine path untouched:

- `--row-codec {affine,mxfp8}` (default `affine`). `mxfp8` is a pure byte repack
  (`_mxfp8_chunk_records`): read the source E4M3 row (256 B) + E8M0 scales (8 B), concatenate,
  write. No dequant/requant.
- Streaming, resumable, ≤ ~200 MB RSS. In-progress bytes go to `engram/engram-L{L}.bin.new`
  **inside the artifact** (same filesystem, so the final rename over the affine bank is atomic);
  the resume journal lives **outside** the artifact under `--state-dir`.
- `main()` stages every layer (`finalize=False`), then flips **all** banks over their affine
  predecessors and rewrites the manifest **together** (`finalize_mxfp8` + `write_manifest`), so
  the artifact never rests half-converted.
- Manifest: top-level `quant {bits:8, group_size:32, mode:"mxfp8", record_bytes:264}`, a no-bias
  `record_layout` (`weight U32[64] @0`, `scales U8[8] @256`), per-bank `sha256`, and
  `source.exact_repack: true`.

Tests (green for both codecs, CPU, `nice -n 19`, no `-n auto`):
`tests/test_ngram_row_cache.py` (mxfp8 geometry + bit-exact dequant), `tests/test_convert_
deepseek_v41_engram.py` (layout constants, full convert+reader bit-exact through the MLX cache,
manifest fields + sha, resume/idempotence/staging, both manifests side-by-side),
`tests/test_engram_v41.py` + `tests/test_engram_residents.py` repointed off the deleted q2 path to
the mxfp4 artifact (`$DSV41_ARTIFACT_DIR` override, mxfp4 default).

```
tests/test_ngram_row_cache.py  tests/test_convert_deepseek_v41_engram.py   25 passed
tests/test_engram_residents.py  test_row_dequant_parity_real_bank           9 passed (real mxfp4 affine bank)
```

---

## 3. Pilot (2,000,000 rows of layer 1)

`~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/engram-pilot/engram-L1.bin` (528,000,000 B) +
`engram-manifest.json` (`mode mxfp8`, `record_bytes 264`, per-bank sha256). Conversion peak RSS
197 MB.

**(a) bit-exact vs the source FP32 dequant** — 5,000 sampled rows (incl. row 0/1/2 and the last
two), 1,280,000 values, `np.array_equal`: **PASS**.

**(b) through the real runtime path** — `EngramBank.open(pilot).cache.dequantize(row_ids)` (the
`NGramRowCache` → `RowGeometry(mode="mxfp8")` → `mx.dequantize(mode="mxfp8")` path
`EngramV41.__call__` uses) is bit-exact; so is `engram_v41.open_engram_row_cache`. The stored
codes and scales equal the source bytes verbatim. **PASS**.

**(c) throughput + projection** — measured on the artifact filesystem (`/dev/disk3s5`), `nice -n 19`:

| measurement | rate |
|---|---|
| pilot conversion record write (warm L1) | ~2,785 MB/s |
| cold slice, layer 14 (shard 48, untouched) | ~1,985 MB/s |
| sustained cold sequential read (8 GiB) | 11.70 GB/s |
| sustained buffered write + fsync (8 GiB) | 10.27 GB/s |
| `hashlib.sha256` (in-RAM) / `_hash_file` over the 528 MB pilot | 3.29 / 2.73 GB/s |

Full-run I/O = read source (202.76 GB, `weight+scale` == output) + write output (202.76 GB) + a
sha256 read-back of each finalized bank (202.76 GB @ ~2.73 GB/s ≈ 74 s). Projected **wall time for
the full 768M-row rewrite of both layers: ~3–5 minutes** on an idle SSD (conversion I/O ~40–60 s +
sha pass ~75 s), scaling the cold pilot linearly gives ~3.8 min. Contention from the concurrent
mxfp4 expert-bank write will stretch this proportionally; the job is I/O-bound with a fixed
~200 MB RSS regardless.

**(d) per-token gather cost vs affine** — 24-row (`n_hash_cols`) gather+dequant per token, CPU,
identical 200,000-row synthetic banks, medians over 1,200 trials (order-independent):

| codec | MISS (cold 24 rows) | HIT (resident 24 rows) |
|---|---:|---:|
| affine | 146.8 µs/token | 89.1 µs/token |
| mxfp8  | 120.0 µs/token | 60.0 µs/token |

mxfp8 is **not a regression** — it is ~18% (miss) / ~33% (hit) cheaper, because the record is 264
vs 272 B and the dequant has no bias array to view/add.

---

## 4. Disk requirement for the full rewrite

The `.bin.new` files are staged alongside the affine banks, so both coexist until the atomic
rename:

- Additional disk to allocate at launch: **202.76 GB** (2 × ~101.38 GB `.bin.new`).
- Peak engram-bank bytes on disk during the run: 208.90 (affine) + 202.76 (mxfp8) = **411.66 GB**.
- After the two `os.replace` renames, the affine bytes are overwritten in place; end state =
  **202.76 GB** (6.14 GB less than the affine banks it replaces).
- Free now: ~615 GiB. Even with the concurrent ~267 GB mxfp4 expert-bank write, > 200 GiB remain.

Nothing else is touched: the q2 artifact is already gone; `experts.bin`, resident shards, the
`engram-residents.safetensors` sidecar, and `engram-manifest.json`'s `residents`/`hashing` blocks
are untouched until the single manifest rewrite at the end.

---

## 5. One-command full run (launch when disk is approved)

```bash
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w19 \
  nice -n 19 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w19/scripts/convert_deepseek_v41_engram.py \
  --mode banks --row-codec mxfp8 --layers 1,14 \
  --src   /Users/davidtai/models/DeepSeek-V4.1-Flash-src \
  --out   /Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/engram \
  --index /Users/davidtai/models/DeepSeek-V4.1-Flash-src/model.safetensors.index.json \
  --state-dir /Users/davidtai/models/dsv41-convert-state/engram-mxfp8
```

Behavior: stages `engram-L1.bin.new` + `engram-L14.bin.new` in the artifact `engram/` dir
(resumable via the journal under `--state-dir`, outside the artifact), verifies each size + sha256,
then atomically renames both over `engram-L{1,14}.bin` and rewrites `engram-manifest.json`
(`mode:"mxfp8"`) in one step. Re-running after an interrupt resumes from the journal; re-running
after completion early-skips. To roll back before the flip, delete the `.bin.new` files — the
affine banks and manifest are untouched until the final step.
