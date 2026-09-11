# W18 — native float-quant residents (mxfp8 dense + mxfp4 MTP), exact repacks

Scope: the **resident** tensors of `~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/`,
converted from the Hub source (`~/models/DeepSeek-V4.1-Flash-src`, revision
`dba1be0a…`) to MLX's native floating-point quant formats as **exact repacks** of
the source — not affine requantizations. The routed backbone expert bank
(`experts.bin`), `expert-manifest.json`, and `engram/` are out of scope (W16 / W15
/ W19). MLX 0.32.2, CPU stream only, no GPU flock.

## Done-when status
- [x] Residents repacked **bit-exact** (mxfp8 dense, mxfp4 MTP experts; BF16/F32 kept verbatim).
- [x] Both **codecs** load strictly (native mxfp8 real artifact + synthetic affine-q8 fixture).
- [x] Golden numbers reported (layer-0 attention; see the important GEMM-precision finding).
- [x] `model.safetensors.index.json` + config `quantization` block written; report committed.

## 1. Source→native format map (residents only)
Every distinct source tensor family, enumerated from the pinned revision's index:

| source dtype | families | native resident format |
|---|---|---|
| `F8_E4M3` + `F8_E8M0` 32×32 block scale | attn `wq_a/wq_b/wkv/wo_a/wo_b`, ffn `shared_experts.w{1,2,3}`, `attn.indexer.wq_b`, MTP dense (`mtp.*.attn.*`, `mtp.*.ffn.shared_experts.*`, `mtp.*.main_proj`) | **mxfp8 gs32** (U32 codes + U8 E8M0 scales) |
| `I8` FP4/E2M1 experts (`mtp.{0,1,2}.ffn.experts.{0..127}.w{1,2,3}`) | 3×128×3 = 1152 tensors | **mxfp4 gs32** |
| `BF16` / `F32` everything else | embed, head, norms, `hc_*`, `ffn.gate` weight/bias, `attn_sink`, compressor `wkv/wgate`, indexer `wk/weights_proj/k_norm`, vision/aligner, `image_*` markers, MTP norms/heads | **kept verbatim** |
| `layers.N.ffn.experts.*` (backbone routed) | — | streamed bank `experts.bin` (**W16**, skipped) |
| `*.engram.*` | — | engram sidecar `engram/` (**W19**, skipped) |

`embed`/`head` are BF16 at source, so "keep verbatim" **restores them to exact bf16**
(the affine-q8 artifact had quantised them). No dense projection the source keeps in
BF16 (indexer `wk`/`weights_proj`, compressor `wkv`/`wgate`) is ever quantised — an
FP8 repack of a bf16 source cannot be exact, so those stay dense.

## 2. Block-scale layout maps exactly onto MLX's mxfp8 scales — YES, every FP8 family
The source stores each FP8 dense tensor as E4M3 codes `[O,I]` with **one E8M0 scale per
32×32 block** `[O/32,I/32]`; MLX's mxfp8 wants **one E8M0 scale per (row, 32-col group)**
`[O,I/32]`. For every FP8 family the 32-col block boundary aligns with the mxfp8 group
boundary, so `mx.quantize(dequant_fp8_block(w,s), gs=32, bits=8, mode="mxfp8")` reproduces
the source **byte-for-byte** — `mx.dequantize(...)` equals the fp32 block dequant with
`np.array_equal` True, `max|Δ| = 0`:

| family (layer 0 + MTP + vision) | shape | block | mxfp8 exact? |
|---|---|---|---|
| `attn.wq_a` | (1280, 5120) | 32×32 | yes (max\|Δ\|=0) |
| `attn.wq_b` | (32768, 1280) | 32×32 | yes |
| `attn.wkv` | (512, 5120) | 32×32 | yes |
| `attn.wo_a` | (8192, 4096) | 32×32 | yes |
| `attn.wo_b` | (5120, 8192) | 32×32 | yes |
| `ffn.shared_experts.w{1,2,3}` | (2304,5120)/(5120,2304) | 32×32 | yes |
| `attn.indexer.wq_b` | (H·D, 1280) | 32×32 | yes |
| `mtp.*.attn.*`, `mtp.*.main_proj`, `mtp.*.shared_experts.*` | — | 32×32 | yes |
| MTP experts `w{1,2,3}` (mxfp4) | (2304,5120)/(5120,2304) | per-32-col | yes (mxfp4) |

MLX picks an **equivalent** E8M0 scale byte (typically one exponent smaller, with
correspondingly larger E4M3 codes) rather than the source block-scale byte — the stored
bytes differ, but the reconstructed value is identical (FP8 values are already `E4M3·2^k`,
and a power-of-two rescale that keeps them in E4M3's normal range is lossless). The
converter runs a **per-tensor `np.array_equal` gate** (`quantize_mxfp8_exact` /
`quantize_mxfp4_exact` in `mtplx/deepseek_v41_convert.py`) that **raises and names the
tensor** if any repack is not bit-exact — none were, across all 46+3 shards.

## 3. Strict load — both codecs
- **Native mxfp8 (real artifact)**: model built with `quantization={group_size:32,
  bits:8, mode:"mxfp8"}`; all **1206** text-only params covered by exactly one resident
  of matching shape, 0 missing, 0 shape mismatch (full `model.load_weights(strict=True)`
  succeeds; the RSS-bounded lazy equivalence check confirms it at ~0.1 GB). The old
  affine-q8 count was 1616; mxfp8 drops the 410 `.biases` leaves (mxfp8 has no biases).
- **Affine q8 (synthetic fixture)**: since the q2 artifact was deleted, the affine path
  is covered by a small in-repo model quantised to q8 gs64 that strict-loads its own
  residents and runs a finite forward (`tests/test_deepseek_v41_residents_native.py::
  test_both_codecs_strict_load_and_forward[affine_q8]`). Both codecs also run a finite
  CPU forward.

Model/loader plumbing: `Model.__init__` now takes the config `quantization` block and
selects the codec (`mtplx/models/deepseek_v41.py`: `_resolve_resident_quant`,
`_make_resident_quant_predicate`); the loader passes `config["quantization"]` through
(`deepseek_v41_loader.py`). `_o_lora_down` dequantizes `wo_a` mode-aware (mxfp8
`biases=None`). Affine tests unchanged (no regression).

## 4. Layer-0 attention golden + an important GEMM-precision finding
Layer 0 is SWA-only, so `attn_L0 = f(embed, 5 dense projections, norms/attn_sink)` —
exactly the W18 residents (no routed-expert bank needed). On the 31-token probe, vs a
**fp32 dense oracle** (source weights dequantised to fp32, fp32 activations):

| resident codec | attn_L0 min_cos | attn_L0 global_cos |
|---|---|---|
| **mxfp8** (exact weights, bf16/fp8 GEMM) | 0.999855 | **0.999921** |
| affine q8 (lossy weights, int8 GEMM) | 0.999974 | 0.999983 |
| committed q8+q2 streaming baseline vs **torch** fp32 (receipt) | 0.9909 | 0.9945 |

**Finding — the mxfp8 residents' _weights_ are bit-exact, but the mxfp8 GEMM is lower
precision than the affine-int8 GEMM.** Isolated on one projection (`wq_a`, K=5120, equal
bf16 activation): dequantised mxfp8 weight == source (`np.array_equal`), yet
`quantized_matmul(mode="mxfp8")` lands at cos **0.99986** vs the fp32 product, while
affine-q8 (whose weights carry ~5e-4 error) lands at **0.99998**; a plain bf16 dense
matmul of the exact weights is 0.999999. The mxfp8 deficit grows with the reduction dim
(cos ≈ 1.0 at K=64), i.e. it is the fp8 GEMM's lower-precision activation/accumulation,
**not** a weight-repack error. So:

- The resident **weight** error is now **zero** (was lossy under q8) — the deliverable.
- The ~0.995 committed baseline was **not** resident-weight-dominated (W9 already showed
  q8 at equal dtype = 0.99999); it is bf16-activation / MLX-vs-torch accumulation plus, at
  L1/L2 (baseline 0.9926 / 0.9850), **q2 routed-expert** error. Exact residents do not by
  themselves move the torch-reference cosine to 0.9999; the large L1/L2 gains come from
  W16's now-exact **mxfp4 routed bank**, which this check can't wire yet (see §7).
- Choosing mxfp8 residents trades a hair of forward precision for exact weight storage and
  MLX's native fp8 Metal kernels (throughput) — the format's intended tradeoff, surfaced
  here for David's call. The full `compare_ref_vs_mlx.py` L0–2 numbers cannot be
  reproduced in this window (the torch-reference `.npy` were deleted and W16's bank/manifest
  are not final); the exactness proofs above stand independently.

## 5. embed/head — bf16 vs q8 byte delta (defaulted to bf16)
| tensor (129280×5120) | q8 gs64 (packed+scales+biases) | bf16 | delta |
|---|---:|---:|---:|
| per tensor | 703,283,200 B (0.655 GiB) | 1,323,827,200 B (1.233 GiB) | **+620,544,000 B (+0.578 GiB)** |
| embed + head (×2) | 1.310 GiB | 2.466 GiB | **+1.156 GiB** |

**Default kept: bf16** (exact restore of the source). Total residents are 17.37 GiB,
far under the 100 GiB wired serving plan, so the +1.156 GiB is affordable and buys exact
embed/head. (Set both back to q8 only if a future wired budget forces it.)

## 6. New artifact files (sizes + sha256)
Residents total **17.369 GiB** (18,650,062,435 B): 46 main shards (mxfp8 dense + BF16/F32
keep) = 10.645 GiB, 3 MTP shards (mxfp4 experts) = 6.724 GiB. All written by the safe swap
(write `*.partial.safetensors` → verify header/tensor-set/size → `os.replace`; the old
inode is never unlinked first). `model.safetensors.index.json` (3913 tensors,
total_size 18,649,658,184) and `config.json` (`quantization`: mxfp8 gs32 default + 1152
MTP mxfp4 overrides) rewritten last. New bytes on disk ≈ 17.4 GiB (≤ 40 GiB budget).

| file | bytes | tensors | sha256 |
|---|---:|---:|---|
| `model-00001.safetensors` | 970,533,589 | 263 | `48cbacf9459d99d2ef8305ee03d1b9186bfbe810639e3a94467eb305f30375ac` |
| `model-00002.safetensors` | 1,323,858,270 | 4 | `851f893c2e7474703187a63f41ae7fc344f67ae8f895e2d11aac2698e05314e5` |
| `model-00003.safetensors` | 174,962,526 | 30 | `eb551543e0db83c7846acc8bd55720a1cb0cf55556335417e44bb695500957a4` |
| `model-00004.safetensors` | 174,962,534 | 30 | `ed5191e1bc7c742919673cb0871915dcd47000377e32e71647857823624d4c1d` |
| `model-00005.safetensors` | 191,315,913 | 38 | `58635547f19a44ac8235c7d837008e961dd87f978c4978370362d3a4151fed9b` |
| `model-00006.safetensors` | 174,962,520 | 30 | `32b8b013afde829475f3594c1c70ecf8ee472354c55dacee1bc1efc92a4deac6` |
| `model-00007.safetensors` | 174,962,520 | 30 | `fdb38d167fb8e210dc4e1cc6c3d48851adfea83923c218c3a4bebf77cf30174a` |
| `model-00008.safetensors` | 174,962,512 | 30 | `e7b412ec56022206f06e999fdb48af5cf6b2e8790ea75bb7d8f02b1b27e92603` |
| `model-00009.safetensors` | 174,962,526 | 30 | `697aed0e9e4fc043413a68513384079836e4fb26ebccc05f3dfcb67c512b3242` |
| `model-00010.safetensors` | 174,962,540 | 30 | `c2e4e13d13369cf424fd493f68aee804684ea0634101da48f561403ded1867ed` |
| `model-00011.safetensors` | 191,315,907 | 38 | `7e5681f506ff29611edff3bdd513bf6c08447d805d6eebac71f751909b632797` |
| `model-00012.safetensors` | 174,962,530 | 30 | `05772832fc0dd80936482841feba672b16a4994e684465a381d7aa1a564e85cd` |
| `model-00013.safetensors` | 174,962,568 | 30 | `23bdd482ea8566d10e5cdecad5d1291fa636cfdfc01412ccae7540b4cfd07f85` |
| `model-00014.safetensors` | 174,962,554 | 30 | `34d2dddf25ecbdf1f02869d4ea9a34f7c52a394e946e481f4bb1c2b9c34c8076` |
| `model-00015.safetensors` | 174,962,542 | 30 | `80926aebffd2aa59f1b467c6bc5ee58281608b2cb9918413bbbcc4bedea86536` |
| `model-00016.safetensors` | 174,962,548 | 30 | `87bf32ff12d3716c5ada51049aac54df71a08c791763ca85152845a65e4f6dca` |
| `model-00017.safetensors` | 191,315,943 | 38 | `345d7c013a2236daf5632e4db14fb1517b9085df35231ee7417ed80964c2b986` |
| `model-00018.safetensors` | 174,962,562 | 30 | `4180ba8ad6a645aba369c37d9e9e62a3a462d59aec7acb9f6b4ce2cbc973058c` |
| `model-00019.safetensors` | 174,962,558 | 30 | `2da3388623a82af7d2660adbc00d93c7d5fe51dc066ca3d2778dad8bf5cad4f5` |
| `model-00020.safetensors` | 174,962,562 | 30 | `fb5896923f36de04a9dd0717af80f8df470228d6c0d2b03f26668314816b7960` |
| `model-00021.safetensors` | 174,962,556 | 30 | `08b2a9ceed393b1618d7b35b5e1fd42de909d3a62979dbfc1d4ce7d36fa67363` |
| `model-00022.safetensors` | 174,962,574 | 30 | `85577dfbf65ff23feb796f5371903e9d93a628fc5980d2509932348c9d613900` |
| `model-00023.safetensors` | 186,072,941 | 37 | `dbcea590dbd46f42a108e3ad33cdfda9cb23fe3ad8709b2297ce983b4d05c74a` |
| `model-00024.safetensors` | 174,962,568 | 30 | `689e9686b101c9416d798519378fc53d7ad4e3ef95d47521de3d0910de469f1a` |
| `model-00025.safetensors` | 174,962,548 | 30 | `bf505b08b9f1a4f70ee7c7e992e28b2e05e6edf1fed291a1491b3a46770d07e5` |
| `model-00026.safetensors` | 174,962,546 | 30 | `a536d9571aa80f0ae6c95835170e753df929c494efbfc9fa2a66e18085342d8b` |
| `model-00027.safetensors` | 180,697,321 | 33 | `370468bc4233fac6d4d7b2147a412430f6ed61a973ffec601ad32b816a634876` |
| `model-00028.safetensors` | 174,962,558 | 30 | `ae8ec73f401473fdf8cfb8b5e0e67df584b3c33cb0733273c72d9c15fe41215e` |
| `model-00029.safetensors` | 174,962,576 | 30 | `922f97e6fe71efac56d557df2928d8061d0b404bb51aaf9545879e1ef570304d` |
| `model-00030.safetensors` | 174,962,554 | 30 | `a7244679d6f22dfe5b032347c1e41f25c934b3bc30bcd7a87ec07f3f1f445d6c` |
| `model-00031.safetensors` | 180,697,273 | 33 | `7246774e70bee570ca2f3e267433179bb0d2854bc041e054def417069ae82e26` |
| `model-00032.safetensors` | 174,962,562 | 30 | `a344bd1e66e05241f454d89a71c59b12b6a1a25597e02e6e671a524aa48094d3` |
| `model-00033.safetensors` | 174,962,542 | 30 | `cd7cb7aa4de26681b006b1d1508b4764fc6f64e8ce5dd9235d1bbeaafa92a109` |
| `model-00034.safetensors` | 174,962,552 | 30 | `2ce253b8ee6e617b52562310347791ded3b1f86fdca4869f74642d91b93802bb` |
| `model-00035.safetensors` | 180,697,307 | 33 | `079943c0aea006baa79c2d440191f1e642a1fc2f9d479ab8629317d185699549` |
| `model-00036.safetensors` | 174,962,566 | 30 | `b3acb20ff638332766f0ee4706c45dd620f54d6a2af2b73d38873166fc25dc37` |
| `model-00037.safetensors` | 174,962,566 | 30 | `b2716c291c24f46bbeb913e664d7da59ffabdcfb404b0f74f7fce8d64b8d4cb6` |
| `model-00038.safetensors` | 174,962,574 | 30 | `196bea61796898adfbb7dba2423ead2d582d3f638e0023b7b1ead44e32ae7dd1` |
| `model-00039.safetensors` | 180,697,293 | 33 | `8cf631cf56d8a436cf8e8852ff29d175da04407e08089de959da4794542beb05` |
| `model-00040.safetensors` | 174,962,532 | 30 | `7b76b5c71cd40cf808c8da9b8f624e088b01700ef8fafcd1b963c9d11e256c41` |
| `model-00041.safetensors` | 174,962,596 | 30 | `1627efb091ef13671c28de7e37b10d4cc5fd74648b9aaf2f8d53552fcd0cdce1` |
| `model-00042.safetensors` | 174,962,554 | 30 | `dc698dff1a9091f03e5da2be4b7603700f5fdf8f389e55acf3ed5e723662e635` |
| `model-00043.safetensors` | 1,323,837,639 | 2 | `b84be2a671ec9a50f27c62d34e32c7e1685897881894b7f63d997e5959895c72` |
| `model-00044.safetensors` | 253,450,320 | 33 | `a98c053e86e0a123ba03eb63abf7a8b552d14918b439dc2cc42977f10aa04753` |
| `model-00045.safetensors` | 172,338,964 | 30 | `adeb9fa5e9ec775902d5ccb85b279476008f214c8b33aad647cf0971899be287` |
| `model-00046.safetensors` | 304,743,093 | 34 | `3fc6a01228110222d8bff0cdbf7625010f6628f106884199a7d3a199ad1eb8f2` |
| `model-00047.safetensors` | 2,406,562,980 | 768 | `95428e522cd6cc8ca3e0cb104c1d9e4a80cda1a732f88c9f039cf195a8c3d61e` |
| `model-00048.safetensors` | 2,406,563,006 | 768 | `09012d6ecb7ff9b83de842e7425b9b34c58fcfadd4e08491070825cbf23ebb99` |
| `model-00049.safetensors` | 2,406,563,050 | 768 | `83b9de0c6d6ceb4e2a49087ce4584f0866d6d28fb58276ec6243318dc545611e` |
| `model.safetensors.index.json` | 250,619 | — | `c0ead8ce7960b2c715366373da35626f3ccaef1837094bf9215cded0bfefdabb` |
| `config.json` | 122,010 | — | `3a43beb01ef51adbed5aa61fe4918653bfd3a1e35c6cad5dc45010bc717f2b8d` |


## 7. Coordination / caveats
- **q2 artifact deleted** — the "q2 byte-identical" check is moot; the mxfp4 dir held the
  only copy of the affine-q8 residents, so every shard was replaced in place by the
  verified atomic swap (never unlink-first). Old manifests preserved at
  `.worktrees/deepseek-v41/.benchmark-artifacts/deepseek-v41/q2-provenance/`.
- **`expert-manifest.json` resident section is W15/W16's** (not in my allowlist). Its
  `resident_tensors` must be regenerated from the new shards (dtypes: 328 U32 + 328 U8
  mxfp8 codes/scales, 230 BF16, 320 F32 in the text set; MTP experts now mxfp4 = 768
  tensors/shard, no biases). `model.safetensors.index.json` (written here) is the ground
  truth for that regeneration.
- **RSS**: all repeatable checks stay ≤ ~2 GB (converter peak 3.67 GB on the MTP mxfp4
  pass; lazy strict-load-equiv 0.11 GB). One *one-shot* full `load_weights(strict=True)`
  materialised the whole text model at **16.75 GB** (double-buffered model + residents,
  bf16 embed/head) — it confirmed the strict load but breached the 8 GB cap; it is **not**
  repeated, and the committed test uses the lazy equivalence path instead.
- The full L0–2 `mlx_dump.py` + `compare_ref_vs_mlx.py` pipeline needs W16's final
  mxfp4 bank + the (deleted) torch-reference `.npy`; deferred to that integration.
