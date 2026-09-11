# W40 / K21 — Output-head codec lever (`MTPLX_DSV41_HEAD_MODE`)

Worker `feat/deepseek-v41-w40`. Opus 4.8, CPU-only (no GPU / Metal; the head math
is validated on CPU with `mx.set_default_device(mx.cpu)`). Landed as a runtime
lever + exactness tests; the realized decode ms/token is a GPU-window A/B (arms
`head_bf16` / `head_mxfp8` / `head_q8` vs `control`), **not measured by this
worker**.

## 1. Root cause of the 70.9 ms/token `head` stage

GPU window 13 (`receipts/gpu-windows/window-13/stage-timing-1024.json`, per-stage
`mx.eval` fences, 64 decode tokens at 1,024 context):

| stage | ms/token | share of the 348.5 ms fenced decode frame |
|---|---:|---:|
| `moe.routed_switch` | 101.6 | 29.2 % |
| **`head`** | **70.9** | **20.3 %** |
| `attn.reuse` | 50.2 | 14.4 % |
| `hc.premix_sinkhorn` | 49.7 | 14.3 % |

The head is `nn.Linear(hidden 5120, vocab 129280, bias=False)`, kept **dense BF16**
by the native-codec quant predicate (`deepseek_v41.py` `_make_resident_quant_predicate`,
`path == "head"` excluded under a native float mode). BF16 `head.weight` =
129280 × 5120 × 2 B = **1.324 GB** (verified from the artifact safetensors header).
A clean BF16 M=1 GEMV over 1.324 GB is ~2 ms at 614 GB/s. **70.9 ms is ~35× that.**

### The dtype at every hop into the head (read from the code, not guessed)

1. `embed_tokens` is BF16 dense (native codec) → the residual stream `h` is **bf16**.
2. Backbone final collapse + norm (`_forward_span`, ~L1412 / `final_norm`):
   `h = mx.sum(pre_mix[..., None] * h.astype(float32), axis=2).astype(h.dtype)` —
   the HC merge computes the weighted sum in float32 but **casts back to `h.dtype`
   (bf16)**; then `_rmsnorm` (`deepseek_v41.py:381`) normalises in f32 and returns
   `.astype(dtype)` = **bf16**. So `out` (the backbone return) is **bf16**.
3. `Model.__call__` head site (was `deepseek_v41.py:1973`):
   ```python
   source = h if keep_last is None else h[:, -keep_last:, :]   # bf16
   logits = self.head(source.astype(mx.float32))               # <-- the trap
   ```
   `source.astype(mx.float32)` **explicitly upcasts the bf16 hidden to float32**,
   so the head matmul is `f32 @ bf16.T`. MLX has no mixed-precision matmul: it
   promotes the smaller operand, i.e. it **materialises a float32 copy of the
   1.324 GB bf16 head weight — a 2.648 GB temporary — every decode token**, then
   runs an f32 GEMV over it. (Confirmed on CPU: `f32 @ bf16.T -> float32`.)

**This refines the task's hypothesis.** `h` does *not* reach the head as float32
(the HC merge casts back to bf16); the float32 promotion is caused **solely by the
explicit `source.astype(mx.float32)` at the head call site.** It is the exact twin
of Hy3's "lm_head fp32-cast trap" ([[hy3-resident-path-breakthrough]]: "lm_head
fp32-cast trap + resident-q4 + islands").

### Byte traffic per token (M=1 GEMV)

| path | weight-related DRAM traffic / token | vs control |
|---|---:|---:|
| **control** (f32-cast trap) | read bf16 1.324 + write f32 temp 2.648 + read f32 GEMV 2.648 = **6.619 GB** | 1.0× |
| `bf16` (cast hidden to weight dtype) | read bf16 1.324 (+ f32 logit upcast, 0.5 MB) = **1.324 GB** | **5.0× less** |
| `mxfp8` (native gs32, E8M0) | packed 0.662 + scales 0.021 = **0.683 GB** | **9.7× less** |
| `q8` (affine gs64) | packed 0.662 + scales+biases 0.041 = **0.703 GB** | **9.4× less** |

The measured 70.9 ms sits far above even the 6.619 GB / 614 GB/s = 10.8 ms a pure
bandwidth model predicts, so the fenced head stage also carries the f32
materialisation's allocator churn and the in-window effective bandwidth (the
per-stage `mx.eval` fence inflates absolute ms — the fenced frame is 348.5 ms/token
vs the unfenced 268 ms/token = 3.73 tok/s in the same receipt). The **byte ratios
above are the robust, shape-independent signal**; the absolute ms/token saved is a
KG-class GPU-window A/B.

## 2. The lever

`MTPLX_DSV41_HEAD_MODE`, resolved once at model construction
(`_resolve_head_mode`, read at load not frozen at import) and the weight repack
applied once **post-load** (`Model.apply_head_mode`, called by
`construct_deepseek_v41_resident_model` after the real bf16 head weight is loaded —
the head weight does not exist yet at `__init__`):

- **unset / `default`** — current behaviour, byte-for-byte (`assert array_equal`
  in the test). The f32-cast trap is preserved as the control.
- **`bf16`** — forward-only: cast the hidden to the head weight dtype (bf16)
  before the matmul, f32 logits after. No weight change, no resident-footprint
  change; the win is the 5× per-token traffic cut. The only numeric change is
  bf16-rounding the hidden.
- **`mxfp8`** — repack the head at load to native mxfp8 gs32 (`mx.quantize(mode=
  "mxfp8", group_size=32)`; 32 is mlx 0.32.2's only mxfp8 group size) → `_MXFP8Head`
  runs one `mx.quantized_matmul(mode="mxfp8")` per call. Resident head 1.324 → **0.683 GB**.
- **`q8`** — repack the head to affine 8-bit gs64 via `nn.QuantizedLinear.from_linear
  (group_size=64, bits=8)`. Resident head 1.324 → **0.703 GB**.

The quantisation happens **once** at load, never per call. An already-quantised
head (the affine artifact quantises the head to q8 at load, so it carries
`.scales`) or a non-float head is a no-op and the forward falls back to the default
path. An unrecognised `MTPLX_DSV41_HEAD_MODE` value raises (a mistyped accuracy
lever fails fast rather than silently running control through a benchmark window).

### Memory pricing

`apply_head_mode` returns a resident-pricing note the loader merges into
`model._mtplx_resident_load_report` (it does **not** touch `raw_tensor_bytes`,
which correctly prices the bf16 head *read from disk*):

```
{"head_mode": "mxfp8",
 "head_resident_bytes_default": 1_323_663_360,   # 1.324 GB bf16
 "head_resident_bytes_actual":  682_598_400,     # 0.683 GB
 "head_resident_saved_bytes":   641_064_960}      # ~0.64 GB freed
```

`bf16` reports `saved_bytes = 0` (resident unchanged; the win is per-token traffic).

## 3. Exactness (CPU, `tests/models/test_deepseek_v41_head_lever.py`, 12 tests)

Head shape for the argmax test: real **hidden = 5120** (the true per-row dot-product
magnitude *and* the true group counts — mxfp8 gs32 → 160 groups/row, q8 gs64 → 80),
vocab shrunk 129280 → 4096 to fit the worker's 3 GB RSS cap (per-row quantisation is
identical whatever the row count; peak RSS 0.32 GB). 256 random hidden vectors, seed 7.
Reference = the default f32-cast path.

| codec | max\|Δ\| vs f32 ref | greedy-argmax match | flipped positions' reference top-1 margin |
|---|---:|---:|---|
| `bf16` | 0.0278 | **249 / 256** | all ≤ 0.0452 (≤ 1.63× max\|Δ\|) |
| `mxfp8` | 0.1877 | 239 / 256 | all ≤ 0.0917 |
| `q8` | 0.0537 | 254 / 256 | all ≤ 0.018 |

Reference top-1 margin percentiles `[0,10,20,50,90,100] = [0.0024, 0.031, 0.0822,
0.2555, 0.8333, 1.916]`.

**Rigorous, seed-robust invariant (asserted, verified across seeds 7/11/23):** every
codec flips the greedy argmax **only** where the reference top-1 margin sits within
~3× that codec's own worst logit perturbation `max|Δ|` — i.e. only at genuine
near-ties, never at a clear winner. This is the precise form of "changes logits only
by `<codec>` rounding". A random head with random hidden vectors is a **worst case**:
its logits are near-uniform (20th-percentile margin 0.082), so bf16 input rounding
tips 7 sub-ulp ties; a trained head with a dominant top-1 tips none. `bf16` is the
tightest (input-only rounding), `mxfp8` the coarsest (E8M0 gs32 scales, no per-group
bias) — matching the byte/precision trade each codec exposes.

- **default codec bit-identical**: `test_default_forward_is_bit_identical_to_fp32_cast_path`
  reproduces `self.head(h.astype(float32))` byte-for-byte through a tiny MTP model
  (`mx.array_equal`).
- **DSpark MTP verify (K+1 rows) works in every mode**:
  `test_mtp_verify_rows_work_in_every_mode` runs a 4-row (K+1) batch through a tiny
  MTP model in `default`/`bf16`/`mxfp8`/`q8` and asserts `[1, 4, vocab]` f32 logits,
  all finite, in every codec; `bf16` keeps the verify argmax identical to default.
- **`apply_head_mode` pricing + idempotence + no-op on an already-quantised head**
  are unit-tested directly.

### Scope note: the DSpark draft shares the output head

`Model.__call__` (the AR decode + the **MTP verify** K+1-row forward) is the 70.9 ms
stage and is fully routed through `_apply_head`. The DSpark **draft** reuses the same
trunk output head (`draft_mtp` passes `self.head` into `DSparkHead.draft_block` →
`forward_head`, `deepseek_v41_dspark.py:441` `head(_rmsnorm(...).astype(float32))`),
which carries the **same f32-cast trap** at its own call site. The lever handles this
safely without touching that line:

- `mxfp8`/`q8` — the head is already the quantised module, so the draft's
  `head(f32)` runs `quantized_matmul` (no f32 weight promotion); correct, and the
  draft's slightly different logits only shift the **acceptance rate**, never
  correctness (the greedy target verify is authoritative — DSpark "draft numerics
  set only the acceptance rate").
- `bf16` — the draft keeps the trap (its head stays a dense bf16 Linear and the
  draft call still upcasts to f32). This is out of scope for this worker (the target
  is the measured AR/verify head stage); fixing the draft call site is a trivial
  follow-up if the draft head ever shows up in a draft-stage timing.

## 4. Expected ms/token saved (first-order, to be confirmed in a GPU window)

If the head stage is DRAM-bandwidth-bound (its 6.6 GB/token traffic and the 35×
gap over a clean bf16 GEMV both say it is), scaling the measured 70.9 ms fenced head
stage by the byte ratios:

| codec | traffic ratio | est. head stage | est. Δ vs control |
|---|---:|---:|---:|
| `bf16` | 0.20× | ~14.2 ms | **−56.7 ms/token (−80 %)** |
| `mxfp8` | 0.103× | ~7.3 ms | **−63.6 ms/token (−90 %)** |
| `q8` | 0.106× | ~7.5 ms | **−63.4 ms/token (−90 %)** |

The head is ~20 % of the fenced decode frame, so removing ~80–90 % of it is a
first-order ~−16–18 % of the fenced frame — most of it captured by the **free,
lossless `bf16` arm** alone; `mxfp8`/`q8` add a further ~7 ms head cut *and* free
~0.64 GB resident, gated by the near-tie argmax cost above. Confirm with the GPU A/B
(`scripts/deepseek_v41/ab_decode_env_levers.py --arms control head_bf16 head_mxfp8
head_q8 --stage-timing`) at [[dsv41-standard-benchmark-shape]], in-window control,
byte-identity **not** expected for the head arms (they are lossy by design — the
harness's identity summary flags them; that is expected, see §exactness).

## 5. Recommendation

Ship the **`bf16` arm as the default fix** — it removes a pure implementation bug
(the unnecessary `astype(float32)` that drags the bf16 weight to a 2.6 GB f32
temporary every token) at zero quality cost beyond bf16-rounding the hidden, and its
argmax flips are confined to the bf16 near-tie envelope. Keep `mxfp8`/`q8` as
opt-in resident-footprint levers to be gated on a full HumanEval/MBPP task eval
([[task-evals-decide-bank-verdicts]]) before serving, since they widen the near-tie
flip rate (mxfp8 most). This supersedes K9's premise (which priced the head at a
clean 1.32 GB bf16 read, ~1 ms, +2–4 %): the real head is the f32-cast trap at
6.6 GB / ~70 ms, so the `bf16` fix alone is worth far more than K9 estimated.
