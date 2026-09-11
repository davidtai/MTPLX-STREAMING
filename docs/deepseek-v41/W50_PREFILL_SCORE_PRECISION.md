# W50 — Prefill score-path precision + split-K online softmax (K25)

Branch `feat/deepseek-v41-w50`. Scope: the attention score/softmax/value path in
`mtplx/models/deepseek_v41.py` for **prefill (rows > 1)**. Env levers, all default
OFF, all prefill-only. No GPU/Metal executed here; all local evidence is CPU (MLX
0.32.2, `mx.set_default_device(mx.cpu)`), no artifact load, peak RSS 1.62 GB.

> **The sections below (Verdict … "What is NOT measured") are the pre-GPU W50
> write-up. Window-20 measured the arms and overturned the K25 roofline — read the
> "Window-20 follow-up" section next for the correction, the two regressions
> explained from the code, the `lean` pass-cut path, the SDPA finding, and the
> probe. The bf16/split-K roofline claims in the Verdict are superseded there.**

## Window-20 follow-up (the K25 roofline was WRONG — score stage is pass/bandwidth-bound)

Integration `7a789731d`, 16,384-token prompt, layer-major, 60 GiB plan,
`MTPLX_DSV41_PREFILL_CHUNK=1024`, **all tokens byte-identical**. Receipt:
`docs/deepseek-v41/receipts/gpu-windows/window-20/prefill-16384-ladder.json`.

| arm | TTFT | Δ vs baseline | peak GB |
|---|---|---|---|
| `layer_major` (baseline) | 352.8 s (46.4 tok/s) | — | 76.6 |
| `score_chunked` (KEY_CHUNK=2048) | 418.8 s | **−16 %** | 65.1 (**−11.5**) |
| `score_bf16` | 535.1 s | **−34 %** | 71.9 |
| `prefill_fast` (bf16 + chunk + dense) | 346.5 s | +1.8 % | — |

**The K25 FLOP roofline was wrong.** It predicted bf16 ≈ 2× on the QK^T/PV matmuls
(≈ −60 to −90 s). Instead bf16 **regressed −34 %** and split-K **regressed −16 %**
(while saving 11.5 GB). The score stage is **pass/bandwidth- and dispatch-bound, not
matmul-FLOP-bound** — so the lever is to cut *passes over the `[rows,64,T]`
transient*, not FLOPs.

### The two regressions, from the code

**bf16 (−34 %).** `_sparse_attend_oneshot` in the bf16 arm runs two full passes over
the `[rows,64,T]` transient that the f32 path never runs:
1. the QK^T output is bf16; the f32 softmax needs f32, so `scores.astype(float32)`
   is a **T-wide bf16→f32 cast** (read 2 B, write 4 B ≈ 6.4 GB at rows=1024, T=16 384,
   *per score call, per layer*);
2. before PV, `w.astype(bfloat16)` is a **T-wide f32→bf16 cast** (a second such pass),
   plus `KV.astype(bfloat16)`.
The matmul I/O halves to 2 B, but (a) the softmax/mask/scale stay f32 (the bandwidth
floor is unchanged) and (b) MLX 0.32.2's bf16 GEMM at **K=512, large N=T, transpose**
does not beat the f32 kernel on this shape. So the two cast passes are pure added
bandwidth → −34 %. *(bf16 stays as a measured-loser arm; not removed.)*

**split-K (−16 %, −11.5 GB).** The online-softmax **rescale of the running output**
is the cost. Per key chunk (`⌈T/n⌉ ≈ 8` at n=2048, T=16 384) `_sparse_attend_chunked`
does `acc = acc*corr + pv` where `acc` is `[rows,64,hd]` = **134 MB at rows=1024** —
that is **2 passes (multiply + add) over 134 MB every chunk**, ≈ 8× vs one-shot's
single accumulation, plus `corr = exp(m−m_new)` and the `denom` rescale, plus **≈8×
the kernel dispatches** for qk/mask/softmax/pv. The transient is `[rows,64,n]` not
`[rows,64,T]`, so the **−11.5 GB peak is real** → split-K is a **peak-GB lever, not a
throughput lever** (use it to admit a larger prefill chunk / longer context, not to
cut TTFT).

### `lean` — cut passes, keep f32 (`MTPLX_DSV41_PREFILL_SCORE_PATH=lean`)

The f32 pass-cut one-shot, the throughput play now that the stage is bandwidth-bound:
- **scale `q` once** (`[rows,64,512]`) instead of the scores (`[rows,64,T]`) → one
  fewer T-wide pass;
- **fold the value-0 sink into the denominator** (reference `_k_sparse_attn`, manual
  max/exp/sum) → no sink `concatenate` (`[rows,64,T+1]` alloc) and no post-softmax
  slice (`[rows,64,T]` copy); the normalize divides the `[rows,64,512]` output, not
  the T-wide `w`.

Net: **three fewer passes / two fewer T-wide allocations** per score call, all f32.
**Reassociation-level vs control** — real DSV4.1-shape (H=64, d=512, rows 128, T 4096)
max |Δ| = **1.2e-6** (rel 1.6e-6), **greedy-identical**; NOT bit-identical (scale-into-q
and normalize-after-PV reorder the float ops). Isolated CPU deltas: `fuse_scale` only
1.2e-6, `fold_sink` only 5.4e-7. Arms: **`score_lean`**, and **`prefill_lean`** =
`layer_major` + `prefill_dense_experts` (K26) + `lean` (the f32 successor to the
bf16-carrying `prefill_fast`).

### SDPA is not usable here (the per-head sink blocks it)

`mx.fast.scaled_dot_product_attention` **accepts head_dim 512 on CPU in MLX 0.32.2**
(it does not raise), but it has **no per-head value-0 sink**: plain SDPA matches a
no-sink softmax (Δ 1.8e-6) but differs from our sink semantics by **2.1e-3** — it
would change the attention output (and greedy tokens on near-ties). Splitting
head_dim into **2×256** is only a reassociated QK^T (max |Δ| 7.6e-5 vs full) that
still can't fold the sink into SDPA's fused softmax, so it buys nothing over `lean`.
**No SDPA arm is shipped** — it cannot be exact here.

### Probe: `attn_breakdown` decomposes `attn.<mode>.score`

`_sparse_attend*` now emit `stage_attn` sub-brackets — `attn.<mode>.score.{qk_matmul,
cast, scale_mask_sink, softmax, online_softmax, pv_matmul, combine, out_proj}` — into
a new `attn_breakdown` report section (parallel to `switch_breakdown`; kept OUT of the
flat partition sum so it never double-counts `attn.<mode>.score`). No-op and
byte-identical off the prefill stage-timing probe. This is what attributes the 153 s
reuse-score term in the next 16K window. (`deepseek_v41_stage_timing.py`: new
`stage_attn` channel + `_attn_sums`/`_attn_counts`.)

---

## Verdict

- `attn.*.score` is **201 s of the 370 s 16K TTFT** (W47: reuse 153 + reindex 23 +
  full 19 + swa_only 6; layer-major, 60 GiB plan) — the single largest prefill term.
  head_dim **512 is not an MLX fused-SDPA dim**, so the path materializes the full
  `[rows,64,T]` f32 score transient (6 GiB at the last 16K chunk) and the reuse
  matmuls run at **≈7 TFLOPS — f32-matmul territory; the M5 Max does ~2× that in bf16**.
- **`MTPLX_DSV41_PREFILL_SCORE_DTYPE=bf16`** runs the QK^T / PV matmuls in bf16
  (inputs cast; MLX accumulates in f32), softmax kept in f32. **Est. −60 to −90 s off
  the score term (TTFT 370 → ~290–310 s, −16 to −22 %)** — a GPU-window estimate.
  LOSSY by design: real-shape attention-output max |Δ| ≈ **1.8–3.9e-3** (rel 6e-3 to
  1.1e-2); greedy tokens can flip on near-ties.
- **`MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK=<n>`** is the gemma4 D512 two-pass split-K
  online softmax; caps the transient at `[rows,64,n]` (n=2048 → 0.54 GB, ~8× smaller).
  f32-exact up to reassociation: real-shape max |Δ| = **6.1e-7** (rel 2e-6),
  **greedy-identical on every row**. Compute-neutral; the win is peak GB.
- **f32 (default, unset) is byte-identical to the shipped path** — proven by
  `array_equal` at the op level and end-to-end (tiny-model logits).

## (1) Dtype audit — prefill score path

Everything below is `mtplx/models/deepseek_v41.py` unless noted. The score/softmax/
value path is `Attention._sparse_attend` (L660) → `_sparse_attend_oneshot` (L679) /
`_sparse_attend_chunked` (L706), reached from `_attend` at L910–911
(`attn.<mode>.score` stage).

| Quantity | Shipped (pre-W50) dtype | Where |
|---|---|---|
| q (into score) | activation dtype, RoPE (f32 cos/sin, L409) promotes the rope tail; **re-cast in the score fn** | L876–878 build; cast L683/685 |
| KV = window ⧺ compressed | activation dtype (post-RoPE window store L880–881); **re-cast in the score fn** | L911 arg; cast L683/685 |
| **QK^T inputs** | both cast to **f32** (`q.astype(f32)`, `KV.astype(f32)`) | L683 |
| **QK^T output** | f32 | L683 |
| scale (`head_dim**-0.5`, py float) | f32 multiply | L625, L690 |
| candidate/reach mask apply | f32 (`mx.where(attend, scores, -inf)`) | L691 |
| per-head sink (`attn_sink`, L638) | `attn_sink.astype(f32)`, concat, value-0 slot | L692–693 |
| softmax | **f32** (`mx.softmax` over `[scores, sink]`, drop sink col) | L694 |
| **PV inputs** | w (f32 from softmax) × `KV.astype(f32)` | L696 |
| **PV output** | f32 | L696 |

Downstream: `_o_lora_down` (L949) casts the attention output to f32 anyway
(`o.astype(f32)`), so the score fn's return dtype only needs to be f32-castable.

**CSA modes.** The four modes (`swa_only` / `full` / `reindex` / `reuse`, L230–238)
select only *which* KV rows and mask reach the score fn (`_attend`: window-only vs
`concatenate([window, compress_kv])`, and reuse reads the source layer's `topk_mask`).
They route through the **same** `_sparse_attend`, so the lever applies uniformly to
all four `attn.<mode>.score` stages. Reuse still runs its own per-layer score matmul.

**Sink / candidate masking is NOT the score path.** The data-dependent row pick
(`Indexer.select`, L539–550: `q.astype(f32)`, `index_k.astype(f32)`,
`weights.astype(f32)`) is the separate `attn.<mode>.select` stage and stays f32 under
K25 — so the bf16 arm leaves the **set of attended rows (`topk_mask`) bit-identical**;
only the attention *over* those rows changes. This is deliberate (never trade the
selection precision, cf. [[acceptance-rate-is-a-primary-bottleneck]]).

**Torch reference (`scripts/deepseek_v41/torchref/ref_forward.py`).** The oracle is
pure f32: `torch.set_default_dtype(torch.float32)` (L42), `_k_sparse_attn` (L135) does
`scores = torch.einsum(q.to(float32), kvg) * softmax_scale` (L146) and
`o = torch.einsum(ex, kvg)` (L154) — QK^T and PV both f32, softmax stable-f32. The
shipped MLX default (f32) **matches the oracle**. The oracle is explicitly a "clean
full-precision oracle" (ref_forward header L10) — the real DeepSeek deployment runs
bf16, so **bf16 scores are within the model's native precision; the f32 oracle is
stricter than the reference's own compute.**

## (2) Implementation

Two resolvers, read at call time (never frozen at import — the serving harness stamps
keys after import, [[env-flags-read-at-use-not-import]]):

- `_resolve_prefill_score_dtype` (L1045): unset/`f32`/`fp32`/`default` → `mx.float32`;
  `bf16`/`bfloat16` → `mx.bfloat16`; anything else raises (fail fast).
- `_resolve_prefill_score_key_chunk` (L1071): unset/`0`/`off` → `None` (one-shot); a
  positive int → chunk width; negative/garbage raises.

`_sparse_attend` (L660) gates on `q.shape[1]`: **`s == 1` (decode / M=1) always runs
the shipped f32 one-shot**, regardless of env. `s > 1` (prefill; also MTP verify's K+1
rows if the flag is ever set during decode) reads the two levers.

- `_sparse_attend_oneshot(score_dtype)` (L679): `score_dtype == f32` inserts **no
  casts** (byte-identical to control); else QK^T/PV inputs cast to `score_dtype`, the
  matmul accumulates in f32, and the result is `.astype(f32)` before the f32 scale/
  mask/sink/softmax — so the **only** numerical change is the two matmuls' input+output
  bf16 rounding.
- `_sparse_attend_chunked(score_dtype, key_chunk)` (L706): flash-style online softmax
  over `key_chunk`-wide key blocks. The value-0 sink **seeds** the running state
  (`m = attn_sink` finite, `denom = exp(0) = 1`, `acc = 0`), so a fully-masked chunk
  gives `corr = exp(m−m) = 1`, `p = 0` — no `-inf−(−inf)` NaN. Scores are bit-identical
  to one-shot (the QK^T reduces over head_dim, **not** the chunked T axis); only the
  softmax denom + value sum reassociate. Composes with `score_dtype` (the `both` arm).

## (2b) The bf16-accumulation proof (core claim)

MLX matmul accumulates in f32 and rounds the result to the input dtype — so a bf16
matmul's error is exactly the **output** rounding, not a K-growing accumulation error.
Proven on CPU at the QK^T shape (K=512 reduction):

```
QK^T bf16 == round_bf16(f32-accumulated)  ->  True (mx.array_equal)
max|bf16mm − f32acc|  = 0.20388   ==   max bf16 output-rounding of f32acc = 0.20388
```

(`tests/models/test_deepseek_v41_prefill_score_precision.py::test_bf16_matmul_accumulates_in_f32`.)
This is what licenses bf16 for the score matmuls: the accumulation is f32-clean.

## (3) Exactness evidence

CPU, `tests/models/test_deepseek_v41_prefill_score_precision.py` (16 tests, all pass;
peak RSS 1.62 GB).

**Real DSV4.1-shaped random layer (H=64, head_dim=512), max |Δ| on attention output vs
one-shot f32:**

| shape | bf16 one-shot | split-K f32 | split-K bf16 |
|---|---|---|---|
| rows=128, T=4096 | 1.76e-3 (rel 5.9e-3) | **6.11e-7 (rel 2.0e-6)** | 1.39e-3 |
| rows=256, T=2048 | 3.94e-3 (rel 1.1e-2) | **7.15e-7 (rel 1.9e-6)** | 4.25e-3 |
| rows=64,  T=4096 | 2.16e-3 (rel 7.8e-3) | **6.11e-7 (rel 2.2e-6)** | 1.65e-3 |

- split-K f32 is at the **f32 reassociation floor** (~1e-6), orders below bf16.
- split-K exactness holds for chunk widths `{1,7,32,64,128,512,4096}` incl. a width >
  T (degenerates to one-shot) and a non-divisor (7).
- Fully-masked chunk: finite (no NaN) and `array_equal` to one-shot (both collapse to
  the value-0 sink → zeros).

**End-to-end tiny-model prefill (greedy-argmax parity of the final logits, 30-token
prompt):**

- control == explicit `f32` (`array_equal`).
- **split-K f32**: logits max |Δ| < 1e-4, **greedy argmax identical on every row**
  (reassociation ≪ logit gaps).
- **bf16**: runs finite, logits Δ at bf16 scale; on this **untrained** double it flips
  some intermediate-row argmax (last row happened to match). **Greedy parity is NOT
  guaranteed** and is not asserted.

**Expected effect on greedy tokens (honest).** split-K f32 (and the default f32) are
greedy-identical to control. **bf16 is lossy** — like `head_bf16` (K21) it can tip
near-tie argmax, so exactness is not the ship bar; the gate is a task eval (HumanEval
164, [[deepseek-v4-quality-verdict]] / [[task-evals-decide-bank-verdicts]]). bf16
scores sit inside the model's native deployment precision (the f32 oracle is stricter).

## (4) Arms + ledger

`scripts/deepseek_v41/ab_decode_env_levers.py` (keys pinned in `_preset`, all 12 lever
keys force-set/unset per arm; dry-run test extended, `tests/test_deepseek_v41_ab_env_levers.py`
35 tests pass):

- `score_bf16` — `MTPLX_DSV41_PREFILL_SCORE_DTYPE=bf16`. Lossy (like head_bf16).
- `score_chunked` — `MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK=2048`. f32, greedy-identical.
- `score_bf16_chunked` — both.
- `prefill_fast` — the stacked 16K prefill candidate (rebased onto W51): `layer_major`
  + `prefill_dense_experts` (K26) + `score_bf16` + `score_chunked`. Lossy (dense fp32
  accumulation order + bf16 score rounding); task-eval gated.

KERNEL_LEDGER **K25** carries the FLOP/byte arithmetic (score matmuls ≈ 4·rows·H·T·d
FLOP → the 1.12 PFLOP W47 estimate; bf16 2× → est. −60 to −90 s of the 201 s score term
→ TTFT −16 to −22 %; transient 6 GiB → 0.54 GB at n=2048, 0.27 GB bf16+chunked).

## What is NOT measured here

No GPU/Metal ran. The realized TTFT / peak-GB delta is the **GPU gate KG-g** (K6
two-pass + tiling on the K16 base: *TTFT −≥15 % beyond K16 AND peak −≥8 GB, parity on
the long-prompt A/B*). The −60 to −90 s figure is a roofline estimate (softmax/mask/
dispatch stay f32, so not the full 2×); split-K's per-chunk peak reduction assumes the
MLX executor frees each chunk's score buffer after its update (flash-style), to be
confirmed in-window.
