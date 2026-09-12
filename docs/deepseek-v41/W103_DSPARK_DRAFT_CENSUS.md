# W103 — DSpark DRAFT-step dispatch census + head fp32-cast trap

Worker `w103/dspark-draft-census`. Opus 4.8, CPU-only (`mx.set_default_device(mx.cpu)`,
tiny seeded real-structure DSpark head, no artifact, no GPU). Subject: the DSpark
**draft phase** only (`DSparkHead.draft_block` → `forward_embed` / 3 `DSparkBlock`
stages / `forward_head`). The verify path, the expert-streaming runtime, and the
harness DSpark generator are out of scope (W100's lane / other lanes).

Baseline is GPU window 39 (`receipts/gpu-windows/window-39/dspark-d5-ring-v2.json`,
depth 5, 54 cycles, `tokens_per_cycle` 4.76, `accept_rate` 0.903). Its arm ran
`MTPLX_DSV41_HEAD_MODE=bf16` **and** `MTPLX_DSV41_DRAFT_COMPILE=None` (draft eager),
so the census mirrors it: eager, DRAFT_COMPILE OFF.

Reproduce:
```
PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \
  scripts/deepseek_v41/dispatch_census.py --dspark-draft --seed 1 --out receipt.json
```

## 0. Headline

Per cycle the draft is **≈243 ms** (`dspark.draft`), and it is dominated by **two
≈83 ms items**, not five 50 ms "steps":

| item | window-39 ms/cycle | share | nature |
|---|---:|---:|---|
| `dspark.head` | **82.7** | 34 % | **bandwidth — the fp32-cast trap** (a 2.65 GB f32 head temporary/cycle) |
| resident 128-expert MoE (3 stages) | ≈83 (243 − 160 bracketed) | 34 % | real `gather_qmm` compute — **expert-streaming runtime, off-limits** |
| `dspark.attn.*` | ≈35.6 | 15 % | `out_prep` 24.2 dominates (o-LoRA einsum + `wo_b`) + dispatch |
| `dspark.forward_embed` | 20.4 | 8 % | main-proj + embed + broadcast |
| `dspark.markov` | 11.7 | 5 % | 5 sequential markov steps (cheap) |
| `dspark.hc.*` | 9.4 | 4 % | attn_prep 4.1 + ffn_prep 3.9 + moe_combine 1.4 |
| `dspark.confidence` | 0.25 | — | one batched matmul |

The single clear-cut, self-contained draft lever is **removing the head fp32-cast
trap** — the exact twin of the backbone W40/K21 fix, which
[[dsv41-head-fp32-cast-trap]] explicitly names as the DSpark follow-up. Implemented
here behind `MTPLX_DSV41_DRAFT_HEAD_BF16` (default OFF, byte-identical on the f32
path). Everything else is either an existing lever to *arm* (K33 DRAFT_COMPILE), a
rounding-class fusion out of scope for this window (K29 attention core), or another
lane (MoE / verify).

## 1. Dispatch table per draft step (eager, DRAFT_COMPILE off, tiny real-structure)

`prim` = graph primitives (every `export_to_dot` rectangle). `nonview` = primitives
that are not pure views/metadata (Reshape/Broadcast/Transpose/ExpandDims/Squeeze/
Slice/Flatten/Unflatten/Split…) — the proxy for the ~100 µs/kernel host-encode floor
([[b1-decode-dispatch-removal-hides]]). Stages fire 3×/cycle (once per DSpark stage)
except embed/head/markov/confidence (1×).

### ONE draft step (`block_size = 1` — the per-cycle FIXED cost)

| group | stage | calls | prim | nonview | top ops |
|---|---|---:|---:|---:|---|
| embed | `dspark.forward_embed` | 1 | 24 | 11 | Broadcast:6 Multiply:3 Full:1 |
| attn | `dspark.attn.main_kv` | 3 | 135 | 63 | Broadcast:27 Multiply:24 Slice:12 |
| attn | `dspark.attn.qkv_prep` | 3 | 267 | 114 | Broadcast:51 Multiply:45 Slice:24 |
| attn | `dspark.attn.sdpa` | 3 | 237 | 98 | Broadcast:46 Multiply:30 Transpose:22 |
| attn | `dspark.attn.out_prep` | 3 | 114 | 33 | Reshape:27 Slice:12 Multiply:12 |
| **attn** | | | **753** | **308** | |
| hc | `dspark.hc.attn_prep` | 3 | 242 | 123 | Broadcast:71 Multiply:30 Add:30 |
| hc | `dspark.hc.ffn_prep` | 3 | 267 | 129 | Broadcast:75 Multiply:33 Add:33 |
| hc | `dspark.hc.moe_combine` | 3 | 33 | 9 | Transpose:9 ExpandDims:6 |
| **hc** | | | **542** | **261** | |
| moe | `moe.gate_topk` | 3 | 78 | 54 | Broadcast:15 GatherAxis:9 AsType:6 |
| moe | `moe.routed_switch` | 3 | 63 | 30 | Transpose:9 Arange:9 AsType:9 GatherMM |
| moe | `moe.shared_expert` | 3 | 24 | 15 | Matmul:9 Transpose:9 |
| moe | `moe.combine` | 3 | 18 | 9 | Multiply:3 Sum:3 |
| **moe** | | | **183** | **108** | |
| head | `dspark.head` | 1 | 21 | 11 | Broadcast:5 Multiply:4 **Matmul:1 AsType:1** |
| glue | `dspark.markov` | 1 | 14 | 7 | ExpandDims:3 Squeeze:3 AsType:2 Gather:1 |
| glue | `dspark.confidence` | 1 | 7 | 2 | Concatenate:1 Matmul(f32) |
| **TOTAL / cycle** | | | **1544** | **708** | |

### Full depth-5 phase (`block_size = 5`) vs one step

| | prim / cycle | nonview / cycle |
|---|---:|---:|
| one step (block 1) | 1544 | 708 |
| full phase (block 5) | 1605 | 730 |
| **Δ over +4 draft rows** | **+61 (+4 %)** | **+22** |

**The draft is ≈96 % per-cycle FIXED cost.** Emitting 5 draft tokens costs +4 %
dispatch over emitting 1 — only `dspark.markov` (+50 prim, the 5 sequential markov
steps) and `dspark.confidence`/`qkv_prep`/`sdpa` (a few rows) scale with depth; the
3-stage backbone-block forward and the head are paid once. So the "251 ms / 5 steps ≈
50 ms/step" arithmetic is an amortization of one fixed forward, **not** five
independent steps — which is why raising depth amortizes the fixed cost over more
accepted tokens nearly for free, and why the levers below all target per-cycle cost.

## 2. Head bytes per draft step (dtype and shape actually materialized)

`forward_head` did `base_logits = head(_rmsnorm(x, …).astype(mx.float32))`. With a
dense **bf16** head (the native artifact keeps it bf16; `HEAD_MODE=bf16` leaves the
weight bf16 — it repairs only `Model._apply_head`, never this draft site), MLX has no
mixed-precision matmul, so `f32 @ bf16.T` **promotes the whole `[vocab, hidden]`
weight to an f32 temporary** before the GEMV.

Graph-verified on the tiny bf16 head (`_dspark_head_bytes`): `xf @ W.T` with `xf` a
pre-eval'd **f32** leaf and `W` a pre-eval'd **bf16** leaf carries exactly **1
`AsType`** — and it vanishes when `W` is already f32 — so the f32 cast lands on the
**weight**, not the tiny input. The bf16 fix's only `AsType` is the tiny f32-logits
cast.

| path | f32 temporary materialized | weight-related DRAM traffic / cycle |
|---|---|---:|
| **trap** (`head(x.astype(f32))`) | **whole weight → f32**: `[129280, 5120]` = **2.648 GB** | read bf16 1.324 + write f32 2.648 + read f32 2.648 = **6.619 GB** |
| **fix** (`head(x.astype(bf16)).astype(f32)`) | only logits `[1, block, 129280]` f32 = **2.6 MB** | read bf16 1.324 (+2.6 MB) = **1.326 GB** |

**Trap = 5.0× the traffic of the fix, block-size-independent** (the weight dominates
the 5 draft rows). Matches the backbone W40 accounting byte-for-byte (`head_resident_
bytes_default` 1,323,827,200 B). This is why `dspark.head` is **dispatch-light (11
non-view kernels) but 82.7 ms/cycle**: it is bandwidth, not dispatch — a hidden f32
temporary a dispatch census alone would miss, which is exactly how W40 found it.

## 3. Where the ≈50 ms/step (243 ms/cycle) goes

- **`dspark.head` 82.7 ms — BANDWIDTH (the fp32-cast trap).** Not dispatch (11
  non-view kernels): the 2.648 GB f32 weight promotion + allocator churn. Removable,
  self-contained. **Lever 1.**
- **Resident 128-expert MoE ≈83 ms — real `gather_qmm` compute.** The unbracketed
  remainder of `dspark.draft` (243 − 160 bracketed). `moe.routed_switch` in the
  receipt (869 ms/cycle, count 2322) is **verify-polluted** — it also brackets the
  backbone verify's many-layer streamed MoE — so the draft's share is read from the
  wall gap, not that counter. This is the **expert-streaming runtime and is not
  dispatch-bound; out of scope.**
- **Dispatch-bound remainder ≈57 ms floor.** attn (308) + hc (261) = **569 non-view
  kernels/cycle** × ~100 µs ≈ 57 ms of pure host-encode — consistent with the
  receipt's attn (35.6) + hc (9.4) + embed (20.4). This is the **K33 DRAFT_COMPILE**
  tape target (existing lever, OFF in window 39). **Lever 2.**
- **`dspark.attn.out_prep` 24.2 ms** is the single biggest attn substage (the
  query-RoPE-removal + grouped o-LoRA down einsum + `wo_b`). **Lever 3.**
- markov 11.7 / embed 20.4 / hc 9.4 / confidence 0.25 — small.

## 4. Lever plan, ranked by expected ms saved / cycle

| # | lever | flag | expected ms/cycle | exactness | status |
|---|---|---|---:|---|---|
| 1 | **Remove the head fp32-cast trap** — cast the draft hidden to the head weight dtype (bf16 GEMV, f32 logits after) instead of promoting the weight | `MTPLX_DSV41_DRAFT_HEAD_BF16` | **≈55–65** (82.7 → ~15–25 bf16-GEMV floor; 5× less traffic) | rounding-class on bf16 head; **byte-identical when head is f32**; draft numerics only set acceptance (greedy verify == AR) | **IMPLEMENTED (default OFF)** |
| 2 | **Arm the existing K33 draft tape collapse** — replay the pure attn/HC/gate/combine chains from `mx.compile` tapes instead of rebuilding per cycle | `MTPLX_DSV41_DRAFT_COMPILE` (exists) | ≈20–40 (collapses the 569 dispatch-bound non-view kernels; census 1591→1145 prim, attn 759→591, hc 542→284) | **byte-identical** (K33 tests); fixed-shape + row-capped tapes, so no `mx.compile` reduction reassociation on the tiny draft rows (K4/K35 caveat handled) | exists, OFF in window 39 — **arm it next window** |
| 3 | Fuse the draft attention core (K29-style: RoPE-remove + o-LoRA einsum + `wo_b` as one tape/kernel) | new | ≈8–15 (`out_prep` 24.2) | rounding-class (einsum reassociation) | **deferred** — not clear-cut/self-contained enough for this window |
| — | Resident 128-expert MoE (`gather_qmm`) ≈83 ms | — | — | — | **off-limits: expert-streaming runtime** |
| — | Verify SSD ≈1475 ms/cycle | — | — | — | **another lane** (W100 / verify) |

**Why lever 1 is byte-identical-classed, not `mx.compile`-classed:** it is a single
`astype` on the head input, not an `mx.compile` of a reduction chain, so there is no
reassociation (the K4/K35 trap). On the tiny f32 census head the cast is a no-op →
byte-identical (test 1, 64 steps). On the real bf16 head it is a clean bf16-round of
the hidden before a bf16 GEMV — exactly the backbone `HEAD_MODE=bf16` codec, which
was **token-byte-identical to control in window 14** and gave **+31 % decode**. The
draft's greedy target verify is authoritative, so even a draft-logit rounding change
cannot change the emitted sequence — only the acceptance rate.

**Ceiling effect.** With verify free, the draft-limited ceiling is
`4.76 / 0.243 = 19.6 tok/s`; lever 1 alone → `4.76 / ~0.183 = ~26 tok/s` (+33 %, the
W40 class). End-to-end gain is small until the verify SSD cost is fixed (another
lane), after which lever 1 is the top draft lever.

## 5. What was implemented (part C)

`mtplx/models/deepseek_v41_dspark.py`:

- `MTPLX_DSV41_DRAFT_HEAD_BF16` (default OFF, read at use never frozen at import;
  module-global pin `_DRAFT_HEAD_BF16` for tests/census). Helper
  `_draft_head_source_dtype(head)`: a dense float head → its own weight dtype (bf16
  GEMV, no promotion); a quantized head (packed weight + `.scales`) → f32
  (`quantized_matmul` dequantises per group, never promotes) so the flag is a no-op
  there.
- `forward_head` now branches: ON → `head(hidden_n.astype(_draft_head_source_dtype
  (head))).astype(mx.float32)`; OFF → the historical `head(hidden_n.astype(f32))`
  (byte-for-byte the old path).
- Process-cumulative engagement counters `_DRAFT_HEAD_BF16_CALLS` /
  `_DRAFT_HEAD_DEFAULT_CALLS` (+ `_reset_draft_head_calls` / `_draft_head_calls`),
  mirroring the W38/K3 Sinkhorn counters — one int add, so an A/B census can tell
  whether the bf16 branch actually engaged.

`scripts/deepseek_v41/dispatch_census.py`: `--dspark-draft` (this census: per-stage
prim/non-view/top-ops for block_size 1 and 5, plus `_dspark_head_bytes`).

`tests/test_deepseek_v41_w103_draft.py` (CPU-pinned, 5 tests): byte-identical draft
tokens/logits/confidence flag on-vs-off over **64 draft steps** on the tiny f32 head;
engagement counter (one head call/cycle on the flag's branch); env read-at-use +
global-pin override; `_draft_head_source_dtype` per codec; and a bf16-head structural
test proving the flag removes the weight-promotion `AsType` (with the counter engaged
and f32 logits out). `5 passed`. Existing K33 draft-compile suite still `12 passed`.
