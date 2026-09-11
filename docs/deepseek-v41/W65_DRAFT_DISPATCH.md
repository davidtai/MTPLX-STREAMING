# W65 — DSpark draft-block dispatch census + tape collapse (kernel-ledger K33)

Branch `feat/deepseek-v41-w65` off `feat/deepseek-v41-streaming` @ `d89044e34`.
Files: `mtplx/models/deepseek_v41_dspark.py` (the drafter — the K33 tapes + the
census stage brackets), `scripts/deepseek_v41/dispatch_census.py` (a `--draft`
mode over the draft block), `tests/models/test_deepseek_v41_dspark_draft_compile.py`
(new, 12 tests), `scripts/deepseek_v41/ab_decode_env_levers.py` +
`tests/test_deepseek_v41_ab_env_levers.py` (the `draft_compile` arm), this report,
`docs/deepseek-v41/KERNEL_LEDGER.md` (§K33 + KG-n). CPU-only, tiny synthetic
config, no artifact, no GPU/Metal. Peak RSS: census tool **0.10 GB** (`/usr/bin/time
-l` maximum resident set size 107,806,720 B), the new exactness suite **0.12 GB**
(124,141,568 B).

## Verdict

The DSpark-DIRECT draft block (W57 §6d) measured **~261 ms/cycle** on the GPU — 3
shallow stages (each a full V4.1 decoder block: sliding-window attention + a
RESIDENT 128-expert top-3 MoE) over `block_size` draft rows, plus `forward_embed`
and a markov autoregression over `block_size` sequential steps — **~87 ms/stage
for a tiny amount of math**: pure dispatch count, the same regime the backbone's
K22 (attention tapes) and K4 (Hyper-Connection tapes) addressed
([[b1-decode-dispatch-removal-hides]]). This worker (1) extends the W41 census
tool with a draft-block mode and produces the primitives-per-stage table, and (2)
replays the draft block's **pure, fixed-shape** chains from `mx.compile` tapes
behind `MTPLX_DSV41_DRAFT_COMPILE` (default OFF).

**Census result (tiny real-structure DSpark head — 3 stages, `block_size` 4,
resident 8-expert top-2 MoE == the 128-expert top-3 structure at small scale):**
the draft cycle drops from **1,591 → 1,145 primitives (−446, −28 %)**, all
byte-identical: flag on vs off is `mx.array_equal` on the draft **tokens AND
logits AND confidence** (0/4 seeds diverge), the greedy-verify == AR gate stays
green with the flag on (depth 1/2/3), and flag-off is byte-for-byte the shipped
eager drafter. The realized GPU per-cycle delta is a GPU-window measurement
(**KG-n**), not flipped in serving.

## 1. The census (`scripts/deepseek_v41/dispatch_census.py --draft`)

Same machinery as the W41 backbone census — `mx.export_to_dot` node-counting via
`count_prims`, the `_CensusProbe` installed into the W37 stage singleton — but over
the **draft block** instead of a backbone decode forward. The drafter now carries
`_stime.stage(...)` brackets (no-ops unless a probe is armed, exactly like the
backbone), so the same probe tiles the draft cycle: `forward_embed`, the 3 stages'
attention prep (`main_kv` / `qkv_prep` / `sdpa` / `out_prep`) + Hyper-Connection
prep (`attn_prep` / `ffn_prep` / `moe_combine`) + the reused MoE (`gate_topk` /
`routed_switch` / `shared_expert` / `combine`), then the last stage's `head` +
`markov` + `confidence`. One `draft_block` == one "token" (draft cycle); counted
before/after `MTPLX_DSV41_DRAFT_COMPILE`.

The tiny config uses power-of-2 reductions (hidden 32, `hc*dim` 128, `q_lora` 16,
`head_dim` 16) so every compiled chain is bit-exact vs eager — dodging the K22
tiny-config RMSNorm reassociation caveat (`q_lora_rank=12` diverges; 16 does not).

## 2. Census table — primitives per draft cycle per stage

`MTPLX_DSV41_DRAFT_COMPILE` before=OFF (eager) vs after=ON (K33), CPU:

| stage | calls/cycle | prim/call (before) | prim/cycle before | prim/cycle after | Δ |
|---|---:|---:|---:|---:|---:|
| `dspark.attn.qkv_prep` | 3 | 90.0 | 270.0 | 171.0 | **99** |
| `dspark.hc.ffn_prep` | 3 | 89.0 | 267.0 | 138.0 | **129** |
| `dspark.hc.attn_prep` | 3 | 80.7 | 242.0 | 122.0 | **120** |
| `dspark.attn.sdpa` | 3 | 80.0 | 240.0 | 240.0 | 0 |
| `dspark.attn.main_kv` | 3 | 45.0 | 135.0 | 87.0 | **48** |
| `dspark.attn.out_prep` | 3 | 38.0 | 114.0 | 93.0 | **21** |
| `moe.gate_topk` | 3 | 26.0 | 78.0 | 63.0 | **15** |
| `moe.routed_switch` | 3 | 21.0 | 63.0 | 63.0 | 0 |
| `dspark.markov` | 1 | 52.0 | 52.0 | 52.0 | 0 |
| `dspark.hc.moe_combine` | 3 | 11.0 | 33.0 | 24.0 | **9** |
| `dspark.forward_embed` | 1 | 24.0 | 24.0 | 24.0 | 0 |
| `moe.shared_expert` | 3 | 8.0 | 24.0 | 24.0 | 0 |
| `dspark.head` | 1 | 20.0 | 20.0 | 20.0 | 0 |
| `moe.combine` | 3 | 6.0 | 18.0 | 15.0 | **3** |
| `dspark.confidence` | 1 | 11.0 | 11.0 | 9.0 | **2** |
| `sample` | 1 | 0.0 | 0.0 | 0.0 | 0 |
| **TOTAL** | | | **1591.0** | **1145.0** | **446.0** |

Grouped: **Hyper-Connection prep 542 → 284 (−258)**, **attention prep 519 → 351
(−168)**, MoE gate-prefix + combine folds 96 → 78 (−18), markov + confidence 63 →
61 (−2). The per-attention-call cuts match K22 exactly (qkv −33/call, out −7/call);
the HC cuts match K4's mix/Sinkhorn/RMSNorm collapse (−40/attn-prep, −43/ffn-prep).

**Is the resident 128-expert MoE a `gather_qmm` or a Python loop?** A `gather_qmm`.
`moe.routed_switch` is **~21 primitives per stage** (NOT O(n_experts)): the DSpark
stage's `self.mlp` is the reused `MoE`, whose `switch_mlp = SwitchGLU(...)` reshapes
the `[b, block_size, dim]` draft to `[block_size, dim]` and issues **one
`mx.gather_qmm` over the `block_size*top_k` rows** (mlx-lm's quantised switch), not a
per-expert loop. So the draft's MoE is already one gather over the 4 rows × top-3
(the K10/K27 shape); K33 folds the pure **gate prefix + combine** around it (−18)
and leaves the gather itself untouched.

## 3. What was compiled (and why it is byte-identical)

A DSpark stage IS structurally a backbone V4.1 layer (HC pre/post around attention
+ MoE), and the DSpark attention IS a base-`Attention` with the same projection
codec and head geometry, so K33 **reuses the backbone tapes** rather than
re-deriving them:

1. **Attention prep** — the draft QKV chain (`q = rope(unflatten(wq_b(rmsnorm(
   wq_a(x)))))`, `kv = rope(rmsnorm(wkv(x)))`) replays through `deepseek_v41.
   _attn_qkv_prep` (K22), and the output chain (query-RoPE removal + grouped o-LoRA
   einsum + `wo_b`) through `_attn_out_prep` (K22). A small **local** `_draft_kv_prep`
   tape carries the window-KV chain `rope(rmsnorm(wkv(m)))` (main positions). Each
   projection is applied by `_apply_lin` exactly as `nn.Linear` / `nn.QuantizedLinear`
   (`mx.quantized_matmul` replays as one primitive compile never reassociates), so
   the tape is bit-identical to the eager module call whether residents are dense
   (tiny/native-BF16) or quantised (mxfp4/mxfp8/q8).
2. **Hyper-Connection prep** — the two mix + Sinkhorn + `pre_mix`-collapse + RMSNorm
   chains and the moe-combine post replay through `deepseek_v41._hc_compiled`
   (`attn_prep` / `ffn_prep` / `moe_combine`, K4). The Sinkhorn boundary is the same
   opaque `hc_split_sinkhorn` (recurrence on CPU, the W32 Metal kernel when armed on
   GPU), so the tape reads no dynamic `.shape` and never perturbs the numerics.
3. **MoE gate prefix + combine** — fired by arming the K22 `ATTN_COMPILE` window
   (`_moe_compile_window`, a scoped save/restore of `deepseek_v41._ATTN_COMPILE`)
   around the resident switch call, so the reused `MoE`'s existing K22 folds
   (`_gate_prefix` / `_moe_combine`) engage. The data-dependent argpartition/argsort
   top-k stays eager.
4. **Markov autoregression** — each greedy step's `embed + head-matmul + base-logit
   add + argmax` folds into one compiled tape (`_draft_markov_step`), and the markov
   embed for the confidence head is gathered **once** over the sampled block
   (`embed(output_ids[:, :block_size])`) instead of stacked per step. The confidence
   head is one batched compiled matmul.

Every tape is **fixed-shape + row-cap** (`_DRAFT_COMPILE_MAX_ROWS = 32`): compile
fires only at the tiny repeating draft-row shape (`block_size` rows), the eager body
runs unchanged above the cap and with the flag off. The switch reads the env at USE
(module-global pin for tests/census, else the key — [[env-flags-read-at-use-not-import]]).

## 4. What was NOT compiled, and why

- **The SDPA (`_sparse_attend`, 240 prim/cycle).** Materialises the `[b, T, H, Wp+T]`
  score whose key length grows with the window — a dynamic shape — and D512 is not a
  fused-SDPA dim. Kept eager between the two attention tapes (the K29 decode-attention
  kernel is the separate GPU lever for this tail on the verify path).
- **The resident routed gather (`moe.routed_switch`, 63 prim/cycle).** Already one
  `gather_qmm` (§2); a tape cannot fold an opaque gather. K10/K27 own its shape.
- **The MoE routing barrier** (argpartition/argsort/take_along_axis) — data-dependent;
  only the pure gate prefix is folded.
- **The markov step's dispatch count is FLAT (52 → 52).** Its ops — an embedding
  gather, a `rank → vocab` matmul, and an argmax — are non-elementwise, so compile has
  no adjacent elementwise chain to fuse. The tape still removes the per-step **Python
  graph rebuild** (the host-encode lever [[b1-decode-dispatch-removal-hides]] the
  node count does not capture) and keeps the loop **sync-free** (argmax stays lazy);
  the census reports node count, so it shows 0 there honestly.

## 5. Exactness

The bar for the DSpark head is greedy-verify == AR (the runtime's target argmax is
authoritative); K33 is a pure dispatch cut, so the bar here is stricter — **byte-
identical draft output** flag on vs off:

- `test_draft_block_flag_on_off_byte_identical` — `mx.array_equal` on draft tokens
  AND logits AND confidence, seeds {0,1,2,3} (0/4 diverge, `max|Δ| = 0`).
- `test_flag_off_matches_the_shipped_eager_drafter` — flag OFF == the pre-K33 eager
  path byte-for-byte.
- `test_greedy_verify_equals_ar_with_draft_compile_on` — greedy `generate_mtpk` ==
  `generate_ar` over 48 tokens at depth {1,2,3} with the flag ON (the draft block is
  reached by both the DIRECT lane and the generic engine via `mtp_forward`).
- `test_draft_compile_on_and_off_spec_agree` — the lane's spec tokens are identical
  on and off.
- `test_flag_defaults_off_and_reads_env_at_use` / `test_row_cap_confines_compile_...`
  — the env truthiness + module-global pin + the row cap.
- `test_draft_census_total_and_targeted_stages_drop` — asserted **from the census
  tool**: total drops; every compiled chain drops; the SDPA / routed gather / shared
  expert are unchanged; the routed gather is `< 64` prim/call (one gather, not a loop).

**Verification (CPU, `nice -n 19`, no `-n auto`, one file at a time under the 1.5 GB
box guard):**

- `test_deepseek_v41_dspark_draft_compile.py` — **12 passed** (0.12 GB peak).
- `test_deepseek_v41_dspark.py` — **15 passed** (default-off greedy == AR unchanged);
  and **36 passed** re-run with `MTPLX_DSV41_DRAFT_COMPILE=1` (drafter + decode).
- `test_deepseek_v41_dspark_decode.py` — **21 passed**.
- `test_deepseek_v41_attn_compile.py` — **12 passed** (the reused K22 tapes intact).
- `test_deepseek_v41_ab_env_levers.py` — **53 passed** (the `draft_compile` arm +
  independence).

## 6. The arm + default OFF

`MTPLX_DSV41_DRAFT_COMPILE` is opt-in so every existing gate stays green and the
eager per-cycle graph stays the serving default. The draft-phase win is a GPU-window
measurement (**KG-n**): `draft_compile` vs control on `--decode-mode dspark` —
byte-identical draft tokens (flag on == off) + greedy-verify == AR + draft-phase ms
down. The arm is wired into `ARM_PRESETS` (pinning every lever key, so it is
independent of prior arms). Only the DIRECT lane runs the draft block, so K33
composes with the decode/verify levers (K29/K30/K31) on the target forward; it is the
**drafter half** of getting the DIRECT lane's per-cycle cost down (K31 is the verify
half). At 3.78 tokens/cycle (W57 §6b) the draft is a minority of the cycle wall
(verify dominates), so credit no end-to-end tok/s to K33 without the in-window A/B.

## 7. Not changed

- The backbone attention / HC / MoE / switch files (`deepseek_v41.py`,
  `deepseek_v41_moe.py`, `expert_mlx.py`) — untouched; K33 only IMPORTS and reuses
  their K22/K4 tapes (`_attn_qkv_prep` / `_attn_out_prep` / `_hc_compiled` / the
  `_ATTN_COMPILE` gate-prefix window), gated on the new draft flag.
- The DSpark seed / rollback / cache seam, the decode lane's verify/accept/commit
  (`deepseek_v41_dspark_decode.py`), the K29/K30/K31 verify levers — unchanged; all
  byte-identical with the draft flag off, and the greedy-verify == AR contract holds
  with it on.
