# W41 — Dispatch census + attention-chain tape collapse (kernel-ledger K22)

Branch `feat/deepseek-v41-w41` off `feat/deepseek-v41-streaming` @ `a8f43b95a`.
Files: `mtplx/models/deepseek_v41.py` (attention chains + the compile infra),
`mtplx/models/deepseek_v41_moe.py` (the gate-prefix and combine folds),
`scripts/deepseek_v41/dispatch_census.py` (new census tool),
`tests/models/test_deepseek_v41_attn_compile.py` (new, 11 tests),
`scripts/deepseek_v41/ab_decode_env_levers.py` + `tests/test_deepseek_v41_ab_env_levers.py`
(the `attn_compile` arm), this report, `docs/deepseek-v41/KERNEL_LEDGER.md` (§K22
+ gate KG-i). CPU-only, tiny synthetic config, no artifact, no GPU/Metal. Peak
RSS of the new test file: **0.12 GB** (`/usr/bin/time -l` maximum resident set
size 116,391,936 B); census tool 0.11 GB.

## Verdict

W37/window-13 stage timing said *where* the 246 ms/token decode goes but not how
many **primitives** each stage dispatches — the lever whole-chain `mx.compile`
moves. This worker (1) builds a CPU dispatch census that counts graph primitives
per decode stage via `mx.export_to_dot` node counting, and (2) uses it to compile
the two largest **pure, fixed-shape** attention chains (plus the pure MoE gate
prefix and combine) behind `MTPLX_DSV41_ATTN_COMPILE` (default OFF).

**Census result (tiny real-structure model, all four CSA modes + HC + gate +
combine + a synthetic engram hook):** the compiled tapes remove **40 primitives
from every attention call in every CSA mode** (QKV-prep 81→48, output-prep 38→31),
the gate prefix 8→3, and the combine 6→5, dropping the whole decode token from
**6,631 → 6,263 primitives/token**. Flag on vs off is `mx.array_equal` (f32 CPU)
over decode (n=1), a K+1 verify batch, and chunked + layer-major prefill; the KV /
compress / index cache is identical on/off (the tapes are pure); the census
reduction is asserted; every existing DSV4.1 suite is green.

The realized GPU decode/dispatch delta is a GPU-window measurement (KERNEL_LEDGER
**KG-i**), not flipped in serving.

## 1. The census tool (`scripts/deepseek_v41/dispatch_census.py`)

**How it counts primitives reliably on CPU.** `mx.export_to_dot(buf, *outs)`
writes the lazy MLX graph rooted at `outs`; every `[label ="Op", shape=rectangle]`
node is one primitive. This is the same graph the K3 worker's "~119 primitives
per Sinkhorn call" counts (validated here: `_sinkhorn_ops` is **198** eager
rectangle nodes — 79 Broadcast + 40 Add + 39 Sum + 39 Divide + 1 Softmax — of
which the 78-op reduce+divide core matches the ledger's 119 once Broadcast is
excluded). Crucially `mx.compile` **fuses** elementwise chains into single
`Compiled*` nodes, so the dot node count drops exactly as the tape collapses
(measured: the compiled Sinkhorn is **80** nodes, an 8-op elementwise chain
collapses 8→1). Inputs that are already `mx.eval`'d appear as source nodes, not
rectangles, so a subgraph rooted at one stage's outputs — with the previous
stages eval'd — counts **only that stage's** primitives.

**Two censuses.**

* **Full-model per-stage.** A `_CensusProbe` is installed into the W37 stage
  singleton (`deepseek_v41_stage_timing._ACTIVE`); it duck-types the probe surface
  (`enter_forward`/`_stage`/`_frame`/`snapshot`) and reuses the model's own
  `with _stime.stage(name)` brackets, but at each stage boundary counts the graph
  primitives of that stage's output arrays **before** `mx.eval`ing them (so the
  next stage's subgraph is disjoint and the per-stage counts tile the token). It
  runs the tiny real-structure model — all four CSA modes (swa_only / full /
  reindex / reuse), HC, gate, combine, and a synthetic engram hook on the engram
  layers (the real 104 GiB engram artifact is not loadable here; K22 does not
  touch engram, so a representative additive-residual `engram.apply` bracket keeps
  the engram-layer structure in the census).
* **Per-chain micro-census.** Each K22-compiled chain counted eager vs compiled in
  isolation (rows=1), the cleanest before/after evidence with op-type breakdown.

## 2. Census table — primitives/token per decode stage

Tiny real-structure model (8 layers: 2 swa_only, 2 full, 3 reuse, 1 reindex),
`MTPLX_DSV41_ATTN_COMPILE` before=OFF (eager) vs after=ON (K22), CPU:

| stage | calls/tok | prim/call (before) | prim/tok before | prim/tok after | Δ |
|---|---:|---:|---:|---:|---:|
| `hc.premix_sinkhorn` | 16 | 259.0 | 4144.0 | 4144.0 | 0 |
| `attn.full` | 2 | 303.5 | 607.0 | 527.0 | **80** |
| `attn.reuse` | 3 | 162.0 | 486.0 | 366.0 | **120** |
| `attn.swa_only` | 2 | 160.5 | 321.0 | 241.0 | **80** |
| `attn.reindex` | 1 | 284.7 | 284.7 | 244.7 | **40** |
| `moe.routed_switch` | 8 | 27.0 | 216.0 | 216.0 | 0 |
| `moe.gate_topk` | 8 | 25.0 | 200.0 | 160.0 | **40** |
| `hc.combine` | 16 | 10.5 | 168.0 | 168.0 | 0 |
| `moe.shared_expert` | 8 | 14.0 | 112.0 | 112.0 | 0 |
| `moe.combine` | 8 | 6.0 | 48.0 | 40.0 | **8** |
| `final_norm` | 1 | 16.0 | 16.0 | 16.0 | 0 |
| `engram.apply` (synthetic) | 2 | 6.0 | 12.0 | 12.0 | 0 |
| `embed` | 1 | 9.0 | 9.0 | 9.0 | 0 |
| `head` | 1 | 4.0 | 4.0 | 4.0 | 0 |
| `sample` | 1 | 3.0 | 3.0 | 3.0 | 0 |
| **TOTAL** | | | **6630.7** | **6262.7** | **368.0** |

Per-chain micro-census (rows=1, isolated, eager → compiled):

| chain | eager | compiled | reduction |
|---|---:|---:|---:|
| `attn.qkv_prep` | 81 | 48 | 33 |
| `attn.out_prep` | 38 | 31 | 7 |
| `moe.gate_prefix` | 8 | 3 | 5 |
| `moe.combine` | 6 | 5 | 1 |

The per-attention-call reduction is a **mode-invariant 40** (qkv 33 + out 7): the
two compiled chains are the projection/norm/RoPE and output prep that every layer
runs identically; the mode-specific work (the Indexer, the SDPA, the KV writes)
stays eager and unchanged, so `attn.reuse`'s 162→122 and `attn.full`'s 303→263
differ only by that constant. `hc.premix_sinkhorn` (4,144 prim/tok, the
Sinkhorn-dominated top source) and `hc.combine` are **K4/HC-compile territory**
(`MTPLX_DSV41_HC_COMPILE`) — untouched here, so they read identical before/after,
which the test asserts.

## 3. What was compiled (and why it is bit-exact)

Two `mx.compile` tapes in `deepseek_v41.py`, one each shared across all 40 layers
(same projection codec + head geometry → one tape, weights as **inputs**), keyed
in `_ATTN_COMPILED` by a structural signature (`_lin_desc` of each projection +
geometry):

1. **`_attn_qkv_prep`** — the pre-SDPA chain: `q = rope(unflatten(wq_b(rmsnorm(
   wq_a(x)))))`, `kv_new = rope(rmsnorm(wkv(x)))`, threading `qr` out for the
   Indexer. Pure. The KV-cache write (`append_window`), the window mask, and the
   SDPA stay **outside** it.
2. **`_attn_out_prep`** — the post-SDPA chain: query-RoPE removal, the grouped
   o-LoRA down-projection einsum (the dequantized `wo_a` weight `w_ol` derived by
   the *same* `_o_lora_dense_weight()` path as eager and fed as an input), then
   `wo_b`. Pure (the SDPA that mutates nothing runs between the two tapes).

Plus two folds in `deepseek_v41_moe.py` under the same flag/row-cap (lazy import
of `deepseek_v41._attn_use_compile` — the `deepseek_v41 ← deepseek_v41_moe` import
is one-directional):

3. **`_gate_prefix`** — the MoE gate's pure prefix (score GEMM / `gate_temp`,
   `sqrt(softplus)`, correction bias). The data-dependent argpartition/argsort/
   top-k that builds the fenced routing barrier stays eager.
4. **`_moe_combine`** — the weighted routed sum (f32 accumulator) + shared add.

**Bit-exactness.** Each projection is applied inside the tape by `_apply_lin`
*exactly* as `nn.Linear.__call__` (`x @ w.T`) / `nn.QuantizedLinear.__call__`
(`mx.quantized_matmul(..., transpose=True, group_size, bits, mode)`) — so the tape
is bit-identical to the eager module call whether residents are dense (tiny
config / native-BF16-kept projections) or quantized (q8 gs64 / mxfp8 / mxfp4 /
nvfp4). A quantized matmul is one primitive, which `mx.compile` never
reassociates; `mx.quantized_matmul` compiled == eager was verified bit-exact on
CPU. `mx.unflatten`/`mx.flatten` replace the `.reshape(b,s,…)` calls so the tape
reads no dynamic `.shape` (byte-identical contiguous split, verified). Following
K4, the tape is **fixed-shape + row-cap** (`_ATTN_COMPILE_MAX_ROWS`, default 32),
not shapeless: the projection matmul is `array_equal` with eager up to **7 rows**
and reassociates at **≥ 8** (measured, exactly K4's regime), so the tests keep
every compiled shape at rows ≤ 7 (decode 1 / verify K+1=4 / chunk 5) and the
production cap of 32 confines the tape to decode/verify while prefill chunks fall
to the eager body.

**Tiny-config RMSNorm caveat.** The default hc-compile config's `q_lora_rank=12`
is a **non-power-of-2** reduction that `mx.mean` reassociates under `mx.compile`
(a tiny-config artifact — measured: dim 12 diverges 2/8 seeds, all power-of-2 dims
and the real model's `q_lora_rank=1280` / `head_dim=512` are bit-exact 0/200). The
K22 census/tests use `q_lora_rank=16` so every reduction is compile-stable, matching
the real model's regime.

## 4. What could NOT be compiled, and why

1. **The SDPA (`_sparse_attend`).** It materialises the `[b,s,H,T]` score whose
   `T` (window + compressed rows) grows every token — a dynamic shape — and D512
   is not a fused-SDPA dim (that split-K is K6, a separate prefill lever); a hand
   MLA kernel is Dead-here in the ledger. Kept eager between the two tapes.
2. **The Indexer selection (`Indexer.select` / candidate blocks).** The score is
   over a **dynamic** `n_comp` (grows per token) and the top-k / candidate-block
   mask is data-dependent (`argpartition`, `where`, `cumsum`) — the "Reindex / CSA
   index selection" the task says to keep outside the tape. Only a few layers run
   it; the dominant `attn.reuse` (30 layers) does not.
3. **The MoE routing barrier** (`argpartition`/`argsort`/`take_along_axis` → the
   fenced `indices`) — data-dependent; only the gate's pure score prefix is folded.
4. **The streamed routed switch** (`switch_mlp`, `expert_mlx.py`) — W42's; and the
   **lm head** — W40's; both out of scope and left untouched.
5. **`hc.premix_sinkhorn` / `hc.combine`** — already the K4 lever
   (`MTPLX_DSV41_HC_COMPILE`); its ~4.1k prim/tok Sinkhorn is the single biggest
   source but is K3/K4 territory, not re-implemented here. The census shows it
   unchanged by K22 (asserted).

The KV-cache mutation (`append_window` / `append_compress` / `append_index_k`)
stays outside every tape, so the tapes are pure and the on/off cache state is
identical (asserted).

## 5. Tests (`tests/models/test_deepseek_v41_attn_compile.py`, CPU, tiny, no artifact)

11 tests, all `mx.set_default_device(mx.cpu)`, `_TEST_CAP = 7`:

1. `test_decode_flag_on_off_identical` — 4 decode steps (n=1), `array_equal`.
2. `test_verify_batch_flag_on_off_identical` — a K+1 = 4-row verify forward.
3. `test_prefill_chunked_flag_on_off_identical` — chunk 5 ≤ cap (compiled span).
4. `test_prefill_layer_major_flag_on_off_identical` — same under W30 layer-major.
5. `test_compile_inert_above_row_cap` — s=12 > cap, flag on == flag off.
6. `test_cache_state_identical_flag_on_off` — window / compress_kv / index_k
   identical on/off after prefill + 3 decode steps (the tapes mutate no cache).
7. `test_census_micro_reduction_per_chain` — every compiled chain has strictly
   fewer primitives than eager (asserted **from the census tool**).
8. `test_census_full_model_total_drops` — total prim/tok drops; every `attn.*`
   stage drops; HC / routed-switch / shared-expert are unchanged; gate + combine
   drop.
9. `test_attn_use_compile_gating` — flag off → never; rows > cap → eager.
10. `test_env_default_off` — the import-time default matches the env truthiness.
11. `test_one_qkv_tape_shared_across_layers` — all 8 layers share ONE qkv tape and
    ONE out tape (weights as inputs).

**Verification (CPU, `nice -n 19`, no `-n auto`):**

- `test_deepseek_v41_attn_compile.py` — **11 passed**.
- `test_deepseek_v41_ab_env_levers.py` — **19 passed** (the `attn_compile` arm +
  independence).
- Full `tests/models/test_deepseek_v41_*.py` + `tests/test_deepseek_v41_*.py` —
  **292 passed, 40 skipped** (artifact-gated), no failures. The K22 flag is default
  OFF and the eager body is byte-identical, so every existing gate is unchanged.

## 6. Not changed

- `expert_mlx.py` (streamed switch, W42) and the lm head (`Model.head`, W40) —
  untouched.
- `hc_split_sinkhorn` / `_sinkhorn_*` (K3/W32), the K4 HC tapes, the attention
  SDPA / Indexer selection, `deepseek_v41_cache.py`, the W30/W20 prefill drivers,
  one-shot / decode paths — unchanged; all byte-identical with the flag off.

## 7. Default OFF — the follow-up

`MTPLX_DSV41_ATTN_COMPILE` is opt-in so every existing gate stays green and the
eager per-call graph stays the serving default. The decode/dispatch win is a
GPU-window measurement (**KG-i**): `attn_compile` vs control (and folded into the
K4 `all_levers` stack) — argmax parity (byte-identical decode/verify) + decode +,
`mx.eval`/dispatch counted per arm. The `attn_compile` arm is wired into
`scripts/deepseek_v41/ab_decode_env_levers.py` (`ARM_PRESETS`, `all_levers`,
`ALL_LEVER_ENVS`). On a pass a one-line default flip lands it in serving.
