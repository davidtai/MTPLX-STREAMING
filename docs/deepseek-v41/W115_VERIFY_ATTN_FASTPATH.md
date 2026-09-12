# W115 — Verify-attention fast path (decode SDPA core for the K+1 verify batch)

Lever: `MTPLX_DSV41_VERIFY_ATTN_FASTPATH` (default OFF) + `MTPLX_DSV41_VERIFY_ATTN_MAX_ROWS`
(default 8). Arm: `cell16k_ring_v2_draft_attn_vfast` = `cell16k_ring_v2_draft_attn` + the lever.
Branch: `w115/verify-attn-fastpath` off `int/w97f-lanes` (7ca25d61a).

## 0. Verdict up front

The DSpark verify is a small-M (`1 < rows = K+1 <= 8`) target forward that already
runs through the *same* `Attention._attend` as the M=1 AR decode. The M=1 decode
machinery — W101 fused projections, the W97 cached pre-transposed `wo_a`, W99 lean
casts, and the fused/compiled SDPA **core** (the K29 kernel on GPU, the W97
core-compile tape on CPU/GPU) — is row-generic and accepts the K+1 batch. **But the
shipping DSpark attention arm `cell16k_ring_v2_draft_attn` deliberately keeps the
eager SDPA core** (K29 and core-compile both OFF — see its preset comment: "The K29
fused decode core … is DELIBERATELY NOT armed … so the direct A/B … isolates … the
lean stack + fused proj without the fused core"). So the verify runs the eager
per-row selected-key core, which scales ~linearly with the row count — window-43
measured the verify attention at ~372 ms/cycle for the 6-row batch vs ~64 ms/token
(1.6 ms/layer) at M=1, i.e. ~6×.

W115 adds a single, phase-scoped, default-OFF lever that **auto-arms the fused decode
SDPA core (K29 + core-compile) and the W101 fused projections for the verify rows
only** (`1 < rows <= MAX_ROWS`, scoped to the `decode_verify` attention phase),
independent of whether the M=1 core levers are armed. `rows == 1` is untouched. The
batched verify then reuses the same single-dispatch core as M=1 — one SDPA over the
selected window/compress KV with the per-query causal mask among the K+1 rows.

Numerics: **ROUNDING-CLASS, not byte-identical** to the eager per-row core (the
fused/compiled core reassociates the fp32 softmax) — greedy-identical, never
bit-identical, so the lever is in `ROUNDING_CLASS_ENVS`. The greedy verify stays
authoritative, so greedy DSpark == greedy AR still holds.

## 1. As-is — which functions the 6-row verify traverses (and why it costs ~6× M=1)

The verify forward is `model(mx.array([[primary, d1..dK]]), cache=…)`
(`deepseek_v41_dspark_decode.py:822`, inside `_verify_routing_context` which stamps
`attention_phase("decode_verify")` and `expert_routing_phase(DECODE)`). `b=1, s=K+1`,
so every backbone layer's attention runs `Attention._attend` with `rows = b*s = K+1`.

Per-layer traversal of the verify (`mtplx/models/deepseek_v41.py`), for the shipping
`cell16k_ring_v2_draft_attn` env (`selected_keys=1`, `attn_fused_proj=1`,
`attn_lean_casts=1`, `wo_a_cache=1`; `decode_attn_kernel` and `attn_core_compile`
OFF):

| stage | function (file:line) | M=1 | K+1 verify (as-is) |
|---|---|---|---|
| qkv-prep | `_attend` → `_qkv_prep_fused` (`deepseek_v41.py:1427`, `:1648`) | fused (`_fused_proj_use(1)`) | **fused** (`_fused_proj_use(K+1)` — rows≤8, so already engages on GPU) |
| SDPA core | `_attend` → `_sparse_attend_selected` (`:1492` `use_selected`, `:1140`) | eager gathered core, but 1 row | **eager gathered core, K+1 rows ≈ K+1×** |
| — K29 gate | `_decode_attn_kernel_use(q)` (`:1198`, def `:2180`) | declined (lever off) | **declined (lever off)** → eager |
| — core-compile gate | `_resolve_attn_core_compile()` (`:1231`) | declined (lever off) | **declined (lever off)** → eager |
| out-prep | `_attend` → `_out_prep_fused` (`:1662`) | fused | **fused** |

The eager selected core is the block at `deepseek_v41.py:1238-1266` (`_sparse_attend_selected`):
`_window_selected_idx` + `_gather_rows` build a **per-query** gathered operand
`KVg [b, s, k, hd]` (`:1179`), then `mx.einsum("bshd,bskd->bshk", …)` (QK), a masked
sink-softmax, and `mx.einsum("bshk,bskd->bshd", …)` (PV). At M=1 this is `[1,1,H,k]`;
at K+1 it is `[1,K+1,H,k]`, so the gather, the two einsums and the softmax are all
~(K+1)× the M=1 work, and the shared KV read is never amortised across the K+1 rows.
That is the ~6× (window-43 `verify_stage_timing`: `attn.reuse` 8.5 ms/call at K+1 vs
~1.7 ms at M=1; `attn.full`/`attn.reindex`/`attn.swa_only` similar; Σ ≈ 372 ms/cycle).

The projections do NOT cost 6×: `_fused_proj_use(K+1)` already returns True (rows≤8),
so the K+1 qkv/out projections run the W101 fused kernels (row-generic, one dispatch
each — W105 priced the five projections at M=6 = 415 µs/layer mxfp8). The remaining
6× is the **eager SDPA core only**. (The window-43 receipt shows
`fused_proj_engagement.rows = 10240 = 256×40` = the AR pass only, because the bench
reads that counter after the AR pass and never re-reads it for the DSpark pass — a
counter-plumbing gap, fixed in §3, not evidence the verify's projections ran eager.)

## 2. Design

`_verify_attn_fastpath_use(rows)` (`deepseek_v41.py:2532`) returns True iff: the lever
is armed, the current attention phase is `decode_verify`, and `1 < rows <=
min(MAX_ROWS, _DECODE_ATTN_KERNEL_MAX_ROWS=8)`. The three small-M core gates OR this
in for `rows > 1`, so the verify auto-arms the fused core even with the M=1 levers off:

- `_decode_attn_kernel_use(q)` (`:2180`): `if not _resolve_decode_attn_kernel(): … if
  not _verify_attn_fastpath_use(rows0): return False`. On GPU the verify batch routes
  through the K29 fused decode kernel — one dispatch over the gathered `[rows,k,hd]`
  operand with the `[rows,k]` valid mask (the reference already handles `b*s>1`; see
  `test_deepseek_v41_decode_attn_kernel.py::test_decode_and_verify_route_through_kernel`).
- the core-compile gate (`:1231`): `(_resolve_attn_core_compile() or
  _verify_attn_fastpath_use(rows)) and rows <= _ATTN_CORE_COMPILE_MAX_ROWS`. On CPU
  (and on GPU when K29 declines) the verify runs the fixed-shape `mx.compile` core
  tape keyed on `(b*s, k)` geometry.
- `_fused_proj_use(rows)` (`:2439`): `not (_resolve_attn_fused_proj() or
  _verify_attn_fastpath_use(rows))` — makes the lever self-contained (the verify
  projections fuse even if `attn_fused_proj` is unset).

Because the gate keys off the SAME `_attend`/`_sparse_attend_selected` path, the fast
path is literally "one SDPA over the selected window/compress KV with the correct
causal mask among the K+1 rows": the per-query causal mask is `_window_selected_idx`
(`:1178`), which for verify positions `[T, T+1, …, T+K]` makes each query attend only
to window rows `<= its own position` (intra-block causality is free — positions
increase within the block). The SWA ring exposes rows `[T, T+K]` because
`append_window(kv_new)` (`:1476`) appends all K+1 new rows before the SDPA. The
compress/index (ratio) lanes see the same rows the prefill path would: `_compressed`
publishes `topk_mask`/`selected_idx` once on the index-source layer and the reuse
layers read it, unchanged by the lever (the lever only swaps the SDPA math, never the
selection). Nothing about which keys/experts are gathered changes — only the fp32
softmax accumulation order, hence rounding-class.

**Why not just arm K29 in the arm?** K29 is rounding-class, and the arm was designed
to isolate the lean stack + fused proj *without* the fused core. W115 keeps that arm
intact and adds a separate, measurable lever so the batched-verify core is an
independent A/B knob with its own engagement census. Phase-scoping (`decode_verify`)
means the auto-arm never trips for a short (`<=8`-row) prefill chunk — a caveat the
raw M=1 core gates carry but this one does not.

### K4/K35 compile compatibility

- **K29 (`decode_attn_kernel`)** is the fused decode core on GPU; it already accepts
  the `b*s` verify batch (each row is its own S=1 "batch"), so the batched shape is
  handled, not a fallback. On CPU it declines (no Metal) and the core-compile tape
  runs instead.
- **W97 core-compile (K-lesson K35)** compiles the selected-key core to ONE fixed-shape
  tape per `(b*s, k)` signature (`_attn_core_compiled`, `:2477`-ish). The verify's
  `b*s = K+1` is a distinct signature from M=1's `b*s = 1`, so it builds its own tape
  (bounded: one per distinct verify width). It handles the batched shape; it does NOT
  fall back. Verified on CPU: the verify's `attn_core_compile_engagement.compiled`
  counts every verify layer-call (§4).
- **K22 attn-chain compile** (`_attn_use_compile`, rows≤32) is superseded by the W101
  fused projections when `attn_fused_proj`/the verify lever is on (fused proj is
  checked first at `:1449`/`:1554`), so it is inert here — no interaction.
- The lever composes with `--stage-timing`: the timed pass forces the core eager (the
  W37 recording guard), so the compiled/K29 core is measured OFF there by design; the
  untimed headline pass carries the engaged number (mirrors K35/`small_stages`).

## 3. Counters + env

Env (read at use, never frozen at import):

- `MTPLX_DSV41_VERIFY_ATTN_FASTPATH` — bool, default OFF (`_resolve_verify_attn_fastpath`).
- `MTPLX_DSV41_VERIFY_ATTN_MAX_ROWS` — int ≥ 2, default 8 (`_resolve_verify_attn_max_rows`).
  Effective cap is `min(MAX_ROWS, 8)` (the K29/core tapes hard-cap at 8).

Both are in `ab_decode_env_levers.ALL_LEVER_ENVS`, in `_preset`, and in
`openai._DSV41_LEVER_ENV_KEYS` (the served-log superset drift guard). The lever is in
`ROUNDING_CLASS_ENVS`, so `cell16k_ring_v2_draft_attn_vfast` is classified
rounding-class automatically (a verify token-id sha mismatch vs control is EXPECTED,
not a failed exact lever).

Engagement census (`deepseek_v41._verify_attn_fastpath_engagement()`):

- `verify_attn_fastpath_calls` — verify (`decode_verify`) `_attend` calls the fast
  path governed (= verify layers × cycles).
- `verify_attn_fastpath_rows` — Σ `b*s` over them (= calls × (K+1) per depth).
- `verify_attn_fastpath_fallbacks` — `{reason: count}` for lever-on verify calls that
  could NOT take the fast core: `rows_gt_max` (batch wider than `MAX_ROWS`) or
  `selected_keys_off` (the fused/compiled core lives on the selected-key path; without
  `SELECTED_KEYS` the verify runs the masked `_sparse_attend` core, so only the
  projections can fuse).

Receipt wiring (`ab_decode_env_levers.py`): the AR-pass top-level receipt carries
`verify_attn_fastpath_engagement` (0 — M=1 has no verify batch). The **dspark block**
resets the decode-core / fused-proj / verify-fast-path counters right before the
UNTIMED headline pass and reports them after it, so the receipt's `dspark` block now
carries `verify_attn_fastpath_engagement`, `attn_core_compile_engagement`,
`decode_attn_kernel_engagement`, and **`fused_proj_engagement` counting the verify
rows** — closing the window-43 "AR only" gap.

## 4. Tests (`tests/test_deepseek_v41_w115_verify_attn_fastpath.py`, CPU, no Metal)

13 tests, all CPU-pinned (MLX default cpu), no model download, no checkpoint, <1.5 GB RSS:

- `test_fastpath_resolver_default_off_and_parse`, `test_max_rows_resolver_default_and_parse`
  — resolvers (default off/8, truthy parse, fail-fast; `MAX_ROWS < 2` raises).
- `test_use_gate_requires_phase_rows_and_flag` — `_verify_attn_fastpath_use` fires only
  when armed AND phase == `decode_verify` AND `1 < rows <= min(MAX_ROWS, 8)`.
- `test_verify_autoarms_k29_and_fused_proj_when_m1_levers_off` — with the M=1 levers
  OFF and a spied GPU, the verify batch auto-arms K29 + fused proj; `rows == 1` and the
  `prefill` phase never do; above the cap declines.
- `test_m1_kernel_lever_still_governs_rows1` — the widening is additive (arming the M=1
  `decode_attn_kernel` behaves exactly as before, independent of the verify lever).
- `test_greedy_dspark_fastpath_equals_ar[1..5]` — greedy DSpark-with-fastpath ==
  greedy AR over the tiny model at depths 1..5; engagement `calls>0`, `rows ==
  calls*(K+1)`, no fallbacks, `attn_core_compile.compiled >= calls`.
- `test_verify_logits_rounding_class_and_argmax_exact` — per-row verify-logit max|Δ|
  fast-core vs eager-core, argmax-exact.
- `test_fallback_rows_gt_max`, `test_fallback_selected_keys_off` — the two fallback
  reasons populate.

Tails:

```
$ .venv/bin/python3 -m pytest tests/test_deepseek_v41_w115_verify_attn_fastpath.py -s
[W115] verify logits per-row max|delta| = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (max 0.000e+00);
       argmax_on=[37, 31, 62, 37, 13, 61] argmax_off=[37, 31, 62, 37, 13, 61]
======================== 13 passed, 2 warnings in 4.45s ========================
```

Per-row max|Δ| is **0.0 on CPU**: the W97 core-compile tape is bit-exact to the eager
core inside MLX's CPU `mx.compile` regime (same fp32 ops, same order), so on CPU the
fast path is byte-identical, not merely greedy-identical. The **rounding-class**
difference the lever is classified for is the GPU K29 kernel's online-softmax tile
reduction (W60: greedy-identical, max|Δ| ~1e-6 vs eager), exercised only under
`MTPLX_GPU_PARITY=1` on a Metal host (`test_deepseek_v41_decode_attn_kernel.py`).

Named parity tests still green (CPU): `test_deepseek_v41_verify_single_barrier.py`,
`test_deepseek_v41_device_route_parity.py`, `tests/models/test_deepseek_v41_dspark_decode.py`,
plus `decode_attn_kernel`, `attn_fused_proj_w101`, `attn_core_compile_w97`,
`selected_keys`, `verify_switch_batched`, `lever_child_env_w46`, `ab_env_levers`,
`w110_record_hash`, `bench_scripts` (the superset + arm-classification guards).

## 5. GPU A/B — do NOT run here (orchestrator, under the flock)

One cell, DSpark depth 5, on the standard 16,384 + 1,024 sweep cell prompt, in
`--decode-mode dspark --stage-timing`:

```
cell16k_ring_v2_draft_attn        # control: eager verify SDPA core
cell16k_ring_v2_draft_attn_vfast  # candidate: + MTPLX_DSV41_VERIFY_ATTN_FASTPATH=1
```

Proof to read from the two `dspark` receipt blocks (control vs candidate):

1. **`fused_proj_engagement.rows` includes the verify rows** on both, and the
   `verify_attn_fastpath_engagement` block shows `calls = verify_layers × cycles`,
   `rows = calls × (K+1)`, `fallbacks = {}` on the candidate (and `calls = 0` on the
   control — lever off).
2. **`decode_attn_kernel_engagement.calls` > 0** on the candidate (the verify routed
   through K29) and `0` on the control.
3. **`per_cycle_ms.verify_ms` down** and the timed `verify_stage_timing` attention
   stages (`attn.full`/`reindex`/`reuse`/`swa_only`) down.
4. **`route_probe` `hot.eval_indices` sums down**: that barrier realises the lazy
   per-layer graph (attention included) before the MoE route host-sync, so a cheaper
   verify attention shortens the in-barrier wait.
5. **Greedy verify authoritative**: byte-identity to the AR reference is NOT the bar
   (rounding-class); the dspark divergence classifier must show tie-flip-only (if any).

Expected: verify attention ~372 → ~85–110 ms/cycle (the batched core ≈ 1.2–1.5× the
M=1 64 ms/token forward, the 16K KV read once), and the finding's in-barrier
(`eval_indices`) ~408 → ~100 ms/cycle. Watch peak memory (the fused path + wo_a cache
each hold a per-layer wo_a copy resident at the 16K cell).
