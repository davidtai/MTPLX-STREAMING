# F5 — DSpark verify-decode compile levers: composition audit

Scope: audit the dispatch-reduction compile levers for the retained DeepSeek-V4.1
Q4 13.87-TPS cell (docs/deepseek-v41/receipts/extension-bank-20260919), decide how
each composes with the installed packed decode lane, and stage a measurement window.
No GPU/Metal was run for this audit (another session held the exclusive lock); all
tests are CPU-pinned. Every claim carries a file:line. Paths are relative to the
repo root; model lines are `mtplx/models/deepseek_v41.py` unless noted.

## Retained ground truth (from `.../extension-bank-20260919/full/result.json`)

- arm `cell16k_ring_v2_draft_attn_pf0`; `dspark.token_ids_sha256 =
  0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac` (the window pin).
- `arm_env`: `HC_COMPILE=0 ATTN_COMPILE=0 ATTN_WIN_MEMO=0` (bound false);
  `ATTN_CORE_COMPILE`/`SMALL_STAGES_FUSED`/`HC_PREMIX_KERNEL` unset; **`DRAFT_COMPILE=1`
  and `SINKHORN_METAL=1` (ON)**; `ATTN_FUSED_PROJ=1 ATTN_LEAN_CASTS=1 ATTN_WO_A_CACHE=1
  RUNNER=v2 WINDOW_RING=1 PREFILL_LAYER_MAJOR=1 PREFILL_SCORE_PATH=lean HEAD_MODE=bf16`.
- `bound_model_levers = {HC_COMPILE:false, ATTN_COMPILE:false, ATTN_WIN_MEMO:false}`.
- `dspark`: 198 cycles / 198 verify_calls, tokens_per_cycle 5.17, accept_rate 0.916,
  decode_tok_s 13.869, decode_wall 73.76 s; `per_cycle_ms {draft 9.95, verify 349.27,
  accept 0.34, commit 0.77}`; `phase_time_s {verify 69.15}` (verify = 93.8% of decode).
- `headline_pass: "untimed"`; `divergence_policy: tie_or_identical`;
  `byte_identical_vs_ar: False` — the retained run itself has ONE accepted `tie_flip`
  at index 297 (`dspark.divergence.class="tie_flip"`, `capture_index_matches_first=true`,
  `ar_top2_margin 0.0`, `dspark_top2_margin 0.25`, `tie_band_used 0.75`).
- engagement: `attn_core_compile_engagement {compiled:0, eager:8640}`;
  `small_stages_engagement {fused_layer_forwards:0, eager_layer_forwards:7920,
  engaged:false}` — 7,920 = 40 backbone layers × 198 verify cycles = the "7,920 layer
  calls" the estimate multiplies.

## Task 1 — per-lever audit

Read-site classes: **import-bound module globals** (read once at import into a global,
re-frozen at construction by `bind_model_levers`) vs **read-at-use** (env or a module
override read every call). This distinction is the toggle mechanism (below).

| Lever | Env / read site | Engagement gate | Engages verify M2..8? | At M=8 (>cap) | Exactness — CPU / native | Memory |
|---|---|---|---|---|---|---|
| **HC_COMPILE** (K4) | `MTPLX_DSV41_HC_COMPILE` (2781); import→global `_HC_COMPILE` (2784); use reads the GLOBAL in `_hc_use_compile` (2911,2924) | `not _HC_COMPILE or _stime.recording()` (2924); `rows ≤ _HC_COMPILE_MAX_ROWS=7` (2795,2929) | M2..7 yes | **eager** (8>7, lever inert) | tiny CPU `mx.array_equal` (test_..._hc_compile.py); **NOT bit-identical at native width** (comment 2769‑2772; HC screen max|Δ| 3.7e‑4) → rounding-class | `_HC_COMPILED` tape cache (2864), 1 tape per (kind,consts) |
| **ATTN_COMPILE** (K22) | `MTPLX_DSV41_ATTN_COMPILE` (1869); import→global `_ATTN_COMPILE` (1870); use reads GLOBAL in `_attn_use_compile` (2658,2669) | `not _ATTN_COMPILE or _stime.is_prefill()` (2669) — decode timing does **not** force eager; `rows ≤ _ATTN_COMPILE_MAX_ROWS=32` (1878,2671) | **M2..8 yes (cap 32)** | engages | **rounding-class even on tiny CPU** — measured max|Δ| 5–7e‑7 at M2..8 here (not the existing test's specific shapes); rounding-class at native width | `_ATTN_COMPILED` (1882), 1 tape per (kind,sig) |
| **ATTN_WIN_MEMO** (K24) | `MTPLX_DSV41_ATTN_WIN_MEMO` (1902); import→global `_ATTN_WIN_MEMO` (1903); used at ~1585 (window mask reuse) | fires only when query positions are the SAME object and (T,window,b,s) match across layers (comment 1896‑1899) | window-masked layers | still engages (position-object match, not row-capped) | **byte-identical** (pure host-dispatch mask reuse; reused mask is the same array) | retains ONE `[b,s,T]` window mask on the per-forward `shared` runtime for the whole forward — a **prefill-memory interaction** (why it was bound false for provenance) |
| **ATTN_CORE_COMPILE** | `MTPLX_DSV41_ATTN_CORE_COMPILE` (2392); **read-at-use** `_resolve_attn_core_compile` (2442‑2456); used at 1320 | `_resolve_attn_core_compile() and rows ≤ _ATTN_CORE_COMPILE_MAX_ROWS=8` (1320,2395) | **M2..8 yes (cap 8)** | engages at 8 | rounding-class; **inert on the tiny CSA config** (needs native selected-key geometry — measured compiled=0); receipt `attn_core_compile_engagement` (2432) | `_ATTN_CORE_COMPILED` (2404), 1 tape per (b*s, CSA k) |
| **SMALL_STAGES_FUSED** (K35) | `MTPLX_DSV41_SMALL_STAGES_FUSED` (2988); **read-at-use** `_small_stages_fused_enabled`→`_env_truthy` (3026‑3030); use `_small_stages_use` (3033,3044) | `not enabled or _stime.recording()` (3044); `rows ≤ _SMALL_STAGES_MAX_ROWS=7` (2996,3049) | M2..7 yes | **eager** (8>7) | tiny CPU `mx.array_equal` (test_..._small_stages_fused.py); **shares the compiled HC premix** (`_hc_mixes_split`, 2798/2809) → same native-width rounding-class as HC_COMPILE | `_SMALL_STAGES_COMPILED` (3000), 3 tapes/layer geometry; counters 3007 |
| **HC_PREMIX_KERNEL** | `MTPLX_DSV41_HC_PREMIX_KERNEL` (267); **read-at-use** `_hc_premix_use_kernel` (288) | requires env truthy **AND** `mx.metal.is_available()` **AND** `default_device==gpu` (294‑298) | **GPU only — INERT on CPU** | (GPU only) | GPU parity-gated 1e‑6 + argmax-exact (`test_hc_premix_kernel_parity_gpu`); never built/dispatched on CPU | `_HC_PREMIX_KERNELS` (275), 1 kernel per (hc,iters,eps) |
| **DRAFT_COMPILE** (K33, already ON) | `MTPLX_DSV41_DRAFT_COMPILE` (`deepseek_v41_dspark.py:115`); **read-at-use** with module override `_DRAFT_COMPILE` (122) in `_draft_compile_on` (137‑140); use `_draft_use_compile` (228) | `_draft_compile_on() and rows ≤ _DRAFT_COMPILE_MAX_ROWS=32` (126,228) | drives the DSpark DRAFT head chains (not the verify layer forward) | engages (cap 32) | byte-identical (draft tapes; `test_deepseek_v41_dspark_draft_compile.py`) | `_DRAFT_COMPILED` (131) |

Key consequences:
- At **M=8** (the 23 of 198 retained verify cycles at K+1=8): HC_COMPILE and
  SMALL_STAGES fall to **eager** (7-row cap), while **ATTN_COMPILE (32) and
  ATTN_CORE_COMPILE (8) still engage**. So no arm compiles the full M=8 verify unless
  the caps are raised (Task 4).
- The `_stime.recording()`/`is_prefill()` bail-outs (2924/2669/3044) force these levers
  eager under `--stage-timing`, so the retained headline pass is `untimed` (result.json
  `dspark.headline_pass="untimed"`); the window keeps that.

### How run_full.py binds the three levers false, and the minimal decode-only change

Binding (all three are import-bound globals):
1. `_bound_prefill_flags = COMPATIBILITY['explicit_bound_model_levers']` (run_full.py:506)
   = `{HC_COMPILE:false, ATTN_COMPILE:false, ATTN_WIN_MEMO:false}`
   (`.../sources/compat/installation.json:17‑21`).
2. For the arm, run_full.py:507‑510 rewrites `ab.ARM_PRESETS[arm]` to set each bound
   key's env to `"0"` **before** the model is built.
3. At construction `bind_model_levers(model_module)`
   (`scripts/deepseek_v41/bench_standard_shape.py:1477‑1498`) reads those env keys and
   **`setattr`s the module globals** `_HC_COMPILE`/`_ATTN_COMPILE`/`_ATTN_WIN_MEMO`
   (bindings 1483‑1486, `setattr` 1496) → frozen `False`. The result is stored as
   `args._dsv41_bound_model_levers` (`ab_decode_env_levers.py:3261`).
4. run_full.py:587 asserts `loaded_args._dsv41_bound_model_levers == _bound_prefill_flags`
   ("explicit flags were not bound at construction" otherwise).

Because `_hc_use_compile`/`_attn_use_compile` read the GLOBAL, **an env change after
construction does not move them — only writing the global does** (proven: the K22 unit
test flips `dv41._ATTN_COMPILE` directly, `test_deepseek_v41_attn_compile.py:103`; the
in-tree `_moe_compile_window`, `deepseek_v41_dspark.py:243‑248`, sets
`_dv41._ATTN_COMPILE=True` around a scope and restores it).

Minimal construction-time change to enable a chosen set FOR DECODE ONLY, prefill
exactly as retained:
- Keep the three bound levers' env at `"0"` so `bind_model_levers` freezes them False
  for construction **and prefill** (prefill runs identically — crucial for
  ATTN_WIN_MEMO's per-forward retained window mask, the prefill-memory interaction).
- At the **post-prefill quiescent boundary** — the runner's `observe_prefill_boundary`
  (run_full.py:768‑785), AFTER `callback(info)` (779) and `growth_transition()` (781,
  which runs `install_model`) — write the module globals (`deepseek_v41._HC_COMPILE=True`
  etc.). This is implemented in `scripts/deepseek_v41/f5_compile/f5_decode_levers.py`
  (`enable_decode_levers` writes globals for the import-bound levers) and staged into
  the runner by `stage_f5_runner.py` (EDIT 2).
- The read-at-use levers (SMALL_STAGES_FUSED, ATTN_CORE_COMPILE, HC_PREMIX_KERNEL) are
  enabled by setting their env at the same boundary; their row caps already keep them
  out of the large prefill chunks, so prefill is unaffected regardless. DRAFT_COMPILE is
  already on. Since prefill chunk rows ≫ every cap, and ATTN_COMPILE additionally guards
  on `is_prefill()`, the 84-row / 110e9-budget prefill envelope does **not** change.

## Task 2 — composition with the installed packed lane

The retained run replaces every routed layer's `switch._run` with `PackedDecode.run`
(`.../sources/packed/plane_lane.py:311`, `install`). Contract:
- `PackedDecode.run(self, x, indices, *, shared_work)` (plane_lane.py:216) — `shared_work`
  is **keyword-only, no default**; the shared expert is computed **iff `shared_work is not
  None`** (250‑252); returns `(output, shared)` (286).
- The streamed switch's `__call__(x, indices)` calls `self._run(x, indices,
  shared_work=None)` and returns only `output` (`expert_mlx.py:2509‑2515`);
  `run_with_shared_overlap(x, indices, cb)` calls `self._run(..., shared_work=cb)` →
  `(output, shared)` (2517‑2531). `_run` is exactly the attribute `switch._run = runner.run`
  replaces.
- MoE.__call__ (`deepseek_v41_moe.py:467‑495`) routes via `_routed_shared_route` (490);
  the retained `verify_shared_overlap` selects `_run_routed_shared_overlap` (406‑419) →
  `run_switch_with_shared_overlap` (`expert_mlx.py:3895‑3906`) → the switch's
  `run_with_shared_overlap` → `_run(shared_work=cb)` = `PackedDecode.run(shared_work=cb)`.

Per-lever verdict (does it still call `switch._run` correctly; shared computed
twice / once / never):

- **HC_COMPILE (K4)** — eager DecoderLayer branch. `attn_and_moe_input` runs the
  compiled HC prep tapes (3455‑3477) but calls `self.attn(...)` unchanged (3469) and
  returns `moe_input`; `__call__` then calls `self.mlp(moe_input)` (3655) → MoE.__call__
  → the SAME overlap path as control → `PackedDecode.run(shared_work=cb)`. K4 never
  touches the MoE/switch/shared path. **`switch._run` is called with the right args
  (`shared_work=cb`); shared computed ONCE (overlapped).** Composes cleanly.
- **ATTN_COMPILE (K22)** — only swaps the QKV/output prep tapes inside
  `Attention.__call__` (via `_attn_use_compile`); `self.mlp(moe_input)` and the switch
  path are byte-for-byte control. **`switch._run(shared_work=cb)`; shared ONCE.** Cap 32
  → engages at all M2..8.
- **SMALL_STAGES_FUSED (K35)** — DecoderLayer.__call__ takes `_fused_small_decode`
  (3642‑3644), which **bypasses MoE.__call__ / `_routed_shared_route`**. It (a) reads the
  shared expert weights `warrs` (3537‑3538) and computes `shared_out` INSIDE seg2's
  compiled tape (3539‑3544, via the raw weights, not `se.__call__`); (b) calls
  `self.mlp.switch_mlp(xf, indices)` (3546) → `__call__` → `_run(shared_work=None)` =
  `PackedDecode.run(shared_work=None)` → **the lane does NOT compute the shared expert**
  (250 skipped, returns `(routed, None)`); (c) folds `routed + shared_out` in seg3
  (3548‑3550). **`switch._run` IS still called** with valid args (`shared_work=None`, no
  crash). **The shared expert is computed EXACTLY ONCE — in seg2's fused tape, not in the
  lane; not twice, not never.** The K1 `verify_shared_overlap` `shared_work` overlap is a
  **NO-OP under K35** (the overlap route is bypassed) — matches the ledger.
  - CPU-verified: `tests/models/test_deepseek_v41_f5_compile_composition.py`
    `test_switch_run_contract_and_shared_work` (lane sees `shared_work=None` under K35,
    `cb` under control) and `test_shared_expert_computed_exactly_once` (module-level
    shared calls: control n=layers, K35 0; outputs `mx.array_equal` → single correct
    shared under K35).
  - **Performance composition caveat (not a bug):** under control the shared expert is
    computed inside the lane overlapping SSD miss reads (`shared_work()` at
    plane_lane.py:251, right after `begin_split_route`). Under K35 it is computed in seg2
    **before** the switch call, so the shared/miss-read overlap is LOST. K35 folds the
    shared into the fused tape but forfeits K1's overlap — a real tradeoff to measure.
- **Projection scheduling install** (`projection_install.py`): rebinds `switch._run`
  AGAIN to a scheduled variant = `PackedDecode.run` source with `self.issue_next()`
  inserted **before** the `if shared_work is not None:` anchor (`scheduled_run_source`
  52‑60; rebind 108; the pre-anchor position means `issue_next()` fires for BOTH
  `shared_work=None` (K35) and `cb` (control)). The next-layer wo_a expansion is issued
  once per layer regardless of the lever, and the attention output projection
  (`ScheduledOutput`, `owned_projection.py`/`projection_install.py:109`) is invoked inside
  `self.attn(...)`, which K4/K22/K35 all call unchanged. **Composes cleanly — no double,
  no missing.** (It validates `type(runner) is PackedDecode` + `switch._run.__func__ is
  PackedDecode.run` at 92‑96, so it must run before any lever/probe rebind — which it does,
  at load, before decode.)
- **Hybrid lookup install** (`hybrid_install.py`): a host-side rewrite of `_decode_cycles`
  that extends the D5 draft to D7 via causal past-text match (`lookup.py`), asserting the
  MLX-op sequence is UNCHANGED (`rewrite` 9‑43, invariance at 41). It computes the verify
  row count M (6..8) but issues NO `mx` ops and never calls `switch._run`. It composes
  trivially — it FEEDS the M distribution the levers gate on. Because M can be 8 (>7 cap),
  HC_COMPILE/SMALL_STAGES fall eager on the 23 M=8 cycles while ATTN_COMPILE/ATTN_CORE
  still engage.

## Task 4 — raise the 7-row caps to 8 (separate construction-time option)

One-line each: `_HC_COMPILE_MAX_ROWS = 7` (2795) and `_SMALL_STAGES_MAX_ROWS = 7` (2996).
Implemented as a SEPARATE option (`enable_decode_levers(..., caps_at_8=True)` in
`f5_decode_levers.py`, armed by `MTPLX_DSV41_F5_CAPS8=1`) that sets both to 8 at the
boundary and clears the tape caches; **defaults are never changed**
(`test_f5_caps_at_8_option` asserts a fresh import is still 7).

Exactness class: raising to 8 lets the M=8 verify batch (23 cycles) compile. The
≥8-row band **reassociates** (~5e‑7..1.2e‑6 on tiny, comment 2991‑2994; native width is
already rounding-class), so the option moves M=8 from "eager / exact" to
**rounding-class** — greedy-near-tie, not bit-identical. Only defensible under the
`tie_or_identical` policy with the divergence classifier armed (the window does this).

## Task 6 — seconds-per-run estimate (census-only; explicitly tempered)

Inputs: verify phase 69.15 s over 198 cycles (349.27 ms/cycle); **7,920 layer calls**
(40×198); census ~350 small-stage primitives/layer eager with the Sinkhorn Metal kernel
on vs ~107 compiled (K35 table) → ~243 removable/layer for K35 (HC_COMPILE folds the two
HC premix+combine chains, a subset; the tiny-CPU census here shows one HC prep chain
collapsing 259→111 primitives); per-dispatch host-encode 0.02–0.05 ms (comment 2942).

Naive census upper bound (K35): 243 × 7,920 × [2e‑5, 5e‑5] s = **[38.5, 96.2] s**.
HC_COMPILE alone (≈ the two HC chains, ~half): **≈ [19, 48] s**.

**This is an over-count and must not be quoted as achievable.** It exceeds the entire
69.15 s verify phase, because most of those dispatches are HIDDEN behind the ~2 ms/layer
GPU work the routing barrier waits on ([[b1-decode-dispatch-removal-hides]]). The one
full HC screen already measured this: the isolated `[1,6,4,5120]` HC probe was 2.27×
(1.6269→0.7167 ms), yet the **full-cycle verify time per cycle was essentially unchanged
(0.3882→0.3880 s, ~0.05%)** and the +0.86% TPS was confounded by a different token stream
(204 vs 206 cycles) — `post-prefill-cache-growth-20260917/followup-hc-screen.json`.

Credible removable time per run: **0 to a small fraction** of the naive bound (the HC
screen puts HC_COMPILE at ≈0 of a full screen). The A2 `TimedPackedDecode` stamp probe
replaces this guess by MEASURING the only genuinely removable slice — the inter-layer
graph-build gap (`next call t0 − this call t5` within a verify forward) — split from the
draft/accept/commit gap that straddles forwards.

## Prior evidence to design around (coordinator addendum)

`post-prefill-cache-growth-20260917/README.md` "HC follow-up: not promoted" +
`followup-hc-screen.json`: HC_COMPILE screened once on the full model, NOT promoted —
not bit-exact at native width (max post-output |Δ| 3.7e‑4), first candidate/control
difference at **index 480 NEVER classified** (no cached AR-logits row there; only the
AR tie at 297 was cached; `cached_ar_logits_row` raises on a new index, run_full.py:822),
guard exited 4 on the digest gate (run_full.py:834‑836). The F5 window closes that gap:
the staged runner (EDIT 1) keeps the AB driver's LIVE `_ar_logits_row_at_index`
(`ab_decode_env_levers.py:3892`, provenance-clean — forwards the AR reference prefix, no
candidate replay) so a divergence at ANY index gets a W120 tie_flip/divergent verdict
with the contested-margin / tie-band keys (`_dspark_divergence_rule`,
`ab_decode_env_levers.py:3805`). Every arm reports verify_ms/cycle AND cycles, not TPS
alone.

## CPU tests (green)

`nice -n 19 pytest tests/models/test_deepseek_v41_f5_compile_composition.py
tests/models/test_deepseek_v41_f5_timed_plane_lane.py` (MLX pinned to CPU, no `-n auto`):

```
14 passed, 2 warnings in 2.08s
```

Coverage: the `switch._run` contract + shared-work per arm at M∈{2,4,6,7,8};
`lever on==off` (`mx.array_equal` HC/K35; ATTN_COMPILE reported max|Δ| 5–7e‑7 —
rounding-class even on tiny CPU); ATTN_CORE inert on tiny; engagement counters; one
routing barrier per routed-layer call (K35 adds none); shared expert computed exactly
once (control via lane / K35 via seg2, outputs identical); per-layer primitive census at
M=6 (attn-prep 259→111 for HC and K35 seg1); the decode-only enable mechanism; the
separate caps@8 option; and — for A2 — `TimedPackedDecode.run` inserts exactly 6 stamps +
1 note, adds no `mx` op, and removing them recovers `PackedDecode.run` byte-for-byte
(round-trip), composes on the projection-scheduled variant, and the summarizer's bucket /
inter-layer-gap math.

## Files (all under this branch `f5/compile-levers`, not pushed)

- `scripts/deepseek_v41/run_f5_compile_window.sh` — the window (write, do not run): CPU
  preflight → stage patched runner → arms A / A2 / B / C / D / E(optional) / F.
- `scripts/deepseek_v41/f5_compile/f5_decode_levers.py` — post-prefill decode-only lever
  enable + caps@8 + A2 probe arming.
- `scripts/deepseek_v41/f5_compile/timed_plane_lane.py` — `TimedPackedDecode` (stamp
  probe; run derived from `PackedDecode.run` with a round-trip op-identity guarantee).
- `scripts/deepseek_v41/f5_compile/stage_f5_runner.py` — stage the 2-edit patched
  run_full.py copy (round-trip checked).
- `scripts/deepseek_v41/f5_compile/f5_preflight.py` — CPU dependency + 111-row
  admissibility gate (refuses before any service unload).
- `scripts/deepseek_v41/f5_compile/f5_readout.py` — per-arm digest/verdict/verify_ms/
  cycles/engagement readout.
- `tests/models/test_deepseek_v41_f5_compile_composition.py`,
  `tests/models/test_deepseek_v41_f5_timed_plane_lane.py`.

Retained runner sha256 `13bcdfe4fe583d8b465e69e7effbb8bf2f20d4d8b5baacb12e252f85f95149be`;
retained `plane_lane.py` sha256
`1acad9e24c37e5c618b2d8e2e98fb93eb94b5476d5c6de6fa0ee054db468ba54` (both pinned in the
window preflight / probe).
