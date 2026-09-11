# W35 — DSpark MTP serve-gate fix + AR-lane profile shaping

Branch `feat/deepseek-v41-w35` off integration `18c7cff27`. CPU-only, no artifact
model load (manifest/config JSON + `expert_artifact_status` filesystem checks
only), peak worker RSS **152 MB**. Not pushed.

Triggered by the window-11 GPU smoke: `mtplx serve --model <mxfp4>
--generation-mode mtp` exited at startup with `promoted streamed profiles are
AR-only in MTPLX 2.3.1rc1`, and the AR serve (no flag) showed a generic memory
plan (75G engine / 48G session bank / 696K context) with the native-MTP head
installed.

## Fixes (all gated on the native-MTP predicate; hy3/glm + AR byte-identical)

### 1. The second AR-only gate (the actual startup blocker)
`grep "promoted streamed profiles are AR-only"` found **two** sites. W21 gated
`mtplx/expert_cli.py`; the one that fired at startup is
`mtplx/commands/public.py:_streamed_generation_mode_error`, called on the serve
path **before** `expert_streaming_load_kwargs`. It now takes `model_path` and
returns `None` for the DSpark native-MTP artifact (same `_is_native_streamed_mtp`
predicate); both serve call sites pass `runtime_model`. Non-native (hy3/glm)
streamed profiles still get the AR-only error.

### 2. Daemon AR/MTP decision + MTP cap (server/openai.py)
The daemon engagement block previously forced `load_mtp=False,
generation_mode="ar"` for **any** streamed profile — which would defeat the
`--generation-mode mtp` glue even after fix #1. It now reads the resolved
streamed kwargs' `mtp` value: `mtp=True` (DSpark native head requested) keeps MTP
(`generation_mode="mtp"`, and the MLX cap is computed against the
`mtp_included=True` spec so the wired `mtp.*` residents are priced); every other
case forces AR (unchanged).

### 3. Served context = the profile's KV plan (issue #3)
The engagement block now caps `args.context_window` at
`stream_config.max_live_kv_tokens` (16,384 for `deepseek-v41-mxfp4-75`) unless the
operator set one explicitly, so the outer plan is not the machine-bound 696K
window.

### 4. Session bank yields to the expert cache (issue #2)
Added `MTPLX_SESSION_BANK_MAX_BYTES=2GiB` to the profile `child_env`. The engine
auto-sizes the bank to ~48 GiB (half the RAM surplus after weights) with no
awareness of the 64 GiB streamed expert cache, which pushed the allocator past
1.0. Capping it to 2 GiB (inside the 7 GiB runtime reserve headroom) yields to the
expert cache while leaving room for W26 in-memory near-prefix restore (~38 KV
entries at 16K).

## Audit of other AR-only / generation_mode serve gates
- `--generation-mode mtp requires --load-mtp` (public.py + openai.py): fires only
  with `--no-load-mtp` (`load_mtp` defaults True), so it does not block the smoke.
- No other AR-only gate on the promoted-profile serve path.

## Issue #1 (AR loaded the MTP head / weights 17.7G) — analysis
`expert_artifact_status(<mxfp4>)` returns `ok=True, streamed_experts=True` with
the bank size matching (288,777,830,400 B), so `_maybe_enable_expert_streaming`
auto-enables streaming and the CLI forwards `--expert-streaming` to the daemon —
streaming should engage. With streaming engaged and AR (`mtp=False`, fix #2), the
loader takes `partition_text_residents(with_mtp=False)` → text-only 9.73 GB, no
MTP head. The log's "weights 17.7G" is the outer memory plan's **disk scan** of
the resident shards (which physically contain the text + `mtp.*` + vision
residents ≈ 18.6 GB on disk), not the loaded footprint. Needs the next GPU smoke
to confirm the **loaded** weights are text-only and the MTP head is absent in AR.

## What the next GPU smoke should confirm
- `--generation-mode mtp` reaches `/health` (no AR-only exit).
- AR (no flag): loaded weights text-only (~9.7 GB), no "Installing native-MTP
  draft head", memory plan reports the profile (82 GiB cap, 16,384 context, ≤2 GiB
  session bank), allocator fraction ≤ 1.0.
- MTP (`--generation-mode mtp`): weights ~17.7 GB (text + MTP), head active.

## Tests (CPU-only)
`tests/test_deepseek_v41_mtp_gate_w35.py` (7): the public.py gate allows native
MTP / rejects non-native / no-op for AR; the `_is_native_streamed_mtp` predicate
(deepseek_v41 + stages + mtp residents → True; hy3 / no-stages → False); the
profile auto-resolves for the model key; context = 16,384 and the session bank +
engram caps are in `child_env`; the profile is AR by construction. Regression:
`serve_profile`, `mtp_serve_glue`, `expert_cli_runtime`, `serve_streaming_autodetect`,
`expert_profiles`, and the `public_cli` generation-mode subset all green.

---

## Window-12 follow-up (server reached /health; three residual issues)

Re-smoke on `e2aa4a0fa` (W35 v1 included): `--generation-mode mtp` reached
`/health` (gate fix worked, context capped to 16,384, engine budget 75.0G), but
`/health` reported `generation_mode='ar'` and the plan advertised a 48G session
bank. Root causes and fixes:

### (1) generation_mode='ar' — a THIRD forcing site (fixed)
After the AR-only gate passed, the interactive serve path
(`commands/public.py`) **unconditionally** ran `args.no_mtp=True;
load_mtp=False; generation_mode=AR` and forwarded `--generation-mode ar` to the
daemon. Factored the decision into `_streamed_native_mtp_requested(args,
model_path)` (shared with the gate) and gated the forcing: the DSpark native-MTP
artifact + `--generation-mode mtp` now keeps MTP (`generation_mode=mtp`,
`load_mtp=True`) and forwards `--generation-mode mtp`; hy3/glm + AR stay AR.
Confirmed on the **real artifact metadata** (config.json + expert-manifest.json,
no experts.bin): `_streamed_native_mtp_requested(mtp_args, <mxfp4>) is True`,
`ar_args → False`. (The registry's `_apply_runtime_compatibility_mode` does **not**
also flip it: deepseek-v41 is `recognized-backend-pending`, not
`native-ar-only`, and its note says the streamed path is decided by the admitted
manifest, not the catalog gate.)

### (2) "session bank up to 48.0G" — which process reads the cap
Two budgets exist: `engine_session.resolve_session_bank_max_bytes` (the **actual**
bank, in the daemon) honors an explicit `MTPLX_SESSION_BANK_MAX_BYTES` over the
auto/plan sizing → 2 GiB when the env reaches; `memory_plan.plan_memory` computes
its **advertised** `bank_idle_max_bytes` independently as `usable − weights`
clamped to `BANK_CAP_BYTES` (48 GiB) and did **not** read the env → the 48.0G
line. The child_env DOES reach the daemon (`apply_expert_profile_child_env(args,
child_env_base)` runs on the interactive serve path before the daemon spawn, with
`_resolved_expert_profile` set by `expert_streaming_load_kwargs`), so the actual
bank is already 2 GiB. Fix: `plan_memory` gained an optional
`session_bank_max_bytes`, and the daemon passes the resolved explicit cap, so the
plan line now matches the engine bank (2.0G). Other models pass `None` →
byte-identical.

### (3) engine budget 75.0G vs the profile's 82 GiB — consistent
`75 GiB = reconcile_mlx_memory_cap(plan) = memory_limit(82 GiB) −
runtime_reserve(7 GiB)` — the Metal engine budget IS the profile ceiling minus the
reserve; they are the same plan, not competing. The 75 GiB governs weights +
expert cache; the 82 GiB is the process ceiling. Locked by a test.

## Tests added (CPU-only)
`test_real_artifact_generation_mode_resolves_to_mtp` (skips if the artifact is
absent; reads config + manifest only), `test_memory_plan_bank_yields_to_explicit_cap`,
`test_explicit_session_bank_env_wins_over_the_plan`,
`test_engine_budget_is_ceiling_minus_reserve`. Full file: 11 passed. Regression:
`serve_profile`, `mtp_serve_glue`, `memory_plan`, `expert_cli_runtime`, and the
`public_cli` generation-mode subset all green (127 + 3).

## Next GPU smoke should confirm
`/health` reports `generation_mode='mtp'` for `--generation-mode mtp`; the memory
plan line shows the ≤2 GiB session bank; AR (no flag) loads text-only with no MTP
head.

---

## Window-14 follow-up (MTP lane engaged; strict resident load failed)

Re-smoke on `7770960a9`: `--generation-mode mtp` now takes the MTP lane (the gate
+ forcing fixes worked), but load died with `ResidentLoadError: resident
parameter validation failed: Received 3 parameters not in model:
mtp.layers.{0,1,2}.mlp.gate.e_score_correction_bias_vl`.

### Root cause
The artifact ships a vision-language gate-bias variant (`*_vl`) for every routed
layer — 40 `layers.*.ffn.gate.bias_vl` (backbone) and 3 `mtp.*.ffn.gate.bias_vl`
(MTP stages). The backbone model's `sanitize` folds its 40 (the constructed model
declares **no** `*_vl` param at all); the text-only DSpark head (worker W23)
omits its own VL gate bias. `partition_text_residents(with_mtp=True)` kept **all**
`mtp.*` residents including the 3 `*_vl`, which `_map_mtp_residents` maps to
`mtp.layers.{0,1,2}.mlp.gate.e_score_correction_bias_vl` — params the head does
not have — so `load_weights(strict=True)` rejected them. W23's DSpark tests never
hit it (synthetic residents, the 376 GiB artifact was never loaded).

### Fix
`partition_text_residents(with_mtp=True)` now drops `mtp.*_vl` residents (via
`_is_kept_mtp_resident`); the backbone `layers.*_vl` are untouched (sanitize
handles them). So the served MTP resident load provides exactly the head's
declared params.

### Verification (CPU, real config + manifest, lazy zeros → no experts.bin, 468 MB RSS)
- The constructed model (`mtp=True`) has 1438 params; the sanitized MTP partition
  keys are a subset with **0 keys not in the model** (was 3). The only model
  params the residents don't provide are the streamed `switch_mlp` backbone
  experts (bound via `bind_streamed_switches`, not residents).
- The model declares no `*_vl` gate bias; the partition drops the 3 `mtp.*_vl`
  and keeps the 40 backbone `layers.*_vl`.

Tests: `test_served_mtp_partition_matches_constructed_model_params`,
`test_mtp_vl_gate_bias_is_dropped_and_absent_from_the_head` (both skip if the
artifact is absent). W23's synthetic-manifest partition/mapping tests still green
(no `_vl` in their fixtures). Full `test_deepseek_v41_mtp_gate_w35.py`: 13 passed.

### Next GPU smoke should confirm
`--generation-mode mtp` reaches `/health` with `generation_mode='mtp'` and the
DSpark head active (no strict-load failure); AR still text-only.
