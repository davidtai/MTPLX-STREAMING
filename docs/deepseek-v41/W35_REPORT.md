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
