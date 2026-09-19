# W46 — DeepSeek-V4.1 decode levers on the SERVED path

Branch `feat/deepseek-v41-w46` off integration `feat/deepseek-v41-streaming`
(`f8b168b85`). CPU-only: no GPU/Metal, no artifact model load (manifest metadata
reads only via the profile resolver; the served bench builds the prompt with the
artifact tokenizer, a metadata read), tests pin `mx.set_default_device(mx.cpu)`,
peak worker RSS well under the 3 GB cap. Not pushed.

## Why this window exists

Windows 14–16 measured, with the bench harness
(`scripts/deepseek_v41/ab_decode_env_levers.py`, 1,024-token prefill_bench prompt
+ 256 greedy decode), these byte-identical decode levers on the mxfp4 streaming
artifact:

- `MTPLX_DSV41_HEAD_MODE=bf16` — +29–31% (removes the lm-head fp32-cast trap).
- head bf16 + `MTPLX_DSV41_SINKHORN_METAL=1` + `MTPLX_DSV41_ATTN_COMPILE=1` →
  **6.24 tok/s vs 4.05 control (+54%)**.
- `MTPLX_DSV41_ATTN_WIN_MEMO=1` (W45, byte-identical, unmeasured on GPU) and
  `MTPLX_DSV41_SWITCH_FASTPATH=1 + MTPLX_DSV41_SWITCH_SUBMIT=1` (+2.9%) are the
  remaining stack candidates.

But the **served** path (`mtplx serve … via serve_health.sh` with those env vars
exported in the parent shell) reported **3.99 tok/s** — the control rate — on a
34-token completion. Three hypotheses: (a) the env never reached the daemon
child; (b) the served decode path bypasses the model-level flags; (c) the ~18-
token prompt / tiny completion hides the gain.

## Verdict: the parent env DOES reach the daemon child (hypothesis (a) is false)

### How env reaches the daemon child

`mtplx serve` runs the server as a spawned child, and the child's **entire
environment is a dict built in the parent**, then handed to the child:

1. `serve_health.sh` / `serve_bench_1k.sh` launch the server with
   `exec env PYTHONPATH=… "$VENV_PY" -m mtplx.cli serve …`. `env` here only
   *adds* `PYTHONPATH` (no `-i`), so **every exported `MTPLX_DSV41_*` is
   inherited** by the `mtplx serve` process's `os.environ`.
2. In the serve handler (`mtplx/commands/public.py`), the child env is
   `child_env_base = os.environ.copy()` (line ~10070) — this is where the
   parent's exported levers enter — followed by serve-flag stamps
   (paged-KV, ram-session-cache, fan, app-launch-id) and then
   `apply_expert_profile_child_env(args, child_env_base)` (line ~10091).
3. The child is spawned with that dict as its whole environment:
   `os.execvpe(sys.executable, cmd, child_env_base)` (line ~10180), or
   `_run_server_child_with_app_parent_watchdog(cmd, env=child_env_base, …)` when
   an app-parent watchdog is in play.
4. Inside the server child (`mtplx/server/openai.py`, `ServerState.__init__`),
   `apply_expert_profile_child_env(args, os.environ)` (line ~3333) re-applies the
   profile child_env onto the child's own `os.environ` — belt-and-suspenders for
   read-at-use levers.

The lever read sites confirm the env is honoured on the served forward (the
served AR decode uses the same model object as the bench harness):

- `HEAD_MODE` — `_resolve_head_mode()` reads `os.environ` at **model
  construction** (load-time repack), `mtplx/models/deepseek_v41.py`.
- `SINKHORN_METAL` — `_sinkhorn_metal_enabled()` reads `os.environ` **per call**.
- `SWITCH_FASTPATH` / `SWITCH_SUBMIT` / `SHARED_OVERLAP` — `os.environ.get(...)`
  **per forward** (`mtplx/models/expert_mlx.py`, `deepseek_v41_moe.py`).
- `ATTN_COMPILE` / `ATTN_WIN_MEMO` — frozen into module globals **at import**
  (`deepseek_v41.py:886,919`). This still works on the served path: because
  step 2 composes the profile default onto `child_env_base` **before** the child
  is spawned, the key is already in the child's environment at process start, so
  the import-time freeze reads the correct value. An operator's parent-shell
  export is likewise present at start. (The only way these two would miss is a
  spawn that bypasses public.py's apply — the serve scripts do not.)

### Evidence

- Code trace above (public.py env composition + `os.execvpe`; serve scripts'
  `exec env …` inheritance).
- `tests/test_deepseek_v41_lever_child_env_w46.py` reproduces the composition and
  **actually spawns a child process** with the composed env, asserting the child
  sees the levers:
  - `test_default_levers_survive_into_a_spawned_child` — profile defaults land
    when the operator sets nothing.
  - `test_explicit_parent_export_overrides_the_profile_default` — an exported
    `MTPLX_DSV41_HEAD_MODE=mxfp8` survives into the child; profile fills the rest.
  - `test_memory_cap_stays_forced_over_a_serve_flag_value` — the W35 bank cap is
    not overridable by a serve flag.

**So the 3.99 served number is not an env-delivery failure.** It is the tiny-
shape artifact (hypothesis (c)): the lever wins are steady-state decode-rate
gains measured over 256 decode steps at a 1,024-token prefill; an ~18-token
prompt with a ~16–34-token completion is dominated by fixed per-request/first-
token overhead. W46 adds a served 1K bench (below) to re-measure on the bench
shape, and a served startup log line so a window can *see* which levers engaged.

## Served-path lever visibility (startup log)

`mtplx/server/openai.py` now prints, right after the profile child_env is
composed onto the daemon's `os.environ` (gated on the DeepSeek-V4.1 streamed
model so it never fires for other families):

```
[4/6] DeepSeek-V4.1 decode levers (resolved env): HEAD_MODE=bf16 SINKHORN_METAL=1 ATTN_COMPILE=1 ATTN_WIN_MEMO=1 SWITCH_FASTPATH=<unset> SWITCH_SUBMIT=<unset> DEVICE_ROUTE=<unset> HC_COMPILE=<unset> SHARED_OVERLAP=<unset> PREFILL_LAYER_MAJOR=<unset>
```

All ten keys the campaign tracks are reported (`<unset>` when absent).
`DEVICE_ROUTE` is surfaced for operator intent even though no code reads it yet.
Helpers `_dsv41_resolved_lever_env` / `_format_dsv41_lever_env` are unit-tested.

## Promoted served defaults + precedence

Added to the `deepseek-v41-mxfp4-75` profile `child_env`
(`mtplx/data/expert_profiles.json`) — the measured-positive, byte-identical
levers, as served defaults:

| key | value | evidence |
|---|---|---|
| `MTPLX_DSV41_HEAD_MODE` | `bf16` | W40/K21 (+29–31%) |
| `MTPLX_DSV41_SINKHORN_METAL` | `1` | W32/K3 |
| `MTPLX_DSV41_ATTN_COMPILE` | `1` | W41/K22 |
| `MTPLX_DSV41_ATTN_WIN_MEMO` | `1` | W45/K24 (byte-identical, GPU unmeasured) |

Left **off** pending an A/B window: `MTPLX_DSV41_SWITCH_FASTPATH`,
`MTPLX_DSV41_SWITCH_SUBMIT` (+2.9%, unconfirmed served), `MTPLX_DSV41_DEVICE_ROUTE`
(no reader), `MTPLX_DSV41_PREFILL_LAYER_MAJOR` (prefill lever).

### Precedence (two-tier, documented in `apply_expert_profile_child_env`)

`apply_expert_profile_child_env` was changed from a blanket
`environ.update(profile.child_env)` (profile always wins) to a per-key rule:

- **`MTPLX_DSV41_*` lever keys** — applied with `setdefault` semantics:
  **explicit parent-shell export > profile default > code default (off/unset)**.
  An operator A/B-ing one lever for a single window just exports it; no profile
  edit needed.
- **Every other child_env key** — **profile (forced) > inherited/serve-flag
  env**, unchanged. This is load-bearing: `--ram-session-cache` stamps
  `MTPLX_SESSION_BANK_MAX_BYTES` into the child env *before* the profile applies
  (`_apply_ram_session_cache_env`), and the W35 `2GiB` cap must still win or the
  admitted memory plan is defeated (`never-exceed-the-memory-knob`). The memory-
  safety caps (`MTPLX_SESSION_BANK_MAX_BYTES`, `MTPLX_ENGRAM_CACHE_LIMIT`,
  `MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS`) are outside the `MTPLX_DSV41_` namespace
  and therefore stay forced.

Tests: `tests/test_deepseek_v41_lever_child_env_w46.py` (profile carries the
levers; parent overrides them; memory cap forced; double-apply idempotent). The
existing `tests/test_deepseek_v41_serve_profile.py` and
`tests/test_deepseek_v41_mtp_gate_w35.py` stay green (the memory keys are
unaffected: no pre-apply writer sets them for the DeepSeek default serve, and the
forced tier is unchanged).

### Byte-identity caveat (safety)

"Byte-identical" for these levers is the GPU-window finding that the **greedy
output token sequence** is unchanged (`HEAD_MODE=bf16` does a bf16 GEMV vs the
default's fp32-promoted GEMV; the argmax token is robust to the low-bit logit
delta). Promoting `HEAD_MODE=bf16` as a served **default** therefore changes the
default served numerics vs the shipped control, so the served 1K receipt records
the **completion sha** — a window compares candidate-default vs control (levers
off) shas to prove the served output is unchanged before this profile ships.

## Served 1K benchmark

`scripts/deepseek_v41/serve_bench_1k.py` (+ `serve_bench_1k.sh`) sends the **same**
1,024-token prefill_bench prompt the bench harness builds, at 256 greedy decode,
through the running server, and writes a JSON receipt.

- **Exact input**: the prompt is built with the bench's own builder
  (`dump_hidden_states.build_prompt` → `prefill_bench._prompt_build_for_context`,
  raw format) against the **real artifact tokenizer**, giving 1,024 context
  tokens + the reference BOS id 0 = **1,025 input tokens**. It is POSTed as a
  **`list[int]`** to the **raw `/v1/completions`** endpoint. `_encode_prompt`
  (`openai.py`) uses a token-id list **verbatim** (no chat template, no
  `add_special_tokens` BOS), whereas a *string* prompt would be re-tokenized with
  `add_special_tokens=True` and gain a BOS. `/v1/chat/completions` is **not**
  used (it wraps the prompt in the chat template).
- **Server-side timing**: from the non-stream `/v1/completions` `timings` block
  (`_build_timings`): `prompt_per_second` → prefill tok/s, `predicted_per_second`
  → decode tok/s, `prompt_ms` → prefill time (== TTFT for a non-stream request,
  first token emitted only after prefill). `usage` gives the token counts. A
  client wall-clock end-to-end tok/s is recorded as a cross-check.
- **Receipt**: written to `$DSV41_BENCH_RECEIPT` (or `--out`); append-only
  (`never-overwrite-a-measurement`). Carries the completion sha256, head/tail,
  server-side rates, TTFT, and the resolved request.
- **CPU-testable**: `--canned-response FILE` (or `MTPLX_DSV41_CANNED_RESPONSE`)
  treats the file's JSON as the server body and builds the receipt with no
  tokenizer / no network / no MLX — the exact path
  `tests/test_deepseek_v41_serve_bench_1k.py` drives (6 tests).

## Exact window commands

Run inside the GPU-lock window (`gpu_window.sh` holds the exclusive flock and has
booted the resident agent out). Never `:8080`; the scripts pick a free high port.
The profile now ships the four levers as defaults, so **AR needs no lever
exports** — the startup log line confirms what engaged.

```bash
# --- AR, CANDIDATE (profile defaults: head bf16 + Sinkhorn + attn-compile +
#     win-memo). Receipt path is explicit + append-only. ---
DSV41_BENCH_RECEIPT=docs/deepseek-v41/receipts/w46_served_ar_candidate.json \
  bash scripts/deepseek_v41/gpu_window.sh \
       bash scripts/deepseek_v41/serve_bench_1k.sh

# --- AR, CONTROL (all four levers OFF via parent-shell override; proves the
#     served delta and lets you diff the completion sha vs candidate). ---
MTPLX_DSV41_HEAD_MODE=default \
MTPLX_DSV41_SINKHORN_METAL=0 \
MTPLX_DSV41_ATTN_COMPILE=0 \
MTPLX_DSV41_ATTN_WIN_MEMO=0 \
DSV41_BENCH_RECEIPT=docs/deepseek-v41/receipts/w46_served_ar_control.json \
  bash scripts/deepseek_v41/gpu_window.sh \
       bash scripts/deepseek_v41/serve_bench_1k.sh

# --- native MTP head (--generation-mode mtp; W35 serve glue) ---
DSV41_SERVE_EXTRA_ARGS="--generation-mode mtp" \
DSV41_BENCH_RECEIPT=docs/deepseek-v41/receipts/w46_served_mtp_candidate.json \
  bash scripts/deepseek_v41/gpu_window.sh \
       bash scripts/deepseek_v41/serve_bench_1k.sh

# --- A/B a single candidate lever (e.g. add SWITCH_* on top of the defaults) ---
MTPLX_DSV41_SWITCH_FASTPATH=1 MTPLX_DSV41_SWITCH_SUBMIT=1 \
DSV41_BENCH_RECEIPT=docs/deepseek-v41/receipts/w46_served_ar_switch.json \
  bash scripts/deepseek_v41/gpu_window.sh \
       bash scripts/deepseek_v41/serve_bench_1k.sh
```

Read each receipt's `server_side.decode_tok_s` / `prefill_tok_s` / `ttft_s` and
compare `completion_sha256` between control and candidate (must match for a
byte-identical lever under greedy). The server log's
`[4/6] DeepSeek-V4.1 decode levers (resolved env): …` line records what actually
engaged inside the daemon for each arm.

## Files touched

- `mtplx/expert_cli.py` — two-tier `apply_expert_profile_child_env` precedence.
- `mtplx/data/expert_profiles.json` — four lever defaults in the profile child_env.
- `mtplx/server/openai.py` — resolved-lever startup log line + helpers.
- `scripts/deepseek_v41/serve_bench_1k.py`, `serve_bench_1k.sh` — served 1K bench.
- `tests/test_deepseek_v41_lever_child_env_w46.py`,
  `tests/test_deepseek_v41_serve_bench_1k.py` — CPU tests.
- `docs/deepseek-v41/W46_SERVED_STACK.md` — this report.

## W18 follow-up: served completion returned only 1 token (EOS) — root cause + fix

Window 18 ran `serve_bench_1k.sh` twice (candidate + control). The daemon lever
line proved engagement (candidate `HEAD_MODE=bf16 SINKHORN_METAL=1 ATTN_COMPILE=1
ATTN_WIN_MEMO=1`; control `default/0/0/0`) and prefill was healthy (53.9 tok/s,
TTFT 19.0 s, **1,025** prompt tokens — the list[int] prompt reached the model
verbatim). But **both** arms returned `completion_tokens=1`, `finish_reason=stop`,
empty completion (`sha256("") = e3b0c442…`) and a nonsense one-token decode rate.
Server log: `"attempts": 4, "blank_retries": 3, "text_preview": ""`.

### Root cause (not the prompt encoding)

1. **The list[int] prompt is handled correctly.** `_encode_prompt` (openai.py)
   returns a token-id list **verbatim** — no re-tokenization, no BOS re-add, no
   chat template — which is exactly why prefill counted 1,025 tokens (1,024 +
   BOS). A *string* prompt would instead hit `_encode_plain_text` →
   `tokenizer.encode(text, add_special_tokens=True)`, gaining a BOS and re-
   tokenizing (not token-exact). So list[int] was the right choice and is kept.
   Covered by `test_real_encode_prompt_*` against the real handler.
2. **The greedy first token of this prompt is EOS.** The raw prefill_bench
   coding prompt (no chat template / assistant-turn opener) reads as a completed
   document to the instruct model, so greedy argmax at the last prompt position
   is `<｜end▁of▁sentence｜>`. It detokenizes to `""`.
3. **The server honours EOS; the bench harness does not.** The server derives
   stop tokens from the tokenizer (`_default_stop_tokens` → `eos_token_id` …) and
   stops on the first stop token — here token #1 — yielding a blank completion.
   Its blank-retry (`blank_retry_attempts`) retried 3× but greedy re-emits EOS
   every time, so all 4 attempts were blank → `completion_tokens=1`,
   `finish_reason=stop`. The bench harness (`ab_decode_env_levers._generate`)
   force-decodes a fixed `for _ in range(steps)` loop with **no EOS check**, so
   it emits the (invisible) EOS as token #1 and keeps going — that is why the
   direct-model run yields 256 tokens starting "# file_0.py" (the visible code
   is token #2 onward).

So the 1-token result is not an env-delivery or prompt-encoding bug; it is the
mismatch between a fixed-step decode-rate probe (ignore EOS) and an EOS-honouring
completion server.

### Fix

- **`mtplx/generation.py` `_default_stop_tokens`**: opt-in `MTPLX_IGNORE_STOP_TOKENS`
  (off by default) returns an empty stop set, so the served generation treats no
  token as a stop and honours `max_tokens` in full. This is the single chokepoint
  every server path falls back to (AR and MTP), and `_is_stop(token, stop_ids)`
  is the ONLY EOS-stop mechanism in the loop, so one gate suffices. Off by
  default → zero effect on the shared `:8080` serve; only the dedicated benchmark
  server sets it.
- **`serve_bench_1k.sh`**: exports `MTPLX_IGNORE_STOP_TOKENS=1` (override with
  `DSV41_IGNORE_STOP_TOKENS=0`) on its own server process, so a greedy 256-token
  fixed-step completion is produced — matching the harness. The prompt stays a
  raw list[int] with BOS id 0 (identical to the harness).
- **`serve_bench_1k.py`**: records `early_stop`/`warning` in the receipt and
  exits non-zero on a REAL early-stop, so a window can never bank a 1-token
  "rate" as a measurement.

With the fix the served arms decode the full 256 tokens; the completion sha256 is
then a real control-vs-candidate identity check (must match for a byte-identical
lever under greedy). The window commands above are unchanged — `serve_bench_1k.sh`
now sets the env itself.

### CPU coverage of the real handlers

`tests/test_deepseek_v41_serve_bench_1k.py` adds: the W18 empty-completion shape
is flagged (`early_stop`); the real `_encode_prompt` uses a list[int] verbatim
with no BOS and re-tokenizes a string with `add_special_tokens=True`; and the
real `_default_stop_tokens` returns the EOS set normally and an empty set under
`MTPLX_IGNORE_STOP_TOKENS=1`.
