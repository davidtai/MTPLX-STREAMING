# W34 — GPU window 11 tooling fixes

Branch `feat/deepseek-v41-w34` (off `feat/deepseek-v41-streaming` @ `18c7cff27`).
Fix commit: **`d9ef7cb60`**. Report commit: this file.

Two harness failures observed in
`.benchmark-artifacts/deepseek-v41/gpu-window-11-native.log` (step 1
`serve_health AR`, and the env-lever A/B driver). Both are pure CPU tooling
bugs — no runtime/model change. Fixed, reproduced on CPU, and covered by CPU
tests. All work under `nice -n 19`, MLX pinned to CPU, no artifact loads.

## Fix 1 — `scripts/deepseek_v41/ab_decode_env_levers.py` argparse crash

**Symptom.** `_run_arm` raised `AttributeError: 'Namespace' object has no
attribute 'prompt'` at `bench._prompt_args(args, args.context_tokens)`.

**Root cause.** The env-lever driver reuses `bench_standard_shape.py`'s
`_prompt_args`/`build_prompt` prompt path, but its own argparse never defined the
options that helper reads. `_prompt_args` builds a namespace from
`args.prompt`, `args.prompt_format`, `args.bos`, `args.bos_id`, `args.model`;
only `--model` existed, so the first attribute access (`args.prompt`) blew up
before any model load — the arm never started.

**Fix.** Added `--prompt` (default `None`), `--prompt-format` (default `raw`),
`--bos/--no-bos` (`BooleanOptionalAction`, default `True`), `--bos-id`
(default `0`) — byte-for-byte the same names and defaults as
`bench_standard_shape.py`. Enumerated every `args.<x>` on the reused code path
(prompt build + `_load_model` loader path); the loader options (`--model`,
`--memory-limit-gib`, `--max-kv`, `--admit`, `--admission-receipt`,
`--expert-cache-limit-gib`, `--apply-memory-cap`, `--slot-layout`,
`--verify-record-hashes`, `--seed`) were already present, so the four prompt
options were the only gap.

Added `--dry-run` mirroring bench's CPU double: it applies each arm's env and
builds the standard-shape prompt with `bench._FakeTokenizer()` — no model, no
MLX/Metal import — writing the same append-only JSONL receipt with `prompt_build`
+ `arm_env` per arm.

**Coordinator add-on (mid-task).** Two more env arms — `sinkhorn_metal`
(`MTPLX_DSV41_SINKHORN_METAL=1`, K3, merged @ `8982b93c9`) and `hc_compile`
(`MTPLX_DSV41_HC_COMPILE=1`, K4) — plus `all_levers` (all four on). Every preset
now pins all four lever keys (`None` = force-unset via a `_preset(...)` helper),
so applying an arm fully determines the env regardless of what a prior arm in the
same process set — the arms are independent. `both` still means
`shared_overlap` + `layer_major`.

**Audit of `ab_decode_levers.py` (W24).** Confirmed clean: it builds the prompt
inline (`_prompt_build_for_context` + its own BOS prepend), never touching
`args.prompt`/`bench._prompt_args`, and its parser defines every `args.<x>` on
its path. No change needed — consistent with it having reached model load in the
window.

**Test.** `tests/test_deepseek_v41_ab_env_levers.py` (13 cases): the parser now
carries the four prompt options with bench's defaults; `bench._prompt_args(args,
1024)` no longer raises; `_apply_arm_env` sets exactly the expected key per arm
and force-unsets the rest (independence, incl. `all_levers → control`);
`main(--dry-run)` over `control, shared_overlap, layer_major, sinkhorn_metal,
hc_compile, both, all_levers` records the right per-arm `arm_env`; and each arm's
`prompt_build` metadata is byte-for-byte identical to `bench_standard_shape`'s
`--dry-run` for 1024 (`input_tokens == 1025`).

## Fix 2 — `scripts/deepseek_v41/serve_health.sh` JSON parsing

**Symptom.** `/health was not valid JSON` and `chat response was not valid JSON`
even though the server generated successfully (log event
`mtplx_openai_generation`, 34 completion tokens); the served model id also fell
back to `basename` instead of the real `/v1/models` id.

**Root cause.** Each body was parsed inline with

```
printf '%s' "$BODY" | "$PY" - <<'PYEOF'
    ... json.load(sys.stdin) ...
PYEOF
```

That command gives the child **two** writers for fd 0: the pipe (the body) and
the here-doc (the program). Bash wires the pipe first, then applies the here-doc
redirect, so fd 0 ends up on the here-doc temp file: `python3 -` reads its
program from there and, by the time the program runs, `sys.stdin` is that same
file at EOF. `json.load(sys.stdin)` therefore always got an empty string and
raised — for `/health`, the chat response, and `/v1/models` alike (the models
parse swallowed the error and fell back to `basename`). The piped body never
reached Python. Reproduced under `bash`: the exact old block prints
`/health was not valid JSON` on a valid `/health` body.

**Fix.** Moved the three parsers into a new stdlib-only
`scripts/deepseek_v41/serve_health_parse.py`, invoked as a **real file**
(`"$PY" "$PARSE" models|health|chat`) so fd 0 stays free for the piped body. The
script prints the served model id (from `/v1/models`), `generation_mode` +
`profile` (name + `runtime_mode`) + `model_key` (from `/health`), and the
completion text head + tok/s (server-reported `decode_tok_s`/`tok_s` when the
response carries one, plus the wall-clock rate). POSIX-bash preserved; a
missing-parser pre-flight check added.

**Test.** `tests/test_deepseek_v41_serve_health.py` (9 cases): the parser
functions extract model id / generation_mode / profile / completion head / tok/s
from canned bodies and emit the exact "was not valid JSON" lines on non-JSON;
the fixed shell pipe (`printf … | python serve_health_parse.py <mode>`) reaches
stdin end to end; and a regression lock proves the old `python3 - <<'HEREDOC'`
form cannot see the piped body.

## Verification

```
nice -n 19 .venv/bin/python3 -m pytest \
  tests/test_deepseek_v41_ab_env_levers.py \
  tests/test_deepseek_v41_serve_health.py
# 22 passed (13 + 9)
```

`bash -n scripts/deepseek_v41/serve_health.sh` clean; both scripts `py_compile`
clean; `scripts/check_ai_attribution.py --range feat/deepseek-v41-streaming..HEAD`
clean.
