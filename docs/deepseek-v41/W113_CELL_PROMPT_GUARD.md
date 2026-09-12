# W113 — cell-prompt guard, prompt provenance, EOS surfacing

`scripts/deepseek_v41/ab_decode_env_levers.py`

## The finding: windows 39–42 measured the wrong prompt

The 16K decode cell is supposed to be the standard **chat-templated** cell — the
exact server token ids exported by `scripts/fable/server_cell_bench.py`
(schema `mtplx-server-cell-prompt-ids-v1`, `cell=sweep`, `target_tokens=16384`,
`seed=20260829`), which end with a generation prompt `<｜Assistant｜></think>` and
ask the model to answer.

Windows 39–42 instead measured the **raw `prefill_bench` builder** prompt, because
`--prompt-format` defaults to `raw` and the runner fell back to the builder when
`--prompt-ids-file` was not passed. That prompt is:

```
BOS(id 0) + [96× "# file_N.py" code-ladder filler] + DEFAULT_FINAL_REQUEST
```

= **16,385 tokens** (16,384 content + the prepended BOS), with **no chat template
and no generation prompt**. Its greedy first token is **EOS (id 1)**. The decode
loops had `stop_ids=set()` and no EOS check, so **256 forced post‑EOS filler
tokens** were timed (the model just resumed the `# file_N.py` ladder). An
EOS‑honouring server would return an **EMPTY** answer — the codebase already knew
this via the `serve_bench_1k` W18 guard.

### Which receipt fields tell the two cells apart

| field | raw builder (windows 39–42) | standard cell (window 43+) |
|---|---|---|
| `prompt_source` (W113) | `raw-builder` | `prompt-ids-file` |
| `prompt_tokens` | **16385** (16384 + BOS re-prepend) | **16384** (file ids, BOS already inside, no re-prepend) |
| `prompt_chat_templated` (W113) | `false` (no `<｜Assistant｜>`) | `true` (ends `<｜Assistant｜></think>`) |
| `prompt_ids_file` (W113) | `null` | the standard file path |
| first generated token | **1 (EOS)** → `first_token_eos=true` | 666 → `first_token_eos=false` |

Note: `prompt_ids_sha256` (W113) is the sha of the **prompt** ids; the pre‑existing
`token_ids_sha256` is the sha of the **generated** ids — two different quantities.

Window 43 (run on the correct prompt) shows `prompt_tokens=16384`, first token 666
(not EOS), and `eos_index` **None** (arm `cell16k_ring`) / **238** (arm
`cell16k_ring_v2`) — both `answer_valid=true` (an answer was actually produced).

## The guard (real-measurement path only)

`_apply_cell_prompt_guard(args)` runs in `main()` before any arm/model load (it is
skipped for `--dry-run`, which is a CPU argument/env resolution double that
exercises the raw builder with the fake tokenizer on purpose, and stamps
`prompt_source="raw-builder"` so the receipt is never silent about it):

1. **Auto-default** `--prompt-ids-file` to the standard cell file when
   `--context-tokens 16384` and that file exists (path resolved **repo-root
   relative**: `docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/prompt-ids-deepseek-v41.json`).
   Launchers get the standard cell without passing the path. The auto-default also
   stamps `--prompt-seed 20260829` (so the receipt's `prompt_seed` is not left
   null) and **pins the prompt-ids sha** to
   `1a45b35bae742fae0e26d4f40ee0dc1093a2038e5b514460a4f02e9e56d74565`, refusing if
   the file was edited/swapped.
2. **Refuse** (raise `SystemExit` with a clear error naming the standard file) any
   `cell16k`/`cell16k_*` arm, or any `--context-tokens 16384` run, that has no
   `--prompt-ids-file` (and no auto-default available, e.g. the file is missing).
   (The bare `cell16k` preset — no trailing underscore — is treated as a 16K-cell
   arm too.)
3. `--allow-raw-prompt` is the loud diagnostics **escape hatch**: it skips both the
   refusal and the auto-default, runs the raw builder, and stamps
   `prompt_source="raw-builder"` with a warning.

**Precedence**: explicit `--prompt-ids-file` > `--allow-raw-prompt` (raw, loud) >
ctx-16384 auto-default > refusal. A run that is neither a `cell16k_*` arm nor
`--context-tokens 16384` is untouched.

## EOS surfacing

Both AR decode loops (classic and device_sample) still generate the requested N
tokens by default (throughput numbers unchanged). The receipt now records, at the
AR top level **and** in the `dspark` block:

- `first_token_eos` — the first generated token is EOS (served path → EMPTY
  answer). A loud line is printed:
  `[ab] !!! WARNING: first generated token is EOS -- answer would be EMPTY on a served path !!!`
- `eos_index` — first position of the EOS id in the stream, or `null`.
- `tokens_before_eos` — `eos_index`, or the whole stream length if EOS is absent.
- `answer_valid` — `not first_token_eos`: the answer is non-empty iff the first
  token is not EOS. **Cap-independent** — it does not change whether `--stop-on-eos`
  truncated the stream or the full fixed-step decode ran. (The earlier
  `eos_index > 0.5·N` rule was withdrawn: it flipped a correct short answer to
  invalid once `--stop-on-eos` shrank N.)
- `answer_truncated` — `eos_index is null`: the decode hit the token cap without
  the model emitting EOS (the answer may be cut off).
- `post_eos_tokens_timed` — when EOS is present, the number of forced post-EOS
  tokens that were still timed (`n_generated - eos_index - 1`; the wasted filler a
  served path never produces — **256** on the windows 39–42 raw prompt, whose EOS
  was at index 0); `0` when EOS is absent.
- `eos_id`, `n_generated` — for auditability.

This mirrors the `serve_bench_1k` W18 guard semantics (an EOS‑honouring server
returns a blank answer when the first token is EOS).

**DSpark decode rate denominator:** the DSpark headline `decode_tok_s` is computed
over `len(toks) - 1` (decode-only, excluding the prefill/first token), matching the
AR lane's `decode_steps_run`. Before W113 it divided by `steps + 1`, so the DSpark
rate was on a different denominator than AR; it is also now correct under
`--stop-on-eos` (where `toks` is the truncated stream).

## Flags

| flag | default | effect |
|---|---|---|
| `--allow-raw-prompt` | off | escape hatch: measure the raw builder on a 16K cell (skips refusal + auto-default), stamps `prompt_source=raw-builder` loudly |
| `--stop-on-eos` | off | served-parity: stop the AR decode (and, in `--decode-mode dspark`, the speculative decode) at EOS and report `decode_tok_s` over the tokens **actually** generated (`decode_tokens_generated`). Default off keeps the full fixed-step decode. |
| `--eos-id N` | resolve from tokenizer files | override the EOS id used by `--stop-on-eos` and the EOS-surfacing fields |

The EOS id is resolved from the tokenizer files (`tokenizer_config.json` +
`tokenizer.json`'s `added_tokens`) — **no model load** — so it works even on the
`--prompt-ids-file` path that skips the tokenizer. For this model, EOS = id 1,
`<｜Assistant｜>` = 128804.

`--stop-on-eos` with an **unresolvable** EOS id (empty/missing tokenizer files and
no `--eos-id`) is refused in `_run_arm`
(`SystemExit("--stop-on-eos needs an EOS id … pass --eos-id")`) rather than
silently no-op'ing while stamping `stop_on_eos: true`.

**Which passes honour `--stop-on-eos`:** the headline AR and DSpark passes and the
`--warm-repeat` pass (its denominator is the tokens actually generated, and it
stops at the same point as the cold pass so `token_ids_match` holds). The
`--stage-timing` and `--syncs` passes deliberately **ignore** it — they run the
full requested step count for a fenced per-stage / host-sync census whose absolute
tok/s is discarded, so an early stop would only shrink the census sample.

## The exact window launcher line for the cell

```sh
$PY $AB \
  --context-tokens 16384 \
  --decode-tokens 256 \
  --max-kv 17408 \
  --prompt-ids-file <repo>/docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/prompt-ids-deepseek-v41.json \
  --prompt-seed 20260829 \
  --stage-timing \
  --memory-limit-gib 60 \
  --utilization \
  --arms <arm> \
  --out <receipt>
```

(`$AB` = `scripts/deepseek_v41/ab_decode_env_levers.py`, `<repo>` = repo/worktree
root.) With the W113 auto-default, the `--prompt-ids-file` argument is now optional
at `--context-tokens 16384` — the runner defaults it to exactly this file — but
passing it explicitly is still the clearest form for a window receipt.

Every flag above exists on this branch and parses.

### Consistency of `--prompt-ids-file` + `--context-tokens 16384` + `--max-kv 17408`

- **Does the override re-derive the context from the file?** No.
  `bench._prompt_ids_override` uses `--context-tokens` (16384) only to **select**
  the entry whose `target_tokens == context_tokens`; the prompt length is whatever
  the selected entry's `token_ids` is (16,384 here). The receipt's `context_tokens`
  = 16384 and `prompt_tokens` = `len(prompt_ids)` = 16384 are therefore consistent,
  and there is **no BOS re-prepend** (the exported ids already carry BOS at index 0).
- **Does it need `--prompt-seed` to pick prompt[1] vs prompt[0]?** Not for **this**
  file: `_prompt_ids_override` filters on `cell == "sweep"` **and**
  `target_tokens == 16384`, and prompt[0] is `cell=vanity, target_tokens=0` (81
  tokens), so the filter alone selects prompt[1]. `--prompt-seed 20260829` is
  therefore consistent/redundant here — **but it is required in general**: if two
  seeds ever share `(sweep, 16384)`, the override raises an "ambiguous … pass
  `--prompt-seed`" `SystemExit`. Passing it is the safe, future-proof form.
- **`--max-kv 17408`**: `bench.resolve_max_kv([16384], 256, 17408)` needs
  `16384 + 256 + 64 = 16704`; `17408 ≥ 16704`, so it is accepted. (A `cell16k_*`
  arm additionally arms `MTPLX_DSV41_KV_BOUNDED`, which preallocates every KV lane
  to this resolved `max_kv`.) The default `--max-kv 4096` is **below** 16704, so a
  16K run must pass `--max-kv 17408`.

## Where the override could still silently fall back to the raw builder

The guard closes the launcher path, but these routes still use the raw builder if
misused (documented so they are not surprises):

1. **`--dry-run`** — the guard is intentionally skipped (no measurement). The
   dry-run receipt stamps `prompt_source="raw-builder"`, so it is explicit, not
   silent.
2. **Calling `_run_arm` / `_dry_run_arm` directly** (a different entrypoint or a
   test) bypasses `main()`'s guard; the fallback in `bench._resolve_prompt` is
   silent (uses the builder when `prompt_ids_file` is falsy).
3. **`scripts/deepseek_v41/bench_standard_shape.py` `main()`** (the sibling cell
   harness) has **no** guard: run at `--context-tokens 16384` without
   `--prompt-ids-file` it silently uses the raw builder. Out of scope for W113
   (which fixes the `ab_decode_env_levers.py` runner) but worth a follow-up if that
   script is used for window measurements.
