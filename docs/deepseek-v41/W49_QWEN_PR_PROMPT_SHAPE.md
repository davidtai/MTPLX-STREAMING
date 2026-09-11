# W49 — DeepSeek-V4.1-Flash served benchmarks use the Qwen3.8-PR prompt shape

David's directive: the DeepSeek-V4.1-Flash streaming benchmarks must use **the
same prompt construction** as the Qwen3.8 125B MTPLX PRs (#475 / #478 / #482 /
#485 / #488). This brings the harness of record into the repo, makes it produce
the identical shape for DSV4.1, and wires the in-process A/B arms to run on the
exact token ids the served cell used.

All paths here are absolute-from-repo-root of the `mtplx-hy3-ssd` worktree.

---

## 1. What the Qwen-PR construction is (harness of record)

Harness: `scripts/fable/server_cell_bench.py` (copied byte-identically from
`.claude/worktrees/w40-server-bench/scripts/fable/server_cell_bench.py`; the
pure-HTTP client never imports MLX and never touches the GPU). Fixtures copied
byte-identically into `mtplx/benchmarks/prompts/` (hash pin preserved).

The `sweep` prompt for one (context target, seed) is built by
`build_sized_prompt()` (`server_cell_bench.py:1764`):

1. **Filler** = the hash-pinned coding fixture
   `mtplx/benchmarks/prompts/qwen38_generation_context.py`
   (1,752 lines, sha256 `c8ae2b1790c0300aa7c1421b55e7cd5d43c93461f7fba5d3a732fd34e156b4c4`,
   asserted at `EXPECTED_CONTEXT_SHA256`, `:1546`), **rotated** by `seed % 1752`
   lines (`rotate_context()`, `:1552`). For the three production seeds the
   offsets are **701 / 702 / 703** (20260829/30/31 % 1752).
2. **Instruction** = line 1 of
   `mtplx/benchmarks/prompts/qwen38_naturalistic_generation_patch.jsonl`
   (`load_fixture_instruction()`, `:1577`) + `SWEEP_INSTRUCTION_SUFFIX`
   (`:148`, the "SHORT markdown report … three observations … No code." tail).
3. The filler is **sized iteratively** to hit the templated token target within
   `PROMPT_TOKEN_TOLERANCE = 8`, correcting by the MEASURED templated-token
   error each round and keeping the best attempt.
4. The templated count comes from `make_counter()` (`:1603`):
   `tokenizer.apply_chat_template([{"role":"user","content":text}], tokenize=True,
   add_generation_prompt=True, enable_thinking=…, reasoning_effort=…)`.

The request is one streaming `POST /v1/chat/completions` (`stream_chat()`,
`:1210`) with `messages=[{"role":"user","content":prompt}]` and body
(`chat_body()`, `:2648`):

```
max_tokens 1024, temperature 1.0, top_p 0.95, top_k 20, seed,
stream=true, stream_options={"include_usage":true}
```

For Qwen the body ALSO carries `reasoning_effort="xhigh"` and
`enable_thinking=true`. Seeds 20260829 / 20260830 / 20260831; the standard cells
are 1,024 and 16,384 templated tokens (the PR ladder also ran 8K–261K).
`User-Agent` is pinned `server-cell-bench/1` (`BENCH_USER_AGENT`, `:2622`) so the
server's managed-client path never overrides the sampler.

---

## 2. What differs for DeepSeek-V4.1-Flash

The DSV4.1 tokenizer at
`~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4` was inspected directly and
the **real serving code path was exercised in-process** (`mtplx.server.openai.
_encode_messages_uncached` with the DSV4.1 tokenizer, MLX pinned to CPU).

### Chat template / thinking kwargs — OMITTED

- `tokenizer_config.json` has **no `chat_template`**; `tokenizer.json` has no
  embedded template; there is **no `chat_template.jinja` sidecar** in the
  artifact. So `tokenizer.chat_template is None` and
  `apply_chat_template(...)` **raises `ValueError`** ("no chat template is set").
- The server's default chat-template profile is `local` (the tokenizer's own),
  so no template is applied. In `_encode_messages_uncached`, every template
  branch is skipped and the request falls through to the **plain role-prefixed
  render** at `mtplx/server/openai.py:14537`:
  `"\n".join(f"{role}: {content}")` + `"\nassistant:"`, i.e.
  **`"user: " + content + "\nassistant:"`**, encoded with
  `add_special_tokens=False` (`_encode_rendered_chat_text`).
- `enable_thinking` and `reasoning_effort` are **inert**: the served ids are
  **byte-identical with thinking on and off** (verified). **DSV4.1 has no
  thinking mode.** The harness therefore OMITS both kwargs from the counter and
  from the request body for `--model-family deepseek-v41` (`cell_sampling` sets
  them `None`; `chat_body` drops `None`).

### BOS — the served path emits **none**

- `add_bos_token: false` in `tokenizer_config.json`, and `tokenizer.json`'s
  post-processor is **`ByteLevel`**, not a `TemplateProcessing` that inserts BOS.
- Empirically, `tokenizer.encode(text, add_special_tokens=True)` returns the
  **same** ids as `add_special_tokens=False` — **no leading BOS id 0**. The chat
  path uses `add_special_tokens=False`; the completions path
  (`_encode_plain_text`, `openai.py:13207`) uses `add_special_tokens=True`.
  Neither emits BOS, and **no serving/generation code prepends it**.
- Net: **the served DSV4.1 chat prompt begins with `user:` (id 5265), NOT BOS
  id 0.** The model config declares `bos_token_id=0`, but the served path does
  not add it. The exported ids match the server byte-for-byte with no BOS, so
  they carry none.
- **Discrepancy to be aware of:** the in-process reference / A-B path
  (`scripts/deepseek_v41/dump_hidden_states.build_prompt`) DOES prepend BOS id 0
  by default (`--bos`, "the reference always does"). That is a different prompt
  from the served one. To compare like-for-like, feed the A/B arms the exported
  served ids via `--prompt-ids-file` (below): those ids carry no BOS and
  override the builder AND the `--bos` prepend.

The DSV4.1 counter/ids replicate the plain render exactly
(`deepseek_v41_chat_render` / `deepseek_v41_prompt_ids`, `:1665`;
`server_prompt_ids`, `:1691`), verified against `_encode_messages_uncached` for
the vanity + all three 1,024-seed cells (exact match, no BOS). The receipt
records the settings under `template_settings` (`chat_template_source: none`,
`enable_thinking: null`, `reasoning_effort: null`, `bos_id_prepended: false`,
`thinking_mode: false`, `add_special_tokens: false`, render cite).

### Request body: DSV4.1 == Qwen-PR body minus the two unsupported kwargs

Same `max_tokens 1024 / temperature 1.0 / top_p 0.95 / top_k 20 / seed /
stream / stream_options`; **minus** `reasoning_effort` and `enable_thinking`
(pinned by `test_request_body_deepseek_is_pr_body_minus_unsupported_kwargs`).
`User-Agent` stays `server-cell-bench/1`. `MTPLX_IGNORE_STOP_TOKENS` is NOT set:
cells stop naturally, like the PRs.

---

## 3. Exact window commands

Runs INSIDE `gpu_window.sh` (holds the exclusive GPU lock, boots the resident
agent out). The STEP `scripts/deepseek_v41/served_cell_bench.sh` builds the
prompts + exact server token-ids on CPU (before the server starts, so tokenizer
CPU never contaminates a timed cell), starts `mtplx serve` on a free high port
(never :8080), waits for `/health`, runs the harness `--mode cells` (a pure HTTP
client) against it, then stops the server cleanly. Receipts are append-only and
land under `DSV41_RECEIPT_DIR`.

```bash
# AR baseline (profile child_env ships HEAD_MODE=bf16 + SINKHORN_METAL +
# ATTN_COMPILE + ATTN_WIN_MEMO as served defaults)
DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w49-served-cells \
bash scripts/deepseek_v41/gpu_window.sh \
     bash scripts/deepseek_v41/served_cell_bench.sh

# native MTP head
DSV41_SERVE_EXTRA_ARGS="--generation-mode mtp" \
DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w49-served-cells-mtp \
bash scripts/deepseek_v41/gpu_window.sh \
     bash scripts/deepseek_v41/served_cell_bench.sh
```

Env knobs (all overridable): `DSV41_CONTEXTS="1024 16384"`,
`DSV41_SEEDS="20260829 20260830 20260831"` (all three for both 1K and 16K),
`DSV41_MAX_TOKENS=1024`, `DSV41_CELLS=sweep`, sampler
`DSV41_TEMPERATURE=1.0 / DSV41_TOP_P=0.95 / DSV41_TOP_K=20`.

**16K memory plan (W55):** the default `--expert-profile
deepseek-v41-mxfp4-75` planner sets an **82 GiB** process ceiling / **75 GiB**
weight envelope with `max_live_kv_tokens=16,384` (i.e. the KV plan is already
sized for the 16K cell). To cap it explicitly for the 16K prefill, set
`DSV41_MEMORY_LIMIT_GIB=<N>`, passed through as
`mtplx serve --expert-memory-limit <N>GiB` — the accepted flag (NOT
`--memory-budget`, which the `mtplx serve` parser rejects as unrecognized; that
was the Window-21 16K start failure, fixed in W55). The bench harness's 60 GiB
plan peaked at **76.6 GB** with a 16,384-token prefill, so
`DSV41_MEMORY_LIMIT_GIB=60` reproduces that and keeps the box comfortably under
the 100 GiB knob for the 16K cell; the default 82 GiB ceiling is tighter.
Companion knobs `--expert-max-live-kv-tokens` / `--expert-runtime-reserve` are
available via `DSV41_SERVE_EXTRA_ARGS` if finer control is needed.

The harness prints per cell: prefill s + prefill tok/s, TTFT, decode tok/s,
wall, completion tokens, reasoning/answer split (0 for DSV4.1 — no thinking),
finish_reason; then the **fastest-of-seeds** summary per cell (max decode tok/s,
max prefill tok/s, min TTFT, with the min–max range), per
`memory/report-fastest-of-seeds.md`.

### In-process A/B on the SAME ids as the served cells

`served_cell_bench.sh` writes the exact server ids to
`${DSV41_RECEIPT_DIR}/prompt-ids-deepseek-v41.json` (also producible standalone
via `server_cell_bench.py --mode build-prompts|build-prompt-ids
--model-family deepseek-v41 --tokenizer <artifact> --prompt-ids-out <path>`).
Feed them to the decode-lever A/Bs so every arm prefills the identical tokens:

```bash
# one greedy A/B cell on the served 16K seed-20260829 ids (no BOS, no builder)
PYTHONPATH=$PWD .venv/bin/python3 scripts/deepseek_v41/ab_decode_env_levers.py \
    --arms control shared_overlap --context-tokens 16384 --decode-tokens 256 \
    --prompt-ids-file docs/deepseek-v41/receipts/w49-served-cells/prompt-ids-deepseek-v41.json \
    --prompt-seed 20260829 --out <receipt.jsonl>

# same hook on bench_standard_shape.py
PYTHONPATH=$PWD .venv/bin/python3 scripts/deepseek_v41/bench_standard_shape.py \
    --context-tokens 16384 --prompt-ids-file <ids.json> --prompt-seed 20260829 \
    --out-dir <dir>
```

`--prompt-ids-file` overrides the prefill_bench builder AND the `--bos` prepend;
the default (no file) keeps the built prefill_bench prompt so pre-W49 receipts
stay comparable. Both record `prompt_source: prompt-ids-file`, the served-cell
text sha, the token-ids sha, and `bos_prepended: false` in the receipt.

---

## 4. Caveat: token identity is a per-seed check, not byte-identity

The Qwen-PR sampler is **temperature 1.0 / top-p 0.95 / top-k 20** (sampled, not
greedy). The served cells therefore do **not** produce byte-identical
completions across arms or runs, and cross-arm equality can only be judged
**per (seed) with the same seed on both sides** — and even then only holds if
both engines apply the sampler identically (top-p/top-k filter order differs
between MTPLX and mlx-serve; see `ENGINE_FIELD_CAVEATS` in the harness). Use the
receipt's `response_parity` + `request_body_sha256` to confirm both arms were
asked the same thing, and compare completion **token counts / finish_reason /
tok-s**, not the text sha, across sampled arms.

The in-process A/B arms run **greedy** (`ab_decode_env_levers.py` /
`bench_standard_shape.py` decode argmax), so THOSE are byte-identity checks
across arms — but only when they share the same prompt ids, which is exactly
what `--prompt-ids-file` guarantees. The two regimes are distinct: the served
cells measure the PR-shape sampled request; the A/B arms measure greedy decode
on the same prompt tokens.

---

## Files

- `scripts/fable/server_cell_bench.py` — harness of record + `--model-family`
  switch, `--mode build-prompts|build-prompt-ids|cells`, `--prompt-ids-out`.
- `mtplx/benchmarks/prompts/qwen38_generation_context.py`,
  `mtplx/benchmarks/prompts/qwen38_naturalistic_generation_patch.jsonl` —
  hash-pinned fixtures (byte-identical copies).
- `scripts/deepseek_v41/served_cell_bench.sh` — the gpu_window STEP.
- `scripts/deepseek_v41/ab_decode_env_levers.py`,
  `scripts/deepseek_v41/bench_standard_shape.py` — `--prompt-ids-file` +
  `--prompt-seed`.
- `tests/test_server_cell_bench_deepseek_v41.py` — CPU tests.
