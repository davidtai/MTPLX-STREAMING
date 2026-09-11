# W52 — DeepSeek-V4.1-Flash chat format, template, and served BOS

Fixes the two W49 serving-correctness defects: the DSV4.1 served chat path fell
through to a plain `"user: " + content + "\nassistant:"` render, and no BOS id 0
was ever prepended (chat or completions). W52 establishes the official format
from primary sources, installs a byte-for-byte Jinja template in the artifact,
adds a served code fallback + BOS, and rewires the W49 harness.

## 1. The official DeepSeek-V4.1-Flash chat format (primary sources)

All sources are the DeepSeek-V4.1-Flash source checkpoint shipped in
`~/models/DeepSeek-V4.1-Flash-src/` (HF `deepseek-ai/DeepSeek-V4.1-Flash`,
revision `dba1be0a…`, per the artifact `conversion-manifest.json`).

- **`encoding/encoding.py`** — the standalone, self-contained prompt-format
  reference (`encode_messages`). This is authoritative; the template and the
  code fallback are transliterations of it.
- **`encoding/README.md`** — prose spec of the format.
- **`encoding/tests/test_input_*.json` + `test_output_*.txt`** — 5 paired
  reference vectors (the acceptance bar).
- **`inference/generate.py`** — the reference runtime; its CLI default is
  `--thinking-mode "chat"`.
- **`README.md`** (model card) — architecture + the "continuously controllable
  reasoning effort 1–100" description; instruct evals use `reasoning_effort=100`,
  `temperature=1.0, top_p=0.95`.

### Special tokens (verified against the artifact tokenizer)

| token | id | role |
|---|---|---|
| `<｜begin▁of▁sentence｜>` | 0 | BOS (start of conversation) |
| `<｜end▁of▁sentence｜>` | 1 | EOS (end of assistant turn) |
| `<｜User｜>` | 128803 | user turn prefix |
| `<｜Assistant｜>` | 128804 | assistant turn prefix |
| `<｜System｜>` | 128799 | system / reasoning-effort prefix |
| `<think>` / `</think>` | 128821 / 128822 | reasoning block delimiters |
| `｜DSML｜` | 128825 | tool-markup token (used as `<｜DSML｜ calls>` etc.) |
| `<｜latest_reminder｜>` | 128828 | date/locale reminder prefix |
| `<｜action｜>`…`<｜read_url｜>` | 128829… | internal quick-instruction tasks |

`<｜begin▁sys｜>` (128826) / `<｜end▁sys｜>` (128827) exist in the vocab but are
**not** used by the reference encoder — system messages use `<｜System｜>`.

### Structure

Basic multi-turn:
```
<｜begin▁of▁sentence｜>{system}<｜User｜>{u1}<｜Assistant｜>{think}{a1}<｜end▁of▁sentence｜><｜User｜>{u2}<｜Assistant｜>{think}…
```

- **BOS** is prepended once at the very start of the conversation
  (`encode_messages`, `add_default_bos_token=True`).
- A leading **system** message gets a `<｜System｜>` prefix (even in chat mode).
- **chat mode** closes the thinking block immediately: the assistant header is
  `<｜Assistant｜></think>`, so the model answers directly.
- **thinking mode** opens it: `<｜Assistant｜><think>`, and the model reasons
  inside `<think>…</think>` before answering. The **reasoning-effort prefix** is
  injected once, at conversation index 0 only, as a `<｜System｜>` block:
  `<｜System｜>Reasoning Effort: {budget} (range 1-100, the higher the value, the
  more thorough the reasoning)\n\n`. Budget defaults to `"high"` = 75; aliases
  `low/high/max` → 50/75/100; an int 1–100 is passed through.
- **System-prompt placement**: content sits directly after BOS (with the
  `<｜System｜>` prefix); the tool schema and response-format blocks append to it
  with `\n\n` separators.
- **Multi-turn / drop_thinking**: by default reasoning from assistant turns
  *before the last user message* is stripped (those turns render chat-style);
  the active round keeps its `<think>…</think>`. `drop_thinking` is forced OFF
  when tools are present.
- **Tools** (DSML): the schema is injected into the system prompt; assistant
  tool calls render as `<｜DSML｜ calls> / <｜DSML｜ invoke> / <｜DSML｜ parameter>`
  blocks (note the leading space in the V4.1 tag names); `tool` role messages
  merge into a user turn as `<tool_result>…</tool_result>`.
- **`<｜latest_reminder｜>`** and the quick-instruction **task** tokens
  (`<｜action｜>`, …) are supported per the reference.

### Is it a thinking model, and how is it toggled?

DSV4.1-Flash **is** a reasoning model, but thinking is a **toggle**, not always
on. It is toggled by which token closes the assistant header:
`<think>` opens reasoning (thinking mode) vs `</think>` pre-closes it (chat mode)
— i.e. the reference's `thinking_mode` in `{"thinking","chat"}`. There is no
separate `<think></think>` pre-fill variant; chat mode simply places `</think>`
directly after `<｜Assistant｜>`.

W49's claim that "DSV4.1 has no thinking mode" was an artifact of the missing
template (with no template, `enable_thinking` never reached a renderer, so ids
were identical on/off). That is corrected here.

## 2. BOS and thinking-default decisions

- **BOS = id 0, prepended exactly once.** `tokenizer_config.json` has
  `add_bos_token: false` and the `tokenizer.json` post-processor is `ByteLevel`
  (adds nothing), so neither the chat nor the completions encode adds BOS on its
  own. The reference `encode_messages` prepends the literal
  `<｜begin▁of▁sentence｜>` (→ id 0); the in-process bench and PORT_CONTRACT (W8)
  require BOS first. W52 makes the chat render carry it (template/fallback both
  emit the literal token) and the `/v1/completions` path prepend id 0 (idempotent
  when a `list[int]` prompt already starts with 0).

- **Thinking default = OFF (`enable_thinking=false`), recorded choice.** The
  only explicit defaults in the official code are `generate.py --thinking-mode
  "chat"` and the encoding-test harness rendering with `thinking_mode="chat"`.
  Both say chat (thinking OFF), so that is the served/harness default.
  `reasoning_effort` is inert in chat mode and is omitted from the request there.
  DeepSeek's own **instruct-eval methodology uses thinking + reasoning_effort=100**
  (`temperature=1.0, top_p=0.95`); to mirror it, set
  `--dsv41-enable-thinking --dsv41-reasoning-effort 100` (or env
  `DSV41_ENABLE_THINKING=1 DSV41_REASONING_EFFORT=100`).

## 3. The exact single-turn render

Prompt: user content `What is 2+2?`, `add_generation_prompt=True`.

- **chat mode (default)** — text:
  `<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜></think>`
  ids: `[0, 128803, 3085, 344, 223, 20, 13, 20, 33, 128804, 128822]`
- **thinking mode** — text:
  `<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher
  the value, the more thorough the reasoning)\n\n<｜User｜>What is 2+2?<｜Assistant｜><think>`
  ids: begin `[0, 128799, …]`, end `[…, 128804, 128821]`.

Both begin with BOS id 0; chat ends `</think>` (128822), thinking ends `<think>`
(128821). Verified identical from the artifact template, the served code
fallback, and the harness counter.

## 4. What changed

### Artifact (outside git — must be re-uploaded to HF)

- **Added** `~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/chat_template.jinja`
  (sha256 `566c0220…`), a new sidecar file. transformers 5.8 auto-loads it on
  `AutoTokenizer.from_pretrained`, so `tokenizer.chat_template` is now set.
  - No pre-existing `chat_template.jinja` was overwritten (no `.orig` needed).
  - **Manifests unaffected:** neither `conversion-manifest.json` nor
    `expert-manifest.json` (nor `model.safetensors.index.json`) hashes the
    tokenizer files — grepped; `expert-manifest` mentions no tokenizer path and
    its `manifest_sha256` covers experts only. **No digest bump required.**
  - The canonical source of this file is version-controlled at
    `mtplx/templates/deepseek_v41/chat_template.jinja` (identical bytes); the
    artifact copy is a deployment of it.
  - **HF republish:** add `chat_template.jinja` to the
    `OpensourceWTF/DeepSeek-V4.1-Flash-*-MTPLX-streaming*` repo when it is next
    published so downstream loaders get the template.

### Repo (committed)

- `mtplx/chat_encoding.py` — `is_deepseek_v41_tokenizer`,
  `render_deepseek_v41_prompt`, `encode_deepseek_v41_messages`,
  `DEEPSEEK_V41_BOS_ID` (the MLX-free port + detector).
- `mtplx/server/openai.py`:
  - `_encode_messages_uncached` — a deepseek_v41 code fallback (gated on
    `chat_template is None`) so a template-less checkpoint still gets the correct
    render + BOS, never the plain `user:/assistant:` render (which begins id
    5265).
  - `_encode_prompt` — prepends BOS id 0 on the `/v1/completions` path for the
    deepseek_v41 family (idempotent), via `_maybe_prepend_deepseek_v41_bos` and a
    memoized `_is_deepseek_v41_tokenizer_cached`.
  - `_coerce_token_ids` — now unwraps a transformers `BatchEncoding` (a
    `UserDict`, not a `dict`) to its `input_ids`; `apply_chat_template(
    tokenize=True)` returns one on a raw fast tokenizer, which the deepseek
    thinking template path hits. Family-agnostic robustness fix.
- `mtplx/templates/deepseek_v41/chat_template.jinja` — canonical template.
- `scripts/fable/server_cell_bench.py` (W49 harness) — the deepseek-v41 counter,
  `server_prompt_ids`, and render now template the real chat prompt (BOS
  included) instead of the plain render; `cell_sampling` sends
  `enable_thinking=false` and omits `reasoning_effort`; `template_settings`
  records the new truth; `--dsv41-enable-thinking` / `--dsv41-reasoning-effort`
  (and `DSV41_*` env) expose the thinking override.
- Tests: `tests/test_deepseek_v41_chat_template.py` (new) and
  `tests/test_server_cell_bench_deepseek_v41.py` (updated). 25 passed, CPU-only.

## 5. Validation

- Template (auto-loaded sidecar) **and** the code fallback reproduce reference
  vectors 1–4 **byte-for-byte** (tool-call, chat multi-turn+system, thinking +
  mid-conversation system + latest_reminder, task=action), plus the README
  quickstart, single-turn chat/thinking, and `add_generation_prompt=False`.
- Vector 5 is vision (image content); the text serving path drops images, so it
  is **out of scope** (documented; not asserted).
- Template ids == code-fallback ids == harness-counter ids, all BOS-first.
- `/v1/completions` BOS prepend verified idempotent on str and `list[int]`.

Namespaced-tool *schema* merging and the multi-tool-result call-order sort are
Python-port-only refinements (the template targets the OpenAI tools-arg
convention: schema on the first system message). Neither affects the served
benchmark shape or vectors 1–4.
