# W29 — K19: head only the last row at prefill

Worker: `feat/deepseek-v41-w29` (off `feat/deepseek-v41-streaming @ 106fdd48e`). CPU-only, no GPU,
no real artifact loaded. Implements kernel-ledger item **K19** (`docs/deepseek-v41/KERNEL_LEDGER.md
§K19`).

## TL;DR

At a 16,384-token prefill the DeepSeek-V4.1 lm head projected **every** row, building a
`[1, 16384, 129280]` f32 logits transient (**8.472 GB**) and running the `129280×5120` output GEMM over
16,383 rows the autoregressive path never reads — decode seeds only from the **last** token's logits.
The last-row head already existed as the runtime `forward_ar` contract **`logits_keep`** (W23); W29
adds the explicit K19 alias **`logits_rows="last"`** and **activates it for the mxfp4 serve lane**, so
the transient and the wasted GEMM are dropped by default. Output is byte-unchanged for AR decode
(the surviving row's argmax is identical).

## What changed

| File | Change |
|---|---|
| `mtplx/models/deepseek_v41.py` | `Model.__call__` gains a `logits_rows` keyword (`"last"` / `"all"` / int) plus `_resolve_logits_keep` / `_logits_rows_to_keep` helpers. `logits_rows` and the pre-existing `logits_keep` funnel into **one** trailing-row slice `h[:, -keep:, :]` applied before the head. Default (both `None`) heads every row — byte-identical to the pre-K19 forward. Docstring documents the K19 rationale. |
| `mtplx/data/expert_profiles.json` | `deepseek-v41-mxfp4-75` profile `child_env` gains `"MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS": "0"`, which turns the runner's already-wired final-logits-only prefill **on** for this lane. |
| `tests/models/test_deepseek_v41_head_last_row.py` | New — 6 tests (a/b/c below). |
| `tests/test_deepseek_v41_serve_profile.py` | +2 tests locking the K19 lane wiring. |
| `docs/deepseek-v41/KERNEL_LEDGER.md` | K19 landed-note + analytical reduction table + audit. |

### Why no new keyword mechanism and no `generation.py` change

- **`logits_keep` is the K19 mechanism, already present.** W23 added `logits_keep` to
  `Model.__call__` as the uniform `MTPLXRuntime.forward_ar` contract; `logits_keep=1` slices
  `h[:, -1:]` before the head — exactly the K19 slice. A second, independently-named keyword would be
  dead API from the runner's perspective (`forward_ar` threads `logits_keep`, not `logits_rows`), so
  `logits_rows="last"` is a thin, explicit alias over the **same** slice path rather than a parallel
  mechanism. The task's "(or an equally clear name)" is satisfied by `logits_keep`; `logits_rows` is
  added for direct/test callers who want the intent spelled out.
- **The runner already narrows prefill logits**, gated by `generation._final_logits_prefill_enabled()`.
  Every `_prefill*` entry emits **no** logits for the prompt body (`emit_logits=False`) and
  `logits_keep=1` for the last token when the gate is on. The gate defaults **off** (full prefill
  logits) and is turned on per-lane by `MTPLX_SUSTAINED_PREFILL=1` **or**
  `MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS=0`. DSV4.1 deliberately does **not** run the full
  sustained-prefill bundle (it uses its own W20 `MTPLX_DSV41_PREFILL_CHUNK` layout), so it never
  opted into last-row prefill — hence W20 bounded the attention/score transient while the head still
  built the 8.47 GB logits. Setting the single env in the lane's `child_env` is the narrowest fix and
  needs **no** `generation.py` edit.

## Call-site audit — who reads DSV4.1 prefill logits

`grep`ed `mtplx/` for the model forward during prefill (`runtime.forward_ar`, `resident_loader`,
`generation._prefill*`, the served path). Result: every prefill consumer reads only the last row,
except the dedicated prompt-logprobs path.

| Call site (`mtplx/generation.py`) | Rows of prefill logits used | Behaviour under the K19 gate | Needs all rows? |
|---|---|---|---|
| `_prefill` (plain AR) | body `prompt_ids[:-1]` logits computed-then-**discarded** when gate off; returns `logits[:, -1, :]` only | body `emit_logits=False`, last token `logits_keep=1` | **No** |
| `_prefill_committed_mtp_history_streaming` (DSpark MTP) | last row only; body forward returns `hidden` (MTP history), logits discarded | body `emit_logits=False` (still returns hidden), last token `logits_keep=1` | **No** |
| `_prefill_with_hidden_sequence` | last row only | `emit_logits=not final`, last token `logits_keep=1` | **No** |
| `_prefill_restored_prompt_suffix` / `append_history` (session restore) | last row only | `logits_keep=1` — and off for this lane anyway (`MTPLX_SESSION_STORE_ON_PREFILL=0`, W22) | **No** |
| `score_prompt_logprobs` (prompt logprobs / echo) | **every** prompt row | passes `emit_logits=True` unconditionally, **does not** consult the gate; independently chunked (`chunk_size=256`, ≤256×vocab resident) | **YES — left as-is** |
| MTP decode-verify (K+1 rows) | all K+1 rows | **not prefill** (decode traffic); passes neither selector | n/a — unchanged |
| DSpark drafter (W23) | reads `main_hidden`, never the lm head | unaffected by head narrowing | n/a |

The one all-rows consumer, `score_prompt_logprobs`, is the historical 32k memory-balloon root cause;
it is already chunked to bound its own transient and does not go through the last-row gate, so the K19
lane activation cannot regress it.

## Tests (CPU, `mx.set_default_device(mx.cpu)`, tiny synthetic configs)

`tests/models/test_deepseek_v41_head_last_row.py` (6):
- **(a) exactness** — `logits_rows="last"` is **bit-identical** to `logits_keep=1` (`mx.array_equal`);
  the surviving row matches the all-rows tail to ~1.8e-7 with an **identical argmax**, for one-shot and
  W20 chunked prefill (chunks 1/3/5/7). `logits_rows="all"` == the default.
- **(b) head-once** — under chunked prefill (chunk=3 → 9 backbone spans) the head is invoked
  **exactly once**, and in last-row mode it projects **1 row** vs `s` (monkeypatched counting head);
  `emit_logits=False` touches the head **0 times**.
- **(c) DSpark unchanged** — with `logits_rows="last"` the DSpark `main_hidden` is bit-identical
  (`mx.array_equal`) and the drafted logits (`mtp_forward`) are bit-identical — the draft reads
  `main_hidden`, never the lm head.

`tests/test_deepseek_v41_serve_profile.py` (+2): the mxfp4 `child_env` carries
`MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS=0`, and applying it flips `_final_logits_prefill_enabled()` on.

### On the "exact (`mx.array_equal`)" bar

The task asked last-row == all-rows-tail exactly. That is **not attainable and not the point**: the lm
head is a matmul whose rounding depends on its row count M, so heading M=1 row (the K19 win) rounds
differently from M=s by ~1.8e-7 (measured on CPU; argmax identical). Strict equality would require
computing the full head, defeating K19. The exact guarantee W29 holds is
`logits_rows="last"` ≡ `logits_keep=1` ≡ `self.head(h[:, -1:])`; the all-rows comparison uses
`allclose(atol=1e-5)` + argmax equality.

### Suite result (step 4d)

`tests/models/test_deepseek_v41_*.py tests/test_deepseek_v41_*.py` under
`nice -n 19 … -m pytest` (CPU, no `-n auto`): **204 tests → 164 passed, 39 skipped, 1 failed**.
The single failure — `test_deepseek_v41_streaming_clamp.py::test_spec_swiglu_limit_values` — is
**pre-existing and unrelated** (it asserts on `MODEL_SPECS[*].swiglu_limit` in
`expert_streaming_models.py`, a file W29 never touched); verified failing at base `106fdd48e` with all
W29 changes stashed. New/touched K19 tests: **8 added, all pass.**

## Ship note for the orchestrator

The `child_env` key changes the **served default** for the `deepseek-v41-mxfp4-75` lane (last-row
prefill head). Output is byte-identical for AR decode (argmax-exact) and the change only drops memory
(8.47 GB at 16K → 0.49 MB) and the wasted head GEMM, so it is memory-protective for the 100 GB knob.
The ledger schedules K19's **magnitude** (TTFT / peak) as GPU window **KG-g** (after KG-b); that
window measures the win, not correctness. If David prefers to gate the served flip behind KG-g, revert
the one JSON line — the model mechanism and tests stand independently.

## Commit

`feat/deepseek-v41-w29` — SHA `__W29_IMPL_SHA__` (recorded post-commit; see `git log`).
Attribution check: `python3 scripts/check_ai_attribution.py --range feat/deepseek-v41-streaming..HEAD`
prints clean.
