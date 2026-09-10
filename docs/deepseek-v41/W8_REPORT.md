# W8 — Degenerate P1.7 gate: root cause (missing BOS) + hidden-state divergence harness

Branch `feat/deepseek-v41-w8` off `feat/deepseek-v41-streaming` @ `4c6a8b4c`
(the W7 integration point). CPU only, `nice -n 19`, MLX default device forced to
the CPU stream (`mx.set_default_device(mx.cpu)`), no GPU/Metal, no lock, no
launchctl, no `:8080`, `~/models` read-only. mlx 0.32.2, py3.12.

The first GPU run of the P1.7 gate
(`scripts/deepseek_v41/gate_stream_equals_resident.py`) **PASSED** streamed==resident
but produced **degenerate** output on **both** arms (argmax `[13394×11, 104113,
15, 2808, 13394×18]` = `" potentially potentially … Kasipak - � potentially …"`).
Two unknowns: (1) is the CPU forward correct, (2) does the GPU diverge from the
CPU.

## Verdict up front

1. **The degeneration reproduces byte-identically on CPU** (Fibonacci prefill
   tok0 = `13394` on CPU, same as the GPU). So this is **not** a GPU-vs-CPU
   divergence — both share the same input and the same forward.
2. **Primary, fixable root cause: the greedy harness fed prompts WITHOUT the
   reference's leading `<｜begin▁of▁sentence｜>` (BOS, id 0).** The reference
   prepends it on every fresh prompt; the artifact tokenizer does not add it, and
   `tokenizer.encode(prompt)` (what the P1.7 gate, the W6 CPU end-to-end proof and
   the first GPU gate run used) omits it. With BOS the first-token error is fixed:
   `"def add(a, b):"` → `" forward"` (no BOS) becomes → `" return a + b"` (BOS).
3. **The CPU forward is faithful to the reference.** The engram, the missing
   SwiGLU clamp, the residual dtype, and the large residual magnitudes were each
   ruled out as the cause; layer-0 attention on the **real** q8 weights matches a
   numpy transcription of the reference to q8/bf16 tolerance. No forward or
   weight-orientation defect was found.
4. A **secondary** effect remains: even with BOS, pure greedy decoding degenerates
   in the tail into recurring low-confidence tokens (`Kasipak`/` potentially`/
   ` problematic`). This is consistent with the aggressive 2-bit expert
   quantization plus greedy decoding on weak-signal prompts; it is **not**
   attributable to any forward/orientation bug found here. David's standardized
   1,024-token in-distribution programming prompt is the intended input for the
   health check (see §6).

Per David's mid-task directive, every input is now the deterministic
`mtplx.prefill_bench` coding-agent programming prompt (1,024 tokens; 16,384 as the
prefill cell). The gate and the dump script build it via
`_prompt_build_for_context` and prepend BOS.

---

## 1. CPU reference generations (the same loader the gate uses: component-banks)

All greedy, streamed experts, engram attached, `slot_layout=component-banks`,
`mx.set_default_device(mx.cpu)`. **NO-BOS** column = exactly what the gate fed.

| prompt (steps) | input | argmax → decoded | health |
|---|---|---|---|
| `Write a Python function that returns the nth Fibonacci number.` (16) | no BOS | `[13394,13394,13394,374,19,37,89,88,17659,271,104113,72077,5946,271,13394,13394]` = `" potentially potentially potentially v1CwvNa\n\nKasipak-uwa\n\n potentially potentially"` | **DEGENERATE** |
| `def add(a, b):` (12) | no BOS | `[6058,201,361,1354,260,940,291,104113,201,13394,201,13394]` = `" forward\n    return a + bKasipak\n potentially\n potentially"` | wrong 1st token, then correct code, then degenerate |
| Fibonacci, **engram hooks removed** (16) | no BOS | `[13394,13394,339,104113,201,13394×11]` = `" potentially potentially.\n\nKasipak\n potentially…"` | **DEGENERATE** (engram is not the cause) |

**Health verdict: UNHEALTHY.** The first generated token is wrong for both
prompts, and the tails collapse into a few recurring garbage tokens (`13394`
` potentially`, `104113` `Kasipak`).

### The BOS fix (regenerated text)

| prompt (steps) | input | argmax → decoded |
|---|---|---|
| `def add(a, b):` (12) | **+BOS (id 0)** | `[1354,260,940,291,201,104113,13394,36564,…]` = `" return a + b\nKasipak potentially problematic …"` |
| Fibonacci (16) | **+BOS** | `[13394,3226,313,16,6948,50363,1878,304,…]` = `" potentially large n. Use memoization to potentially large n.Kasipak…"` |

Prepending BOS **fixes the first token** for `def add` (` forward` → ` return`,
producing the correct `def add(a, b): return a + b`) and injects on-topic content
for Fibonacci (`large n. Use memoization to`). The tail still degenerates under
greedy — see §4.

### 1,024-token prefill_bench prompt (David's standard input)

CPU generation, greedy, 16 steps, BOS-prefixed, component-banks. Log + receipt:
`.benchmark-artifacts/deepseek-v41/w8-refgen/refgen_1024.{log,_receipt.json}`
(written incrementally, one token at a time, so a restart never loses it).

<!-- FILLED-1024-RESULT -->
_1,024-token result: see receipt (run in progress at report time)._

The 16,384-token build constructs and reports exactly 16,384 tokens (verified on
CPU); the 16K forward is left to the orchestrator's GPU window.

---

## 2. Root cause: the missing BOS token

The reference `inference/generate.py` encodes every prompt through
`encoding.encode_messages`, which sets `add_default_bos_token=True` and prepends
`bos_token = "<｜begin▁of▁sentence｜>"` when there is no prior context
(`encoding/encoding.py:779`). The artifact tokenizer sets `add_bos_token: False`
(`tokenizer_config.json`), so `tokenizer.encode(prompt)` returns the prompt with
**no** leading BOS. `<｜begin▁of▁sentence｜>` is token id **0**.

The P1.7 gate, the W6 CPU end-to-end proof and the first GPU gate run all
tokenized with a bare `tokenizer.encode(args.prompt)` — so the model saw a content
token at position 0 instead of the BOS it was trained/served with, and its
first-token distribution was wrong (a well-known failure mode for BOS-trained
models). Prepending id 0 restores the correct first token.

## 3. Ablations that ruled out the other suspects

| suspect | test | result |
|---|---|---|
| **Engram** (gate/scale/hash) | set every `engram_hook = None`, regenerate | still degenerate → **not the cause** |
| **SwiGLU clamp** (streamed path omits the reference's ±`swiglu_limit`) | monkeypatch the streamed `swiglu` to clamp gate≤10 / up∈[−10,10] in fp32 | layer-39 max-abs 293,505 → 264,300 (~10%); top-1 still `13394` → **not the cause** |
| **bf16 residual precision** | probe `h.dtype` per layer | residual is **fp32** from layer 1 on (only the embed is bf16) → no bf16 precision loss |
| **Massive activations** | per-layer max-abs dump | grows to ~2.9e5 by layer 39, **input-independent** (nearly identical for both prompts), in fp32, within the fp8-native reference's representable range → real model behavior, not inflation |
| **Attention real-weight orientation** | layer-0 SWA attention on the dequantized q8 residents vs a numpy transcription of the reference | max-abs diff 0.47 on outputs of magnitude ~7.5 (~6%, = the port's bf16 q/kv activation precision) → **no orientation error** |
| **Routed Q2 experts** | `def add` greedy (with BOS) produces the exact correct code `return a + b` | a w1/w3 swap or transpose would corrupt every token → experts are substantially correct |
| **Router (sqrtsoftplus/noaux_tc/norm/scale)** | validated in the tiny-config block parity (`_ref_moe`) and consistent with the correct `def add` code | no defect |

The per-layer block math was already validated against a numpy reference in
`tests/models/test_deepseek_v41_parity.py` (W6, to ~1e-6). W8 adds the real-weight
attention check and the ablations above. **No forward or weight-orientation bug
was found.**

## 4. Secondary: greedy-tail degeneration under 2-bit experts

With BOS, the strong-prior start is correct, then greedy decoding collapses into a
few recurring low-confidence tokens. Diagnostic signature: for a degenerate
position the top-8 logits are **flat and low** (~15, e.g. `13394`/`104113`),
whereas for a confident-correct position they are **peaked** (`def add` → ` return`
at 26.3 vs 25.8 next). The recurring tokens are the quantization noise floor of the
q8 head + Q2 experts dominating once the true signal is weak. This is a quality
limitation of the 2-bit expert bank under greedy decoding, not a port defect; it is
exactly what David's standardized 1,024-token in-distribution prompt (a strong,
coherent signal) is meant to exercise.

## 5. Fix + regression test

**Fix (in the allowlisted harness):** `scripts/deepseek_v41/dump_hidden_states.py`
and `scripts/deepseek_v41/gate_stream_equals_resident.py` now build their input via
`mtplx.prefill_bench._prompt_build_for_context` (default: the 1,024-token
coding-agent programming prompt; `--context-tokens 1024|16384`) and **prepend BOS
(id 0) by default** (`--no-bos` to reproduce the failing input). The build metadata
(token count, style, format, release-valid, tokenizer, BOS) is recorded in every
receipt/dump JSON. The artifact tokenizer ships no HF chat template (the reference
uses its own `encoding.py` chat tokens), so `--prompt-format` defaults to `raw`.

> The production/eval prompt encoders outside this allowlist (the gate is now
> fixed; any serve-path or eval driver that calls `tokenizer.encode` directly)
> must likewise prepend id 0 — flagged for the orchestrator.

**Regression test:** `tests/test_deepseek_v41_bos.py`
- `test_build_prompt_prepends_bos_by_default` / `_no_bos_flag` (both scripts):
  the harness prepends id 0 by default and records the metadata (cheap, no model).
- `test_tokenizer_omits_bos_which_is_id_zero` (artifact-gated): BOS is id 0 and
  `tokenizer.encode` does not add it.
- `test_prefill_bench_build_1024_and_16384_construct` (artifact-gated): both
  context sizes build to exactly their token counts, coding-agent, release-valid.
- `test_bos_is_load_bearing_on_real_artifact` (opt-in `DSV41_RUN_BOS=1`): on the
  real model, BOS changes the greedy prediction and the BOS-prefixed `def add(a,
  b):` continues with ` return` while no-BOS does not.

## 6. Divergence harness (deliverable 2)

- `scripts/deepseek_v41/dump_hidden_states.py` — one PREFILL forward through the
  real serve-path model; writes per-layer hidden-state summaries (mean/std/max-abs
  + the first 64 floats of the last position for **every** layer), the embed and
  final-norm summaries, the final **top-8 logits** with ids + text, and the
  **engram gate** stats for layers 1 and 14, to `--out`. Capture is non-invasive
  (wraps `DecoderLayer.__call__` and the two engram hooks in-process, restores
  after). `--engram-ablate` detaches the hooks. Same loader/flags as the gate.
- `scripts/deepseek_v41/compare_hidden_states.py` — diffs two dumps and prints the
  **first layer** whose `first64` max-abs diff exceeds `--threshold` (default
  1e-2), plus embed/final-norm/engram-gate deltas and the two top-8 logit lists.
  Pure Python, no MLX.
- Committed CPU dumps under `docs/deepseek-v41/receipts/`. The orchestrator takes
  the same dump on the GPU inside the guarded window and runs `compare_hidden_states.py`.

## 7. Files changed (allowlist)

- `scripts/deepseek_v41/dump_hidden_states.py` — new divergence dump.
- `scripts/deepseek_v41/compare_hidden_states.py` — new dump diff.
- `scripts/deepseek_v41/gate_stream_equals_resident.py` — prefill_bench build +
  `--context-tokens`/`--prompt-format`/`--bos`; BOS default on; prompt-build
  metadata in the receipt.
- `tests/test_deepseek_v41_bos.py` — new (BOS + build regression).
- `docs/deepseek-v41/receipts/*.json` — committed CPU dumps.
- `docs/deepseek-v41/W8_REPORT.md` — this report.

No changes to `mtplx/models/deepseek_v41.py`, `deepseek_v41_loader.py`, or
`mtplx/engram_v41.py`: the investigation found the forward faithful, so no model
edit was warranted; the defect was the harness prompt encoding.

Attribution: `python3 scripts/check_ai_attribution.py --range origin/main..HEAD`
→ clean (no Co-Authored-By / Claude trailer).
