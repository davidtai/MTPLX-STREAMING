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
3. **BUT BOS is not the whole story — there is a second, unidentified forward
   defect.** A teacher-forced PREFILL of near-deterministic code (probe A, with
   BOS) matches the ground-truth next token at only **16/30** positions and
   **locks onto a fixed junk-token set** (`13394 ' potentially'`, `104113
   'Kasipak'`, `36564 ' problematic'`) at scattered positions — including ones
   whose continuation is certain (pos 19 after `def sub(a, b):\n   ` predicts
   `Kasipak` where ` return` is inevitable). This is a **single-forward prefill
   defect**, not the decode loop (§4). The **cause was NOT identified** — root
   causing is handed to W9 (torch reference goldens) and W10 (a faithful
   transliteration of the reference model).
4. **The earlier "2-bit quality limitation" claim is WITHDRAWN.** A quantization
   noise floor degrades fluency gradually and does not *select which positions to
   spoil* or collapse to a fixed two-token set; probe A shows the prefill is wrong
   at specific positions while nailing others (in-context ` sub`/` mul`). Engram,
   SwiGLU clamp, residual dtype and layer-0 attention orientation were each ruled
   out (§3–§4), but the actual defect remains open.

Per David's mid-task directive, every input is the deterministic
`mtplx.prefill_bench` coding-agent programming prompt (1,024 tokens; 16,384 as the
prefill cell). The gate and the dump script build it via
`_prompt_build_for_context` and prepend BOS. (The 1,024-token CPU generation was
stopped before completing — see §4.)

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
| **Massive activations** | per-layer max-abs dump | grows to ~2.9e5 by layer 39, **input-independent** (nearly identical for both prompts), in fp32 (so not a bf16-precision effect). Whether this magnitude is reference-faithful is **unconfirmed** (needs W9 goldens) and is a candidate mechanism for the §4 fixed-token bias |
| **Attention real-weight orientation** | layer-0 SWA attention on the dequantized q8 residents vs a numpy transcription of the reference | max-abs diff 0.47 on outputs of magnitude ~7.5 (~6%, = the port's bf16 q/kv activation precision) → **no orientation error** |
| **Routed Q2 experts** | `def add` greedy (with BOS) produces the exact correct code `return a + b` | a w1/w3 swap or transpose would corrupt every token → experts are substantially correct |
| **Router (sqrtsoftplus/noaux_tc/norm/scale)** | validated in the tiny-config block parity (`_ref_moe`) and consistent with the correct `def add` code | no defect |

The per-layer block math was validated against a numpy reference in
`tests/models/test_deepseek_v41_parity.py` (W6, to ~1e-6). W8 adds the real-weight
attention check and the ablations above. These **rule out the listed suspects**,
but they do **not** clear the forward: §4 shows a real prefill defect whose exact
site is still open (candidates that remain untested at real dims/weights: the MoE
routed-expert Q2 decode across many experts, the full-stack hyper-connection
threading over 40 real layers, and the final collapse/head under the large
residual). W9 goldens + the W10 transliteration will localize it.

## 4. The prefill forward is wrong at scattered positions (cause NOT identified)

The tail degeneration is **not** a decode-loop bug and **not** a 2-bit quality
floor. A discriminating experiment (`scripts`/`.benchmark-artifacts` runner
`decode_probe.py`, evidence `docs/deepseek-v41/receipts/cpu_decode_probe_ABC.json`)
on the cheap 30-token probe, all with BOS, component-banks, the gate's loader:

**A — teacher-forced PREFILL** of a 3-function file
(`def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef
mul(a, b):`), one forward, argmax at every position vs the ground-truth next
token: **16/30 matches.** The model **locks onto a fixed junk set**
(`13394 ' potentially'`, `104113 'Kasipak'`, `36564 ' problematic'`) at positions
6, 11, 12, 19, 20, 21, 24, 26 — e.g. pos 19 after `def sub(a, b):\n   ` predicts
`Kasipak` where ` return` is certain; pos 12 after `\n\n\n` predicts `Kasipak`
instead of `def`. Yet it **nails** the in-context positions (pos 13 ` sub`, pos 25
` mul`, plus `,`/` b`/`\n\n\n`). The **same** target token ` a` (260) is correct at
pos 8 but junk at pos 20. A quantization floor cannot select which positions to
spoil, and would not collapse to a two-token set — so this is a **real forward
defect exercised by a single prefill**, not decode and not quality.

**B — incremental KV decode** from `def add(a, b):`: `[1354,260,940,291,201,
104113,13394,36564,13394,36564,…]` = `" return a + b\nKasipak potentially
problematic potentially problematic…"` — correct start, then the same lock.

**C — re-prefill from scratch each step**: confirms the same behavior (B ≈ C).
Since A (pure prefill) is already wrong, B/C are only confirmation; the defect is
in the shared forward, reached by prefill.

**Ablations (probe A, 30 positions, WITH BOS):**

| config | matches/30 | note |
|---|---:|---|
| baseline (engram on, full attention) | **16/30** | junk at 6,11,12,19,20,21,24,26 |
| **engram hooks = None** | **13/30** | *worse* — removing engram loses correct predictions at 8,9,10,13,14; **engram is not the cause** |

The engram-off run is *worse*, so the engram is contributing correctly, not
injecting the junk. (The manifest `hash_multipliers` were also verified to match
the reference `compute_hash_multipliers` recipe exactly — seed `10007*layer_id`,
`*2+1`, bound from the compressed vocab 99092.) The SWA-only ablation and the
per-layer good-vs-bad-position bisect were **not run**: per David's decision the
old model file is being replaced by a faithful transliteration of the reference
(W10), and W9 provides torch reference goldens, so further ablation of the current
file was stopped.

**Cause: NOT identified in W8.** What is established: the defect is in the prefill
forward (not the decode loop, not the KV cache, not engram, not the SwiGLU clamp,
not residual dtype, not layer-0 attention orientation); it manifests as a
position-dependent collapse to a fixed junk-token set while in-context predictions
succeed. Root causing is handed to W9 (torch goldens per layer/position) + W10
(faithful port); the dump/compare harness (§6) and the BOS finding (§2) are the
tools they use.

The 1,024-token CPU generation was launched (incremental receipt at
`.benchmark-artifacts/deepseek-v41/w8-refgen/refgen_1024_receipt.json`) but its
1,025-token CPU prefill did not finish within the window and I stopped the process
to free CPU for this discriminating experiment; it emitted no tokens (receipt
stuck at `prefilling`). It should not be re-run until the forward defect is fixed —
its output would be void.

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
- `docs/deepseek-v41/receipts/*.json` — CPU hidden-state dump + A/B/C probe evidence.
- `docs/deepseek-v41/W8_REPORT.md` — this report.

Diagnostic runners (kept under `.benchmark-artifacts/deepseek-v41/w8-refgen/`, not
committed as code): `decode_probe.py` (A/B/C), `ablate_A.py` (ablation matrix),
`bisect_step.py` (layer bisect, unused after the stop).

No changes were made to `mtplx/models/deepseek_v41.py`,
`mtplx/models/deepseek_v41_loader.py`, or `mtplx/engram_v41.py`. The confirmed,
in-allowlist defect (missing BOS) is fixed in the harness (§2/§5). The **second
forward defect (§4) was left un-fixed by design**: David is replacing the model
file with a faithful transliteration (W10) rather than patching it, so an edit here
would be thrown away — the W8 deliverable for that defect is the reproduction, the
ruled-out suspects, and the dump/compare harness that W9/W10 use to localize it.

Attribution: `python3 scripts/check_ai_attribution.py --range origin/main..HEAD`
→ clean (no Co-Authored-By / Claude trailer).
