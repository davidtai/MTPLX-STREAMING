# W6 — DeepSeek-V4.1-Flash parity device fix + Engram end-to-end integration

Branch `feat/deepseek-v41-w6` off `feat/deepseek-v41-streaming` @ `086d6c18`
(W1 model core + W2 engram runtime + W3 loader/spec + W4 engram residents merged).
CPU only, `nice -n 19`, no GPU/Metal (no flock held). mlx 0.32.2, py3.12, no torch.

Three tasks:
1. Fix the device-dependent parity oracle (passed on GPU default, failed on CPU) and
   force the CPU stream in every DeepSeek-V4.1 test.
2. Attach Engram end to end: real `EngramV41` hooks on layers 1 and 14, built from the
   artifact's banks + W4 resident sidecar, with a per-cache `NgramHashState`.
3. Update the pre-W1 "raises until W1 lands" loader test to assert a real construct, and
   run a CPU end-to-end forward proof on the real artifact.

All deepseek_v41 / engram / ngram test files pass together and alone on CPU.

---

## Task 1 — parity was device-dependent (oracle bug, not tolerance)

### Direct device micro-test (fp32 inputs vs float64), mlx 0.32.2, this box

`mx.matmul` / `mx.einsum` on the **CPU stream** vs a numpy float64 reference and vs the
oracle's `_tf32` rounding (`scratchpad/micro_matmul.py`, `micro_einsum.py`):

| shape (M×K×N) | MLX-cpu vs f64 (max abs) | MLX-cpu vs **numpy fp32-accumulate** | MLX-cpu vs the **tf32 oracle** |
|---|---:|---:|---:|
| 32×32×32     | 4.12e-06 | **0.000e+00** | 1.33e-02 |
| 24×512×512   | 7.07e-05 | **0.000e+00** | 7.62e-02 |
| 288×5120×384 | 9.11e-04 | **0.000e+00** | 2.38e-01 |
| 64×6144×512  | 1.02e-03 | **0.000e+00** | 2.44e-01 |

einsum `bshd,btd->bsht` (d=16): MLX-cpu vs f64 = 2.09e-06, vs fp32-accumulate = 2.86e-06.
softmax (elementwise): MLX-cpu vs f64 = 1.2e-08.

**Finding.** MLX 0.32.2's CPU GEMM is **bit-identical to numpy float32-accumulate** (0.0 in
every shape) and its error vs float64 is pure fp32 rounding. It does **not** round operands
to tf32. W1's `_tf32` oracle (keep the top 10 of 23 mantissa bits) models MLX's **GPU/Metal**
matmul, not the CPU — which is exactly why the parity tests passed only while MLX's default
device was the GPU (the engram tests, which force `mx.set_default_device(mx.cpu)`, flipped the
shared process device to CPU and exposed the mismatch when run together). On CPU the tf32
rounding injected 1e-2–2e-1 of spurious error: swa attention 0.0104 (tol 1e-3), block hidden
0.0141 (tol 1e-2), and a csa2 argmax flip at pos 10 (ref top-2 margin 0.0096 > the 0.005 tie
band). The csa2 flip "varied run to run" because the corrupted margins were within the noise
band, not because of any real nondeterminism.

### Fix

`tests/models/test_deepseek_v41_parity.py`: `_mm`/`_einsum` now cast operands to fp32 and
accumulate in fp32 (matching MLX CPU bit-for-bit), then widen to float64 so the surrounding
elementwise reference (rmsnorm/softmax/rope/sinkhorn) stays high-precision. The `_tf32`
helper is removed. Every DeepSeek-V4.1 test file now forces `mx.set_default_device(mx.cpu)`
at module import (parity, config, loader_contract, `test_deepseek_v41_loader`,
`test_deepseek_v41_spec`, and the new engram-attach file; the engram/ngram files already did).

### Residual after the fix (no hidden bug)

| check | before (tf32, CPU) | after (fp32-accumulate) | tolerance |
|---|---:|---:|---:|
| swa attention max abs | 0.0104 (FAIL) | **8.3e-07** | 1e-3 |
| dense block hidden max abs | 0.0141 (FAIL) | **1.5e-06** | 1e-2 |
| csa2 logits max abs | — | **7.3e-06** | — |
| csa2 argmax mismatches | 1 (FAIL) | **0 / 288** | ≤2 tie flips |

The residual (~1e-6, accumulated over 8 layers) is exactly the elementwise fp32-vs-fp64
difference of the reference's rmsnorm/softmax/rope/sinkhorn — **not** a bf16-cast, RMSNorm-eps,
rope, or sink bug. `test_csa2_modes` was run 5× and is stable (0 mismatches every run).

---

## Task 2 — Engram attached end to end

### Wiring seam: `Model.attach_engram(...)`, called by the loader (not `__init__`)

Chosen seam: a new **`Model.attach_engram(engram_dir, *, tokenizer=None, cache_bytes=None)`**
method, invoked by `construct_deepseek_v41_resident_model` after the residents load.

Why not `Model.__init__`: the parity/contract unit tests construct `Model` cheaply and without
the artifact on disk; building real engram in `__init__` would couple every construction to the
tokenizer walk + the two 104 GiB row banks + the 319 MiB resident sidecar. Why not purely in the
loader: which layers are engram layers, how `make_cache` hands each sequence its own history, and
how the shared `NgramHashState` prototype is cloned are model internals — the loader only supplies
the on-disk path and the env cache budget. So the *mechanism* lives on `Model`; the loader is the
*caller*.

What `attach_engram` does, per `engram-manifest.json`'s `hashing.layer_ids` = `[1, 14]`:
- opens each layer's affine-q8 row bank (`EngramBank.open`, byte budget from
  `MTPLX_ENGRAM_CACHE_LIMIT` via `cache_bytes_from_env`) — and **keeps the banks alive on the
  model** (`self._engram_banks`), because `EngramBank.__del__` would otherwise close the
  `NGramRowCache` the hooks hold;
- loads the W4 resident `wkv`/`q_weight`/`k_weight` sidecar (`load_engram_residents`) and builds
  the `EngramV41` hook (`EngramResidents.build_module(row_cache=bank.cache, layer_hash_index=…,
  norm_eps=args.rms_norm_eps)`), attaching it to `model.model.layers[layer_id].engram_hook`;
- builds **one** `NgramHashState` prototype (`from_manifest`), whose **compressed token map is
  built from the artifact's HuggingFace tokenizer** (`tokenizer.json` at the artifact root; the
  parenthetical `encoding/` in the brief is the reference encoding module, not the tokenizer) and
  validated against the manifest's `compressed_vocab_size` = 99092, and stashes it on the
  backbone (`self.model.engram_hash`, mirroring the reference `Transformer.engram_hash`).

Per-sequence state: `make_cache()` clones a fresh streaming `NgramHashState` via the new
`NgramHashState.fresh()` (shares the immutable config arrays — cheap — with an empty history);
a bare `model(ids)` (auto-created cache) also gets its own clone. `norm_eps` = `args.rms_norm_eps`
= 1e-20 and `clamp_value` = 1e-6 match the reference `Engram` exactly. The backbone already
advanced `cache.engram_state` once per step, called `layer.engram_hook(h, input_ids,
engram_state)` before each engram layer, and trimmed `engram_state` on `DeepseekV41Cache.rollback`
in step with the KV rollback; W6 only guards the hook call on `engram_state is not None`.

Reference match (`~/models/DeepSeek-V4.1-Flash-src/inference/{model.py,engram.py}`): the reference
computes all engram hashes once per forward and applies `layer.engram(h, engram_hashes[:, :,
layer_hash_index, :], mask)` before `layer(...)` — identical to the MTPLX backbone.

### Tests (`tests/models/test_deepseek_v41_engram_attach.py`, real-artifact gated)

A reduced backbone (hidden 5120 / hc_mult 4 so the real dim-5120 hooks fit, 15 SWA-only layers,
tiny experts/vocab) hosts the **real** engram hooks (real 104 GiB banks + W4 sidecar):
- `test_attach_sets_hooks_on_exactly_engram_layers`: hooks (EngramV41) on exactly `[1, 14]`,
  `layer_hash_index` correct, prototype built from the tokenizer, `make_cache` clones a fresh
  independent state sharing the config.
- `test_hook_changes_hidden_vs_none`: hooks change the backbone hidden by **6.14 max abs**
  (> 1e-2) vs `engram_hook = None` on a real-token prompt; re-attaching reproduces it exactly.
- `test_rollback_trims_engram_in_step_with_kv`: prefill, mark, decode a divergent tail (engram
  length 20→22), roll back (offset and engram length restored to 20), decode the real tail →
  byte-identical hidden to a clean decode.

---

## Task 3 — real construct test + CPU end-to-end proof

`tests/test_deepseek_v41_loader.py::test_construct_is_wired_and_guarded_until_w1` (asserted a
`ResidentLoadError` before W1) is replaced by **`test_construct_wires_real_model_and_engram`**:
constructs the real model via the serve-path loader and asserts `tensor_count == 1616`,
`raw_tensor_bytes == 8_673_203_648`, `bound_sparse_layers == 40`, all 40 `switch_mlp` are
`HotExpertSwitchGLU`, engram hooks on exactly `[1, 14]` (EngramV41), `_mtplx_engram_layer_ids ==
(1, 14)`, and `make_cache().engram_state is not None`.

### CPU end-to-end forward proof (REAL artifact, FULL 40 layers — feasible)

`scratchpad/dsv41_e2e_proof.py` (also committed as the opt-in
`test_cpu_end_to_end_forward`, gated on `DSV41_RUN_E2E`). Serve path: real
`ensure_expert_admitted` → `open_deepseek_v41_runtime` → resident q8 load →
`bind_streamed_switches` → `attach_engram` → greedy prefill + 2 decode.

| metric | value |
|---|---|
| admission | REAL `ensure_expert_admitted` (trusted bank digest, **no 169 GiB hash**) |
| admission receipt | `<receipt_root>/<sha>.json` (757 B); default root `~/.mtplx/receipts`, here a scratch dir |
| construct | 7.8 s; residents 1,616 / 8,673,203,648 B; 40 bound sparse layers; reader backend `native` |
| engram | hooks on layers [1, 14]; per-cache `NgramHashState` |
| prompt | `"def add(a, b):"` → 6 tokens `[3465, 1258, 6036, 14, 291, 2605]` |
| prefill (6 tok) | 55.5 s wall; **TTFT 55.5 s**; first token id 6058 |
| decode | 11.0 s, 10.95 s (2 steps) → ids 201, 361 |
| routed experts read from `experts.bin` | **1,920** records (= (6+2)×40×6), all misses; 1,374 transient loads; **15,195,340,800 B** read |
| engram rows gathered | layer 1: 192 rows / 3 gathers; layer 14: 192 rows / 3 gathers (6 prefill + 2 decode tokens × 24 rows) |
| produced ids | `[6058, 201, 361]` → `" forward\n   "`; full: `"def add(a, b): forward\n   "` |
| **peak RSS** | **9.49 GB** (residents are mmap'd; well under the 60 GB ceiling) |

The full 40-layer forward ran, so no slice was needed. (A 4-layer slice — layers 0–3 incl.
engram layer 1 — was also run as a warmup: 192 expert reads, engram L1 192 rows, peak RSS
3.98 GB.) The produced tokens are in-range and detokenize to real subword text; this is a raw
base LM under greedy decoding, so `" forward\n   "` is a plausible (not instruction-tuned)
continuation — the proof is of the mechanism (real experts gathered from `experts.bin`, real
engram rows gathered from the banks, a full CPU forward), not of completion quality.

Not exercised on the CPU path: Metal/GPU execution, MTP/DSpark, vision, FP4/FP8 KV quant, and
long-context CSA2 boundaries (6 tokens stay under every compress-ratio boundary). No `~/models`
write except the admission receipt, which the serve-path loader writes by design (into the
`receipt_root`, outside the artifact).

---

## Test results (CPU, `nice -n 19`, no `-n auto`)

All nine deepseek_v41 / engram / ngram test files, run together (canonical order):
```
54 passed, 1 skipped in 103.46s
```
(the 1 skip is the opt-in `test_cpu_end_to_end_forward`; it passes with `DSV41_RUN_E2E=1`
in 152 s.) Order-independent: reversed order (engram/ngram files first) → `51 passed, 1
skipped`; `test_deepseek_v41_parity.py` alone → `4 passed` (it forces the CPU stream itself,
so it no longer depends on another file having flipped the device). `test_csa2_modes` alone
run 5× → stable, 0 mismatches each run.

Attribution: `python3 scripts/check_ai_attribution.py --range origin/main..HEAD` → clean
(no Co-Authored-By / Claude trailer on the W6 commit).

## Files changed (allowlist)
- `mtplx/models/deepseek_v41.py` — backbone `engram_hash` prototype + per-cache clone,
  guarded hook call, `Model.make_cache` engram-state attach, `Model.attach_engram`.
- `mtplx/models/deepseek_v41_loader.py` — call `model.attach_engram` after residents load;
  record `_mtplx_engram_layer_ids`.
- `mtplx/engram_v41.py` — `NgramHashState.fresh()` (cheap per-sequence clone).
- `tests/models/test_deepseek_v41_parity.py` — fp32-accumulate oracle + CPU force.
- `tests/models/test_deepseek_v41_config.py`, `tests/models/test_deepseek_v41_loader_contract.py`,
  `tests/test_deepseek_v41_loader.py`, `tests/test_deepseek_v41_spec.py` — CPU force at import.
- `tests/models/test_deepseek_v41_engram_attach.py` — new (Task 2 tests).
- `tests/test_deepseek_v41_loader.py` — real-construct test + opt-in `test_cpu_end_to_end_forward`.
- `docs/deepseek-v41/W6_REPORT.md` — this report.
