# W7 — P1.7 gate crash: `'bytearray' object has no attribute 'bank'`

Branch `feat/deepseek-v41-w7` off `feat/deepseek-v41-streaming` @ `04d18051`
(the W6 integration point). CPU only, `nice -n 19`, MLX default device forced to
the CPU stream, no GPU/Metal, no lock, no launchctl, no `:8080`, no `~/models`
write except the append-only admission receipt the serve-path loader writes by
design. mlx 0.32.2, py3.12.

The first real GPU run of `scripts/deepseek_v41/gate_stream_equals_resident.py`
(inside `gpu_window.sh`, `--pinned-layers 20`, real artifact) died in **Run A**
(streamed, `island_layers=()`) after ~70 s with peak RSS 60.6 GiB, and the gate
swallowed the traceback behind its one-line summary:

```
gate_stream_equals_resident: could not complete a run
(AttributeError: 'bytearray' object has no attribute 'bank')
```

---

## 1. Traceback captured (CPU reproduction)

Reproduced on the CPU stream with the same gate invocation (`--pinned-layers 20`,
real artifact, but `--steps 1`, a short prompt, `--cpu`, and
`--no-apply-memory-cap`). It crashes at the **first routed layer of the Run A
prefill** — no full 40-layer forward needed — so the repro is fast and its peak
RSS is bounded by the mmap'd residents:

```
File ".../scripts/deepseek_v41/gate_stream_equals_resident.py", line 230, in _greedy_decode
  logits = forward(mx.array([context]))
File ".../mtplx/models/deepseek_v41.py", line 922, in __call__       # Model.__call__
File ".../mtplx/models/deepseek_v41.py", line 847, in __call__       # backbone: layer(...)
File ".../mtplx/models/deepseek_v41.py", line 733, in __call__       # DecoderLayer: x = self.mlp(x)
File ".../mtplx/models/deepseek_v41.py", line 660, in __call__       # MoE: routed = self.switch_mlp(xf, indices)
File ".../mtplx/models/expert_mlx.py", line 1763, in __call__        # HotExpertSwitchGLU.__call__ -> _run
File ".../mtplx/models/expert_mlx.py", line 2446, in _run            # evaluate_bindings(miss_positions, ...)
File ".../mtplx/models/expert_mlx.py", line 1950, in evaluate_component_bindings
  by_bank.setdefault(id(binding.buffer.bank), []).append(
AttributeError: 'bytearray' object has no attribute 'bank'
```

The gate now **always prints the full traceback to stderr before** the summary
line (`main`'s `except` handler: `traceback.print_exc(file=sys.stderr)`), so a
raw dispatch error is never again mistaken for a "missing-W1" `ResidentLoadError`.

## 2. Root cause (file:line)

The component-banks streamed dispatch requires each routed binding's `buffer` to
be an `MlxComponentSlot` — a gather-qmm-ready slot that carries `.bank`
(`mtplx/models/expert_mlx.py:359,364` `self.bank.component_view(...)`); the
grouping at `mtplx/models/expert_mlx.py:1950`
(`by_bank.setdefault(id(binding.buffer.bank), [])`, selected as
`evaluate_bindings` only when `config.slot_layout == "component-banks"`,
`expert_mlx.py:2277`) reads that `.bank`.

Those slots are produced by `make_mlx_component_bank_allocator`
(`mtplx/models/expert_mlx.py:1014`), which a caller must hand to
`ExpertStreamingRuntime.open(..., buffer_allocator=...)`. When no allocator is
supplied the slot pool falls back to a **raw `bytearray`**:

```
mtplx/expert_slots.py:760
self._allocator = buffer_allocator or (lambda size, _label: bytearray(size))
```

`bytearray` has no `.bank` → the `AttributeError`.

**The defect:** `open_deepseek_v41_runtime`
(`mtplx/models/deepseek_v41_loader.py`) opened the runtime **without a
`buffer_allocator`**. The production serve path builds the component-bank
allocator for exactly this layout in `mtplx/runtime.py:1042`
(`slot_allocator = make_mlx_component_bank_allocator(...)`), but
`open_deepseek_v41_runtime` is a **second** runtime-open entry — used by the
P1.7 gate and the CPU end-to-end proof, not by production serving — and it never
wired the same allocator. So under `slot_layout="component-banks"` its slots were
bytearrays.

**Why the W6 CPU end-to-end proof passed through the same loader.** It used the
**default** `slot_layout="direct-slots"`, whose dispatch (`evaluate_direct_bindings`
→ `_run_q4_expert`, and `_component_array` at `mtplx/models/expert_mlx.py:160`)
reads `binding.buffer` as **raw record bytes** — a `bytearray` is exactly what
that path expects, so it never touches `.bank`. Only the component-banks path,
which the gate forces (islands require it), needs the `MlxComponentSlot`. The
difference was the slot layout, not the loader or the model.

## 3. Fix (minimal, in the allowlist)

`mtplx/models/deepseek_v41_loader.py` — new helper `_component_bank_allocator_for(...)`
called by `open_deepseek_v41_runtime` before `ExpertStreamingRuntime.open`
(`deepseek_v41_loader.py:254`). When `config.slot_layout == "component-banks"` it
builds `make_mlx_component_bank_allocator(plan, spec, manifest)` and passes it as
`buffer_allocator`; for any other layout it returns `None` and the bytearray
fallback stands (the direct-slot dispatch wants it). This mirrors the production
`mtplx/runtime.py` branch — the generic streaming machinery, applied to the
loader's own open entry.

The plan handed to the allocator is built **the same way
`ExpertStreamingRuntime.open` builds its own** — island placement resolved
(`resolve_island_placement`), the SWA window priced as `additional_resident_bytes`
(`SWA_WINDOW_BYTES`), the resident-quant discounts applied
(`proj_quant_plan_discount + proj_requant_plan_discount`), mixed-official
per-layer record sizes when applicable — so the allocator's per-bank capacities
match the slot pool's plan exactly. (For this affine-Q2 spec the discounts are 0
and `is_mixed_official` is False, but the helper computes them so any caller that
tunes `proj_quant`/`proj_requant` stays correct; a capacity mismatch would fail
loudly at open, never silently.) `expert_runtime.py`/`expert_mlx.py` were **not**
modified — the defect was the missing wiring in this loader entry, not the
component-bank machinery.

Gate script (`scripts/deepseek_v41/gate_stream_equals_resident.py`), to make it
CPU-reproducible without weakening the GPU proof:
- full traceback to stderr before the summary (§1);
- `--cpu` — force `mx.set_default_device(mx.cpu)` before any array is built
  (default off; the real proof runs on the GPU);
- `--apply-memory-cap` / `--no-apply-memory-cap` (default **on**) — a CPU-only
  reproduction must not touch the MLX/Metal memory cap;
- `--steps` (pre-existing) reduces the decode step count for the repro.

## 4. Regression tests (`tests/test_deepseek_v41_loader.py`)

- `test_component_banks_wires_bank_allocator_not_bytearray` — fast, **no bank
  read**: asserts `_component_bank_allocator_for` returns a real component-bank
  allocator (the closure carries `.banks`/`.close`) for `slot_layout=
  "component-banks"`, and `None` for the default `direct-slots`.
- `test_component_banks_prefill_dispatch_reads_bank_slots` — behavioral, gated on
  the bank being present: opens the runtime with `slot_layout="component-banks"`,
  binds the streamed switches on the loader test double, and drives a **single
  PREFILL dispatch** (T=2 → `RoutingPhase.PREFILL`, the phase Run A crashed in)
  through one routed layer. It asserts the pool's allocator carries `.banks`,
  the dispatch returns finite logits, and a real `MlxComponentSlot` backs the
  layer's bank. This exercises the exact `binding.buffer.bank` path for the hy3
  streamed switch (`HotExpertSwitchGLU`) shared by all component-banks models.

**Counterfactual (test validity).** Monkeypatching `_component_bank_allocator_for`
to return `None` (the pre-fix behavior) and re-running the same one-layer
dispatch reproduces `AttributeError: 'bytearray' object has no attribute 'bank'`
— so the behavioral test fails without the fix and passes with it.

## 5. CPU gate run to completion (both runs)

`--cpu --no-apply-memory-cap --steps 4 --prompt "def add(a, b):" --pinned-layers 20`,
real artifact, receipt in the session scratchpad:

| field | value |
|---|---|
| verdict | **PASS** (`match=True`) |
| slot_layout | `component-banks` |
| Run A (streamed, `island_layers=()`) argmax | `[6058, 201, 361, 1354]` |
| Run B (resident, `island_layers=[20]`) argmax | `[6058, 201, 361, 1354]` |
| switches wrapped / gathered records (each run) | 40 / 1614 |
| prompt tokens / steps / used_kv_cache | 6 / 4 / True |

Both runs completed, the receipt was written, and the two argmax sequences are
byte-identical — streamed serving of layer 20 equals pinning it resident. (The
first tokens `6058, 201, 361` match the W6 CPU end-to-end proof, cross-checking
the forward.) `--steps 1` reproduces the original crash on Run A; `--steps 4`
exercises prefill + several decode steps of both runs.

**Measured peak RSS.** A second run under `/usr/bin/time -l`
(`--cpu --no-apply-memory-cap --steps 2 --pinned-layers 20`, both runs, verdict
PASS) reports **maximum resident set size = 44,399,542,272 B = 41.35 GiB** — well
under the 60 GB ceiling, and lower than the failing GPU run's 60.6 GiB. (`time`'s
"peak memory footprint" reads 87.5 GB, but that counts the reclaimable
file-backed pages of the mmap'd 158 GiB expert bank touched during the reads, not
wired memory; the RSS is the metric that matches the task's ceiling.) The
component-banks cache is what makes this heavier than W6's ~9.5 GB direct-slot
forward: with `expert_cache_limit_bytes=None` the runtime plans a large per-layer
component-bank cache and Run B additionally pins layer 20 resident.

## 6. Test results (CPU, `nice -n 19`, no `-n auto`)

All 16 deepseek_v41 / engram / ngram test files run together:

```
172 passed, 1 skipped, 1 deselected in 113.80s
```

- the 1 skip is the opt-in `test_cpu_end_to_end_forward` (`DSV41_RUN_E2E`);
- the 1 deselected is a **pre-existing, unrelated** failure, see below.

The two new regression tests pass; the previously-crashing component-banks path
is now covered.

### Pre-existing unrelated failure (not this fix, not in the allowlist)

`tests/test_convert_deepseek_v41_streamed.py::test_mxfp4_is_not_bit_exact_in_mlx_032`
fails on this box. It is **byte-identical to the base branch**
(`git diff feat/deepseek-v41-streaming` over that test and
`mtplx/deepseek_v41_convert.py` / `expert_manifest.py` / `expert_streaming_models.py`
= 0 lines), imports none of the modules changed here, and fails standalone. Root
cause: the test pins mlx 0.32.0 behavior (its name says so) and asserts
`mx.quantize(..., mode="mxfp4")` is **not** a bit-exact repack; on this box's
**mlx 0.32.2** mxfp4 dequant of a tensor drawn from the exact E2M1×E8M0 grid now
round-trips bit-exactly, so `assert not np.array_equal(deq, ref)` is False. It is
a convert-time documentation assertion about an MLX version, outside this task's
allowlist and unrelated to the streaming-loader fix; flagging it for the
orchestrator rather than touching it.

## 7. Files changed (allowlist)

- `mtplx/models/deepseek_v41_loader.py` — `_component_bank_allocator_for` helper;
  `open_deepseek_v41_runtime` wires the component-bank allocator for the
  `component-banks` layout (the fix).
- `scripts/deepseek_v41/gate_stream_equals_resident.py` — full traceback before
  the summary; `--cpu` and `--apply-memory-cap`/`--no-apply-memory-cap` flags.
- `tests/test_deepseek_v41_loader.py` — two regression tests (§4).
- `docs/deepseek-v41/W7_REPORT.md` — this report.

Attribution: `python3 scripts/check_ai_attribution.py --range origin/main..HEAD`
→ clean (no Co-Authored-By / Claude trailer on the W7 commit).
