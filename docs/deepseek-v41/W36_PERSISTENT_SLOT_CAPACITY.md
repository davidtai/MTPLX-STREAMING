# W36 — `persistent slot is outside planned capacity` (window-11 ab_decode_levers)

## Symptom

In GPU window 11, `scripts/deepseek_v41/ab_decode_levers.py --context-tokens 1024
--decode-tokens 256 --arms control fanout4 overlap overlap_fanout4
--memory-limit-gib 82 ...` failed on **every** arm — including `control` (no
overrides) — inside the load, with:

```
ValueError('persistent slot is outside planned capacity')
```

Receipt: `docs/deepseek-v41/receipts/gpu-windows/window-11/ab-1024-io-levers.json`.

`bench_standard_shape.py` had loaded and run the same artifact at
`--memory-limit-gib 72` and `92` in earlier windows, so the failure looked
config-specific to the W24 script. It is not.

## Root cause (a real loader bug, not the script, not the generic planner)

The error is raised in the component-bank slot allocator
(`mtplx/models/expert_mlx.py`, `make_mlx_component_bank_allocator` →
`allocate`) when it is asked for a `layer-<L>-persistent-<idx>` slot whose
`idx >= plan.slots_per_layer` of **the allocator's** memory plan.

Two independent memory plans are built on the open path, and they must agree:

1. **Pool plan** — `ExpertStreamingRuntime.open` (`mtplx/expert_runtime.py`
   ~line 2082) builds it and enumerates the persistent slots the pool will hold:
   `for slot_index in range(pool_plan.slots_per_layer)`
   (`mtplx/expert_slots.py` ~line 793), calling the allocator for each.
2. **Allocator plan** — `open_deepseek_v41_runtime` wires a component-bank
   allocator through `_component_bank_allocator_for`
   (`mtplx/models/deepseek_v41_loader.py`), and that allocator sizes each
   per-layer bank's capacity from **its own** `memory_plan(...)` call.

Both call `config.memory_plan(spec, additional_resident_bytes=SWA_WINDOW_BYTES,
resident_discount_bytes=…, layer_record_bytes=…)`. The only difference was the
`resident_discount_bytes` term:

| plan | `resident_discount_bytes` |
|---|---|
| pool (`open`) | `proj_quant_plan_discount` + `proj_requant_plan_discount` + **`text_only_resident_discount`** |
| allocator (loader) | `proj_quant_plan_discount` + `proj_requant_plan_discount` |

`text_only_resident_discount` (added in **W21**, commit `effe73663`) discounts
the residents a phase-1 text-only AR forward never wires — the artifact's
`mtp.*` (mxfp8 dense + mxfp4 experts) and `vision.*`/`aligner.*`/`image_*`
residents. W21 added the term to `open()`'s pool plan **and** to `runtime.py`'s
production pre-flight allocator, but **not** to the loader's
`_component_bank_allocator_for` — the *second* runtime-open entry (the P1.7
gate + CPU end-to-end proof, used by the W24 decode-lever, W28 env-lever, and
`bench_standard_shape.py` scripts). W21's own commit message claims "both the
pool plan (open) and the allocator plan (pre-flight) apply the same discount" —
true for the `runtime.py` allocator, false for this one. The loader's docstring
("the plan handed to the allocator is built the same way
`ExpertStreamingRuntime.open` builds its own … the resident-quant discounts
applied") went stale at W21.

With the discount on the pool side only, the pool has **more** free bytes for
persistent slots than the allocator does, so `pool.slots_per_layer >
allocator.slots_per_layer`. The pool enumerates persistent slot index
`allocator.slots_per_layer`, the allocator's bank cannot hold it, and the load
dies.

## Arithmetic (real manifest, `deepseek-v41-flash-expert-mxfp4`)

- `expert_count = 384`, `routed_layer_count = 40` (all streamed, no islands),
  `top_k = 6`, `expert_record_bytes = 18,800,640` (17.93 MiB).
- `spec.resident_bytes = 18,649,658,184` (17.369 GiB); `SWA_WINDOW_BYTES ≈ 5 MiB`;
  runtime reserve 7 GiB; `max_live_kv_tokens = 4096`.
- `text_only_resident_discount = 8,920,505,736` bytes (**8.308 GiB**) of skipped
  MTP + vision residents. `proj_quant`/`proj_requant` discounts are 0 for this
  artifact (mxfp4 records; no BF16/U32 `*_proj` residents to shrink).
- One uniform persistent slot costs `streamed_bytes_sum = 40 × 18,800,640 ≈
  0.700 GiB`, so the 8.308 GiB discount is worth `⌊8.308 / 0.700⌋ ≈ 11–12`
  extra slots/layer.

`slots_per_layer` for each plan, computed from the real manifest:

| `--memory-limit-gib` | allocator (buggy) | pool (`open`) | Δ | outcome (before fix) |
|---:|---:|---:|---:|---|
| 72 | 67 | 79 | +12 | pool enumerates index 67 → **crash** |
| 82 | 82 | 93 | +11 | pool enumerates index 82 → **crash** |
| 92 | 96 | 108 | +12 | pool enumerates index 96 → **crash** |

The `+11` at 82 GiB matches W21's stated "+11 resident expert slots/layer at the
82 GiB envelope". The mismatch exists at **every** envelope, so this is not
specific to 82 GiB or to the W24 script.

## Why `bench_standard_shape.py` did not hit it

Same code path (`open_deepseek_v41_runtime` → `_component_bank_allocator_for`),
same default `--slot-layout component-banks`, so post-W21 bench crashes at 72
and 92 GiB too (Δ = +12 at both, per the table). Its clean 72/92 GiB runs were
in windows **before** the W21 commit (`effe73663`, 2026-09-10 22:41), when
neither plan applied the discount and the two agreed. Nothing about 82 GiB or
the ab script is special — any component-banks load through
`open_deepseek_v41_runtime` since W21 fails identically.

## Fix

`mtplx/models/deepseek_v41_loader.py`, `_component_bank_allocator_for`: add
`text_only_resident_discount(manifest, spec)` to `resident_discount_bytes`, so
the allocator plan's discount equals the one `ExpertStreamingRuntime.open`
applies to its pool plan (mirroring the W21 fix already present in
`runtime.py`'s pre-flight allocator). The same `spec` object is passed to both
`_component_bank_allocator_for` and `ExpertStreamingRuntime.open`, so the two
`text_only_resident_discount(manifest, spec)` calls are byte-identical (and if a
future caller wires MTP with an `mtp_included=True` spec, both sides see it).
The term is **0** for any manifest with no MTP/vision residents (hy3/glm keep
their MTP head in a separate external artifact), so those resolved plans stay
byte-identical — no other lane changes.

A small testability hook was added: `make_mlx_component_bank_allocator` now
attaches its `plan` to the returned allocator (`allocate.plan`), alongside the
existing `backend`/`slots`/`banks`/`close` attributes. No behavior change.

## Regression test

`tests/test_deepseek_v41_component_bank_plan_agreement.py` (CPU-only; reads only
the real `expert-manifest.json` metadata, never `experts.bin`, never allocates a
bank — tiny RSS; skips if the artifact is absent). At 72/82/92 GiB it asserts
the loader's `_component_bank_allocator_for(...).plan.slots_per_layer` (and
`persistent_slots`, `persistent_cache_bytes`, discounted `resident_bytes`)
equals the plan `ExpertStreamingRuntime.open` builds. Verified to **fail** on
the pre-fix loader (`96 != 108` at 92 GiB, `82 != 93` at 82 GiB, `67 != 79` at
72 GiB) and **pass** after the fix. A premise guard asserts the text-only
discount is material (> 6 GiB) for this artifact.

## Rerun command (next GPU window)

The exact window-11 command is now unblocked; run it inside
`scripts/deepseek_v41/gpu_window.sh` (GPU flock held, Qwen unloaded,
memory-guarded), from a clean worktree checkout of this branch:

```
scripts/deepseek_v41/ab_decode_levers.py \
  --context-tokens 1024 --decode-tokens 256 \
  --arms control fanout4 overlap overlap_fanout4 \
  --memory-limit-gib 82 \
  --out docs/deepseek-v41/receipts/gpu-windows/window-11/ab-1024-io-levers.json
```

(append-only receipt; do not overwrite the window-11 failure receipt — write the
rerun to a new window's receipt path per `never-overwrite-a-measurement`.)

## SHA

Fix + test + this doc committed on `feat/deepseek-v41-w36` as `__COMMIT_SHA__`.
