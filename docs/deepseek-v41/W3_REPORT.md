# W3 report — DeepSeek-V4.1-Flash SSD-streamed MoE serve wiring

Scope: everything between `mtplx serve --model ~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2`
and a constructed model whose routed experts stream from the Q2 bank, so that when the model module
(`mtplx/models/deepseek_v41.py`, worker W1) lands, integration is a one-line import. Template: the
Hy3 lane. All numbers below are verified against the real artifact at
`~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2` (50 shards: 49 `model-000NN.safetensors` q8
residents + `experts.bin`; the task prompt's "33 resident q8 shards" is stale — the artifact ships 49).

Files changed (allowlist):
- `mtplx/expert_streaming_models.py` — the `DEEPSEEK_V41_FLASH_EXPERT_Q2` entry only.
- `mtplx/backends/registry.py` — one `deepseek-v41` architecture entry.
- `mtplx/models/deepseek_v41_loader.py` — new; the loader body.
- `mtplx/resident_loader.py` — the two-line serve-path dispatch (get_streaming_model_classes +
  construct_resident_model delegation) that routes `model_type == "deepseek_v41"` into the loader.
- `tests/test_deepseek_v41_spec.py`, `tests/test_deepseek_v41_loader.py`,
  `tests/deepseek_v41_test_double.py`, `docs/deepseek-v41/W3_REPORT.md` — new.

Loader-location convention (which place hy3's loader lives): the loader body is a **new file**,
`mtplx/models/deepseek_v41_loader.py`. `mtplx/resident_loader.py` (the equivalent place hy3's
loader lives) carries only the two dispatch points, so the production serve path
(`runtime.py` → `construct_resident_model`) reaches the loader with **no `runtime.py` edit**, exactly
like the hy3 lane. The deepseek-specific logic (text-only filter, engram ctor arg) stays out of the
generic `construct_resident_model`.

---

## 1. Pinned spec numbers and their derivation

Changed fields in `DEEPSEEK_V41_FLASH_EXPERT_Q2` (others left as the converter set them; `source_revision`
stays the DeepSeek source-model commit `dba1be0a…`, `router_bytes` stays 157,409,280):

| Field | Was | Now | Derivation |
|---|---|---|---|
| `total_tensor_bytes` | 185,869,312,000 (provisional) | **195,033,235,352** | Measured header-inventory sum: Σ over all 49 `model-000NN.safetensors` headers = **25,163,923,352** resident bytes (5,726 tensors) + routed bank **169,869,312,000** = 195,033,235,352. Equals `conversion-manifest.json.artifact_tensor_bytes` and `expert-manifest.json.artifact.tensor_bytes` exactly — **no delta**. |
| `quant_model` | `local/deepseek-v41-flash-mtplx-streaming-q2` | **`OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2`** | Public HF repo (HF API `private=false`). |
| `quant_revision` | `dba1be0a…` | **`b64980a16283647bb213ab475335f38f516e0d9e`** | Current `main` commit sha from the HF API model info (read-only, 2026-09-10). |
| `kv_bytes_per_token` | 81,920 (placeholder) | **3,200** | PORT_PLAN §3b phase-1 bf16 global CSA2 KV. Per position a Full/kv_source layer stores a compressed-KV latent (512·2 B) + an index-K record (128·2 B) = 1280 B, scaled by 1/compress_ratio; summed over kv_source layers [L2,L8,L14 @ ratio 2; L20 @ ratio 1] = 3·(1280//2) + 1280 = **3200 B/token**. The SWA sliding window is a **fixed** 40·128·(512·2) = **5,242,880 B (5 MiB)**, NOT per token — the loader prices it as an additional-resident reserve, documented in the spec comment. |
| `full_indexer_layers` | `()` | **`(2, 8, 14, 20, 24, 28, 32, 36)`** | `text_config.index_source_layer_ids`, verified from the artifact config.json. Runtime consumption: grep finds **no reader** of `spec.full_indexer_layers` outside `expert_streaming_models.py` — it is a pinned geometry descriptor, structurally validated (sorted/unique/in-range) but not yet consumed by the runtime for behaviour (same status as the GLM-5.2 entry, which likewise pins its index-owning layers here). The KV surcharge accounting the field exists for is not wired. |

Strict spec validation (`validate_expert_manifest_spec(manifest, spec, require_pinned_tensor_bytes=True)`)
**passes** against the real manifest once the spec's source identity is rebased to the manifest's
(see §2). Under that call it checks, and all pass: bits (2), group size (64), affine mode, model_key,
the 40×384 record Cartesian product, each record's 11,059,200 logical bytes, each of the 9 component
segments' dtype/shape/length, routed bytes 169,869,312,000, `artifact_tensor_bytes` == 195,033,235,352,
and `resident_tensor_bytes` == 25,163,923,352.

### Source-identity divergence (publish blocker — surfaced, not worked around)

The shipped `expert-manifest.json` — the local `~/models` copy **and** the HF-uploaded copy (verified by
ranged fetch of the published file) — still carries the **pre-publish** identity
`source_repo="local/deepseek-v41-flash-mtplx-streaming-q2"`, `source_revision="dba1be0a…"`.
`validate_expert_manifest_spec` compares `manifest.source_repo`/`source_revision` against the spec's
`quant_model`/`quant_revision`. With the spec pinned to the HF repo (as instructed), strict validation
and the serve-path admission (`ensure_expert_admitted(root)` → `get_model_spec(manifest.model_key)`)
**fail on source identity only** — every byte, geometry, and record check passes (proven by
`test_hf_pinned_spec_fails_only_on_source_identity`, whose error is exactly
`"manifest source identity does not match the pinned descriptor"`).

Root-cause fix: rebuild the manifest with the HF identity and re-upload. The **local** rebuild is done
(next subsection). The **re-upload** is David's call.

### Manifest identity fix (applied to the local artifact)

The local `~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2/expert-manifest.json` was rebased to the HF
identity so it matches the pinned spec.

- **Path taken — digest reuse, no bank re-hash.** `build_expert_manifest(...)` was **not** the mechanism:
  it re-inventories the safetensors shards and (with `hash_records=True`) re-reads all 15,360 records
  (158 GiB), and moreover the routed experts live in the `experts.bin` **sidecar**, not the safetensors,
  so a from-scratch build cannot reconstruct them. Instead the existing manifest was loaded,
  `dataclasses.replace(manifest, source_repo="OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2",
  source_revision="b64980a16283647bb213ab475335f38f516e0d9e").with_digest()` recomputed only the
  content digest (a ~49 MB canonical-JSON hash, sub-second), and the file was edited surgically
  (three field values: `source_repo`, `source_revision`, `manifest_sha256`) with an atomic write. Every
  recorded record digest, resident tensor, and shard entry is reused byte-for-byte. **No 158 GiB
  re-hash ran** (the GPU flock was held by another process — a bank re-hash would have corrupted its
  host-encode window; digest reuse is also the coordinator's stated preference).
- **Backup**: the pre-fix manifest is at
  `…/scratchpad/expert-manifest.json.pre-hf-identity` (49,228,771 B).
- **`manifest_sha256`**: `608f05f4…e32eb` → `607e1721…005b`. `experts.bin`, the 49 resident shards, and
  `engram/` were untouched.
- **Diff (old vs new)** shows only the three identity fields changed — records/resident_tensors/shards
  byte-identical:
  ```
  4,5c4,5
  <   "source_repo": "local/deepseek-v41-flash-mtplx-streaming-q2",
  <   "source_revision": "dba1be0a40aa45a94ad051997016db3960a90277",
  ---
  >   "source_repo": "OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2",
  >   "source_revision": "b64980a16283647bb213ab475335f38f516e0d9e",
  1875228c1875228
  <   "manifest_sha256": "608f05f4…e32eb"
  ---
  >   "manifest_sha256": "607e1721…005b"
  ```
- **Proof against the REAL local artifact with the PINNED (HF) spec — no source-rebase shim:**
  - `validate_expert_manifest_spec(manifest, get_model_spec("deepseek-v41-flash-expert-q2"),
    require_pinned_tensor_bytes=True)` → **PASS** (`manifest.source_repo == spec.quant_model`).
  - serve-path `ensure_expert_admitted(root)` → **PASS in 1.99 s** via receipt reuse (the receipt was
    seeded from the manifest's recorded `experts.bin` sha256 as a trusted digest, so no bank hash;
    `load_valid_admission_receipt` re-validates the HF spec and the bank stat identity, never hashing).
  - Committed as state-aware tests: `test_pinned_spec_strict_validation` and
    `test_serve_path_admission_with_pinned_spec` (both take the fixed-identity branch here).

> **The HF-uploaded copy of `expert-manifest.json` still carries the old `local/…` identity.** Only the
> local artifact was fixed. A fresh `mtplx serve --model OpensourceWTF/… --download` will still fail
> admission on source identity until the corrected manifest is re-uploaded to the HF repo — **David's
> call** (W3 does not push anything, HF included).

---

## 2. Text-only resident byte total (skip vision + MTP)

The manifest's 25,163,923,352 resident bytes (5,726 tensors) include the q8 MTP dense + q8 MTP experts
and the vision/aligner/image residents that the converter kept in the artifact. Phase-1 text-only AR
skips them at load. Filter: drop any resident whose name starts with `mtp.` / `vision.` / `aligner.` /
`image_`.

| Class | Tensors | Bytes | GiB |
|---|---:|---:|---:|
| **Text kept** (`embed.*`, `head.*`, `layers.*`, `norm.*`) | **1,616** | **8,673,203,648** | 8.078 |
| Skipped `mtp.*` | 3,584 | 15,974,347,224 | 14.878 |
| Skipped vision/aligner/image | 526 | 516,372,480 | 0.481 |
| **Total residents** | 5,726 | 25,163,923,352 | 23.436 |

Skipped total = 4,110 tensors / 16,490,719,704 bytes. Kept + skipped reconcile to the manifest's
`resident_tensor_bytes` exactly.

### Reconciliation with PORT_PLAN §3a (its 8.42 GiB is not stale relative to text-only)

§3a estimated the **text-only dense backbone** at 8.42 GiB. The **measured text-only load is 8.67 GB =
8.078 GiB** — within ~0.34 GiB of the §3a estimate, confirming §3a's scope. §3a is "stale" only relative
to the manifest's **25.16 GB**, which is larger because it additionally carries the q8 MTP (14.88 GiB) and
vision/aligner (0.48 GiB) residents that §3a (and text-only AR) exclude. There is no contradiction: §3a
counted the text backbone; the manifest counts every resident the converter emitted.

---

## 3. Memory-plan table (100 GiB knob)

`plan_expert_memory` with the text-only resident (via `resident_discount_bytes = 16,490,719,704`), the
fixed SWA window as `additional_resident_bytes = 5,242,880`, and the promoted-profile 7 GiB runtime
reserve. Record = 11,059,200 B; one uniform slot spans all 40 streamed layers (no islands).

| Context | KV bytes | Slots/layer | Persistent slots | Expert cache (GiB) | Fits | Unallocated (GiB) |
|---:|---:|---:|---:|---:|:--:|---:|
| 4,096 | 13,107,200 | **205** | 8,200 | 84.457 | yes | 0.386 |
| 16,384 | 52,428,800 | **205** | 8,200 | 84.457 | yes | 0.350 |
| 65,536 | 209,715,200 | **205** | 8,200 | 84.457 | yes | 0.203 |

Resident + reserve fit assertion (worst case, 64K): text-only resident 8,678,446,528 (8.67 GB backbone
+ 5 MiB window) + 7 GiB reserve + 209,715,200 KV + 66,355,200 transient = 16,470,709,696 ≤ 107,374,182,400
(100 GiB). ~205 of 384 experts per layer stay resident-cached; the rest page from the bank — streaming is
mandatory (a full 158 GiB island bank does not fit).

Note on the runtime's internal plan: `ExpertStreamingRuntime.open` builds its plan from the **full**
`spec.resident_bytes` (25.16 GB) with only a `proj_quant` discount, so as wired today it reserves for all
residents and buys **168 slots/layer** at 4K (still fits — safe, conservative). Realizing the text-only
+37 slots/layer inside the runtime needs a text-only resident-discount hook on `open`/`ExpertStreamingConfig`
(would touch `expert_runtime.py`, outside this task's allowlist) — documented follow-up. The 205-slot figure
is the text-only envelope from the standalone `plan_expert_memory` profile.

---

## 4. Interface the loader expects from `mtplx.models.deepseek_v41` (worker W1)

The loader resolves `(Model, ModelArgs)` via the single guarded import in
`deepseek_v41_loader.deepseek_v41_model_classes()`:
`from mtplx.models.deepseek_v41 import Model, ModelArgs`. Required surface:

- **`class ModelArgs`** with `@classmethod from_dict(config: dict) -> ModelArgs`. `config` is the merged
  checkpoint config: top-level `model_type == "deepseek_v41"`, text sub-config
  `text_config.model_type == "deepseek_v41_text"` (num_hidden_layers 40, hidden 5120,
  moe_intermediate 2304, n_routed_experts 384, num_experts_per_tok 6, sliding_window 128,
  kv_source_layer_ids [2,8,14,20], index_source_layer_ids [2,8,14,20,24,28,32,36],
  candidate_source_layer_id 20, compress_ratios 43 entries).
- **`class Model(nn.Module)`** with constructor
  **`Model(model_args: ModelArgs, *, engram_bank_path: str | os.PathLike | None = None)`**. The loader
  passes `engram_bank_path = <artifact>/engram` (or `None` if absent). W1 accepts/stashes it and hands it
  to the engram runtime (worker W2 owns `mtplx/engram*`); W3 does **not** implement engram.
- **Switch seam** (the attribute path `bind_streamed_switches` walks): `model.model.layers[i].mlp.switch_mlp`
  (it tries `model.model.layers` first, then `model.layers`). For every routed layer `i` in `0..39`,
  `layers[i].mlp` must exist and expose a reassignable `switch_mlp` attribute; pre-binding it should be
  `mtplx.models.expert_mlx.UnboundExpertSwitch(i)` (the hy3 placeholder). `bind_streamed_switches`
  overwrites it with `HotExpertSwitchGLU(runtime, i)` and returns the bound count, which must equal
  `routed_layer_count == 40`. **Naming caveat:** DeepSeek's *tensor* namespace is `layers.N.ffn.*`, but the
  MTPLX seam is `.mlp.switch_mlp` (hy3 convention) — W1's `Model` exposes the routed FFN block as
  `layer.mlp` and its `sanitize()` remaps the `layers.N.ffn.*` resident keys to the `.mlp`-based module
  paths (as hy3's `SparseMLP` does).
- **Standard `nn.Module` load surface**: optional `sanitize(weights: dict) -> dict`, plus `eval()`,
  `load_weights(list(items), strict=True)`, `parameters()`. The loader strict-loads exactly the **1,616
  text-only resident keys / 8,673,203,648 bytes** (embed/head/layers/norm) — W1's text-only `Model` must
  declare exactly these parameter paths after `sanitize` and must **not** declare vision/aligner/mtp params
  (those residents are skipped at load).

Once `mtplx/models/deepseek_v41.py` provides the above, `mtplx serve --model <local dir>` reaches a
constructed, switch-bound model through the wiring in this PR with no further code change (module,
resident_loader dispatch, spec, registry, admission are all in place). Full CLI `/health` additionally
needs the model to actually materialize its 8.67 GB residents (Metal) and a running server; the serve
path forces `generation_mode: "ar"` for a streamed artifact (public.py:9681, server/openai.py:3123) and
reports the model key — both facts the loader/runtime already determine (`runtime.spec.key ==
"deepseek-v41-flash-expert-q2"`, `mtp_included == False`).

---

## 5. What was verified (CPU only, no Metal, no full-bank load)

- `ExpertStreamingRuntime.open` on the real artifact (with a matching admission receipt, `apply_memory_cap=False`,
  `expert_cache_limit_bytes=0` for tiny host allocation) constructs a runtime whose plan fits, reader backend
  is `native`, and `bind_streamed_switches` binds all **40** routed layers to `HotExpertSwitchGLU`. Bank I/O is
  lazy — `open` reads the manifest JSON and size-checks `experts.bin` only.
- Admission on the real artifact: model_key `deepseek-v41-flash-expert-q2`; sampled records are 11,059,200 B;
  sample record SHA-256 (read from `experts.bin` via positional reads of a handful of ~10.5 MiB records)
  match the manifest record hashes; a real `admit_expert_artifact` (trusted bank digest → no 169 GiB hash)
  writes a revision/digest-bound receipt into a scratch receipt root.
- Text-only filter exact counts/bytes; memory-plan slot counts; loader end-to-end with the test double;
  `/health`-relevant facts.

### pytest tail

```
tests/test_deepseek_v41_spec.py::test_spec_bytes_are_pinned_to_measured_inventory PASSED
tests/test_deepseek_v41_spec.py::test_spec_kv_and_indexer_pins PASSED
tests/test_deepseek_v41_spec.py::test_quant_model_pinned_to_public_hf_repo PASSED
tests/test_deepseek_v41_spec.py::test_manifest_bytes_match_pinned_spec PASSED
tests/test_deepseek_v41_spec.py::test_pinned_spec_strict_validation PASSED
tests/test_deepseek_v41_loader.py::test_text_only_filter_counts_and_bytes PASSED
tests/test_deepseek_v41_loader.py::test_text_only_filter_predicate_never_keeps_vision_or_mtp PASSED
tests/test_deepseek_v41_loader.py::test_admission_model_key_and_record_bytes PASSED
tests/test_deepseek_v41_loader.py::test_admission_sample_record_digests PASSED
tests/test_deepseek_v41_loader.py::test_real_admission_writes_receipt PASSED
tests/test_deepseek_v41_loader.py::test_serve_path_admission_with_pinned_spec PASSED
tests/test_deepseek_v41_loader.py::test_memory_plan_text_only_slots[4096] PASSED
tests/test_deepseek_v41_loader.py::test_memory_plan_text_only_slots[16384] PASSED
tests/test_deepseek_v41_loader.py::test_memory_plan_text_only_slots[65536] PASSED
tests/test_deepseek_v41_loader.py::test_memory_plan_full_resident_is_conservative PASSED
tests/test_deepseek_v41_loader.py::test_open_runtime_and_bind_end_to_end PASSED
tests/test_deepseek_v41_loader.py::test_construct_is_wired_and_guarded_until_w1 PASSED
tests/test_deepseek_v41_loader.py::test_health_relevant_facts PASSED
======================= 18 passed, 2 warnings in 13.48s ========================
```

State-aware: `test_pinned_spec_strict_validation` and `test_serve_path_admission_with_pinned_spec` take
the fixed-identity branch here (local manifest rebased); against a fresh HF `--download` (still `local/…`
identity) the first falls back to the rebased-spec assertions and the second skips.

Regression check: `tests/test_streamed_models.py tests/test_expert_streaming_models.py
tests/test_expert_manifest.py` → 139 passed, 1 skipped (no regression to the hy3/glm lanes from the spec /
registry / resident_loader edits).

Pre-existing failure, out of scope: `tests/test_convert_deepseek_v41_streamed.py::test_mxfp4_is_not_bit_exact_in_mlx_032`
fails identically with W3's changes stashed — MLX 0.32.0's `mx.quantize(mode="mxfp4")` is bit-exact on this
box, contradicting that converter-worker test's assumption. It exercises `mx.quantize`, not anything W3
touched, and the file is outside the W3 allowlist.
