# W42 — DSV4.1-Flash streamed-switch all-hit hot path (KERNEL_LEDGER K23)

Status: root-caused and implemented behind a switch (`MTPLX_DSV41_SWITCH_FASTPATH`,
default off), byte-identity proven on the fake bank **and** a real streamed
runtime; GPU A/B arm (`switch_fastpath`) ready for the orchestrator. Author: Opus
4.8 worker (`feat/deepseek-v41-w42`, from `feat/deepseek-v41-streaming` @
a8f43b95a). CPU-only; MLX pinned to CPU; no GPU lock, no `experts.bin` load.
mlx 0.32.2.

## 0. Headline

W37 window-13 measured decode at **3.73 tok/s (268 ms/token clean)**, with
`moe.routed_switch` the largest stage at **101.6 ms/token = 2.54 ms per layer-call**
(2,560 calls = 40 layers × 64 tokens). Two window-13 facts locate the cost:

1. **Warm-repeat decode == cold decode** (3.80 vs 3.73 tok/s): with every routed
   expert already resident, decode does **not** speed up. So **miss I/O is not the
   per-layer cost** — the exposed cost is present on the all-hit path.
2. The route probe attributes the all-hit cost to **`hot.allhit_fence_eval`:
   1016.9 ms / 1558 all-hit layer-calls = 0.65 ms each ≈ 15.9 ms/token.** That
   bracket wraps `synchronous_fence` → **`mx.eval(wave_output)`**: a blocking
   device→host round-trip run **every all-hit layer**, a *second* per-layer sync
   on top of the K1 `mx.eval(indices)` routing barrier.

**Root cause:** the DSV4.1 decode lane runs with `deferred_pin_release=False` and
`split_route_release="fenced"` — the dataclass defaults
(`expert_runtime.py:180,205`), which DSV4.1's `build_streaming_config`
(`deepseek_v41_loader.py:284`) never overrides. hy3/glm ship the opposite in their
profiles (`mtplx/data/expert_profiles.json`: `deferred_pin_release: true`,
`split_route_release: "deferred"`), so **they defer the slot release to the next
layer's routing barrier** (which materializes the wave output as an ancestor of the
next `mx.eval(indices)`, for free) and dispatch split waves via `async_eval` — one
blocking sync per layer, not two. DSV4.1 eats the extra sync **and** serialises the
decode: the blocking `mx.eval` drains the GPU every layer, so layer N+1 cannot
dispatch over layer N's gather. That is the warm==cold signature.

The all-hit gather itself is **not** the cost: it is already **one `gather_qmm` per
projection** (gate/up/down) over all top-k experts against the persistent component
bank (`_gather_component_bank`, no per-expert Python loop, no slot-view
re-materialisation), reading 6 × **18,800,640 B** (exact record size, manifest
`records[0].logical_bytes`; W24-consistent) = ~113 MB/layer ≈ **~0.19 ms** @ 600
GB/s — two orders below 2.5 ms.

## 1. The fix (behind `MTPLX_DSV41_SWITCH_FASTPATH`, default off)

Promote hy3/glm's already-shipped deferred-release mechanism for the DSV4.1 lane
**without mutating the config** (so other lanes and callers are untouched and the
arm A/Bs cleanly). In `HotExpertSwitchGLU._run` the flag is OR'd with the config
field at every decision site:

```python
_switch_fastpath   = os.environ.get("MTPLX_DSV41_SWITCH_FASTPATH") == "1"
_fastpath_can_defer = _switch_fastpath and <runtime implements defer + flush>
_deferred_pin_active   = config.deferred_pin_release            or _fastpath_can_defer
_split_deferred_active = config.split_route_release=="deferred" or _fastpath_can_defer
```

- **all-hit** (`_run` all-hit branch): `if wave_index == final_wave and
  _deferred_pin_active:` → `defer_slot_release(ready, wave_output)` **instead of**
  `synchronous_fence` (the blocking `mx.eval`) + `ready.release`.
- **split-route** (final wave): `deferred_split = … and _split_deferred_active and
  _deferred_pin_active` → the hit and each miss part dispatch via `async_eval`
  (`defer=True`) and release through one deferred `_DeferredSplitClose` at the next
  barrier, instead of a blocking `synchronous_fence` per wave part.
- the shared-branch miss-window dispatch uses `async_eval` under `_deferred_pin_active`.

With the flag **off** (and config at the fenced default), `_fastpath_can_defer` is
`False` and all three collapse to the **exact shipped expressions** → byte-identical,
zero behavioural change. The flag engages **only** when the runtime implements
`defer_slot_release` / `flush_deferred_slot_releases` (real `ExpertStreamingRuntime`
does; a fake without them falls back to the shipped fence, never crashing). The next
barrier is the covering eval, so **no blocking `mx.eval` is added** — this removes the
lifecycle fence, it does not delete a decision sync (the a3b caution
[[a3b-decode-roundtrip-is-the-lever]] does not bite; that was a bare decision drain,
this is a genuinely redundant pin/release fence). The routing barrier is **preserved**.

`mtplx/expert_runtime.py` needed **no change**: `defer_slot_release`,
`flush_deferred_slot_releases`, and the row-end/close covering flushes
(`reset():3461`, `close():3723`, both `evaluate=True`) already exist and are
model-agnostic. `mtplx/models/deepseek_v41_moe.py` needed no change (the fast-path is
entirely inside the switch it calls).

## 2. Per-call op table — one all-hit DSV4.1 layer at M=1

Host-sync column = **blocking device→host GPU round-trip**. µs are the W37 window-13
route-probe means (`/1558` all-hit calls) except the barrier (W28 p50) and the gather
GPU time (byte estimate).

| op (in call order) | count/layer | host sync | est. µs | fast-path |
|---|---:|:---:|---:|---|
| `mx.eval(indices)` — routing barrier | 1 | **YES** | ~1000 | **kept** |
| `flush_deferred_slot_releases()` (prev layer's pins) | 1 | no¹ | ~5 | kept (now active) |
| `indices.reshape(-1).tolist()` (route id list) | 1 | no² | ~23 | kept |
| `observe_route` / `route_waves` (host) | 1 | no | ~5 | kept |
| `try_all_hit_route` (pin slots, layer lock) | 1 | no | ~35 | kept |
| `broadcast_to`/reshape → assignment_inputs | 1 | no | ~2 | kept (lazy) |
| `_dispatch_component_bank` → 3× `gather_qmm` (gate/up/down, all 6 experts) | 3 qmm / 1 dispatch-build | no | ~9 build / ~190 GPU | kept (overlaps) |
| **`synchronous_fence` → `mx.eval(wave_output)`** | 1 | **YES** | **~650** | **REMOVED → `defer_slot_release`** |
| `ready.release(synchronize=False)` | 1 | no | ~3 | replaced by deferred release |

¹ the previous layer's covering barrier already materialised the deferred output;
the release itself is host bookkeeping. ² indices already materialised by the
barrier, so `.tolist()` is a pure host copy, not a GPU stall.

**Split-route (final wave, 1 hit + N miss), M=1:** identical prologue, then
`begin_split_route` (host/IO submit, no sync), the hit wave, and each of N miss
parts each run `_dispatch_component_bank` + `fence_bindings(force_sync=True)` →
**one blocking `mx.eval` per wave part**. Fast-path: each dispatches via
`async_eval` and releases through one deferred `_DeferredSplitClose`.

## 3. Host syncs per layer — before → after

| route outcome | blocking device→host `mx.eval` **before** (fenced default) | **after** (fast-path) | removed |
|---|---:|---:|---|
| all-hit | **2** — barrier + wave fence | **1** — barrier only | the wave fence |
| split, 1 hit + N miss | **2 + N** — barrier + hit fence + N miss fences | **1** — barrier only | hit + all N miss fences |
| all-miss, N parts | **1 + N** — barrier + N miss fences | **1** — barrier only | all N miss fences |

Counts are **measured** in the CPU census (`tests/test_deepseek_v41_switch_
fastpath.py`, monkeypatch-wrapped `mx.eval`/`mx.async_eval` + the `hot.eval_indices`
bracket): all-hit 2→1 (+1 deferred release, 0 async), split 3→1 (+2 async, +1
deferred), all-miss 2→1 (+1 async, +1 deferred). **Exactly one routing barrier per
layer in every mode**, before and after — the lever never adds or deletes it.

## 4. Expected ms/layer

- Measured today (`moe.routed_switch`, window-13): **2.54 ms/layer average** — but
  this is a stage-timing **fence-inflated** number; the clean-run exposed per-layer
  cost is the host round-trips, dominated by the barrier (~1 ms) + the wave fence
  (**0.65 ms measured**) + ~0.07 ms host bits.
- The fast-path removes the **0.65 ms/layer all-hit wave fence** outright
  (~40 × 0.65 = **~26 ms/token** of exposed sync) and, by not draining the GPU each
  layer, lets layer N+1's dispatch overlap layer N's gather.
- **Expected all-hit switch: ~2.5 → ~1 ms/layer** — the hy3/glm barrier-only floor
  (they pay one sync per layer for the same switch). Split layers additionally shed
  their per-part fences.
- **Net decode delta is the GPU A/B's call** (per the a3b caution, async-submission
  backpressure may conserve part of a removed sync's block). This a-priori estimate
  was **refuted** by GPU window 14: pure defer measured **−13.4 %**, not a gain — see
  §8 for the root cause (the removed fence left the all-hit path submitting no GPU
  work) and variant B. Default off; nothing ships until an arm beats control.

## 5. Byte-identity proof (`tests/test_deepseek_v41_switch_fastpath.py`, green)

- **census** — all-hit / split / all-miss: blocking `mx.eval` before→after per §3,
  exactly one routing barrier each way, one deferred release queued under the flag.
- **`test_switch_output_bitwise_identical_off_vs_on[all_hit/split/all_miss × M=1,4]`**
  — the switch output array is bitwise-equal flag off vs on, for AR (M=1) and the
  **MTP verify row batch (M=4)**; the fake gather is a non-identity, order-sensitive
  function of the inputs, so a stale / dropped / reordered wave would show as a bit
  difference. (Split at M=1 is undefined for a single assignment — skipped.)
- **`test_fastpath_falls_back_to_fence_when_runtime_cannot_defer`** — a runtime
  without `defer_slot_release` ignores the flag (fences, byte-identical, no crash).
- **`test_integrated_streamed_fastpath_matches_fenced_bitwise`** — a **real** streamed
  runtime (`_integrated_hy3_artifact`) whose config is the DSV4.1 fenced default:
  flag-off (fenced) and flag-on (deferred) produce **bitwise-equal logits**, and the
  flag-on run **holds its slot pins until the covering flush** then drains to zero —
  the slot-safety guarantee the fake bank cannot give.
- Underlying deferred mechanism's slot-safety is already locked by
  `test_streamed_models.py::test_deferred_split_route_release_matches_fenced_bitwise`.

Regression: `test_streamed_models.py` (80), `test_deepseek_v41_sync_overlap.py`,
`test_deferred_pin_release.py`, `test_deepseek_v41_ab_env_levers.py`,
`tests/models/test_deepseek_v41_{moe,stage_timing}.py`, and
`test_deepseek_v41_served_generation.py` all green (flag off ⇒ shipped path verbatim).

## 6. GPU A/B arms

`scripts/deepseek_v41/ab_decode_env_levers.py` gains two presets:
`switch_fastpath` (`MTPLX_DSV41_SWITCH_FASTPATH=1`, pure defer) and
`switch_fastpath_b` (`+ MTPLX_DSV41_SWITCH_SUBMIT=1`, defer + async submit — §8).
Every preset pins all six lever keys; `all_levers` runs variant B. Inside
`scripts/deepseek_v41/gpu_window.sh`:

```
ab_decode_env_levers.py --arms control switch_fastpath switch_fastpath_b \
  --context-tokens 1024 --decode-tokens 256 --syncs 16 \
  --stage-timing --warm-repeat --out <receipt.jsonl>
```

`--syncs` re-censuses `hot.eval_indices` (barriers/token, expected ~40) and the
route brackets; `--stage-timing` re-reads `hot.allhit_fence_eval` (expected → ~0
under the fast-path); byte-identity is the recorded token-id sha256 (a differing sha
FAILS the arm).

**Census-probe caveat (window-14):** `--syncs`'s route-stage counters are **not**
reset between arms in that harness, so the per-arm `stages` totals/counts are
**cumulative across arms** (arm _k_ ≈ _k_× the first arm) and are **not**
per-arm-comparable. The reliable per-arm fields are the clean `sync_census`
scalars (`routing_barriers_per_token`, `sinkhorn_*`) and the top-level tok/s / peak
/ sha. This does not affect the decode measurement; note it when reading the census.

## 7. Scope / caveats

- **Value is A/B-pending, not asserted** (a3b caution). This worker proves the
  mechanism removes the second per-layer sync exactly and adds no blocking sync; the
  net decode tok/s is the GPU window's call.
- **Composes with K1** (barrier overlap): K1 *fills* the barrier's idle window
  (barrier stays); K23 *removes* the separate wave/slot fence. Run both arms.
- **Not gated by model key** (W28 convention): the env name provides the isolation —
  an hy3/glm run simply does not set `MTPLX_DSV41_SWITCH_FASTPATH`, and those lanes
  already defer via their profiles anyway.

## 8. GPU window 14 result + diagnosis + variant B

**Measured (integration 7770960a9, 1,024-token prompt, 256 greedy decode, 82 GiB
plan; receipt `docs/deepseek-v41/receipts/gpu-windows/window-14/ab-1024-head-
switch.json`):**

| arm | decode tok/s | Δ vs control | tokens | peak GB |
|---|---:|---:|:---:|---:|
| control | 4.02 | — | sha 033bfcdf… | 77.68 |
| `switch_fastpath` (pure defer) | **3.48** | **−13.4 %** | **identical** | 77.68 |
| head_bf16 (other worker) | 5.28 | +31 % | identical | 75.94 |
| sinkhorn_metal | 4.01 | −0.4 % | identical | 77.68 |

So the fast-path is **byte-identical but −13.4 % slower** — the a3b backpressure
trap ([[a3b-decode-roundtrip-is-the-lever]]), worse than a3b's −3.27 %.

**Why pure defer lost (root cause).** The clean per-arm signal in the receipt:
`routing_barriers_per_token = 40.0` for **both** arms (the fast-path preserved the
one-barrier-per-layer invariant, as designed), and the `--syncs` census confirmed
the all-hit fence became a deferred release (`hot.allhit_defer` counted, 0
`hot.allhit_fence_eval` in the fast-path's own pass). So the mechanism worked. The
regression is **submission**, not sync count:

- The all-hit **deferred** branch (`_run`) submitted **no GPU work** — it appended
  the wave output to `outputs`, called `defer_slot_release`, and moved on. The
  **split** deferred branch does the opposite: `evaluate_component_bindings(defer=
  True)` calls `async_eval(wave_outputs)` — its comment: *"the GPU still needs the
  part submitted now — without it the device idles."* The all-hit path never got
  that treatment.
- DSV4.1's backbone (`deepseek_v41.py`, W41 — **out of this worker's allowlist**)
  has **no submit cadence**. hy3 pairs `deferred_pin_release=True` with
  **`MTPLX_HY3_SUBMIT_CADENCE=8`** (`mtplx/data/expert_profiles.json`), applied in
  `hy3_mlx.py:1165` as `async_eval(hidden)` every 8 decode layers, with the exact
  comment describing this failure: *"without checkpoints the … segments accumulate
  as unsubmitted lazy graph while Python walks the layers, and the GPU idles until
  the next … eval drains the whole backlog at once."*
- Net: on all-hit layers the fast-path removed the fence but replaced it with
  **nothing**, so the lazy graph accrued and the GPU idled until the next barrier
  drained it in a lump — exactly the mechanism hy3's cadence exists to prevent.

**hy3's companion settings (checked against DSV4.1's config):**

| hy3-oq2e profile setting | DSV4.1 `build_streaming_config` | gap |
|---|---|---|
| `deferred_pin_release: true` | `False` (default) | promoted by the fast-path |
| `split_route_release: "deferred"` | `"fenced"` (default) | promoted by the fast-path |
| **`MTPLX_HY3_SUBMIT_CADENCE: 8`** | **none (no DSV4.1 equivalent)** | **the missing companion — variant B** |
| `MTPLX_SUSTAINED_PREFILL: 1` | n/a (decode) | prefill only |
| `cache_policy: frequency`, `cache_scope: layer`, `slot_layout: component-banks` | same (component-banks confirmed by all-hit path) | matched |

The load-bearing missing companion is the **submit cadence**. Its real home is the
backbone decode loop (W41's allowlist), which this worker cannot touch — so variant
B puts the equivalent submit **inside the switch**.

**Variant B — `switch_fastpath_b` (`MTPLX_DSV41_SWITCH_SUBMIT=1`, default off).**
The all-hit deferred branch now `async_eval`s its wave output (non-blocking
per-layer submit) before deferring the release — keeping the GPU fed during the
host graph-build of the following layers, without the blocking round-trip and
without releasing the pin early. This makes the all-hit path behave exactly like
the already-correct split defer path. Scheduling only → **byte-identical**
(asserted in the test's byte-identity matrix for both variants, and the census:
variant B adds exactly one `async_eval` and still 0 blocking wave fences, barrier
still 1). Host syncs/all-hit layer stay **1** (barrier only); the difference from
pure defer is one **non-blocking** submit per all-hit layer.

**Honest read on whether it can win.** Variant B is the principled fix (it closes
the all-hit vs split submission asymmetry the diagnosis found) and is the natural
next A/B. But there is a real chance the lane **cannot** win within the switch:
DSV4.1 already forces `mx.eval(indices)` **every layer**, so unlike hy3's islands
there is only ever ~one layer of unsubmitted graph between barriers — the window a
within-switch submit can fill is small, and MLX's async backpressure may conserve
the block regardless (the a3b result). If `switch_fastpath_b` does not clear
control in window 15, K23 is **dead on this lane**: the fence is genuinely load-
bearing as a per-layer command-buffer boundary here, and the only real cadence
lever lives in the backbone (W41), not this switch. Either way the fast-path stays
**default off**; nothing ships until an arm beats control.
