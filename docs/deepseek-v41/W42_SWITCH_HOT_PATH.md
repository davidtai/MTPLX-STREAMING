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
  backpressure may conserve part of a removed sync's block): mechanically ~26 ms/token
  of exposed fence is gone and no blocking sync is added, so the floor moves from
  268 ms/token toward ~242 ms/token before counting overlap recovery. Default off
  until the window prices it.

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

## 6. GPU A/B arm

`scripts/deepseek_v41/ab_decode_env_levers.py` gains the `switch_fastpath` preset
(`MTPLX_DSV41_SWITCH_FASTPATH=1`; every preset pins all five lever keys, `all_levers`
includes it). Inside `scripts/deepseek_v41/gpu_window.sh`:

```
ab_decode_env_levers.py --arms control switch_fastpath \
  --context-tokens 1024 --decode-tokens 256 --syncs 16 \
  --stage-timing --warm-repeat --out <receipt.jsonl>
```

`--syncs` re-censuses `hot.eval_indices` (barriers/token, expected ~40) and the
route brackets; `--stage-timing` re-reads `hot.allhit_fence_eval` (expected → ~0
under the fast-path); byte-identity is the recorded token-id sha256 (a differing sha
FAILS the arm).

## 7. Scope / caveats

- **Value is A/B-pending, not asserted** (a3b caution). This worker proves the
  mechanism removes the second per-layer sync exactly and adds no blocking sync; the
  net decode tok/s is the GPU window's call.
- **Composes with K1** (barrier overlap): K1 *fills* the barrier's idle window
  (barrier stays); K23 *removes* the separate wave/slot fence. Run both arms.
- **Not gated by model key** (W28 convention): the env name provides the isolation —
  an hy3/glm run simply does not set `MTPLX_DSV41_SWITCH_FASTPATH`, and those lanes
  already defer via their profiles anyway.
