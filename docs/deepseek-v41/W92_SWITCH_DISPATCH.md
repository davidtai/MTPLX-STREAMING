# W92 — AR-decode streamed-switch dispatch/sync census + the `switch_lean` arm

**Target:** the DeepSeek-V4.1 `moe.routed_switch` on the M=1 AR-decode path (window
33, arm `cell16k_ring`, `ar-16k-cell16k-ring.json`: `moe.routed_switch` 2.59 ms/layer-call,
40 layer-calls/token, decode 2.33 tok/s). The switch is `HotExpertSwitchGLU`
(`mtplx/models/expert_mlx.py`), `slot_layout="component-banks"`, codec affine/mxfp4.

**Method:** CPU-only fake component-bank runtime (`_integrated_hy3_artifact`, top_k=6),
`mx.set_default_device(mx.cpu)`, route-stage probe armed. Metal-side dispatches counted
by wrapping the module-level `mx.*` primitives issued by the switch; host syncs counted
as blocking `mx.eval` calls; fence location read off the probe brackets. No GPU, never
the real model.

## Regime (READ FIRST — the win is small)

At **16K the AR route is ~30% all-hit, not 92%**. Window 33's receipt records
`hot.all_hit` = **3,054 of 10,240 layer-calls (29.8%)**; the other ~70% are split
(miss) layers that **block on SSD reads regardless of any fence**. The fence deferral
removes only the *exposed* (non-overlapped) blocking evals: the second `mx.eval` on the
30% all-hit layers, plus the residual per-part fences on split layers that the miss I/O
did not already hide. Sizing that against the 2.59 ms/layer × 40 layers ≈ 104 ms/token
switch budget, the removable exposed cost is **≤ ~11 ms/token (≈ 2.5%)** — inside
single-prompt seed noise. The measured **+2.86%** for variant B is from the **1K**
regime (window 16, `ab-1024-fastpath-b.json`), where all-hit dominates; do not expect
that at 16K. This arm is a correctness/host-sync-hygiene lever with a small, seed-noise-
adjacent upside, not a headline speedup. The 80→40 barriers/token figure is a **1K**
count and is retained below only as the all-hit-layer illustration.

## Task 1 — per-layer-call dispatch / host-sync census (shipped path)

Per **M=1 all-hit** layer-call (all 6 routed experts persistent-resident):

| item | count | notes |
|---|---|---|
| **host syncs (blocking `mx.eval`)** | **2** | `mx.eval(indices)` routing barrier + `mx.eval(wave_output)` wave fence |
| — `hot.eval_indices` | 1 | the one routing barrier (unavoidable — host plans loads) |
| — `hot.allhit_fence_eval` | 1 | **the removable second sync** (`synchronous_fence`) |
| host read `.tolist()`+`int()` (`hot.route_host`) | 1 | reads the *already-evaluated* small indices array; **not a second GPU barrier**; feeds observe_route / try_all_hit_route / route_waves |
| `mx.gather_qmm` (`hot.allhit_gather_qmm`) | **3** | one per component (gate/up/down) over all 6 routed slots via `rhs_indices` — **already minimal, not per-expert** |
| `mx.broadcast_to` | 1 | assignment inputs `tokens[:,None,:]` → `(1,6,H)` |
| `swiglu` + `mx.clip` + `mx.minimum` | 1 + 1 + 1 | one fused clamped-SwiGLU (clip+min are the DSV4.1 swiglu_limit=10 clamp) |
| `mx.take` / `concat` / `argsort` / `astype` | 0 | one wave, ordered positions → no reorder; no cast on the decode gather |
| Python per-expert loop | 0 | the all-hit branch does **not** loop per expert |

Per **M=1 miss** layer-call (mixed hits + misses, one wave):

| item | 1 miss + 5 hits | 3 miss + 3 hits |
|---|---|---|
| **host syncs (blocking `mx.eval`)** | **3** (indices + hit fence + 1 miss fence) | **5** (indices + hit fence + 3 miss fences) |
| `mx.gather_qmm` (total) | 6 (3 per part) | 12 |
| `mx.take` / `concat` / `argsort` | 3 / 1 / 1 | 5 / 1 / 1 |

**Fence location (proved):** `expert_mlx.py` — `with _route_probe.bracket("hot.allhit_fence_eval"): synchronous_fence(ready, wave_output)`, taken because `wave_index == final_wave and _deferred_pin_active and phase is DECODE` is **False** (`_deferred_pin_active = config.deferred_pin_release (False for DSV4.1) or _fastpath_can_defer (env FASTPATH, unset in window 33)`). `split_route_release="deferred"` only feeds `_split_deferred_active`, which does **not** gate the all-hit fence.

## Task 2 — the dumb parts, named (empirically)

1. **The all-hit synchronous wave fence** — a redundant second blocking `mx.eval` on the
   ~30% all-hit layers; the route's slots are already pinned by `try_all_hit_route`, so
   the release can defer to the next routing barrier without a fence. **Removed by the arm.**
2. **Per-miss serialized fences on split layers** — each miss part fences separately; the
   deferred-split path async-submits every part behind the one barrier. The *net* effect
   is small because split layers block on SSD reads anyway.
3. **The `.tolist()` + `int()` loop** (`hot.route_host`) — host work, **not** a GPU
   barrier (indices already materialized); needed for streaming load planning.

**Hypotheses that did NOT hold (measured):**
- *"per-expert loop issuing 6×3 gather_qmm"* — **false**: the all-hit path issues exactly
  **3** grouped `gather_qmm` (`hot.allhit_gather_qmm` == 3·`hot.all_hit`). The per-expert
  loop lives in `MappedExpertSwitchGLU` (the mmap switch), **not** the streaming path.
- *"unnecessary astype"* — **none** on the M=1 all-hit gather.

**Root cause of window-33's fenced AR decode:** `arm=cell16k_ring` never armed
`MTPLX_DSV41_SWITCH_FASTPATH`/`_SUBMIT` (both `None` in the receipt), so every all-hit
layer synced. The variant-B mechanism already existed, byte-identical and pin-safe.

## Task 3 — the fixes

(a) **Minimal-dispatch all-hit switch** — *already minimal*. Added the all-hit-scoped,
probe-gated counter `hot.allhit_gather_qmm` (a delta on the shared `hot.switch_gather_qmm`
so split parts and prefill waves do **not** inflate it) — the receipt's
`gather_qmm_per_all_hit_call` is exactly **3**.

(b) **Defer the all-hit fence** — armed via `MTPLX_DSV41_SWITCH_FASTPATH=1` +
`MTPLX_DSV41_SWITCH_SUBMIT=1`. Defer = `defer_slot_release(ready, wave_output)`, released
at the next layer's `flush_deferred_slot_releases()` after that layer's
`mx.eval(indices)` (the covering eval). Submit = `async_eval(wave_output)` — non-blocking,
keeps the GPU fed (pure defer lost −13% at 1K, W42 window-14, because the lazy graph
accrued and the device idled; the async submit recovered the small +2.86% at 1K, window
16). **DECODE-only:** the deferral is now self-guarded with `and phase is
RoutingPhase.DECODE` — a multi-group layer-major *prefill* has no covering eval between
groups, so a deferred gather there could read a slot the next group reloaded; prefill
always fences. **Pin-safety (W44 hazard avoided):** `try_all_hit_route` pins the whole
route and `defer_slot_release` holds the `_SlotPinClaim`s until the covering flush, so
during the deferral `slots.invalidate` on any route slot raises `ExpertSlotError`
("cannot invalidate an active expert slot") and eviction is refused/redirected.

(c) **Miss reads concurrent** — *already concurrent* (`begin_split_route` submits every
miss part to the `_split_executor` thread pool; `overlap_miss_reads` batches them). The
removable per-part fences fold into the deferred-split path.

**CRITICAL fix — deferred-route lock vs the KV boundary (`expert_runtime.py`
`_apply_derived_allowance`).** A deferred split/all-hit route keeps its layer
`threading.Lock` held (acquired in `begin_split_route`, released only in
`PendingSplitRoute._finalize_if_ready`) until the next covering flush. On the derived-
policy profile (`deepseek-v41-mxfp4-75`, no `expert_cache_limit_bytes`),
`_apply_derived_allowance` — called by `admit_kv_tokens`/`release_kv_tokens` at every KV
boundary — takes **every** layer lock, so the generation thread self-deadlocks on a lock
only a later forward would release. This is **latent today on the DSpark lane** (the
W61/W81 verify deferral is default-on) and would be armed lane-wide by `switch_lean`.
Fix: `_apply_derived_allowance` now calls `flush_deferred_slot_releases(evaluate=True)`
first (exactly as `reset()`/`close()` do at their boundaries), draining the generation
thread's own already-submitted deferrals before it takes the locks. Regression test:
a deferred split → `admit_kv_tokens` on a thread completes < 5 s.

## Arms + counters (`scripts/deepseek_v41/ab_decode_env_levers.py`)

- **`switch_lean`** = `SWITCH_FASTPATH=1` + `SWITCH_SUBMIT=1` + `VERIFY_SINGLE_BARRIER=1`.
- **`cell16k_ring_switch`** = `cell16k_ring` + those keys (effective delta = the two
  fence-defer keys; `VERIFY_SINGLE_BARRIER` is default-on).

Receipt `switch_dispatch` block (AR pass, `--stage-timing`): `eval_indices`, `all_hit`,
`allhit_fence_synced` vs `allhit_fence_deferred`, `allhit_defer_submit`,
`allhit_deferred_pct`, `split_route`, `begin_split_route`, `allhit_gather_qmm`,
`switch_gather_qmm_total`, `gather_qmm_per_all_hit_call` (= 3), `route_host_tolist`. The
`ENABLED` probe flag is armed/restored in try/finally so a failed arm never leaves it on.

## Before / after (per streamed layer-call, blocking `mx.eval`)

| layer type | shipped `cell16k_ring` | `switch_lean` / `cell16k_ring_switch` |
|---|---|---|
| all-hit (**~30%** of 16K layers; ~92% at 1K) | **2** (indices + wave fence) | **1** (indices only; gather async-submitted, released at next barrier) |
| miss, 1 miss + 5 hits | **3** | **1** (parts async; but the layer still blocks on SSD) |
| miss, 3 miss + 3 hits | **5** | **1** (parts async; SSD-bound regardless) |
| gather_qmm / all-hit call | 3 (already minimal) | 3 (unchanged) |

**Expected real-model effect at 16K: ≤ ~2.5% (≈ ≤11 ms/token), inside seed noise** —
because only ~30% of layers are all-hit and the 70% split layers are SSD-bound. Larger
gains (the +2.86% variant-B figure) belong to the 1K regime (window 16).

**Byte-identity & safety (fake runtime, CPU-only,
`tests/test_deepseek_v41_w92_switch_dispatch.py`, 12 tests):** AR M=1 × 64-step hit/miss
byte-identity; miss-route byte-identity; W81 multi-wave batched split byte-identical
*under the arm* (the fastpath fake never partitions, so this covers that composition on
the real runtime); all-hit gather == 3 and split layers do not inflate the all-hit-scoped
counter; sync-count before/after; the **deadlock regression** (deferred split →
`admit_kv_tokens` off-thread < 5 s); the **non-vacuous pin-safety** test (a control run
with no pins evicts the pin set; the deferred run refuses eviction below the pinned count
and the pin set survives, then is invalidatable only after the covering flush); and the
**prefill-fence** invariant (the arm never defers in prefill). Existing
`tests/test_deepseek_v41_switch_fastpath.py` (17 passed) locks the FASTPATH byte-identity
+ slot-safety on a real streamed runtime.

> GPU A/B still owed: `cell16k_ring` vs `cell16k_ring_switch` (and `control` vs
> `switch_lean`) via `gpu_window.sh --stage-timing`. Expect the `switch_dispatch` block to
> show `allhit_fence_deferred == all_hit` (0 synced) and `gather_qmm_per_all_hit_call == 3`
> on the switch arm, the token-id sha256 to match the ring arm (byte-identical), and the
> decode-tok/s delta to be **small and possibly within seed noise** at 16K (~30% all-hit).
