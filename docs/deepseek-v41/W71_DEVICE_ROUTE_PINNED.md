# W71 — Pinned-guarded barrier-free device route (KERNEL_LEDGER K24 revived)

Status: **REVIVED, default off, byte-identical on CPU; GPU parity window pending.**
Behind env `MTPLX_DSV41_DEVICE_ROUTE_PINNED` (default off). Per-layer component-bank
runtime only. CPU-only design; MLX pinned to CPU; no `experts.bin` load (fake bank /
tiny double); ≤1.5 GB RSS. mlx 0.32.2. Author: Opus 4.8 worker
(`feat/deepseek-v41-w71`, branched off W64 `7810a08ea`). This revives the W44
device route (K24, shelved after GPU window 19) by consuming the W64 pin surface
exactly as W64 §6 prescribed — so correctness no longer depends on the slot
stability of *unpinned* experts, only on the pins, which W64 guarantees.

## 0. The two priors this composes

- **W44 (K24, SHELVED).** Removed the per-layer routing barrier by gathering the
  all-hit route over a device expert→slot LUT **without** `mx.eval(indices)` and
  **deferring** the gather (`async_eval`, read back only at the token-end flush).
  GPU window 19 decoded all-zero garbage (−21%, NOT byte-identical). Root cause
  (W44 §8): the barrier-free path read a bank slot **without pinning** and deferred
  the gather; `gather_qmm` reads the bank at **eval** time, so an admission that
  recycled a read slot **in place** across the decode overwrote the slot's bytes
  before the deferred gather ran → wrong expert's weights → garbage. W44's own
  disposition: "device_route is exact **only** when the resident set is static for
  the entire decode (no miss, no eviction)".

- **W64 (R3-pin).** After prefill, pins the per-layer working set: a pinned expert's
  persistent slot is marked **never-recyclable on normal decode admission**
  (`_victim_slot` excludes `_pinned`). A pinned slot is therefore static for the
  whole decode. W64 §6 spelled out how a revived device route must consume this,
  and W64 deliberately did **not** wire it. **W71 is that wiring.**

## 1. Mechanism — a PINNED-only device LUT + a deferred all-pinned check

W71 changes W44 in three places, all gated by `MTPLX_DSV41_DEVICE_ROUTE_PINNED`:

1. **The LUT is built from PINNED experts only.** `runtime.device_route_pinned_lut(
   layer)` snapshots `{expert: slot}` for the layer's **pinned** experts and writes
   `-1` for everything else — an unpinned-but-resident expert reads `-1` exactly
   like a non-resident one. It is rebuilt on the host **only when the layer's PIN
   set changes**: a pin / refresh re-rank (`_maybe_pin_layer` →
   `_mark_device_route_dirty`) or a memory-forced capacity eviction that unpins an
   expert (`_evict_layer_bank_to_capacity` → `_mark_device_route_dirty` →
   `_mark_device_route_pinned_dirty`). A mere LRU churn of the unpinned free tail
   does **not** dirty it — its own dirty flag (`_device_route_pinned_lut_dirty`) is
   distinct from the W44 resident LUT's.

2. **The device path is taken speculatively; the all-pinned check is deferred.**
   Per layer per token the barrier-free gather is issued over `lut_pinned[indices]`
   (`HotExpertSwitchGLU._run_device_route(pinned=True)`) with **no** `mx.eval(
   indices)` — an all-pinned route's assignments read their pinned slots (exact),
   any non-pinned expert reads `-1` → clamped to row 0 (a void value). `indices` is
   `async_eval`'d and a **pinned** probe is enqueued. At the token boundary
   `flush_device_route_probes()` does **one** batched `mx.eval` of all probed
   indices and, for each pinned probe, flags every routed expert that is **NOT
   currently pinned**. All-pinned → keep (the deferred gather stands); otherwise the
   layer is recomputed on W44's fenced cold-recovery path
   (`DeepseekV41Backbone._device_route_recover`): rewind every layer's cache to
   pre-token, re-run the whole span on a fresh `shared` with the flagged layers
   forced fenced, pay one routing barrier per flagged layer.

3. **Recovery also handles a pinned expert force-evicted mid-token.** The flush
   checks the **CURRENT** pinned set (`bank.pinned_experts`), not the build
   snapshot. So an expert that was pinned when the LUT was built but has since been
   force-evicted (memory hard constraint, `respect_pins=False` →
   `invalidate_expert` unpins it) is flagged for a fenced recompute — closing the
   one window in which W64 permits a pinned slot to move. The force-eviction also
   dirties the pinned LUT, so the next build drops the evicted expert.

Pins are established **out-of-band**: the decode forward calls
`runtime.pin_working_set()` once at the prefill→decode boundary (a pure host
ranking over the resident set — no gather, no barrier), because the W64 switch
hook rides the very `mx.eval(indices)` this route removes (W64 §6 point 1). It is
idempotent (epoch-gated per layer, refresh-aware) and a no-op unless
`MTPLX_DSV41_PIN_WORKING_SET` is armed.

## 2. Exactness — three sentences

An **all-pinned** layer's device gather reads, for every assignment, the pinned
expert's persistent slot — the identical `bank_index` the fenced path would pin —
so the gather runs the identical kernel over the identical rows and is
**bit-for-bit** the fenced output, and because W64 forbids recycling a pinned slot
on normal decode admission, that slot cannot move between the deferred gather's
issue and its eval, so the byte-identity survives the whole decode (the W44 §8
race is gone). A **not-all-pinned** layer (an unpinned-resident, non-resident, or
mid-token force-evicted expert) reads a void row → wrong → but the deferred flush
detects it against the **current** pin set and the backbone re-runs that layer on
the fenced path, which **is** the reference, so the reconciled token is
byte-identical layer for layer. Correctness therefore rests **only** on the pin
invariant W64 guarantees — never on the slot stability of any unpinned expert —
which is exactly the precondition W44 §8 said was missing.

## 3. Barrier count per token

| regime | fenced (control) | pinned device route (W71) |
|---|---:|---:|
| per layer, all-pinned | 1 | **0** |
| per layer, not-all-pinned (fenced on the recovery pass) | 1 | 1 |
| span-end verify (one batched `mx.eval`) | 0 | **1** / token |
| **all-pinned token** | **40** | **1** (batched verify only) |
| **token with *m* not-all-pinned layers** | **40** | **m + 1** |

The per-layer routing barriers drop **40 → m** (only the not-all-pinned layers are
fenced, on the recovery pass); the single batched span-end verify sync is the "+1"
(`flush_device_route_probes` batches all probed indices into **one** device→host
sync — never one per layer). With the `pin_ws` arm (pin all keys) every all-hit
layer is an all-pinned layer, so the barrier-free fraction equals the plain all-hit
fraction (W44 §5: ≈0.61 of layer-calls all-hit at the cold window-12 rate, up to the
warm within-prompt 0.835) — at **zero** recycle risk.

## 4. Test coverage (`tests/test_deepseek_v41_device_route_pinned.py`, 24 tests)

1. all-pinned device output is bitwise-identical to the fenced switch and the
   `_run` performs ZERO host syncs, at M=1 and M=4;
2. the pinned LUT is built from PINNED experts only (an unpinned-but-resident
   expert reads `-1`) and refreshes only when the pin set changes;
3. the deferred pinned probes flag exactly the not-pinned experts per layer over an
   all-pinned / partial / none-pinned sequence, and the flagged layers recover to
   fenced byte-identically, at M=1 and M=4;
4. **(real runtime)** the production `device_route_pinned_lut` maps pinned-only, a
   memory-forced eviction (`invalidate_expert` + dirty mark) drops the pin and the
   next flush flags it, the telemetry counts barrier-free vs recovered layers, and
   **one flush call is exactly ONE batched `mx.eval`** (the "+1");
5. **(backbone end to end)** the whole DSV4.1 decode/verify forward is byte-identical
   to the fully fenced path — logits + per-layer KV/compress/index cache + engram
   `_buf`/`_len` — across all-pinned / partial / multi / none / **forced-eviction**
   layer patterns at M=1 and M=4, and the routing barriers paid equal the number of
   NOT-all-pinned layers (0 / 1 / 3 / 8 / 1);
6. **(W44 race adapted)** a pinned slot is NOT recycled under a 20-expert decode
   churn — the pinned experts keep their exact bank rows while only the free tail
   recycles — so a deferred gather over pinned slots is isolated from the recycle
   that corrupted the shelved W44 route.

Run: `PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 -m pytest
tests/test_deepseek_v41_device_route_pinned.py` → **24 passed**. The W44
(`test_deepseek_v41_device_route.py`, `..._recovery.py`, `..._parity.py`) and W64
(`test_deepseek_v41_pinned_working_set.py`) suites, plus `..._ab_env_levers.py`,
stay green.

## 5. Arm + telemetry

- `scripts/deepseek_v41/ab_decode_env_levers.py`: the **`device_route_pinned`** arm
  (`MTPLX_DSV41_PIN_WORKING_SET=all` + `MTPLX_DSV41_DEVICE_ROUTE=1` +
  `MTPLX_DSV41_DEVICE_ROUTE_PINNED=1`) — pin all keys so every all-hit layer is an
  all-pinned layer taken barrier-free at zero recycle risk. `DEVICE_ROUTE_PINNED_ENV`
  is a first-class lever key (every preset pins it None); the dry-run test covers it.
- Telemetry: `runtime.device_route_pinned_telemetry()` reports, per probe flush
  (≈ per decode token in the steady no-recovery state), `barrier_free_layers`
  (all-pinned, kept) vs `recovered_layers` (not-all-pinned, refenced) and the
  derived `barrier_free_layers_per_flush` — the K24-revived win read. Surfaced in
  both runtime snapshots, the A/B receipt (`device_route_pinned`), and the served
  event (`serve_stream_counters.py`, with the before/after window delta).

## 6. What W71 lands (end to end)

- `expert_runtime.py`: `DEVICE_ROUTE_PINNED_ENV`; the pinned LUT state
  (`_device_route_pinned_lut` / `_snapshot` / `_lut_dirty` + telemetry counters);
  `device_route_pinned_lut` / `device_route_pinned_snapshot`;
  `_mark_device_route_pinned_dirty` (and `_mark_device_route_dirty` now also
  invalidates the pinned view on any residency change); `enqueue_device_route_probe(
  ..., pinned=)` + a `flush_device_route_probes` that checks the current pin set for
  pinned probes and counts the barrier-free/recovered telemetry;
  `device_route_pinned_telemetry`; `reset` clears the pinned LUT state.
- `models/expert_mlx.py`: `HotExpertSwitchGLU._run_device_route(pinned=)` swaps the
  resident LUT/snapshot for the pinned ones; the `_run` device-path gate recognizes
  `MTPLX_DSV41_DEVICE_ROUTE_PINNED` (resident W44 path keeps its 3-arg probe call so
  a pre-W71 runtime double is unaffected).
- `models/deepseek_v41.py`: `_device_route_active` arms recovery for either device
  flag; `_forward_span` establishes pins out-of-band once at the decode-span start
  when the pinned flag is set. The W44 `_device_route_recover` machinery is reused
  **unchanged** (miss layers forced fenced, rewind-all + fresh-`shared` re-run).
- `models/deepseek_v41_cache.py`: **no change** — the length-based `rollback` / `mark`
  already rewind window/compress/index/frontier and leave the engram untouched.
- `serve_stream_counters.py`: forwards the `device_route_pinned` block + window delta.
- `scripts/deepseek_v41/ab_decode_env_levers.py`: the `device_route_pinned` arm +
  `_device_route_pinned_telemetry` in the receipt.
- Tests: `tests/test_deepseek_v41_device_route_pinned.py` (§4);
  `tests/test_deepseek_v41_ab_env_levers.py` extended for the new arm + env keys;
  the W64 runtime double mirrors the new `_device_route_pinned_lut` internal.

## 7. Pending — the GPU parity window

W71 is CPU-exact and default off. The shelved W44 route failed the real-artifact
`MTPLX_GPU_PARITY` window (window 19). Before promotion, re-run
`test_gpu_parity_device_route_vs_fenced` **with `MTPLX_DSV41_DEVICE_ROUTE_PINNED=1`
and `MTPLX_DSV41_PIN_WORKING_SET=all`** inside a GPU flock window: the pin guard
should make it byte-identical where the unpinned W44 route was not. The
decode-tok/s win on the all-pinned fraction (barrier removal vs the cold recovery
compute) is the A/B's call (`device_route_pinned` vs `control` / `pin_ws`), out of
scope for this CPU-only worker. K24 status: **revived-pending-window**.
