# W64 — Post-prefill pinned working set (R3-pin; the safety fix W44 needs)

Status: **LANDED, default off, byte-identical when off; CPU-verified.** Behind env
`MTPLX_DSV41_PIN_WORKING_SET` (default off). Per-layer component-bank runtime only.
CPU-only design; MLX pinned to CPU; no `experts.bin` load (fake bank / tiny double);
≤1.5 GB RSS. mlx 0.32.2. Author: Opus 4.8 worker (`feat/deepseek-v41-w64`). This
lever does **not** re-enable the W44 device route — it supplies the slot-stability
precondition a future device route must consume (§6).

## 0. Why this exists — the W44 window-19 race, restated

W44 (KERNEL_LEDGER K24) removed the per-layer routing barrier by issuing the all-hit
`gather_qmm` over a device-side expert→slot LUT **without** `mx.eval(indices)` and
**deferring** the gather (`async_eval`, read back only at the token-end flush). GPU
window 19 decoded all-zero garbage (−21%, NOT byte-identical). Root cause proven on
CPU (`test_deferred_device_gather_races_with_slot_recycle`, W44 §8): the barrier-free
path reads a bank slot **without pinning** and **defers** the gather; `gather_qmm`
reads the bank at **eval** time, so any admission that recycles a read slot **in
place** — the LRU eviction/admission that happens across a 256-token decode —
overwrites the slot's bytes before the deferred gather runs. It reads the wrong
expert's weights → catastrophic garbage.

W44's own disposition (§8): "device_route is exact **only** when the resident set is
static for the entire decode (no miss, no eviction)". The census (W24) is why it is
not static today: a 64-token decode touches **~102 distinct experts/layer** (of 384),
and at **~115 slots/layer** (the only budget that fits 82 GiB) LRU == Belady — every
slot is live, so a decode admission recycles a slot every few tokens. Once the MTP
head is resident, cold slots became scarce enough to cost AR **5.86 → 2.12 tok/s**
(window 25). No barrier-free-and-exact fix exists **for a churning LRU bank** because
safety needs the host slot ids the lever removed.

**W64 makes a subset of the bank static on purpose.** After prefill it pins the
per-layer working set — the top experts by prefill routing frequency — and marks
those slots **never-recyclable on normal decode admission**. A pinned expert's slot
is then stable for the whole decode, so a deferred device gather over pinned experts
**cannot** race a recycle. That is the exact safety property W44 §8 said was missing;
this lever supplies it without reintroducing the routing barrier.

## 1. What W64 lands (end to end)

- `expert_streaming.py` (`LayerExpertSlotBank`): the pin state + eviction change.
  - `_pinned: set[int]` — the never-recycled working set (empty until pinned).
  - `_prefill_route_freq: Counter` — per-expert prefill routing frequency,
    accumulated in `prepare_prefill_seed` so the working set can be ranked by
    prompt-frequent experts with no caller-supplied count.
  - `pin_working_set(*, top_k=None, free_tail=0, experts=None) -> tuple[int, …]`
    — ranks resident experts by `_pin_rank` (prefill freq, then decode score, then
    id) and pins the top set, capped so at least `free_tail` persistent slots stay
    unpinned for misses; or pins exactly `experts` (the `pin_ws` arm passes the full
    resident set → a fully static layer).
  - `_victim_slot(*, pinned, respect_pins=True)` — normal decode admission now
    excludes `_pinned` from eviction; a **memory-forced capacity eviction** passes
    `respect_pins=False` (memory is the hard constraint, [[never-exceed-the-memory-knob]])
    and `invalidate_expert` then unpins the evicted expert.
  - `pinned_experts` / `pinned_count` / `pinned_static` / `route_all_pinned(ids)`
    — the query surface a device route consumes (§6).
- `expert_runtime.py`: the trigger + telemetry.
  - `parse_pin_working_set` / `parse_pin_refresh_tokens` — env parsers (read at USE,
    [[env-flags-read-at-use-not-import]]).
  - `pin_working_set_hook(layer, expert_ids, phase)` — the switch-side per-route
    hook: no-op (one env read) unless armed; on a DECODE route it pins the layer's
    working set on the first decode route after prefill (refreshed every
    `MTPLX_DSV41_PIN_REFRESH_TOKENS` decode epochs when set) and records the
    all-pinned-hit telemetry.
  - `pin_working_set(layer=None)` — **out-of-band** force-pin (for a backbone that
    prefers to pin explicitly at the prefill→decode boundary rather than lazily on
    the first decode route; §6), `clear_working_set_pins`, `layer_pinned_static`,
    `route_all_pinned`, `pinned_working_set_telemetry`.
  - Capacity eviction (`_evict_layer_bank_to_capacity`) uses `respect_pins=False`.
  - `snapshot()` and `resource_telemetry_snapshot()` carry a `pin_working_set` block.
- `models/expert_mlx.py` (`HotExpertSwitchGLU._run`): ONE guarded call —
  `pin_working_set_hook(layer, expert_ids, phase)` after `observe_route` /
  `prepare_prefill_seed`, before the route executes (so this token's route respects
  the fresh pins). Guarded with `getattr`, so a runtime double without the hook is
  unaffected. The K27/K31 gather paths are untouched.
- `serve_stream_counters.py`: forwards the snapshot `pin_working_set` block onto the
  served event with the window's all-pinned-hit rate (before/after delta).
- `scripts/deepseek_v41/ab_decode_env_levers.py`: the `pin_ws` arm
  (`MTPLX_DSV41_PIN_WORKING_SET=all`) + `pin_working_set` telemetry in the receipt.
- `tests/test_deepseek_v41_pinned_working_set.py`: the CPU tests (§4).

## 2. The env knob

`MTPLX_DSV41_PIN_WORKING_SET`:

| value | meaning |
|---|---|
| unset / `0` / `off` / `false` / `no` / empty | **off** (default; byte-identical) |
| `all` / `keys` / `1.0` / `100%` | pin the whole resident set per layer → fully static layers (the `pin_ws` arm) |
| a fraction `f` in `(0,1]` (`0.5`, `50%`) | pin `round(f × persistent_slots)`, leave the rest as a free tail |
| a positive integer `K` | pin the top-`K` resident experts, leave `persistent_slots − K` as a free tail |

Ambiguity rule: a bare integer is a **slot count**; write `1.0` / `100%` / `all` for
"the whole set". `MTPLX_DSV41_PIN_REFRESH_TOKENS=N` re-ranks the pinned set every `N`
decode epochs (0 / unset = pin once after prefill and never refresh).

## 3. Exactness — byte-identical with pinning on OR off

**Off:** `_pinned` is empty, so `_victim_slot`'s `blocked` set is exactly the route's
own hit set (as before), the hook returns after one env read, and no telemetry or
LUT-dirty mark fires. Every output, counter, and residency trajectory is bitwise
identical to the pre-W64 code (locked by the existing 382-test sweep; the two
pre-existing `DenseIslandSwitchGLU`/multi-wave failures on the base branch are
unrelated to W64 and present without these changes).

**On:** pinning is a **pure cache-policy change**. For any route the runtime serves
the *requested* experts — a hit resolves to the expert's persistent slot, a miss is
admitted (persistent free-tail or transient) and served from the correct expert's
bytes. Pinning changes only **which** slot an expert occupies and the hit/miss split,
never **which** expert is served. So the gather output is a pure function of the
routed ids and is identical with pinning on or off. The CPU test
`test_served_gather_byte_identical_pin_on_off` proves this concretely: it models the
physical component bank (slot → resident expert, written by each load, read by the
gather), asserts every resolved slot holds the *requested* expert through a churny
decode, and checks the concatenated per-assignment gather output is bitwise identical
across `off` / top-K / `all`. (A wrong slot — the W44 recycle-before-read — would show
as a wrong expert here; it does not, because pinned slots never recycle.)

## 4. Test coverage (`tests/test_deepseek_v41_pinned_working_set.py`, 10 tests)

1. pinned experts survive a 40-cold-expert churn that recycles only the free tail;
2. a decode miss still admits into the free tail (pinned untouched);
3. `pin_ws` (pin all keys) freezes the layer — no cold expert takes a persistent
   slot, `pinned_static` holds, `route_all_pinned` is True for resident routes;
4. the working set is ranked by prefill routing frequency;
5. a memory-forced capacity eviction MAY evict a pinned expert and unpins it;
6. served-gather byte-identity with pinning on vs off (the model in §3);
7–9. the runtime hook pins after prefill, honours the refresh cadence, is a no-op
   when off, and the telemetry reports pinned-count-per-layer + all-pinned-hit rate;
10. the env parsers.

Run: `PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 -m pytest
tests/test_deepseek_v41_pinned_working_set.py` → **10 passed**.

## 5. Telemetry

`pinned_working_set_telemetry()` (in both runtime snapshots, the A/B receipt, and the
served event) reports, per session/window:

- `pinned_by_layer` / `pinned_total` — pinned count per layer (a gauge);
- `static_layer_count` — layers with an active pinned working set;
- `decode_routes` / `all_pinned_routes` / `all_pinned_hit_rate` — the fraction of
  decode layer-routes whose experts are **all pinned** (i.e. the fraction a
  barrier-free device route could take race-free). Cumulative counters, so the
  served-event delta reports the per-window rate.

`all_pinned_hit_rate` is the single number that decides §6: it is the measured share
of the decode that a pinning-guarded device route would keep barrier-free.

## 6. How the W44 device route should consume `pinned_static` (NOT wired here)

W64 deliberately does **not** re-enable the device route. When it is re-enabled under
a pinning redesign, it must consume this surface as follows:

1. **Establish pins out-of-band, once, at the prefill→decode boundary.** The switch
   hook (§1) runs on the fenced path *after* `mx.eval(indices)`; the device route
   removes exactly that barrier, so it must **not** rely on the hook. Instead the
   decode forward calls `runtime.pin_working_set()` once when prefill ends (a pure
   host ranking over the already-resident set — no gather, no barrier). Pins are then
   in place before the first device gather.
2. **Gate the barrier-free path per route on `route_all_pinned`, not on all-hit.**
   Today the device path engages on any all-hit layer; under W64 it must engage only
   when `runtime.route_all_pinned(layer, indices)` — every routed expert pinned →
   slot-stable. Because that check needs the ids on the host, it is done device-side:
   fold a `pinned_mask` (int32 `[expert_count]`, 1 where pinned) into the LUT build,
   `mx.take(pinned_mask, indices)` on device, and treat a route as barrier-free only
   when the reduced mask is all-ones (verified in the deferred token-end flush, the
   same batched read that already checks residency — no new per-layer barrier). A
   route touching an unpinned expert takes the fenced path for that layer.
3. **Pinned slots never recycle, so the deferred gather is safe.** For a route that
   is `route_all_pinned`, every read slot is in `_pinned` and cannot be recycled by
   decode admission (`_victim_slot` excludes it) — the W44 §8 race cannot occur. The
   free-tail slots still churn, but no all-pinned route reads them.
4. **Capacity eviction is the one exception** (memory hard constraint): it may unpin
   an expert, which marks the LUT dirty and shrinks `pinned_experts`, so the next
   `route_all_pinned` for a route touching it returns False → fenced. Correct by
   construction.

The `pin_ws` arm (pin all keys) is the maximal case: every layer is fully static, so
`route_all_pinned` == all-hit, and the device route's barrier-free fraction equals the
plain all-hit fraction — at zero recycle risk.

## 7. Expected effect on the all-hit (barrier-free-eligible) layer fraction

Pinning is **not** a hit-rate lever — W24 already proved a static top-N pin ≈ LRU
*within* a prompt (the ~102-distinct-experts working set fills the ~115 slots either
way). W64 does not try to raise the all-hit rate; it makes the **existing** all-hit
layers **safe** for a barrier-free gather. So the metric it moves is the fraction of
all-hit layer-routes a device route may take **without** the routing barrier:

- **Today (W44, no pinning): 0 %** — the barrier-free path is not exact on *any*
  churning layer (window 19: all tokens garbage), so it is shelved.
- **With `pin_ws` (pin all keys): the full all-hit fraction** — every all-hit route
  is an all-pinned route. From the census that is ≈ **0.61 of layer-calls all-hit at
  the cold window-12 rate** (W44 §5: 1558/2560 layer-calls all-hit) and up to the
  **warm within-prompt hit 0.835** (W24) once residency settles. Those layer-routes
  become barrier-free-eligible where before they were unusable.
- **With a fractional pin (top-`K` + free tail):** the eligible fraction is the share
  of routes whose experts all fall in the pinned top-`K`. It trades barrier-free
  coverage for a free tail that still absorbs cold misses into persistent slots (so
  the fenced miss path stays fast). `all_pinned_hit_rate` (§5) measures this share
  live; the fraction rises monotonically with `K` toward the `pin_ws` ceiling as the
  free tail shrinks to zero.

The honest read: W64 converts W44 from "shelved, exact on 0 % of a real decode" to
"exact on the all-pinned fraction", and the telemetry reports that fraction directly.
Whether the barrier removal on that fraction nets a decode-tok/s win is the GPU A/B
(`pin_ws` + a pinning-guarded `device_route`, window 16+) — out of scope for this
CPU-only worker and gated on the `MTPLX_GPU_PARITY` window being clean under the
guard in §6.
