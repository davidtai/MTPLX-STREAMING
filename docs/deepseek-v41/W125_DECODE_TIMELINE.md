# W125 — host-side decode timeline probe (`mtplx/dsv41_decode_timeline.py`)

## Why

AR decode on the 16K cell runs 5.5–5.9 tok/s (169–181 ms/tok) with GPU idle share
0.73–0.75. Window 50 fanned each SSD miss record read into concurrent sub-reads
(read_ns/wall 2.45, same bytes) for only +7%; prefetch-off gained +3.5%. So ~120+
ms of every token is host-side wait that no receipt localized. `--stage-timing`
can't answer it: it forces the eager path (226 ms/tok) and kills the compile
levers, so its breakdown is of a different, slower graph.

This probe records the host timeline **on the real compiled v2 runner path**
(levers on), per decode token and per MoE layer, and aggregates it in-process into
per-phase mean/p50/p95 (per layer and per token) plus a host-gap accounting. It is
env-gated (`MTPLX_DSV41_DECODE_TIMELINE=1`), a no-op when unset, byte-identical
on/off (pure host timestamps — no `mx` op, no tensor read), and stamps its own
overhead (< 0.5 ms/token budget).

## Reading the phases under MLX lazy eval (IMPORTANT)

MLX is lazy. The **only per-layer blocking sync** is the switch's
`mx.eval(indices)` (expert_mlx.py `HotExpertSwitchGLU._run`), which the marks
bracket as **`gate_to_barrier`** (`moe_start → routing_barrier_done`). That eval
materializes the **previous** layer's routed gather **+ this** layer's attention
**+** the gate. Consequences:

- **`gate_to_barrier`** = "barrier wait incl. GPU (prev gather + attention + gate)"
  — dominated by GPU **execution** + the sync round-trip, not host build.
- **`attn`**, **`combine`**, **`head_dispatch`**, **`interlayer_gap`** are lazy
  **graph-build microseconds**, not the attention/combine/head compute (that runs
  at the next `gate_to_barrier` eval).
- **`host_gap`** is defined as **`post_barrier_host` = `expert_dispatched −
  routing_barrier_done`**, summed over layers: the pure host span **after** the
  barrier sync (route `.tolist` + plan/pin + gather graph-build), during which no
  GPU work is in flight. It deliberately **excludes** the big GPU chunk the barrier
  eval forces. (The pre-fix definition — `expert_dispatched − attn_end` — swallowed
  that GPU compute and read as ~150 ms/tok; see window 52 below.)

### Fenced split path caveat (`switch_config.fenced_split_path`)

DSV4.1's `build_streaming_config` leaves `deferred_pin_release` **False** and
`split_route_release` **"fenced"** (neither is in `_PROFILE_PLAN_FIELDS`), and with
`MTPLX_DSV41_SWITCH_FASTPATH` off the split path runs `evaluate_bindings(
force_sync=True, defer=False)` — a **blocking** `mx.eval` of the routed gather.
So when `switch_config.fenced_split_path == true`:

- the **`ready_to_dispatch`** phase and the **`miss_issue_to_ready`** PHASE
  (timestamp delta) can include the **GPU gather execution**;
- only **`miss_wait_total`** (the ACC accumulator, measured inside
  `iter_ready_misses`) is the **pure host SSD read wait**.

The runtime stamps `switch_config` (fenced_split_path, deferred_pin_release,
split_route_release, switch_fastpath, overlap_miss_reads, device_route) once, so a
reader can tell which regime produced the numbers. It is `{}` if no routed layer
reported the barrier (e.g. a device-route arm — see below).

## Phase / metric glossary

Per layer (aggregated across tokens in `per_layer.<phase>.<layer>`), per token
(aggregated across layers in `per_token`):

| key | delta | nature |
|---|---|---|
| `attn` | attn_end − attn_start | graph-build µs |
| `gate_to_barrier` | routing_barrier_done − moe_start | barrier wait **incl. GPU** (prev gather + attention + gate) + sync |
| `barrier_to_issue` | miss_issue − routing_barrier_done | host route plan/pin (miss layers) |
| `miss_issue_to_ready` | miss_ready − miss_issue | issue→slots-ready; **incl. gather fence on the fenced path** |
| `ready_to_dispatch` | expert_dispatched − miss_ready | ready→switch-returned; **incl. blocking gather eval (fenced)** |
| `post_barrier_host` | expert_dispatched − routing_barrier_done | **pure host after the sync — this is `host_gap`** |
| `combine` | layer_end − expert_dispatched | graph-build µs |
| `moe_total` | layer_end − moe_start | GPU-inclusive |
| `layer_total` | layer_end − layer_start | GPU-inclusive |

Per-token accumulators / counts:
- `miss_wait_total` (ACC) — **pure exposed host SSD read wait**, measured on the
  generation thread inside `iter_ready_misses` (`as_completed`/`future.result`).
- `reconcile_total` (ACC) — gate-oracle prefetch reconcile await (0 if ring off).
- `routing_barrier_total` = Σ `gate_to_barrier`.
- `token_total` — consecutive `token_start` deltas = true inter-token wall (1/tok_s).
- `head_dispatch`, `sample_sync_tail` — head matmul dispatch and trailing sync.
- `miss_layers` / `hit_layers` / `unrouted_layers` — **counts** (keys
  `mean`/`p50`/`p95`, no `_ms`): fenced-miss vs all-hit vs no-barrier
  (device-route / dense-island) layers per token.

### n_semantics (populations differ — compare like with like)

PHASE metrics are timestamp deltas summed per token, so their `n` counts only
tokens with ≥1 layer where **both** endpoints fired (`miss_issue_to_ready.n` =
tokens with ≥1 miss layer). The `*_total` accumulators span **all** recorded
tokens (`n = tokens_recorded`), 0 where absent. The snapshot stamps `n_semantics`
and `phase_semantics` so a reader never conflates the two.

## First reading — window 52 (v1 probe, feat 8f4ac0a53 + probe)

`docs/deepseek-v41/receipts/gpu-windows/window-52/ar-v2-attn-pf0-fanout4-timeline.json`
(the **v1** probe, before this review's fixes) over 256 tokens:

- `token_total` mean **164 ms/tok** (~6.1 tok/s).
- `gate_to_barrier` mean **49 ms/tok** — the `mx.eval(indices)` barrier, **GPU-inclusive**.
- `miss_wait_total` (ACC, pure host SSD wait) mean **55 ms/tok**.
- `reconcile_total` ~13, `ready_to_dispatch` ~19.
- old `host_gap` (attn_end→dispatch) **150 ms/tok** — the artifact this review
  fixed: it swallowed the GPU compute the barrier forces. `switch_config` was
  absent (v1); `overhead.mark_cost` 28 ns (the calibration undercount, now fixed).

**Interpretation:** the ~120 ms host budget splits into ~49 ms barrier round-trip
(GPU + sync, hidden behind the eval) and ~55 ms exposed SSD read wait, plus
reconcile/dispatch host. The v2 probe below reports `post_barrier_host` and
`switch_config` so the barrier (GPU) and the post-barrier host are no longer
conflated, and so a reader knows the fenced gather is inside `ready_to_dispatch`.

## Next window (v2 probe)

Arm `cell16k_ring_v2_attn_hr8` (= `cell16k_ring_v2_attn` +
`MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB=8`, plan target 69.1783), add the launch env:

```
MTPLX_DSV41_DECODE_TIMELINE=1 \
python scripts/deepseek_v41/ab_decode_env_levers.py \
  --arms cell16k_ring_v2_attn_hr8 \
  --context-tokens 16384 --decode-tokens 256 \
  --prompt-ids-file .../window-28b/ar-16k/prompt-ids-deepseek-v41.json \
  --prompt-seed 20260829 --out <window-XX/ar-v2-attn-hr8-timeline.jsonl>
```

The bench sizes the probe to `--decode-tokens` (`configure(n_layers,
max_tokens=steps)`), so a 1024-token run is not truncated at the 512 default.
Confirm `decode_timeline.overhead.within_budget == true` before trusting the
numbers, and read `switch_config.fenced_split_path` before comparing
`ready_to_dispatch`/`miss_issue_to_ready` against `miss_wait_total`.

## Notes / limits

- **DSpark verify forwards carry no block here.** A multi-row forward (prefill, or
  the DSpark K+1 verify) never calls `token_begin` (gated on `input_ids.shape[1]
  == 1`), so it never records — the timeline is AR-decode only. A verify's own
  `mx.eval(indices)` barriers are therefore **not** in these numbers.
- Device-route presets (`MTPLX_DSV41_DEVICE_ROUTE[_PINNED]`) bypass
  `observe_route`, so `barrier_done`/`all_hit`/`miss_*` never fire and
  `switch_config` stays `{}`; those layers surface as `unrouted_layers`.
- The served snapshot memoises the aggregate on `(_NTOK, _MARKS)` (it is called
  twice per request and the aggregation is O(MAXTOK×NL×phases)); `reset` clears it.
- `perf_counter_ns` is never 0 in a real run, so `0.0` is a safe "unset" sentinel;
  tests that script the clock use a nonzero base.
