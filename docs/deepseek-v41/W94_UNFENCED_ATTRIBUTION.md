# W94 — Unfenced whole-token frame-wall attribution (DeepSeek-V4.1-Flash decode)

Author: Opus 4.8 worker (window 94). Analysis + harness; no GPU run from this file.

## Why

The fenced per-stage census (`mtplx/models/deepseek_v41_stage_timing.py`, driven by the W37
`--stage-timing` decode pass and by `metal_decode_attn_bisect.py --in-model`) brackets each stage
with `mx.eval`. MLX is lazy and async, so a bracket that forces its stage's arrays pays a **GPU
pipeline drain + refill** — a host round-trip that serialises the stage. That makes the bracket wall
= dispatch **plus a full pipeline stall**, i.e. **latency, not compute**.

Measured symptom (windows 36–37 in-model census):

- Every attention bracket reads **~8 ms/layer in EVERY configuration** — 1K and 16K, selected-keys on
  and off, hot and after a 180 s cooldown, at GPU clocks from 700 to 1,000 MHz — while the **same
  kernels cost ~2 ms isolated** (the empty-process microbench). The ~6 ms gap is mode-independent: it
  is the fence's pipeline stall, not attention work.
- The fenced frame wall is ~**700 ms/token** at 16K (`window-36/in-model-16k-cool0`), but the served
  AR loop pays only ~**460 ms/token** (`window-37/ar-ring-ref` = 2.17 tok/s). The fences add ~50%.

So the fenced absolute per-stage ms cannot be summed or quoted as in-situ cost. Only the **ratios**
between stages are meaningful. (This is now warned in the stage-timing module docstring and in
`KERNEL_LEDGER.md` §4a.)

## What the unfenced mode measures

`scripts/deepseek_v41/metal_decode_attn_bisect.py --in-model --unfenced` runs the **production decode
loop** (the only `mx.eval` per token is the sampler / next-token id — exactly
`ab_decode_env_levers._generate`'s classic argmax loop) with **no stage recorder armed and no
per-stage fences**, and times the **whole-token frame wall**. It then stubs components out and reads
the frame wall again. `full − stubbed` is the component's TRUE in-situ cost — including whatever
latency it *causes* (the DVFS downclock in the sync gap, the serialized SSD wait) but **excluding**
probe latency, because there is no probe.

Per pass it records: mean and median ms/token over the decode steps (default 64; the first 8 excluded
as warmup), tok/s (from the post-warmup mean), macmon utilization (GPU busy% / MHz / power via the
W90 sampler under `--utilization`), and the decode-scoped expert-streaming counters (hits / misses /
bytes per token, from `serve_stream_counters` on the bare runtime, working after W87).

### The five passes

| # | key | stub | isolates |
|---|-----|------|----------|
| 1 | `full` | none | the served decode (its token sha is the served-path comparison) |
| 2 | `expert_stub` | routed switch → zeros, **routing barrier KEPT** (`_ZeroSwitch(keep_routing_barrier=True)`: still `mx.eval(indices)` per layer, but skips the SSD gather + `gather_qmm`) | (1)−(2) = the switch's **SSD/gather** cost |
| 3 | `attn_stub` | attention → zeros (KV offset still advances) | (1)−(3) = **attention** incl. any latency it causes |
| 4 | `expert_stub_nobarrier` | routed switch → zeros, **barrier REMOVED** (`keep_routing_barrier=False`: the stub never touches `indices` on host) | (2)−(4) = the **~40 host syncs** alone; (1)−(4) = the whole switch |
| 5 | `small_stages_floor` | attention zeros **and** switch stubbed no-barrier | (5) = the **floor** of the small per-layer stages (norms / HC / Sinkhorn / gate / shared / combine) + head + sample |

Why pass 2 keeps the barrier and pass 4 drops it: the real streamed switch pays a per-layer host
round-trip — `mx.eval(indices)` (`mtplx/models/expert_mlx.py`, the `hot.eval_indices` barrier the
ledger prices at ~40/token) — to read the route before it can stream. A plain zeros stub never
touches `indices` on host, so it silently drops that barrier *and* the gather together. To split
them, pass 2 keeps the barrier (so `full − 2` is the SSD/gather cost, which should track
misses × bytes / bandwidth) and pass 4 drops it (so `2 − 4` is the barrier — the ~40 host syncs —
alone). The default `_ZeroSwitch()` (barrier off) keeps the fenced 3-pass mode byte-for-byte
unchanged; that mode already owns the barrier at its `moe.gate_topk` fence.

### Derived attribution (identities)

```
switch total          = (1) − (4)
  of which SSD-bound   = (1) − (2)          [cross-checked vs bytes/tok ÷ SSD BW]
  of which sync/barrier= (2) − (4)          (~40 host syncs/token)
attention             = (1) − (3)
small stages floor    = (5)
sum of parts          = attention + switch total + small stages floor
unattributed residual = full − sum of parts     (the interaction/overlap; 0 under additivity)
```

By construction `switch total == SSD-bound + sync/barrier` and
`sum of parts == attention + switch total + floor` exactly; the residual is the only free quantity and
measures latency that appears only when the components coexist. The SSD-bound estimate is
`bytes_read_per_token ÷ --ssd-bandwidth-gibs` (default 4.4 GiB/s, the measured M5 Max SSD threshold);
the receipt carries `misses_per_token` and `bytes_read_per_token` so any bandwidth can be re-applied.

### eval(indices) time, measured directly (pass 1)

Pass (1) also arms the route-stage probe over its decode loop, so the receipt carries the
`hot.eval_indices` barrier time **directly** (`route_probe_sums_ns` / `route_probe_counts`, surfaced as
`attribution.eval_indices_ms_per_token_measured`). The probe's `bracket` only wraps the `mx.eval(indices)`
the production switch already runs with a `perf_counter` — it adds **no** fence and does not force the
eager path — so the frame wall is unperturbed. This is an independent cross-check of the `(2)-(4)`
sync/barrier delta.

The counters are cleared **immediately before** the pass's before-snapshot and nowhere between it and
the after-snapshot, so the recorded sums/counts are the exact decode deltas (baseline 0, never
negative). This is the same discipline as the companion fix in `ab_decode_env_levers.py::_generate`:
that function used to clear the route probe *after* its `after_prefill` snapshot, so the decode
`route_probe_sums_ns` delta (`end - after_prefill`) subtracted the prefill accumulation and went
**negative** for prefill-heavy stages (window-37 `ar-ring-ref` had `hot.begin_split_route` sum
`-3.8e8`), making the "≥315 ms of the token inside `mx.eval(indices)`" figure only a lower bound. With
the clear moved before the before-snapshot, that delta is exact.

## Exact GPU-window command (16,384-token cell)

60 GiB plan, `--max-kv 17408`, `cell16k_ring` arm — the standard 16K cell. Run through the flock
(`scripts/deepseek_v41/gpu_window.sh`) like every Metal exec on this box; `$PY` = the venv python3,
`$MODEL` = the streaming artifact, `$OUT` = the window receipt dir:

```bash
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-XX
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
mkdir -p "$OUT"; cd "$WT"
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
  export PYTHONPATH=$WT
  nice -n 19 $PY scripts/deepseek_v41/metal_decode_attn_bisect.py \
    --in-model --unfenced --gpu --model $MODEL --arms cell16k_ring \
    --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408 \
    --in-model-steps 64 --utilization \
    --out $OUT/in-model-16k-unfenced.json"
```

No `--prompt-ids-file`: the standard-cell builder resolves the same 16,384-token prompt the
`cell16k_ring` AR reference uses, so pass (1)'s `token_ids_sha256` (1 prefill + 64 decode ids) is
comparable to an ab `cell16k_ring` receipt run at `--decode-tokens 64`.

## Receipt schema (`in_model_unfenced`)

```
worker "W94" · mode "in_model_unfenced" · device · tiny · arm · steps · warmup_steps
ssd_bandwidth_gibs · dims · prompt · memory{active_end_gib, peak_gib}
utilization · cooldown · token_ids_sha256 · n_token_ids · token_ids   # pass (1) hoisted
route_probe_counts · route_probe_sums_ns                              # pass (1), exact decode deltas
passes.<key>.summary  { mean_ms_per_token, median_ms_per_token, tok_s, steps,
                        warmup_steps, measured_steps, decode_wall_s, decode_wall_tok_s,
                        ttft_s, gpu_busy_pct, gpu_freq_mhz, gpu_power_w,
                        misses_per_token, bytes_read_per_token, records_streamed_per_token,
                        hit_rate, ssd_bound_estimate_ms,
                        eval_indices_ms_per_token, eval_indices_barriers_per_token }
passes.<key>.{ utilization, cooldown, token_ids_sha256, n_token_ids, token_ids,
               route_probe_counts, route_probe_sums_ns }   # route_probe: pass (1) only, else null
attribution { full, attention, switch_total, switch_ssd_bound, switch_sync_barrier,
              eval_indices_ms_per_token_measured, eval_indices_barriers_per_token,
              small_stages_floor, sum_of_parts, unattributed_residual,
              ssd_bound_independent_estimate_ms, misses_per_token, bytes_read_per_token,
              hit_rate, ssd_bandwidth_gibs }   (all …_ms_per_token)
```

## Validation

`--tiny` runs all five passes on a fake CPU model (no artifact, no GPU): all passes complete, the
table prints, the JSON is written, the attribution identities hold, and the barrier flag gates the
per-layer `mx.eval(indices)`. Covered by `tests/models/test_metal_decode_attn_bisect.py`
(`test_unfenced_*`, `test_zeroswitch_*`, `test_apply_stub_barrier_flag_propagates`). CPU magnitudes are
meaningless (the CPU backend does not reproduce Metal latency); the GPU window measures the real
in-situ frame walls.

The companion `_generate` route-probe delta fix is covered by
`tests/test_deepseek_v41_ab_env_levers.py::test_generate_route_probe_delta_is_exact_and_non_negative`:
a fake model bumps a route counter heavily on prefill and lightly per decode step; the test asserts the
`after_prefill` route-probe baseline is empty (the clear ran before it) and the decode delta equals the
direct decode count and is non-negative for every stage.
