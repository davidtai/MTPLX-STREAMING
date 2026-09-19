# W37 — DeepSeek-V4.1-Flash decode stage-timing probe

**Task class:** measurement, not a fix. Window 12 measured the streaming decode at
**4.06 tok/s (246 ms/token)**, TTFT 34 s, peak 77.7 GB, and established that decode
is **not** SSD-bandwidth-bound (634 MB/token realized at 2.3 GiB/s against a ~13
GB/s SSD; the queue-depth / fanout / overlap I/O arms all moved ±1%). The earlier
4.5 GB/token cost model is wrong. **40 routing barriers/token** (`mx.eval(indices)`,
one per layer) are confirmed. This probe answers *where the 246 ms/token goes* by
attributing wall time to each decode stage.

Receipts context: `docs/deepseek-v41/receipts/gpu-windows/window-12/` (integration
worktree `.worktrees/deepseek-v41`, read-only).

## Why the existing route-stage probe reads 0.0 ms

`mtplx/expert_route_probe.py` (`MTPLX_ROUTE_STAGE_PROBE`) brackets the streamed
switch (`hot.eval_indices`, `hot.begin_split_route`, `hot.try_all_hit`, …). MLX is
lazy, so a bracket that wraps only **graph construction** exits before the GPU runs
— it *counts* accurately but *times* ~0 ms for every bracket that does not itself
contain an `mx.eval`/blocking read (window 12's stage totals). It cannot tile the
whole decode forward either — it only sees the switch.

## What this probe does

`mtplx/models/deepseek_v41_stage_timing.py` is a **module-level singleton** armed
around the decode loop (`begin()` / `end()`), a **parallel timer** to the route
probe. When armed *and* the current forward is a decode step (`s == 1`), each stage
bracket places an explicit **`mx.eval` fence** on the arrays that stage produces, so
the bracket's wall time is the stage's **dispatch + GPU execution**, not just the
Python encode. Each stage fences its own outputs, so consecutive stages measure
disjoint work and their sum tiles the per-token frame wall.

* **OFF is free and byte-identical.** With no session armed, `stage()` returns a
  shared no-op context manager, `recording()` is `False`, and no array is touched —
  the forward is the shipped path bit-for-bit (gate `test_probe_on_off_logits_
  identical_hc_off`, `test_report_none_and_not_recording_when_off`).
* **Prefill is never fenced.** `Model.enter_forward(s)` arms recording only on
  `s == 1`; a prefill forward (`s > 1`, whose transients are multi-GB) records
  nothing (gate `test_prefill_forward_not_recorded`).
* **Fences inflate absolute time.** An extra host round-trip serialises every
  stage, so a stage-timing pass's tok/s is meaningless — the **ratios** between
  stages are the signal. The clean tok/s pass runs with the probe OFF.

Exposed as `model.stage_timing_report()`; embedded per-arm by
`scripts/deepseek_v41/ab_decode_env_levers.py --stage-timing`.

## Stages (per decode token)

| stage | where | count / token |
|---|---|---|
| `embed` | `Backbone._forward_span` (embed + hc broadcast + pre_mix) | 1 |
| `engram.advance` | `NgramHashState.advance` (rolling n-gram hash, once/step) | 1 (engram attached) |
| `engram.hash` | `EngramV41` per-layer row-id read | n_engram_layers (1 & 14 → 2) |
| `engram.row_fetch` | `EngramV41` byte-budgeted row-cache dequantize (I/O half) | n_engram_layers |
| `engram.apply` | `EngramV41` wkv + q/k gate + additive residual write | n_engram_layers |
| `attn.<mode>` | `Attention.__call__`, split by CSA mode `swa_only`/`full`/`reindex`/`reuse` | Σ = n_layers |
| `hc.premix_sinkhorn` | `DecoderLayer._mixes` + hc-pre collapse + input RMSNorm (attn & ffn) | 2·n_layers |
| `hc.combine` | `_hc_post_impl` (attn HC post + ffn HC post) | 2·n_layers |
| `moe.gate_topk` | `Gate` scoring + top-k **+ routing-barrier fence** (`mx.eval(indices)`) | n_layers |
| `moe.routed_switch` | `switch_mlp` — streamed: miss-I/O wait + gather_qmm hits (barrier already paid) | n_layers |
| `moe.shared_expert` | shared `Expert` (control arm; folded into `routed_switch` under `MTPLX_DSV41_SHARED_OVERLAP`) | n_layers |
| `moe.combine` | weighted routed sum + shared add | n_layers |
| `final_norm` | hc collapse + final RMSNorm | 1 |
| `head` | `lm_head` GEMM | 1 |
| `sample` | argmax + host token round-trip (in the harness decode loop) | 1 |

The report also carries `stage_sum_ms`, `frame_wall_ms` (their reference), per-stage
`mean_ms` / `mean_ms_per_token` / `count`, and — when `MTPLX_ROUTE_STAGE_PROBE` is
also armed — the route probe snapshot under `route_stage` for the switch-internal
breakdown (the confirmed ~40 `hot.eval_indices` barriers/token, and streamed-only
miss-I/O / gather brackets). `--stage-timing` arms the route probe automatically and
clears its counters before the decode loop so `route_stage` reflects that window.

### How the fences are placed

* Each `with _stime.stage(name) as st:` block ends with `st.add(<this stage's output
  arrays>)`; on `__exit__` the probe does `mx.eval(collected)` **only when
  recording**, then books the elapsed ns under `name` and increments its count.
* `moe.gate_topk` fences `indices`: that fence **is** the per-layer routing barrier,
  so the streamed switch's own internal `mx.eval(indices)` becomes a no-op and
  `moe.routed_switch` owns only the subsequent miss-I/O + gather. On the resident
  (test) path there is no internal barrier; the fence attributes the gate compute
  here, keeping the stage split path-independent.
* `frame()` wraps one whole decode iteration (forward + sample) without fencing —
  the inner stages already tile the work — giving `frame_wall`, the denominator the
  per-stage sum should match (measured ratio ≈ 0.98 on the CPU test double).

## Known bias

1. **Fences inflate absolute time** (extra host syncs). Ratios survive; totals do
   not — never quote a stage-timing pass's tok/s.
2. **The eager Hyper-Connection path is forced while recording**
   (`_hc_use_compile` returns `False` under `_stime.recording()`): a compiled tape
   is one opaque call the per-stage fences cannot split, so timing always reflects
   the eager HC path — exactly the shipped `control` arm — regardless of
   `MTPLX_DSV41_HC_COMPILE`. To time the compiled path, measure it separately.
3. **`MTPLX_DSV41_SHARED_OVERLAP` distorts under timing.** The early `moe.gate_topk`
   indices fence serialises the barrier the overlap is designed to hide, and the
   overlap fuses routed + shared into one call (`moe.shared_expert` is then absent).
   Measure the overlap lever's tok/s with the probe OFF.
4. **Attention KV-cache frontier.** `attn.<mode>` fences the attention output, which
   forces the window/compress/index appends it depends on; the compressor frontier
   (`comp_state.raw_kv/raw_score`) retained for the *next* step is forced at the next
   stage/forward, so a hair of that retention cost can land outside the attention
   bracket. Immaterial to the ratio.

## Warm-repeat (`--warm-repeat`)

After the measured cold pass, re-run prefill+decode of the **same** prompt a second
time in the same process. A fresh `model.make_cache()` resets the KV window and
hands a fresh engram-history clone, while the expert-bank LRU (and engram row cache)
stay warm from the cold pass — so the warm pass bounds the **no-miss decode
ceiling**. Reported as `warm_ttft_s` / `warm_prefill_tok_s` / `warm_decode_tok_s` /
`warm_peak_gb`. Greedy decode is deterministic, so the warm token ids must match the
cold pass: `token_ids_match` + both sha256 are **recorded, not asserted** — a
mismatch is reported, never a crash.

## Exact GPU-window command

```
scripts/deepseek_v41/ab_decode_env_levers.py \
  --context-tokens 1024 --decode-tokens 64 \
  --arms control --stage-timing --warm-repeat \
  --memory-limit-gib 82 --out <receipt.jsonl>
```

Runs inside `scripts/deepseek_v41/gpu_window.sh` (GPU flock held, Qwen unloaded,
memory-guarded). The `control` arm's clean tok/s pass is unchanged; the
`stage_timing` and `warm` blocks are appended to the same receipt.

## Caveat at the pinned base (bfd361424)

The compiled Hyper-Connection path has a **pre-existing NameError** at this base
commit: `_hc_mixes_split` (the compiled tape body) calls an undefined
`hc_split_sinkhorn` (should be `_hc_split_sinkhorn`), so any arm arming
`MTPLX_DSV41_HC_COMPILE=1` (`hc_compile`, `all_levers`) crashes on the first
decode/verify forward. It is measured on the clean base (6 of 8 `test_deepseek_v41_
hc_compile.py` gates fail identically without this branch's changes) and is already
fixed on the advanced `feat/deepseek-v41-streaming` tip (dc39b50f). This probe forces
the eager path while recording, so it is unaffected; the `control` window arm never
takes the compiled path.

## Tests (CPU, tiny test-double, no artifact, MLX pinned to CPU)

* `tests/models/test_deepseek_v41_stage_timing.py` — OFF byte-identical + no fences;
  ON exact per-token stage counts + `stage_sum ≈ frame_wall` partition + report
  schema; recording forces eager (`_hc_use_compile` False) and matches plain eager;
  engram hook hash/row_fetch/apply split (fake row cache); prefill not recorded.
* `tests/test_deepseek_v41_ab_env_levers.py` — `--stage-timing` / `--warm-repeat`
  parser + dry-run flow (flags flow into the receipt), route-probe env arming
  wiring, pass-helper signatures.
