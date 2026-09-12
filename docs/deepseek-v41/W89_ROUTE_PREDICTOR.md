# W89 — DSV4.1-Flash learned route predictor (feasibility harness + overlap model)

**Job.** Feasibility only, no runtime integration. Build the harness that answers
one question: can a cheap, per-layer *route predictor* — reading the hidden state
one (or two) layers before layer L — name L's top-6 experts well enough to issue
their SSD reads a layer early, so the expert-read time overlaps the current
layer's compute instead of adding to it? W85 showed the compulsory miss floor is
flat against capacity and *temporal* (token t vs t−1, same layer) prediction is
only ~19 % precise; a **learned** predictor is the open question. This window
delivers (1) a GPU-window trace collector, (2) a CPU-only offline
trainer/evaluator, (3) this overlap arithmetic + runtime design sketch. **No
weight, format, or output changes**: the collector is byte-identical to an
unhooked decode (gated), and a prefetch mispredict may only waste a read.

Author: Opus 4.8 worker (`w89/route-predictor`, off `0ba388c2d`). Scripts:
`scripts/deepseek_v41/collect_route_traces.py`,
`scripts/deepseek_v41/train_route_predictor.py`; tests
`tests/models/test_route_predictor.py` (11, green, CPU). The trace itself is a
GPU-window job (a window is running now); the harness is validated on CPU with
`--tiny` on the fake model from `tests/models/test_deepseek_v41_stage_timing.py`.

---

## 0. Headline

> **Even a *perfect* one-layer-ahead route predictor caps the AR-16K decode gain
> at ~+24 % (2.24 → 2.78 tok/s) today, because the expert-read time is only
> I ≈ 87 ms of the T ≈ 447 ms token (≈19 % — exactly W85's "decode is not
> SSD-bound"). Overlap turns the additive `T = C + I` into `max(C, I)`; since
> I < C, the whole prize is I, reached only as the prefetch coverage
> `r = miss_reduction@K → 1`. The predictor's own cost is negligible (~0.24 ms/
> token, <0.1 % of the wall). So the route predictor is a *second-order* lever on
> AR: it becomes first-order only after the compute stack (W80 ring + W71
> barrier-free route) cuts C toward the ~165 ms 1K-forward floor — then
> `C + I = 252 → max = 165 ms`, a **~34 %** cut, and the SSD turns co-binding
> (W82's ~6–8 tok/s wall). On DSpark the prize is larger now (I ≈ 132 ms/accepted
> token, ~37 % of the cycle's per-accepted cost) because the 4-row verify streams
> 1.65 GiB/accepted.** The empirical precision — whether a cheap linear/MLP head
> reaches `r → 1` at prefetch width 8–12 — is what the real trace decides; the
> harness measures exactly that, train on the prefill tail, test on decode.

---

## 1. What the collector records (and why it is exact)

`collect_route_traces.py` runs the standard cell (`--arm cell16k`,
`--memory-limit-gib 60`, `--max-kv 17408`, the window-28b AR-16K prompt ids,
`--prompt-seed 20260829`) and captures, for **every routed layer** and every
token of the **prefill's last 2,048 positions** plus **256 AR-decode tokens**:

| field | tensor | what a predictor does with it |
|---|---|---|
| `router_in(L)` | pre-MoE, post-attention hidden feeding the gate (`MoE.__call__`'s `xf`, `[n,5120]`) | predictor (a): re-run the router (zero overlap; the sanity ceiling) |
| `layer_in(L)` | residual **entering** layer L, mean over the hc copies (`mean(h,axis=2)` — the pre-layer hidden the DSpark head reads) | predictors (b)/(c): read at L−1 / L−2, predict L (one/two layers of overlap) |
| `top6(L)` | router top-6 expert ids `(scores+bias)`-descending — the shipped route | the label |
| `gate(L)` | the routed weights at those ids | route weight (diagnostics) |
| per token | token id, absolute position, phase (0 tail / 1 decode) | predictor (d)'s token feature; the train/test split |

**Capture mechanism.** Two class-level hooks: `MoE.__call__` re-runs the *pure*
gate (`self.gate(xf)` — a linear + top-k, no streaming side effect) to stash the
route ids/weights and the router input; `DecoderLayer.__call__` (the common path
for decode and non-layer-major prefill) records the layer-input residual and the
absolute positions, then combines. The recorded tail forward is issued
`prefill_layer_major=False` so it flows through `_forward_span →
DecoderLayer.__call__` with clean absolute positions (`arange(cache.offset, …)`);
the head prefill is unrecorded and uses the arm's own schedule. Splitting the
prefill and forcing chunk-major on the tail is **route-faithful** — a token's
gate depends only on its own causal hidden, and schedule only reassociates fp
reductions at the argmax-stable ~1e-6 level the cell already carries.

**Non-invasive (proven).** The hooks call the *unchanged* `MoE.__call__` /
`DecoderLayer.__call__` and only *read* intermediates; decode logits are
`np.array_equal` with the hooks enabled vs a clean decode
(`test_capture_is_byte_identical_to_unhooked_decode`). Nothing changes a weight,
a format, or an output.

**Storage.** Hiddens are bf16 (the runtime dtype). numpy 2.x cannot buffer a
bf16 array and this box has no `ml_dtypes`, so each hidden is stored as the
**exact** bf16 bit pattern in uint16 (`bf16_bits`/`bf16_to_f32` round-trip is a
tested fixed point). Size at the standard cell:
`2,304 tok × 40 layers × 2 hiddens × 5,120 × 2 B ≈ 1.9 GB` — held in RAM as the
final uint16 (no float32 shadow), written once. `--hidden-stride N` subsamples
the hidden axis if a run must fit under `--max-bytes` (default 3.0 GB; the
collector aborts before capture if the estimate exceeds it).

### Exact GPU-window command (run inside the guarded window; NOT run by this worker)

```
scripts/deepseek_v41/gpu_window.sh \
  env PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \
  scripts/deepseek_v41/collect_route_traces.py \
    --arm cell16k --memory-limit-gib 60 --max-kv 17408 \
    --context-tokens 16384 --prefill-tail 2048 --decode-tokens 256 \
    --prompt-ids-file docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/prompt-ids-deepseek-v41.json \
    --prompt-seed 20260829 \
    --out docs/deepseek-v41/receipts/gpu-windows/window-89/route-traces
```

(`collect_route_traces.py --print-window-command` echoes this verbatim.) The
loader defaults `--expert-profile deepseek-v41-mxfp4-75` (transient_slots 48),
i.e. the **served plan** — the routes are capacity-independent, so the profile
only sets the (unused-for-routing) cache size.

---

## 2. What the trainer measures

`train_route_predictor.py` (CPU-only, one layer at a time so the working set
stays small) trains four predictors per layer L, **train on the prefill-tail
rows, test on the held-out decode rows**, and reports precision/recall@6 and
miss-reduction@{6,8,12}:

- **(a) `router_in(L) → top6(L)`** — re-runs the router. Zero overlap; the
  **sanity ceiling**. On a genuinely linear router with enough samples the ridge
  head recovers it (`test_ridge_recovers_a_recoverable_linear_router`, prec 0.96);
  a shortfall on the real trace quantifies the finite-sample/nonlinearity gap
  (2,048 tail rows vs a 5,120-wide gate is under-determined — a real finding to
  read off (a)).
- **(b) `layer_in(L−1) → top6(L)`** — **one layer ahead, the useful one**: the
  read overlaps a full layer of attention+MoE compute.
- **(c) `layer_in(L−2) → top6(L)`** — two layers ahead (more overlap, expected
  lower precision — the overlap/precision trade the runtime tunes).
- **(d) cheap features** — previous layer's routed ids (multi-hot) + the token id
  (one-hot over the train vocab; the learned column is that token's embedding
  into expert-logit space; unseen decode tokens fall back to the route features).
  The "can we skip the hidden state entirely" floor.

**Predictor class.** Default is ridge (closed-form multi-output least squares to
the multi-hot route) — fast, deterministic, and it *is* the linear head the task
asks for; `--model logistic` / `--model mlp` (one 512-hidden layer) train by GD
on mlx-cpu for comparison. Model sizes are tiny per the brief (a 5120→384 linear
is ~2 M params; the 40-layer bank is ~80 M floats ≈ 320 MB, processed one layer
at a time). **Precision@k == recall@k** at k = top_k (both sets are size top_k);
**miss_reduction@K** = mean over decode tokens of `|true_top6 ∩ predicted_topK| /
top_k` — the fraction of that layer's compulsory misses already prefetched at
width K.

> The `--tiny` path self-generates a synthetic trace from the fake model and runs
> the whole pipeline in seconds; its absolute numbers are a smoke test (random
> weights, 8 experts), **not** a feasibility read. The feasibility numbers come
> from the real-trace window.

---

## 3. The overlap arithmetic

**Model (the problem's framing).** A layer's experts are known only after that
layer's attention, so without a predictor the per-token cost is additive over
layers:

```
T_today = Σ_L (compute_L + io_L) = C + I
```

with `C = Σ compute_L` (attention + MoE compute + dispatch, GPU-bound) and
`I = Σ io_L`, `io_L = misses_L · record / bandwidth` (the flat-out SSD read time
for layer L's cold experts). A one-layer-ahead predictor issues layer L's reads
during layer L−1, so the fraction it correctly prefetches (`r =
miss_reduction@K`, the recall of L's true experts by the width-K prefetch)
overlaps compute; the mispredicted `(1−r)` are known only after L's attention and
stay serial:

```
T_overlap(r) ≈ C + (1−r)·I          (when io fits under compute, r·I ≤ C, per layer)
             → max(C, I)   as r → 1
```

So overlap converts `C + I` into `max(C, I)`; the entire prize is `r·I`, and the
**SSD wall becomes `max(C, I)` only as `r → 1`**.

**AR-16K today** (measured, W82/W85): record 17.93 MiB, 61.9 misses/token →
`I = 1.084 GiB / 12.5 GiB/s = 86.7 ms`; the unfenced wall is 447 ms/token
(2.237 tok/s), so under the additive reading `C = 360 ms`. Because `I < C`, io
fully hides at `r = 1`:

| prefetch coverage r = missRed@K | token T = C+(1−r)I | tok/s | Δ vs today |
|---:|---:|---:|---:|
| 0.0 (today) | 447.0 ms | 2.237 | — |
| 0.3 | 421.0 ms | 2.375 | +6.2 % |
| 0.5 | 403.6 ms | 2.477 | +10.7 % |
| 0.7 | 386.3 ms | 2.589 | +15.7 % |
| 0.9 | 369.0 ms | 2.710 | +21.2 % |
| **1.0** | **360.3 ms** | **2.776** | **+24.1 %** (= max(C,I)) |

The predictor's **own cost** is a single 5120×384 matvec per layer ≈ 3.9 MB
read/layer × 40 = 157 MB/token, DRAM-bound at ~614 GB/s ≈ **0.24 ms/token**
(<0.1 % of the wall); an MLP (5120→512→384) ≈ 0.37 ms. Negligible either way — it
never has to *pay for itself in compute*, only prefetch correctly.

**What precision is needed for `max(C, I)`.** The wall reaches `max(C, I)` only
when the exposed io `(1−r)·I → 0`, i.e. **`miss_reduction@K → 1`**: the width-K
one-layer-ahead prefetch must contain essentially *all* 6 true experts. This is a
**recall@K** target, so widening K (prefetch 8 or 12 to cover the top-6) is the
lever — at the cost of `(K−6)` wasted reads/layer/token when the extra slots
miss. The gate here is `miss_reduction@12`: if even a width-12 prefetch a layer
ahead cannot cover the true 6 (i.e. it stays near W85's 19 % temporal floor), the
lever is dead; if a learned head lifts it toward ~1, AR gains up to the +24 %
above and DSpark more (below). **This single number is the feasibility verdict,
and it is exactly the trainer's headline column.**

**Per-layer refinement.** The aggregate hides that some layers may have
`compute_L < io_L` (miss-heavy, cheap-compute layers); their io cannot fully hide
even at r = 1 and they set a residual floor. The trainer's *worst-5-layers by
miss_reduction* surfaces the layers where the predictor is weakest — the same
layers W85 flagged as diffuse (0–7, 37–39). A runtime would prefetch two layers
ahead (predictor c) for those to buy a wider overlap window.

**DSpark.** 94 misses/accepted × 18.0 MiB = 1.652 GiB → `I = 132 ms/accepted`.
Against a per-accepted cost of ~250–350 ms (W82 cycle model), io is ~37 % — a
**larger** prize than AR, because the 4-row verify streams 1.45× AR's bytes per
accepted token. So if the predictor works, it helps DSpark more; the harness
traces AR (the clean per-(layer,token) regime), and the same predictor weights
apply to the verify's per-row routes (a follow-on window would trace the verify
union directly).

**Why this is second-order on AR today, first-order later.** W85/W82 already
proved AR-16K is compute-bound (io 19 % of the wall, realized SSD BW 2.1 GB/s ≪
12.5 GiB/s). So the +24 % ceiling is real but modest **until the compute stack
lands**: with C recovered toward the 1K-forward floor (~165 ms via W80 ring +
W71 barrier-free route), `C + I = 252 ms → max(C, I) = 165 ms`, a **~34 %** cut,
and the SSD becomes co-binding exactly where W82 put the ~6–8 tok/s wall. The
route predictor is the lever that keeps the SSD *off* the critical path once
compute stops hiding it — it should be built and measured **alongside** W80/W71,
not before.

---

## 4. Runtime design sketch (not built this window)

A pure cache-warming prefetch; the MoE gather always uses the **real** indices,
so outputs are bitwise-identical and a mispredict only wastes a read.

1. **Predict at L−1.** After layer L−1's attention (or at its entry, for a
   two-ahead predictor reading `layer_in(L−1)`), run the per-layer predictor head
   → width-K expert ids for layer L. One matvec (~6 µs/layer); off the critical
   path (it overlaps L−1's MoE).
2. **Speculative loads into a bounded tier.** Enqueue the K predicted experts as
   **speculative** reads into the streaming pool, drawn from a small dedicated
   speculative-slot budget (e.g. ≤ K per in-flight layer, capped well under the
   transient pool). Speculative reads have **lower queue priority than demand
   reads** and **never evict a resident/persistent slot or a real in-flight
   read** — so speculation can only consume otherwise-idle SSD bandwidth and
   spare slots, never displace a needed expert.
3. **Reconcile at L.** When layer L's real route resolves (after L's attention):
   real ∩ predicted are already resident/in-flight (the overlapped hits); real \
   predicted are cold and issued now as demand reads (identical to today —
   **never block waiting on the predictor**); predicted \ real are wasted
   speculative reads, evicted at lowest priority.
4. **Correctness invariant.** The gather reads the real `indices`; the predictor
   only pre-warms the cache. Therefore output is byte-identical and a mispredict
   costs at most one wasted 18 MB read (bounded by the speculative-slot budget) —
   satisfying "a prefetch mispredict may only waste a read, never change a
   result." Demand reads preempt speculative ones in the SSD queue, so a wrong
   prediction cannot delay a correct one.
5. **Depth knob.** One-ahead (predictor b) maximises precision; two-ahead
   (predictor c) maximises the overlap window for miss-heavy/cheap-compute layers
   at some precision cost. The runtime picks per-layer from the measured
   miss_reduction table (worst-5 layers → two-ahead).

**Gate before any integration:** the real-trace `miss_reduction@{8,12}` for
predictor (b). If it clears the additive table's r needed for a worthwhile cut
(and it must beat W85's 19 % temporal baseline by a wide margin), integrate
**stacked on W80/W71**; if it does not, the predictor is dead for the same reason
temporal prefetch was, and the residency verdict (W85) stands unchanged.

---

## 5. Deliverables + provenance

| artifact | path |
|---|---|
| trace collector (GPU window; `--tiny` CPU validation) | `scripts/deepseek_v41/collect_route_traces.py` |
| offline trainer/evaluator (CPU; `--tiny` self-generates) | `scripts/deepseek_v41/train_route_predictor.py` |
| tests (both scripts; 11, CPU, `nice -n 19`) | `tests/models/test_route_predictor.py` |
| this analysis | `docs/deepseek-v41/W89_ROUTE_PREDICTOR.md` |

Inputs: `W82_CYCLE_MODEL.md` (cost model, AR wall, DSpark verify bytes),
`../dsv41-w85/docs/deepseek-v41/W85_RESIDENCY_PROGRAM.md` (compulsory miss floor,
19 % temporal predictability, per-layer diffuse layers). Numbers marked
"measured" are from those receipts; the overlap table is arithmetic over them;
the predictor precision is **pending the real-trace window** (the harness is the
instrument). CPU-validated on the fake model; no GPU, no artifact load, ≤1.5 GB.
