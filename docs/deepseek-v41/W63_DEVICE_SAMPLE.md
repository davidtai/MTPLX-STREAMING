# W63 — Device-side sampling AR decode (one-step-lag pipeline) — K32

**Lane:** `MTPLX_DSV41_DEVICE_SAMPLE=1` (default OFF). **Scope:** the AR / greedy
decode loop for the deepseek_v41 lane. **Correctness:** greedy is byte-identical
to the classic argmax loop (proven on the CPU double); sampled is a documented
seed-mapping + narrower-nucleus deviation, default-off.

## 0. The problem (why decode is dispatch-bound)

DSV4.1 streaming decode at 1K is dispatch-bound at ~160 ms/token (stack 6.0
tok/s, [[dsv41-decode-lever-ledger]]). Every token ends with a device→host round
trip: `logits → argmax/sample on device → the token id is read to the host
(mx.eval / .item()) → detokenize/stop check → the id is fed back as the next
input embedding`. That read is a **full GPU drain per token**, and the next
step's graph cannot be encoded until it returns — so the GPU idles across the
host read and re-encode on the critical path of every token.

## 1. What the lane does

`mtplx.models.deepseek_v41_dspark_decode.run_device_sample_decode` keeps the
sampled token **on device** across the whole decode:

- **Token stays on device.** The sampled/greedy id is a lazy `mx.array` fed
  straight into the next forward. The model's embedding lookup consumes it
  directly (`mx.take` on the embedding with the id array) — the id never
  round-trips to the host to become the next input.
- **One-step lag.** The host reads token *t* only after step *t+1*'s forward has
  already been submitted (`mx.async_eval` on the step outputs), so the GPU never
  idles on the host read. Concretely, each iteration: submit `forward(tok_lazy)`
  and its device sample (`async_eval`), *then* materialize the previous step's
  `tok_lazy.item()`, emit it, and only then advance. This is the mlx-lm
  double-buffer pattern (cf. the shipped `MTPLX_AR_PIPELINE` lane), applied to
  the deepseek_v41 lane and extended to greedy.
- **Bounded discarded work.** At a stop token, at `max_tokens`, or on abort, the
  loop has already submitted **exactly one** extra forward (the just-emitted
  token's step) whose sampled successor is discarded. The classic loop breaks
  *before* forwarding its final token; the lag pipeline computes that one step
  and drops it. `run_device_sample_decode` returns this count as its third value
  (`extra_forward_steps`, always ≤ 1); the receipts surface it as
  `device_sample_extra_forwards`. Greedy output is unaffected — the discarded
  step's token is never emitted.

## 2. Correctness

### 2.1 Greedy — byte-identical (hard gate)

`argmax` is deterministic and the forward math is identical (same cache, same
inputs), so the device-side argmax selects the same integer id the classic
`int(mx.argmax(...).item())` loop would, and that same id feeds the next forward
either way. Proven on the CPU double (`tests/models/
test_deepseek_v41_device_sample.py`):

- `run_device_sample_decode` greedy == an independent argmax AR reference,
  **byte-for-byte over 256 tokens**.
- `generate_ar` with `MTPLX_DSV41_DEVICE_SAMPLE=1` == `generate_ar` with it off,
  **byte-for-byte**, streamed deltas == committed tokens, same `finish_reason`.
- Stop handling on the lagged read: the emitted ids equal the classic
  up-to-stop output (the stop token is included exactly once, last), at both the
  model level and the served level.
- `extra_forward_steps == 1` at every length / stop finish (bounded).
- `ab_decode_env_levers._generate` and `bench_standard_shape.bench_one_cell`
  produce byte-identical greedy ids with the pipeline on vs off — so the
  AR-reference byte-identity gate the DSpark lane asserts against still holds.

### 2.2 Sampled — documented deviation (default-off)

The sampled path reuses the **shipped** device shaped sampler
`mtplx.generation._mx_lazy_sample` (temp → top-k → top-p → categorical) — the
same one the qwen4_exp `MTPLX_AR_PIPELINE` lane uses, so **no other model's
sampler is touched** and none is modified. It is **not** token-for-token equal to
the host numpy path, by two deviations:

1. **RNG stream (seed-mapping change).** The device path draws from an
   `mx.random` key seeded from the request seed, not the numpy
   `default_rng(seed)` stream `_sample_from_logits` uses. Same seed → same
   distribution, **different sequence**. (The first token is still host-sampled
   from the prefill logits, so on the served lane token 0 uses the numpy stream
   and the rest use the device stream — the same split the `MTPLX_AR_PIPELINE`
   lane already ships.)
2. **Narrower top-p nucleus.** `_mx_lazy_sample` computes the top-p nucleus over
   the **renormalized top-k softmax**, whereas the host `apply_top_p_top_k`
   computes it over the **full-vocab softmax**. When top-k truncates real tail
   mass, the device nucleus is narrower (measured on the CPU double: 12–18 kept
   vs the host's 17–20).

Both are **conservative**: the device sampled support is a **subset** of the
host top-k / host nucleus support (verified on the CPU double across trials), so
the device never draws a token the host would not — it only ever samples from a
prefix of the host's admissible set. Sampled callers that need the exact host
distribution keep the default (device sample off) and stay on the classic path.
The greedy path — which is the DSV4.1 standard benchmark shape
([[dsv41-standard-benchmark-shape]]: 1,024-token greedy prefill_bench, 256
decode) — is exact.

We do **not** "fix" `_mx_lazy_sample` to match the host nucleus, because it is a
shared function on the qwen4_exp lane's critical path; changing it would touch
another model's path (out of scope) and ship an unvetted device sampler this
CPU-only window cannot validate on the real vocab/dtype.

## 3. Integration + gating

- **Bench loops first** (`--device-sample`, default follows
  `MTPLX_DSV41_DEVICE_SAMPLE`): `scripts/deepseek_v41/ab_decode_env_levers.py`
  (`_generate`) and `scripts/deepseek_v41/bench_standard_shape.py`
  (`bench_one_cell`). Only the real-MLX AR path uses it; the dry-run `_FakeOps`
  double keeps the classic loop. Receipts record `device_sample` and
  `device_sample_extra_forwards`.
- **Served lane** (`mtplx/generation.py` `generate_ar`): a device-sample lane
  block, a sibling of the `MTPLX_AR_PIPELINE` lane, that drains into the shared
  stats/finish tail. It engages **only** when: `MTPLX_DSV41_DEVICE_SAMPLE=1` AND
  `rt.model.model_type == "deepseek_v41"` AND no constraint / repetition-stop /
  loop-guard / thinking-guard / AR-hidden / session final-state capture, AND
  `max_tokens > 1`, AND the sampler is eligible (greedy always; sampled needs
  `top_k > 1`, no presence/frequency penalties). Any other case falls through to
  the existing classic / pipeline path unchanged. Verified on the CPU double: a
  non-deepseek_v41 `model_type` never engages the lane.
- **Final-state capture.** The lane is disabled when `capture_final_state` is
  True (multi-turn session-bank commits) — its extra forward leaves the cache one
  token ahead of the classic loop's invariant, so rather than replicate the
  `MTPLX_AR_PIPELINE` lane's `_lane_final_row` bookkeeping, session-capturing
  requests keep the classic path. Benchmark / single-shot requests
  (`capture_final_state=False`) get the pipeline. If the served
  single-turn-benchmark path is ever what we A/B and it sets
  `capture_final_state`, the follow-up is to thread the discarded final row out
  of `run_device_sample_decode` (it already computes it).

## 4. Expected ms/token saved

The lane removes the **exposed per-token device→host read + next-step graph
re-encode** from the decode critical path (the ~160 ms/token, 6.0 tok/s stack at
1K). The realized delta is a **GPU-window measurement this CPU-only worker did
not take** ([[cpu-heavy-work-voids-flock-windows]], no GPU here). Bracketing:

- The a3b precedent for removing the one exposed decision-sync/cycle was **+1.1%**
  ([[a3b-decode-roundtrip-is-the-lever]]); on the more dispatch-bound DSV4.1
  streaming lane the exposed read is framed as a *full GPU drain per token*, so
  the ceiling is higher.
- **Expected ~5–15 ms/token saved at 1K (≈ +3–9% decode, 6.0 → ~6.2–6.6
  tok/s)** — the exposed host read + encode gap the lag now overlaps with GPU
  compute. Confirm in a paired 1K decode window (KG-d family, one lever/arm,
  `mx.eval` counted, [[dsv41-standard-benchmark-shape]]).

## 5. Companion changes

- **`mlx_buffer_500` arm (K14 re-falsifier).** `MLX_MAX_MB_PER_BUFFER` is now a
  pinned lever key + a receipt field (`mlx_max_mb_per_buffer`), and the arm
  `mlx_buffer_500` sets it to 500 MB (the value the Qwen lane measured +1.6% at).
  **Caveat:** MLX binds `MLX_MAX_MB_PER_BUFFER` **once at Metal init**, so an
  in-process arm switch after init does not rebind the buffer — a genuine A/B
  runs this arm in its **own process** with the value exported before launch
  (the a3b K14 verdict was measured the same way). The arm pins + records it so
  the run is reproducible. The dry-run test covers the pin + receipt.
- **`prefill_best_sel` preset.** `prefill_best_nok28` + `selected_keys` (K30):
  the current best f32 prefill stack (layer-major dense experts + lean pass-cut
  score path + K27 sorted routed gather) with the score-WIDTH lever added, no K28
  kernel. LOSSY vs control (dense fp32 accumulation order + score reassociation),
  task-eval gated like `prefill_best_nok28`.

## 6. Tests (CPU, pinned to `mx.cpu`, tiny seeded model)

- `tests/models/test_deepseek_v41_device_sample.py` — greedy byte-identity
  (model + served, 256 tokens), stop-lag correctness (model + served), length /
  extra-forward bound, eligibility rules, sampled subset-of-host support, the
  non-deepseek_v41 gate, and `device_sample_enabled` env parsing.
- `tests/test_deepseek_v41_ab_env_levers.py` — extended for the `mlx_buffer_500`
  and `prefill_best_sel` arms, the `MLX_MAX_MB_PER_BUFFER` lever pin/clear, the
  `mlx_max_mb_per_buffer` + `device_sample` receipt fields, and the
  `--device-sample` flag (default None → follows env).

## 7. Not done here / follow-ups

- No GPU measurement (CPU-only worker). The §4 estimate needs a paired 1K decode
  window; add a `device_sample` A/B arm's decode tok/s to the ledger then.
- The served `capture_final_state=True` path stays classic (see §3); thread the
  discarded final row out of the helper if a session-capturing served benchmark
  becomes the A/B shape.
- Sampled exact-host-distribution parity would need a full-vocab device nucleus
  (or a host-matching device sampler that does not touch the qwen4_exp
  `_mx_lazy_sample`); out of scope for a default-off greedy-headline lever.
