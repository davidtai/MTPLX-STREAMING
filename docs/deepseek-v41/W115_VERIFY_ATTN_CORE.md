# W115 — DSpark verify attention CORE (honest A/B) + verify-scoped engagement

Arm: `cell16k_ring_v2_draft_attn_eager` = `cell16k_ring_v2_draft_attn` with the verify
core K29 pinned OFF. Branch: `w115/verify-attn-fastpath` off `int/w97f-lanes` (7ca25d61a).

## 0. Verdict / correction up front

An earlier draft of W115 added a `MTPLX_DSV41_VERIFY_ATTN_FASTPATH` lever to "auto-arm"
the fused decode core for the K+1 verify, on the premise that the verify was running the
**eager** SDPA core. **That premise is false** (red-team). The DSpark lane arms the
fused decode core for the verify **by default**:

- `ab_decode_env_levers._run_arm` does `os.environ.setdefault(K29="1", K30="1")` from
  `dspark_decode_kernel_env_defaults()` for **every** `--decode-mode dspark` arm
  (`ab_decode_env_levers.py:3690-3695`), AFTER `_apply_arm_env`. The served lane's
  `arm_dspark_decode_kernels()` does the same.
- So `cell16k_ring_v2_draft_attn` runs the verify through the **K29 fused decode
  kernel** — window-43 shows `arm_env.MTPLX_DSV41_DECODE_ATTN_KERNEL="1"` and
  `decode_attn_kernel_engagement.calls>0`.
- `_decode_attn_kernel_use` short-circuits on `_resolve_decode_attn_kernel()` being
  True, so any additional "fast path" gate is unreachable, the compile core is
  unreachable on GPU, and `vfast` vs `draft_attn` would be a **null A/B**. The auto-arm
  also re-armed K29 under an explicit `DECODE_ATTN_KERNEL=0` / `DSPARK_VERIFY_K29=0`
  opt-out, and its census counted a "call" regardless of whether any gate flipped
  (false proof). The fastpath lever + `vfast` arm were dropped.

The verify's ~M× attention cost is **structural** — per-row gathered sparse cores, a
dispatch/fence problem, NOT bandwidth (§2 corrects the earlier byte arithmetic by ~3
orders) — not a missing lever (see §1/§2). K29 itself is shelved (W60: −38% vs eager at
M=1, parity ~1e-3), so whether K29 even helps the *wider* verify is an open question —
exactly the A/B `MTPLX_DSV41_DSPARK_VERIFY_K29` was built for. W115 ships that as a
first-class arm plus the engagement plumbing to read it. NOTE the arm is whole-arm:
`DECODE_ATTN_KERNEL` also flips the draft + AR core, so only `verify_ms` isolates the
verify (§3/§5).

## 1. Corrected as-is — the K+1 verify runs the K29 gathered core, ~M× is structural

Verify forward: `model([[primary,d1..dK]])` (`deepseek_v41_dspark_decode.py:822`),
`b=1, s=K+1`, under `attention_phase("decode_verify")` + `expert_routing_phase(DECODE)`.
Every backbone layer runs `Attention._attend` (`deepseek_v41.py:1412`) → (SELECTED_KEYS
on by setdefault) `_sparse_attend_selected` (`:1130`):

1. `_window_selected_idx(positions,…)` (`:1104`) builds a **per-query** window index set
   — for verify positions `[T,…,T+K]`, query `i` selects window rows `<= T+i` (intra-block
   causality is free).
2. `_gather_rows(window_all, win_idx, …)` (`:1169`) materialises `kvg_win [b, s, W, hd]`
   — a **separate gather per query row** (`s = K+1` copies of ~W keys, windows overlapping
   heavily), concatenated with the selected compressed rows into `KVg [b, s, k, hd]`
   (`:1178`), `k = window + index_topk`.
3. The core (`:1188`): with K29 armed (the default) and a GPU, `_decode_attn_kernel_use(q)`
   is True and the gathered operand goes to the K29 kernel reshaped to `[b*s,1,H,hd]` /
   `[b*s,k,hd]` — **each of the K+1 rows is its own S=1 batch with its own gathered KV**.
   One kernel launch, but the *operand* is `(K+1) × k × hd` and the KV read is not shared
   across rows. Without K29 the eager block below runs the same `[b,s,H,k]` einsums.

So the per-layer verify work — the gather in (2) and the score/softmax/PV over `[s,H,k]`
— scales ~linearly in `s = K+1`: window-43 `verify_stage_timing` shows `attn.reuse`
8.5 ms/call at K+1 vs ~1.7 ms at M=1, `attn.full`/`reindex`/`swa_only` similar, Σ ≈ 372
ms/cycle. The projections do NOT scale that way (W101 fused kernels are row-generic —
W105 priced the five projections at M=6 = 415 µs/layer mxfp8). **The ~M× is the gathered
sparse core, and it is structural**: K29 makes it one *launch*, not one *read*.

(The window-43 receipt's `fused_proj_engagement.rows = 10240 = 256×40` is the **AR pass
only** — the bench reads that counter after `_generate` and never re-reads it for the
DSpark pass. A counter-plumbing gap, fixed in §3 — not evidence the verify projections
ran eager.)

## 2. The cost is dispatch/fence overhead, NOT bandwidth (byte floor, corrected)

An earlier draft called read-amplification "the lever". **That is wrong by ~3 orders.**
The real config (`ModelArgs`, `deepseek_v41.py:458/477/486`): `head_dim = 512` (the KV
latent `hd`, no `kv_lora_rank`), `sliding_window = 128`, `index_topk = 512`, so the
gathered operand is `k = window 128 + index_topk 512 = 640` keys of `hd = 512`. MLA has
one KV latent, and `_sparse_attend_selected` passes the **same** `KVg` buffer as both K
and V (`deepseek_v41.py:1191-1194`), so the K+1 verify reads, per cycle:

```
bytes ≈ k × hd × 2 B(bf16) × ROWS × 40 layers
      = 640 × 512 × 2 × 6 × 40  ≈ 157 MB   (≈ 315 MB if the kernel reads KVg twice)
```

At the box's ~500 GB/s realized decode bandwidth ([[test-machines-bandwidth-file]]) that
is **~0.3–0.6 ms/cycle** — against the **372 ms/cycle** measured. Bandwidth is 600–1200×
too small to explain the cost: the verify attention is **dispatch / per-row gather /
fence-overhead bound**, not read-bound. The ~M× is the per-row structure — 6 rows ≈ 6×
the M=1 in-model attention chain (window-37: ~1.6 ms/layer at M=1 → ~9.3 ms/layer at
rows=6, ≈ 372 ms/cycle over 40 layers), the dozens of tiny per-row gather / reshape /
kernel / softmax / out-proj ops the GPU never saturates ([[b1-decode-dispatch-removal-
hides]]).

So the real work item is a **dispatch-count** target, not a bandwidth one: **one launch
per layer that handles all K+1 rows with the gathers INSIDE the kernel** — a
`[K+1, H, hd]` query set against a shared `[k_union, hd]` KV tile (the K+1 rows' selected
windows overlap by `W − K`, so `k_union ≈ k + K`, not `(K+1)·k`), one `[K+1, k_union]`
valid/causal mask among the block rows (positions increase within the block), sink-softmax
and PV in a single tiled launch, no `s` separate `_gather_rows` copies and no
`[b*s,k,hd]` re-materialisation. It collapses the per-row op chain, not the bytes.

### 2b. Next probe (W116, step 1): fenced per-sub-op census at rows=6 vs rows=1

Before writing any kernel, measure **which sub-op scales with rows**. Run one verify
layer under a fenced (mx.eval-per-sub-op) census at `rows = 6` and `rows = 1`, breaking
`_attend`/`_sparse_attend_selected` into: `_window_selected_idx`, `_gather_rows` (the
per-row gather), the `[b*s,k,hd]` reshape, the K29 kernel launch (or the eager
einsum/softmax/PV), and the out-proj. The sub-op whose ms grows ~linearly in rows is the
dispatch target the §2 kernel must fold in; a sub-op flat in rows is not worth touching.
This replaces the (wrong) "~372 → ~85–110 ms" bandwidth projection with a measured
per-sub-op scaling curve.

## 3. Ships in W115

**Arm `cell16k_ring_v2_draft_attn_eager`** = `cell16k_ring_v2_draft_attn` with:
`decode_attn_kernel="0"` (the RUNTIME knob — an explicit "0" beats the `setdefault`,
which only fills unset keys; the preset must pin "0", not None, or `_apply_arm_env` pops
it and the setdefault re-arms "1") **and** `dspark_verify_k29="0"` (drops K29 from
`dspark_decode_kernel_env_defaults()` entirely). Both pinned so nothing re-arms it.
Everything else identical (SELECTED_KEYS + fused proj + lean casts + wo_a cache all on).
New env constant `MTPLX_DSV41_DSPARK_VERIFY_K29` in `ALL_LEVER_ENVS`, `_preset`, and
`openai._DSV41_LEVER_ENV_KEYS` (served-log superset). `_rounding_class_keys` now treats an
explicit OFF value ("0"/"off"/…) as unset, so pinning K29="0" is not mis-credited as a
rounding reason (both arms stay rounding-class via fused proj + bf16 draft head).

**SCOPE CAVEAT — `DECODE_ATTN_KERNEL` is whole-arm, not verify-only.** The same gate
(`_decode_attn_kernel_use`) governs the DSpark **draft** attention
(`deepseek_v41_dspark.py:591`) and the **AR reference** decode (M=1), so this arm flips
K29 for draft + AR + verify at once. W60: K29 is −38% vs eager at M=1, so the eager arm's
draft may be *faster* and its headline tok/s / `stats.draft_ms` are **confounded** — only
`dspark.per_cycle_ms.verify_ms` (and `verify_stage_timing`'s attn stages) isolate the
verify core (see §5).

**Verify-scoped engagement plumbing** (`_reset_dspark_engagement_counters` /
`_capture_dspark_engagement`): the dspark headline pass zeros and re-reads the
decode-core / fused-proj / attn-core-compile counters, and the `dspark` receipt block
carries `decode_attn_kernel_engagement`, `attn_core_compile_engagement`, and
**`fused_proj_engagement` counting the K+1 VERIFY rows** — closing the window-43
"AR only" gap. The top-level blocks stay AR-scoped.

## 4. Tests (`tests/test_deepseek_v41_w115_verify_attn_core.py`, CPU, no Metal)

9 tests, CPU-pinned, no model load, <1.5 GB RSS:

- `test_eager_arm_turns_verify_core_off_under_dspark_defaults` — under the real dspark
  setdefault, `draft_attn` → `DECODE_ATTN_KERNEL="1"`, `_decode_attn_kernel_use(q6)` True
  (spied GPU); `eager` → `"0"`, `_resolve_decode_attn_kernel()` False,
  `_decode_attn_kernel_use(q6)`/`(q1)` False; SELECTED_KEYS stays "1".
- `test_eager_arm_flips_draft_and_ar_too` — the whole-arm scope: on `draft_attn` K29 is
  on at M=1 (draft/AR) AND K+1 (verify) and OFF at 9 rows (>8 cap); on `eager` it is off
  for all three — so headline/draft_ms are confounded and a depth-8 control keeps
  `calls>0` from the M=1 draft while its verify is eager (read rows, run at depth ≤ 7).
- `test_dspark_defaults_drop_k29_only_when_verify_k29_off` — `DSPARK_VERIFY_K29=0` drops
  K29 (K30 stays) from the lane defaults.
- `test_eager_preset_pins_zero_not_none` — the eager preset pins the runtime knob "0"
  (not None) so the setdefault can't re-arm; the base arm leaves it None.
- `test_eager_arm_rounding_class_does_not_credit_off_kernel`,
  `test_off_values_not_counted_generally` — an OFF-pinned lever is not a rounding reason;
  both arms still rounding-class via fused proj.
- `test_off_values_are_off_in_every_runtime_resolver` — `_LEVER_OFF_VALUES` is the
  INTERSECTION of OFF across every rounding-class runtime resolver; guards the "none"
  removal (the `_env_truthy` / draft-head resolvers treat "none" as ON).
- `test_verify_k29_env_in_lever_lists` — `DSPARK_VERIFY_K29` in `ALL_LEVER_ENVS` + the
  served superset; every preset pins it.
- `test_engagement_reset_and_capture` — the dspark engagement helpers reset+capture
  `fused_proj`/`decode_attn_kernel`/`attn_core_compile` (verify rows counted).

```
$ .venv/bin/python3 -m pytest tests/test_deepseek_v41_w115_verify_attn_core.py
======================== 9 passed, 2 warnings in ~1s ========================
```

Existing guards still green (CPU): `test_deepseek_v41_ab_env_levers.py`,
`test_deepseek_v41_lever_child_env_w46.py` (ALL_LEVER_ENVS ⊆ served keys),
`test_deepseek_v41_w110_record_hash.py`, `test_deepseek_v41_bench_scripts.py`,
`tests/models/test_deepseek_v41_dspark_decode.py`, `decode_attn_kernel`, `selected_keys`.

## 5. Window-45 A/B — do NOT run here (orchestrator, under the flock)

One cell, DSpark depth 5, standard 16,384 + 1,024 sweep cell, `--decode-mode dspark
--stage-timing`:

```
cell16k_ring_v2_draft_attn        # K29 verify core (the shipping default)
cell16k_ring_v2_draft_attn_eager  # eager gathered verify core (K29 pinned off)
```

Run at **DSpark depth ≤ 7** (K+1 ≤ 8): a depth-8 verify is >8 rows and runs eager on
BOTH arms while control's `decode_attn_kernel_engagement.calls` stays >0 from the M=1
draft/AR — so calls alone would mislead (read rows too).

Because `DECODE_ATTN_KERNEL` is whole-arm (§3), read the RIGHT signals:

1. **`decode_attn_kernel_engagement` — read `calls` AND `rows`**: on `draft_attn`,
   `calls > 0` with `rows > 1` (verify rows actually engaged, not just the M=1 draft/AR);
   on `eager`, `calls == 0`. This confirms the A/B toggled the core AND that the verify
   (not only the draft) was on the kernel in control.
2. **`dspark.per_cycle_ms.verify_ms`** (and `verify_stage_timing`'s attn stages) — the
   ONLY clean read of the verify core. Do **not** compare headline tok/s or
   `stats.draft_ms`: the eager arm also flipped the draft (K29 is −38% vs eager at M=1,
   so its draft may be *faster*), confounding both. Note also that on `draft_attn` the
   fused K29 core has NO `attn.<mode>.score.qk_matmul/.softmax/.pv_matmul` sub-stage
   leaves (they live inside the one kernel), while `eager` does — so compare at the
   outer `attn.<mode>` stage, not the `score.*` leaves (absent by construction on
   control).
3. **`fused_proj_engagement.rows`** includes the verify rows on both (the census fix).
4. Greedy DSpark stays authoritative (byte-identity to AR is not the bar; both cores are
   greedy-identical — K29 rounding-class, eager byte-identical).

This A/B tells us whether K29 is worth keeping on the verify at all, and gives the
baseline the dispatch-collapsing kernel in §2 must beat. It does **not** itself deliver a
verify-attention speedup — that needs the W116 per-sub-op census (§2b) then the kernel.
