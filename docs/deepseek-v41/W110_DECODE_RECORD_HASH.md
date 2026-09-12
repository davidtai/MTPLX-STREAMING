# W110 — Decode-path per-record sha256: the "drop hashing" lever is VOID; kept as a bench diagnostic

Worker `w110/decode-record-hash`, Opus 4.8, CPU-only (a GPU benchmark window holds
the Metal lock; no model loaded, no Metal touched, no `~/models` read). Base
`bd9feb982` (int/w95f-lanes-2: W95+W106+W107 integrated).

**Headline (corrected after red-team): there is nothing to drop.** Decode-path
per-record sha256 has been **OFF on every ab/bench path and OFF in the served
profile**, so the W109 §1.b lever ("drop `verify_record_hashes` on decode to save
~174–226 ms/verify") removes **zero** ms — it would turn OFF what is already OFF.
This document supersedes that framing. What ships from this window is: (1) a
**bench-only diagnostic** that turns hashing ON to *measure* its cost, (2) the
io-thread engagement counters, (3) a corrected receipt (`serve_stream_counters` io
delta + `resolved_plan` gate-prefetch fix). No perf lever.

---

## 0. The premise correction (why the lever is void)

`ExpertStreamingConfig.verify_record_hashes` defaults **True** *in the dataclass*
(`expert_runtime.py:162`). W96's as-is audit noted exactly that —
"`verify_record_hashes` default `True` in the runtime dataclass"
(`W96_RUNNER_AS_IS_AUDIT.md` ~L287) — and W109 §1.b read it as "the standard
`cell16k_ring_v2` arm hashes every record." **That inference is false**: every
real path overrides the dataclass default to **False** before it reaches the io
threads:

- **ab/bench** (`scripts/deepseek_v41/ab_decode_env_levers.py`): `--verify-record-hashes`
  is `action=BooleanOptionalAction, default=False` (`:1088`), passed straight to
  `load_deepseek_v41_streaming(verify_record_hashes=False)` (`:1945`). So the
  `cell16k_ring_v2*` cell has hashed **nothing** on decode since the flag existed
  (W11/W34-era harness).
- **served** (`expert_profiles.build_expert_streaming_config`): the profile
  `deepseek-v41-mxfp4-75` ships `"verify_record_hashes": false` in its config block
  (`mtplx/data/expert_profiles.json:113`), and the served builder applies
  `**profile.config`. It also **never reads** `MTPLX_DSV41_VERIFY_RECORD_HASHES`
  (it hand-replicates only the gate-prefetch/runner env hooks) — so a loader-side
  env lever is **dead on the served path** regardless.
- the DSV4.1 helper scripts (`decode_probe.py`, `routing_census.py`,
  `ab_decode_levers.py`, `torchref/*`, …) all hardcode `verify_record_hashes=False`.

So the window-41 verify's 1,312 ms/cycle contained **no** per-record sha256 time.
W109 §1.b and its §5 lever #1 are **void**. (Left as a note here; W109's own branch
is not edited.)

---

## 1. What ships instead

### 1.a Bench-only diagnostic (measure the cost, don't remove it)

`MTPLX_DSV41_VERIFY_RECORD_HASHES=0|1` is read **at use** in
`build_streaming_config` (`mtplx/models/deepseek_v41_loader.py`), **authoritative
when set** (overrides an explicit caller value; `.strip().lower()`, so
`"1"/"true"/"on"/"yes"` → on and `"0"/"false"/"off"/"no"` → off, case-insensitive;
empty/whitespace → unset), leaving current behaviour when unset. It is honoured
**only on the loader/bench builder**; the served profile builder deliberately does
not carry the hook, so this env cannot affect a served daemon.

Because decode hashing is already OFF everywhere, the useful direction is **ON**:

- arm `cell16k_ring_v2_hash` = `cell16k_ring_v2` + `VERIFY_RECORD_HASHES=1`. The
  A/B `cell16k_ring_v2_hash` vs `cell16k_ring_v2` **measures** the io-thread cost of
  per-record sha256 (should a future policy ever require decode-time integrity),
  the reverse of the withdrawn "drop it" framing.

The env is **not** a served lever: it is **absent** from
`ab_decode_env_levers.ALL_LEVER_ENVS` and from
`mtplx/server/openai.py:_DSV41_LEVER_ENV_KEYS` (where it would have been a dead
served lever), and the W90 superset drift-guard still holds.

**Guard/stamp:** `resolved_plan.verify_record_hashes` records
`runtime.config.verify_record_hashes` for **every** arm, so a hash-vs-parent A/B can
never be control-vs-control silently (parent stamps `false`, `cell16k_ring_v2_hash`
stamps `true`).

### 1.b Engagement counters

`ExpertIOMetrics` (`mtplx/expert_io.py`, in `as_dict()`) — incremented at all three
read/hash sites (single, batched v2-overlap, rANS decode-on-miss):

| counter | meaning |
|---|---|
| `records_hashed` | records whose bytes were sha256-verified in the io thread after the read |
| `records_unhashed` | records read with verify OFF (bytes landed, no re-check) |
| `hash_thread_ns_total` | **summed** hashing time across **all** io-pool threads (not wall) — divide by the io-pool width (`max_inflight_io_bytes // record_bytes`) for an upper bound on exposed wall |

Surfaced as `snapshot["io"]` in `ExpertStreamingRuntime.snapshot()` (the served
stream-counter path, which nests `slots` and otherwise had no top-level `io`) and in
`resource_telemetry_snapshot()`.

### 1.c Receipt fixes

- **`serve_stream_counters` io delta** (`stream_counters_delta`): `_delta_map` used to
  difference the whole io block, turning the cumulative-since-open **float**
  `read_mib_per_second` into a garbage "delta". Now the non-counter float keys are
  dropped before the delta, and the decode-**window** read rate is derived from the
  `read_bytes`/`read_ns` counter deltas as **`io.read_gb_per_s_window`** (bytes/ns ==
  GB/s; `read_ns` is summed io-thread read time, so this is aggregate io-thread read
  throughput while reading — the per-window SSD read rate W109 wanted in the
  receipt). The block also reports `records_hashed[_per_token]`,
  `records_unhashed[_per_token]`, `hash_fraction`, `hash_thread_ms[_per_token]`.
- **`resolved_plan` gate-prefetch** (see §3).

---

## 2. Integrity argument (verified in code) — this is CURRENT production behaviour

Because decode hashing is already off, the "corrupt record with hashing off" case
below is **how production runs today**, not a risk this window introduces.

**Checked at open.** `ExpertStreamingRuntime.open` calls `verify_expert_manifest`
(`expert_runtime.py` ~L2325 → `expert_manifest.py:2388`) whenever
`verify_artifact_headers` (default **True**) or `verify_sidecar_hash_at_open`
(default **False**). With the shipped defaults it verifies `validate_structure()`,
manifest self-consistency (`manifest_sha256`), the authoritative safetensors resident
inventory, each shard's provenance (`size`, `header_bytes`, safetensors **header**
`header_sha256`), and the **sidecar file SIZE** (`fstat`). It does **not**
content-hash the sidecar record bytes (`verify_records`/`verify_shard_hashes`/
`verify_sidecar_hash` all False). Descriptors are then **pinned**
(`ADMITTED_DESCRIPTOR_SECURITY_BOUNDARY`, `expert_io.py:53`): pinned fds prevent
pathname replacement after admission.

**What a corrupt record does with hashing OFF (i.e. today).** A record whose on-SSD
bytes disagree with the manifest's trusted `record.sha256` — a same-user in-place
write to the retained pinned inode after admission that **preserves the file size** —
is **not detected**: open only checked size (unchanged) and decode runs no re-check;
the corrupt bytes land in the slot and are used as expert weights → **wrong output,
no crash/exception** (`test_corrupt_record_lands_silently_with_hashing_off`). This is
exactly the residual risk the boundary comment declares **outside** the local-artifact
threat model. With hashing **ON** the record is rejected — `ExpertIOIntegrityError`
"hash mismatch", `integrity_errors` incremented, fail-closed
(`test_corrupt_record_rejected_with_hashing_on`).

**If defense-in-depth is ever wanted** at zero per-token cost: set
`verify_sidecar_hash_at_open=True` (hashes the whole sidecar once at open). That is a
separate config field; the W110 env never touches it.

---

## 3. `resolved_plan` gate-prefetch fix + "armed ≠ engaged"

`ab_decode_env_levers._resolved_plan` derived `gate_prefetch_armed` / `gate_prefetch_k`
from the explicit `MTPLX_DSV41_GATE_PREFETCH` env **only**, so it missed the v2
runner's auto-arm: **window-39/41 receipts' `gate_prefetch_armed=False` / `k=0` are
historically WRONG** (the runner had `prefetch_committed=10513`, `k=6`). The fix reads
the actual state from the runtime object the loader built:

- `gate_prefetch_armed` = `config.prefetch_slots > 0` (the ring was built);
- `gate_prefetch_k` = the width the runtime resolved (its runner receipt block's
  `prefetch_k`, via `_runtime_gate_prefetch_k`), **not** `prefetch_slots//2` — the v2
  ring is sized `2*24=48` to buffer the verify union, so `//2` misreports `24` not `6`;
- `gate_prefetch_env` keeps the raw env for provenance.

The W93 explicit-lever guard (explicit env armed but no ring → fail loud) is preserved.

**Armed ≠ engaged.** A built ring is only *armed*; the gate oracle *engages* in
specific phases (`_maybe_stash_gate_prefetch`, `deepseek_v41.py` ~L2880–2915):

- it runs only in the **DECODE routing phase** (`current_expert_routing_phase`), so
  **PREFILL (any T) never predicts** — a short 2..8-token prefill has the same shape
  as a verify but must not speculate;
- **AR decode** (T==1) predicts the single row's route;
- the **DSpark verify** (T=K+1, 2..MAX rows) predicts the per-row union, but **only
  under the v2 runner** — an AR-only `MTPLX_DSV41_GATE_PREFETCH` stays inert at T>1; a
  verify wider than MAX never predicts.

So a receipt that shows a large ring but flat `prefetch_issued` outside decode is
expected, not a bug — read `prefetch_issued`/`prefetch_committed`, not `armed`, to
know it *engaged*.

---

## 4. GPU command (DO NOT RUN — a benchmark window holds the Metal lock)

Diagnostic only (measure the cost of hashing that is currently OFF). Run under the
box's GPU flock, one arm per process:

```
nice -n 19 python3 scripts/deepseek_v41/ab_decode_env_levers.py \
  --arms cell16k_ring_v2_hash cell16k_ring_v2 \
  --context-tokens 16384 --decode-tokens 256 --max-kv 17408 \
  --memory-limit-gib 60 --decode-mode dspark --dspark-depth 5 \
  --out <receipt>/dspark-d5-hashcost.jsonl
# gate: resolved_plan.verify_record_hashes == true on cell16k_ring_v2_hash and
#   false on cell16k_ring_v2 (the A/B is real, not control-vs-control);
#   token_ids_sha256 identical (byte-neutral);
#   serve_stream_counters.io.records_hashed >0 (hash arm) vs 0 (parent),
#   hash_thread_ms the SUMMED io-thread hashing cost (divide by io-pool width for a
#   wall upper bound); io.read_gb_per_s_window the per-window SSD read rate.
# (optional) MTPLX_DSV41_RESOURCE_TELEMETRY=1 to also record reader_pool concurrency.
```

There is **no** `cell16k_ring_v2_draft_nohash`/`_nohash` arm — those were withdrawn
with the void lever.

---

## 5. Provenance

Base `bd9feb982`. Code: `expert_io.py` (ExpertIOMetrics `records_hashed` /
`records_unhashed` / `hash_thread_ns_total` + `as_dict`; three read/hash sites),
`expert_runtime.py` (`snapshot["io"]` in `snapshot` + `resource_telemetry_snapshot`),
`serve_stream_counters.py` (io pass-through; io delta drops the float rate + adds
`read_gb_per_s_window`), `models/deepseek_v41_loader.py`
(`MTPLX_DSV41_VERIFY_RECORD_HASHES` bench-only hook, case-insensitive),
`scripts/deepseek_v41/ab_decode_env_levers.py` (`_resolved_plan` fix +
`_runtime_gate_prefetch_k` + `verify_record_hashes` stamp; the bench-only diagnostic
arm `cell16k_ring_v2_hash`). Tests: `tests/test_deepseek_v41_w110_record_hash.py`
(CPU, synthetic bank). Void: the W109 §1.b / §5 lever #1 ("drop decode hashing");
premise traced to a misread of `verify_record_hashes` being the dataclass default
(W96 ~L287) while every real path overrides it to False.
