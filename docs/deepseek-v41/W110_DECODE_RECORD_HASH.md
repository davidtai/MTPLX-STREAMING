# W110 — Drop the decode-path per-record sha256 (byte-identical)

Worker `w110/decode-record-hash`, Opus 4.8, CPU-only (a GPU benchmark window holds
the Metal lock; no model loaded, no Metal touched, no `~/models` read). Base
`bd9feb982` (int/w95f-lanes-2: W95+W106+W107 integrated). Follows the corrected
verify audit `W109_VERIFY_IO_FANOUT.md` §1.b, lever #1.

**Lever #1 of the W109 ranking, built.** On the DSpark verify the io threads read
5.42 GB/verify and then sha256 every record **in the same io thread, right after
each read**, serialized before the slot is ready (`expert_io.py`
`read_component_records_into` / `read_record_into`). W109 measured that at
**~174–226 ms of io-thread time per verify** (3.29 GB/s single-thread, ~31 GB/s at
11 threads). This lever removes it. It is **byte-identical**: hashing never changes
the bytes landed in the slot; it only re-checks integrity the admission receipt
already covers at open.

---

## 1. Mechanism

`MTPLX_DSV41_VERIFY_RECORD_HASHES=0|1` gates the per-record sha256 re-check on the
**DECODE/verify streaming path**.

- Read **at use** in `build_streaming_config`
  (`mtplx/models/deepseek_v41_loader.py`), not at import (cf.
  `memory/env-flags-read-at-use-not-import.md`). When set to `0` (also `false`/`no`/
  `off`) it forces `verify_record_hashes=False`; `1` forces `True`.
- **AUTHORITATIVE when set**: it overrides an explicit caller
  `verify_record_hashes=` so an A/B arm (`cell16k_ring_v2_nohash`) forces the
  decode-path hashing off even though the ab harness passes an explicit
  `--verify-record-hashes` value. **Unset (or empty)** leaves current behaviour: the
  caller value, else the `ExpertStreamingConfig` default (`True`).
- `verify_record_hashes` reaches the io reader through
  `ExpertStreamingRuntime.open` →
  `verify_hashes = config.verify_record_hashes and not config.verify_sidecar_hash_at_open`
  → `ExpertSlotPool(verify_hashes=…)` → `reader.read_*_into(verify_hash=…)`
  (`expert_runtime.py` ~L2465). Off → each io thread does `read(record)` and lands
  the bytes; on → it also runs `hashlib.sha256` over the record and compares to
  `record.sha256`, raising `ExpertIOIntegrityError` on mismatch.

The lever is DECODE-path **only**. It never touches `verify_artifact_headers`
(default `True`) or `verify_sidecar_hash_at_open` (default `False`) — separate config
fields that drive the admission/open verification (§2) — so open-time integrity
stays on regardless of the lever.

---

## 2. Integrity argument (verified in code)

**Where the bank's manifest/hashes are checked at open.**
`ExpertStreamingRuntime.open` calls `verify_expert_manifest(manifest, artifact_root,
verify_sidecar_hash=config.verify_sidecar_hash_at_open)` whenever
`config.verify_artifact_headers` (default **True**) **or**
`config.verify_sidecar_hash_at_open` (default **False**) is set
(`expert_runtime.py` ~L2325). With the shipped defaults that call
(`expert_manifest.py:2388`) verifies:

- `manifest.validate_structure()` and manifest self-consistency
  (`manifest_sha256 == with_digest().manifest_sha256`);
- for an authoritative artifact, the resident safetensors inventory (shard set,
  header inventory, per-tensor offset/length/dtype/shape);
- **each shard's provenance**: `size`, `header_bytes`, and the safetensors
  **header** `header_sha256`;
- **the sidecar file SIZE** (`fstat` / `stat`, `experts.bin` part sizes).

With the defaults it passes `verify_records=False`, `verify_shard_hashes=False`,
`verify_sidecar_hash=False`, so the **content bytes of the streamed sidecar
records are NOT hashed at open** — only the sidecar's total size is checked. After
this, descriptors are **pinned** (`ADMITTED_DESCRIPTOR_SECURITY_BOUNDARY`,
`expert_io.py:53`): pinned fds prevent pathname replacement after admission;
MTPLX-controlled installs must stage a separate file and atomically replace the
pathname, never mutate an admitted inode.

So in the shipped config the per-record sha256 on the decode path is the **only**
content-level re-check of the streamed sidecar record bytes. It re-checks the exact
risk the boundary comment names: *"Uncooperative same-user writes to that retained
inode after construction are outside the local artifact threat model; per-record
hash verification, when enabled, detects that residual risk."*

**What a corrupt record does with hashing OFF.** A record whose on-SSD bytes
disagree with the manifest's trusted `record.sha256` — a same-user in-place write to
the retained (pinned) inode after admission that **preserves the file size** —
is **not detected**: open already ran and only checked size (unchanged), and the
decode path runs no re-check. The corrupt bytes land in the slot and are used as
expert weights → **wrong/garbage expert output, no crash and no exception**
(`test_corrupt_record_lands_silently_with_hashing_off`). With hashing **ON** the
same record is **rejected** — `ExpertIOIntegrityError` "expert record hash mismatch",
`integrity_errors` incremented, fail-closed
(`test_corrupt_record_rejected_with_hashing_on`).

**Net.** Turning the lever OFF returns to the declared local-artifact threat-model
boundary (trust the admitted, pinned inode; rely on the open-time structure/size
checks). Defense-in-depth content verification at open, at zero per-token cost, is
available separately via `verify_sidecar_hash_at_open=True` (hashes the whole
sidecar once at open); this lever never touches that flag.

---

## 3. Engagement counters (exact names)

`ExpertIOMetrics` (`mtplx/expert_io.py`) gains three cumulative io-thread counters,
in `as_dict()`:

| counter | meaning |
|---|---|
| `records_hashed` | records whose bytes were sha256-verified in the io thread after the read |
| `records_unhashed` | records read with verify OFF (bytes landed, no re-check) |
| `hash_ns_total` | cumulative io-thread wall (ns) spent in `hashlib.sha256` |

They increment at all three read/hash sites: the single-record path, the batched
(v2 `overlap_miss_reads`) path, and the rANS decode-on-miss path.

**On the receipt.** The runtime surfaces the reader metrics as `snapshot["io"]` in
both `ExpertStreamingRuntime.snapshot()` (served stream-counter path) and
`resource_telemetry_snapshot()` (bench sampler). `serve_stream_counters` passes the
`io` block through and the **decode-scoped delta** (`_stream_counters_block` →
`serve_stream_counters` block, on every AR/DSpark A/B receipt) reports:

```
serve_stream_counters.io = {
  records_hashed, records_unhashed, hash_ns_total,          # decode-window deltas
  records_hashed_per_token, records_unhashed_per_token,
  hash_fraction,        # records_hashed / (hashed+unhashed); None if no reads
  hash_ms, hash_ms_per_token   # hash_ns_total in ms
}
```

Gate for the A/B: control (hashing ON) shows `records_hashed>0`, `records_unhashed=0`,
`hash_ms>0`; the `_nohash` arm shows `records_hashed=0`, `records_unhashed>0`,
`hash_ms=0`; and `token_ids_sha256` is **identical** between the two (byte-identical
by construction).

**Companion receipt fix (W110).** `ab_decode_env_levers._resolved_plan` now reports
the **actual** gate-prefetch armed state from the runtime, not the raw env: the v2
runner auto-arms the ring (`config.prefetch_slots>0`) with `MTPLX_DSV41_GATE_PREFETCH`
unset, so the old env-only read printed `gate_prefetch_armed=False`/`k=0` while the
runner actually prefetched at `k=6` (window 41: `prefetch_committed=10513`).
`gate_prefetch_armed` now = ring built; `gate_prefetch_k` = the width the runtime
resolved (its runner block's `prefetch_k`, **not** `prefetch_slots//2` — the v2 ring
is sized `2*24=48` to double-buffer the verify union, so `//2` would misreport `24`);
`gate_prefetch_env` keeps the raw env for provenance. The W93 explicit-lever guard
(explicit env armed but no ring → fail loud) is preserved.

---

## 4. Arms

- `cell16k_ring_v2_nohash` = `cell16k_ring_v2` + `MTPLX_DSV41_VERIFY_RECORD_HASHES=0`.
- `cell16k_ring_v2_draft_nohash` = `cell16k_ring_v2_draft` +
  `MTPLX_DSV41_VERIFY_RECORD_HASHES=0`.

The env is registered in `ALL_LEVER_ENVS` and in the served-log snapshot
`mtplx/server/openai.py:_DSV41_LEVER_ENV_KEYS` (the W90 drift guard keeps the latter
a superset of the former).

---

## 5. GPU A/B command (DO NOT RUN — a benchmark window holds the Metal lock)

Run under the box's GPU flock, one arm per process. **Pass `--verify-record-hashes`
so the CONTROL arm hashes** (hashing is what we are measuring); the `_nohash` arm's
env forces it OFF regardless (the lever is env-authoritative), so the single flag is
correct for both arms:

```
nice -n 19 python3 scripts/deepseek_v41/ab_decode_env_levers.py \
  --arms cell16k_ring_v2_draft_nohash cell16k_ring_v2_draft \
  --verify-record-hashes \
  --context-tokens 16384 --decode-tokens 256 --max-kv 17408 \
  --memory-limit-gib 60 --decode-mode dspark --dspark-depth 5 \
  --out <receipt>/dspark-d5-nohash.jsonl
# gate: token_ids_sha256 == control (byte-identical);
#   read dspark.per_cycle_ms.verify_ms down (vs 1,312 at window 41);
#   serve_stream_counters.io.records_hashed 0 on _nohash / >0 on control,
#   hash_ms 0 vs >0; resolved_plan.gate_prefetch_armed/k now correct (W110).
```

(Optional, to measure the io-thread concurrency the hashing serializes behind, arm
`MTPLX_DSV41_RESOURCE_TELEMETRY=1` per W109 §1.d.)

---

## Provenance

Base `bd9feb982`. Code: `expert_io.py` (ExpertIOMetrics `records_hashed` /
`records_unhashed` / `hash_ns_total` + `as_dict`; the three read/hash sites),
`expert_runtime.py` (`snapshot["io"]` in `snapshot` + `resource_telemetry_snapshot`),
`serve_stream_counters.py` (io pass-through + decode-scoped delta),
`models/deepseek_v41_loader.py` (`MTPLX_DSV41_VERIFY_RECORD_HASHES` in
`build_streaming_config`), `scripts/deepseek_v41/ab_decode_env_levers.py`
(`_resolved_plan` receipt fix + `_runtime_gate_prefetch_k`; env const, `ALL_LEVER_ENVS`,
`_preset`, the two `_nohash` arms), `server/openai.py` (`_DSV41_LEVER_ENV_KEYS`).
Tests: `tests/test_deepseek_v41_w110_record_hash.py` (18, CPU, synthetic bank).
sha256 throughput figures measured on this box (M5 Max, CPU, `nice -n 19`) in W109.
