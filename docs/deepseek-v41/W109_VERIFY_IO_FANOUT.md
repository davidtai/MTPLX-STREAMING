# W109 — DSpark verify SSD queue depth + concurrent union read (T1 of W108)

Worker `w109/verify-io-fanout`, Opus 4.8. Base `1e1856ea6` (the W100 merge).
Authored CPU-only (a GPU benchmark window holds the Metal lock and is reading the
expert bank from SSD; no model was loaded, no Metal touched, no SSD read run against
the model files). This implements **T1 / issue I1** of the W108 audit
(`docs/deepseek-v41/W108_DSPARK_RUNNER_AUDIT.md`): get the verify's ~5.4 GB/token
miss reads off the critical path by raising SSD queue depth and issuing a layer's
miss union as concurrent batched reads. Byte-identical by construction — only I/O
scheduling changes; the gather consumes the same `indices` and the records land in
the same slots.

Receipt anchor for the as-is numbers: window-41
`dspark-d5-ring-v2-budget0.json` (verify 1,312 ms/cycle, draft 230, 4.76 tok/cycle,
`transient_slots=48`, `slots_per_layer=49`, `prefetch_slots=48`).

---

## 1. As-is audit of the verify miss-read path (file:line)

### 1.1 Read primitive — `preadv` + `F_NOCACHE` (direct), NOT mmap

The streamed expert bank is read with **positional `preadv`** into caller-owned slot
buffers, with the page cache **bypassed** (`F_NOCACHE`) when `bypass_page_cache` is on
(it is, on the DSV4.1 streamed profile). There is an optional native backend
(`mtplx_native_expert_io.pread_exact_into`) with the identical contract; it is the
default when the extension is importable.

- `mtplx/expert_io.py:695` — the Python path: `os.preadv(fd, [target], source_offset + read_total)`.
- `mtplx/expert_io.py:677` — the native path: `native_read_into(fd, source_offset + read_total, target)` (`pread`-exact).
- `mtplx/expert_io.py:389` and `:471` — `fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)` on the pinned/leased fd when `bypass_page_cache`.
- `mtplx/expert_io.py:776` — the scatter path `_readv_range_into` (one `os.preadv` over an iovec of ≤ `IOV_MAX` component views) used by the coalesced/batched reads.
- `mtplx/expert_io.py:415` — `backend` = `"native"` or `"python-preadv"`; `:419` — `cache_mode` = `"f-nocache"` or `"buffered"`.

**There is no mmap on the streamed decode path.** `mmap` in this codebase is a
separate `slot_layout="metal-mmap"` / `mmap_island_layers` feature
(`expert_runtime.py:185,198,443,578`); the DSV4.1 v2 cell runs `slot_layout=
"component-banks"` with `preadv`+`F_NOCACHE`. So the ~4.4 GiB/s page-cache/mmap
plateau from memory note *mmap WILLNEED unwired* does **not** apply here — this is a
direct-read path whose queue depth is set by how many `preadv` calls are in flight
concurrently, which is exactly the lever.

Existing intra-record fanout: `PositionalExpertReader` already has an
`io_read_fanout` knob (`expert_io.py:240,288-295,586-637`) that splits **one** record
larger than `max_read_chunk_bytes` (8 MiB) into N contiguous sub-`preadv`s on a
`mtplx-io-fanout` ThreadPoolExecutor. That is *within-record* concurrency and is
bypassed by the batched **scatter** path (`_readv_range_into` does one `preadv` per
coalesced run and never fans out). It is therefore orthogonal to — and does not
deliver — the *cross-record* concurrency this task needs.

### 1.2 How misses are discovered per layer — needs the routing barrier

Per verify layer, inside `HotExpertSwitchGLU._run` (`mtplx/models/expert_mlx.py`):

1. `mx.eval(indices)` — the **per-layer routing barrier** (`expert_mlx.py:2658-2665`,
   the W108 V-c). Materializes this layer's `indices` so the host can read the routed
   expert ids. 40 of these per verify forward.
2. `indices.reshape(-1).tolist()` (`expert_mlx.py:2672`) — host read of the routed ids.
3. The W81 single-barrier verify split (`expert_mlx.py:3143-3241`) groups the routed
   ids into `route_waves` (≤ transient capacity → one wave; window-39/41 always one
   wave) and calls `runtime.begin_split_route(layer, wave.experts, phase=DECODE)`
   (`expert_mlx.py:3156-3161`).

So the miss set for a layer is **not** knowable until that layer's routing barrier has
drained — the reads are inherently reactive per layer. Cross-layer read concurrency is
**not** possible on the demand path without the barrier (the ids do not exist yet);
that is what the *prefetch* lane (I2/T2, the gate oracle + draft-window prefetch)
targets and is out of scope here. What **is** available at this layer, once its
barrier drains, is concurrency **across the layer's own miss union** — that is T1.

### 1.3 How each read is issued — split executor, per-expert vs one batched future

`ExpertStreamingRuntime.begin_split_route` (`mtplx/expert_runtime.py:3291-3466`):

- `plan, policy_txn = self._plan_route_transaction(...)`; `miss_plan =
  self._subset_route_plan(plan, hits=False)` (`:3332-3338`).
- **Grouping** (`:3363-3374`): `batch_misses = config.overlap_miss_reads and DECODE`.
  - `overlap_miss_reads` **False** → `miss_parts = self._miss_route_parts(miss_plan)`
    (`:3156-3204`): **one part per expert**, each its own future.
  - `overlap_miss_reads` **True** (armed by the v2 runner via
    `deepseek_v41_loader.py:353-355` / `expert_profiles.py:399-407`) → `miss_parts =
    (miss_plan,)`: the **whole union as ONE future** → one thread issuing sequential
    scatter `preadv`s. **This is what window-41 `dspark-d5-ring-v2` ran** (RUNNER=v2,
    OVERLAP unset-but-armed): a layer's whole ~11-miss union read by a single worker,
    so realized SSD QD is ~1–2 regardless of the 48-wide pool.
- **Submission** (`:3422-3456`): each part → `self._split_executor.submit(ensure, layer, part, ...)`
  where `ensure = self.slots.ensure_route_part` on DECODE. The executor is
  `ThreadPoolExecutor(max_workers=max(1, plan.transient_slots))` (`:2043-2044`) —
  **48 workers** on this cell — so the per-expert regime *can* run all ~11 misses
  concurrently, but the armed one-batched-future regime funnels them through one
  worker. `ensure_route_part` reads a multi-record part via
  `read_component_records_into` → `_readv_range_into` (adjacency-coalesced scatter,
  `expert_io.py:1151-1278`); a per-expert part reads one record via
  `read_record_into` → `_read_range_into` (fanout-capable).
- `plan.transient_slots=48 ≥ union (~24.5) ≥ any reasonable fanout`, so the pool never
  bounds an ≤8/16-way verify fanout; **no executor resize is needed**.

### 1.4 Where the generation thread waits

`PendingSplitRoute.iter_ready_misses` (`expert_runtime.py:1089-1169`) — the generation
thread consumes miss futures in completion order via `as_completed`; the blocking
`future.result()` / iterator-advance is **the SSD read wait** (W108 V-h). The verify
split loop drives it at `expert_mlx.py:3184` (`for _miss_ready in
_pending.iter_ready_misses():`), doing the deferred per-part `gather_qmm` between
yields. Because a generator is *suspended* at `yield` while the consumer runs the
gather, the wall spent **inside** the generator between resume and the next yield is
exactly the exposed read wait (this is what `verify_io_wait_ms_total` measures, §3).

**Verdict.** Read primitive = `preadv`(+native `pread`) with `F_NOCACHE`, direct (no
mmap). Misses need the per-layer routing barrier (no cross-layer demand concurrency
without prefetch). Within a layer, the v2 arm currently reads the whole miss union
through **one** worker (armed `overlap_miss_reads`), so the 48-wide pool is idle; the
lever is to fan that union into N concurrent batched `preadv` groups. Byte-identical:
scheduling only.

---

## 2. Design

Two composable levers reshape ONLY the DECODE-phase miss submission in
`ExpertStreamingRuntime.begin_split_route` (`mtplx/expert_runtime.py`). Both are
read AT USE (`_verify_io_fanout_setting()` / `_verify_union_read_setting()`), so an
arm that exports them after import still engages, and both default to today's exact
grouping when unset.

- `MTPLX_DSV41_VERIFY_IO_FANOUT=N` — issue a layer's miss union as up to `N`
  concurrent groups. Each group is one future on the existing 48-wide
  `_split_executor`, so `N` reads run in parallel; a group with adjacent records
  still coalesces into one scatter `preadv`. New helper `_miss_route_groups(plan, N)`
  partitions the unique miss experts into `N` ceil-balanced contiguous chunks (it is
  the generalization of `_miss_route_parts`, which is the `N == union` case, and of
  the single-union part, the `N == 1` case).
- `MTPLX_DSV41_VERIFY_UNION_READ=1` — force the whole union into ONE batched
  submission even when the `overlap_miss_reads` config knob is off (`miss_parts =
  (miss_plan,)`, identical shape to the overlap path).

Grouping resolution (DECODE, misses present, a lever armed):

| FANOUT | UNION_READ | groups (miss_parts) |
|---|---|---|
| unset | unset | today's behavior (per-expert, or one union if overlap config on) |
| unset | 1 | `1` (one batched submission) |
| N | any | `max(1, min(N, unique_misses))`, with `1 -> (miss_plan,)` and `>= union -> _miss_route_parts` (both reuse the existing, tested regimes verbatim) |

The composite arm `cell16k_ring_v2_verifyio` sets `FANOUT=8 + UNION_READ=1`: on a
~11-miss verify layer that is `min(8,11)=8` concurrent batched-scatter groups vs the
current **one** serial batched future (the v2 arm runs `overlap_miss_reads` armed, so
window-41 read each layer's whole union through a single worker). No executor resize
is needed: `plan.transient_slots=48` already exceeds any reasonable fanout, and every
regrouping partitions the SAME loads/slots exactly once, so records land in the same
slots.

**Byte-identity by construction.** The gather consumes the true `indices`; the
records' bytes and their consumers are unchanged. Only which records share a syscall
and how many reads run concurrently changes. `_miss_route_groups` reuses the exact
RoutePlan-subset shape of `_miss_route_parts`; the `N==1` and `N>=union` cases return
the already-shipped `(miss_plan,)` / `_miss_route_parts(miss_plan)` objects verbatim.
Reads never exceed the resident plan (same slots as today).

---

## 3. Env + engagement counters

Env (read at use; both default to current behavior when unset):

- `MTPLX_DSV41_VERIFY_IO_FANOUT` — integer `>= 1` (invalid / `<1` / empty -> treated
  as unset). Concurrent batched miss groups per DECODE layer.
- `MTPLX_DSV41_VERIFY_UNION_READ` — truthy (`1`/`true`/`yes`/`on`).

Counters on `ExpertSlotMetrics` (`mtplx/expert_slots.py`), surfaced in
`slots.snapshot()["metrics"]`, in the v2 `runner.verify_io` receipt block
(`_runner_snapshot`), and in the AB harness receipt under `verify_io`
(`ab_decode_env_levers.py::_verify_io_telemetry`). All zero unless a lever is armed:

| counter | meaning |
|---|---|
| `verify_io_reads_issued` | expert records scheduled on the DECODE miss path under the lever (= misses/layer summed) |
| `verify_io_batches` | concurrent batched groups (futures) they were submitted as |
| `verify_io_max_inflight` | peak concurrently in-flight miss reads observed (a real gauge via a wrapped `ensure`, not `len(parts)`) |
| `verify_io_wait_ms_total` | ms the generation thread was blocked inside `iter_ready_misses` awaiting these reads (surfaced from the `verify_io_wait_ns_total` ns counter) |

`verify_io_wait_ms_total` is measured as the wall the miss generator itself executes
(it is *suspended* at `yield` while the consumer runs the gather), so it is the
exposed read wait excluding the interleaved GPU dispatch. `per_cycle_ms.verify_ms`
remains the authoritative headline; these are engagement/attribution counters.

Per-cycle plumbing into the DSpark `per_cycle_ms` block was **not** added: that block
(`deepseek_v41_dspark_decode.py::_dspark_decode_wall_accounting`) accumulates phase
wall-time seconds, not slot-metric deltas, so there is no seam to carry a per-cycle
counter without new plumbing (out of T1 scope). The counters are cumulative over the
decode window in the receipt.

---

## 4. Test evidence

`tests/test_deepseek_v41_w109_verify_io.py` (CPU-pinned, synthetic component-bank
artifact reused from `test_expert_overlap_split`), run one file per process under
`nice -n 19`, `PYTHONPATH` pinned to the worktree:

```
$ nice -n 19 python3 -m pytest tests/test_deepseek_v41_w109_verify_io.py -q
.......                                                                  [100%]
7 passed, 2 warnings in 0.75s
```

The seven cases:
- `test_verify_io_default_off_keeps_per_expert_parts` — env unset -> 4 per-expert
  futures (current behavior); all four counters zero; `pending._verify_io_timed` False.
- `test_verify_union_read_forces_single_submission` — `UNION_READ=1` (overlap config
  off) -> ONE submission covering `{0,1,2,3}`; `batches==1`, `reads_issued==4`.
- `test_verify_io_fanout_splits_union_into_n_groups` — `FANOUT=2` over 4 misses -> 2
  disjoint groups covering the whole union; `batches==2`, `reads_issued==4`.
- `test_verify_io_fanout_at_or_above_union_is_per_expert` — `FANOUT=8 + UNION=1`
  (the arm shape) over 4 misses -> `min(8,4)==4` per-expert parts.
- `test_fanout_reads_run_concurrently` — fake recording reader: `FANOUT=2` with
  adjacent groups -> `max_active==2` concurrent reader calls, `verify_io_max_inflight
  ==2`, `verify_io_wait_ns_total>0` (N-way concurrency proven).
- `test_union_read_is_a_single_reader_call` — fake recording reader: `UNION_READ=1`
  -> exactly ONE batched reader call covering all four union records (union batching
  proven).
- `test_fanout_1_and_8_are_byte_identical` — a full `HotExpertSwitchGLU` forward over
  the same synthetic bank emits bit-identical output at `FANOUT=1` and `FANOUT=8`
  (`np.array_equal`), with the lever engaged in both (`verify_io_batches>=1`).

Existing suites still green with the change:
`tests/test_expert_overlap_split.py` (10 passed),
`tests/test_deepseek_v41_verify_single_barrier_split.py` (14 passed),
`tests/test_expert_slots_runtime.py` + `tests/test_expert_io_metrics.py` (191
passed), `tests/test_deepseek_v41_w95_runner_v2.py` (17 passed).

### 4.1 Microbench validation (temp file only)

`scripts/deepseek_v41/ssd_read_fanout_bench.py --self-test` builds a small temp file
(no model, no real bank) and sweeps fanout 1/2/4/8/16 with the runtime's primitive
(native `pread_exact_into` when present, else `os.preadv`) + `F_NOCACHE`:

```
$ nice -n 19 python3 scripts/deepseek_v41/ssd_read_fanout_bench.py --self-test
{ "backend": "native", "cache_mode": "f-nocache", "record_bytes": 65536,
  "num_offsets": 48, "fanout": { "1": {...}, "2": {...}, ... "16": {...} },
  "best_fanout": 1, "peak_gb_per_s": 31.6, "self_test": true }
```

(On a tiny cached temp file fanout=1 wins — thread overhead dominates 64 KiB reads;
the sweep only validates the code path + JSON shape. The real bank is 269 GiB with
`F_NOCACHE` bypassing the cache and 18.8 MB records, where queue depth matters — the
orchestrator runs that in the lock gap, §5.)

---

## 5. GPU A/B command (DO NOT RUN — a benchmark window holds the Metal lock)

The DSpark d5 paired A/B on the standard 16K cell (the T1 done-when read is realized
BW up and `verify_ms` down at an unchanged `token_ids_sha256`; run under the box's
GPU flock/guarded wrapper, one arm per process):

```
nice -n 19 python3 scripts/deepseek_v41/ab_decode_env_levers.py \
  --arms cell16k_ring_v2_verifyio cell16k_ring_v2 \
  --context-tokens 16384 --decode-tokens 256 --max-kv 17408 \
  --memory-limit-gib 60 --decode-mode dspark --dspark-depth 5 \
  --out docs/deepseek-v41/receipts/gpu-windows/window-XX/dspark-d5-verifyio.jsonl
```

Read from the receipt: `runner.verify_io` (fanout/union_read + reads_issued/batches/
max_inflight/wait_ms_total) and top-level `verify_io`; `dspark.per_cycle_ms.verify_ms`
(vs 1,312 ms control); `serve_stream_counters` bytes/hit; and `token_ids_sha256` ==
the control arm (byte-identity gate).

### 5.1 SSD read-fanout microbench for the real bank (orchestrator, in the lock gap)

Expert bank: `/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/experts.bin`
(288,777,830,400 bytes; single-file mxfp4 sidecar). Representative record size
18,800,000 bytes (~18.80 MB/record, window-41 receipt):

```
nice -n 19 python3 scripts/deepseek_v41/ssd_read_fanout_bench.py \
  /Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/experts.bin \
  --record-bytes 18800000 --num-offsets 128 --fanouts 1,2,4,8,16 --repeats 3 \
  --output docs/deepseek-v41/receipts/gpu-windows/window-XX/ssd_read_fanout.json
```

This characterizes realized GB/s vs queue depth on the drive with the runtime's exact
read primitive, quantifying the 4.71 GB/s -> ? ceiling the FANOUT lever climbs and the
fanout at which the drive saturates (informing the default N; the arm ships N=8).
