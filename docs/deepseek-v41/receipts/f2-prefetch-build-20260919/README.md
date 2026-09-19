# F2b next-layer expert prefetch — build receipt (2026-09-19)

Next-layer expert prefetch for the DeepSeek-V4.1 Q4 D5/M≤8 verify decode, composed onto
the retained 13.87 TPS packed run. **F2b** design: a lane-private HOST ring plus
reader-level interception — the runtime keeps `prefetch_slots == 0` and its packed_phase /
packed_admission / config stay byte-for-byte retained (except the equal-capacity row cap);
the runtime still sees an ordinary decode MISS, but the miss is fulfilled from RAM instead
of the SSD. This supersedes the earlier runtime-ring design (deleted: FullPrefetchConfig,
install_f2_growth, the ring-charge staging, the PrefetchDecode/PriorityReads lane), which
`packed_phase.install_growth` refuses at construction (it rejects any ring; packed_phase.py:
41-42/54/64-65).

Code: `scripts/deepseek_v41/f2/` (host_ring, reader_intercept, predictor, speculative,
install, stage_f2_runner, window_preflight). CPU proofs: `tests/test_dsv41_f2b.py`,
`tests/test_dsv41_f2_stage.py`, kept `tests/test_dsv41_f2_prefetch.py`. Branch
`f2/next-layer-prefetch`. No GPU touched.

## Design

- **Host ring** (`host_ring.py`): R=32 records × 3 planes of page-aligned anonymous RAM
  (`mmap.mmap(-1, n)` → `np.frombuffer`). Each entry is ONE plane, keyed by its ABSOLUTE
  `experts.bin` offset (`record.sidecar_offset + {0, 6,266,880, 12,533,760}`) — no
  layer/expert identity at the reader. Per-plane states QUEUED→READING→READY + a refcount
  during a demand copy; FIFO recycling skips READING/referenced entries. Buffers are sized
  to the max plane length; each entry carries its exact plane length (gate/up 6,266,880 vs
  down 5,160,960) read at install from a live slot's `component_view` (weights only; scales
  resident) — never hard-coded. One lock for metadata; byte copies (`np.copyto` on
  `np.frombuffer` views, GIL released) happen outside it.
- **Reader interception** (`reader_intercept.py`): the replacement for
  `reader.read_record_into` / `read_component_records_into` is DERIVED from the retained
  `plane_lane.bind_reader` source by a single anchored line insertion with a round-trip
  check (the f5_compile/timed_plane_lane.py discipline; retained lane pinned to sha256
  `1acad9e2…ba54`) and exec'd in `plane_lane`'s namespace, so it is byte-for-byte the
  retained reader (job construction, gate/up-first/down-last, fanout submit, early gate/up
  witness `publish_read_components`, error joining, metrics, view release) EXCEPT the
  per-plane `read(job)` first tries the ring: READY → copy + `prefetch_plane_hits`; READING
  → wait + copy + `prefetch_plane_waits`; QUEUED → mark cancelled + `prefetch_plane_cancelled`
  → normal pread; absent → normal pread. It reuses the installed lane's `local` witness (via
  `runner.executor.local`). No eligible-or-stock branch: the ring lookup IS the lane's work
  on the enabled path (AGENTS.md).
- **Speculative reads** (`speculative.py`): a private N-worker pool (default 3) calling the
  SAME primitive the retained lane uses — `reader._readv_range_into('experts.bin', offset,
  (view,))` — into ring buffers, gate/up/down order. A per-target "demand imminent" flag
  stops workers from STARTING new planes for a target once its forward begins (in-flight
  finish; unstarted dropped).
- **Predictor + wrapper** (`predictor.py`, `install.py`): the parameter-free predictor is
  reused — device `merged = max over rows of the next layer's native biased gate score`
  (`_gate_prefix`/`_gate_prefix_impl`, exactly what `Gate.__call__` ranks). Each source
  layer's `switch._run` is OUTER-wrapped: the wrapper computes `merged` for L+1 and evals it
  on the SAME barrier as indices (`mx.eval(indices, merged)`; the original scheduled run's
  `mx.eval(indices)` is then a no-op), sets the target's demand-imminent flag, calls the
  original run, then ranks on the host (top k=3 of L+1 not resident in its bank and not
  already in the ring) and enqueues those records' planes. Sources 3..38 → targets 4..39;
  layers 0..3 unpredicted; layer 39 (target-only) still sets its demand-imminent flag.

## Install-point analysis (file:line)

F2b installs as the LAST step of `observe_seed_prefill` (run_full.py:737, right after
`projection_owner_report.update(prime_model(target))`), via one anchored, round-trip-checked
staged edit of run_full.py calling `f2.install.install_from_env(target)` (no-op unless
`MTPLX_DSV41_F2B=1`).

- `projection_install.install_model` (projection_install.py:63-121, runs during
  growth_transition) validates `type(runner) is PackedDecode` and `switch._run.__func__ is
  PackedDecode.run` (:92-96) then rebinds `switch._run = MethodType(scheduled_run, runner)`
  (:108, `scheduled_run` = the `self.issue_next()` variant, scheduled_run_source :52-60) and
  sets `runner.issue_next` (:107).
- `prime_model` (projection_install.py:124-130, runs in observe_seed_prefill after
  `grow_rows`) only calls `store.issue(0)` + `mx.eval` — it does **NOT** re-validate
  `switch._run` or the reader.
- `verify_retirement` (projection_install.py:133-151, runs AFTER the measured request) checks
  only `attn._out_prep_fused_impl` (the ScheduledOutput lane) and the projection store — it
  does **NOT** touch `switch._run` or the reader.

So nothing re-validates `switch._run` or the reader after `prime_model`. The wrapper is an
outer wrap that CALLS the original scheduled `switch._run` (so `runner.issue_next` and the
projection scheduling stay intact) and reuses the runner instance; the reader intercept
reuses the lane's `local`. Compatible.

`observe_seed_prefill` has no hiding except; a failed F2b install propagates. The retained
`observe_prefill_boundary` except (run_full.py:782-783) raises a bare `SystemExit` with no
traceback — the stager inserts `traceback.print_exc()` there (failure path only) so a hidden
transition/install error is visible.

## Memory arithmetic

The ring is HOST memory, not MLX, and is NOT admitted through packed_admission. At R=32 the
ring reserves `32 × 3` plane buffers sized to the max plane (6,266,880 B) = **601,620,480 B**
of host RAM (the record's weight bytes are 17,694,720; the ring holds ~566–602 MB depending
on whether down planes use the full max buffer). One decode row per layer is
`40 × 17,694,720 = 707,788,800 B` of **MLX active** memory. The candidate arm runs
`F2_MAX_ROWS-1` rows and the controls `F2_MAX_ROWS`, so the candidate frees 707,788,800 B of
MLX active — larger than the host ring — hence

    physical = baseline + host + MLX_active
    candidate_total = baseline + (host + ring) + (MLX_active − 707,788,800)  ≤  control_total

i.e. the candidate's total physical footprint is ≤ a control's despite the ring (no admission
edit; the ring lives in host RAM under the 110 GB whole-machine budget, not the MLX cap).

## Counter schema (`<arm>/f2b_counters.json`, dumped once after decode via atexit)

Plain ints updated off the measured main thread (reader + speculative worker threads):
`planes_issued` (enqueued), `planes_completed` (speculative reads that reached READY),
`planes_hits` (demand plane served from a READY entry), `planes_waits` (demand plane that
waited on an in-flight READING entry then copied), `planes_cancelled` (demand plane that hit
a QUEUED entry → cancelled + preaded), `planes_wasted` (a READY entry recycled unread/
unconsumed), `bytes_speculative` (bytes read speculatively), `records_full` /
`records_partial` (records whose 3 / 1–2 planes reached READY). Not per-token proof counters
— aggregate engagement, AGENTS.md.

## Barrier count

Per source layer-call, unchanged at 1 routing + 1 miss-drain: the wrapper's
`mx.eval(indices, merged)` IS the routing barrier (forces indices + the prediction), and the
original scheduled run's `mx.eval(indices)` becomes a no-op. The host ranking + enqueue after
the run add no `mx` op; the speculative reads are on worker threads. Proven:
`test_wrapper_barrier_parity_one_eval`.

## CPU proofs (green this build)

`tests/test_dsv41_f2b.py` (15) + `tests/test_dsv41_f2_stage.py` (10) +
`tests/test_dsv41_f2_prefetch.py` (12) = **37 passed** (`PYTHONPATH=<worktree> nice -n 19
.venv/bin/python3 -m pytest …`). Coverage: ring READY-hit/QUEUED-cancel/READING-wait/
failure-evict/recycle-under-refcount; the derived reader == the retained reader when the ring
is empty (byte-identical destinations + identical metrics + identical early-witness ordering,
on a fake reader with the real offsets + tiny plane lengths); a READY plane serves from RAM
with NO pread; QUEUED → cancel + pread; the speculative pool fills the ring + window-stop
drops unstarted planes; the derivation round-trips and pins the retained lane sha; predictor
ranking == the offline scorer `f2_predictor.merge_rank_exclude`; wrapper barrier parity; and
end-to-end reader lane-on (ring pre-filled) vs lane-off (retained) → identical routed bytes.

The full preflight ran green (exit 0) against the REAL run worktree
(`.worktrees/dsv41-run-d5f15e7a`, HEAD == pin) + the archived sources: source pin (11 runtime
sources match), archived-helper shas vs the packed/compat `installation.json`, seam
resolution against the run worktree's classes, and the reader-intercept derivation +
`projection_install` resolution.

## AGENTS.md compliance

Construction-time install once (the lane is selected by which `switch._run`/reader is bound,
never a per-call check); the ring lookup is the lane's actual work, not an eligible-or-stock
fallback; engagement is read once after decode from aggregate ring counters; the pinned
runtime sources are untouched (the reader intercept is a derived-source rebind; the predictor
is an outer wrap); the retained reader logic cannot drift (anchored round-trip + sha pin).

## What stays unverified until the guarded window

The ring/reader/predictor/pool run on real objects on CPU. Only the full in-process run under
Metal is a window measurement: that the ring bytes read via `_readv_range_into` from the real
`experts.bin` equal the demand bytes (so the digest stays `0d54d9b2…417ac`), the throughput/
peak-GB/TTFT deltas, and the hit/wait/cancel/wasted mix on the real routing. Because the ring
serves the SAME bytes the pread would (proven byte-identical on CPU with a fake reader), the
digest must equal the control's; a mismatch is a bug, not a tie.

## Launch (orchestrator, under the guard — not the author)

```
bash /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f2-prefetch/scripts/deepseek_v41/run_f2_prefetch_window.sh
```

Runs the CPU preflight, then the equal-capacity ladder `control_a`@`F2_MAX_ROWS`(108) /
`candidate`@`F2_MAX_ROWS-1`+F2b / `control_b`@108 (optional `control_low`@107 via
`F2_INCLUDE_CONTROL_LOW=1`), each from the detached run worktree with a fresh
`/tmp/dsv41-110-stage/<stem>.jsonl` (sidecars copied into the arm dir), `guard.exit` handling
(0 continue; 4 = digest-mismatch FAILURE; else abort), `GPU_WINDOW_LOCK_TIMEOUT=${F2_LOCK_TIMEOUT:-7200}`,
the token-id sha256 gate, the per-arm `f2b_counters.json`, and each arm's exact expanded
command in `command.txt`. `F2_RING_RECORDS`/`F2_WORKERS`/`F2_ARMS` are overridable.
