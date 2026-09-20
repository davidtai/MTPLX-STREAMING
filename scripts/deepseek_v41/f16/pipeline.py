"""F16 verify-row-group pipeline: interleave two causal MTP-verify row groups so
one group's SSD miss-read wait overlaps the other group's routing barrier + submit.

The lever is admissible because splitting the M<=8 verify rows into group A (the
first 4 rows) and group B (the rest), with B attending to A's KV, is ALREADY a
supported arithmetic -- the retained decode loop's ``verify_chunks``.  Fable ran it
sequentially as (4,4): output digest
172830a96d84dbdac631c058fe0dfaf956df15b6c353f41fc05f604bc92c6393.  This module's
``pipelined_forward`` must be BIT-IDENTICAL to running ``forward(A)`` then
``forward(B)`` sequentially (same per-group row counts => same kernels), so a GPU
run reproduces that digest exactly.

How it interleaves WITHOUT restructuring the model, and WITHOUT a second thread
(MLX GPU streams are THREAD-BOUND -- a helper thread cannot touch the generation
thread's stream), so both groups run as GREENLETS on the ONE calling thread:
  * The driver (``pipelined_forward``'s frame) is the PARENT greenlet.  ``gA``/``gB``
    are child greenlets (group A / B forwards).  A group hands control back at ONE
    yield point -- ``self._f16_yield()`` -> ``greenlet.getcurrent().parent.switch()``
    -- inserted (by :func:`f16_run_source`) into the retained SCHEDULED
    ``PackedDecode.run`` immediately before ``ready_iter =
    pending.iter_ready_misses()`` (after the demand reads + hit gate_up + shared
    expert are submitted, before the completion loop blocks on the SSD reads).
  * ``_f16_yield`` is a no-op unless the current greenlet is a pipeline group (an
    ``_f16_group`` attribute on the greenlet, set only by the driver); no thread-local.
  * Driver order (leader A one layer ahead of trailer B): switch A -- A runs [layer 0
    to yield] but does NOT hand back at its first ``_f16_skip`` hand-offs (reads mode
    skips 1; F18 barrier mode skips 2 -- barrier(0) and reads(0), stopping at
    barrier(1)), continues [finish 0, layer 1]; then the driver alternates B, A, B, A ...;
    when one group is dead it keeps switching the other until it is dead too.  Because
    ``begin_split_route(L)`` takes the per-layer lock and it is released only by the
    NEXT run's ``flush_deferred_slot_releases`` (expert_runtime begin_split_route:4149
    acquire / _DeferredSplitClose->close:1552 release, flushed at plane_lane run head),
    B may enter layer L only after A entered layer L+1 -- so A has always written layer
    L's KV (via ``layer(...)``; LayerAttentionCache.advance:1479-1484 only moves the
    offset, the layer forward grows the store) before B reaches layer L.  Causality
    holds; ``cache.advance`` runs once, after both groups, in the driver.  Every MLX op
    of both groups issues on the ONE calling thread + stream, so acquire/release of the
    per-layer route locks and the deferred-close drain are all single-threaded (the
    ``flush_deferred_slot_releases`` "same generation thread" invariant is satisfied).

Faithfulness of the per-group forward: it is a hand clone of
``DeepseekV41Backbone._forward_span`` + the ``Model.__call__`` head/logits/main_hidden
tail, with exactly three differences (positions from a shared ``offset0``; engram
advanced up front for A then B and read per group through ``_ChunkEngramView`` as
``_forward_layer_major`` does; no per-group ``cache.advance``).  The device-route
recovery is asserted OFF (never cloned).  The source of every function this clone
mirrors is SHA-pinned below; :func:`verify_source_pins` fails loudly on drift, and
the CPU test diffs the clone's per-line arithmetic against the live source.

MLX-free at import except ``mlx.core``; CPU-testable.
"""
from __future__ import annotations

import contextvars
import hashlib
import inspect
import textwrap
from types import MethodType
from typing import Any, Callable

import mlx.core as mx
import greenlet  # private target dir .f16-site on PYTHONPATH (never the shared venv)

from . import stamps  # host-only diagnostic sink (no mx); active only when configured

# --- SHA pins: the retained/pinned sources this lever derives from or clones. ----
# A drift in any of these breaks an assumption; verify_source_pins / preflight fail.
RETAINED_PLANE_LANE_SHA256 = (
    "1acad9e24c37e5c618b2d8e2e98fb93eb94b5476d5c6de6fa0ee054db468ba54"
)
FORWARD_SPAN_SHA256 = (
    "c5750a4d25a216b562aeeef5d495dd071645704cc3d1ea735c7a7ff485b3f432"
)
MODEL_CALL_SHA256 = (
    "677f0d5025e3fd408e85ff2b37340ef0fee5372876ae4bd889ae43261aa2f2b6"
)
CHUNK_ENGRAM_VIEW_SHA256 = (
    "38567d1f60a766be987769cf3dc840d49e1123d945e743692d4c0945a9618b37"
)
FORWARD_LAYER_MAJOR_SHA256 = (
    "b0cc782462dbc8fb2cf5849f4f7f50d2697a2dc4457e9fca3a2b6685c7cac66e"
)
DEVICE_ROUTE_ACTIVE_SHA256 = (
    "1c8788ac30e9899d0f6799d0f508fc02e1195a0884b636debb36925fb9e31e8a"
)

# The verify row split. Forwards of <= A_ROWS rows are never split (single group). Above that
# the LEADER group size is a construction-time choice (``Pipeline(split=...)``):
#   "fixed4"   -> 4 rows, the rest trail        (oracle: sequential chunks (4, n-4))
#   "balanced" -> ceil(n/2) rows, floor(n/2)    (oracle: sequential chunks (ceil, floor))
# Balanced groups overlap better: with 4+2 on a 6-row cycle the trailer has ~2 ms of reads (and
# no misses at all in ~45% of its layer calls) while the leader's ~4 ms of reads outlast the
# trailer's ~2.5 ms of barrier + host work. Each split is its OWN arithmetic (per-group row
# counts pick the kernels), so each needs its own sequential oracle digest + tie classification.
A_ROWS = 4
SPLITS = {
    "fixed4": lambda rows: A_ROWS,
    "balanced": lambda rows: (rows + 1) // 2,
}
# Greenlet roles (leader runs one layer ahead of the trailer).
LEADER, TRAILER = 0, 1
NONGROUP = 2  # a non-pipeline caller (single-group forward / stock control); stamps role

# Construction-time hand-off mode (``MTPLX_DSV41_F16_HANDOFF``) -> how many hand-offs
# per layer, and how many the LEADER skips to get one layer ahead (``_f16_skip``):
#   "reads"   (default) = 1 hand-off/layer (the reads yield); leader skips 1.
#   "barrier" (F18)     = 2 hand-offs/layer (barrier + reads yield); leader skips 2
#                         (pass barrier(0) and reads(0), stop at barrier(1)).
HANDOFFS = {"reads": 1, "barrier": 2}

# The one anchor the reads yield is inserted before, and the inserted line.  The insert
# issues no ``mx`` op (a pure host greenlet hand-off), so the lane's MLX op sequence is
# unchanged -- the per-group forward is byte-identical to the unyielded scheduled run.
_YIELD_ANCHOR = "ready_iter = pending.iter_ready_misses()"
_YIELD_INSERT = "self._f16_yield()"

# F18: the barrier hand-off is inserted immediately BEFORE the retained blocking
# ``mx.eval(indices)``.  The inserted line issues no ``mx`` op; the single
# ``mx.async_eval(indices)`` lives in ``Pipeline.barrier`` (the one deliberate
# exception, documented there), never in the derived source.
_BARRIER_ANCHOR = "mx.eval(indices)"
_BARRIER_INSERT = "self._f16_barrier(indices)"

# F18 stamp inserts (diagnostic; only in the STAMPED derived variant).  Each is a pure
# host call, uniquely anchored, round-trip checked, and contains no ``mx`` text.
_STAMP_ENTRY_ANCHOR = "tokens = x.reshape(-1,5120)"          # after -> stamp 0
_STAMP_EVAL_ANCHOR = _BARRIER_ANCHOR                          # after mx.eval -> stamp 3
_STAMP_ROUTE_ANCHOR = "parts = tuple(self.executor.parts)"   # after -> route ctx + stamp 4
_STAMP_DEFER_ANCHOR = (
    "runtime.defer_slot_release(_DeferredSplitClose(pending,tuple(leased)),tuple(outputs))"
)  # before -> stamp 7
_STAMP_RETURN_ANCHOR = (
    "return mx.take(joined,order,axis=0).reshape((*indices.shape,5120)),shared"
)  # before -> stamp 8


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dedented_source(obj) -> str:
    return textwrap.dedent(inspect.getsource(obj))


# ---------------------------------------------------------------------------
# Source pins
# ---------------------------------------------------------------------------
def verify_source_pins() -> dict:
    """Assert the live pinned-runtime sources this lever clones/derives from match
    the SHAs pinned above.  Raises on the FIRST mismatch so a GPU window cannot run
    a lever built against a drifted backbone.  CPU-safe (pure text)."""
    from mtplx.models import deepseek_v41 as dv

    checks = {
        "_forward_span": (dv.DeepseekV41Backbone._forward_span, FORWARD_SPAN_SHA256),
        "Model.__call__": (dv.Model.__call__, MODEL_CALL_SHA256),
        "_ChunkEngramView": (dv._ChunkEngramView, CHUNK_ENGRAM_VIEW_SHA256),
        "_forward_layer_major": (
            dv.DeepseekV41Backbone._forward_layer_major,
            FORWARD_LAYER_MAJOR_SHA256,
        ),
        "_device_route_active": (
            dv.DeepseekV41Backbone._device_route_active,
            DEVICE_ROUTE_ACTIVE_SHA256,
        ),
    }
    report = {}
    for name, (fn, pin) in checks.items():
        got = _sha(_dedented_source(fn))
        if got != pin:
            raise RuntimeError(
                f"F16: pinned-runtime {name} drifted from the cloned source "
                f"(sha {got} != pinned {pin}); re-derive the F16 clone before a GPU run"
            )
        report[name] = got
    return report


# ---------------------------------------------------------------------------
# Yield-capable run derivation (from the retained SCHEDULED PackedDecode.run)
# ---------------------------------------------------------------------------
def _insert_line(source: str, anchor: str, insert: str, what: str, *, after: bool) -> str:
    """Insert ``insert`` on its own line before/after the UNIQUE ``anchor`` line,
    asserting the insert adds no ``mx`` op and a byte-for-byte round trip (removing the
    inserted line recovers ``source``).  The shared discipline for every F16/F18 line
    insertion (the yield, the barrier, and the stamps)."""
    if "mx." in insert:
        raise RuntimeError(f"F16 {what} insert would add an mx op: {insert!r}")
    if insert in source:
        raise RuntimeError(f"F16 {what} already inserted in this run source")
    orig = source.splitlines()
    hits = [i for i, ln in enumerate(orig) if ln.strip() == anchor]
    if len(hits) != 1:
        raise RuntimeError(
            f"F16 {what} anchor is not unique ({len(hits)}x): {anchor!r} -- "
            "the retained scheduled run changed; re-pin the F16 insertion"
        )
    i = hits[0]
    indent = orig[i][: len(orig[i]) - len(orig[i].lstrip())]
    lines = list(orig)
    lines.insert(i + 1 if after else i, indent + insert)
    recovered = [ln for ln in lines if ln.strip() != insert]
    if recovered != orig:
        raise RuntimeError(f"F16 {what} insertion changed the scheduled run beyond one line")
    return "\n".join(lines)


def f16_run_source(scheduled_source: str) -> str:
    """Insert ``self._f16_yield()`` immediately before the miss-completion iterator
    (the reads-mode single hand-off), unique anchor + round trip, no ``mx`` op."""
    return _insert_line(scheduled_source, _YIELD_ANCHOR, _YIELD_INSERT, "yield", after=False)


def f16_barrier_run_source(scheduled_source: str) -> str:
    """F18 (barrier mode): two pure line insertions from the SCHEDULED run source --
    ``self._f16_barrier(indices)`` immediately before the retained ``mx.eval(indices)``,
    and the reads ``self._f16_yield()`` before the miss iterator.  Each is uniquely
    anchored, round-trip checked and mx-free; the barrier's single ``async_eval`` lives
    in ``Pipeline.barrier``, not here."""
    step = _insert_line(scheduled_source, _BARRIER_ANCHOR, _BARRIER_INSERT, "barrier", after=False)
    return _insert_line(step, _YIELD_ANCHOR, _YIELD_INSERT, "yield", after=False)


def f16_stamp_run_source(derived_source: str) -> str:
    """Insert the six in-run stamp calls into an already reads/barrier-derived source
    (stamps 1/2/6 live in the stamped hand-off helpers).  Each insert is pure host,
    uniquely anchored, round-trip checked and mx-free."""
    s = _insert_line(derived_source, _STAMP_ENTRY_ANCHOR, "self._f16_stamp0(tokens)", "stamp0", after=True)
    s = _insert_line(s, _STAMP_EVAL_ANCHOR, "self._f16_stamp3()", "stamp3", after=True)
    s = _insert_line(s, _STAMP_ROUTE_ANCHOR, "self._f16_stamproute(experts, parts, pending)", "stamproute", after=True)
    s = _insert_line(s, _YIELD_INSERT, "self._f16_stamp5()", "stamp5", after=False)
    s = _insert_line(s, _STAMP_DEFER_ANCHOR, "self._f16_stamp7()", "stamp7", after=False)
    s = _insert_line(s, _STAMP_RETURN_ANCHOR, "self._f16_stamp8()", "stamp8", after=False)
    return s


def _scheduled_run_source() -> str:
    """The retained scheduled run source (``self.issue_next()`` variant that
    projection_install installs), derived from the pinned plane_lane at run time."""
    import plane_lane
    import projection_install

    lane_src = inspect.getsource(plane_lane)
    if _sha(lane_src) != RETAINED_PLANE_LANE_SHA256:
        raise RuntimeError("plane_lane.py differs from the pinned retained lane")
    base = _dedented_source(plane_lane.PackedDecode.run)
    return projection_install.scheduled_run_source(base)


def build_yield_run():
    """Compile the yield-capable run in plane_lane's namespace (so it closes over the
    identical helpers) from the retained scheduled source.  Returns (run_fn, shas)."""
    import plane_lane

    scheduled = _scheduled_run_source()
    yielded = f16_run_source(scheduled)
    namespace = dict(plane_lane.__dict__)
    exec(compile(yielded, "<f16_yield_run>", "exec"), namespace)  # noqa: S102
    return namespace["run"], {
        "scheduled_run_sha256": _sha(scheduled),
        "f16_yield_run_sha256": _sha(yielded),
        "retained_plane_lane_sha256": RETAINED_PLANE_LANE_SHA256,
    }


def build_barrier_run():
    """Compile the F18 barrier+yield run in plane_lane's namespace from the retained
    scheduled source.  Returns (run_fn, shas)."""
    import plane_lane

    scheduled = _scheduled_run_source()
    derived = f16_barrier_run_source(scheduled)
    namespace = dict(plane_lane.__dict__)
    exec(compile(derived, "<f16_barrier_run>", "exec"), namespace)  # noqa: S102
    return namespace["run"], {
        "scheduled_run_sha256": _sha(scheduled),
        "f16_barrier_run_sha256": _sha(derived),
        "retained_plane_lane_sha256": RETAINED_PLANE_LANE_SHA256,
    }


def build_stamped_run(*, barrier: bool):
    """Compile the STAMPED variant of the derived run (separate compiled function; the
    stamp inserts are pure host, round-trip checked).  Returns (run_fn, shas)."""
    import plane_lane

    scheduled = _scheduled_run_source()
    base = f16_barrier_run_source(scheduled) if barrier else f16_run_source(scheduled)
    stamped = f16_stamp_run_source(base)
    namespace = dict(plane_lane.__dict__)
    exec(compile(stamped, "<f16_stamped_run>", "exec"), namespace)  # noqa: S102
    key = "f16_stamped_barrier_run_sha256" if barrier else "f16_stamped_reads_run_sha256"
    return namespace["run"], {
        "scheduled_run_sha256": _sha(scheduled),
        key: _sha(stamped),
        "stamped_barrier": bool(barrier),
        "retained_plane_lane_sha256": RETAINED_PLANE_LANE_SHA256,
    }


def scheduled_run_cocode() -> bytes:
    """Bytecode of the retained scheduled run, for the install-time refuse check
    (F16 stage 1 is mutually exclusive with F2b and the F5 stamp probe: the enabled
    hot path must be the retained scheduled run, not a wrapped/stamped variant)."""
    import plane_lane

    scheduled = _scheduled_run_source()
    namespace = dict(plane_lane.__dict__)
    exec(compile(scheduled, "<f16_scheduled_ref>", "exec"), namespace)  # noqa: S102
    return namespace["run"].__code__.co_code


# ---------------------------------------------------------------------------
# Greenlet hand-off: read by the injected run (_f16_yield) + the wrapped issue_next.
# ---------------------------------------------------------------------------
class _PipelineAborted(BaseException):
    """Thrown by the driver into the SUSPENDED partner greenlet when the other group
    raised, so ``PackedDecode.run``'s ``except BaseException`` cleanup (mx.synchronize,
    pending.abort, pending.close) runs and its layer lock is released before we re-raise
    the original error.  A BaseException so ``except Exception`` cannot swallow it."""


def _f16_handoff(g) -> None:
    """Hand control back to the driver, unless one of the leader's head-start hand-offs
    is still pending (``_f16_skip`` counts them down).  Shared by both hand-off kinds
    (the barrier and the reads yield); O(1), allocation-free."""
    if getattr(g, "_f16_skip", 0) > 0:
        g._f16_skip -= 1
        return
    g.parent.switch()  # cooperative hand-off back to the driver (same thread)


def f16_yield() -> None:
    """The reads hand-off point, bound onto every runner as ``_f16_yield``.  A no-op
    unless the current greenlet is a pipeline group (O(1), allocation-free attribute
    reads); the leader's ``_f16_skip`` head start keeps it one layer ahead."""
    g = greenlet.getcurrent()
    if not getattr(g, "_f16_group", False):
        return
    _f16_handoff(g)


def f16_yield_stamped() -> None:
    """Stamped reads hand-off (bound as ``_f16_yield`` only in stamps mode): the reads
    hand-off plus stamp 6 (resumed after the reads hand-off).  Pure host beyond the
    identical greenlet switch."""
    g = greenlet.getcurrent()
    if not getattr(g, "_f16_group", False):
        return
    _f16_handoff(g)
    stamps.stamp(getattr(g, "_f16_rec", None), 6)


# -- in-run stamp helpers (bound per runner as MethodType only in stamps mode) --------
def _stamp0(runner, tokens) -> None:
    g = greenlet.getcurrent()
    rec = stamps.begin(getattr(g, "_f16_role", NONGROUP), runner.layer, int(tokens.shape[0]))
    g._f16_rec = rec  # current run's record; read by the later stamps + hand-off helpers


def _stamp3(runner) -> None:
    stamps.stamp(getattr(greenlet.getcurrent(), "_f16_rec", None), 3)


def _stamproute(runner, experts, parts, pending) -> None:
    rec = getattr(greenlet.getcurrent(), "_f16_rec", None)
    hit = getattr(pending, "hit_ready", None)
    n_hits = len(hit.bindings) if hit is not None else 0
    stamps.annotate(rec, n_parts=len(parts), n_hits=n_hits, n_unique=len(set(experts)))
    stamps.stamp(rec, 4)


def _stamp5(runner) -> None:
    stamps.stamp(getattr(greenlet.getcurrent(), "_f16_rec", None), 5)


def _stamp7(runner) -> None:
    stamps.stamp(getattr(greenlet.getcurrent(), "_f16_rec", None), 7)


def _stamp8(runner) -> None:
    stamps.stamp(getattr(greenlet.getcurrent(), "_f16_rec", None), 8)


def issue_suppressed() -> bool:
    """True on the trailer group's greenlet (the leader already issued that layer's
    next-projection; re-issuing would clobber a buffer the leader still needs)."""
    return getattr(greenlet.getcurrent(), "_f16_suppress_issue", False)


# ---------------------------------------------------------------------------
# The pipeline object (injected into the hybrid decode namespace as _F16_PIPELINE)
# ---------------------------------------------------------------------------
class Pipeline:
    """Holds the target model and drives the two-group verify pipeline.

    When ``armed`` is False (MTPLX_DSV41_F16 != 1) ``pipelined_forward`` is a pure
    passthrough to the retained ``forward`` -- an explicit construction-time route,
    so a staged tree run with F16 off is a byte-for-byte stock A/B control.
    """

    def __init__(self, model, *, armed: bool, split: str = "fixed4", handoff: str = "reads"):
        self.model = model
        self.backbone = model.model
        self.armed = bool(armed)
        if split not in SPLITS:
            raise RuntimeError(f"F16 split must be one of {sorted(SPLITS)}; got {split!r}")
        self.split = split
        if handoff not in HANDOFFS:
            raise RuntimeError(
                f"F16 handoff must be one of {sorted(HANDOFFS)}; got {handoff!r}"
            )
        self.handoff = handoff
        self._skip = HANDOFFS[handoff]             # leader head-start hand-offs; bound once
        self._leader_rows = SPLITS[split]          # bound once; no per-call mode branch
        self._orphans: list = []                   # dead group's deferred closes (barrier mode)
        self.counters = {
            "calls": 0,
            "single_forwards": 0,
            "pipelined_forwards": 0,
            "handoffs": 0,  # number of driver greenlet switches (both hand-off kinds)
        }
        # Bind the driver strategy ONCE at construction (no per-switch mode test): reads
        # keeps today's shared-list driver; barrier installs the per-group deferred-list
        # driver with orphan hand-off (design 3.4).  Stamps do not change the driver.
        if handoff == "barrier":
            self._run_groups = self._run_groups_barrier

    # -- entry (replaces the ONE verify forward call) -----------------------
    def pipelined_forward(self, forward: Callable[[Any, Any], tuple], ids, cache):
        if not self.armed:
            return forward(ids, cache)
        self.counters["calls"] += 1
        rows = int(ids.shape[1])
        if rows <= A_ROWS:
            # Runtime M-route (logical row count genuinely varies): a single group
            # needs no split/greenlets.  Runs the rebound yield run; _f16_yield no-ops
            # (the calling greenlet has no _f16_group attribute).
            self.counters["single_forwards"] += 1
            return forward(ids, cache)
        self.counters["pipelined_forwards"] += 1
        return self._run_pipeline(ids, cache)

    # -- two-group interleave ----------------------------------------------
    def _run_pipeline(self, ids, cache):
        backbone = self.backbone
        rows = int(ids.shape[1])
        rows_a = int(self._leader_rows(rows))
        rows_b = rows - rows_a
        ids_a, ids_b = ids[:, :rows_a], ids[:, rows_a:]
        offset0 = int(cache.offset)

        # The clone omits the device-route cold-recovery block, so it MUST be off.
        if backbone._device_route_active(cache, rows_a) or backbone._device_route_active(
            cache, rows_b
        ):
            raise RuntimeError("F16 pipeline requires the device route OFF")

        # Admit the whole forward once, up front (mirrors Model->backbone.__call__).
        admit = getattr(cache, "assert_can_admit", None)
        if callable(admit):
            admit(rows_a + rows_b)

        # Engram history advanced for A then B up front, in position order (identical
        # buffer/length to running the two spans sequentially -- the _forward_layer_
        # major discipline); each group reads its own captured advance via a view.
        engram_state = getattr(cache, "engram_state", None)
        if engram_state is not None:
            cur_a = engram_state.advance(ids_a)
            cur_b = engram_state.advance(ids_b)
        else:
            cur_a = cur_b = None

        results = self._run_groups(
            lambda: self._group_forward(
                ids_a, cache, offset0=offset0, start=0, engram_current=cur_a
            ),
            lambda: self._group_forward(
                ids_b, cache, offset0=offset0, start=rows_a, engram_current=cur_b
            ),
        )

        # Both groups complete: advance the cache once, then concatenate exactly as
        # the sequential chunk loop concatenates its parts.
        cache.advance(rows_a + rows_b)
        logits_a, mh_a = results[LEADER]
        logits_b, mh_b = results[TRAILER]
        logits = mx.concatenate([logits_a, logits_b], axis=1)
        main_hidden = (
            None
            if (mh_a is None or mh_b is None)
            else mx.concatenate([mh_a, mh_b], axis=1)
        )
        return logits, main_hidden

    # -- drive two group callables as cooperative greenlets on THIS thread ------
    def _run_groups(self, group_leader, group_trailer) -> list:
        """Run two zero-arg group callables as child greenlets of the calling
        (driver) greenlet, on the ONE calling thread and stream.  The leader runs one
        layer ahead (it skips its first yield); thereafter the driver alternates.  On a
        group error the driver throws ``_PipelineAborted`` into the still-suspended
        partner so its ``PackedDecode.run`` cleanup releases its layer lock, then
        re-raises the original error.  A suspended run is NEVER dropped (its pending
        route would wedge the layer lock).  Returns ``[leader_result, trailer_result]``.
        Each greenlet inherits the caller's routing context via ``gr_context`` (a fresh
        greenlet otherwise starts with an EMPTY context -> the phase ContextVars would
        be lost -> a verify group would route as PREFILL)."""
        results: list = [None, None]

        def wrap(role, fn):
            def run():
                results[role] = fn()

            return run

        gA = greenlet.greenlet(wrap(LEADER, group_leader))
        gA.gr_context = contextvars.copy_context()
        gA._f16_group = True
        gA._f16_skip = self._skip                 # reads = 1; leader skips its first yield
        gA._f16_suppress_issue = False
        gA._f16_role = LEADER
        gB = greenlet.greenlet(wrap(TRAILER, group_trailer))
        gB.gr_context = contextvars.copy_context()
        gB._f16_group = True
        gB._f16_skip = 0
        gB._f16_suppress_issue = True
        gB._f16_role = TRAILER

        switches = 0
        try:
            gA.switch()  # prime the leader (skips its first yield -> runs to the 2nd)
            switches += 1
            while not (gA.dead and gB.dead):
                if not gB.dead:
                    gB.switch()
                    switches += 1
                if not gA.dead:
                    gA.switch()
                    switches += 1
        finally:
            # A raising group leaves the OTHER suspended inside run(); unwind it so its
            # BaseException route cleanup runs (else its pending route wedges the lock).
            for g in (gB, gA):
                if not g.dead:
                    try:
                        g.throw(_PipelineAborted)
                    except BaseException:  # noqa: BLE001 - cleanup ran; original re-raises
                        pass
        self.counters["handoffs"] += switches
        return results

    # -- F18 barrier hand-off (bound onto every runner as _f16_barrier) ---------
    def _barrier_adopt_orphans(self) -> None:
        """Adopt a dead partner's remaining deferred closes into the runtime's CURRENT
        (this group's, driver-installed) deferred list, BEFORE the barrier submit -- so
        the orphan waves precede this barrier graph in the single stream and this
        group's post-``mx.eval`` flush covers them (design 3.4 rule 2).  This frees the
        last layer's lock for the trailer; without it the trailer deadlocks at layer 39.
        Appended (not fenced): each release is independent and the covering eval bounds
        them all.  ``_mtplx_expert_runtime`` may be absent on a CPU model with no
        streamed switch -- then there are no real closes and this is a no-op."""
        orphans = self._orphans
        if not orphans:
            return
        runtime = getattr(self.model, "_mtplx_expert_runtime", None)
        if runtime is None:
            return
        pending = getattr(runtime, "_deferred_slot_releases", None)
        if pending is None:
            pending = []
            runtime._deferred_slot_releases = pending
        pending.extend(orphans)
        self._orphans = []

    def barrier(self, indices) -> None:
        """Submit the routing barrier early and hand off so the partner group uses the
        GPU wait.  ``mx.async_eval(indices)`` adds NO array and NO op: it submits the
        exact graph the next line's retained ``mx.eval(indices)`` waits on, only earlier
        -- the one deliberate ``mx`` call the barrier lane issues, and it lives in this
        helper, never in the derived source.  A non-group caller (single-group forward /
        stock control) returns at once with NO async_eval: the retained blocking
        ``mx.eval(indices)`` follows unchanged, so that path stays byte-identical."""
        g = greenlet.getcurrent()
        if not getattr(g, "_f16_group", False):
            return
        self._barrier_adopt_orphans()
        mx.async_eval(indices)
        _f16_handoff(g)

    def _barrier_stamped(self, indices) -> None:
        """Stamped barrier (bound as ``_f16_barrier`` only in stamps+barrier mode): the
        barrier hand-off plus stamp 1 (after async_eval = encode) and stamp 2 (resumed
        after the hand-off).  Identical MLX behaviour to :meth:`barrier`."""
        g = greenlet.getcurrent()
        if not getattr(g, "_f16_group", False):
            return
        self._barrier_adopt_orphans()
        mx.async_eval(indices)
        stamps.stamp(getattr(g, "_f16_rec", None), 1)
        _f16_handoff(g)
        stamps.stamp(getattr(g, "_f16_rec", None), 2)

    # -- F18 driver: per-group deferred lists + orphan hand-off (design 3.4) ----
    def _run_groups_barrier(self, group_leader, group_trailer) -> list:
        """Like :meth:`_run_groups`, but each group OWNS its deferred-close list, which
        the driver swaps onto ``runtime._deferred_slot_releases`` around every hand-off,
        so a group's evaluate-free flush only ever releases closes whose waves precede
        that group's own ``mx.eval(indices)`` (the per-group stream-FIFO argument that
        replaces the shared-list one, which barrier mode breaks -- a partner runs a slice
        between this group's async_eval and eval).  A dead group's remaining list becomes
        ``self._orphans`` (adopted by the survivor inside ``barrier`` before its submit).
        At the end -- normal or error -- every remaining close is put back on the runtime
        list so the retained boundary flush (``flush_deferred_slot_releases``) behaves
        exactly as today; nothing is dropped.  The leader inherits whatever list the
        runtime held when the forward began (those waves precede everything)."""
        runtime = getattr(self.model, "_mtplx_expert_runtime", None)
        results: list = [None, None]

        def wrap(role, fn):
            def run():
                results[role] = fn()

            return run

        gA = greenlet.greenlet(wrap(LEADER, group_leader))
        gA.gr_context = contextvars.copy_context()
        gA._f16_group = True
        gA._f16_skip = self._skip                 # barrier = 2 (pass barrier(0)+reads(0))
        gA._f16_suppress_issue = False
        gA._f16_role = LEADER
        gB = greenlet.greenlet(wrap(TRAILER, group_trailer))
        gB.gr_context = contextvars.copy_context()
        gB._f16_group = True
        gB._f16_skip = 0
        gB._f16_suppress_issue = True
        gB._f16_role = TRAILER

        if runtime is not None:
            state = {gA: getattr(runtime, "_deferred_slot_releases", None), gB: None}
            runtime._deferred_slot_releases = None
        else:
            state = {gA: None, gB: None}
        self._orphans = []

        def switch(g, *, throw=False):
            if runtime is not None:
                runtime._deferred_slot_releases = state[g]
            try:
                if throw:
                    g.throw(_PipelineAborted)
                else:
                    g.switch()
            finally:
                if runtime is not None:
                    state[g] = getattr(runtime, "_deferred_slot_releases", None)
                    runtime._deferred_slot_releases = None
                if g.dead:
                    self._orphans.extend(state[g] or [])
                    state[g] = None

        switches = 0
        try:
            try:
                switch(gA)  # prime the leader (skips barrier(0) + reads(0))
                switches += 1
                while not (gA.dead and gB.dead):
                    if not gB.dead:
                        switch(gB)
                        switches += 1
                    if not gA.dead:
                        switch(gA)
                        switches += 1
            finally:
                # A raising group leaves the OTHER suspended; unwind it (its list is
                # installed first) so its BaseException route cleanup runs.
                for g in (gB, gA):
                    if not g.dead:
                        try:
                            switch(g, throw=True)
                        except BaseException:  # noqa: BLE001 - cleanup ran; original re-raises
                            pass
        finally:
            # Rule 3: put every remaining close back on the runtime list (nothing
            # dropped) so the retained boundary flush behaves exactly as today.
            if runtime is not None:
                remaining: list = []
                remaining.extend(state[gA] or [])
                remaining.extend(state[gB] or [])
                remaining.extend(self._orphans)
                existing = getattr(runtime, "_deferred_slot_releases", None) or []
                runtime._deferred_slot_releases = (
                    remaining + list(existing) if (remaining or existing) else None
                )
            self._orphans = []
        self.counters["handoffs"] += switches
        return results

    # -- per-group forward: faithful clone of _forward_span + Model head tail -
    def _group_forward(self, ids_group, cache, *, offset0, start, engram_current):
        """One verify row group through every layer, appending to the shared cache
        WITHOUT advancing it, and returning ``(logits, main_hidden)`` -- the exact
        tuple ``model(ids, cache=cache, return_hidden=True)`` returns for these rows.

        Byte-for-byte vs ``_forward_span`` + ``Model.__call__`` (verify path:
        emit_logits=True, logits_keep/logits_rows=None -> head every row) except:
          (i)  positions from the shared ``offset0`` + ``start`` (not cache.offset),
          (ii) engram advanced up front by the driver; this reads its captured rows
               through ``_ChunkEngramView`` (as ``_forward_layer_major`` does),
          (iii) no ``cache.advance`` here (the driver advances once, after both).
        The device-route recovery is asserted off (never entered).  Host-only stage/
        timeline instrumentation (``_stime``/``_tl``, no mx ops, assumes the single
        generation thread) is omitted; the arithmetic is unchanged."""
        from mtplx.models.deepseek_v41 import _ChunkEngramView, _rmsnorm

        backbone = self.backbone
        model = self.model
        b, s = ids_group.shape

        positions = mx.arange(offset0 + start, offset0 + start + s)  # (i)

        h = backbone.embed_tokens(ids_group)
        h = mx.broadcast_to(h[:, :, None, :], (b, s, backbone.hc_mult, h.shape[-1]))
        pre_mix = mx.concatenate(
            [mx.ones((b, s, 1)), mx.zeros((b, s, backbone.hc_mult - 1))], axis=-1
        ).astype(mx.float32)

        engram_state = getattr(cache, "engram_state", None)
        engram_view = (
            _ChunkEngramView(engram_current) if engram_current is not None else None
        )
        shared = cache.new_shared_runtime()  # own shared runtime per group
        want_main = bool(backbone._mtp_target_layer_ids)
        main_hiddens: list = []

        if backbone._device_route_active(cache, s):  # never cloned; must be off
            raise RuntimeError("F16 group forward requires the device route OFF")

        for layer in backbone.layers:
            if layer.engram_hook is not None and engram_state is not None:
                h = layer.engram_hook(h, ids_group, engram_view)  # (ii)
            if want_main and layer.layer_id in backbone._mtp_target_layer_ids:
                main_hiddens.append(
                    mx.mean(h.astype(mx.float32), axis=2).astype(h.dtype)
                )
            h, pre_mix = layer(
                h, pre_mix, positions, cache.layers[layer.layer_id], shared
            )
        # (iii) no cache.advance here.

        h = mx.sum(pre_mix[..., None] * h.astype(mx.float32), axis=2).astype(h.dtype)
        out = _rmsnorm(h, backbone.norm_weight, backbone.args.rms_norm_eps)
        main_hidden = mx.concatenate(main_hiddens, axis=-1) if main_hiddens else None

        # Model.__call__ tail for the verify path: head every row, return both.
        logits = model._apply_head(out)
        return logits, main_hidden


class LazyPipeline:
    """What the staged hybrid rewrite injects as ``_F16_PIPELINE`` (review 2026-09-19).

    ``hybrid_install.install`` runs at the START of the DSpark pass -- before prefill -- while
    ``f16.install.install_from_env`` stashes ``model._f16_pipeline`` only at the post-prime
    boundary (after prefill + growth + seed). Binding ``model._f16_pipeline`` at hybrid-install
    time would raise AttributeError before the prefill even starts, so the injected object
    resolves the pipeline at CALL time; the first verify forward happens after the install."""

    __slots__ = ("_model",)

    def __init__(self, model) -> None:
        self._model = model

    def pipelined_forward(self, forward, ids, cache):
        return self._model._f16_pipeline.pipelined_forward(forward, ids, cache)


def bind_yield_run(runners) -> dict:
    """Rebind each retained ``PackedDecode`` runner's ``switch._run`` to the
    yield-capable run and give each runner an ``_f16_yield`` (the shared hand-off).
    Reuses the existing runner instances (``issue_next``, executor, ops preserved).
    ``runners`` maps layer -> (switch, runner)."""
    run_fn, shas = build_yield_run()
    bound = 0
    for _layer, (switch, runner) in runners.items():
        runner._f16_yield = f16_yield
        switch._run = MethodType(run_fn, runner)
        bound += 1
    shas["yield_run_bound_layers"] = bound
    return shas


def bind_barrier_run(runners, pipeline) -> dict:
    """F18 barrier mode: rebind each runner's ``switch._run`` to the barrier+yield run
    and give each runner both hand-off helpers -- ``_f16_yield`` (shared reads hand-off)
    and ``_f16_barrier`` (this pipeline's bound ``barrier``)."""
    run_fn, shas = build_barrier_run()
    bound = 0
    for _layer, (switch, runner) in runners.items():
        runner._f16_yield = f16_yield
        runner._f16_barrier = pipeline.barrier   # bound method (adopts orphans, submits)
        switch._run = MethodType(run_fn, runner)
        bound += 1
    shas["barrier_run_bound_layers"] = bound
    return shas


def _bind_stamp_helpers(runner) -> None:
    runner._f16_stamp0 = MethodType(_stamp0, runner)
    runner._f16_stamp3 = MethodType(_stamp3, runner)
    runner._f16_stamproute = MethodType(_stamproute, runner)
    runner._f16_stamp5 = MethodType(_stamp5, runner)
    runner._f16_stamp7 = MethodType(_stamp7, runner)
    runner._f16_stamp8 = MethodType(_stamp8, runner)


def bind_stamped_run(runners, pipeline, *, barrier: bool) -> dict:
    """Diagnostic (``MTPLX_DSV41_F16_STAMPS``): rebind each runner's ``switch._run`` to
    the STAMPED variant of the derived run and bind the stamped hand-off + in-run stamp
    helpers.  Selected once at construction; the sink is host-only (no mx)."""
    run_fn, shas = build_stamped_run(barrier=barrier)
    bound = 0
    for _layer, (switch, runner) in runners.items():
        runner._f16_yield = f16_yield_stamped
        if barrier:
            runner._f16_barrier = pipeline._barrier_stamped
        _bind_stamp_helpers(runner)
        switch._run = MethodType(run_fn, runner)
        bound += 1
    shas["stamped_run_bound_layers"] = bound
    return shas
