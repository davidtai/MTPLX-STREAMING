"""F5 arm A2: a zero-distortion critical-path stamp probe over the retained
packed decode lane.

``TimedPackedDecode`` records ``time.perf_counter_ns()`` at six points inside the
EXACT retained ``plane_lane.PackedDecode.run`` (extension-bank-20260919,
sha256 1acad9e24c37e5c618b2d8e2e98fb93eb94b5476d5c6de6fa0ee054db468ba54), with
NO extra ``mx.eval`` / ``mx.async_eval`` / ``mx.synchronize`` and no new arrays.
The stamped ``run`` is not hand-written: it is DERIVED from the installed run's
source by pure line-insertion (:func:`stamp_run_source`), which asserts that
removing the inserted stamp/metadata lines recovers the original byte-for-byte and
that no inserted line issues an ``mx`` op.  So the timed lane issues exactly the
same MLX operations in the same order as the untimed lane -- the property the CPU
test (tests/models/test_deepseek_v41_f5_timed_plane_lane.py) pins.

Stamp points (per the F5 addendum):
  t0  entry to run
  t1  after mx.eval(indices)                 (routing barrier)
  t2  after runtime.begin_split_route(...)   (demand reads submitted)
  t3  after the hit gate_up/down + shared_work async_eval (before miss iterate)
  t4  after the completion loop              (all miss parts consumed)
  t5  just before return                     (post: concat/argsort/finally)
Per call it also records (layer, rows, n_parts, n_hits) via a pure-host ``_note``.

Buckets derived in post: barrier=t1-t0, host_pre_read=t2-t1, hit_submit=t3-t2,
miss_wait=t4-t3, post=t5-t4, and inter_layer_build = next-call.t0 - this-call.t5
WITHIN one verify forward (the gap ACROSS forwards is draft/accept/commit and is
reported separately).

This module composes with the retained projection scheduling install: when
``scheduled=True`` the stamps are inserted into
``scheduled_run_source(PackedDecode.run)`` (the ``self.issue_next()`` variant that
projection_install.py installs), so the timed lane keeps the next-layer projection
expansion.  The stamp anchors are present in both the plain and scheduled sources.

GPU-only at runtime (the lane dispatches Metal kernels); this file is import-safe
and CPU-testable because :func:`stamp_run_source` is a pure text transform.
"""
from __future__ import annotations

import gzip
import json
import statistics
from time import perf_counter_ns
from typing import Any


# The retained lane the stamps are pinned to.  A structural drift (upstream edit
# to plane_lane.PackedDecode.run) breaks the unique-anchor check in
# stamp_run_source, so this pin fails loudly rather than timing a changed lane.
RETAINED_PLANE_LANE_SHA256 = (
    "1acad9e24c37e5c618b2d8e2e98fb93eb94b5476d5c6de6fa0ee054db468ba54"
)

# (stripped anchor line, "before"|"after", inserted stripped statement).
# Every inserted statement is a pure-host call (self._stamp / self._note); NONE
# issues an mx op, so the MLX op sequence is unchanged.  Each anchor must appear
# EXACTLY once in the run source (plain or scheduled), else the transform raises.
_STAMP_EDITS = (
    ("runtime, ops = self.runtime, self.ops", "before", "self._stamp(0)"),
    ("mx.eval(indices)", "after", "self._stamp(1)"),
    ("pending = runtime.begin_split_route(self.layer,experts,phase=RoutingPhase.DECODE)",
     "after", "self._stamp(2)"),
    ("parts = tuple(self.executor.parts)", "after",
     "self._note(self.layer, tokens.shape[0], len(parts), pending)"),
    ("ready_iter = pending.iter_ready_misses()", "before", "self._stamp(3)"),
    ("for _ in ready_iter:", "before", "self._stamp(4)"),
    ("return mx.take(joined,order,axis=0).reshape((*indices.shape,5120)),shared",
     "before", "self._stamp(5)"),
)

_INSERTED = frozenset(ins for _, _, ins in _STAMP_EDITS)

# Names an inserted line may reference (pure host).  Guards the "no mx op added"
# invariant: an inserted line must not contain "mx." (an mx dispatch).
_FORBIDDEN_IN_INSERT = ("mx.",)


def stamp_run_source(source: str) -> str:
    """Return ``source`` (a ``PackedDecode.run`` body, plain or scheduled) with the
    six timing stamps + one metadata note inserted, asserting op-sequence identity.

    Guarantees, all checked here so a caller can trust the result:
      * each anchor line occurs exactly once (unique insertion point);
      * no inserted line issues an ``mx`` op (only ``self._stamp`` / ``self._note``);
      * removing every inserted line recovers ``source`` line-for-line.
    """
    for _, _, ins in _STAMP_EDITS:
        for bad in _FORBIDDEN_IN_INSERT:
            if bad in ins:
                raise RuntimeError(f"stamp line would add an mx op: {ins!r}")
    orig_lines = source.splitlines()
    lines = list(orig_lines)
    for anchor, position, ins in _STAMP_EDITS:
        hits = [i for i, ln in enumerate(lines) if ln.strip() == anchor]
        if len(hits) != 1:
            raise RuntimeError(
                f"packed run anchor is not unique ({len(hits)}x): {anchor!r} -- "
                "the retained lane changed; re-pin the stamp probe"
            )
        i = hits[0]
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        lines.insert(i if position == "before" else i + 1, indent + ins)
    recovered = [ln for ln in lines if ln.strip() not in _INSERTED]
    if recovered != orig_lines:
        raise RuntimeError(
            "stamp insertion changed the packed run body beyond the stamp lines"
        )
    return "\n".join(lines)


class ProbeSink:
    """Shared, append-only collector for all layers' stamps in call order.

    A single sink across the 40 layer runners gives ONE global timeline, so the
    post pass can pair this-call.t5 with next-call.t0 to recover the inter-layer
    graph-build gap within a verify forward.  Only Python list appends on the hot
    path (no arrays, no eval)."""

    __slots__ = ("t_layer", "t_idx", "t_ns", "m_layer", "m_rows", "m_parts", "m_hits")

    def __init__(self) -> None:
        # One row per stamp (6 per run call), in global call order.
        self.t_layer: list[int] = []
        self.t_idx: list[int] = []
        self.t_ns: list[int] = []
        # One row per run call.
        self.m_layer: list[int] = []
        self.m_rows: list[int] = []
        self.m_parts: list[int] = []
        self.m_hits: list[int] = []

    def stamp(self, layer: int, idx: int) -> None:
        self.t_layer.append(layer)
        self.t_idx.append(idx)
        self.t_ns.append(perf_counter_ns())

    def note(self, layer: int, rows: int, parts: int, pending: Any) -> None:
        hit = getattr(pending, "hit_ready", None)
        self.m_layer.append(int(layer))
        self.m_rows.append(int(rows))
        self.m_parts.append(int(parts))
        self.m_hits.append(0 if hit is None else len(hit.bindings))


def _build_timed_run(base_run_source: str, *, scheduled: bool):
    """Compile the timed run from the installed run's source.

    ``scheduled`` selects the projection-scheduled source (``self.issue_next()``
    inserted, exactly as projection_install.py builds it) so arm A2 keeps the
    retained next-layer projection expansion.  Returns the compiled ``run`` fn and
    the two sha256s so the receipt can prove what was timed."""
    import hashlib
    import textwrap

    import plane_lane  # retained lane (receipt sources on PYTHONPATH)

    base = textwrap.dedent(base_run_source)
    if scheduled:
        from projection_install import scheduled_run_source
        base = scheduled_run_source(base)
    timed = stamp_run_source(base)
    namespace = dict(plane_lane.__dict__)
    exec(compile(timed, "<timed_packed_run>", "exec"), namespace)  # noqa: S102
    return (
        namespace["run"],
        hashlib.sha256(base.encode()).hexdigest(),
        hashlib.sha256(timed.encode()).hexdigest(),
    )


def make_timed_decode_class(*, scheduled: bool):
    """Build a ``TimedPackedDecode`` subclass whose ``run`` is the stamped variant
    of the installed lane.  Kept a factory so the plain and scheduled variants are
    distinct classes with their own compiled ``run`` (no per-call branch)."""
    import inspect

    import plane_lane

    base_src = inspect.getsource(plane_lane.PackedDecode.run)
    run_fn, base_sha, timed_sha = _build_timed_run(base_src, scheduled=scheduled)

    class TimedPackedDecode(plane_lane.PackedDecode):
        """PackedDecode with the six critical-path stamps; identical MLX ops."""

        base_run_sha256 = base_sha
        timed_run_sha256 = timed_sha

        def __init__(self, *args, sink: ProbeSink, **kwargs):
            super().__init__(*args, **kwargs)
            self._sink = sink

        def _stamp(self, idx: int) -> None:
            self._sink.stamp(self.layer, idx)

        def _note(self, layer, rows, parts, pending) -> None:
            self._sink.note(layer, rows, parts, pending)

    TimedPackedDecode.run = run_fn
    return TimedPackedDecode


def summarize(sink: ProbeSink, *, verify_forward_layers: int) -> dict:
    """Reduce the raw stamps to per-bucket mean/p50/p90 (seconds), split by rows M
    and by layer index, plus per-run totals that reconcile with verify_ms.

    ``verify_forward_layers`` = the number of routed-layer run calls in ONE verify
    forward (40 for the retained backbone).  A new forward starts every
    ``verify_forward_layers`` calls; the t0-minus-previous-t5 gap that STRADDLES a
    forward boundary is draft/accept/commit and is bucketed separately."""
    t_layer, t_idx, t_ns = sink.t_layer, sink.t_idx, sink.t_ns
    n_stamp = len(t_ns)
    if n_stamp % 6 != 0:
        raise RuntimeError(f"stamp stream not a multiple of 6 ({n_stamp})")
    n_calls = n_stamp // 6
    if len(sink.m_layer) != n_calls:
        raise RuntimeError("metadata/stamp call count mismatch")

    # Reshape into per-call [t0..t5]; verify the phase indices are 0..5 in order.
    calls = []
    for c in range(n_calls):
        base = c * 6
        idxs = t_idx[base:base + 6]
        if idxs != [0, 1, 2, 3, 4, 5]:
            raise RuntimeError(f"call {c} stamp order corrupt: {idxs}")
        calls.append(t_ns[base:base + 6])

    NS = 1e-9
    bucket_names = ("barrier", "host_pre_read", "hit_submit", "miss_wait", "post")
    per_call_buckets = []   # list of (layer, rows, [5 bucket seconds])
    for c in range(n_calls):
        ts = calls[c]
        b = [(ts[k + 1] - ts[k]) * NS for k in range(5)]
        per_call_buckets.append((sink.m_layer[c], sink.m_rows[c], b))

    # inter-layer graph build: this-call.t5 -> next-call.t0, only when the next
    # call is the SAME verify forward (call index not on a forward boundary).
    inter_same, inter_across = [], []
    for c in range(n_calls - 1):
        gap = (calls[c + 1][0] - calls[c][5]) * NS
        if (c + 1) % verify_forward_layers == 0:
            inter_across.append(gap)   # forward boundary -> draft/accept/commit
        else:
            inter_same.append(gap)

    def stats(xs):
        if not xs:
            return {"n": 0}
        xs2 = sorted(xs)
        return {
            "n": len(xs2),
            "mean": statistics.fmean(xs2),
            "p50": xs2[len(xs2) // 2],
            "p90": xs2[min(len(xs2) - 1, int(0.9 * len(xs2)))],
            "sum": sum(xs2),
        }

    # by bucket (all calls), by rows M, by layer.
    by_bucket = {name: stats([bc[2][k] for bc in per_call_buckets])
                 for k, name in enumerate(bucket_names)}
    rows_seen = sorted({bc[1] for bc in per_call_buckets})
    by_rows = {
        int(m): {name: stats([bc[2][k] for bc in per_call_buckets if bc[1] == m])
                 for k, name in enumerate(bucket_names)}
        for m in rows_seen
    }
    layers_seen = sorted({bc[0] for bc in per_call_buckets})
    by_layer = {
        int(L): {name: stats([bc[2][k] for bc in per_call_buckets if bc[0] == L])
                 for k, name in enumerate(bucket_names)}
        for L in layers_seen
    }

    total_in_run = sum(sum(bc[2]) for bc in per_call_buckets)
    total_inter_same = sum(inter_same)
    return {
        "n_calls": n_calls,
        "verify_forward_layers": verify_forward_layers,
        "buckets_seconds": by_bucket,
        "by_rows_seconds": by_rows,
        "by_layer_seconds": by_layer,
        "inter_layer_build_seconds": stats(inter_same),
        "across_forward_gap_seconds": stats(inter_across),
        "totals_seconds": {
            "in_run_all_buckets": total_in_run,
            "inter_layer_build": total_inter_same,
            "in_run_plus_inter_layer": total_in_run + total_inter_same,
            "note": ("in_run_plus_inter_layer should reconcile with the verify "
                     "phase seconds (phase_time_s.verify) within a few %; the "
                     "across-forward gap is draft/accept/commit, reported apart."),
        },
    }


def dump_probe(sink: ProbeSink, out_stem: str, *, verify_forward_layers: int = 40) -> dict:
    """Write the raw stamps (json.gz) + the summary (json) to ``out_stem`` and
    return the summary.  Called once after decode by the arm-A2 runner."""
    summary = summarize(sink, verify_forward_layers=verify_forward_layers)
    with gzip.open(f"{out_stem}.raw.json.gz", "wt") as fh:
        json.dump(
            {
                "t_layer": sink.t_layer, "t_idx": sink.t_idx, "t_ns": sink.t_ns,
                "m_layer": sink.m_layer, "m_rows": sink.m_rows,
                "m_parts": sink.m_parts, "m_hits": sink.m_hits,
            },
            fh,
        )
    with open(f"{out_stem}.summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def install_timed(runners: dict, switches: dict, *, scheduled: bool) -> dict:
    """Rebind each retained ``PackedDecode`` runner's ``switch._run`` to the timed
    lane, sharing one :class:`ProbeSink`.

    ``runners`` maps layer -> the ``PackedDecode`` instance (from
    ``plane_lane.install``); ``switches`` maps the SAME layer -> its switch module.

    MUST run AFTER plane_lane.install and (if used) projection_install.install_model
    -- projection_install validates ``type(runner) is PackedDecode`` and
    ``switch._run.__func__ is PackedDecode.run`` BEFORE its own rebind, so the timed
    rebind (which changes both) has to come last.  ``scheduled`` must match whether
    projection scheduling was installed, so the timed source is derived from the
    same run body that was live (``self.issue_next()`` kept when scheduled=True).
    The existing runner instance is reused, so ``runner.issue_next`` (wired by
    projection_install) stays intact.  Returns the sink + provenance."""
    Timed = make_timed_decode_class(scheduled=scheduled)
    sink = ProbeSink()
    from types import MethodType

    if set(runners) != set(switches):
        raise RuntimeError("runner/switch layer sets differ")
    bound = 0
    for layer, runner in runners.items():
        runner._sink = sink
        runner._stamp = MethodType(Timed._stamp, runner)
        runner._note = MethodType(Timed._note, runner)
        switches[layer]._run = MethodType(Timed.run, runner)
        bound += 1
    return {
        "timed_lane_installed_layers": bound,
        "scheduled": bool(scheduled),
        "base_run_sha256": Timed.base_run_sha256,
        "timed_run_sha256": Timed.timed_run_sha256,
        "retained_plane_lane_sha256": RETAINED_PLANE_LANE_SHA256,
        "sink": sink,
    }
