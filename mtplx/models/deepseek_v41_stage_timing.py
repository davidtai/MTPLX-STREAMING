"""Opt-in per-token / per-chunk stage timing for DeepSeek-V4.1-Flash (W37 + W47).

Window 12 measured 246 ms/token decode but could not say *where* the time goes;
window 16 measured 16,384-token prefill at 490 s chunk-major / 370 s layer-major
against a ~20 s bank read and a ~100 s compute floor -- also unattributed.  The
existing route-stage probe (``mtplx.expert_route_probe``,
``MTPLX_ROUTE_STAGE_PROBE``) brackets the streamed switch, but MLX is lazy, so a
bracket that wraps only graph construction records ~0 ms -- it counts, it does not
time.  This probe fixes that for the whole forward by placing an explicit
``mx.eval`` fence at each stage boundary, so a bracket's wall time is the stage's
dispatch **plus** its GPU execution, not just the Python encode.

Two session kinds
-----------------
* **decode** (W37): one query row (``s == 1``) per forward; the harness wraps each
  decode iteration in :func:`frame` and the argmax in a ``sample`` stage.  Per-token
  means; the per-stage sum tiles ``frame_wall``.
* **prefill** (W47): the 16K prompt runs as chunks through either schedule
  (``_forward_span`` per chunk = *chunk-major*, or ``_forward_layer_major`` =
  *layer-major*).  Attention is split (q/kv proj, indexer/candidate select,
  score+softmax+value, cache appends); the streamed switch is split into
  admission / route-plan / miss-submit host stages plus the fenced gather total.
  Brackets are tagged by chunk index via :func:`chunk`, so the report can show
  whether cost grows with chunk index (attention over growing T) or is flat (MoE).

Contract
--------
* OFF is the default and free: no session armed -> :func:`stage` returns a shared
  no-op context manager, :func:`recording` is ``False``, nothing is touched, so
  the forward is byte-for-byte the shipped path.
* A session is armed by :func:`begin` (kind="decode" default, or "prefill") and
  torn down by :func:`end`.  :meth:`_Probe.enter_forward` gates recording: decode
  records only ``s == 1`` forwards; prefill records the whole prefill forward.
* Every fence **inflates** absolute time (an extra host round-trip serialises the
  stage), so a stage-timing pass's tok/s is NOT throughput -- the *ratios* between
  stages are the signal.  Run the clean tok/s pass with the probe OFF.

The probe is a module-level singleton (no threading through every call site).
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

import mlx.core as mx

__all__ = [
    "begin",
    "end",
    "active",
    "is_active",
    "recording",
    "is_prefill",
    "stage",
    "stage_prefill",
    "stage_nested",
    "stage_attn",
    "frame",
    "chunk",
    "set_schedule",
    "tally",
    "report",
]

_KIND_DECODE = "decode"
_KIND_PREFILL = "prefill"
#: Max query rows a DECODE-kind session records: 1 (AR decode) .. 8 (the DSpark
#: K+1 speculative-verify batch, matching the K29 fused decode-attention row cap).
#: A forward wider than this is a genuine prefill and is left unrecorded so its
#: multi-GB transients never pollute the per-token decode census (W57).
_DECODE_STAGE_MAX_ROWS = 8

#: The armed probe for the current process, or ``None`` when no session is open.
#: Read on every stage bracket, so keep it a bare module global (one attribute
#: load) -- installed by :func:`begin`, cleared by :func:`end`.
_ACTIVE: "Optional[_Probe]" = None


class _NullFence:
    """The fence handed to a no-op bracket: ``add`` is a no-op (no array touched)."""

    __slots__ = ()

    def add(self, *_arrays) -> None:
        return None

    fence = add


class _Fence:
    """Collects the arrays a stage produces so the bracket can force exactly this
    stage's work at its close (and nothing the next stage will build)."""

    __slots__ = ("_arrays",)

    def __init__(self) -> None:
        self._arrays: list = []

    def add(self, *arrays) -> None:
        for a in arrays:
            if isinstance(a, mx.array):
                self._arrays.append(a)

    fence = add


class _NoopCM:
    """A reusable, stateless no-op context manager (returned when the probe is off
    or not recording), so the off path allocates nothing per site."""

    __slots__ = ()

    def __enter__(self) -> _NullFence:
        return _NULL_FENCE

    def __exit__(self, *_exc) -> bool:
        return False


_NULL_FENCE = _NullFence()
_NOOP_CM = _NoopCM()


class _Probe:
    """Per-session accumulator: nanosecond sums + counts per stage, a unit counter
    (decode tokens or prefill chunks), a frame/chunk-wall total, plus -- for
    prefill -- per-(chunk, stage) sums and a nested switch breakdown."""

    __slots__ = (
        "_kind", "_sums", "_counts", "_tokens", "_frame_ns", "_recording_now",
        "_chunk_idx", "_chunk_sums", "_chunk_counts", "_chunk_wall",
        "_nested_sums", "_nested_counts", "_nested_tallies",
        "_attn_sums", "_attn_counts", "_schedule",
    )

    def __init__(self, kind: str = _KIND_DECODE) -> None:
        self._kind = kind
        self._sums: dict[str, int] = defaultdict(int)
        self._counts: dict[str, int] = defaultdict(int)
        self._tokens: int = 0
        self._frame_ns: int = 0
        #: Set per forward by :meth:`enter_forward`.
        self._recording_now: bool = False
        #: Current chunk index (prefill) or ``None`` (decode / untagged).
        self._chunk_idx: "Optional[int]" = None
        self._chunk_sums: dict[tuple, int] = defaultdict(int)
        self._chunk_counts: dict[tuple, int] = defaultdict(int)
        self._chunk_wall: dict[int, int] = defaultdict(int)
        #: Nested switch breakdown (miss-I/O submit / route-plan / admission);
        #: kept OUT of the flat partition sum -- it decomposes moe.routed_switch.
        self._nested_sums: dict[str, int] = defaultdict(int)
        self._nested_counts: dict[str, int] = defaultdict(int)
        #: Integer magnitudes recorded by :meth:`_tally` (e.g. rows routed dense vs
        #: gather, experts under the dense threshold), summed across the forward and
        #: exported as ``switch_tallies`` -- counts of *things*, not wall time.
        self._nested_tallies: dict[str, int] = defaultdict(int)
        #: W50 nested attention-score breakdown (qk_matmul / scale_mask_sink /
        #: softmax / pv_matmul / cast / out_proj, per CSA mode); kept OUT of the
        #: flat partition sum -- it decomposes ``attn.<mode>.score`` (like the
        #: switch breakdown decomposes moe.routed_switch), so it never double-counts.
        self._attn_sums: dict[str, int] = defaultdict(int)
        self._attn_counts: dict[str, int] = defaultdict(int)
        self._schedule: "Optional[str]" = None

    def _tally(self, name: str, value: int) -> None:
        self._nested_tallies[name] += int(value)

    def enter_forward(self, seq_len: int) -> None:
        """Arm recording for this forward.

        decode: a single query row (``s == 1``) OR a small-M speculative-verify
        batch (``2 <= s <= _DECODE_STAGE_MAX_ROWS`` == 8, the DSpark K+1 verify /
        K29 cap) -- both are per-token decode traffic with tiny transients, so the
        W37 probe records them (W57: the 4-row verify was invisible before this,
        so ``--decode-mode dspark --stage-timing`` produced an empty verify table).
        A genuine prefill forward (``s > 8``) is NOT recorded here -- it would fence
        multi-GB transients and pollute the per-token census.  prefill kind: record
        the whole forward (the chunk loop inside tags each chunk).  The flag
        persists after the forward returns so the harness' ``sample`` bracket --
        which runs *between* decode forwards -- still records."""
        if self._kind == _KIND_PREFILL:
            self._recording_now = seq_len >= 1
        else:
            self._recording_now = 1 <= seq_len <= _DECODE_STAGE_MAX_ROWS

    @contextmanager
    def _stage(self, name: str, nested: bool = False, attn: bool = False):
        fence = _Fence()
        t0 = time.perf_counter_ns()
        try:
            yield fence
        finally:
            if fence._arrays:
                mx.eval(fence._arrays)
            dt = time.perf_counter_ns() - t0
            if attn:
                self._attn_sums[name] += dt
                self._attn_counts[name] += 1
            elif nested:
                self._nested_sums[name] += dt
                self._nested_counts[name] += 1
            else:
                self._sums[name] += dt
                self._counts[name] += 1
                idx = self._chunk_idx
                if idx is not None:
                    self._chunk_sums[(idx, name)] += dt
                    self._chunk_counts[(idx, name)] += 1

    @contextmanager
    def _frame(self):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self._frame_ns += time.perf_counter_ns() - t0
            self._tokens += 1

    @contextmanager
    def _chunk(self, idx: int):
        prev = self._chunk_idx
        self._chunk_idx = int(idx)
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self._chunk_wall[int(idx)] += time.perf_counter_ns() - t0
            self._chunk_idx = prev

    def snapshot(self) -> dict:
        tokens = self._tokens or 1
        names = sorted(set(self._sums) | set(self._counts))
        total_ns = sum(self._sums.values())
        report: dict = {
            "enabled": True,
            "kind": self._kind,
            "tokens": self._tokens,
            "stage_sum_ms": total_ns / 1e6,
            "stage_sum_ms_per_token": total_ns / 1e6 / tokens,
            "frame_wall_ms": self._frame_ns / 1e6,
            "frame_wall_ms_per_token": self._frame_ns / 1e6 / tokens,
            "stages": {
                name: {
                    "total_ms": self._sums.get(name, 0) / 1e6,
                    "count": self._counts.get(name, 0),
                    "mean_ms": (
                        self._sums[name] / self._counts[name] / 1e6
                        if self._counts.get(name)
                        else None
                    ),
                    "mean_ms_per_token": self._sums.get(name, 0) / 1e6 / tokens,
                }
                for name in names
            },
        }
        if self._kind == _KIND_PREFILL:
            report.update(self._prefill_views())
        return report

    def _prefill_views(self) -> dict:
        chunk_ids = sorted(
            {c for (c, _n) in self._chunk_sums} | set(self._chunk_wall)
        )
        by_chunk: dict = {}
        for c in chunk_ids:
            stages = {
                n: {
                    "total_ms": self._chunk_sums[(cc, n)] / 1e6,
                    "count": self._chunk_counts[(cc, n)],
                }
                for (cc, n) in self._chunk_sums
                if cc == c
            }
            by_chunk[str(c)] = {
                "wall_ms": self._chunk_wall.get(c, 0) / 1e6,
                "stage_sum_ms": sum(s["total_ms"] for s in stages.values()),
                "stages": stages,
            }
        switch_breakdown = {
            name: {
                "total_ms": self._nested_sums[name] / 1e6,
                "count": self._nested_counts[name],
                "mean_ms": (
                    self._nested_sums[name] / self._nested_counts[name] / 1e6
                    if self._nested_counts.get(name)
                    else None
                ),
            }
            for name in sorted(self._nested_sums)
        }
        attn_breakdown = {
            name: {
                "total_ms": self._attn_sums[name] / 1e6,
                "count": self._attn_counts[name],
                "mean_ms": (
                    self._attn_sums[name] / self._attn_counts[name] / 1e6
                    if self._attn_counts.get(name)
                    else None
                ),
            }
            for name in sorted(self._attn_sums)
        }
        return {
            "schedule": self._schedule,
            "chunks": len(chunk_ids),
            "chunk_wall_sum_ms": sum(self._chunk_wall.values()) / 1e6,
            "by_chunk": by_chunk,
            "switch_breakdown": switch_breakdown,
            "switch_tallies": dict(sorted(self._nested_tallies.items())),
            "attn_breakdown": attn_breakdown,
        }


# ---------------------------------------------------------------------------
# module-level session control + the call-site surface
# ---------------------------------------------------------------------------
def begin(kind: str = _KIND_DECODE) -> "_Probe":
    """Open a stage-timing session.  ``kind="decode"`` (default) records single-row
    decode forwards; ``kind="prefill"`` records the prefill forward and its chunks.
    Replaces any open session, so a fresh census starts each pass."""
    global _ACTIVE
    _ACTIVE = _Probe(kind=kind)
    return _ACTIVE


def end() -> "Optional[_Probe]":
    """Close the session and return it (its :meth:`snapshot` is still readable)."""
    global _ACTIVE
    probe = _ACTIVE
    _ACTIVE = None
    return probe


def active() -> "Optional[_Probe]":
    return _ACTIVE


def arm_recording() -> None:
    """Force ``_recording_now`` on for the active session (no-op when off).

    The DSpark draft (``draft_block``) does not go through ``Model.__call__``, so
    it never calls :meth:`_Probe.enter_forward`; without this the draft's coarse
    ``dspark.draft`` bracket would only record from the cycle *after* the first
    verify (the stale flag). The caller arms recording at each cycle start so both
    the draft and the K+1 verify are recorded deterministically from cycle 0 (W57).
    The verify's own ``enter_forward`` re-arms it inside the target forward."""
    if _ACTIVE is not None:
        _ACTIVE._recording_now = True


def is_active() -> bool:
    return _ACTIVE is not None


def _prefill_recording(p) -> bool:
    """Is ``p`` an armed probe recording a *prefill* forward?

    ``_kind`` is an OPT-IN prefill marker: a probe declares ``_kind = "prefill"``
    to select the finer prefill sub-brackets; anything else -- including a minimal
    probe DOUBLE that predates ``_kind`` and never sets it (e.g. the W41 dispatch
    census in ``scripts/deepseek_v41/dispatch_census.py``, which installs itself as
    the active probe and reuses the ``stage()`` brackets with decode semantics) --
    defaults to decode.  So this reads ``_kind`` through ``getattr`` with a decode
    default, and every prefill-only accessor goes through here; a double only needs
    ``_recording_now`` (and ``_stage``) to work."""
    return (
        p is not None
        and p._recording_now
        and getattr(p, "_kind", _KIND_DECODE) == _KIND_PREFILL
    )


def recording() -> bool:
    """True while a forward of an armed session is being timed (either kind).  Read
    by ``deepseek_v41._hc_use_compile`` / ``_attn_use_compile`` to force the eager
    Hyper-Connection / attention path when timing (a compiled tape is opaque to
    per-stage fences), so the census reflects one code path regardless of the K4 /
    K22 compile flags."""
    p = _ACTIVE
    return p is not None and p._recording_now


def is_prefill() -> bool:
    """True while a *prefill* session is recording -- selects the finer prefill
    attention/switch sub-brackets over the single decode ``attn.<mode>`` stage.
    A probe without ``_kind`` (a lightweight double) reads as decode."""
    return _prefill_recording(_ACTIVE)


def stage(name: str):
    """A timing bracket for one stage (fires in either session kind while
    recording).  Returns a shared no-op context manager otherwise, so the ``with``
    body runs untouched and the yielded fence's ``add`` is a no-op.  When active,
    fences the arrays handed to ``fence.add(...)`` on exit and books the elapsed
    wall time under ``name`` (and under the current chunk, if one is set)."""
    p = _ACTIVE
    if p is None or not p._recording_now:
        return _NOOP_CM
    return p._stage(name)


def stage_prefill(name: str):
    """A prefill-only timing bracket (the finer attention / cache sub-stages).
    No-op unless a prefill session is recording -- so the same code, run under a
    decode session or off, is byte-identical and never double-counts the single
    decode ``attn.<mode>`` bracket."""
    p = _ACTIVE
    if not _prefill_recording(p):
        return _NOOP_CM
    return p._stage(name)


def stage_nested(name: str):
    """A prefill-only *nested* bracket for the streamed switch breakdown
    (admission / route-plan / miss-submit).  Recorded into ``switch_breakdown`` and
    kept OUT of the flat partition sum, since it decomposes moe.routed_switch."""
    p = _ACTIVE
    if not _prefill_recording(p):
        return _NOOP_CM
    return p._stage(name, nested=True)


def stage_attn(name: str):
    """A prefill-only nested bracket for the attention-score breakdown (W50):
    qk_matmul / scale_mask_sink / softmax / pv_matmul / cast / out_proj, per CSA
    mode.  Recorded into ``attn_breakdown`` and kept OUT of the flat partition sum,
    since it decomposes ``attn.<mode>.score`` -- so it never double-counts the flat
    stage and is a no-op unless a prefill session is recording."""
    p = _ACTIVE
    if not _prefill_recording(p):
        return _NOOP_CM
    return p._stage(name, attn=True)


def frame():
    """Times one whole decode iteration (forward + sample) and counts it as a
    token.  Does NOT fence -- the inner :func:`stage` brackets already tile the
    work.  A no-op when no session is armed."""
    p = _ACTIVE
    if p is None:
        return _NOOP_CM
    return p._frame()


def chunk(idx: int):
    """Tags every :func:`stage` bracket inside the block with prefill chunk ``idx``
    and times the block as that chunk's wall.  A no-op unless a prefill session is
    recording, so decode / one-shot paths are untouched."""
    p = _ACTIVE
    if not _prefill_recording(p):
        return _NOOP_CM
    return p._chunk(idx)


def set_schedule(name: str) -> None:
    """Record which prefill schedule ran (``chunk_major`` / ``layer_major`` /
    ``one_shot``).  A no-op unless a prefill session is recording."""
    p = _ACTIVE
    if _prefill_recording(p):
        p._schedule = name


def tally(name: str, value: int) -> None:
    """Accumulate an integer magnitude under ``name`` into ``switch_tallies`` (e.g.
    the W51 dense-path row/expert counters).  Prefill-only and guarded exactly like
    :func:`set_schedule`, so decode / off / a lightweight probe double are no-ops."""
    p = _ACTIVE
    if _prefill_recording(p):
        p._tally(name, value)


def report() -> "Optional[dict]":
    """The active session's snapshot, or ``None`` when no session is armed."""
    return _ACTIVE.snapshot() if _ACTIVE is not None else None
