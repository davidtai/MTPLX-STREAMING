"""Opt-in per-token decode stage timing for DeepSeek-V4.1-Flash (W37).

Window 12 measured 246 ms/token at 4.06 tok/s but could not say *where* the time
goes: the existing route-stage probe (``mtplx.expert_route_probe``,
``MTPLX_ROUTE_STAGE_PROBE``) brackets the streamed switch, but MLX is lazy, so a
bracket that wraps only graph construction records ~0 ms -- it counts, it does not
time.  This probe fixes that for the whole decode forward by placing an explicit
``mx.eval`` fence at each stage boundary, so a bracket's wall time is the stage's
dispatch **plus** its GPU execution, not just the Python encode.

Contract
--------
* OFF is the default and free: when no session is armed, :func:`stage` returns a
  shared no-op context manager and :func:`recording` is ``False`` -- no fence, no
  allocation, no array touched, so the forward is byte-for-byte the shipped path.
* A session is armed by :func:`begin` (around the *decode* loop, after prefill)
  and torn down by :func:`end`.  While armed, :meth:`_Probe.enter_forward` gates
  recording on ``s == 1`` so prefill forwards (which would fence multi-GB
  transients) stay untouched; only decode (one query row) is timed.
* Every fence **inflates** absolute time (an extra host round-trip serialises the
  stage), so the reported tok/s of a stage-timing pass is NOT a throughput
  number.  What survives the inflation is the *ratio* between stages, which is the
  question W37 asks.  Run the clean tok/s pass with the probe OFF.

The probe is a module-level singleton (no threading through every call site).
:func:`begin` installs it; the model/engram/attention/MoE call sites read it via
:func:`stage`; :func:`report` / :meth:`Model.stage_timing_report` read it back.
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
    "stage",
    "frame",
    "report",
]

#: The armed probe for the current process, or ``None`` when no session is open.
#: Read on every stage bracket, so keep it a bare module global (one attribute
#: load) -- installed by :func:`begin`, cleared by :func:`end`.
_ACTIVE: "Optional[_Probe]" = None


class _NullFence:
    """The fence handed to a no-op bracket: ``add`` is a no-op (no array touched)."""

    __slots__ = ()

    def add(self, *_arrays) -> None:
        return None

    # ``fence`` reads the same as ``add`` at the call sites.
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
    or not recording this forward), so the off path allocates nothing per site."""

    __slots__ = ()

    def __enter__(self) -> _NullFence:
        return _NULL_FENCE

    def __exit__(self, *_exc) -> bool:
        return False


_NULL_FENCE = _NullFence()
_NOOP_CM = _NoopCM()


class _Probe:
    """Per-session accumulator: nanosecond sums + counts per stage, a decode-token
    counter, and a frame-wall total (the reference the per-stage sum tiles)."""

    __slots__ = ("_sums", "_counts", "_tokens", "_frame_ns", "_recording_now")

    def __init__(self) -> None:
        self._sums: dict[str, int] = defaultdict(int)
        self._counts: dict[str, int] = defaultdict(int)
        self._tokens: int = 0
        self._frame_ns: int = 0
        #: Set per forward by :meth:`enter_forward`; ``True`` only for a decode
        #: (single query row) forward while the session is armed.
        self._recording_now: bool = False

    def enter_forward(self, seq_len: int) -> None:
        """Arm recording for this forward iff it is a decode step (``s == 1``).

        Prefill forwards (``s > 1``) build multi-GB transients whose fences would
        both crater the process and pollute the per-token census, so they stay
        untouched; the flag persists after the forward returns so the harness'
        ``sample`` bracket -- which runs *between* forwards -- still records."""
        self._recording_now = seq_len == 1

    @contextmanager
    def _stage(self, name: str):
        fence = _Fence()
        t0 = time.perf_counter_ns()
        try:
            yield fence
        finally:
            if fence._arrays:
                mx.eval(fence._arrays)
            self._sums[name] += time.perf_counter_ns() - t0
            self._counts[name] += 1

    @contextmanager
    def _frame(self):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self._frame_ns += time.perf_counter_ns() - t0
            self._tokens += 1

    def snapshot(self) -> dict:
        tokens = self._tokens or 1
        names = sorted(set(self._sums) | set(self._counts))
        total_ns = sum(self._sums.values())
        return {
            "enabled": True,
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


# ---------------------------------------------------------------------------
# module-level session control + the call-site surface
# ---------------------------------------------------------------------------
def begin() -> "_Probe":
    """Open a stage-timing session (call around the decode loop, after prefill).
    Replaces any open session, so a fresh census starts each pass."""
    global _ACTIVE
    _ACTIVE = _Probe()
    return _ACTIVE


def end() -> "Optional[_Probe]":
    """Close the session and return it (its :meth:`snapshot` is still readable)."""
    global _ACTIVE
    probe = _ACTIVE
    _ACTIVE = None
    return probe


def active() -> "Optional[_Probe]":
    return _ACTIVE


def is_active() -> bool:
    return _ACTIVE is not None


def recording() -> bool:
    """True while a decode forward of an armed session is in flight.  Read by
    ``deepseek_v41._hc_use_compile`` to force the eager Hyper-Connection path when
    timing (the compiled tape is opaque to per-stage fences), so the stage census
    always reflects one code path regardless of ``MTPLX_DSV41_HC_COMPILE``."""
    p = _ACTIVE
    return p is not None and p._recording_now


def stage(name: str):
    """A timing bracket for one decode stage.

    Returns a shared no-op context manager unless a session is armed *and* the
    current forward is a decode step; otherwise the ``with`` body runs untouched
    and the yielded fence's ``add`` is a no-op.  When active, the bracket fences
    the arrays handed to ``fence.add(...)`` on exit and books the elapsed wall
    time under ``name``."""
    p = _ACTIVE
    if p is None or not p._recording_now:
        return _NOOP_CM
    return p._stage(name)


def frame():
    """Times one whole decode iteration (forward + sample) and counts it as a
    token.  Does NOT fence -- the inner :func:`stage` brackets already tile the
    work -- so ``frame_wall`` is the reference the per-stage sum should match, not
    a double count.  A no-op when no session is armed."""
    p = _ACTIVE
    if p is None:
        return _NOOP_CM
    return p._frame()


def report() -> "Optional[dict]":
    """The active session's snapshot, or ``None`` when no session is armed."""
    return _ACTIVE.snapshot() if _ACTIVE is not None else None
