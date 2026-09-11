"""W53: per-token served-decode stage timer.

A near-zero-overhead probe that attributes each decode step's wall time to the
loop stages (forward, sampling, detokenize/emit, guards/stop-check, eval sync,
cache/session stores, MTP draft/verify/accept). Armed by the environment so it
never touches the shipping decode path:

    MTPLX_SERVE_STAGE_TIMING=1          arm the probe
    MTPLX_SERVE_STAGE_TIMING_RECEIPT=…  optional dir/file for a JSON receipt

When disarmed every method is a single boolean test and returns immediately, so
the generation loop keeps its historical timing byte-for-byte. Flags are read
at construction (once per request) — never at import — so a server that stamps
optimization keys after importing this module still arms correctly.

The timer records only host ``time.perf_counter()`` deltas. It issues no
``mx.eval`` and no GPU synchronization of its own; a stage's wall time is
whatever the loop already forces to materialize inside that stage. It is CPU
testable with a stub model (see tests/test_serve_stage_timing.py).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def stage_timing_enabled() -> bool:
    """Read the arm flag at call time (env-at-use, not import-frozen)."""
    return _env_truthy("MTPLX_SERVE_STAGE_TIMING")


class StageTimer:
    """Accumulates per-stage wall time across decode steps.

    Usage inside a decode loop::

        timer = StageTimer(enabled=stage_timing_enabled())
        ...
        timer.begin()                 # start the clock for this step
        logits_row = ...              # sampling region
        timer.lap("sample")           # bank perf_counter()-last into "sample"
        emit_token(token)
        timer.lap("emit")
        ...
        rt.forward_ar(...)
        timer.lap("forward")
        _eval(...)
        timer.lap("eval")
        timer.tick_token()            # one committed token this step

    ``lap`` banks the time since the last ``begin``/``lap`` under ``name`` and
    resets the clock, so consecutive laps partition the step with no gaps. When
    disabled every call is one boolean test.
    """

    __slots__ = ("enabled", "_totals", "_counts", "_last", "_tokens", "_wall0")

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._last: float = 0.0
        self._tokens: int = 0
        self._wall0: float = time.perf_counter() if self.enabled else 0.0

    def begin(self) -> None:
        if not self.enabled:
            return
        self._last = time.perf_counter()

    def lap(self, name: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._totals[name] = self._totals.get(name, 0.0) + (now - self._last)
        self._counts[name] = self._counts.get(name, 0) + 1
        self._last = now

    def add(self, name: str, seconds: float) -> None:
        """Bank an externally measured duration (does not touch the clock)."""
        if not self.enabled:
            return
        self._totals[name] = self._totals.get(name, 0.0) + float(seconds)
        self._counts[name] = self._counts.get(name, 0) + 1

    def tick_token(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self._tokens += int(n)

    def summary(self) -> dict[str, Any]:
        """Per-stage table. Empty dict when disabled (nothing to export)."""
        if not self.enabled:
            return {}
        tokens = max(1, self._tokens)
        total_wall = time.perf_counter() - self._wall0
        stages: dict[str, Any] = {}
        measured = 0.0
        for name, total in self._totals.items():
            count = self._counts.get(name, 0)
            measured += total
            stages[name] = {
                "total_s": round(total, 6),
                "count": count,
                "mean_ms": round(total / count * 1e3, 4) if count else 0.0,
                "per_token_ms": round(total / tokens * 1e3, 4),
                "share": round(total / total_wall, 4) if total_wall > 0 else 0.0,
            }
        return {
            "enabled": True,
            "tokens": self._tokens,
            "wall_s": round(total_wall, 6),
            "measured_s": round(measured, 6),
            "unattributed_s": round(max(0.0, total_wall - measured), 6),
            "per_token_wall_ms": round(total_wall / tokens * 1e3, 4),
            "stages": stages,
        }


def write_stage_timing_receipt(
    summary: dict[str, Any],
    *,
    request_id: str | None,
    mode: str,
) -> str | None:
    """Write ``summary`` to MTPLX_SERVE_STAGE_TIMING_RECEIPT if set.

    The env value may be a directory (a timestamped, request-suffixed file is
    created inside it — append-only, never an overwrite) or a full file path
    (one JSON object written to it). Returns the path written, or None when the
    env is unset or the summary is empty. Failures are swallowed: a broken
    receipt sink must never fail a served response.
    """
    if not summary:
        return None
    target = str(os.environ.get("MTPLX_SERVE_STAGE_TIMING_RECEIPT", "")).strip()
    if not target:
        return None
    payload = {
        "event": "mtplx_serve_stage_timing",
        "request_id": request_id,
        "generation_mode": mode,
        **summary,
    }
    try:
        if target.endswith(".json") or (
            os.path.exists(target) and not os.path.isdir(target)
        ):
            path = target
        else:
            os.makedirs(target, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            rid = (request_id or "req").replace("/", "_")[:48]
            path = os.path.join(target, f"stage-timing-{mode}-{stamp}-{rid}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None
