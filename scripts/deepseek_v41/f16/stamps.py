"""F16/F18 diagnostic stamps: one global, append-only, host-only sink (default OFF).

``MTPLX_DSV41_F16_STAMPS=<stem>`` selects, at construction, a stamped variant of the
derived run (a SEPARATE compiled function; the inserted lines are pure host and
round-trip checked) plus stamped hand-off helpers.  Every record here is pure host
state -- ``time.perf_counter_ns()`` only, no ``mx`` array, no ``mx.eval`` -- so the
sink never touches Metal or the GPU lock and adds no op to the measured graph.

Per run call the sink records role (0 leader / 1 trailer / 2 non-group), layer, rows,
n_parts, n_hits, unique experts, and the nine stamps below (indices 0..8):

    0 run entry (before the barrier helper)
    1 after ``mx.async_eval(indices)`` returned (barrier mode; = encode time)
    2 resumed after the barrier hand-off (barrier mode)
    3 after ``mx.eval(indices)`` (GPU residual wait)
    4 after ``begin_split_route`` (reads submitted)
    5 before the reads hand-off (hits + projection + shared submitted)
    6 resumed after the reads hand-off
    7 after the completion loop (read residual + miss kernels submitted)
    8 before ``return``

At exit :func:`write` emits ``<stem>.raw.json.gz`` (every record) and
``<stem>.summary.json`` (:func:`summarize`).  ``stamp_readout.py <stem>`` prints it.
"""
from __future__ import annotations

import gzip
import json
import time
from typing import Optional

N_STAMPS = 9  # ids 0..8

# Named intervals derived from the stamps (see summarize).  gpu_wait uses 3-2 when the
# barrier stamps are present (barrier mode) and 3-0 otherwise (reads mode).
_SIMPLE_INTERVALS = (
    ("encode", 0, 1),
    ("parked_at_barrier", 1, 2),
    ("host_pre", 3, 4),
    ("hit_submit", 4, 5),
    ("parked_at_reads", 5, 6),
    ("miss_wait", 6, 7),
    ("post", 7, 8),
)


class _Sink:
    __slots__ = ("stem", "records")

    def __init__(self, stem: str) -> None:
        self.stem = stem
        self.records: list = []  # append-only, in schedule order

    def begin(self, role: int, layer: int, rows: int) -> dict:
        rec = {
            "role": int(role),
            "layer": int(layer),
            "rows": int(rows),
            "n_parts": None,
            "n_hits": None,
            "n_unique": None,
            "s": [None] * N_STAMPS,
        }
        rec["s"][0] = time.perf_counter_ns()
        self.records.append(rec)
        return rec

    def write(self) -> Optional[str]:
        raw_path = self.stem + ".raw.json.gz"
        with gzip.open(raw_path, "wt") as fh:
            json.dump({"n_stamps": N_STAMPS, "records": self.records}, fh)
        with open(self.stem + ".summary.json", "w") as fh:
            json.dump(summarize(self.records), fh, indent=2, sort_keys=True)
        return raw_path


_SINK: Optional[_Sink] = None


def configure(stem: str) -> _Sink:
    """Install the global sink (construction time, once)."""
    global _SINK
    _SINK = _Sink(stem)
    return _SINK


def sink() -> Optional[_Sink]:
    return _SINK


def begin(role: int, layer: int, rows: int) -> Optional[dict]:
    """Start one run's record (sets stamp 0).  No-op (returns None) if unconfigured."""
    if _SINK is None:
        return None
    return _SINK.begin(role, layer, rows)


def stamp(rec: Optional[dict], sid: int) -> None:
    if rec is not None:
        rec["s"][sid] = time.perf_counter_ns()


def annotate(rec: Optional[dict], *, n_parts=None, n_hits=None, n_unique=None) -> None:
    if rec is None:
        return
    if n_parts is not None:
        rec["n_parts"] = int(n_parts)
    if n_hits is not None:
        rec["n_hits"] = int(n_hits)
    if n_unique is not None:
        rec["n_unique"] = int(n_unique)


def write() -> Optional[str]:
    """Flush the sink to disk (registered at exit by the installer)."""
    if _SINK is None:
        return None
    return _SINK.write()


# ---------------------------------------------------------------------------
# Summary (pure host; robust to missing/None stamps).
# ---------------------------------------------------------------------------
def _p(values: list, q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def _stats(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "sum": None, "mean": None, "p50": None, "p90": None}
    return {
        "n": len(vals),
        "sum": sum(vals),
        "mean": sum(vals) / len(vals),
        "p50": _p(vals, 0.50),
        "p90": _p(vals, 0.90),
    }


def _diff(rec: dict, a: int, b: int) -> Optional[int]:
    sa, sb = rec["s"][a], rec["s"][b]
    if sa is None or sb is None:
        return None
    return sb - sa


def _gpu_wait(rec: dict) -> Optional[int]:
    # barrier mode records stamps 1/2; reads mode does not -> fall back to 3-0.
    if rec["s"][2] is not None:
        return _diff(rec, 2, 3)
    return _diff(rec, 0, 3)


def _assign_forwards(records: list) -> list:
    """Group records into forwards.  Each forward runs layers 0..N per role; a role's
    layer decreasing marks a new forward.  Returns a forward index per record."""
    last_layer: dict = {}
    fwd_for_role: dict = {}
    out = []
    max_fwd = -1
    for rec in records:
        role, layer = rec["role"], rec["layer"]
        prev = last_layer.get(role)
        if prev is None or layer < prev:
            fwd_for_role[role] = fwd_for_role.get(role, -1) + 1
        last_layer[role] = layer
        f = fwd_for_role[role]
        max_fwd = max(max_fwd, f)
        out.append(f)
    return out


def summarize(records: list) -> dict:
    roles = {0: "leader", 1: "trailer", 2: "nongroup"}
    interval_names = [n for n, _, _ in _SIMPLE_INTERVALS] + ["gpu_wait", "build"]

    fwd_idx = _assign_forwards(records)

    # build: next same-role record's stamp 0 minus this record's stamp 8, same forward.
    build_by_role: dict = {r: [] for r in roles}
    prev_by_role: dict = {}  # role -> (record, forward)
    for rec, f in zip(records, fwd_idx):
        role = rec["role"]
        prev = prev_by_role.get(role)
        if prev is not None and prev[1] == f:
            prec = prev[0]
            if prec["s"][8] is not None and rec["s"][0] is not None:
                build_by_role[role].append(rec["s"][0] - prec["s"][8])
        prev_by_role[role] = (rec, f)

    per_role: dict = {}
    for role, name in roles.items():
        rr = [rec for rec in records if rec["role"] == role]
        if not rr:
            continue
        block = {"n_records": len(rr)}
        for iname, a, b in _SIMPLE_INTERVALS:
            block[iname] = _stats([_diff(rec, a, b) for rec in rr])
        block["gpu_wait"] = _stats([_gpu_wait(rec) for rec in rr])
        block["build"] = _stats(build_by_role[role])
        per_role[name] = block

    overall = {"n_records": len(records)}
    for iname, a, b in _SIMPLE_INTERVALS:
        overall[iname] = _stats([_diff(rec, a, b) for rec in records])
    overall["gpu_wait"] = _stats([_gpu_wait(rec) for rec in records])
    overall["build"] = _stats([v for vs in build_by_role.values() for v in vs])

    # per-forward wall (last stamp8 - first stamp0) and across-forward gap.
    fwd_span: dict = {}
    for rec, f in zip(records, fwd_idx):
        s0, s8 = rec["s"][0], rec["s"][8]
        cur = fwd_span.setdefault(f, [None, None])
        if s0 is not None and (cur[0] is None or s0 < cur[0]):
            cur[0] = s0
        if s8 is not None and (cur[1] is None or s8 > cur[1]):
            cur[1] = s8
    forwards = sorted(fwd_span)
    totals = [
        fwd_span[f][1] - fwd_span[f][0]
        for f in forwards
        if fwd_span[f][0] is not None and fwd_span[f][1] is not None
    ]
    gaps = []
    for a, b in zip(forwards, forwards[1:]):
        end_a, start_b = fwd_span[a][1], fwd_span[b][0]
        if end_a is not None and start_b is not None:
            gaps.append(start_b - end_a)

    return {
        "n_records": len(records),
        "n_forwards": len(forwards),
        "interval_names": interval_names,
        "per_role": per_role,
        "overall": overall,
        "total_per_forward_ns": _stats(totals),
        "across_forward_gap_ns": _stats(gaps),
        "units": "nanoseconds (time.perf_counter_ns)",
    }
