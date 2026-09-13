"""W125 -- low-overhead host-side decode timeline probe for DSV4.1 (AR v2 runner).

Motivation (docs W125): AR decode on the 16K cell runs 5.5-5.9 tok/s with GPU idle
share 0.73-0.75. Window 50 proved the SSD miss *bytes* are not the whole story
(fanning each record read into concurrent sub-reads at read_ns/wall 2.45 gained
only +7%), so ~120+ ms of every token is spent waiting on something host-side that
no existing receipt shows. ``--stage-timing`` cannot answer this: it forces the
eager path (226 ms/tok) and kills the compile levers.

This module records, *on the real compiled v2 runner path*, per decode token and
per MoE layer, monotonic host timestamps at the events that bracket the decode
work, then aggregates them in-process into per-phase mean/p50/p95 (per layer and
per token) plus a "host gap" accounting. It is a strict superset, in resolution,
of :mod:`mtplx.expert_route_probe` (which sums one bucket per stage name across all
layers/tokens with no percentiles and no per-token structure).

Design contract (matching the W125 spec):

* Env-gated by ``MTPLX_DSV41_DECODE_TIMELINE=1``. When unset the public marks are a
  single module-global boolean test + return -- a no-op, asserted by the unit
  tests.
* No per-event Python objects on the hot path. Each mark reads
  ``time.perf_counter_ns()`` and writes ONE float into a preallocated flat list
  indexed by ``(token, layer, event)``. Duration accumulators (reconcile await,
  exposed miss wait) take a caller-held ``int`` start stamp -- no context managers,
  no closures, no tuples allocated per mark.
* Overhead budget < 0.5 ms/token; the mean per-mark cost is calibrated once at
  arm time and the resulting per-token overhead is stamped into the snapshot.

The generation forward is single-threaded on the main thread, so the module-global
"current token" needs no lock. The SSD reads run on IO-pool threads, but those are
never marked here: the *exposed* miss wait is measured on the generation thread
inside ``PendingSplitRoute.iter_ready_misses`` (expert_runtime), which is exactly
the host stall this probe exists to expose.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from typing import Any

_perf = time.perf_counter_ns

# --------------------------------------------------------------------------- #
# Event layout
# --------------------------------------------------------------------------- #
# Per-(token, layer) timestamp events. Index order is the host-time order of a
# single MoE decode layer (attention then routed switch).
L_START = 0            # DecoderLayer.__call__ entry (host)                  [first]
ATTN_START = 1         # Attention.__call__ entry                            [first]
ATTN_END = 2           # Attention.__call__ return                          [last]
MOE_START = 3          # just before the routed switch (self.mlp)            [first]
BARRIER_DONE = 4       # routing indices barrier done (observe_route, post   [last]
#                        mx.eval(indices) / device LUT sync)
RECONCILE_DONE = 5     # gate-oracle prefetch reconcile returned             [last]
MISS_ISSUE = 6         # demand miss set planned + reads submitted           [first]
MISS_READY = 7         # last miss part became ready (slots ready)           [last]
ALL_HIT = 8            # try_all_hit_route returned a ready (no miss I/O)    [first]
EXPERT_DISPATCHED = 9  # routed switch returned (gather built + submitted)   [last]
L_END = 10             # DecoderLayer.__call__ return (after moe_combine)    [last]
_NE = 11

_EVENT_NAMES = {
    L_START: "layer_start",
    ATTN_START: "attn_start",
    ATTN_END: "attn_end",
    MOE_START: "moe_start",
    BARRIER_DONE: "routing_barrier_done",
    RECONCILE_DONE: "reconcile_done",
    MISS_ISSUE: "miss_issue",
    MISS_READY: "miss_ready",
    ALL_HIT: "all_hit",
    EXPERT_DISPATCHED: "expert_dispatched",
    L_END: "layer_end",
}
# Events whose FIRST occurrence per (token, layer) is authoritative (they may fire
# more than once on a multi-wave layer); everything else keeps the LAST value.
_FIRST_WRITE = frozenset({L_START, ATTN_START, MOE_START, MISS_ISSUE, ALL_HIT})

# Per-(token, layer) accumulated durations (ns), summed across repeats.
ACC_RECONCILE = 0      # time inside _reconcile_prefetch_for_route
ACC_MISS_WAIT = 1      # exposed wait in iter_ready_misses (host blocked on SSD)
_NA = 2

# Per-token timestamp events.
T_START = 0            # decode forward entry (Model.__call__, s == 1)
T_HEAD_DONE = 1        # lm_head matmul dispatched
T_END = 2             # forward end (fallback total for the final token)
_NTE = 3

# --------------------------------------------------------------------------- #
# Module-global probe state (the module IS the singleton, like expert_route_probe)
# --------------------------------------------------------------------------- #
_ON = False                     # hot-path gate: True only when armed AND sized
_ENV_KEY = "MTPLX_DSV41_DECODE_TIMELINE"
_PATH = os.environ.get(
    "MTPLX_DSV41_DECODE_TIMELINE_PATH", "/tmp/mtplx-dsv41-decode-timeline.json"
)
_MAXTOK = 0
_NL = 0
# 0.0 is the "unset" sentinel below. ``time.perf_counter_ns()`` counts ns from an
# arbitrary boot-relative epoch and is always a large positive integer in a real
# run, so it never collides with 0.0 (tests that script the clock must use a
# nonzero base).
_TS: list[float] = []           # (MAXTOK * NL * NE) timestamps, 0.0 == unset
_ACC: list[float] = []          # (MAXTOK * NL * NA) accumulated ns
_TTS: list[float] = []          # (MAXTOK * NTE) per-token timestamps
_CUR = -1                       # current decode-token index (-1 == none open)
_REC = False                    # True only inside a single-row decode forward
_NTOK = 0                       # decode tokens seen (may exceed MAXTOK -> clamped)
_DROPPED = 0                    # decode tokens past MAXTOK (not recorded)
_MARKS = 0                      # total layer/acc mark writes (for overhead stamp)
_NOW_CALLS = 0                  # now() calls while recording (overhead stamp)
_MARK_COST_NS = 0.0             # calibrated cost of one full mark call
_NOW_COST_NS = 0.0             # calibrated cost of one now() call
_CONFIG_FLAGS: dict = {}        # runtime switch-path flags, stamped into the block
_SNAP_CACHE: dict | None = None       # memoised aggregate (O(MAXTOK*NL) is not free)
_SNAP_CACHE_KEY: tuple | None = None  # (_NTOK, _MARKS): changes on any new mark


def env_armed() -> bool:
    """Whether ``MTPLX_DSV41_DECODE_TIMELINE=1`` requests the probe. Read at USE
    (not frozen at import), so a harness that stamps the env after importing this
    module -- the [[env-flags-read-at-use-not-import]] trap -- is still honoured.
    ``configure`` still has to run to size the arrays before ``enabled`` is True."""
    return os.environ.get(_ENV_KEY, "").strip() == "1"


def enabled() -> bool:
    return _ON


def config_noted() -> bool:
    """Whether the runtime has already stamped its switch-path config (a once-guard
    so the hot path builds the flag dict at most once)."""
    return bool(_CONFIG_FLAGS)


def note_switch_config(**flags: Any) -> None:
    """Stamp which switch path the runtime is using (fenced_split_path,
    deferred_pin_release, split_route_release, device_route). Called once by the
    runtime; carried verbatim into the snapshot's ``switch_config`` so a reader can
    tell whether ``ready_to_dispatch``/``miss_issue_to_ready`` include a blocking
    GPU gather fence (see the phase semantics)."""
    if _ON:
        _CONFIG_FLAGS.update(flags)


def _calibrate() -> tuple[float, float]:
    """Calibrate the FULL public mark path (function call + ``_REC`` check + index
    math + ``perf_counter_ns`` + list store + counter increment) AND ``now()``
    separately, on scratch cells, so the stamped overhead reflects real per-mark
    cost -- not just the raw list store (~26 ns), which undercounts the ~100+ ns
    real path. Never gates the hot path."""
    global _REC, _CUR, _MARKS, _NOW_CALLS
    if not _TS:
        return 0.0, 0.0
    save = (_REC, _CUR, _MARKS, _NOW_CALLS)
    _REC = True
    _CUR = 0
    reps = 20000
    t0 = _perf()
    for _ in range(reps):
        attn_end(0)  # full public mark path (last-write branch)
    cost_mark = (_perf() - t0) / reps
    t0 = _perf()
    for _ in range(reps):
        now()
    cost_now = (_perf() - t0) / reps
    _TS[(0 * _NL + 0) * _NE + ATTN_END] = 0.0  # undo the scratch write
    _REC, _CUR, _MARKS, _NOW_CALLS = save     # restore state + counters
    return cost_mark, cost_now


def configure(n_layers: int, *, max_tokens: int | None = None) -> None:
    """Arm and size the probe. Idempotent: a second call with a compatible layer
    count is a no-op, so the model may call it every forward without cost. Does
    nothing unless the env flag is set."""
    global _ON, _MAXTOK, _NL, _TS, _ACC, _TTS, _MARK_COST_NS, _NOW_COST_NS
    if not env_armed():
        return
    n_layers = int(n_layers)
    # Re-size when the layer count grows OR a larger token budget is requested
    # (the bench passes max_tokens == decode steps so a 1024-token run is not
    # truncated at the 512 default).
    if _ON and n_layers <= _NL and (max_tokens is None or int(max_tokens) <= _MAXTOK):
        return
    if max_tokens is None:
        raw = os.environ.get("MTPLX_DSV41_DECODE_TIMELINE_MAXTOK", "512")
        try:
            max_tokens = max(1, int(raw))
        except ValueError:
            max_tokens = 512
    _MAXTOK = max(int(max_tokens), _MAXTOK)
    _NL = max(n_layers, _NL, 1)
    _TS = [0.0] * (_MAXTOK * _NL * _NE)
    _ACC = [0.0] * (_MAXTOK * _NL * _NA)
    _TTS = [0.0] * (_MAXTOK * _NTE)
    _ON = True
    _MARK_COST_NS, _NOW_COST_NS = _calibrate()


def reset() -> None:
    """Zero the buffers and token cursor without re-sizing (a fresh measurement
    over the same arming). Cheap; safe to call between bench arms."""
    global _CUR, _REC, _NTOK, _DROPPED, _MARKS, _NOW_CALLS
    global _SNAP_CACHE, _SNAP_CACHE_KEY
    if not _ON:
        return
    for i in range(len(_TS)):
        _TS[i] = 0.0
    for i in range(len(_ACC)):
        _ACC[i] = 0.0
    for i in range(len(_TTS)):
        _TTS[i] = 0.0
    _CUR = -1
    _REC = False
    _NTOK = 0
    _DROPPED = 0
    _MARKS = 0
    _NOW_CALLS = 0
    _CONFIG_FLAGS.clear()
    _SNAP_CACHE = None
    _SNAP_CACHE_KEY = None


# --------------------------------------------------------------------------- #
# Token boundary (driven by Model.__call__ so both bench and served paths work)
# --------------------------------------------------------------------------- #
def token_begin(n_layers: int | None = None) -> None:
    """Open a new decode token. Call once per single-row decode forward, at the top
    of ``Model.__call__``. Sizing happens lazily on the first call so callers need
    not know the layer count up front."""
    global _CUR, _REC, _NTOK, _DROPPED
    if not env_armed():
        return
    if not _ON:
        if n_layers is None:
            return
        configure(n_layers)
        if not _ON:
            return
    _NTOK += 1
    nxt = _CUR + 1
    if nxt >= _MAXTOK:
        # Past capacity: keep timing enabled for the head mark but stop recording
        # per-layer cells (the cursor stays clamped at the last valid token).
        _DROPPED += 1
        _REC = False
        return
    _CUR = nxt
    _REC = True
    _TTS[_CUR * _NTE + T_START] = _perf()


def token_head_done() -> None:
    """Stamp the lm_head dispatch for the current token (after the head matmul)."""
    if not _REC:
        return
    _TTS[_CUR * _NTE + T_HEAD_DONE] = _perf()


def forward_end() -> None:
    """Close the current decode forward. Stamps a fallback per-token end and stops
    recording, so a subsequent multi-row (prefill / DSpark verify) forward -- which
    never calls ``token_begin`` -- is not attributed to this token."""
    global _REC
    if not _REC:
        return
    _TTS[_CUR * _NTE + T_END] = _perf()
    _REC = False


def finalize() -> None:
    """Alias for :func:`forward_end` used by a harness at end of decode."""
    forward_end()


# --------------------------------------------------------------------------- #
# Hot-path marks. Model side passes ``layer`` explicitly (self.layer_id); runtime
# side passes the route ``layer_index``. Both index the same 0..n_layers space.
# --------------------------------------------------------------------------- #
def _layer_ts(event: int, layer: int, first: bool) -> None:
    # Precondition: _REC is True (checked by the thin public wrappers). Kept as a
    # single flat store with no allocation.
    global _MARKS
    if layer < 0 or layer >= _NL:
        return
    idx = (_CUR * _NL + layer) * _NE + event
    if first and _TS[idx] != 0.0:
        _MARKS += 1
        return
    _TS[idx] = _perf()
    _MARKS += 1


def layer_start(layer: int) -> None:
    if _REC:
        _layer_ts(L_START, layer, True)


def attn_start(layer: int) -> None:
    if _REC:
        _layer_ts(ATTN_START, layer, True)


def attn_end(layer: int) -> None:
    if _REC:
        _layer_ts(ATTN_END, layer, False)


def moe_start(layer: int) -> None:
    if _REC:
        _layer_ts(MOE_START, layer, True)


def barrier_done(layer: int) -> None:
    if _REC:
        _layer_ts(BARRIER_DONE, layer, False)


def reconcile_done(layer: int) -> None:
    if _REC:
        _layer_ts(RECONCILE_DONE, layer, False)


def miss_issue(layer: int) -> None:
    if _REC:
        _layer_ts(MISS_ISSUE, layer, True)


def miss_ready(layer: int) -> None:
    if _REC:
        _layer_ts(MISS_READY, layer, False)


def all_hit(layer: int) -> None:
    if _REC:
        _layer_ts(ALL_HIT, layer, True)


def expert_dispatched(layer: int) -> None:
    if _REC:
        _layer_ts(EXPERT_DISPATCHED, layer, False)


def layer_end(layer: int) -> None:
    if _REC:
        _layer_ts(L_END, layer, False)


# --------------------------------------------------------------------------- #
# Duration accumulators. The caller holds an int start stamp from ``now()`` -- no
# object is allocated. A repeat on the same (token, layer) accumulates.
# --------------------------------------------------------------------------- #
def now() -> int:
    """Monotonic ns, or 0 when not recording (so a guarded caller can cheaply skip
    the paired ``add_*`` too)."""
    global _NOW_CALLS
    if not _REC:
        return 0
    _NOW_CALLS += 1
    return _perf()


def _add_acc(slot: int, layer: int, start_ns: int) -> None:
    global _MARKS
    if not start_ns or layer < 0 or layer >= _NL:
        return
    _ACC[(_CUR * _NL + layer) * _NA + slot] += _perf() - start_ns
    _MARKS += 1


def add_reconcile(layer: int, start_ns: int) -> None:
    if _REC:
        _add_acc(ACC_RECONCILE, layer, start_ns)


def add_miss_wait(layer: int, start_ns: int) -> None:
    if _REC:
        _add_acc(ACC_MISS_WAIT, layer, start_ns)


# --------------------------------------------------------------------------- #
# Aggregation (OFF the hot path)
# --------------------------------------------------------------------------- #
def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q / 100.0 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _stats(values: list[float]) -> dict[str, Any]:
    """mean/p50/p95 of a list of NANOSECOND durations, reported in milliseconds."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "n": n,
        "mean_ms": sum(values) / n / 1e6,
        "p50_ms": _pct(values, 50) / 1e6,
        "p95_ms": _pct(values, 95) / 1e6,
    }


def _count_stats(values: list[float]) -> dict[str, Any]:
    """mean/p50/p95 of a list of plain COUNTS (no ns->ms scaling), e.g. how many
    layers per token took the miss vs all-hit path."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": n,
        "mean": sum(values) / n,
        "p50": _pct(values, 50),
        "p95": _pct(values, 95),
    }


#: Derived phase -> (end_event, start_event). Positive-delta cells only are kept.
#: NOTE on lazy eval (HIGH-1): MLX is lazy and the only per-layer blocking sync is
#: the switch's ``mx.eval(indices)`` (expert_mlx.py), which the marks bracket as
#: ``gate_to_barrier`` (MOE_START->BARRIER_DONE). That eval materializes the PREVIOUS
#: layer's routed gather + THIS layer's attention + gate, so ``gate_to_barrier`` is
#: dominated by GPU EXECUTION + the sync, and ``attn``/``combine`` are lazy
#: graph-BUILD microseconds, not the attention/combine compute. The true host gap is
#: ``post_barrier_host`` (BARRIER_DONE->EXPERT_DISPATCHED): the .tolist / route plan
#: / gather graph-build the host runs after the sync. See ``phase_semantics`` below.
_PHASES = {
    "attn": (ATTN_END, ATTN_START),
    "gate_to_barrier": (BARRIER_DONE, MOE_START),
    "barrier_to_issue": (MISS_ISSUE, BARRIER_DONE),
    "miss_issue_to_ready": (MISS_READY, MISS_ISSUE),
    "ready_to_dispatch": (EXPERT_DISPATCHED, MISS_READY),
    "post_barrier_host": (EXPERT_DISPATCHED, BARRIER_DONE),
    "combine": (L_END, EXPERT_DISPATCHED),
    "moe_total": (L_END, MOE_START),
    "layer_total": (L_END, L_START),
}

#: How to read each phase (GPU-inclusive vs graph-build vs pure host), stamped into
#: the snapshot so a reader never mistakes a graph-build microsecond for real work.
_PHASE_SEMANTICS = {
    "attn": "graph-build us (attention runs lazily; its compute is forced at the "
            "next gate_to_barrier eval, not here)",
    "gate_to_barrier": "barrier wait incl. GPU (prev layer's routed gather + this "
                       "layer's attention + gate) + the mx.eval(indices) sync",
    "barrier_to_issue": "host: .tolist route + plan/pin hits before demand reads "
                        "are submitted (miss layers only)",
    "miss_issue_to_ready": "issue->slots-ready by timestamp; on the FENCED split "
                           "path this can include a blocking gather fence on "
                           "multi-part routes -- use miss_wait_total (ACC) for the "
                           "pure host SSD wait (see switch_config.fenced_split_path)",
    "ready_to_dispatch": "ready->switch-returned; on the FENCED split path this "
                         "includes the blocking mx.eval of the routed gather",
    "post_barrier_host": "pure host after the barrier sync (route plan + gather "
                         "graph-build); this is the host_gap component",
    "combine": "graph-build us (HC combine is lazy)",
    "moe_total": "moe_start->layer_end (spans the barrier, so GPU-inclusive)",
    "layer_total": "layer_start->layer_end (GPU-inclusive)",
}


def _ntok_recorded() -> int:
    return 0 if _CUR < 0 else min(_CUR + 1, _MAXTOK)


def snapshot() -> dict[str, Any]:
    """Aggregate the recorded timeline. Safe to call when disabled (returns a
    small stub). The O(MAXTOK*NL*phases) percentile work runs off the hot path and
    is MEMOISED on ``(_NTOK, _MARKS)`` -- the served snapshot() is called twice per
    request, and without the cache each call re-aggregated (~tens of ms). Any new
    mark changes _MARKS and invalidates the cache; ``reset`` clears it."""
    global _SNAP_CACHE, _SNAP_CACHE_KEY
    if not _ON:
        return {"enabled": False, "env_armed": env_armed()}

    key = (_NTOK, _MARKS)
    if _SNAP_CACHE_KEY == key and _SNAP_CACHE is not None:
        return _SNAP_CACHE

    ntok = _ntok_recorded()

    def cell(t: int, l: int, e: int) -> float:
        return _TS[(t * _NL + l) * _NE + e]

    def acc(t: int, l: int, slot: int) -> float:
        return _ACC[(t * _NL + l) * _NA + slot]

    # ---- per-phase, per-layer and per-token ---------------------------------
    per_layer: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in _PHASES}
    per_token_phase: dict[str, list[float]] = {name: [] for name in _PHASES}
    # host-gap components, per token
    tok_miss_wait: list[float] = []
    tok_reconcile: list[float] = []
    tok_barrier: list[float] = []
    tok_attn: list[float] = []
    tok_dispatch: list[float] = []
    tok_host_gap: list[float] = []      # attn_end -> expert_dispatched (GPU-idle host span)
    tok_interlayer_gap: list[float] = []  # layer_end -> next layer_start
    tok_moe: list[float] = []
    tok_layer_sum: list[float] = []
    tok_total: list[float] = []
    tok_head: list[float] = []
    tok_sample_sync: list[float] = []
    miss_layers_per_tok: list[float] = []
    hit_layers_per_tok: list[float] = []

    layer_phase_vals: dict[str, dict[int, list[float]]] = {
        name: {} for name in _PHASES
    }

    unrouted_per_tok: list[float] = []

    for t in range(ntok):
        sums = {name: 0.0 for name in _PHASES}
        mw = rc = bar = at = dsp = hg = ilg = moe = lyr = 0.0
        n_miss = n_hit = n_unrouted = 0
        prev_layer_end = 0.0
        for l in range(_NL):
            l_start = cell(t, l, L_START)
            if l_start == 0.0:
                continue  # layer not exercised this token
            for name, (e_end, e_start) in _PHASES.items():
                a = cell(t, l, e_start)
                b = cell(t, l, e_end)
                if a > 0.0 and b > a:
                    d = b - a
                    sums[name] += d
                    layer_phase_vals[name].setdefault(l, []).append(d)
            a_end = cell(t, l, ATTN_END)
            disp = cell(t, l, EXPERT_DISPATCHED)
            b_done = cell(t, l, BARRIER_DONE)
            # host_gap (HIGH-1): the PURE host span after the routing-barrier sync,
            # barrier_done -> expert_dispatched (route plan + gather graph-build). The
            # older attn_end->dispatch span wrongly swallowed all the GPU compute the
            # barrier eval forces, so it is NOT used for host_gap.
            if b_done > 0.0 and disp > b_done:
                hg += disp - b_done
            if prev_layer_end > 0.0 and l_start > prev_layer_end:
                ilg += l_start - prev_layer_end
            le = cell(t, l, L_END)
            if le > 0.0:
                prev_layer_end = le
            mw += acc(t, l, ACC_MISS_WAIT)
            rc += acc(t, l, ACC_RECONCILE)
            m_start = cell(t, l, MOE_START)
            if m_start > 0.0 and b_done > m_start:
                bar += b_done - m_start
            a_start = cell(t, l, ATTN_START)
            if a_start > 0.0 and a_end > a_start:
                at += a_end - a_start
            m_ready = cell(t, l, MISS_READY)
            if disp > 0.0 and m_ready > 0.0 and disp > m_ready:
                dsp += disp - m_ready
            if m_start > 0.0 and le > m_start:
                moe += le - m_start
            if l_start > 0.0 and le > l_start:
                lyr += le - l_start
            if cell(t, l, MISS_ISSUE) > 0.0:
                n_miss += 1
            if cell(t, l, ALL_HIT) > 0.0:
                n_hit += 1
            # MEDIUM: a device-route (or dense-island) layer bypasses observe_route,
            # so BARRIER_DONE never fires though the layer ran -- count it so a
            # zero barrier/miss population is explained, not silently dropped.
            if b_done == 0.0:
                n_unrouted += 1
        for name in _PHASES:
            if sums[name] > 0.0:
                per_token_phase[name].append(sums[name])
        tok_miss_wait.append(mw)
        tok_reconcile.append(rc)
        tok_barrier.append(bar)
        tok_attn.append(at)
        tok_dispatch.append(dsp)
        tok_host_gap.append(hg)
        unrouted_per_tok.append(float(n_unrouted))
        tok_interlayer_gap.append(ilg)
        tok_moe.append(moe)
        tok_layer_sum.append(lyr)
        miss_layers_per_tok.append(float(n_miss))
        hit_layers_per_tok.append(float(n_hit))
        head = cell_head(t, T_HEAD_DONE)
        tstart = cell_head(t, T_START)
        if head > 0.0 and tstart > 0.0 and head > tstart:
            tok_head.append(head - tstart)
        # per-token total from consecutive starts (true inter-token wall)
        nxt_start = cell_head(t + 1, T_START) if t + 1 < ntok else 0.0
        if nxt_start > 0.0 and tstart > 0.0 and nxt_start > tstart:
            tok_total.append(nxt_start - tstart)
            if head > 0.0 and nxt_start > head:
                tok_sample_sync.append(nxt_start - head)

    for name in _PHASES:
        for l, vals in layer_phase_vals[name].items():
            per_layer[name][str(l)] = _stats(vals)

    # ---- overhead stamp (full mark path + now() calls) -----------------------
    overhead_ns_total = _MARKS * _MARK_COST_NS + _NOW_CALLS * _NOW_COST_NS
    overhead_ms_per_token = (
        overhead_ns_total / max(1, ntok) / 1e6 if ntok else None
    )

    result = {
        "enabled": True,
        "env_armed": env_armed(),
        "schema": "w125.decode_timeline.v2",
        "tokens_recorded": ntok,
        "tokens_seen": _NTOK,
        "tokens_dropped_over_capacity": _DROPPED,
        "n_layers": _NL,
        "max_tokens": _MAXTOK,
        # Which switch path ran (stamped by the runtime once). fenced_split_path
        # True => ready_to_dispatch and the miss_issue_to_ready PHASE include a
        # blocking GPU gather fence; only miss_wait_total (ACC) is the pure host
        # SSD wait. Empty {} if no routed layer reported (e.g. a device-route arm).
        "switch_config": dict(_CONFIG_FLAGS),
        # Per-token phase sums (aggregated across layers), then over tokens.
        "per_token": {
            **{name: _stats(per_token_phase[name]) for name in _PHASES},
            "attn_total": _stats(tok_attn),
            "routing_barrier_total": _stats(tok_barrier),
            "reconcile_total": _stats(tok_reconcile),
            "miss_wait_total": _stats(tok_miss_wait),
            "ready_to_dispatch_total": _stats(tok_dispatch),
            "moe_total_sum": _stats(tok_moe),
            "layer_total_sum": _stats(tok_layer_sum),
            "host_gap": _stats(tok_host_gap),
            "interlayer_gap": _stats(tok_interlayer_gap),
            "head_dispatch": _stats(tok_head),
            "sample_sync_tail": _stats(tok_sample_sync),
            "token_total": _stats(tok_total),
            # counts, not durations -- keys mean/p50/p95 (no _ms scaling)
            "miss_layers": _count_stats(miss_layers_per_tok),
            "hit_layers": _count_stats(hit_layers_per_tok),
            "unrouted_layers": _count_stats(unrouted_per_tok),
        },
        # Per-layer phase percentiles (aggregated across tokens).
        "per_layer": per_layer,
        "phase_semantics": dict(_PHASE_SEMANTICS),
        "n_semantics": (
            "PHASE metrics (attn, gate_to_barrier, miss_issue_to_ready, "
            "ready_to_dispatch, ...) are timestamp deltas summed per token, so their "
            "n counts only tokens with >=1 layer where BOTH endpoints fired (e.g. "
            "miss_issue_to_ready.n = tokens with >=1 miss layer). The *_total keys "
            "(miss_wait_total, reconcile_total, ready_to_dispatch_total, ...) are "
            "summed over ALL recorded tokens (n = tokens_recorded), 0 where absent. "
            "Compare like with like."
        ),
        "host_gap_definition": (
            "per token: sum over layers of (expert_dispatched - barrier_done) == "
            "post_barrier_host: the PURE host span after the mx.eval(indices) sync "
            "(route .tolist + plan + gather graph-build) during which no GPU work is "
            "in flight. NOT attn_end->dispatch (that swallows the GPU compute the "
            "barrier eval forces). The barrier round-trip itself (GPU-inclusive) is "
            "routing_barrier_total; the pure exposed SSD read wait is miss_wait_total."
        ),
        "overhead": {
            "mark_cost_ns_calibrated": _MARK_COST_NS,
            "now_cost_ns_calibrated": _NOW_COST_NS,
            "marks_total": _MARKS,
            "now_calls_total": _NOW_CALLS,
            "overhead_ms_per_token": overhead_ms_per_token,
            "budget_ms_per_token": 0.5,
            "within_budget": (
                overhead_ms_per_token is not None
                and overhead_ms_per_token < 0.5
            ),
        },
    }
    _SNAP_CACHE = result
    _SNAP_CACHE_KEY = key
    return result


def cell_head(t: int, e: int) -> float:
    if t < 0 or t >= _MAXTOK:
        return 0.0
    return _TTS[t * _NTE + e]


def _dump() -> None:
    if _ON and _NTOK:
        try:
            with open(_PATH, "w") as handle:
                json.dump(snapshot(), handle, indent=2, sort_keys=True)
        except Exception:
            pass


atexit.register(_dump)
