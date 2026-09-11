"""W53: per-request expert-streaming counter deltas for the served decode path.

The seed spread on the DeepSeek-V4.1 streaming artifact (same prompt, three
seeds -> 2.3x per-token decode time) is content routing to different experts.
To attribute a served window's per-token cost to misses directly, this module
snapshots the streaming counters at the start and end of the DECODE loop and
emits the deltas (hits, misses, miss rate, records streamed, bytes, plus
per-completion-token averages) onto the generation event and stage-timing
receipt.

Three sources, all best-effort and defensive so a stub runtime (or a build
without one of them) simply omits that sub-block:

  expert_cache        rt.expert_streaming_snapshot()["cache"]  (CacheCounters:
                      route_calls / expert_hits / expert_misses / bytes_read /
                      persistent+transient loads = records streamed) and
                      ["incremental_misses"] (routes / parts).
  engram_row_cache    sum of rt.model._engram_banks[*].cache.stats  (NGramRowCache:
                      hits / misses / reads / rows_read / gathers / evictions).
  route_probe         mtplx.expert_route_probe module counters, only when
                      MTPLX_ROUTE_STAGE_PROBE=1 (per-(phase,stage) call counts
                      and ns sums, e.g. split-route vs all-hit layer calls).

Snapshots are plain counter reads (no mx.eval, no GPU sync, no I/O), so the
probe adds only a few dict copies per request.
"""
from __future__ import annotations

from typing import Any

_EXPERT_CACHE_KEYS = (
    "route_calls",
    "expert_requests",
    "unique_expert_requests",
    "shared_expert_assignments",
    "expert_hits",
    "expert_misses",
    "persistent_loads",
    "transient_loads",
    "evictions",
    "bytes_read",
    "prefetch_issued",
    "prefetch_committed",
)


def _engram_row_cache_totals(rt: Any) -> dict[str, int] | None:
    """Sum NGramRowCache.stats across the model's per-layer engram banks."""
    model = getattr(rt, "model", None)
    banks = getattr(model, "_engram_banks", None)
    if not banks:
        return None
    agg: dict[str, int] = {}
    found = False
    for bank in banks:
        cache = getattr(bank, "cache", None)
        stats = getattr(cache, "stats", None)
        if isinstance(stats, dict):
            found = True
            for key, value in stats.items():
                try:
                    agg[str(key)] = agg.get(str(key), 0) + int(value)
                except (TypeError, ValueError):
                    continue
    return agg if found else None


def snapshot_stream_counters(rt: Any) -> dict[str, Any]:
    """Best-effort flat snapshot of the per-request-attributable counters.

    Every source is guarded: an unavailable one is absent from the result
    (a stub runtime returns ``{}``). Never raises.
    """
    out: dict[str, Any] = {}

    # 1. Expert streaming cache + incremental miss structure.
    snap = None
    try:
        snap = rt.expert_streaming_snapshot()
    except Exception:
        snap = None
    if isinstance(snap, dict):
        cache = snap.get("cache")
        if isinstance(cache, dict):
            out["expert_cache"] = {
                key: int(cache.get(key, 0)) for key in _EXPERT_CACHE_KEYS
            }
        inc = snap.get("incremental_misses")
        if isinstance(inc, dict):
            out["incremental_misses"] = {
                "routes": int(inc.get("routes", 0)),
                "parts": int(inc.get("parts", 0)),
            }
        # W64 (R3-pin): pinned-working-set gauges + cumulative all-pinned-route
        # counters (empty-ish when the lever is off). Cumulative so the delta
        # below reports the window's all-pinned-hit rate.
        pin = snap.get("pin_working_set")
        if isinstance(pin, dict):
            out["pin_working_set"] = {
                "enabled": bool(pin.get("enabled", False)),
                "decode_routes": int(pin.get("decode_routes", 0) or 0),
                "all_pinned_routes": int(pin.get("all_pinned_routes", 0) or 0),
                "pinned_total": int(pin.get("pinned_total", 0) or 0),
                "static_layer_count": int(pin.get("static_layer_count", 0) or 0),
            }

    # 2. Engram row cache (per-layer NGramRowCache stats, summed).
    engram = _engram_row_cache_totals(rt)
    if engram is not None:
        out["engram_row_cache"] = engram

    # 3. Route-stage probe (module-global; only when armed).
    try:
        from mtplx import expert_route_probe as _rp

        if getattr(_rp, "ENABLED", False):
            out["route_probe_counts"] = {
                str(name): int(count) for name, count in dict(_rp._COUNTS).items()
            }
            out["route_probe_sums_ns"] = {
                str(name): int(total) for name, total in dict(_rp._SUMS).items()
            }
    except Exception:
        pass

    return out


def _delta_map(before: dict, after: dict) -> dict[str, int]:
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def stream_counters_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    tokens: int,
    phase: str = "decode",
) -> dict[str, Any]:
    """Per-source deltas (after - before) with rates and per-token averages.

    ``tokens`` is the completion-token count the deltas are averaged over.
    Returns ``{}`` when neither snapshot carried any counters (stub runtime).
    """
    if not before and not after:
        return {}
    tok = max(1, int(tokens))
    out: dict[str, Any] = {"phase": phase, "tokens": int(tokens)}

    b_ec, a_ec = before.get("expert_cache"), after.get("expert_cache")
    if isinstance(b_ec, dict) and isinstance(a_ec, dict):
        d = _delta_map(b_ec, a_ec)
        hits = d.get("expert_hits", 0)
        misses = d.get("expert_misses", 0)
        total = hits + misses
        records = d.get("persistent_loads", 0) + d.get("transient_loads", 0)
        d["records_streamed"] = records
        d["hit_rate"] = round(hits / total, 6) if total else None
        d["miss_rate"] = round(misses / total, 6) if total else None
        d["misses_per_token"] = round(misses / tok, 4)
        d["records_streamed_per_token"] = round(records / tok, 4)
        d["bytes_read_per_token"] = round(d.get("bytes_read", 0) / tok, 2)
        out["expert_cache"] = d

    b_im, a_im = before.get("incremental_misses"), after.get("incremental_misses")
    if isinstance(b_im, dict) and isinstance(a_im, dict):
        d = _delta_map(b_im, a_im)
        d["routes_per_token"] = round(d.get("routes", 0) / tok, 4)
        d["parts_per_token"] = round(d.get("parts", 0) / tok, 4)
        out["incremental_misses"] = d

    b_er, a_er = before.get("engram_row_cache"), after.get("engram_row_cache")
    if isinstance(b_er, dict) and isinstance(a_er, dict):
        d = _delta_map(b_er, a_er)
        r_hits = d.get("hits", 0)
        r_misses = d.get("misses", 0)
        r_total = r_hits + r_misses
        d["hit_rate"] = round(r_hits / r_total, 6) if r_total else None
        d["miss_rate"] = round(r_misses / r_total, 6) if r_total else None
        d["misses_per_token"] = round(r_misses / tok, 4)
        d["rows_read_per_token"] = round(d.get("rows_read", 0) / tok, 4)
        out["engram_row_cache"] = d

    b_pin, a_pin = before.get("pin_working_set"), after.get("pin_working_set")
    if isinstance(a_pin, dict):
        b_pin = b_pin if isinstance(b_pin, dict) else {}
        routes = int(a_pin.get("decode_routes", 0)) - int(b_pin.get("decode_routes", 0))
        all_pinned = int(a_pin.get("all_pinned_routes", 0)) - int(
            b_pin.get("all_pinned_routes", 0)
        )
        out["pin_working_set"] = {
            "enabled": bool(a_pin.get("enabled", False)),
            "decode_routes": routes,
            "all_pinned_routes": all_pinned,
            "all_pinned_hit_rate": round(all_pinned / routes, 6) if routes else None,
            "pinned_total": int(a_pin.get("pinned_total", 0)),
            "static_layer_count": int(a_pin.get("static_layer_count", 0)),
        }

    a_rc = after.get("route_probe_counts")
    if isinstance(a_rc, dict):
        b_rc = before.get("route_probe_counts") or {}
        out["route_probe_counts"] = _delta_map(b_rc, a_rc)
        a_rs = after.get("route_probe_sums_ns")
        if isinstance(a_rs, dict):
            b_rs = before.get("route_probe_sums_ns") or {}
            out["route_probe_sums_ns"] = _delta_map(b_rs, a_rs)

    return out
