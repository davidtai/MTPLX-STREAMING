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
    "pool_loads",
    "scan_inserts",
    "promotions",
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
        # W87 HIGH-1: the served daemon passes an MTPLXRuntime
        # (expert_streaming_snapshot()); the DSV4.1 in-process bench passes the BARE
        # ExpertStreamingRuntime (snapshot()) the loader attaches as
        # model._mtplx_expert_runtime.  Prefer the MTPLXRuntime accessor (served
        # path unchanged), else resolve the streaming runtime and snapshot it --
        # otherwise every bench receipt silently loses its expert_cache/cold_start.
        getter = getattr(rt, "expert_streaming_snapshot", None)
        if callable(getter):
            snap = getter()
        else:
            es = getattr(rt, "expert_streaming", None) or rt
            snap = es.snapshot()
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
        # W71: pinned device-route barrier-free-layer counters (cumulative;
        # the delta below reports the window's barrier-free layers per token).
        drp = snap.get("device_route_pinned")
        if isinstance(drp, dict):
            out["device_route_pinned"] = {
                "enabled": bool(drp.get("enabled", False)),
                "flushes": int(drp.get("flushes", 0) or 0),
                "barrier_free_layers": int(drp.get("barrier_free_layers", 0) or 0),
                "recovered_layers": int(drp.get("recovered_layers", 0) or 0),
            }

        # W87 cold-start decode telemetry: pass through so the DELTA can compute
        # the first-N-token vs steady-state decode hit rate (always on -- both the
        # two-tier and single-slot-pool paths populate the same fields).
        cs = snap.get("cold_start")
        if isinstance(cs, dict):
            out["cold_start"] = {str(k): v for k, v in cs.items()}

        # W95f: pass through the gate-oracle prefetch + v2 runner receipt blocks so
        # the served daemon's stream-counter path logs the SSD-hiding counters
        # (prefetch hit/wasted, demand vs speculative bytes, budget_skips, margin,
        # ring size, per-decode-token normalisations). Absent before -- this
        # snapshot filtered the cache to _EXPERT_CACHE_KEYS -- and present now only
        # when the ring / v2 runner is armed (the snapshot carries the block).
        for _block_key in ("gate_prefetch", "runner"):
            block = snap.get(_block_key)
            if isinstance(block, dict):
                out[_block_key] = block

        # W110: pass through the io-thread reader metrics (per-record sha256
        # engagement) so the decode-scoped delta can report records_hashed /
        # records_unhashed / hash_thread_ns_total per verify/decode window -- the
        # MTPLX_DSV41_VERIFY_RECORD_HASHES lever's engagement counters.
        io_metrics = snap.get("io")
        if isinstance(io_metrics, dict):
            out["io"] = {str(k): io_metrics[k] for k in io_metrics}

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

    # W110: per-record sha256 engagement over THIS decode window (the bench-only
    # MTPLX_DSV41_VERIFY_RECORD_HASHES diagnostic's counters). ``records_hashed`` +
    # ``records_unhashed`` = records streamed off SSD during decode; with hashing ON
    # unhashed is 0 and ``hash_thread_ms`` is the SUMMED io-thread hashing time (across
    # all io-pool threads, NOT wall -- divide by the io-pool width for an upper bound
    # on exposed wall); with hashing OFF hashed is 0 and no re-check ran (bytes are
    # byte-identical either way).
    b_io, a_io = before.get("io"), after.get("io")
    if isinstance(b_io, dict) and isinstance(a_io, dict):
        # _delta_map differences EVERY key, but ``read_mib_per_second`` is a
        # cumulative-since-open FLOAT RATE (from ExpertIOMetrics.as_dict), so
        # differencing it yields a garbage "delta". Drop the non-counter (derived
        # float) keys before the delta and derive the decode-WINDOW read rate from the
        # ``read_bytes`` / ``read_ns`` counter deltas instead.
        # W123: DERIVED FLOAT keys (not monotonic counters) must be dropped
        # before the delta -- differencing a cumulative-since-open rate/mean, or a
        # running PEAK, yields garbage (window 50 showed read_inflight_max=1 from
        # peak-differencing and read_inflight_depth_mean=-2 from float-
        # differencing). ``read_inflight_max`` is a peak: take the AFTER snapshot
        # directly (the deepest concurrency observed by window end). The mean /
        # realized-BW / realized-QD are recomputed below from the counter deltas.
        _NON_COUNTER_IO = (
            "read_mib_per_second",
            "read_inflight_depth_mean",
            "read_realized_gb_per_s",
            "read_realized_qd",
            "read_inflight_max",
        )
        b_c = {k: v for k, v in b_io.items() if k not in _NON_COUNTER_IO}
        a_c = {k: v for k, v in a_io.items() if k not in _NON_COUNTER_IO}
        d = _delta_map(b_c, a_c)
        hashed = d.get("records_hashed", 0)
        unhashed = d.get("records_unhashed", 0)
        total = hashed + unhashed
        d["records_hashed_per_token"] = round(hashed / tok, 4)
        d["records_unhashed_per_token"] = round(unhashed / tok, 4)
        d["hash_fraction"] = round(hashed / total, 6) if total else None
        d["hash_thread_ms"] = round(d.get("hash_thread_ns_total", 0) / 1e6, 3)
        d["hash_thread_ms_per_token"] = round(d.get("hash_thread_ns_total", 0) / 1e6 / tok, 4)
        # W123 read-pool depth over THIS window, from monotonic-counter deltas:
        #   read_wall_ns = UNION of in-flight intervals (drive-busy wall);
        #   read_ns      = SUM of per-sub-read durations (io-thread time).
        # realized GB/s uses the UNION wall (the honest aggregate SSD throughput,
        # unlike the old bytes/read_ns which sub-read fanout makes meaningless);
        # realized QD = thread-time / union = mean reads outstanding while busy
        # (~1 == serialized QD1). read_gb_per_s_window keeps its name but now uses
        # the union wall. read_inflight_max is the peak from the AFTER snapshot.
        _read_ns = d.get("read_ns", 0)
        _read_wall_ns = d.get("read_wall_ns", 0)
        _samples = d.get("read_inflight_samples", 0)
        d["read_inflight_max"] = int(a_io.get("read_inflight_max", 0))
        d["read_inflight_depth_mean"] = (
            round(d.get("read_inflight_depth_sum", 0) / _samples, 4)
            if _samples
            else None
        )
        d["read_realized_qd"] = (
            round(_read_ns / _read_wall_ns, 4) if _read_wall_ns else None
        )
        d["read_gb_per_s_window"] = (
            round(d.get("read_bytes", 0) / _read_wall_ns, 4) if _read_wall_ns else None
        )
        # Keep the old io-thread-time throughput under an explicit name for
        # continuity with pre-W123 receipts (sum-of-durations, not wall).
        d["read_thread_gb_per_s_window"] = (
            round(d.get("read_bytes", 0) / _read_ns, 4) if _read_ns else None
        )
        out["io"] = d

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

    b_drp, a_drp = before.get("device_route_pinned"), after.get("device_route_pinned")
    if isinstance(a_drp, dict):
        b_drp = b_drp if isinstance(b_drp, dict) else {}
        flushes = int(a_drp.get("flushes", 0)) - int(b_drp.get("flushes", 0))
        bfree = int(a_drp.get("barrier_free_layers", 0)) - int(
            b_drp.get("barrier_free_layers", 0)
        )
        recovered = int(a_drp.get("recovered_layers", 0)) - int(
            b_drp.get("recovered_layers", 0)
        )
        out["device_route_pinned"] = {
            "enabled": bool(a_drp.get("enabled", False)),
            "flushes": flushes,
            "barrier_free_layers": bfree,
            "recovered_layers": recovered,
            "barrier_free_layers_per_flush": round(bfree / flushes, 6) if flushes else None,
        }

    # W87 cold-start decode telemetry (always on -- both slot-pool paths populate
    # it, so cell16k_ring vs cell16k_ring_pool read the first-N-token hit rate from
    # the SAME receipt field).
    b_cs, a_cs = before.get("cold_start"), after.get("cold_start")
    if isinstance(a_cs, dict):
        b_cs = b_cs if isinstance(b_cs, dict) else {}
        d_fh = int(a_cs.get("first_64_steps_hits", 0)) - int(
            b_cs.get("first_64_steps_hits", 0)
        )
        d_fr = int(a_cs.get("first_64_steps_requests", 0)) - int(
            b_cs.get("first_64_steps_requests", 0)
        )
        d_sh = int(a_cs.get("steady_hits", 0)) - int(b_cs.get("steady_hits", 0))
        d_sr = int(a_cs.get("steady_requests", 0)) - int(
            b_cs.get("steady_requests", 0)
        )
        out["cold_start"] = {
            "single_slot_pool": bool(a_cs.get("single_slot_pool", False)),
            "measurement_basis": a_cs.get("measurement_basis"),
            "cold_start_decode_steps": int(a_cs.get("cold_start_decode_steps", 0)),
            "decode_steps_observed": int(a_cs.get("decode_steps_observed", 0))
            - int(b_cs.get("decode_steps_observed", 0)),
            "first_64_steps_hits": d_fh,
            "first_64_steps_requests": d_fr,
            "steady_hits": d_sh,
            "steady_requests": d_sr,
            "decode_hit_rate_first_64_steps": (
                round(d_fh / d_fr, 6) if d_fr else None
            ),
            "decode_hit_rate_steady_state": (
                round(d_sh / d_sr, 6) if d_sr else None
            ),
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
