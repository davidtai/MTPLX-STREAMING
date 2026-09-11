"""W53 per-request expert-streaming counter deltas.

CPU only, no runtime/model load: fake runtimes exercise the snapshot sources
(expert cache, engram row cache, route-stage probe) and the delta math
(hit/miss rates, records streamed, per-token averages).
"""

from types import SimpleNamespace

from mtplx import expert_route_probe
from mtplx.serve_stream_counters import (
    snapshot_stream_counters,
    stream_counters_delta,
)


def _fake_rt(*, cache=None, incremental=None, engram_banks=None, has_snapshot=True):
    def _snap():
        out = {}
        if cache is not None:
            out["cache"] = cache
        if incremental is not None:
            out["incremental_misses"] = incremental
        return out

    model = SimpleNamespace()
    if engram_banks is not None:
        model._engram_banks = engram_banks
    rt = SimpleNamespace(model=model)
    if has_snapshot:
        rt.expert_streaming_snapshot = _snap
    return rt


def _bank(stats):
    return SimpleNamespace(cache=SimpleNamespace(stats=stats))


# ---- snapshot sources -----------------------------------------------------


def test_snapshot_empty_when_no_streaming_runtime():
    # A runtime with no expert_streaming_snapshot and no engram banks.
    rt = SimpleNamespace(model=SimpleNamespace())
    assert snapshot_stream_counters(rt) == {}


def test_snapshot_reads_expert_cache_and_incremental():
    rt = _fake_rt(
        cache={"expert_hits": 100, "expert_misses": 20, "bytes_read": 5000,
               "persistent_loads": 3, "transient_loads": 17, "route_calls": 10},
        incremental={"routes": 4, "parts": 9},
    )
    snap = snapshot_stream_counters(rt)
    assert snap["expert_cache"]["expert_hits"] == 100
    assert snap["expert_cache"]["expert_misses"] == 20
    assert snap["incremental_misses"] == {"routes": 4, "parts": 9}


def test_snapshot_sums_engram_banks():
    rt = _fake_rt(
        engram_banks=[
            _bank({"hits": 10, "misses": 2, "rows_read": 240, "gathers": 10}),
            _bank({"hits": 5, "misses": 3, "rows_read": 96, "gathers": 8}),
        ],
    )
    snap = snapshot_stream_counters(rt)
    assert snap["engram_row_cache"]["hits"] == 15
    assert snap["engram_row_cache"]["misses"] == 5
    assert snap["engram_row_cache"]["rows_read"] == 336


def test_snapshot_route_probe_only_when_armed(monkeypatch):
    monkeypatch.setattr(expert_route_probe, "ENABLED", False, raising=False)
    rt = _fake_rt(cache={"expert_hits": 1, "expert_misses": 0})
    assert "route_probe_counts" not in snapshot_stream_counters(rt)

    monkeypatch.setattr(expert_route_probe, "ENABLED", True, raising=False)
    monkeypatch.setattr(
        expert_route_probe, "_COUNTS", {"split_route": 7, "all_hit": 55}, raising=False
    )
    monkeypatch.setattr(
        expert_route_probe, "_SUMS", {"split_route": 1200}, raising=False
    )
    snap = snapshot_stream_counters(rt)
    assert snap["route_probe_counts"] == {"split_route": 7, "all_hit": 55}
    assert snap["route_probe_sums_ns"] == {"split_route": 1200}


# ---- delta math -----------------------------------------------------------


def test_delta_empty_when_both_empty():
    assert stream_counters_delta({}, {}, tokens=10) == {}


def test_expert_cache_delta_rates_and_per_token():
    before = {"expert_cache": {"expert_hits": 100, "expert_misses": 20,
                               "bytes_read": 1000, "persistent_loads": 2,
                               "transient_loads": 8}}
    after = {"expert_cache": {"expert_hits": 180, "expert_misses": 40,
                              "bytes_read": 3000, "persistent_loads": 3,
                              "transient_loads": 27}}
    d = stream_counters_delta(before, after, tokens=40)
    ec = d["expert_cache"]
    assert ec["expert_hits"] == 80
    assert ec["expert_misses"] == 20
    assert ec["records_streamed"] == (3 - 2) + (27 - 8)  # 20
    assert ec["hit_rate"] == round(80 / 100, 6)
    assert ec["miss_rate"] == round(20 / 100, 6)
    assert ec["misses_per_token"] == round(20 / 40, 4)
    assert ec["bytes_read_per_token"] == round(2000 / 40, 2)
    assert d["phase"] == "decode"
    assert d["tokens"] == 40


def test_engram_delta_and_route_probe_delta():
    before = {
        "engram_row_cache": {"hits": 10, "misses": 2, "rows_read": 240},
        "route_probe_counts": {"split_route": 3, "all_hit": 20},
    }
    after = {
        "engram_row_cache": {"hits": 40, "misses": 10, "rows_read": 960},
        "route_probe_counts": {"split_route": 9, "all_hit": 80},
    }
    d = stream_counters_delta(before, after, tokens=30)
    er = d["engram_row_cache"]
    assert er["hits"] == 30 and er["misses"] == 8
    assert er["miss_rate"] == round(8 / 38, 6)
    assert er["rows_read_per_token"] == round(720 / 30, 4)
    assert d["route_probe_counts"] == {"split_route": 6, "all_hit": 60}


def test_delta_handles_zero_total_gracefully():
    before = {"expert_cache": {"expert_hits": 5, "expert_misses": 5}}
    after = {"expert_cache": {"expert_hits": 5, "expert_misses": 5}}
    d = stream_counters_delta(before, after, tokens=8)
    assert d["expert_cache"]["hit_rate"] is None  # no new hits/misses this window
    assert d["expert_cache"]["miss_rate"] is None
