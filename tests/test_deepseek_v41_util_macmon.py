"""W90 -- macmon utilization sampler for the DSV4.1 decode benches (CPU, mocked).

The GPU DVFS-downclocks in the gaps between B=1 decode dispatch bursts (macmon on a
live 16K decode: 71 C, ~45% busy, freq 580-1381 MHz) -- the mode-independent in-situ
decode floor.  ``scripts/deepseek_v41/util_macmon.py`` samples ``macmon pipe`` over
the timed decode into a ``utilization`` receipt block.  These tests exercise the
parse/summary/census as pure functions and the background sampler with the READER
MOCKED (injected JSON lines / a nonexistent binary) -- no Metal, no sudo, no macmon
process.  Run under ``nice -n 19``; no ``pytest -n auto``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "deepseek_v41" / "util_macmon.py"
)
_spec = importlib.util.spec_from_file_location("dsv41_util_macmon", _PATH)
um = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(um)


def _payload(freq, power, gpu_usage, gpu_active, cpu_usage, gpu_c, cpu_c):
    # gpu_usage_ratio and gpu_active_ratio are DIFFERENT macmon metrics -> distinct args
    return {
        "gpu_freq_mhz": freq,
        "gpu_power": power,
        "gpu_usage_ratio": gpu_usage,
        "gpu_active_ratio": gpu_active,
        "cpu_usage_ratio": cpu_usage,
        "temp": {"gpu_temp_avg": gpu_c, "cpu_temp_avg": cpu_c},
    }


# --- parse -----------------------------------------------------------------
def test_parse_extracts_fields():
    p = _payload(1030.0, 8.2, 0.45, 0.198, 0.18, 71.0, 77.0)
    s = um.parse_macmon_payload(p)
    assert s["gpu_freq_mhz"] == 1030.0
    assert s["gpu_power_w"] == 8.2
    assert s["gpu_usage_ratio"] == 0.45
    assert s["gpu_active_ratio"] == 0.198
    assert s["cpu_usage_ratio"] == 0.18
    assert s["gpu_temp_c"] == 71.0
    assert s["cpu_temp_c"] == 77.0


def test_parse_usage_and_active_are_distinct():
    # the two occupancy metrics are recorded SEPARATELY, not collapsed
    p = _payload(900.0, 6.0, 0.070, 0.198, 0.10, 60.0, 65.0)
    s = um.parse_macmon_payload(p)
    assert s["gpu_usage_ratio"] == 0.070
    assert s["gpu_active_ratio"] == 0.198
    assert s["gpu_usage_ratio"] != s["gpu_active_ratio"]


def test_parse_missing_fields_are_zero():
    s = um.parse_macmon_payload({})
    for k in ("gpu_freq_mhz", "gpu_power_w", "gpu_usage_ratio", "gpu_active_ratio",
              "gpu_temp_c"):
        assert s[k] == 0.0


# --- summarize -------------------------------------------------------------
def test_summarize_min_mean_max_and_series():
    samples = [um.parse_macmon_payload(_payload(f, 8.0, 0.4, 0.2, 0.2, 70.0, 76.0))
               for f in (580.0, 1030.0, 1381.0)]
    for i, s in enumerate(samples):
        s["t"] = float(i)
    summ = um.summarize_utilization(samples)
    assert summ["samples"] == 3
    assert summ["gpu_freq_mhz"]["min"] == 580.0
    assert summ["gpu_freq_mhz"]["max"] == 1381.0
    assert abs(summ["gpu_freq_mhz"]["mean"] - (580 + 1030 + 1381) / 3) < 1e-6
    assert len(summ["series"]) == 3
    assert set(summ["series"][0]) >= {"gpu_freq_mhz", "gpu_power_w", "gpu_usage_ratio",
                                      "gpu_active_ratio", "t"}


def test_summarize_empty():
    assert um.summarize_utilization([]) == {"samples": 0}


def test_summarize_drop_series():
    samples = [um.parse_macmon_payload(_payload(1000.0, 8.0, 0.4, 0.2, 0.2, 70.0, 76.0))]
    summ = um.summarize_utilization(samples, keep_series=False)
    assert "series" not in summ
    assert summ["samples"] == 1


# --- census ----------------------------------------------------------------
def test_census_line_prints_gpu_usage_ratio_as_busy():
    # busy% must be gpu_usage_ratio (0.45), NOT gpu_active_ratio (0.198)
    samples = [um.parse_macmon_payload(_payload(1030.0, 8.2, 0.45, 0.198, 0.18, 71.0, 77.0))]
    line = um.census_line(um.summarize_utilization(samples, keep_series=False))
    assert "gpu" in line and "W" in line and "MHz" in line and "% busy" in line
    assert "8.2 W" in line
    assert "45% busy" in line  # gpu_usage_ratio, not the 20% active_ratio
    assert "20% busy" not in line


def test_census_line_no_samples():
    assert "no samples" in um.census_line({"samples": 0})


# --- sampler (reader mocked via injected JSON lines) -----------------------
def test_sampler_with_injected_lines():
    lines = [
        json.dumps(_payload(580.0, 8.0, 0.30, 0.15, 0.15, 70.0, 75.0)),
        "not json - must be skipped",
        json.dumps(_payload(1381.0, 9.0, 0.60, 0.30, 0.20, 72.0, 78.0)),
    ]
    with um.UtilizationSampler(_lines=lines) as s:
        pass  # the background loop drains the injected lines
    s._thread.join(timeout=3)
    summ = s.summarize()
    assert summ["samples"] == 2  # the bad line was skipped
    assert summ["gpu_freq_mhz"]["min"] == 580.0
    assert summ["gpu_freq_mhz"]["max"] == 1381.0
    assert "MHz" in s.census() and "busy" in s.census()  # census renders from the trace


def test_sampler_missing_binary_is_graceful_noop():
    with um.UtilizationSampler(macmon_bin="/nonexistent/macmon-xyz") as s:
        pass
    assert s.samples == []
    assert "no samples" in s.census()


# --- read_once / cooldown (reader mocked via nonexistent binary) -----------
def test_read_once_missing_binary_returns_empty():
    assert um.read_once(macmon_bin="/nonexistent/macmon-xyz") == {}


def test_cooldown_zero_is_noop_and_returns_block():
    block = um.cooldown(0.0, macmon_bin="/nonexistent/macmon-xyz", label="test")
    assert block["seconds"] == 0.0
    assert block["start"] == {} and block["end"] == {}
