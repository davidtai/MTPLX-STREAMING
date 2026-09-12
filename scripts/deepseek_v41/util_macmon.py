"""W90 macmon utilization sampler for the DSV4.1 decode benches (shared, CPU-safe).

macmon during a live 16,384-token decode read 71 C (NOT thermal), ~45% GPU-busy,
GPU frequency swinging 580-1381 MHz: the GPU DVFS-downclocks in the sync/latency
gaps between B=1 dispatch bursts, so each post-gap burst runs at a reduced clock --
the mode-independent in-situ decode floor (the isolated bench keeps the GPU awake).
This module samples ``macmon pipe`` in a background thread over the TIMED DECODE of
both ``ab_decode_env_levers.py`` and ``metal_decode_attn_bisect.py --in-model`` and
folds the trace into a ``utilization`` receipt block (min/mean/max + per-sample
series) plus a one-line census.

Sudo-free: reuses ``/opt/homebrew/bin/macmon`` exactly as
``scripts/fable/server_cell_bench.py`` (``read_machine_temperature`` /
``MacmonTrace``).  Graceful no-op when macmon is absent (empty samples, never
raises).  The parse/summary are pure functions and the sampler accepts an injected
line source, so it is unit-tested with the reader mocked -- no Metal, no sudo.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
from typing import Any, Iterable, Optional

MACMON = "/opt/homebrew/bin/macmon"


def _resolve_bin(macmon_bin: Optional[str] = None) -> str:
    """The macmon binary to use: an explicit argument wins, else the
    ``MTPLX_MACMON_BIN`` env override (a test points it at a nonexistent path for a
    graceful no-op), else the default.  Read at use, never frozen at import."""
    if macmon_bin is not None:
        return macmon_bin
    return os.environ.get("MTPLX_MACMON_BIN") or MACMON

#: The utilization fields the W90 discriminator needs, per macmon sample.  Names
#: follow the coordinator's ``macmon pipe`` schema; read with fallbacks so a macmon
#: version that spells a field differently (e.g. ``gpu_active_ratio``) still lands.
_FIELDS = (
    "gpu_freq_mhz", "gpu_power_w", "gpu_busy_ratio",
    "cpu_busy_ratio", "gpu_temp_c", "cpu_temp_c",
)


def parse_macmon_payload(payload: dict) -> dict:
    """Extract the W90 utilization fields from one macmon JSON sample as floats.

    Robust to field-name drift across macmon versions (``gpu_usage_ratio`` vs
    ``gpu_active_ratio``) and to missing fields (0.0)."""
    temp = payload.get("temp") or {}

    def _f(*keys, src=payload) -> float:
        for k in keys:
            v = src.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return 0.0

    return {
        "gpu_freq_mhz": _f("gpu_freq_mhz"),
        "gpu_power_w": _f("gpu_power"),
        "cpu_power_w": _f("cpu_power"),
        "all_power_w": _f("all_power"),
        "gpu_busy_ratio": _f("gpu_usage_ratio", "gpu_active_ratio"),
        "cpu_busy_ratio": _f("cpu_usage_ratio", "cpu_active_ratio"),
        "gpu_temp_c": _f("gpu_temp_avg", src=temp),
        "cpu_temp_c": _f("cpu_temp_avg", src=temp),
    }


def summarize_utilization(samples: list, *, keep_series: bool = True) -> dict:
    """Fold parsed samples into ``{samples, <field>:{min,mean,max}, [series]}``."""
    if not samples:
        return {"samples": 0}
    out: dict[str, Any] = {"samples": len(samples)}
    for f in _FIELDS:
        xs = [s[f] for s in samples if f in s and s[f] is not None]
        if xs:
            out[f] = {"min": min(xs), "mean": statistics.fmean(xs), "max": max(xs)}
    if keep_series:
        out["series"] = [
            {k: s.get(k) for k in (*_FIELDS, "t")} for s in samples
        ]
    return out


def census_line(summary: dict) -> str:
    """One-line GPU census: ``gpu <W> / <MHz> / <busy%>`` + freq range + temps."""
    if not summary or not summary.get("samples"):
        return "utilization: no samples (macmon unavailable)"

    def _m(field: str, key: str = "mean"):
        d = summary.get(field)
        return d.get(key) if isinstance(d, dict) else None

    def _n(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) else "?"

    gb = _m("gpu_busy_ratio")
    return (
        f"utilization[n={summary['samples']}]: "
        f"gpu {_n(_m('gpu_power_w'), '{:.1f}')} W / "
        f"{_n(_m('gpu_freq_mhz'), '{:.0f}')} MHz / "
        f"{_n(gb * 100 if isinstance(gb, (int, float)) else None, '{:.0f}')}% busy "
        f"(freq {_n(_m('gpu_freq_mhz', 'min'), '{:.0f}')}-"
        f"{_n(_m('gpu_freq_mhz', 'max'), '{:.0f}')} MHz), "
        f"gpu {_n(_m('gpu_temp_c'), '{:.0f}')}C cpu {_n(_m('cpu_temp_c'), '{:.0f}')}C"
    )


class UtilizationSampler:
    """Background ``macmon pipe -s 0 -i <interval>`` sampler over a timed region.

    Use as a context manager wrapping the TIMED DECODE only.  A single long-lived
    macmon process (not one spawn per sample) so the sampling never perturbs what it
    measures.  Graceful no-op if macmon is missing (``samples`` stays empty).  For
    tests pass ``_lines`` (an iterable of macmon JSON strings) instead of spawning a
    subprocess -- the reader is then fully mocked."""

    def __init__(self, interval_ms: int = 2000, macmon_bin: Optional[str] = None,
                 _lines: Optional[Iterable[str]] = None) -> None:
        self.interval_ms = int(interval_ms)
        self.macmon_bin = _resolve_bin(macmon_bin)
        self.samples: list[dict] = []
        self._lines = _lines
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._t0: Optional[float] = None

    def _record(self, payload: dict) -> None:
        s = parse_macmon_payload(payload)
        s["t"] = time.monotonic() - (self._t0 or 0.0)
        self.samples.append(s)

    def _loop(self, lines: Iterable[str]) -> None:
        for line in lines:
            if self._stop.is_set():
                break
            try:
                self._record(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue

    def __enter__(self) -> "UtilizationSampler":
        self._t0 = time.monotonic()
        if self._lines is not None:
            self._thread = threading.Thread(
                target=self._loop, args=(list(self._lines),), daemon=True
            )
            self._thread.start()
            return self
        try:
            self._proc = subprocess.Popen(
                [self.macmon_bin, "pipe", "-s", "0", "-i", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except OSError:
            self._proc = None
            return self
        self._thread = threading.Thread(
            target=self._loop, args=(self._proc.stdout,), daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> bool:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return False

    def summarize(self, *, keep_series: bool = True) -> dict:
        return summarize_utilization(self.samples, keep_series=keep_series)

    def census(self) -> str:
        return census_line(self.summarize(keep_series=False))


def read_once(macmon_bin: Optional[str] = None) -> dict:
    """One parsed macmon sample (for the cooldown thermal readout); ``{}`` if the
    reader is unavailable."""
    macmon_bin = _resolve_bin(macmon_bin)
    try:
        proc = subprocess.Popen(
            [macmon_bin, "pipe", "-s", "1", "-i", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except OSError:
        return {}
    try:
        line = proc.stdout.readline() if proc.stdout else ""
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    if not line:
        return {}
    try:
        return parse_macmon_payload(json.loads(line))
    except (json.JSONDecodeError, TypeError):
        return {}


def cooldown(seconds: float, *, macmon_bin: Optional[str] = None,
             label: str = "dsv41") -> dict:
    """Idle ``seconds`` AFTER prefill and BEFORE the timed decode (so TTFT, measured
    from the prefill, is unaffected).  Reads a thermal/util sample at start and end.
    Returns a ``cooldown`` receipt block.  ``seconds <= 0`` is a no-op readout."""
    macmon_bin = _resolve_bin(macmon_bin)
    seconds = float(seconds or 0.0)
    start = read_once(macmon_bin)
    if seconds > 0:
        print(
            f"[{label}] cooldown {seconds:.0f}s (post-prefill, pre-decode); "
            f"start gpu {start.get('gpu_temp_c', '?')}C "
            f"{start.get('gpu_freq_mhz', '?')}MHz",
            flush=True,
        )
        time.sleep(seconds)
    end = read_once(macmon_bin)
    return {"seconds": seconds, "start": start, "end": end}
