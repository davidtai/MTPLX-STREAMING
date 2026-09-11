"""Server-side cell benchmark: one OpenAI-compatible endpoint, many contexts.

Measures what David asked for, per request, against ANY OpenAI-compatible
server (the branch's ``mtplx.server.openai``, upstream MTPLX v2.10.2, or
mlx-serve):

    prefill time s (and derived prefill tok/s), decode tok/s, peak memory GB,
    TTFT s, wall time s

The client is pure HTTP + stdlib: this module NEVER imports mlx and never
touches the GPU, so the report/table/chart half is unit-testable on CPU while
a guarded GPU window is running.

Two cells:

``vanity``
    A pinned ~100-token Python task, no system prompt, ``max_tokens`` 1024,
    natural stop. Zero prefill to speak of -- this is the "how fast does it
    feel" number. It samples EXACTLY like a sweep cell (temperature 1,
    top-p 0.95, top-k 20, seeded from the same production seeds) and differs
    from one in the prompt and in ``enable_thinking`` alone: thinking OFF,
    carried by ``enable_thinking=false`` with NO ``reasoning_effort`` field.
    See :data:`VANITY_ENABLE_THINKING` for why that spelling, and only that
    spelling, means the same thing on all three engines.

``sweep``
    The same 1,024-token Python task at the END of the prompt, with filler
    context in front so the TEMPLATED prompt hits the target exactly. The
    1,024 point is the task alone with no filler. Targets:

        1,024 / 8,192 / 16,384 / 32,768 / 65,536 / 131,072 / 262,144

    temperature 1 / top-p 0.95 / top-k 20, reasoning ``xhigh``, seeds
    20260829/20260830/20260831 via the API ``seed`` field, ``max_tokens`` 1024.

Every timed request is preceded by the SAME 40 C thermal gate the ABBA driver
uses (``scripts.fable.abba_driver.wait_for_temperature``), and the ready
temperature lands in the receipt.

Peak memory has two independent sources and the receipt says which is which:

``server``
    Whatever the server reports for the request (MTPLX exposes a metrics
    envelope; see ``--memory-field``). Absent on servers that report nothing.

``client_rss``
    ``ps -o rss=`` on the server pid, sampled at 0.5 s for the life of the
    request, maximum taken. Always available when ``--server-pid`` is given.

Sanity-check gate: ``--stop-after-context`` (default 16384) refuses to run a
larger context than that, so a battery pauses for review after the 16K point.
Resume by passing the remaining sizes to ``--contexts``; receipts merge into
one report because ``--report`` reads the whole receipt directory.

The branch arm's retained ``MTPLX_FABLE_*`` set is FILE-DRIVEN
(``--fable-flags-file``, repeatable and multi-valued, KEY=VALUE per line).
Given at all it REPLACES the harness's ``DEFAULT_FABLE_FLAGS`` -- a flag the
files do not name is off -- merges OVER the derived family env, and stays
UNDER ``--env``. Every file's sha256 and the resolved key set with its
per-key provenance land in both receipts, so a battery is reproducible from
the receipt alone. The harness refuses, before anything boots, a key the
server owns (:data:`AUTO_ARMED_KEYS`), an ``MTPLX_FABLE_*`` name the served
tree never reads, and anything outside the ``MTPLX_`` namespace.

DEPENDENCY. There is no typed registry of the ``MTPLX_FABLE_*`` space to
check against: W61's ``mtplx/full_stack_env.py`` covers MTPLX_QWEN4_ /
MTPLX_QSA_ / MTPLX_FRSPEC_ only, and as of 2026-09-02 it is on
``worker/w61-restack-profile`` and NOT merged into
``experiments/fable-qwen38-80tps`` (14d18189). So the known set is scanned
out of the served tree's own sources -- ``--flag-registry-root``, defaulting
to ``--server-cwd`` on a branch arm and otherwise to
:data:`MTPLX_SOURCE_ROOT`. When W61 lands, its declared names are unioned in
automatically.

``--dry-run`` prints the exact argv, the exact env and the per-cell request
plan this invocation would run, then exits 0 without starting a server or
sending a request. The output carries no clock, no pid and no receipt name,
env and flag keys are sorted, and cells appear in plan order, so two runs are
byte-identical and three engines' plans diff against each other.

PARITY. Every cell carries a ``body_sha256`` over the request with the
engine-specific fields removed (:data:`PARITY_TRANSPORT_FIELDS`, which is
``model`` and nothing else). Three engines whose cells share a digest asked
for the same thing, and the dry run proves it before the GPU is touched. The
receipt records the same digest computed from the request that ACTUALLY went
out, plus ``response_parity`` -- completion tokens, finish reason, and the
thinking-phase size -- because identical requests do not prove identical
honouring, and an engine that ignores a field shows up only on that side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import ctypes
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The full battery, ascending. ``--contexts`` selects a subset.
CONTEXT_BATTERY: tuple[int, ...] = (
    1_024,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
)

#: The sanity-check pause. Nothing above this runs without an explicit
#: ``--stop-after-context`` raise, so a battery stops for review at 16K.
DEFAULT_STOP_AFTER_CONTEXT = 16_384

PRODUCTION_SEEDS: tuple[int, ...] = (20260829, 20260830, 20260831)

DEFAULT_MAX_TOKENS = 1_024

#: Pinned vanity prompt. Its templated token count is recorded per run rather
#: than asserted here, because it depends on the server's chat template.
VANITY_PROMPT = (
    "Write a Python function `is_palindrome(text: str) -> bool` that checks "
    "whether a string is a palindrome, ignoring case and any non-alphanumeric "
    "characters. Give it a docstring explaining the normalisation rule, and "
    "follow it with exactly three `assert` statements covering: a phrase with "
    "punctuation and mixed case, the empty string, and a non-palindrome."
)
VANITY_PROMPT_SHA256 = hashlib.sha256(VANITY_PROMPT.encode("utf-8")).hexdigest()

#: The 1,024-token coding task that ends every sweep prompt.
SWEEP_INSTRUCTION_SUFFIX = (
    "\n\nAnswer with a SHORT markdown report: one sentence of summary, then a "
    "bulleted list of exactly three observations about the module above. "
    "No code."
)

METRIC_SPECS: tuple[dict[str, Any], ...] = (
    {"key": "prefill_time_s", "label": "Prefill s", "unit": "s", "lower_is_better": True},
    {"key": "prefill_tok_s", "label": "Prefill tok/s", "unit": "tok/s", "lower_is_better": False},
    {"key": "decode_tok_s", "label": "Decode tok/s", "unit": "tok/s", "lower_is_better": False},
    {"key": "peak_memory_gb", "label": "Peak mem GB", "unit": "GB", "lower_is_better": True},
    {"key": "ttft_s", "label": "TTFT s", "unit": "s", "lower_is_better": True},
    {"key": "wall_s", "label": "Wall s", "unit": "s", "lower_is_better": True},
)


# ---------------------------------------------------------------------------
# Pure metric arithmetic (unit-tested; no I/O)
# ---------------------------------------------------------------------------


def decode_tok_s(completion_tokens: int, decode_s: float) -> float | None:
    """Tokens per second over the DECODE phase only.

    The first token is excluded from the numerator because ``decode_s`` is
    measured from the first delta, not from the request start: ``n`` tokens
    arriving after the first one span ``n - 1`` inter-token gaps.
    """

    tokens = int(completion_tokens)
    elapsed = float(decode_s)
    if tokens <= 1 or elapsed <= 0:
        return None
    return (tokens - 1) / elapsed


def prefill_tok_s(new_prefill_tokens: int, prefill_s: float) -> float | None:
    tokens = int(new_prefill_tokens)
    elapsed = float(prefill_s)
    if tokens <= 0 or elapsed <= 0:
        return None
    return tokens / elapsed


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def aggregate(values: Iterable[Any]) -> dict[str, Any]:
    """mean / spread over the seeds of one (server, context) point.

    ``spread`` is the population standard deviation, and is ``0.0`` for a
    single sample rather than ``None`` -- a one-seed point has no measured
    spread, which the report renders as a bare mean.
    """

    numbers = _finite(values)
    if not numbers:
        return {"n": 0, "mean": None, "spread": None, "min": None, "max": None}
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "spread": statistics.pstdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def cell_key(record: Mapping[str, Any]) -> tuple[str, int]:
    """``(cell, context)`` -- the vanity cell sorts before every sweep point."""

    cell = str(record.get("cell") or "sweep")
    if cell == "vanity":
        return ("vanity", -1)
    return ("sweep", int(record.get("target_tokens") or 0))


def summarize_records(
    records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Group receipts into ``server -> point -> metric -> aggregate``.

    Records that failed (``ok`` false) are counted but contribute no numbers,
    so a server that OOMs at 256K shows an explicit failure rather than a
    silently missing row.
    """

    servers: list[str] = []
    points: list[tuple[str, int]] = []
    for record in records:
        server = str(record.get("server") or "unknown")
        if server not in servers:
            servers.append(server)
        key = cell_key(record)
        if key not in points:
            points.append(key)
    points.sort(key=lambda item: (item[0] != "vanity", item[1]))

    table: dict[str, dict[str, Any]] = {}
    for server in servers:
        per_point: dict[str, Any] = {}
        for key in points:
            matching = [
                r
                for r in records
                if str(r.get("server") or "unknown") == server
                and cell_key(r) == key
            ]
            ok = [r for r in matching if r.get("ok", True)]
            failed = [r for r in matching if not r.get("ok", True)]
            metrics = {
                spec["key"]: aggregate(r.get(spec["key"]) for r in ok)
                for spec in METRIC_SPECS
            }
            per_point[point_name(key)] = {
                "attempts": len(matching),
                "ok": len(ok),
                "failed": len(failed),
                "failures": [
                    str(r.get("error") or "unknown") for r in failed
                ],
                "completion_tokens": aggregate(
                    r.get("completion_tokens") for r in ok
                ),
                "prompt_tokens": aggregate(r.get("prompt_tokens") for r in ok),
                "finish_reasons": sorted(
                    {str(r.get("finish_reason") or "?") for r in ok}
                ),
                "memory_sources": sorted(
                    {str(r.get("peak_memory_source") or "none") for r in ok}
                ),
                "metrics": metrics,
            }
        table[server] = per_point
    return {
        "servers": servers,
        "points": [point_name(key) for key in points],
        "by_server": table,
    }


def point_name(key: tuple[str, int]) -> str:
    cell, context = key
    if cell == "vanity":
        return "vanity"
    if context >= 1024 and context % 1024 == 0:
        return f"{context // 1024}K"
    return str(context)


def point_context(name: str) -> int:
    """Inverse of :func:`point_name` for chart x-axes; vanity sorts to 0."""

    if name == "vanity":
        return 0
    if name.endswith("K"):
        return int(name[:-1]) * 1024
    return int(name)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:,.{digits}f}"


def _fmt_mean_spread(entry: Mapping[str, Any], digits: int = 2) -> str:
    mean = entry.get("mean")
    if mean is None:
        return "-"
    spread = entry.get("spread")
    if not spread:
        return _fmt(mean, digits)
    return f"{_fmt(mean, digits)}±{_fmt(spread, digits)}"


def render_markdown_table(summary: Mapping[str, Any]) -> str:
    """One table: rows = context point, one column group per server."""

    servers = list(summary["servers"])
    points = list(summary["points"])
    by_server = summary["by_server"]

    header = ["Point"]
    for server in servers:
        for spec in METRIC_SPECS:
            header.append(f"{server} {spec['label']}")
        header.append(f"{server} Completion tok")
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    for point in points:
        row = [point]
        for server in servers:
            entry = by_server.get(server, {}).get(point)
            if entry is None:
                row.extend(["-"] * (len(METRIC_SPECS) + 1))
                continue
            if entry["ok"] == 0:
                reason = entry["failures"][0] if entry["failures"] else "no data"
                row.extend([f"FAIL ({reason})"] + ["-"] * len(METRIC_SPECS))
                continue
            for spec in METRIC_SPECS:
                digits = 1 if spec["key"] in {"prefill_tok_s"} else 2
                row.append(_fmt_mean_spread(entry["metrics"][spec["key"]], digits))
            row.append(_fmt(entry["completion_tokens"].get("mean"), 0))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _mermaid_label(text: str) -> str:
    """Quote a mermaid axis/series label.

    ``xychart-beta`` takes double-quoted strings; a literal quote inside one
    ends the string, so strip them rather than emit a chart that fails to
    render on GitHub.
    """

    return re.sub(r'["\n\r]', "", str(text))


def render_mermaid_chart(
    summary: Mapping[str, Any], metric_key: str
) -> str:
    """A GitHub-renderable ``xychart-beta`` line chart: metric vs context.

    One ``line`` per server. Mermaid has no per-series legend, so the series
    order is spelled out in the title.
    """

    spec = next(s for s in METRIC_SPECS if s["key"] == metric_key)
    servers = list(summary["servers"])
    points = list(summary["points"])
    by_server = summary["by_server"]

    series: dict[str, list[float | None]] = {}
    for server in servers:
        values: list[float | None] = []
        for point in points:
            entry = by_server.get(server, {}).get(point)
            mean = (
                entry["metrics"][metric_key]["mean"]
                if entry and entry.get("ok")
                else None
            )
            values.append(mean)
        series[server] = values

    everything = _finite(v for values in series.values() for v in values)
    if not everything:
        return ""
    ceiling = max(everything)
    floor = min(everything)
    if ceiling == floor:
        ceiling = floor + 1.0
    pad = (ceiling - floor) * 0.1
    y_min = max(0.0, floor - pad)
    y_max = ceiling + pad

    order = ", ".join(servers)
    title = f"{spec['label']} vs context  (series order: {order})"
    lines = [
        "```mermaid",
        "xychart-beta",
        f'    title "{_mermaid_label(title)}"',
        "    x-axis ["
        + ", ".join(f'"{_mermaid_label(p)}"' for p in points)
        + "]",
        f'    y-axis "{_mermaid_label(spec["label"])}" '
        f"{y_min:.4g} --> {y_max:.4g}",
    ]
    for server in servers:
        # Mermaid cannot render a gap; a missing point is carried as the
        # y-axis floor and the table above is authoritative for failures.
        rendered = [
            f"{value:.6g}" if value is not None else f"{y_min:.6g}"
            for value in series[server]
        ]
        lines.append("    line [" + ", ".join(rendered) + "]")
    lines.append("```")
    return "\n".join(lines)


def render_all_mermaid(summary: Mapping[str, Any]) -> str:
    blocks = []
    for spec in METRIC_SPECS:
        chart = render_mermaid_chart(summary, spec["key"])
        if chart:
            blocks.append(f"### {spec['label']}\n\n{chart}")
    return "\n\n".join(blocks)


def render_report(
    summary: Mapping[str, Any], *, notes: Sequence[str] = ()
) -> str:
    parts = ["## Table", "", render_markdown_table(summary), ""]
    if notes:
        parts.extend(["### Notes", ""])
        parts.extend(f"- {note}" for note in notes)
        parts.append("")
    parts.extend(["## Charts", "", render_all_mermaid(summary), ""])
    return "\n".join(parts)


def render_png(summary: Mapping[str, Any], path: Path) -> Path | None:
    """Five stacked panels, one per metric. ``None`` if matplotlib is absent."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    servers = list(summary["servers"])
    points = list(summary["points"])
    by_server = summary["by_server"]
    x = list(range(len(points)))

    fig, axes = plt.subplots(
        len(METRIC_SPECS), 1, figsize=(10, 3.1 * len(METRIC_SPECS)), sharex=True
    )
    if len(METRIC_SPECS) == 1:
        axes = [axes]
    for ax, spec in zip(axes, METRIC_SPECS):
        for server in servers:
            ys = []
            for point in points:
                entry = by_server.get(server, {}).get(point)
                ys.append(
                    entry["metrics"][spec["key"]]["mean"]
                    if entry and entry.get("ok")
                    else None
                )
            ax.plot(x, ys, marker="o", label=server)
        ax.set_ylabel(f"{spec['label']}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize="small")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(points)
    axes[-1].set_xlabel("prefill context")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Thermal gate + process memory sampling
# ---------------------------------------------------------------------------


#: Pinned to ``abba_driver.DEFAULT_THERMAL_MAX_C``. The gate is reimplemented
#: here rather than imported because importing ``abba_driver`` pulls in mlx,
#: which would create a SECOND Metal context inside the benchmark client while
#: the server under test owns the GPU. ``test_thermal_threshold_matches_abba``
#: parses the driver's source (no import) so the two cannot drift apart.
DEFAULT_THERMAL_MAX_C = 40.0

ABBA_DRIVER_PATH = Path(
    "/Users/davidtai/projects/OpenSourceWTF/.worktrees/"
    "qwen38-fable-80tps/scripts/fable/abba_driver.py"
)

MACMON = "/opt/homebrew/bin/macmon"


def read_machine_temperature() -> dict[str, float]:
    process = subprocess.Popen(
        [MACMON, "pipe", "-s", "1", "-i", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("macmon did not expose stdout")
        line = process.stdout.readline()
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    payload = json.loads(line)
    temperature = payload.get("temp") or {}
    cpu_c = float(temperature["cpu_temp_avg"])
    gpu_c = float(temperature["gpu_temp_avg"])
    return {"cpu_c": cpu_c, "gpu_c": gpu_c, "max_c": max(cpu_c, gpu_c)}


def wait_for_temperature(
    max_celsius: float = DEFAULT_THERMAL_MAX_C, *, deadline_s: float = 3600.0
) -> dict[str, Any]:
    """Block outside the timed region until CPU and GPU are both cool."""

    started = time.monotonic()
    deadline = started + float(deadline_s)
    samples = 0
    initial_celsius: float | None = None
    sensor = "macmon:max(cpu_temp_avg,gpu_temp_avg)"
    while True:
        reading = read_machine_temperature()
        samples += 1
        celsius = float(reading["max_c"])
        if initial_celsius is None:
            initial_celsius = celsius
        print(
            f"[server-cell] thermal gate {sensor}={celsius:.1f}C "
            f"target<={max_celsius:.1f}C",
            flush=True,
        )
        if celsius <= max_celsius:
            return {
                "threshold_c": max_celsius,
                "initial_c": initial_celsius,
                "ready_c": celsius,
                "ready_cpu_c": float(reading["cpu_c"]),
                "ready_gpu_c": float(reading["gpu_c"]),
                "sensor": sensor,
                "wait_s": time.monotonic() - started,
                "samples": samples,
            }
        if time.monotonic() >= deadline:
            raise RuntimeError(f"thermal gate timed out at {celsius:.1f}C")
        time.sleep(10)


# ---------------------------------------------------------------------------
# Page-cache prewarm, clock wake, and the 1 Hz thermal trace
# ---------------------------------------------------------------------------

PREWARM_CHUNK_BYTES = 64 * 1024 * 1024

#: Files to pre-read after a server launch, by stack. The in-process ABBA
#: arms read the n-gram table end to end before timing; the server path never
#: did, and an as-found page cache costs the same trajectory 56 tok/s against
#: 68.8 tok/s pre-read. Every arm must therefore start from the same regime.
NGRAM_TABLE_GLOBS: dict[str, tuple[str, ...]] = {
    "branch": ("ngram-table.safetensors",),
    "upstream": ("ngram-table.safetensors",),
    "mlx-serve": ("ngram_table.bin", "ngram-table.safetensors"),
}


def prewarm_file(path: Path, chunk_bytes: int = PREWARM_CHUNK_BYTES) -> dict[str, Any]:
    """Read one file end to end so its pages are resident before timing."""

    size = path.stat().st_size
    total = 0
    started = time.monotonic()
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            total += len(block)
    elapsed = time.monotonic() - started
    if total != size:
        raise RuntimeError(f"prewarm read {total} of {size} bytes from {path}")
    return {
        "path": str(path),
        "bytes": total,
        "seconds": elapsed,
        "gib_per_s": (total / 1024**3) / elapsed if elapsed > 0 else None,
        "chunk_bytes": int(chunk_bytes),
    }


def prewarm_model_tables(model_dir: Path, stack: str) -> list[dict[str, Any]]:
    """Pre-read every n-gram/PLE table the pack ships. Empty list if none."""

    results: list[dict[str, Any]] = []
    for name in NGRAM_TABLE_GLOBS.get(stack, ()):
        for path in sorted(model_dir.glob(name)):
            record = prewarm_file(path)
            print(
                "[server-cell] prewarm " + json.dumps(record, sort_keys=True),
                flush=True,
            )
            results.append(record)
    if not results:
        print(
            f"[server-cell] prewarm: no n-gram/PLE table found under {model_dir} "
            f"for stack {stack}",
            flush=True,
        )
    return results


class MacmonTrace:
    """1 Hz macmon trace for the life of one timed request.

    A single long-lived ``macmon pipe`` process rather than one spawn per
    sample: spawning costs ~100 ms, which at 1 Hz would be 10% of the sampling
    interval and would itself perturb what it measures.
    """

    def __init__(self, interval_ms: int = 1000) -> None:
        self.interval_ms = int(interval_ms)
        self.samples: list[dict[str, float]] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if self._stop.is_set():
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            temperature = payload.get("temp") or {}
            try:
                self.samples.append(
                    {
                        "t": time.monotonic(),
                        "cpu_c": float(temperature.get("cpu_temp_avg") or 0.0),
                        "gpu_c": float(temperature.get("gpu_temp_avg") or 0.0),
                        "gpu_power_w": float(payload.get("gpu_power") or 0.0),
                        "gpu_freq_mhz": float(payload.get("gpu_freq_mhz") or 0.0),
                        "gpu_active_ratio": float(payload.get("gpu_active_ratio") or 0.0),
                        "all_power_w": float(payload.get("all_power") or 0.0),
                    }
                )
            except (TypeError, ValueError):
                continue

    def __enter__(self) -> "MacmonTrace":
        try:
            self._process = subprocess.Popen(
                [MACMON, "pipe", "-s", "0", "-i", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self._process = None
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def summarize(self, *, split_at: float | None = None) -> dict[str, Any]:
        """Fold the trace into the per-phase numbers the report needs.

        ``split_at`` is the monotonic timestamp of the first token, so the
        prefill phase and the decode phase get separate maxima.
        """

        if not self.samples:
            return {"samples": 0}

        def window(rows: list[dict[str, float]]) -> dict[str, Any]:
            if not rows:
                return {}
            return {
                "max_cpu_c": max(r["cpu_c"] for r in rows),
                "max_gpu_c": max(r["gpu_c"] for r in rows),
                "mean_gpu_power_w": statistics.fmean(r["gpu_power_w"] for r in rows),
                "max_gpu_power_w": max(r["gpu_power_w"] for r in rows),
                "min_gpu_freq_mhz": min(r["gpu_freq_mhz"] for r in rows),
                "max_gpu_freq_mhz": max(r["gpu_freq_mhz"] for r in rows),
                "mean_gpu_freq_mhz": statistics.fmean(r["gpu_freq_mhz"] for r in rows),
                "n": len(rows),
            }

        peak = [max(r["cpu_c"], r["gpu_c"]) for r in self.samples]
        interval_s = self.interval_ms / 1000.0
        prefill = (
            [r for r in self.samples if split_at is None or r["t"] <= split_at]
            if split_at is not None
            else self.samples
        )
        decode = (
            [r for r in self.samples if r["t"] > split_at]
            if split_at is not None
            else []
        )
        return {
            "samples": len(self.samples),
            "interval_s": interval_s,
            "max_c": max(peak),
            "seconds_above_85c": sum(1 for c in peak if c > 85.0) * interval_s,
            "seconds_above_95c": sum(1 for c in peak if c > 95.0) * interval_s,
            "prefill": window(prefill),
            "decode": window(decode),
            "trace": [
                {k: round(v, 3) for k, v in row.items() if k != "t"}
                for row in self.samples
            ],
        }


class VmStatSampler:
    """vm_stat at 1 Hz for the life of a request; keeps first, last, delta.

    A pressure trim already fired inside a 16K cell and the wake request that
    followed took 12.41 s against 0.87 s for its neighbours. Without this a
    paging stall is indistinguishable from an engine regression.
    """

    FIELDS = (
        "Pageins", "Pageouts", "Swapins", "Swapouts",
        "Decompressions", "Compressions",
        "Pages free", "Pages inactive", "Pages wired down", "Pages active",
    )

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = float(interval_s)
        self.first: dict[str, int] = {}
        self.last: dict[str, int] = {}
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            snap = vm_stat_pages()
            if snap:
                self.samples += 1
                if not self.first:
                    self.first = snap
                self.last = snap
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "VmStatSampler":
        self.first = vm_stat_pages()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.last = vm_stat_pages() or self.last

    def summary(self) -> dict[str, Any]:
        if not self.first or not self.last:
            return {"samples": self.samples}
        page = self.last.get("_page_bytes") or 16384
        delta = {
            k: self.last.get(k, 0) - self.first.get(k, 0)
            for k in self.FIELDS
            if k in self.last and k in self.first
        }
        paging = (
            delta.get("Swapins", 0) > 0
            or delta.get("Swapouts", 0) > 0
            or delta.get("Pageins", 0) > 1000
        )
        return {
            "samples": self.samples,
            "page_bytes": page,
            "delta_counts": delta,
            "delta_free_bytes": delta.get("Pages free", 0) * page,
            "delta_wired_bytes": delta.get("Pages wired down", 0) * page,
            # A paging cell must be flagged, never quietly averaged in.
            "paging_suspected": paging,
        }


def scan_pressure_events(log_path: Path | None, since_offset: int) -> dict[str, Any]:
    """Memory-pressure guard lines the server emitted during one request."""

    if not log_path or not log_path.exists():
        return {"available": False, "events": [], "offset": since_offset}
    try:
        with log_path.open("r", errors="replace") as handle:
            handle.seek(since_offset)
            chunk = handle.read()
            offset = handle.tell()
    except OSError:
        return {"available": False, "events": [], "offset": since_offset}
    events = re.findall(r"memory pressure guard \{[^}]*\}", chunk)
    return {
        "available": True,
        "events": events[:8],
        "event_count": len(events),
        "offset": offset,
    }


class MetricsPoller:
    """Poll a JSON metrics endpoint at 1 Hz and keep per-key maxima.

    mlx-serve reports memory only as server-wide Prometheus gauges, never per
    request, so the only way to attribute memory to a call is to watch the
    gauge across it.
    """

    def __init__(self, url: str | None, keys: Sequence[str], interval_s: float = 1.0):
        self.url = url
        self.keys = list(keys)
        self.interval_s = float(interval_s)
        self.maxima: dict[str, float] = {}
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _record(self, key: str, value: Any) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if key not in self.maxima or number > self.maxima[key]:
            self.maxima[key] = number

    def _sample(self) -> None:
        if not self.url:
            return
        # /metrics.json did not expose the gauges under the documented names
        # in the smoke test, so fall back to the Prometheus text endpoint and
        # parse it directly rather than silently reporting no memory at all.
        try:
            payload = http_get_json(self.url, timeout=3.0)
            self.samples += 1
            found = False
            for key in self.keys:
                value = _dig(payload, (key,))
                if value is not None:
                    found = True
                    self._record(key, value)
            if found:
                return
        except Exception:  # noqa: BLE001 - a missing sample must not fail a run
            pass
        text_url = self.url[: -len(".json")] if self.url.endswith(".json") else self.url
        try:
            with urllib.request.urlopen(text_url, timeout=3.0) as response:
                body = response.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name = parts[0].split("{", 1)[0].strip()
            for key in self.keys:
                if name == key:
                    self._record(key, parts[1])

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "MetricsPoller":
        if self.url:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


RUSAGE_INFO_V4 = 4


class _RUsageInfoV4(ctypes.Structure):
    """``rusage_info_v4`` from libproc, with the v5/v6 tail included.

    The trailing ``ri_flags`` belongs to v5+, so this struct is 304 bytes
    against the kernel's 296 for RUSAGE_INFO_V4. Harmless -- the kernel writes
    only its own 296 bytes and that field is never read -- but it is why the
    name and the layout do not match exactly.

    ``ri_phys_footprint`` is the number Activity Monitor calls Memory and is
    the only cheap per-process figure that counts MLX's unified-memory
    buffers; ``ps -o rss`` does not, which is why it read 49-53 GB against a
    server reporting 77 GB. ~1 us per call, so it can be sampled per request
    without spawning ``footprint`` or ``vmmap``.
    """

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
            "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
            "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups", "ri_child_pageins",
            "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
            "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time",
            "ri_logical_writes", "ri_lifetime_max_phys_footprint",
            "ri_instructions", "ri_cycles", "ri_billed_energy",
            "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time", "ri_flags",
        )
    ]


_LIBPROC = None


def _libproc() -> Any:
    global _LIBPROC
    if _LIBPROC is None:
        try:
            _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        except OSError:
            _LIBPROC = False
    return _LIBPROC


def proc_rusage(pid: int) -> dict[str, int] | None:
    """phys_footprint / lifetime max / pageins for one pid, or None."""

    lib = _libproc()
    if not lib:
        return None
    info = _RUsageInfoV4()
    rc = lib.proc_pid_rusage(
        ctypes.c_int(int(pid)), ctypes.c_int(RUSAGE_INFO_V4), ctypes.byref(info)
    )
    if rc != 0:
        return None
    return {
        "phys_footprint": int(info.ri_phys_footprint),
        "lifetime_max_phys_footprint": int(info.ri_lifetime_max_phys_footprint),
        "interval_max_phys_footprint": int(info.ri_interval_max_phys_footprint),
        "pageins": int(info.ri_pageins),
        "resident_size": int(info.ri_resident_size),
        "wired_size": int(info.ri_wired_size),
    }


def vm_stat_pages() -> dict[str, int]:
    """Page counters for residency deltas across a request."""

    try:
        text = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {}
    out: dict[str, int] = {}
    page = 16384
    header = re.search(r"page size of (\d+)", text)
    if header:
        page = int(header.group(1))
    out["_page_bytes"] = page
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        digits = "".join(c for c in raw if c.isdigit())
        if digits:
            out[key.strip()] = int(digits)
    return out


def memory_pressure_level() -> dict[str, Any]:
    """Raw kernel memory-pressure report, kept verbatim.

    Stored rather than reduced to a bool: a cell that ran under anything but
    the normal level is suspect, and the level itself is the evidence.
    """

    try:
        text = subprocess.check_output(
            ["memory_pressure", "-Q"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}
    lowered = text.lower()
    level = next(
        (n for n in ("critical", "warn", "normal") if n in lowered), "unknown"
    )
    return {"available": True, "level": level, "raw": text.splitlines()[-1][:160]}


class ProcSampler:
    """Sample phys_footprint at 0.5 s for one pid and keep the maximum."""

    def __init__(self, pid: int | None, interval_s: float = 0.5) -> None:
        self.pid = int(pid) if pid else None
        self.interval_s = float(interval_s)
        self.max_footprint: int | None = None
        self.first: dict[str, int] | None = None
        self.last: dict[str, int] | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            snap = proc_rusage(self.pid) if self.pid else None
            if snap:
                self.samples += 1
                if self.first is None:
                    self.first = snap
                self.last = snap
                value = snap["phys_footprint"]
                if self.max_footprint is None or value > self.max_footprint:
                    self.max_footprint = value
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "ProcSampler":
        if self.pid is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "samples": self.samples,
            "max_phys_footprint": self.max_footprint,
        }
        if self.first and self.last:
            out["pageins_delta"] = self.last["pageins"] - self.first["pageins"]
            out["lifetime_max_phys_footprint"] = self.last[
                "lifetime_max_phys_footprint"
            ]
        return out


class RssSampler:
    """Sample ``ps -o rss=`` on a pid at 0.5 s and keep the maximum.

    ``ps`` reports resident set size in KiB. This is the client-side memory
    number: it is the SERVER PROCESS footprint, which on Apple silicon
    includes the unified-memory buffers, and it is the only memory source
    available for a server that reports nothing itself.
    """

    def __init__(self, pid: int | None, interval_s: float = 0.5) -> None:
        self.pid = int(pid) if pid else None
        self.interval_s = float(interval_s)
        self.max_rss_bytes: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> int | None:
        if self.pid is None:
            return None
        try:
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(self.pid)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, OSError):
            return None
        digits = output.strip()
        if not digits.isdigit():
            return None
        return int(digits) * 1024

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self._sample_once()
            if value is not None:
                self.samples += 1
                if self.max_rss_bytes is None or value > self.max_rss_bytes:
                    self.max_rss_bytes = value
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "RssSampler":
        if self.pid is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Streaming OpenAI client
# ---------------------------------------------------------------------------


def _dig(payload: Any, keys: Sequence[str]) -> Any:
    """First non-null value found at any of ``keys``, searched depth-first."""

    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        for value in payload.values():
            found = _dig(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = _dig(value, keys)
            if found is not None:
                return found
    return None


SERVER_PREFILL_KEYS = ("prompt_eval_time_s", "prefill_time_s", "prompt_eval_duration_s")
#: mlx-serve reports llama.cpp-server-style timings in milliseconds.
SERVER_PREFILL_MS_KEYS = ("prompt_ms",)
SERVER_DECODE_MS_KEYS = ("predicted_ms",)
SERVER_PREFILL_TOKS_KEYS = ("prefill_tok_s", "prompt_tps", "prefill_compute_tok_s", "prompt_per_second")
SERVER_TTFT_KEYS = ("ttft_s",)
SERVER_DECODE_S_KEYS = ("decode_elapsed_s",)
SERVER_DECODE_TOKS_KEYS = ("decode_tok_s", "display_decode_tok_s", "predicted_per_second")
SERVER_NEW_PREFILL_KEYS = ("new_prefill_tokens", "prompt_n")
SERVER_CACHED_KEYS = ("cached_tokens", "cached_n")
SERVER_MEMORY_KEYS = (
    "peak_memory_bytes",
    "peak_memory",
    "active_memory_bytes",
)


def stream_chat(
    *,
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    enable_thinking: bool | None,
    timeout_s: float,
    api_key: str | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One streaming chat completion, fully instrumented.

    TTFT is the wall time to the first delta carrying either ``content`` or
    ``reasoning_content``: with thinking ON the first visible token is a
    reasoning token, and treating that as "not yet started" would inflate TTFT
    by the entire thinking phase.
    """

    body = chat_body(
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
    )
    body["messages"] = [{"role": "user", "content": prompt}]
    if extra_body:
        body.update(dict(extra_body))

    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": BENCH_USER_AGENT,
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )

    started = time.monotonic()
    first_delta_at: float | None = None
    last_delta_at: float | None = None
    content: list[str] = []
    reasoning: list[str] = []
    delta_count = 0
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    trailing: list[Any] = []
    server_error: dict[str, Any] | None = None

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            trailing.append(chunk)
            # MTPLX reports a mid-stream failure IN BAND: a chunk carrying
            # {"error": {...}} plus finish_reason "error", over HTTP 200. A
            # client that only reads choices/usage sees a successful request
            # that produced no tokens.
            if isinstance(chunk, Mapping) and chunk.get("error"):
                server_error = dict(chunk["error"])
            if chunk.get("usage"):
                usage = dict(chunk["usage"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                think = delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    content.append(str(piece))
                if think:
                    reasoning.append(str(think))
                if piece or think:
                    now = time.monotonic()
                    if first_delta_at is None:
                        first_delta_at = now
                    last_delta_at = now
                    delta_count += 1
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
    ended = time.monotonic()

    text = "".join(content)
    reasoning_sha256 = hashlib.sha256(
        "".join(reasoning).encode("utf-8")
    ).hexdigest()
    completion_tokens = int(usage.get("completion_tokens") or 0) or delta_count
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    ttft = (first_delta_at - started) if first_delta_at is not None else None
    decode_s = (
        (last_delta_at - first_delta_at)
        if (first_delta_at is not None and last_delta_at is not None)
        else None
    )
    # A request is successful only if it produced tokens AND the server did
    # not report an in-band error. Zero tokens with finish_reason "error" is a
    # failure no matter what the HTTP status said.
    failure: str | None = None
    if server_error is not None:
        failure = (
            f"server error [{server_error.get('code')}] "
            f"{server_error.get('message')}"
        )
    elif finish_reason == "error":
        failure = "finish_reason=error with no error payload"
    elif completion_tokens <= 0:
        failure = f"no tokens generated (finish_reason={finish_reason})"

    return {
        "ok": failure is None,
        "error": failure,
        "first_delta_at": first_delta_at,
        "server_error": server_error,
        "wall_s": ended - started,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "delta_count": delta_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "reasoning_sha256": reasoning_sha256,
        "reasoning_chars": sum(len(part) for part in reasoning),
        "request_body": {k: v for k, v in body.items() if k != "messages"},
        "usage": usage,
        "server_stats": _collect_server_stats(trailing),
        # Engagement must be read from the response, never from the launch
        # env: a mode can silently stop engaging mid-run (mlx-serve reports
        # runtime_disabled=true once it gives up on MTP). Keep the raw blocks.
        "raw_engagement": _collect_raw_engagement(trailing),
    }


RAW_ENGAGEMENT_KEYS = ("mtplx_stats", "timings", "usage", "mtp", "spec")


def _collect_raw_engagement(chunks: Sequence[Any]) -> dict[str, Any]:
    """Verbatim copies of whatever engagement blocks the server returned."""

    out: dict[str, Any] = {}
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        for key in RAW_ENGAGEMENT_KEYS:
            value = chunk.get(key)
            if isinstance(value, Mapping) and value:
                out[key] = dict(value)
    return out


def _collect_server_stats(chunks: Sequence[Any]) -> dict[str, Any]:
    """Pull MTPLX's metrics envelope out of whatever chunk carries it."""

    prefill_s = _dig(chunks, SERVER_PREFILL_KEYS)
    if prefill_s is None:
        prefill_ms = _dig(chunks, SERVER_PREFILL_MS_KEYS)
        prefill_s = (float(prefill_ms) / 1000.0) if prefill_ms is not None else None
    decode_s = _dig(chunks, SERVER_DECODE_S_KEYS)
    if decode_s is None:
        decode_ms = _dig(chunks, SERVER_DECODE_MS_KEYS)
        decode_s = (float(decode_ms) / 1000.0) if decode_ms is not None else None
    return {
        "prompt_eval_time_s": prefill_s,
        "prefill_tok_s": _dig(chunks, SERVER_PREFILL_TOKS_KEYS),
        "ttft_s": _dig(chunks, SERVER_TTFT_KEYS),
        "decode_elapsed_s": decode_s,
        "decode_tok_s": _dig(chunks, SERVER_DECODE_TOKS_KEYS),
        "new_prefill_tokens": _dig(chunks, SERVER_NEW_PREFILL_KEYS),
        "cached_tokens": _dig(chunks, SERVER_CACHED_KEYS),
        "peak_memory_bytes": _dig(chunks, SERVER_MEMORY_KEYS),
    }


def _mtp_accept_rate(raw: Mapping[str, Any]) -> float | None:
    """accepted / drafted, from whichever engagement block the server sent."""

    stats = raw.get("mtplx_stats") or {}
    accepted = stats.get("accepted_drafts")
    drafted = stats.get("drafted_tokens")
    if accepted is None or not drafted:
        timings = raw.get("timings") or {}
        accepted = timings.get("accepts")
        drafted = timings.get("drafted")
    try:
        drafted_f = float(drafted)  # type: ignore[arg-type]
        if drafted_f <= 0:
            return None
        return float(accepted) / drafted_f  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


#: Characters kept at each end of a completion in the receipt. Bounded on
#: purpose: two excerpts diagnose a divergence, a full transcript per cell
#: would multiply every receipt by the generated length for a field that is
#: read only when two arms disagree.
TEXT_EXCERPT_CHARS = 240


def build_record(
    *,
    server: str,
    cell: str,
    target_tokens: int,
    seed: int | None,
    repeat: int,
    call: Mapping[str, Any],
    rss_bytes: int | None,
    thermal: Mapping[str, Any] | None,
    prompt_sha256: str,
    prefer_server_timings: bool,
) -> dict[str, Any]:
    """Fold one call plus its side-channels into the receipt row.

    Prefill time has two possible sources and the receipt names the one used:
    the server's own ``prompt_eval_time_s`` when it reports one, otherwise
    TTFT, which on a cold prefill is prefill plus one decode step.
    """

    stats = dict(call.get("server_stats") or {})
    server_prefill = stats.get("prompt_eval_time_s") if prefer_server_timings else None
    if server_prefill:
        prefill_s: float | None = float(server_prefill)
        prefill_source = "server:prompt_eval_time_s"
    elif call.get("ttft_s") is not None:
        prefill_s = float(call["ttft_s"])
        prefill_source = "client:ttft"
    else:
        prefill_s = None
        prefill_source = "none"

    new_prefill = stats.get("new_prefill_tokens")
    if new_prefill is None:
        new_prefill = call.get("prompt_tokens") or target_tokens

    server_memory = stats.get("peak_memory_bytes")
    if server_memory:
        peak_bytes: int | None = int(server_memory)
        memory_source = "server:peak_memory_bytes"
    elif rss_bytes:
        peak_bytes = int(rss_bytes)
        memory_source = "client:ps_rss"
    else:
        peak_bytes = None
        memory_source = "none"

    decode_s = call.get("decode_s")
    return {
        "server": server,
        "cell": cell,
        "target_tokens": int(target_tokens),
        "seed": seed,
        "repeat": int(repeat),
        "ok": bool(call.get("ok")),
        "prompt_tokens": call.get("prompt_tokens"),
        "new_prefill_tokens": int(new_prefill or 0),
        "completion_tokens": call.get("completion_tokens"),
        "finish_reason": call.get("finish_reason"),
        "prefill_time_s": prefill_s,
        "prefill_time_source": prefill_source,
        "prefill_tok_s": prefill_tok_s(int(new_prefill or 0), prefill_s or 0.0),
        "ttft_s": call.get("ttft_s"),
        "decode_s": decode_s,
        "decode_tok_s": decode_tok_s(
            int(call.get("completion_tokens") or 0), float(decode_s or 0.0)
        ),
        "wall_s": call.get("wall_s"),
        "peak_memory_bytes": peak_bytes,
        "peak_memory_gb": (peak_bytes / 1e9) if peak_bytes else None,
        "peak_memory_source": memory_source,
        "client_rss_bytes": rss_bytes,
        "server_peak_memory_bytes": server_memory,
        "text_sha256": call.get("text_sha256"),
        # A bounded window on the completion itself. `text_sha256` answers
        # "did two arms produce the same text"; when the answer is no, a sha
        # cannot say HOW they diverged, and the full text would put ~4 KB per
        # cell into every receipt. Head and tail are enough to see whether two
        # arms started differing immediately or only near the end, which is
        # the difference between a different code path and a late sampling
        # divergence. Never used for equality -- that stays the sha.
        "text_chars": len(str(call.get("text") or "")),
        "text_head": str(call.get("text") or "")[:TEXT_EXCERPT_CHARS],
        "text_tail": str(call.get("text") or "")[-TEXT_EXCERPT_CHARS:],
        "reasoning_chars": call.get("reasoning_chars"),
        "reasoning_sha256": call.get("reasoning_sha256"),
        "thermal_gate": dict(thermal) if thermal else None,
        "prompt_sha256": prompt_sha256,
        "request_body": call.get("request_body"),
        "server_stats": stats,
        "usage": call.get("usage"),
        "server_error": call.get("server_error"),
        "error": call.get("error"),
        # Computed in stream_chat; it was being dropped here, which is why the
        # <=16K MTPLX rows carry no engagement evidence at all.
        "raw_engagement": call.get("raw_engagement") or {},
        "mtp_accept_rate": _mtp_accept_rate(call.get("raw_engagement") or {}),
    }


# ---------------------------------------------------------------------------
# Prompt construction (CPU only; run BEFORE the guard window opens)
# ---------------------------------------------------------------------------

MODEL = Path(
    "/Users/davidtai/.mtplx/models/"
    "Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
)
MODEL_ID = "mtplx-flash-next-optimized-speed"
# In-repo, hash-pinned fixtures (copied byte-identically from the
# qwen4-queue-first-draft worktree; context sha still checked against
# EXPECTED_CONTEXT_SHA256 below). Falls back to the original location if a
# consumer runs this file from a tree that does not carry the fixtures.
FIXTURES = ROOT / "mtplx" / "benchmarks" / "prompts"
if not (FIXTURES / "qwen38_generation_context.py").exists():
    FIXTURES = Path(
        "/Users/davidtai/projects/OpenSourceWTF/.worktrees/"
        "qwen4-queue-first-draft/mtplx/benchmarks/prompts"
    )
EXPECTED_CONTEXT_SHA256 = (
    "c8ae2b1790c0300aa7c1421b55e7cd5d43c93461f7fba5d3a732fd34e156b4c4"
)
PROMPT_TOKEN_TOLERANCE = 8


def rotate_context(context: str, seed: int) -> str:
    """Rotate the pinned coding context by ``seed`` lines.

    Deterministic and content-preserving: every seed sees the same bytes in a
    different order, so the seeds are genuinely distinct prompts without
    introducing material the fixture hash does not cover.
    """

    lines = context.splitlines()
    if not lines:
        raise ValueError("coding context is empty")
    offset = int(seed) % len(lines)
    return "\n".join(lines[offset:] + lines[:offset])


def load_fixture_context() -> str:
    text = (FIXTURES / "qwen38_generation_context.py").read_text()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != EXPECTED_CONTEXT_SHA256:
        raise RuntimeError(
            f"coding context fixture drifted: {digest} != {EXPECTED_CONTEXT_SHA256}"
        )
    return text


def load_fixture_instruction() -> str:
    return json.loads(
        (FIXTURES / "qwen38_naturalistic_generation_patch.jsonl")
        .read_text()
        .splitlines()[0]
    )["prompt"]


def templated_ids(result: Any) -> list[int]:
    """Normalise ``apply_chat_template(tokenize=True)`` across transformers.

    transformers 5.x returns a ``BatchEncoding`` whose ``len()`` is the number
    of KEYS, not tokens; 4.x returns a flat list; both can return a batch of
    one. Normalise all three to the flat id list.
    """

    if isinstance(result, Mapping):
        result = result["input_ids"]
    values = list(result)
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise RuntimeError(f"chat template returned {len(values)} sequences")
        values = list(values[0])
    return [int(value) for value in values]


def make_counter(
    tokenizer: Any, *, enable_thinking: bool, reasoning_effort: str
) -> Callable[[str], int]:
    """Count the FULL templated prompt -- what the prefill chunker sees.

    The template kwargs must match the request that will be sent: thinking on
    with ``xhigh`` renders a different prefix than thinking off, and sizing a
    prompt under the wrong one misses the target by that difference.
    """

    def count(text: str) -> int:
        return len(
            templated_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    reasoning_effort=reasoning_effort,
                )
            )
        )

    return count


# ---------------------------------------------------------------------------
# Model-family switch (W49): the Qwen3.8 PRs template with a chat template that
# consumes ``enable_thinking``/``reasoning_effort``; DeepSeek-V4.1-Flash ships
# NO chat template (tokenizer.chat_template is None), so on the served path
# ``mtplx.server.openai._encode_messages_uncached`` falls through every
# template branch to the plain role-prefixed render at openai.py:14537
# (``"user: " + content + "\nassistant:"``) encoded with
# ``add_special_tokens=False``. enable_thinking/reasoning_effort are inert for
# it (verified: identical ids with thinking on and off), and the tokenizer's
# ByteLevel post-processor with add_bos_token=False prepends NO leading BOS
# id 0 -- neither the chat path (add_special_tokens=False) nor the completions
# path (_encode_plain_text, add_special_tokens=True) emits it, and no serving
# code prepends it. So the DSV4.1 counter/ids replicate that plain render and
# the request omits the two kwargs. Keyed on --model-family or the served
# model id. Qwen behaviour is untouched (family defaults to qwen38).
QWEN38_FAMILY = "qwen38"
DEEPSEEK_V41_FAMILY = "deepseek-v41"
MODEL_FAMILIES: tuple[str, ...] = (QWEN38_FAMILY, DEEPSEEK_V41_FAMILY)

# W52: DSV4.1 DOES have a chat template + a thinking mode. The artifact now
# carries chat_template.jinja (a byte-for-byte port of the official reference
# encoder) and the served path adds BOS id 0, so the counter/ids template the
# real chat prompt instead of the old plain "user:/assistant:" render. The
# OFFICIAL default is thinking OFF: the reference generate.py defaults
# --thinking-mode "chat", and the reference encoding-test harness renders with
# thinking_mode="chat". So the served cells default to enable_thinking=False;
# reasoning_effort is inert in chat mode (only rendered in thinking mode) and is
# omitted there. Override with --dsv41-enable-thinking / --dsv41-reasoning-effort
# (or DSV41_ENABLE_THINKING / DSV41_REASONING_EFFORT). DeepSeek's own instruct
# evals use thinking + reasoning_effort=100; flip the knob to mirror them.
DEEPSEEK_V41_DEFAULT_ENABLE_THINKING = False
DEEPSEEK_V41_DEFAULT_REASONING_EFFORT: str | None = None


def _parse_bool_flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "on", "true", "yes"}


def deepseek_v41_thinking_settings(
    args: argparse.Namespace | None,
) -> tuple[bool, str | None]:
    """Resolve (enable_thinking, reasoning_effort) for the DSV4.1 sweep cell.

    Precedence: explicit CLI flag > DSV41_* env > official default (thinking
    OFF). reasoning_effort is returned only when thinking is on (it is inert in
    chat mode and omitted from the request body there).
    """
    et: Any = getattr(args, "dsv41_enable_thinking", None) if args is not None else None
    if et is None:
        env = os.environ.get("DSV41_ENABLE_THINKING")
        et = _parse_bool_flag(env) if env is not None else DEEPSEEK_V41_DEFAULT_ENABLE_THINKING
    et = bool(et)
    eff: str | None = getattr(args, "dsv41_reasoning_effort", None) if args is not None else None
    if eff is None:
        eff = os.environ.get("DSV41_REASONING_EFFORT") or DEEPSEEK_V41_DEFAULT_REASONING_EFFORT
    return et, (str(eff) if (et and eff) else None)


def resolve_model_family(explicit: str | None, model_id: str | None) -> str:
    """--model-family wins; else detect from the served/model id substring."""

    if explicit:
        value = str(explicit).strip().lower()
        if value not in MODEL_FAMILIES:
            raise ValueError(
                f"--model-family must be one of {MODEL_FAMILIES}, got {explicit!r}"
            )
        return value
    mid = str(model_id or "").lower()
    if "deepseek" in mid or "dsv4" in mid or "v4.1" in mid or "-v41" in mid:
        return DEEPSEEK_V41_FAMILY
    return QWEN38_FAMILY


def _cell_thinking_settings(
    family: str,
    is_vanity: bool,
    *,
    reasoning_effort: str | None,
    dsv41_enable_thinking: bool,
    dsv41_reasoning_effort: str | None,
) -> tuple[bool, str | None]:
    """(enable_thinking, reasoning_effort) for one family+cell.

    Both families run the vanity cell thinking-OFF (a short sanity prompt).
    Qwen sweep = thinking ON + reasoning_effort; DSV4.1 sweep = the resolved
    DSV4.1 setting (default thinking OFF). reasoning_effort is dropped when
    thinking is off (inert; keeps the wire body minimal)."""

    if family == DEEPSEEK_V41_FAMILY:
        if is_vanity:
            return False, None
        return dsv41_enable_thinking, (dsv41_reasoning_effort if dsv41_enable_thinking else None)
    if is_vanity:
        return VANITY_ENABLE_THINKING, VANITY_REASONING_EFFORT
    return True, reasoning_effort


def templated_prompt_ids(
    tokenizer: Any,
    text: str,
    *,
    enable_thinking: bool,
    reasoning_effort: str | None,
) -> list[int]:
    """Ids for one user-turn prompt via the tokenizer's chat template.

    For DSV4.1 the artifact's chat_template.jinja emits a leading BOS id 0 (and
    the served path also prepends it), so these ids match the server
    byte-for-byte WITH BOS. reasoning_effort is passed only when thinking is on."""

    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if enable_thinking and reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return templated_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": text}], **kwargs)
    )


def deepseek_v41_prompt_ids(
    tokenizer: Any,
    text: str,
    *,
    enable_thinking: bool = DEEPSEEK_V41_DEFAULT_ENABLE_THINKING,
    reasoning_effort: str | None = None,
) -> list[int]:
    """The exact ids the DSV4.1 server prefills for one user turn: the chat
    template render (BOS id 0 first). Kept as a named helper for the tests."""

    return templated_prompt_ids(
        tokenizer, text, enable_thinking=enable_thinking, reasoning_effort=reasoning_effort
    )


def make_deepseek_v41_counter(
    tokenizer: Any,
    *,
    enable_thinking: bool = DEEPSEEK_V41_DEFAULT_ENABLE_THINKING,
    reasoning_effort: str | None = None,
) -> Callable[[str], int]:
    def count(text: str) -> int:
        return len(
            deepseek_v41_prompt_ids(
                tokenizer,
                text,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        )

    return count


def server_prompt_ids(
    tokenizer: Any,
    family: str,
    entry: Mapping[str, Any],
    *,
    reasoning_effort: str | None,
    dsv41_enable_thinking: bool = DEEPSEEK_V41_DEFAULT_ENABLE_THINKING,
    dsv41_reasoning_effort: str | None = None,
) -> list[int]:
    """Exactly the token ids the server will prefill for one built prompt.

    Both families now go through the tokenizer's chat template with the SAME
    kwargs the request carries (per :func:`_cell_thinking_settings`). DSV4.1's
    template emits BOS id 0; Qwen's does not."""

    text = str(entry.get("text") or "")
    is_vanity = entry.get("cell") == "vanity"
    enable_thinking, effort = _cell_thinking_settings(
        family,
        is_vanity,
        reasoning_effort=reasoning_effort,
        dsv41_enable_thinking=dsv41_enable_thinking,
        dsv41_reasoning_effort=dsv41_reasoning_effort,
    )
    return templated_prompt_ids(
        tokenizer, text, enable_thinking=enable_thinking, reasoning_effort=effort
    )


def template_settings_for_family(
    family: str,
    *,
    reasoning_effort: str | None = "xhigh",
    dsv41_enable_thinking: bool = DEEPSEEK_V41_DEFAULT_ENABLE_THINKING,
    dsv41_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """What the receipt records about how prompts were templated for a family."""

    if family == DEEPSEEK_V41_FAMILY:
        effort = dsv41_reasoning_effort if dsv41_enable_thinking else None
        return {
            "model_family": family,
            "chat_template_source": (
                "artifact chat_template.jinja (W52 port of the official reference "
                "encoder) + served code fallback "
                "(mtplx.chat_encoding.encode_deepseek_v41_messages)"
            ),
            "render": (
                "DeepSeek-V4.1 reference chat format: BOS + [<｜System｜>effort/"
                "system] + <｜User｜>content + <｜Assistant｜> + <think>|</think>"
            ),
            "render_cite": (
                "mtplx/templates/deepseek_v41/chat_template.jinja; "
                "DeepSeek-V4.1-Flash-src/encoding/encoding.py::encode_messages"
            ),
            "add_special_tokens": False,
            "enable_thinking": dsv41_enable_thinking,
            "reasoning_effort": effort,
            "thinking_mode": bool(dsv41_enable_thinking),
            "bos_id_prepended": True,
            "bos_token_id": 0,
            "note": (
                "DSV4.1 now templates the real chat prompt (BOS id 0 first). The "
                "OFFICIAL default is thinking OFF (reference generate.py "
                "--thinking-mode 'chat'; the encoding-test harness renders "
                "thinking_mode='chat'); reasoning_effort is inert in chat mode "
                "and omitted there. DeepSeek's instruct evals use thinking + "
                "reasoning_effort=100 -- set --dsv41-enable-thinking "
                "(DSV41_ENABLE_THINKING) + --dsv41-reasoning-effort to mirror them. "
                "The exported ids match the server byte-for-byte WITH BOS."
            ),
        }
    return {
        "model_family": family,
        "chat_template_source": "tokenizer (apply_chat_template)",
        "sweep": {"enable_thinking": True, "reasoning_effort": reasoning_effort},
        "vanity": {
            "enable_thinking": VANITY_ENABLE_THINKING,
            "reasoning_effort": VANITY_REASONING_EFFORT,
        },
    }


def build_sized_prompt(
    *,
    context: str,
    instruction: str,
    seed: int,
    target_tokens: int,
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    count_templated: Callable[[str], int],
    tolerance: int = PROMPT_TOKEN_TOLERANCE,
    max_rounds: int = 8,
) -> dict[str, Any]:
    """A rotated slice of the pinned context sized to an exact token target.

    Corrects the context budget by the MEASURED error rather than doing
    sentinel arithmetic on the template, so it stays correct if the template
    changes. Keeps the BEST attempt rather than the last: a one-token budget
    step can move the templated count by two, so the sequence can oscillate
    around the target without ever landing on it.
    """

    if int(target_tokens) <= 0:
        raise ValueError("target_tokens must be positive")
    body = str(instruction).strip() + SWEEP_INSTRUCTION_SUFFIX
    rotated = rotate_context(context, seed)
    context_ids = list(encode(rotated.rstrip() + "\n"))
    if not context_ids:
        raise ValueError("coding context encoded to zero tokens")

    def assemble(budget: int) -> str:
        take = max(1, int(budget))
        repeats = (take + len(context_ids) - 1) // len(context_ids)
        ids = (context_ids * repeats)[:take]
        return decode(ids).rstrip() + "\n\n" + body

    budget = max(1, int(target_tokens) - len(list(encode(body))))
    history: list[dict[str, int]] = []
    best: tuple[int, str, int, int] | None = None
    text = assemble(budget)
    for _ in range(int(max_rounds)):
        measured = int(count_templated(text))
        history.append({"budget": int(budget), "templated_tokens": measured})
        error = int(target_tokens) - measured
        if best is None or abs(error) < best[0]:
            best = (abs(error), text, measured, int(budget))
        if error == 0:
            break
        budget = max(1, budget + error)
        text = assemble(budget)
    assert best is not None
    drift, text, measured, budget = best
    if drift > int(tolerance):
        raise RuntimeError(
            f"prompt sizing did not converge for seed {seed} target "
            f"{target_tokens} (best miss {drift} > {tolerance}): {history}"
        )
    return {
        "seed": int(seed),
        "target_tokens": int(target_tokens),
        "templated_tokens": measured,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
        "rounds": history,
    }


def build_prompt_cache(
    *,
    contexts: Sequence[int],
    seeds: Sequence[int],
    path: Path,
    tokenizer_path: str | Path | None = None,
    model_id: str | None = None,
    model_family: str | None = None,
    reasoning_effort: str | None = "xhigh",
    dsv41_enable_thinking: bool = DEEPSEEK_V41_DEFAULT_ENABLE_THINKING,
    dsv41_reasoning_effort: str | None = None,
    ids_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build every prompt the battery will send and cache it to JSON.

    Run this OUTSIDE the guarded window. Loading the tokenizer and running the
    sizing loop is seconds of CPU and disk, and doing it between GPU arms is
    exactly the contamination that made W39's page-cache prewarm uneven.
    """

    from transformers import AutoTokenizer

    family = resolve_model_family(model_family, model_id)
    tok_path = str(tokenizer_path or MODEL)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=False)
    encode = lambda text: list(tokenizer.encode(text))  # noqa: E731
    decode = lambda ids: str(tokenizer.decode(list(ids)))  # noqa: E731

    context = load_fixture_context()
    instruction = load_fixture_instruction()

    if family == DEEPSEEK_V41_FAMILY:
        # Chat template (artifact sidecar): count the real templated prompt,
        # BOS included. Vanity runs thinking-off like Qwen; sweep uses the
        # resolved DSV4.1 setting (default thinking-off).
        sweep_count = make_deepseek_v41_counter(
            tokenizer,
            enable_thinking=dsv41_enable_thinking,
            reasoning_effort=(dsv41_reasoning_effort if dsv41_enable_thinking else None),
        )
        vanity_count = make_deepseek_v41_counter(
            tokenizer, enable_thinking=False, reasoning_effort=None
        )
    else:
        sweep_count = make_counter(
            tokenizer, enable_thinking=True, reasoning_effort=reasoning_effort
        )
        vanity_count = make_counter(
            tokenizer, enable_thinking=False, reasoning_effort="low"
        )

    entries: list[dict[str, Any]] = [
        {
            "cell": "vanity",
            "target_tokens": 0,
            "seed": None,
            "text": VANITY_PROMPT,
            "text_sha256": VANITY_PROMPT_SHA256,
            "templated_tokens": vanity_count(VANITY_PROMPT),
        }
    ]
    for target in contexts:
        for seed in seeds:
            built = build_sized_prompt(
                context=context,
                instruction=instruction,
                seed=seed,
                target_tokens=int(target),
                encode=encode,
                decode=decode,
                count_templated=sweep_count,
            )
            built["cell"] = "sweep"
            entries.append(built)

    payload = {
        "schema": "mtplx-server-cell-prompts-v1",
        "model": tok_path,
        "model_family": family,
        "served_model_id": model_id,
        "template_settings": template_settings_for_family(
            family,
            reasoning_effort=reasoning_effort,
            dsv41_enable_thinking=dsv41_enable_thinking,
            dsv41_reasoning_effort=dsv41_reasoning_effort,
        ),
        "context_sha256": EXPECTED_CONTEXT_SHA256,
        "vanity_prompt_sha256": VANITY_PROMPT_SHA256,
        "contexts": [int(c) for c in contexts],
        "seeds": [int(s) for s in seeds],
        "prompts": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    if ids_path is not None:
        write_prompt_ids(
            entries,
            tokenizer=tokenizer,
            family=family,
            reasoning_effort=reasoning_effort,
            dsv41_enable_thinking=dsv41_enable_thinking,
            dsv41_reasoning_effort=dsv41_reasoning_effort,
            path=Path(ids_path),
            tokenizer_path=tok_path,
            model_id=model_id,
        )
    return payload


def write_prompt_ids(
    entries: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    family: str,
    reasoning_effort: str | None,
    path: Path,
    tokenizer_path: str,
    model_id: str | None,
    dsv41_enable_thinking: bool = DEEPSEEK_V41_DEFAULT_ENABLE_THINKING,
    dsv41_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Export every built prompt as the EXACT token-id list the server will
    prefill (per cell, per seed), so an in-process arm can be handed the same
    ids the served cell used (``--prompt-ids-file`` on the A/B scripts)."""

    id_entries: list[dict[str, Any]] = []
    for entry in entries:
        ids = server_prompt_ids(
            tokenizer,
            family,
            entry,
            reasoning_effort=reasoning_effort,
            dsv41_enable_thinking=dsv41_enable_thinking,
            dsv41_reasoning_effort=dsv41_reasoning_effort,
        )
        id_entries.append(
            {
                "cell": entry.get("cell"),
                "target_tokens": int(entry.get("target_tokens") or 0),
                "seed": entry.get("seed"),
                "text_sha256": entry.get("text_sha256"),
                "templated_tokens": entry.get("templated_tokens"),
                "input_tokens": len(ids),
                "token_ids": ids,
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(ids).encode("utf-8")
                ).hexdigest(),
                "bos_id_prepended": ids[:1] == [0] if ids else False,
            }
        )
    payload = {
        "schema": "mtplx-server-cell-prompt-ids-v1",
        "model": str(tokenizer_path),
        "model_family": family,
        "served_model_id": model_id,
        "template_settings": template_settings_for_family(
            family,
            reasoning_effort=reasoning_effort,
            dsv41_enable_thinking=dsv41_enable_thinking,
            dsv41_reasoning_effort=dsv41_reasoning_effort,
        ),
        "context_sha256": EXPECTED_CONTEXT_SHA256,
        "prompts": id_entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return payload


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

#: The documented ceiling for this 128 GiB box. Raised from 96/90 GiB after
#: run 1: 100 GiB covers the predicted 256K peak (~93.1 GiB, from the memory
#: plan's own 24,576 B/token KV + 7,872 B/token aux over the measured 16K
#: peak) and still leaves 28 GiB to macOS. Higher is refused: a 112 GiB knob
#: leaving 16 GiB to the OS is the documented kernel-panic regime.
MEMORY_LIMIT_BYTES = 100 * 1024**3
WIRED_LIMIT_BYTES = 100 * 1024**3
MAX_SAFE_LIMIT_BYTES = 100 * 1024**3

#: The recommended branch stack. ``MTPLX_SESSION_BANK_MAX_BYTES`` is
#: deliberately absent: ``auto`` keeps the bank's yielding ceiling, and the 8G
#: pin is an agentic-serving recommendation, not a battery setting.
BRANCH_ENV: dict[str, str] = {
    "MTPLX_FABLE_HC_M4": "1",
    "MTPLX_FABLE_OPDIET": "1",
    "MTPLX_FABLE_COMPILED_DRAFT": "1",
    "MTPLX_FABLE_BLOCK_VERIFY": "1",
    "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD": "1",
    "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE": "2048",
    "MTPLX_PREFILL_CHUNK_SIZE": "4096",
    "MTPLX_QSA_PREFILL_COMPILE_ROWS": "4096",
    "MTPLX_GDN_BLOCKED_PREFILL": "1",
    # W57 is fixing the geometry guard properly. Until that merges this is
    # required: at 512/2560 the ladder's chunk width clamps to 256, collides
    # with the pinned COMPILE_ROWS=4096, and PrefillChunkGeometryError kills
    # BOTH warmup steps on every branch boot -- so the branch arm entered
    # every timed cell colder than upstream, which completed both.
    "MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH": "1",
    "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE": "1",
    "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL": "1",
    "MTPLX_QWEN4_M4_ROUTED_GLU": "1",
}

#: The restack the ABBA driver sets via FLAGS. CORRECTION: four of these are
#: NOT missing at server defaults. openai.py:846-869 auto-stamps
#: MTPLX_COMPILED_VERIFY, MTPLX_QWEN4_FIXED_M4_VERIFY, MTPLX_QWEN4_M4_STAGE3
#: and MTPLX_QSA_M4_FUSED_KV_GATHER via setdefault whenever
#: _served_model_is_qwen4_fixed_m4() holds -- and it holds for the served
#: pack (is_qwen4_fixed_verify_config(config) == True, verified). Upstream
#: v2.10.2 has no such block at all. Listing them here is harmless because an
#: explicit operator export wins (overrides.pop) with the same value.
#:
#: DERIVED, not hand-written: this is
#: abba_driver.build_family_overrides(abba_window.CONTROL_FLAGS) minus every
#: key the server already owns, plus the FR-Spec pair (the env equivalent of
#: the driver's --full-frspec, which the driver installs programmatically
#: rather than through the env). Regenerate with
#: tests/fixtures/abba_control_family_overrides.json; a test compares the two
#: so the harness and the measured control cannot drift apart.
#:
#: MTPLX_QWEN4_RELAXED_DRAFT_TIES is deliberately ABSENT: --compiled-mtp-prepare
#: does not imply it and it is not in CONTROL_FLAGS, so exporting it would make
#: the full-stack arm EXCEED the arm it is supposed to reproduce.
#:
#: The auto-stamped keys are DELIBERATELY ABSENT from this block so both
#: branch arms reach them by the server's own setdefault path. Exporting them
#: is a no-op only by coincidence -- turbo happens to set COMPILED_VERIFY="1"
#: too -- and openai.py:870-880 is a LIVE precedent for breaking exactly that
#: coincidence: it pins MTPLX_NAX_VERIFY="0" against turbo's "1" for this
#: family until it "earns a family receipt". If anyone does the same to a key
#: we export, the exported arm would fall through overrides.pop to whatever
#: turbo then says while the defaults arm keeps the setdefault value, and the
#: arms would diverge silently on a key no row label mentions.
FULL_STACK_ENV: dict[str, str] = {
    "MTPLX_QSA_GATHER": "1",
    "MTPLX_COMPILED_GDN": "1",
    "MTPLX_AR_PIPELINE": "1",
    "MTPLX_FAMILY_CAPTURE_COMMIT": "1",
    "MTPLX_FUSED_HC_V3": "1",
    "MTPLX_FUSED_GDN_INPROJ": "1",
    "MTPLX_FUSED_GATE_UP": "1",
    "MTPLX_FUSED_GDN_CONVNORM": "1",
    "MTPLX_FUSED_GDN_STEP": "1",
    "MTPLX_FUSED_CONVNORM_VERIFY": "1",
    "MTPLX_QWEN4_COMPILED_MTP_PREPARE": "1",
    "MTPLX_FRSPEC_DRAFT": "1",
    "MTPLX_FRSPEC_VOCAB": "builtin:qwen38-code-64k",
    # The one profile conflict left to resolve by hand. BATCH_TARGET_ARRAYS
    # and LAZY_TARGET_DISTRIBUTIONS are NOT here: openai.py:840-845 already
    # setdefaults them to exactly these values (1 and 0) on this family.
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "0",       # turbo sets 1
}

#: Identical Metal caps on both servers. Upstream's own default wired cap is
#: ~83.34 GiB, so without this the two servers would be scored under different
#: eviction pressure and the comparison would be meaningless.
COMMON_ENV: dict[str, str] = {
    "MTPLX_MEMORY_LIMIT_BYTES": str(MEMORY_LIMIT_BYTES),
    "MTPLX_WIRED_LIMIT_BYTES": str(WIRED_LIMIT_BYTES),
    "MTPLX_SESSION_BANK_MAX_BYTES": "auto",
}

#: The ``--server-defaults`` arm exports ONLY the memory caps. Every other key,
#: including the session-bank budget, is the server's to decide: the served
#: tree now defaults ``MTPLX_SESSION_BANK_MAX_BYTES`` to ``8G`` for Flash-Next
#: packs, so exporting ``auto`` here would read as an operator override and
#: turn one retained key off on the very arm that asks what the defaults are.
SERVER_DEFAULTS_ENV: dict[str, str] = {
    "MTPLX_MEMORY_LIMIT_BYTES": str(MEMORY_LIMIT_BYTES),
    "MTPLX_WIRED_LIMIT_BYTES": str(WIRED_LIMIT_BYTES),
}


# ---------------------------------------------------------------------------
# Fable flag files: the retained decode/prefill set, driven from disk
# ---------------------------------------------------------------------------

FABLE_FLAG_PREFIX = "MTPLX_FABLE_"

#: The tree the BRANCH server actually runs from. Both the profile allowlist
#: and the MTPLX_FABLE_* known-flag set are read out of it as TEXT -- never
#: imported, because importing ``mtplx`` pulls mlx and this module must stay
#: GPU-free so the suite can run inside a guarded window.
MTPLX_SOURCE_ROOT = Path(
    "/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps"
)

#: W61's typed registry (``mtplx/full_stack_env.py``). Its prefixes are
#: MTPLX_QWEN4_/MTPLX_QSA_/MTPLX_FRSPEC_, so it does NOT enumerate the
#: MTPLX_FABLE_* space; it is unioned in when present so that a later
#: registry entry is honoured, but the FABLE known-set comes from the source
#: scan below either way. DEPENDENCY: as of 2026-09-02 W61 lives on
#: ``worker/w61-restack-profile`` and is NOT merged into
#: ``experiments/fable-qwen38-80tps`` (14d18189), so this file is absent and
#: the scan is the only source.
FULL_STACK_ENV_PY_NAME = "mtplx/full_stack_env.py"

#: Keys that live in ``BRANCH_ENV`` but are NOT part of the retained tuning
#: set: they are boot requirements. A flags file that simply does not mention
#: them must not switch them off, so they stay in the always-applied base.
#: Without the compile-rows mismatch allowance the branch server raises
#: PrefillChunkGeometryError on BOTH warmup ladder steps and enters every
#: timed cell colder than the control. A file MAY still override one.
BRANCH_BOOT_FABLE_KEYS: tuple[str, ...] = (
    "MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH",
)

#: The hard-coded retained set. Used ONLY when no --fable-flags-file is given.
DEFAULT_FABLE_FLAGS: dict[str, str] = {
    key: value
    for key, value in BRANCH_ENV.items()
    if key.startswith(FABLE_FLAG_PREFIX) and key not in BRANCH_BOOT_FABLE_KEYS
}

#: Everything in BRANCH_ENV that a flags file does not replace: the family and
#: runtime keys plus the boot requirements above. Always applied to a branch
#: arm. ``BRANCH_BASE_ENV | DEFAULT_FABLE_FLAGS == BRANCH_ENV`` by
#: construction, and a test pins it, so the no-file path is byte-identical to
#: what the battery ran before this option existed.
BRANCH_BASE_ENV: dict[str, str] = {
    key: value for key, value in BRANCH_ENV.items() if key not in DEFAULT_FABLE_FLAGS
}

#: Where a resolved key came from, when it did not come from a file.
DEFAULT_FLAG_SOURCE = "harness:DEFAULT_FABLE_FLAGS"


class FlagFileError(ValueError):
    """A flags file the harness refuses to run with."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


_FLAG_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_fable_flags_file(path: str | Path) -> dict[str, str]:
    """``KEY=VALUE`` per line -> an ordered dict.

    Blank lines and lines whose first non-space character is ``#`` are
    ignored. There is NO inline comment syntax: everything after the first
    ``=`` is the value, stripped of surrounding whitespace only, because real
    values in this set are comma lists (``qsa_rope,qsa_rope_idx``) and a
    ``#``-stripping rule would silently truncate one some day.

    A duplicate key inside ONE file is refused rather than resolved: it is a
    hand-edit mistake, and picking a winner would hide it.
    """

    target = Path(path)
    try:
        text = target.read_text()
    except OSError as error:
        raise FlagFileError(f"{target}: cannot read: {error}") from error
    entries: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep:
            raise FlagFileError(
                f"{target}:{number}: not KEY=VALUE: {raw!r}"
            )
        if not _FLAG_KEY_RE.match(key):
            raise FlagFileError(f"{target}:{number}: not a valid env key: {key!r}")
        if key in entries:
            raise FlagFileError(
                f"{target}:{number}: duplicate key {key} (already set on an "
                "earlier line of this file)"
            )
        entries[key] = value.strip()
    return entries


def known_fable_flags(root: str | Path | None = None) -> dict[str, Any]:
    """Every ``MTPLX_FABLE_*`` name that appears in the served tree's source.

    A typo guard, not an authority: a name that only appears in a comment
    still counts as known. What it catches is the failure this harness has
    already had once -- most of these keys are read by a bare
    ``os.environ.get(name, default)``, so a misspelling does not raise, it
    silently runs a lesser stack.
    """

    base = Path(root) if root is not None else MTPLX_SOURCE_ROOT
    package = base / "mtplx"
    if not package.is_dir():
        return {
            "available": False,
            "root": str(base),
            "reason": f"no mtplx/ package under {base}",
            "keys": (),
            "known_count": 0,
            "scanned_files": 0,
            "full_stack_env_py": False,
        }
    pattern = re.compile(r"MTPLX_FABLE_[A-Z0-9_]+")
    keys: set[str] = set()
    scanned = 0
    for source in sorted(package.rglob("*.py")):
        try:
            keys.update(pattern.findall(source.read_text(errors="replace")))
        except OSError:
            continue
        scanned += 1
    registry_py = base / FULL_STACK_ENV_PY_NAME
    registry_present = registry_py.is_file()
    if registry_present:
        # W61's registry, when it lands: union its declared names so a key it
        # introduces is accepted even before a reader site spells it out.
        try:
            keys.update(pattern.findall(registry_py.read_text(errors="replace")))
        except OSError:
            registry_present = False
    return {
        "available": True,
        "root": str(base),
        "source": f"{package}/**/*.py",
        "keys": tuple(sorted(keys)),
        "known_count": len(keys),
        "scanned_files": scanned,
        "full_stack_env_py": registry_present,
    }


def resolve_fable_flags(
    paths: Sequence[str | Path] | None,
    *,
    stack: str = "branch",
    registry_root: str | Path | None = None,
    default: Mapping[str, str] | None = None,
    server_defaults: bool = False,
) -> dict[str, Any]:
    """Resolve the branch arm's retained flag set from files, or the default.

    Precedence inside the set is file order: a key set by a later
    ``--fable-flags-file`` beats an earlier one, and the shadowed entry is
    recorded rather than dropped silently. The whole set then merges OVER the
    derived family env, and an explicit ``--env`` still beats all of it.

    Refuses, before anything boots:

    * a key the SERVER owns (:data:`AUTO_ARMED_KEYS`) -- exporting one of
      those makes the two branch arms reach it by different precedence paths;
    * an ``MTPLX_FABLE_*`` name the served tree never reads -- a typo there
      does not raise at the read site, it silently no-ops;
    * anything outside the ``MTPLX_`` namespace -- a flags file is not a place
      to set ``PATH``.
    """

    default_map = dict(DEFAULT_FABLE_FLAGS if default is None else default)
    registry = known_fable_flags(registry_root)
    # A --server-defaults arm injects nothing, so it must RESOLVE nothing. The
    # env builder already ignores this set, but a printed block claiming six
    # resolved keys that the process never got is exactly the "receipt claims
    # a set the process did not get" defect the file-driven set exists to
    # prevent -- so the two agree here rather than only in the env dump.
    applied = stack == "branch" and not server_defaults
    files: list[dict[str, Any]] = []
    resolved: dict[str, str] = {}
    sources: dict[str, str] = {}
    shadowed: list[dict[str, str]] = []
    problems: list[str] = []

    for raw_path in list(paths or []):
        path = Path(raw_path)
        if not path.is_file():
            raise FlagFileError(f"{path}: no such flags file")
        entries = parse_fable_flags_file(path)
        text_path = str(path.resolve())
        for key, value in entries.items():
            if key in AUTO_ARMED_KEYS:
                problems.append(
                    f"{path}: {key} is SERVER-OWNED (openai.py stamps it for "
                    "this family); exporting it makes the two branch arms "
                    "reach it by different precedence paths"
                )
                continue
            if not key.startswith("MTPLX_"):
                problems.append(
                    f"{path}: {key} is outside the MTPLX_ namespace; a flags "
                    "file sets engine flags, not process environment"
                )
                continue
            if key.startswith(FABLE_FLAG_PREFIX):
                if not registry["available"]:
                    problems.append(
                        f"{path}: cannot validate {key}: "
                        f"{registry.get('reason')}"
                    )
                    continue
                if key not in registry["keys"]:
                    near = _nearest_flag(key, registry["keys"])
                    hint = f"; did you mean {near}?" if near else ""
                    problems.append(
                        f"{path}: {key} is not read anywhere under "
                        f"{registry['root']}/mtplx{hint}"
                    )
                    continue
            if key in resolved:
                shadowed.append(
                    {
                        "key": key,
                        "value": resolved[key],
                        "path": sources[key],
                        "shadowed_by": text_path,
                    }
                )
            resolved[key] = value
            sources[key] = text_path
        files.append(
            {
                "path": text_path,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "keys": sorted(entries),
            }
        )

    default_used = applied and not files
    if default_used:
        resolved = dict(default_map)
        sources = {key: DEFAULT_FLAG_SOURCE for key in resolved}

    if problems:
        raise FlagFileError(
            "refusing the flags file(s):\n  - " + "\n  - ".join(problems)
        )

    if not applied:
        resolved, sources = {}, {}

    return {
        "applied": applied,
        "stack": stack,
        "server_defaults": bool(server_defaults),
        "default_used": default_used,
        "dropped_vs_default": (
            sorted(set(default_map) - set(resolved)) if applied else []
        ),
        "files": files,
        "resolved": dict(sorted(resolved.items())),
        "sources": dict(sorted(sources.items())),
        "shadowed": shadowed,
        "registry": {
            "root": registry["root"],
            "available": bool(registry["available"]),
            "known_count": int(registry["known_count"]),
            "scanned_files": int(registry["scanned_files"]),
            "full_stack_env_py": bool(registry["full_stack_env_py"]),
        },
    }


def _nearest_flag(name: str, candidates: Iterable[str]) -> str | None:
    """Cheapest useful hint: the known key with the longest shared prefix."""

    best: tuple[int, str] | None = None
    for candidate in candidates:
        shared = len(os.path.commonprefix([name, candidate]))
        if shared >= len(FABLE_FLAG_PREFIX) + 2 and (best is None or shared > best[0]):
            best = (shared, candidate)
    return best[1] if best else None


def render_fable_flags(resolution: Mapping[str, Any]) -> str:
    """The block that goes in the preflight report and the dry run.

    Stable and diffable: files in the order given, keys sorted.
    """

    lines: list[str] = []
    registry = resolution.get("registry") or {}
    lines.append(
        f"registry root={registry.get('root')} "
        f"known={registry.get('known_count')} "
        f"scanned={registry.get('scanned_files')} "
        f"full_stack_env_py={'yes' if registry.get('full_stack_env_py') else 'no'}"
    )
    if not resolution.get("applied", True):
        if resolution.get("server_defaults"):
            lines.append(
                "not applied: --server-defaults -- the harness injects no "
                "MTPLX_FABLE_* set at all, so whatever the stack reports as "
                "armed came from the SERVER's own defaults"
            )
        else:
            lines.append(
                f"not applied: stack={resolution.get('stack')} takes no "
                "MTPLX_FABLE_* set"
            )
        return "\n".join(lines)
    if resolution.get("default_used"):
        lines.append("source: harness DEFAULT_FABLE_FLAGS (no --fable-flags-file)")
    for key in resolution.get("dropped_vs_default") or []:
        lines.append(
            f"dropped {key}={DEFAULT_FABLE_FLAGS.get(key)} "
            "(in DEFAULT_FABLE_FLAGS, not in the files -> OFF this run)"
        )
    for entry in resolution.get("files") or []:
        lines.append(
            f"file {entry['path']} sha256={entry['sha256']} "
            f"bytes={entry['bytes']} keys={len(entry['keys'])}"
        )
    for item in resolution.get("shadowed") or []:
        lines.append(
            f"shadowed {item['key']}={item['value']} from {item['path']} "
            f"by {item['shadowed_by']}"
        )
    sources = resolution.get("sources") or {}
    resolved = resolution.get("resolved") or {}
    lines.append(f"resolved {len(resolved)} key(s)")
    for key in sorted(resolved):
        lines.append(f"  {key}={resolved[key]}  <- {sources.get(key, '?')}")
    return "\n".join(lines)


def cli_env_overrides(pairs: Sequence[str] | None) -> dict[str, str]:
    """``--env KEY=VALUE``, applied last and recorded separately.

    Kept out of ``server_env_overrides`` so the receipt distinguishes the
    harness's own blocks from an operator's one-off, and refused when it is
    not KEY=VALUE rather than silently setting an empty variable.
    """

    out: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = str(pair).partition("=")
        if not sep or not _FLAG_KEY_RE.match(key.strip()):
            raise SystemExit(f"--env expects KEY=VALUE, got {pair!r}")
        out[key.strip()] = value
    return out


def branch_env_overrides(
    *,
    stack: str,
    full_stack: bool,
    fable_flags: Mapping[str, str],
    server_defaults: bool = False,
) -> dict[str, str]:
    """Exactly what the harness adds on top of ``os.environ``.

    ONE definition, used to build the launch env AND to write
    ``server_env_overrides`` into the receipt, so the receipt cannot claim a
    set the process did not get.

    ``server_defaults`` strips the harness back to :data:`SERVER_DEFAULTS_ENV`
    -- the two memory caps and nothing else. No ``BRANCH_BASE_ENV``, no
    ``FULL_STACK_ENV``, no ``MTPLX_FABLE_*``, and no session-bank budget (the
    served tree defaults it; an exported ``auto`` reads as an operator
    override). The caps stay because every arm has to be scored under one
    memory envelope or the comparison is meaningless; everything else the
    server must supply itself, which is the whole question a defaults arm asks.
    """

    if stack == "mlx-serve":
        return dict(MLXSERVE_ENV)
    if server_defaults:
        return dict(SERVER_DEFAULTS_ENV)
    env = dict(COMMON_ENV)
    if stack == "branch":
        env.update(BRANCH_BASE_ENV)
        if full_stack:
            env.update(FULL_STACK_ENV)
        env.update(fable_flags)
    return env


#: Installs a stderr logging handler, then runs the server module unchanged.
#: The server itself configures no handler, so this is the only way to see
#: the [qwen4-*] install reports. NOT used for timed cells by default: an
#: INFO handler also turns on any per-request logging, whose I/O would land
#: inside the very measurement it is there to validate.
SERVER_LOG_BOOTSTRAP = (
    "import logging,sys,runpy;"
    "logging.basicConfig(level=logging.{level},stream=sys.stderr,"
    "format='%(message)s');"
    "sys.argv[0]='mtplx.server.openai';"
    "runpy.run_module('mtplx.server.openai',run_name='__main__')"
)


def build_server_argv(
    *,
    python: str | Path,
    port: int,
    host: str = "127.0.0.1",
    log_level: str | None = None,
    profile: str | None = "turbo",
) -> list[str]:
    """The model's own serving defaults: turbo, MTP depth 3, serial.

    Reasoning is NOT pinned here -- each request carries its own
    ``enable_thinking`` / ``reasoning_effort``, so one server serves both the
    thinking-off vanity cell and the ``xhigh`` sweep.

    ``profile=None`` omits ``--profile`` from the argv altogether, so the
    server picks its OWN default. That is the only way to measure what a user
    who types ``mtplx serve`` actually gets: passing the profile name we
    expect the default to be would prove nothing, because it would arm the
    stack by hand and then congratulate the server for it.
    """

    launcher = (
        ["-c", SERVER_LOG_BOOTSTRAP.format(level=log_level.upper())]
        if log_level
        else ["-m", "mtplx.server.openai"]
    )
    return [
        str(python),
        *launcher,
        "--model", str(MODEL),
        "--model-id", MODEL_ID,
        "--host", host,
        "--port", str(int(port)),
        *(() if profile is None else ("--profile", str(profile))),
        "--generation-mode", "mtp",
        "--load-mtp",
        "--depth", "3",
        "--scheduler-mode", "serial",
        "--ssd-session-cache", "off",
        "--no-auth",
    ]


MLXSERVE_BINARY = Path(
    "/Users/davidtai/projects/OpenSourceWTF/.tools/mlx-serve-macos-arm64/mlx-serve"
)
MLXSERVE_MODEL = Path(
    "/Users/davidtai/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit"
)
MLXSERVE_MODEL_ID = "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit"

#: The ONE thinking setting all three engines honour identically on this
#: checkpoint, replacing the per-engine NO_THINK_EFFORT map that made the
#: vanity cell a different experiment on each server.
#:
#: The two served packs ship the BYTE-IDENTICAL chat template
#: (sha256 c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041 on
#: both Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed and
#: ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit), and that template reads
#: ``reasoning_effort`` ONLY inside
#: ``{%- if enable_thinking is undefined or enable_thinking is true %}``
#: (chat_template.jinja:46-57). With thinking off it emits no reasoning
#: preamble at all and closes the block in the generation prompt
#: (``<think>\n\n</think>\n\n``, chat_template.jinja:163-169). So on the OFF
#: arm the effort word is INERT and only ``enable_thinking`` matters.
#:
#: Sending it with no ``reasoning_effort`` at all is what makes the three
#: agree:
#:
#: * MTPLX (branch and upstream 2.10.2, identical code):
#:   ``_thinking_enabled_for_request`` takes the request's ``enable_thinking``
#:   over the server default, and ``_reasoning_effort_for_state`` returns
#:   None the moment thinking is off, so chat_encoding.py:284 never puts
#:   ``reasoning_effort`` in the template kwargs.
#: * mlx-serve: this checkpoint's template does NOT contain "Disabling
#:   thinking is not supported", so ``refuses_nothink`` is false and
#:   ``serializeExtraContext`` passes ``enable_thinking:false`` straight
#:   through (chat.zig:1098-1104). It also passes
#:   ``qwen38EffortFor(null, false) == "low"``, which the template cannot
#:   read on that arm.
#:
#: What must NOT be sent is the old pairing. ``reasoning_effort:"low"`` with
#: ``enable_thinking:false`` makes mlx-serve's ``resolveEnableThinking``
#: compute ``false or effort_cfg.enable`` == TRUE (server.zig:4791-4798;
#: reasoningEffortFromWord, server.zig:4743, returns enable=true for every
#: word except "none") -- which is exactly how the first mlx-serve vanity
#: cell ran with thinking ON and had to be voided. Omitting the field removes
#: the trap rather than routing around it with a third spelling.
#:
#: SOURCE OF RECORD for every mlx-serve citation here: the tree the SHIPPED
#: binary was built from, :data:`MLXSERVE_SOURCE_ROOT` -- v26.8.11 / 5afa398,
#: identified by a byte-identical NOTICE, the embedded mlx-c pin 56b2d39fc831,
#: and the `generate.Generator.sampleLazy` symbol that commit introduced. NOT
#: `.worktrees/mlxserve-qwen4-ple-pread-workers`, which is 20 commits ahead
#: and whose server.zig line numbers are offset by +84 in this region.
VANITY_ENABLE_THINKING = False
VANITY_REASONING_EFFORT: str | None = None

#: Identity this client presents, pinned EXPLICITLY rather than left to
#: urllib's default, because MTPLX silently voids the request's sampler for a
#: client it recognises. When ``_app_managed_client_hint``
#: (openai.py:14073-14088) fires, ``_client_controls_allowed`` goes false and
#: temperature, top-p and top-k are replaced by the server's own values and
#: stamped ``client_sampler_fields_ignored`` -- no error, no 4xx, and NO
#: change to the request digest. The battery would still show byte-identical
#: bodies while one arm sampled differently.
#:
#: Two ways in, and they are not the same rule:
#:
#: * an EXPLICIT identity -- ``X-MTPLX-Client``, ``X-Client-Name``, or
#:   ``metadata.client``/``client_label`` -- is managed when it is in
#:   {browser, chat, hermes, mtplx, mtplx_app, mtplxapp, opencode, openwebui,
#:   pi} OR simply starts with ``mtplx_`` after normalisation
#:   (``-`` and space both become ``_``);
#: * a USER-AGENT is mapped to an identity only by the four substrings in
#:   :data:`MANAGED_USER_AGENT_SUBSTRINGS` (openai.py:14082-14101), plus an
#:   ``X-OpenWebUI-*`` header.
#:
#: So the harness sends no client-identity header at all, and its UA avoids
#: all four substrings AND the ``mtplx`` prefix -- the prefix is not reachable
#: through the UA today, but a name one rule change away from being managed is
#: not a name worth keeping on a fairness harness. Tests pin both properties.
BENCH_USER_AGENT = "server-cell-bench/1"

#: Header names that would hand MTPLX an explicit identity. The harness must
#: never send one; the test asserts their absence by name.
MTPLX_CLIENT_IDENTITY_HEADERS: tuple[str, ...] = (
    "X-MTPLX-Client",
    "X-Client-Name",
)

#: The ONLY User-Agent substrings that become a client identity
#: (openai.py:14082-14101). "mtplx" is deliberately NOT among them: it is
#: managed via the explicit-header path, not via the UA.
MANAGED_USER_AGENT_SUBSTRINGS: tuple[str, ...] = (
    "claude-cli",
    "opencode",
    "android",
    "jetbrains",
    "ai-sdk",
)

#: Fields whose value is LEGITIMATELY engine-specific, and therefore the only
#: ones excluded from the per-cell parity digest. Everything else in the body
#: must match across the three engines or the cells are not comparable.
PARITY_TRANSPORT_FIELDS: tuple[str, ...] = ("model",)


def chat_body(
    *,
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    """Every field of a request EXCEPT the prompt itself.

    One definition, used to build the live request AND to describe it in the
    dry run and the receipt, so what the plan claims and what the wire
    carries cannot drift.
    """

    body: dict[str, Any] = {
        "model": model_id,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if top_k is not None:
        body["top_k"] = int(top_k)
    if seed is not None:
        body["seed"] = int(seed)
    if reasoning_effort is not None:
        body["reasoning_effort"] = str(reasoning_effort)
    if enable_thinking is not None:
        body["enable_thinking"] = bool(enable_thinking)
    return body


def parity_body(
    body: Mapping[str, Any], *, prompt_sha256: str, prompt_chars: int
) -> dict[str, Any]:
    """One request in engine-agnostic canonical form.

    ``model`` is dropped because the three engines legitimately serve the
    request under different ids. ``messages`` is replaced by the prompt's
    sha256 and length: the text is identical across engines, but a 255K-token
    array would make the plan undiffable and the receipt enormous, and the
    digest pins it exactly either way.

    ONE construction, used by the dry run to describe the plan and by the
    battery to record what actually went out, so the two are comparable
    digest-for-digest.
    """

    out = {
        key: value
        for key, value in body.items()
        if key not in PARITY_TRANSPORT_FIELDS and key != "messages"
    }
    out["prompt_sha256"] = str(prompt_sha256)
    out["prompt_chars"] = int(prompt_chars)
    return dict(sorted(out.items()))


def parity_digest(canonical: Mapping[str, Any]) -> str:
    """sha256 of :func:`parity_body`'s output.

    Two engines whose cells carry the same digest ASKED for the same thing.
    It is not proof that both HONOURED it -- ``response_parity`` in the same
    record is what speaks to that -- but a digest mismatch settles the
    question before any number is read.
    """

    return hashlib.sha256(
        json.dumps(dict(canonical), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


#: Where a server may report the thinking-phase token count, best first. The
#: OpenAI spelling is the nested one; some servers flatten it; a server that
#: reports neither leaves the client with the streamed reasoning_content
#: length, which is characters, not tokens, and must be labelled as such.
REASONING_TOKEN_PATHS: tuple[tuple[str, ...], ...] = (
    ("completion_tokens_details", "reasoning_tokens"),
    ("reasoning_tokens",),
    ("completion_tokens_details", "reasoning"),
)


def reasoning_tokens_from_usage(
    usage: Mapping[str, Any] | None,
) -> tuple[int | None, str]:
    """The thinking-phase token count and WHICH field carried it."""

    if not isinstance(usage, Mapping):
        return None, "none"
    for path in REASONING_TOKEN_PATHS:
        node: Any = usage
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
            if node is None:
                break
        if isinstance(node, (int, float)):
            return int(node), "usage:" + ".".join(path)
    return None, "none"


#: What MTPLX echoes back about the sampler it ACTUALLY used, allowlisted in
#: its own PUBLIC_MTPLX_STATS_KEYS. Reading these turns "we asked for top-k 20"
#: into "the server says it used top-k 20", which is the only way to catch a
#: request whose controls were voided -- MTPLX substitutes its own values and
#: stamps ``client_sampler_fields_ignored`` without an error, and the request
#: digest is unchanged, so nothing else in this harness would notice.
APPLIED_SAMPLER_FIELDS: tuple[tuple[str, str], ...] = (
    ("temperature", "effective_temperature"),
    ("top_p", "effective_top_p"),
    ("top_k", "effective_top_k"),
)


def applied_sampler(call: Mapping[str, Any]) -> dict[str, Any]:
    """What the server says it used, when the server says anything.

    mlx-serve echoes nothing, so ``available`` is False there and every field
    is None -- reported as unknown rather than as agreement.
    """

    stats = (call.get("raw_engagement") or {}).get("mtplx_stats") or {}
    if not isinstance(stats, Mapping):
        stats = {}
    out: dict[str, Any] = {
        "available": any(key in stats for _, key in APPLIED_SAMPLER_FIELDS),
        "source": "mtplx_stats" if stats else "none",
    }
    for name, key in APPLIED_SAMPLER_FIELDS:
        out[name] = stats.get(key)
        out[f"requested_{name}"] = stats.get(f"request_{name}")
    out["seed"] = stats.get("server_seed")
    # MTPLX names this itself when it overrode the request's sampler.
    out["client_sampler_fields_ignored"] = stats.get(
        "client_sampler_fields_ignored"
    )
    out["control_owner"] = stats.get("mtplx_control_owner")
    # "truncated_inside_reasoning" is ROUTINE at Qwen 3.8 xhigh with
    # max_tokens 1024: the model spent the whole budget thinking and the
    # visible answer is empty. It is not a failure and must not be read as one.
    out["content_empty_reason"] = stats.get("content_empty_reason")
    out["reasoning_tokens"] = stats.get("reasoning_tokens")
    out["answer_tokens"] = stats.get("answer_tokens")
    return out


def sampler_honoured(
    applied: Mapping[str, Any], sampling: Mapping[str, Any]
) -> dict[str, Any]:
    """Per field: did the server use what the cell asked for?

    ``None`` means the server reported nothing, which is NOT agreement. A
    False here is the loudest signal this harness can produce: it means two
    cells with identical request digests did not run the same experiment.
    """

    verdict: dict[str, Any] = {}
    for name, _ in APPLIED_SAMPLER_FIELDS:
        got = applied.get(name)
        want = sampling.get(name)
        if got is None or want is None:
            verdict[name] = None
            continue
        verdict[name] = bool(abs(float(got) - float(want)) < 1e-9)
    if applied.get("client_sampler_fields_ignored"):
        verdict["client_sampler_fields_ignored"] = True
    return verdict


def response_parity(
    call: Mapping[str, Any], sampling: Mapping[str, Any]
) -> dict[str, Any]:
    """What the engine actually DID with the settings every engine was sent.

    Request parity proves the three were asked the same thing. This is the
    other half: an engine that silently ignores ``top_k``, caps the thinking
    phase, or refuses to think at all produces different numbers here even
    though the request digests match.
    """

    tokens, source = reasoning_tokens_from_usage(call.get("usage"))
    chars = call.get("reasoning_chars")
    applied = applied_sampler(call)
    return {
        # what the server says it actually used, and whether that matches
        "applied_sampler": applied,
        "sampler_honoured": sampler_honoured(applied, sampling),
        "completion_tokens": call.get("completion_tokens"),
        "finish_reason": call.get("finish_reason"),
        "reasoning_chars": chars,
        "reasoning_tokens": tokens,
        "reasoning_tokens_source": source,
        # the honest fallback when no server reports reasoning tokens: a
        # nonzero character count proves a thinking phase happened, and zero
        # proves one did not. It is NOT a token count and is not labelled one.
        "thinking_observed": (
            None if chars is None else bool(int(chars) > 0)
        ),
        "thinking_requested": bool(sampling.get("enable_thinking")),
        "reasoning_effort_requested": sampling.get("reasoning_effort"),
        "temperature_requested": sampling.get("temperature"),
        "top_p_requested": sampling.get("top_p"),
        "top_k_requested": sampling.get("top_k"),
        "seed_requested": sampling.get("seed"),
    }


def cell_sampling(
    args: argparse.Namespace, prompt: Mapping[str, Any], seed: int | None
) -> dict[str, Any]:
    """The sampling and thinking settings for one cell, engine-independent.

    The vanity cell differs from a sweep cell in its PROMPT and in nothing
    else but ``enable_thinking``: same temperature, top-p, top-k, seed and
    max_tokens. It used to run greedy and unseeded, which made it a second
    experiment rather than the same experiment on a short prompt.
    """

    is_vanity = prompt.get("cell") == "vanity"
    family = getattr(args, "model_family_resolved", None) or QWEN38_FAMILY
    if family == DEEPSEEK_V41_FAMILY:
        # DSV4.1 now has a real chat template + thinking mode. Send
        # enable_thinking explicitly so the served render matches the counter;
        # reasoning_effort is inert in chat mode and dropped there. Default is
        # the official thinking-OFF (see DEEPSEEK_V41_DEFAULT_ENABLE_THINKING).
        dsv_thinking, dsv_effort = deepseek_v41_thinking_settings(args)
        enable_thinking, reasoning_effort = _cell_thinking_settings(
            DEEPSEEK_V41_FAMILY,
            is_vanity,
            reasoning_effort=None,
            dsv41_enable_thinking=dsv_thinking,
            dsv41_reasoning_effort=dsv_effort,
        )
    else:
        reasoning_effort = VANITY_REASONING_EFFORT if is_vanity else args.reasoning
        enable_thinking = VANITY_ENABLE_THINKING if is_vanity else True
    return {
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "seed": None if args.no_seed else seed,
        "reasoning_effort": reasoning_effort,
        "enable_thinking": enable_thinking,
    }

#: The tree the SHIPPED mlx-serve binary was built from. Every mlx-serve
#: citation in this module is a line number in THIS tree. Pointing at a
#: newer worktree is how the first pass mis-cited the effort resolution by
#: 84 lines: the release is v26.8.11 / 5afa398, and the qwen4-ple-pread
#: worktree is 20 commits past it.
MLXSERVE_SOURCE_ROOT = Path(
    "/Users/davidtai/projects/OpenSourceWTF/.tools/mlx-serve-v26.8.11"
)

#: Fields an engine ACCEPTS on the wire and does not honour. The request
#: digest proves the three engines were ASKED the same thing; it cannot prove
#: they DID the same thing, and a field that is parsed, stored and then
#: dropped is invisible on both sides unless it is written down.
#:
#: Verified against the shipped build, not a newer worktree.
ENGINE_FIELD_CAVEATS: dict[str, tuple[dict[str, str], ...]] = {
    "branch": (
        {
            "field": "top_p+top_k order",
            "honoured": "yes, but top_p is applied BEFORE top_k",
            "effect": (
                "Both values are applied verbatim, but the FILTER ORDER "
                "differs from mlx-serve, which applies top_k first "
                "(generate.zig:8026 then :8034 on the AR path, :7663 then "
                ":7669 on the MTP verify path). At top_k=20 / top_p=0.95 the "
                "two orders can admit different candidate sets, so this is a "
                "real difference in the sampled distribution, not a "
                "formality. It is inherent to the engines and is not "
                "something the request can equalise."
            ),
            "cite": (
                "mtplx/sampling.py:121-130 -- 'Local mlx_lm applies top-p "
                "before top-k, so MTPLX's NumPy reference path mirrors that "
                "order'"
            ),
            "remedy": (
                "disclose it. Sending only ONE of the two would remove the "
                "difference, but it would also change the sampler the model "
                "ships with, so the honest move is the footnote rather than a "
                "quieter benchmark."
            ),
        },
    ),
    "upstream": (
        {
            "field": "top_p+top_k order",
            "honoured": "yes, but top_p is applied BEFORE top_k",
            "effect": (
                "Both values are applied verbatim, but the FILTER ORDER "
                "differs from mlx-serve, which applies top_k first "
                "(generate.zig:8026 then :8034 on the AR path, :7663 then "
                ":7669 on the MTP verify path). At top_k=20 / top_p=0.95 the "
                "two orders can admit different candidate sets, so this is a "
                "real difference in the sampled distribution, not a "
                "formality. It is inherent to the engines and is not "
                "something the request can equalise."
            ),
            "cite": (
                "mtplx/sampling.py:121-130 -- 'Local mlx_lm applies top-p "
                "before top-k, so MTPLX's NumPy reference path mirrors that "
                "order'"
            ),
            "remedy": (
                "disclose it. Sending only ONE of the two would remove the "
                "difference, but it would also change the sampler the model "
                "ships with, so the honest move is the footnote rather than a "
                "quieter benchmark."
            ),
        },
    ),
    "mlx-serve": (
        {
            "field": "top_p+top_k order",
            "honoured": "yes, but top_k is applied BEFORE top_p",
            "effect": (
                "The opposite order to both MTPLX arms, which apply top_p "
                "first (mtplx/sampling.py:121-130). At top_k=20 / top_p=0.95 "
                "the two orders can admit different candidate sets. Applies "
                "on the AR path and on the MTP verify distribution alike."
            ),
            "cite": (
                "generate.zig:8026 then :8034 (sampleTokenLazy), and :7663 "
                "then :7669 (probsAllPositions) -- mlx-serve v26.8.11 / "
                "5afa398"
            ),
            "remedy": (
                "disclose it; no request field reorders an engine's own "
                "sampler."
            ),
        },
        {
            "field": "seed",
            "honoured": "partial",
            "effect": (
                "The AR sampler and the accept-test PRNG ARE seeded "
                "(generate.zig:8054 seedKey, generate.zig:2422). But the "
                "correction/bonus token committed on EVERY MTP round is "
                "drawn with a null key -- MLX's process-global RNG, seeded "
                "once from the wall clock at main.zig:1078. So under --mtp a "
                "seeded request is NOT reproducible, and the harness runs "
                "mlx-serve with --mtp."
            ),
            "cite": (
                "generate.zig:5268-5277, and the DEFAULT batched arm "
                "generate.zig:4827-4831 (mlx-serve v26.8.11 / 5afa398)"
            ),
            "remedy": (
                "no request field and no server flag fixes it. The seeded "
                "control is --mlxserve-no-mtp, which this harness already "
                "offers. Otherwise disclose it: mlx-serve's three seeds are "
                "REPEATS, not reproductions, and its spread is a sample of "
                "run-to-run variance rather than a seed effect."
            ),
        },
    ),
}

def engine_field_caveats(stack: str) -> tuple[dict[str, str], ...]:
    """Fields this engine accepts and drops. () when it honours everything."""

    return ENGINE_FIELD_CAVEATS.get(stack, ())


def render_field_caveats(stack: str) -> list[str]:
    """One block per dropped field, for the dry run and the launch log."""

    lines: list[str] = []
    for item in engine_field_caveats(stack):
        lines.append(
            f"CAVEAT {stack} {item['field']}: honoured={item['honoured']}"
        )
        lines.append(f"  effect:  {item['effect']}")
        lines.append(f"  cite:    {item['cite']}")
        lines.append(f"  remedy:  {item['remedy']}")
    return lines


#: Undocumented (absent from --help); raw bytes, no suffix accepted. This is
#: MLX's reclaimable buffer pool, NOT a hard wired ceiling, so it is matched
#: numerically to the MTPLX cap rather than claimed to be equivalent.
MLXSERVE_ENV: dict[str, str] = {
    "MLX_SERVE_CACHE_LIMIT": str(100 * 1024**3),
}

#: GET /props exposes memory.peak_bytes == mlx_get_peak_memory(), the SAME
#: quantity MTPLX reports as peak_memory_bytes. This is the comparable number;
#: the Prometheus gauges and ps-rss are not.
MLXSERVE_PROPS_PATH = "/props"

MLXSERVE_METRIC_KEYS = (
    "mlx_serve:memory_mb",
    "mlx_serve:mlx_active_bytes",
    "mlx_serve:mlx_cache_bytes",
)


def build_mlxserve_argv(
    *, port: int, host: str = "127.0.0.1", mtp: bool = True
) -> list[str]:
    """mlx-serve with the prefix cache OFF and MTP forced on.

    MTP defaults OFF for a MoE target, and the MTPLX arms run MTP depth 3, so
    leaving it off would compare a speculative engine against a plain one.
    The prefix cache is disabled outright: the battery sends three related
    prompts per size and a cross-seed cache hit would fabricate a prefill win.
    """

    return [
        str(MLXSERVE_BINARY),
        "--model", str(MLXSERVE_MODEL),
        "--serve",
        "--host", host,
        "--port", str(int(port)),
        "--mtp" if mtp else "--no-mtp",
        "--no-pld",
        "--metrics",
        # mlx-serve's preflight counts only free+inactive, so the previous
        # arm's mmap'd weights sitting in the page cache read as unavailable
        # and it refuses to start (68.4 GB of weights against a reported
        # 56.9 GB available on a 128 GB box). Those pages are file-backed and
        # reclaimable; the override is the documented escape.
        "--skip-mem-preflight",
        "--prefix-cache-entries", "0",
        "--prefix-cache-mem", "0",
    ]


def power_source() -> dict[str, Any]:
    """AC or battery at cell start. A cell on battery is throttled and void."""

    try:
        text = subprocess.check_output(
            ["pmset", "-g", "batt"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}
    lowered = text.lower()
    on_ac = "ac power" in lowered
    match = re.search(r"(\d+)%", text)
    return {
        "available": True,
        "on_ac": on_ac,
        "source": "AC" if on_ac else "battery",
        "charge_pct": int(match.group(1)) if match else None,
        "raw": text.strip().splitlines()[:2],
    }


def fan_state() -> dict[str, Any]:
    """Fan mode and actual RPM. Recorded, never changed mid-battery."""

    for binary in (
        # root-owned copy first: this is the one the sudoers rule trusts and
        # the one `thermalforge max` actually drives.
        Path("/usr/local/bin/thermalforge"),
        Path.home() / ".mtplx" / "bin" / "thermalforge",
    ):
        if not binary.exists():
            continue
        try:
            payload = json.loads(
                subprocess.check_output(
                    [str(binary), "status"], text=True, stderr=subprocess.DEVNULL
                )
            )
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        fans = payload.get("fans") or []
        modes = {str(f.get("mode")) for f in fans}
        # "max" is a mode the CLI accepts; `status` reports the resulting mode
        # as "manual". Judge by RPM against each fan's own ceiling rather than
        # by the word, and require every fan, not just fan 0.
        at_max = bool(fans) and all(
            float(f.get("actual_rpm") or 0) >= 0.95 * float(f.get("max_rpm") or 1)
            for f in fans
        )
        return {
            "available": True,
            "mode": (fans[0].get("mode") if fans else None),
            "modes": sorted(modes),
            "all_manual": modes == {"manual"},
            "at_max": at_max,
            "binary": str(binary),
            "fans": [
                {
                    k: f.get(k)
                    for k in ("index", "mode", "actual_rpm", "target_rpm", "max_rpm")
                }
                for f in fans
            ],
        }
    return {"available": False}


PROFILES_PY = MTPLX_SOURCE_ROOT / "mtplx" / "profiles.py"


def profile_allowlist() -> set[str]:
    """Keys the profile layer validates. Everything else is unchecked."""

    try:
        text = PROFILES_PY.read_text()
    except OSError:
        return set()
    start = text.find("MODEL_RUNTIME_ENV_OVERRIDE_KEYS")
    if start < 0:
        return set()
    return set(re.findall(r'"(MTPLX_[A-Z0-9_]+)"', text[start : start + 4000]))


def report_unvalidated_env(env: Mapping[str, str]) -> dict[str, Any]:
    """Name every MTPLX_* key the server will NOT validate.

    15 of the 20 restack keys are read by a bare ``os.environ.get(name,
    default)`` at their use site, so a misspelling silently no-ops and the arm
    quietly runs a lesser stack -- which is exactly how the server arms ran
    without the compiled fixed-M4 verifier for a whole battery. Printing them
    does not validate them, but it makes the unvalidated surface visible.
    """

    allow = profile_allowlist()
    ours = sorted(k for k in env if k.startswith("MTPLX_"))
    unvalidated = [k for k in ours if k not in allow]
    print(
        f"[server-cell] MTPLX_* env: {len(ours)} set, {len(unvalidated)} NOT "
        f"profile-validated (a typo in these silently no-ops):",
        flush=True,
    )
    for key in unvalidated:
        print(f"    UNVALIDATED {key}={env[key]}", flush=True)
    return {
        "allowlist_size": len(allow),
        "set_keys": ours,
        "unvalidated_keys": unvalidated,
    }


#: Markers a server emits only when the lane actually installs, split by
#: whether a client can SEE them.
#:
#: The server installs no logging handler (no basicConfig/addHandler anywhere
#: in mtplx/server/openai.py), so ``logger.info`` output goes nowhere:
#: runtime.py's own comment says as much -- "logger.info alone is invisible
#: under `python -m mtplx.server.openai`". Only ``print`` reaches the log.
#: The three M4 reports are therefore UNOBSERVABLE by default, and gating on
#: them would refuse every cell no matter what installed. Their patterns are
#: kept and corrected to the real runtime strings so they match whenever the
#: handler is installed (--server-log-level).
ENGAGEMENT_MARKERS: dict[str, dict[str, str]] = {
    # -- visible: print(..., file=sys.stderr) / print("[mtplx] ...")
    "frspec_installed": {
        "pattern": r"\[frspec\] install report.*'installed': True",
        "visibility": "print",  # draft_lm_head.py:361
    },
    "frspec_disabled": {
        "pattern": r"\[frspec\] disabled",
        "visibility": "print",
    },
    "gdn_blocked_prefill": {
        "pattern": r"\[mtplx\] gdn-blocked-prefill .*'installed': True",
        "visibility": "print",  # kernels/gdn_blocked_prefill.py:446
    },
    "prefill_chunk_override": {
        "pattern": r"\[mtplx\] profile env override: MTPLX_PREFILL_CHUNK_SIZE",
        "visibility": "print",  # profiles.py:961
    },
    # -- logger.info only: invisible unless a handler is installed
    "m4_fixed_verify": {
        "pattern": r"\[qwen4-fixed-M4-verify\] .*'installed': True",
        "visibility": "logger",  # runtime.py:1093
    },
    "m4_stage3": {
        "pattern": r"\[qwen4-M4-stage3\] .*'installed': True",
        "visibility": "logger",  # runtime.py:1112
    },
    "compiled_mtp_prepare": {
        "pattern": r"\[qwen4-compiled-MTP-prepare\] ",
        "visibility": "logger",  # runtime.py:1081
    },
    # -- mlx-serve
    "spec_stats": {"pattern": r"\[spec-stats\] mode=", "visibility": "print"},
    "qsa_gather": {"pattern": r"\[qsa-gather\] engaged", "visibility": "print"},
}

#: Auto-stamped by the server itself (openai.py:836-869) when the served pack
#: matches the fixed-M4 geometry. Recorded per arm so the two branch rows can
#: be compared directly: they must arm the SAME four routes, and a divergence
#: is a real signal rather than something to reason about from precedence.
AUTO_ARMED_KEYS = (
    "MTPLX_COMPILED_VERIFY",
    "MTPLX_QWEN4_FIXED_M4_VERIFY",
    "MTPLX_QWEN4_M4_STAGE3",
    "MTPLX_QSA_M4_FUSED_KV_GATHER",
    "MTPLX_BATCH_TARGET_ARRAYS",
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
    # openai.py:870-880 pins this to "0" for this family against turbo's "1";
    # the control arm sets the same 0, so it is server-owned either way.
    "MTPLX_NAX_VERIFY",
)

#: What --require-full-stack can actually PROVE from a default server launch.
OBSERVABLE_REQUIRED = ("frspec_installed", "gdn_blocked_prefill")

#: Real, but unprovable without --server-log-level. Reported, never gated on.
LOGGER_ONLY = ("m4_fixed_verify", "m4_stage3", "compiled_mtp_prepare")


def scan_engagement(log_path: Path | None) -> dict[str, Any]:
    """Read a server log and report which engagement markers appeared."""

    if not log_path or not log_path.exists():
        return {"available": False, "reason": "no server log"}
    try:
        text = log_path.read_text(errors="replace")
    except OSError as error:
        return {"available": False, "reason": str(error)}
    found = {
        name: bool(re.search(spec["pattern"], text))
        for name, spec in ENGAGEMENT_MARKERS.items()
    }
    logging_on = any(found[name] for name in LOGGER_ONLY)
    frspec_n = re.search(r"'n':\s*(\d+)", text)
    ladder = re.findall(r'"kind":\s*"ladder",\s*"context":\s*(\d+),\s*"state":\s*"(\w+)"', text)
    return {
        "available": True,
        "markers": found,
        "logger_handler_installed": logging_on,
        "unobservable_by_default": [
            n for n in LOGGER_ONLY if not found[n]
        ],
        "frspec_n": int(frspec_n.group(1)) if frspec_n else None,
        "ladder": [{"context": int(c), "state": st} for c, st in ladder],
        "ladder_all_ok": bool(ladder) and all(st == "ok" for _, st in ladder),
    }


def require_preflight(
    engagement: Mapping[str, Any], *, full_stack: bool
) -> list[str]:
    """Proofs a PREFLIGHT boot must show. [] == pass.

    A preflight runs with a logging handler installed, so the three
    [qwen4-*] install reports ARE visible here and can be required -- which
    they cannot be during a timed cell. Only a full-stack configuration is
    held to them: shipped upstream legitimately has none of them, so its
    preflight records what it found and asserts only the ladder.

    W61 is making those reports print at install time the way frspec already
    does; once that lands this whole mode can go and the timed cells can
    carry their own proof again.
    """

    problems = require_full_stack(engagement) if full_stack else []
    if not engagement.get("available"):
        return [f"no engagement evidence: {engagement.get('reason')}"]
    if not engagement.get("ladder_all_ok"):
        if not any("ladder" in p for p in problems):
            problems.append(
                f"warmup ladder not all ok: {engagement.get('ladder')}"
            )
    if not full_stack:
        return problems
    markers = engagement.get("markers") or {}
    if not engagement.get("logger_handler_installed"):
        problems.append(
            "no [qwen4-*] report seen at all: run the preflight with "
            "--server-log-level INFO, or the handler did not install"
        )
    for name in LOGGER_ONLY:
        if not markers.get(name):
            problems.append(f"{name} did not install (no {name} report line)")
    return problems


def require_full_stack(engagement: Mapping[str, Any]) -> list[str]:
    """Proofs a 'branch (full stack)' cell must show. [] == pass.

    Gates ONLY on markers a default server launch actually emits. The three
    M4 reports are logger.info with no handler, so requiring them would
    refuse every cell regardless of what installed -- a gate that can never
    pass is not a gate. They are reported as unverified instead, and
    --server-log-level makes them provable when that matters more than the
    per-request log I/O it adds.
    """

    problems: list[str] = []
    if not engagement.get("available"):
        return [f"no engagement evidence: {engagement.get('reason')}"]
    markers = engagement.get("markers") or {}
    if not markers.get("frspec_installed"):
        problems.append(
            "frspec did not install (expected \"[frspec] install report "
            "... 'installed': True ... 'n': 65536\")"
        )
    elif engagement.get("frspec_n") != 65536:
        problems.append(f"frspec n={engagement.get('frspec_n')}, expected 65536")
    if not markers.get("gdn_blocked_prefill"):
        problems.append(
            "no \"[mtplx] gdn-blocked-prefill ... 'installed': True\" line"
        )
    if not engagement.get("ladder_all_ok"):
        problems.append(f"warmup ladder not all ok: {engagement.get('ladder')}")
    return problems


def wake_request(
    base_url: str,
    model_id: str,
    *,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """An untimed request that ramps the GPU clock before a timed one.

    The first request after a cool thermal gate pays a clock-ramp cost the
    rest do not: run 2's first vanity request measured 0.26 s of prefill
    against 0.83-0.85 s for the gated repeats that followed. This burns that
    ramp on a throwaway call so the timed request sees a machine already at
    speed. Its result is discarded and it is counted nowhere.

    It carries the SAME thinking setting as the vanity cell on every engine,
    so the wake cannot itself be a per-engine difference. It used to take a
    ``stack`` and look the effort word up per engine; there is one setting
    now (see :data:`VANITY_ENABLE_THINKING`) and no lookup left to do.
    """

    started = time.monotonic()
    try:
        stream_chat(
            base_url=base_url,
            model_id=model_id,
            prompt=(
                "Reply with the single word OK. Do not explain, do not "
                "elaborate, and do not add punctuation beyond the word."
            ),
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            seed=None,
            reasoning_effort=VANITY_REASONING_EFFORT,
            enable_thinking=VANITY_ENABLE_THINKING,
            timeout_s=timeout_s,
        )
        return {"ok": True, "elapsed_s": time.monotonic() - started}
    except Exception as error:  # noqa: BLE001 - never fail a run on the wake
        return {
            "ok": False,
            "elapsed_s": time.monotonic() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def http_get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(
    base_url: str,
    process: subprocess.Popen[Any],
    *,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Poll ``/health`` until ok, failing fast if the server process dies."""

    deadline = time.monotonic() + timeout_s
    last_error = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with {process.returncode} before becoming ready"
            )
        try:
            health = http_get_json(f"{base_url}/health", timeout=10.0)
            if health.get("ok") or str(health.get("status") or "") == "ok":
                return health
            last_error = f"health not ok: {health!r}"
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            last_error = repr(error)
        time.sleep(3.0)
    raise RuntimeError(f"server never became healthy: {last_error}")


def wait_for_background_warmup(
    base_url: str, *, timeout_s: float = 1800.0
) -> dict[str, Any]:
    """Wait out the server's own background warmup.

    Timing a request that races startup GPU work measures the warmup, not the
    model. A server with no ``warmup.background`` block is treated as already
    warm rather than as an error.
    """

    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            health = http_get_json(f"{base_url}/health", timeout=15.0)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(3.0)
            continue
        background = (health.get("warmup") or {}).get("background")
        if not background:
            return {"state": "absent", "health": health}
        last = dict(background)
        state = str(background.get("state") or background.get("status") or "")
        if background.get("done") or state in {"done", "complete", "completed", "ready"}:
            return {"state": state or "done", "background": last}
        time.sleep(3.0)
    return {"state": "timeout", "background": last}


def terminate_server(process: subprocess.Popen[Any], *, grace_s: float = 180.0) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    process.terminate()
    try:
        return int(process.wait(timeout=grace_s))
    except subprocess.TimeoutExpired:
        process.kill()
        return int(process.wait(timeout=30))


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def cell_plan(
    *,
    prompts: Sequence[Mapping[str, Any]],
    contexts: Sequence[int],
    repeats: int,
    cells: str = "both",
    only_seed: int | None = None,
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> list[dict[str, Any]]:
    """Ascending order: vanity, then each context by seed.

    Ascending matters: the server's ``peak_memory_bytes`` is a process
    high-water mark, so only a monotonically growing battery lets each point's
    reading be read as that point's peak.

    Vanity repeats carry the SAME production seeds the sweep uses, one per
    repeat. They used to be unseeded, which left the one cell every reader
    looks at first as the only unreproducible row in the battery.
    """

    plan: list[dict[str, Any]] = []
    if cells in {"both", "vanity"}:
        vanity = [p for p in prompts if p.get("cell") == "vanity"]
        pool = [int(s) for s in seeds] or [None]
        for index in range(int(repeats)):
            for prompt in vanity:
                plan.append(
                    {
                        "prompt": prompt,
                        "repeat": index,
                        "seed": pool[index % len(pool)],
                    }
                )
    if cells == "vanity":
        return plan
    for target in sorted(int(c) for c in contexts):
        for prompt in sorted(
            (
                p
                for p in prompts
                if p.get("cell") == "sweep"
                and int(p["target_tokens"]) == target
                and (only_seed is None or int(p["seed"]) == int(only_seed))
            ),
            key=lambda p: int(p["seed"]),
        ):
            plan.append({"prompt": prompt, "repeat": 0, "seed": int(prompt["seed"])})
    return plan


# ---------------------------------------------------------------------------
# Engine version: measured per run, never guessed
# ---------------------------------------------------------------------------

#: One-liner run under the SERVER's own interpreter. importlib.metadata reads
#: dist metadata only -- it imports no model code, allocates no Metal, and is
#: safe to run inside a guarded window.
MTPLX_VERSION_PROBE = (
    "import importlib.metadata as m;print(m.version('mtplx'))"
)
MLX_VERSION_PROBE = "import importlib.metadata as m;print(m.version('mlx'))"

#: `mlx-serve --version` prints a block; these are the lines worth keeping.
#: Parsed by leading token so a reordered or extended block still resolves.
MLXSERVE_VERSION_LINES: tuple[tuple[str, str], ...] = (
    ("mlx-serve", "engine"),
    ("mlx-c", "mlx_c"),
    ("mlx", "mlx"),
)

VERSION_PROBE_TIMEOUT_S = 120.0


def _probe(argv: Sequence[str], *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Run a short command and capture it, never raising."""

    try:
        done = subprocess.run(
            [str(a) for a in argv],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=VERSION_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if done.returncode != 0:
        return {
            "ok": False,
            "error": f"exit {done.returncode}: {(done.stderr or '').strip()[:400]}",
        }
    return {"ok": True, "stdout": done.stdout, "stderr": done.stderr}


def parse_mlxserve_version(text: str) -> dict[str, Any]:
    """Pull engine / mlx / mlx-c out of a --version block.

    Longest token first, so the ``mlx-c`` line is never eaten by the ``mlx``
    prefix. Lines the block gains later are ignored rather than mis-parsed.
    """

    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        for token, field in MLXSERVE_VERSION_LINES:
            if field in out:
                continue
            head, sep, rest = line.partition(" ")
            if sep and head == token and rest.strip():
                out[field] = rest.strip()
                break
    return out


def probe_engine_version(args: argparse.Namespace) -> dict[str, Any]:
    """What build this arm actually is, measured on the box that runs it.

    NEVER guesses. A probe that fails omits the version and records why, so
    the report falls back to its hand-maintained operator string and says out
    loud that it is doing so -- which is the honest outcome, and strictly
    better than a plausible number nobody measured.
    """

    if args.stack == "mlx-serve":
        got = _probe([MLXSERVE_BINARY, "--version"])
        if not got["ok"]:
            return {"available": False, "reason": f"mlx-serve --version: {got['error']}"}
        raw = got["stdout"]
        parsed = parse_mlxserve_version(raw)
        if not parsed.get("engine"):
            return {
                "available": False,
                "reason": "mlx-serve --version printed no 'mlx-serve <version>' line",
                "raw": raw.strip(),
            }
        return {
            "available": True,
            "source": "mlx-serve --version",
            "engine": parsed["engine"],
            "mlx": parsed.get("mlx"),
            "mlx_c": parsed.get("mlx_c"),
            "git_head": None,
            "raw": raw.strip(),
        }

    if not args.server_python:
        return {"available": False, "reason": "no --server-python to probe with"}
    cwd = args.server_cwd or None
    got = _probe([args.server_python, "-c", MTPLX_VERSION_PROBE], cwd=cwd)
    if not got["ok"]:
        return {
            "available": False,
            "reason": f"importlib.metadata:mtplx: {got['error']}",
        }
    engine = got["stdout"].strip().splitlines()[-1].strip() if got["stdout"].strip() else ""
    if not engine:
        return {"available": False, "reason": "importlib.metadata:mtplx printed nothing"}

    mlx_got = _probe([args.server_python, "-c", MLX_VERSION_PROBE], cwd=cwd)
    mlx = (
        mlx_got["stdout"].strip().splitlines()[-1].strip()
        if mlx_got["ok"] and mlx_got["stdout"].strip()
        else None
    )
    # The dist version is stamped at INSTALL time; an editable install lags
    # the working tree by however many commits have landed since. For a
    # branch arm the sha is the version that matters.
    git_head = None
    git_reason = None
    if cwd:
        head = _probe(["git", "-C", str(cwd), "rev-parse", "HEAD"])
        if head["ok"] and head["stdout"].strip():
            git_head = head["stdout"].strip().splitlines()[-1].strip()
        else:
            git_reason = head.get("error") or "git rev-parse printed nothing"
    else:
        git_reason = "no --server-cwd"
    block: dict[str, Any] = {
        "available": True,
        "source": "importlib.metadata:mtplx",
        "engine": engine,
        "mlx": mlx,
        "mlx_c": None,
        "git_head": git_head,
        "raw": f"mtplx {engine}" + (f"\nmlx {mlx}" if mlx else ""),
    }
    if git_reason:
        block["git_head_reason"] = git_reason
    if not mlx_got["ok"]:
        block["mlx_reason"] = mlx_got.get("error")
    return block


def engine_version_block(args: argparse.Namespace) -> dict[str, Any] | None:
    """The receipt's ``engine_version``, or None with the reason recorded.

    Shape is the one ``server_cell_report.engine_version_of`` reads:
    ``source`` / ``engine`` / ``raw`` / ``mlx`` / ``git_head``.
    """

    probed = probe_engine_version(args)
    if not probed.get("available"):
        return None
    return {
        key: probed.get(key)
        for key in ("source", "engine", "raw", "mlx", "mlx_c", "git_head")
        if key in probed
    } | {
        key: probed[key]
        for key in ("git_head_reason", "mlx_reason")
        if key in probed
    }


#: The label a plain branch arm carries. A full-stack arm MUST NOT reuse it:
#: the report pools receipts by ``server``, so two different stacks under one
#: label become one mean and the difference the battery exists to measure
#: disappears into it.
DEFAULT_BRANCH_SERVER = "branch"

#: Profile names that mean "full stack" even without --full-stack.
FULL_STACK_PROFILE_NAMES = ("turbo-full-stack", "full-stack")

#: What to call the arm instead.
FULL_STACK_SERVER_LABEL = "branch-fullstack"

#: What a ``--server-defaults`` arm must be called. Same pooling argument as
#: the full-stack label: the report keys receipts on ``--server``, so a
#: defaults arm wearing the bare "branch" label would be averaged in with the
#: configured cells and the one number it exists to produce -- what a user who
#: types `mtplx serve` gets -- would vanish into the mean.
SERVER_DEFAULTS_LABEL = "branch-default"


def check_server_defaults(args: argparse.Namespace) -> str | None:
    """Refuse a defaults arm that is not actually running on defaults.

    The arm's whole claim is "nothing here was configured by the harness". So
    anything that configures it is refused loudly at parse time rather than
    silently ignored: a receipt that says ``server_defaults`` while a flags
    file was applied is worse than no receipt.

    ``--env`` is deliberately NOT refused. It is the operator's explicit last
    word, it is recorded separately from the harness's own blocks in
    ``cli_env_overrides``, and the opt-out arm needs it
    (``--env MTPLX_FABLE_DISABLE=all``).
    """

    if not args.server_defaults:
        return None
    if args.stack == "mlx-serve":
        return (
            "--server-defaults is an MTPLX option; mlx-serve is launched from "
            "its own argv builder and takes no profile or MTPLX_FABLE_* set"
        )
    conflicts = []
    if args.fable_flags_file:
        conflicts.append("--fable-flags-file")
    if args.full_stack:
        conflicts.append("--full-stack")
    if args.profile_explicit:
        conflicts.append(f"--profile {args.profile}")
    if conflicts:
        return (
            "--server-defaults measures what the SERVER arms on its own, so "
            f"it cannot be combined with {', '.join(conflicts)}: that would "
            "arm the stack by hand and then credit the server for it. Drop "
            "the flag(s), or drop --server-defaults."
        )
    if str(args.server) == DEFAULT_BRANCH_SERVER:
        return (
            "--server-defaults with --server "
            f"{DEFAULT_BRANCH_SERVER!r} would pool this arm's receipts with "
            "the configured branch cells and average away the only number it "
            f"exists to produce. Pass --server {SERVER_DEFAULTS_LABEL} (or "
            "another distinct label) and re-run."
        )
    return None


def full_stack_requested(args: argparse.Namespace) -> bool:
    """Either switch turns this into a full-stack arm."""

    if getattr(args, "server_defaults", False):
        # The server may well default TO the full stack -- that is the thing
        # this arm measures -- but the harness did not request it, and this
        # predicate is about what the invocation asked for.
        return False
    return bool(args.full_stack) or str(args.profile) in FULL_STACK_PROFILE_NAMES


def check_arm_label(args: argparse.Namespace) -> str | None:
    """Refuse a full-stack arm wearing the plain-branch label. None == ok."""

    if not full_stack_requested(args):
        return None
    if str(args.server) != DEFAULT_BRANCH_SERVER:
        return None
    why = (
        "--full-stack" if args.full_stack else f"--profile {args.profile}"
    )
    return (
        f"{why} makes this a FULL-STACK arm, but --server is the bare default "
        f"{DEFAULT_BRANCH_SERVER!r}. The report pools receipts by --server, so "
        "these cells would be averaged together with the plain-branch cells "
        "and the difference this arm exists to measure would vanish into the "
        f"mean. Pass --server {FULL_STACK_SERVER_LABEL} (or another distinct "
        "label) and re-run."
    )


def check_server_cwd(args: argparse.Namespace) -> str | None:
    """Refuse an MTPLX launch with no explicit --server-cwd. None == ok.

    The interpreter puts the CURRENT DIRECTORY on sys.path, so an MTPLX
    server launched without one imports whichever tree the shell happened to
    be sitting in. That is not hypothetical: it measured the wrong code once
    already. mlx-serve is a self-contained binary and is exempt.
    """

    if args.stack == "mlx-serve":
        return None
    if args.server_cwd:
        return None
    return (
        "--server-cwd is required for an MTPLX arm: the interpreter imports "
        "whichever mtplx tree sits on the current directory, so an inherited "
        "cwd silently measures the wrong code. Pass the worktree this arm is "
        "meant to serve (the same tree --server-python lives in)."
    )


def resolve_contexts(args: argparse.Namespace) -> list[int]:
    """The sizes this invocation will run, with the sanity gate applied."""

    contexts = [int(c) for c in (args.contexts or "").split(",") if c.strip()]
    if not contexts:
        contexts = [c for c in CONTEXT_BATTERY if c <= int(args.stop_after_context)]
    too_big = [c for c in contexts if c > int(args.stop_after_context)]
    if too_big:
        raise SystemExit(
            f"refusing contexts {too_big} above --stop-after-context "
            f"{args.stop_after_context}; raise it deliberately"
        )
    return contexts


def server_command(args: argparse.Namespace) -> list[str]:
    """The engine's argv. ONE definition, so the dry run cannot drift."""

    if args.stack == "mlx-serve":
        return build_mlxserve_argv(port=int(args.port), mtp=not args.mlxserve_no_mtp)
    return build_server_argv(
        python=args.server_python,
        port=int(args.port),
        log_level=args.server_log_level,
        profile=None if getattr(args, "server_defaults", False) else args.profile,
    )


def request_body(
    args: argparse.Namespace, prompt: Mapping[str, Any], seed: int | None
) -> dict[str, Any]:
    """The per-cell request, with the prompt reduced to its digest.

    Built by the SAME :func:`chat_body` the live call uses, so the plan and
    the wire cannot drift; only the ``messages`` array is swapped for the
    prompt's sha256 and length, because a 255K-token body would make the plan
    undiffable while the digest pins the text exactly.
    """

    body = chat_body(
        model_id=MLXSERVE_MODEL_ID if args.stack == "mlx-serve" else MODEL_ID,
        **cell_sampling(args, prompt, seed),
    )
    return parity_body(
        body,
        prompt_sha256=str(prompt["text_sha256"]),
        prompt_chars=len(str(prompt.get("text") or "")),
    )


def dry_run_tail(args: argparse.Namespace) -> list[str]:
    """New sections go HERE, after the plan.

    Appending rather than inserting keeps every earlier line at the same
    offset, so a dry run captured before this change still diffs cleanly
    against one captured after it.
    """

    probed = probe_engine_version(args)
    block = engine_version_block(args)
    lines = ["--- engine version ---"]
    if block:
        for key in ("source", "engine", "mlx", "mlx_c", "git_head"):
            if block.get(key) is not None:
                lines.append(f"{key}: {block[key]}")
        for key in ("git_head_reason", "mlx_reason"):
            if block.get(key):
                lines.append(f"{key}: {block[key]}")
    else:
        lines.append(f"UNRECORDED: {probed.get('reason')}")
    lines.append("--- pooling identity ---")
    lines.append(f"server-label: {args.server}")
    lines.append(f"full-stack-arm: {'yes' if full_stack_requested(args) else 'no'}")
    lines.append(
        "profile: "
        + ("-" if args.stack == "mlx-serve"
           else "SERVER DEFAULT (--profile omitted)" if args.server_defaults
           else str(args.profile))
    )
    lines.append(f"server-defaults: {'yes' if args.server_defaults else 'no'}")
    lines.append(f"server-cwd: {args.server_cwd or '-'}")
    return lines


DRY_RUN_SCHEMA = "server-cell-dry-run-v1"


def render_dry_run(args: argparse.Namespace) -> str:
    """Everything this invocation would do, and nothing it would start.

    Deterministic by construction: no clock, no pid, no receipt name, env and
    flag keys sorted, cells in plan order. Two runs of the same invocation
    produce byte-identical text, and three engines' dry runs concatenate into
    one auditable battery plan.
    """

    overrides = branch_env_overrides(
        stack=args.stack,
        full_stack=bool(args.full_stack),
        fable_flags=args.fable_flags["resolved"],
        server_defaults=bool(getattr(args, "server_defaults", False)),
    )
    cli_env = cli_env_overrides(args.env)
    argv = server_command(args)
    contexts = resolve_contexts(args) if args.mode == "run" else []
    lines = [
        f"### {DRY_RUN_SCHEMA}",
        f"mode: {args.mode}",
        f"engine: {args.server}",
        f"stack: {args.stack}",
        "profile: " + ("-" if args.stack == "mlx-serve"
                       else "SERVER DEFAULT (--profile omitted)"
                       if args.server_defaults else str(args.profile)),
        f"server-defaults: {'yes' if args.server_defaults else 'no'}",
        f"port: {int(args.port)}",
        f"server-python: {args.server_python or '-'}",
        f"server-cwd: {args.server_cwd or '-'}",
        f"server-log: {args.server_log or '-'}",
        f"server-log-level: {args.server_log_level or '-'}",
        f"model-id: {MLXSERVE_MODEL_ID if args.stack == 'mlx-serve' else MODEL_ID}",
        f"model-dir: {MLXSERVE_MODEL if args.stack == 'mlx-serve' else MODEL}",
        f"receipt-dir: {args.receipt_dir}",
        f"prompt-cache: {args.prompt_cache}",
        f"full-stack: {'yes' if args.full_stack else 'no'}",
        f"require-full-stack: {'yes' if args.require_full_stack else 'no'}",
        f"require-fan-max: {'yes' if args.require_fan_max else 'no'}",
        f"thermal-gate-max-c: {float(args.thermal_gate_max_c)}",
        f"timeout-s: {float(args.timeout_s)}",
        f"parity: body_sha256 covers every field but {'/'.join(PARITY_TRANSPORT_FIELDS)}"
        + ("  PARITY RISK: --no-seed drops the seed on this engine only" if args.no_seed else ""),
        f"contexts: {','.join(str(c) for c in contexts) or '-'}",
        f"cells: {args.cells}",
        f"repeats: {int(args.repeats)}",
        f"only-seed: {args.only_seed if args.only_seed is not None else '-'}",
        f"argv: {shlex.join(str(a) for a in argv)}",
        "--- field caveats ---",
        "\n".join(render_field_caveats(args.stack))
        or f"none recorded for stack={args.stack}",
        "--- fable flags ---",
        render_fable_flags(args.fable_flags),
        f"--- server env overrides ({len(overrides)}) ---",
    ]
    lines.extend(f"{key}={overrides[key]}" for key in sorted(overrides))
    lines.append(f"--- cli env overrides ({len(cli_env)}) ---")
    lines.extend(f"{key}={cli_env[key]}" for key in sorted(cli_env))

    if args.mode == "serve-hold":
        lines.append("--- plan ---")
        lines.append(
            "serve-hold: boot, wait for health + background warmup, scan the "
            "log for engagement, pre-read the model tables, send ONE untimed "
            "warm request, announce ready, hold, tear down. No timed cell."
        )
        lines.append(f"hold-seconds: {float(args.hold_seconds)}")
        lines.append(f"ready-file: {args.ready_file or '-'}")
        lines.append(
            "hold-command: "
            + (shlex.join(str(a) for a in args.hold_command)
               if args.hold_command else "- (sleep for the hold window)")
        )
        lines.append(
            "required: "
            + (", ".join(("full-stack engagement",))
               if args.require_full_stack else "nothing beyond health+warmup")
        )
        lines.extend(dry_run_tail(args))
        return "\n".join(lines) + "\n"

    if args.mode == "preflight":
        lines.append("--- plan ---")
        lines.append(
            "preflight: boot with logging ON, wait for health and background "
            "warmup, scan the log, shut down. No timed request."
        )
        lines.append(
            "required: "
            + ", ".join(
                OBSERVABLE_REQUIRED
                + (LOGGER_ONLY if args.full_stack else ())
                + ("warmup ladder all ok",)
            )
        )
        lines.extend(dry_run_tail(args))
        return "\n".join(lines) + "\n"

    cache_path = Path(args.prompt_cache)
    try:
        prompts = json.loads(cache_path.read_text())["prompts"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"dry run needs the prompt cache {cache_path}: {error}. Build it "
            "first with --mode build-prompts."
        ) from error
    plan = cell_plan(
        prompts=prompts,
        contexts=contexts,
        repeats=int(args.repeats),
        cells=args.cells,
        only_seed=args.only_seed,
    )
    if args.cells != "vanity":
        have = {
            int(p["target_tokens"])
            for p in prompts
            if p.get("cell") == "sweep"
        }
        for target in contexts:
            if target not in have:
                lines.append(
                    f"MISSING: no sweep prompt cached for target {target}; "
                    f"this size would silently contribute ZERO cells "
                    f"(cached: {','.join(str(t) for t in sorted(have))})"
                )
    lines.append(f"--- plan: {len(plan)} cell(s) ---")
    for index, item in enumerate(plan, start=1):
        prompt = item["prompt"]
        label = (
            "vanity"
            if prompt.get("cell") == "vanity"
            else point_name(("sweep", int(prompt["target_tokens"])))
        )
        body = request_body(args, prompt, item["seed"])
        lines.append(
            f"{index:04d} {args.server} {label} repeat={item['repeat']} "
            f"templated={prompt.get('templated_tokens')} "
            f"body_sha256={parity_digest(body)} "
            + json.dumps(body, sort_keys=True)
        )
    lines.extend(dry_run_tail(args))
    return "\n".join(lines) + "\n"


def run_preflight(args: argparse.Namespace) -> int:
    """Boot once with logging on, prove what installed, shut down.

    Separated from the timed cells on purpose: the logging handler that makes
    the install reports visible also turns on per-request logging, whose I/O
    would land inside the measurement. Run this ONCE per server
    configuration per session, not per cell.
    """

    overrides = branch_env_overrides(
        stack=args.stack,
        full_stack=bool(args.full_stack),
        fable_flags=(args.fable_flags["resolved"]),
        server_defaults=bool(getattr(args, "server_defaults", False)),
    )
    cli_env = cli_env_overrides(args.env)
    env = dict(os.environ)
    env.update(overrides)
    env.update(cli_env)

    print(
        "[server-cell] fable flags:\n"
        + textwrap.indent(render_fable_flags(args.fable_flags), "    "),
        flush=True,
    )

    probed = probe_engine_version(args)
    version_block = engine_version_block(args)
    version_reason = None if version_block else probed.get("reason")
    print(
        "[server-cell] engine version: "
        + (json.dumps(version_block, sort_keys=True) if version_block
           else f"UNRECORDED -- {version_reason}"),
        flush=True,
    )

    base_url = f"http://127.0.0.1:{int(args.port)}"
    argv = build_server_argv(
        python=args.server_python,
        port=int(args.port),
        log_level="INFO",
        profile=None if args.server_defaults else args.profile,
    )
    log_path = Path(args.server_log) if args.server_log else None
    handle = log_path.open("w") if log_path else subprocess.DEVNULL
    print(f"[server-cell] PREFLIGHT {args.server}: logging ON", flush=True)
    started = time.time()
    process = subprocess.Popen(
        argv, env=env, stdout=handle, stderr=subprocess.STDOUT,
        cwd=str(Path(args.server_cwd)) if args.server_cwd else None,
    )
    health: dict[str, Any] = {}
    engagement: dict[str, Any] = {}
    try:
        health = wait_for_health(base_url, process)
        wait_for_background_warmup(base_url)
        engagement = scan_engagement(log_path)
    finally:
        returncode = terminate_server(process)
        if handle is not subprocess.DEVNULL:
            handle.close()

    problems = require_preflight(engagement, full_stack=bool(args.full_stack))
    receipt = {
        "schema": "mtplx-server-cell-preflight-v1",
        "server": args.server,
        "stack": args.stack,
        "full_stack": bool(args.full_stack),
        "profile": args.profile,
        "server_cwd": str(args.server_cwd) if args.server_cwd else None,
        "engine_version": version_block,
        "engine_version_reason": version_reason,
        "server_argv": argv,
        "server_env_overrides": overrides,
        "cli_env_overrides": cli_env,
        "fable_flags": args.fable_flags,
        "server_log": str(log_path) if log_path else None,
        "started_epoch_s": started,
        "elapsed_s": time.time() - started,
        "health": health,
        "stack_engagement": engagement,
        # for the cross-arm comparison: what each arm actually EXPORTED for
        # the auto-stamped keys (absent == the server stamped it itself)
        "auto_armed_env": {
            k: env.get(k) for k in AUTO_ARMED_KEYS
        },
        "problems": problems,
        "passed": not problems,
        "server_returncode": returncode,
    }
    out_dir = Path(args.receipt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"preflight-{args.server}-{int(started)}.json"
    out.write_text(json.dumps(receipt, indent=1, default=str))
    print(f"[server-cell] wrote {out}", flush=True)
    if problems:
        print(
            "[server-cell] PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems),
            flush=True,
        )
        return 1
    print("[server-cell] PREFLIGHT PASSED", flush=True)
    return 0


#: Written into the ready file the moment a serve-hold server is usable.
SERVE_HOLD_READY_SCHEMA = "mtplx-server-cell-serve-hold-ready-v1"

#: The line an external caller can grep for on stdout.
SERVE_HOLD_READY_LINE = "[server-cell] SERVE-HOLD READY"


def recorded_profile(args: argparse.Namespace) -> str | None:
    """The profile this invocation actually put on the server's command line.

    ``None`` in the two cases where no ``--profile`` reaches the engine:
    mlx-serve, which has no MTPLX profile at all, and ``--server-defaults``,
    which omits the flag so the server picks its own. Naming the harness
    default in either case would put a lane name on a receipt whose engine
    never read one -- and for a defaults arm it would name the very thing the
    arm exists to find out.

    Kept next to the hold because the ready file is handed to an EXTERNAL
    caller: a quality eval that reads ``profile`` out of it and writes it into
    its own scoreboard must not learn a profile the server was never told.
    """

    if args.stack == "mlx-serve":
        return None
    if getattr(args, "server_defaults", False):
        return None
    return args.profile


def serve_hold_ready_payload(
    *,
    args: argparse.Namespace,
    argv: Sequence[str],
    pid: int,
    base_url: str,
    model_id: str,
    started_epoch: float,
    health: Mapping[str, Any],
    warmup: Mapping[str, Any],
    engagement: Mapping[str, Any],
) -> dict[str, Any]:
    """What a caller needs to send its own requests at this server.

    ``model_id`` is in here on purpose: the caller must not have to know which
    engine spells the model which way, and a caller that guesses wrong gets a
    404 in the middle of a long eval instead of at the ready gate.
    """

    return {
        "schema": SERVE_HOLD_READY_SCHEMA,
        "server": args.server,
        "stack": args.stack,
        "profile": recorded_profile(args),
        "base_url": base_url,
        "port": int(args.port),
        "model_id": model_id,
        "pid": int(pid),
        "server_argv": [str(a) for a in argv],
        "server_cwd": str(args.server_cwd) if args.server_cwd else None,
        "started_epoch_s": float(started_epoch),
        "ready_epoch_s": time.time(),
        "health": dict(health),
        "warmup": dict(warmup),
        "stack_engagement": dict(engagement),
    }


def run_serve_hold(args: argparse.Namespace) -> int:
    """Boot ONE engine, hold it up for a caller, tear it down. No timed cells.

    Why this mode exists
    --------------------
    Every other mode in this harness owns both ends: it boots a server, sends
    its own requests and shuts down. A quality eval (HumanEval pass@1) is a
    different client with its own scoring pipeline, and re-implementing the
    launch there would fork the definition of "the branch arm" -- the 20-key
    profile, the resolved MTPLX_FABLE_* set, the mlx-serve prefix-cache-off
    argv, the upstream interpreter and cwd. So the harness keeps ownership of
    the launch and lends the endpoint out instead.

    Shape of one hold
    -----------------
    1. build the SAME launch env as the battery (:func:`launch_env`) and the
       SAME argv (:func:`server_command`),
    2. wait for ``/health`` and the background warmup,
    3. scan the server log for stack engagement, and refuse the hold if
       ``--require-full-stack`` was asked for and the log does not prove it,
    4. pre-read the model's n-gram/PLE tables and send one untimed warm
       request -- exactly what :func:`run_battery` does before its first timed
       cell, so a caller's first request is not paying for graph compilation,
    5. announce readiness (stdout line and, with ``--ready-file``, a JSON file),
    6. run ``--hold-command`` while the server is up, or simply sleep
       ``--hold-seconds``,
    7. tear the server down and write a receipt.

    The receipt deliberately carries NO ``records`` key, so both report readers
    (``server_cell_report.load`` and :func:`load_records`) ignore it exactly as
    they ignore a preflight receipt.
    """

    env, overrides, cli_env = launch_env(args)

    print(
        "[server-cell] fable flags:\n"
        + textwrap.indent(render_fable_flags(args.fable_flags), "    "),
        flush=True,
    )
    probed = probe_engine_version(args)
    version_block = engine_version_block(args)
    version_reason = None if version_block else probed.get("reason")
    print(
        "[server-cell] engine version: "
        + (json.dumps(version_block, sort_keys=True) if version_block
           else f"UNRECORDED -- {version_reason}"),
        flush=True,
    )
    for line in render_field_caveats(args.stack):
        print(f"[server-cell] {line}", flush=True)

    base_url = f"http://127.0.0.1:{int(args.port)}"
    argv = server_command(args)
    model_id = MLXSERVE_MODEL_ID if args.stack == "mlx-serve" else MODEL_ID
    model_dir = MLXSERVE_MODEL if args.stack == "mlx-serve" else MODEL

    fans_start = fan_state()
    power_start = power_source()
    if args.require_fan_max and fans_start.get("available") and not fans_start.get("at_max"):
        raise SystemExit(
            "[server-cell] REFUSING to serve: fans are not at max "
            f"({fans_start.get('modes')}, rpm "
            f"{[f.get('actual_rpm') for f in fans_start.get('fans') or []]}). "
            "Every arm must share one cooling regime; run "
            "`sudo -n /usr/local/bin/thermalforge max` first."
        )

    log_path = Path(args.server_log) if args.server_log else None
    log_handle = log_path.open("w") if log_path else subprocess.DEVNULL
    ready_path = Path(args.ready_file) if args.ready_file else None
    if ready_path is not None and ready_path.exists():
        # A stale ready file from a previous hold would let a caller start
        # firing at a port nothing is listening on yet.
        ready_path.unlink()

    print(
        f"[server-cell] SERVE-HOLD {args.server}: "
        + shlex.join(str(a) for a in argv),
        flush=True,
    )
    started_epoch = time.time()
    process = subprocess.Popen(
        argv,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(Path(args.server_cwd)) if args.server_cwd else None,
    )
    health: dict[str, Any] = {}
    warmup: dict[str, Any] = {}
    engagement: dict[str, Any] = {"available": False, "reason": "not reached"}
    prewarm: list[dict[str, Any]] = []
    problems: list[str] = []
    hold: dict[str, Any] = {"ran": False}
    ready_epoch: float | None = None
    try:
        health = wait_for_health(base_url, process)
        warmup = wait_for_background_warmup(base_url)
        print(
            f"[server-cell] healthy: model={health.get('model')} "
            f"profile={(health.get('profile') or {}).get('name')} "
            f"mtp={health.get('mtp_enabled')} depth={health.get('depth')} "
            f"warmup={warmup.get('state')}",
            flush=True,
        )
        engagement = scan_engagement(log_path)
        print(
            "[server-cell] engagement " + json.dumps(engagement, sort_keys=True),
            flush=True,
        )
        if args.require_full_stack:
            problems = require_full_stack(engagement)
            if problems:
                raise SystemExit(
                    "[server-cell] REFUSING to hold this server: "
                    "--require-full-stack was asked for but the server did "
                    "not install it:\n  - " + "\n  - ".join(problems)
                )
            print("[server-cell] full-stack engagement VERIFIED", flush=True)

        prewarm = prewarm_model_tables(model_dir, args.stack)
        stream_chat(
            base_url=base_url, model_id=model_id, prompt="ping",
            max_tokens=8, temperature=0.0, top_p=1.0, top_k=None, seed=None,
            reasoning_effort=VANITY_REASONING_EFFORT,
            enable_thinking=VANITY_ENABLE_THINKING, timeout_s=600.0,
        )

        ready = serve_hold_ready_payload(
            args=args, argv=argv, pid=process.pid, base_url=base_url,
            model_id=model_id, started_epoch=started_epoch, health=health,
            warmup=warmup, engagement=engagement,
        )
        ready_epoch = float(ready["ready_epoch_s"])
        if ready_path is not None:
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a caller polling for the file never sees a
            # half-written one and never parses a truncated JSON body.
            staging = ready_path.with_name(ready_path.name + ".partial")
            staging.write_text(json.dumps(ready, indent=1, default=str))
            staging.replace(ready_path)
        print(
            f"{SERVE_HOLD_READY_LINE} {args.server} {base_url} "
            f"model={model_id} pid={process.pid} "
            f"after={ready_epoch - started_epoch:.0f}s",
            flush=True,
        )

        hold = hold_the_server(
            command=args.hold_command,
            hold_seconds=float(args.hold_seconds),
            process=process,
            server=args.server,
        )
    finally:
        returncode = terminate_server(process)
        if log_handle is not subprocess.DEVNULL:
            log_handle.close()
        if ready_path is not None and ready_path.exists():
            # The endpoint is gone; leaving the file would advertise a server
            # that no longer exists.
            ready_path.unlink()

    receipt = {
        "schema": "mtplx-server-cell-serve-hold-v1",
        "server": args.server,
        "stack": args.stack,
        "full_stack": bool(args.full_stack),
        # mlx-serve has no MTPLX profile and --server-defaults omits the flag;
        # recording the unused default would put a lane name on a receipt
        # whose engine never read one.
        "profile": recorded_profile(args),
        "server_cwd": str(args.server_cwd) if args.server_cwd else None,
        "engine_version": version_block,
        "engine_version_reason": version_reason,
        "server_argv": [str(a) for a in argv],
        "server_env_overrides": overrides,
        "cli_env_overrides": cli_env,
        "fable_flags": args.fable_flags,
        "server_log": str(log_path) if log_path else None,
        "ready_file": str(ready_path) if ready_path else None,
        "model_id": model_id,
        "base_url": base_url,
        "started_epoch_s": started_epoch,
        "ready_after_s": (ready_epoch - started_epoch) if ready_epoch else None,
        "elapsed_s": time.time() - started_epoch,
        "health": health,
        "warmup": warmup,
        "stack_engagement": engagement,
        "prewarm": prewarm,
        "auto_armed_env": {k: env.get(k) for k in AUTO_ARMED_KEYS},
        "require_full_stack": bool(args.require_full_stack),
        "require_full_stack_problems": problems,
        "hold": hold,
        "session_fans_start": fans_start,
        "session_fans_end": fan_state(),
        "session_power_start": power_start,
        "session_power_end": power_source(),
        "server_returncode": returncode,
    }
    out_dir = Path(args.receipt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"serve-hold-{args.server}-{int(started_epoch)}.json"
    out.write_text(json.dumps(receipt, indent=1, default=str))
    print(f"[server-cell] wrote {out}", flush=True)
    code = int(hold.get("returncode") or 0)
    held = hold.get("held_s")
    print(
        f"[server-cell] SERVE-HOLD {args.server} done: "
        f"hold={hold.get('mode')} rc={code} "
        f"held={held:.0f}s" if isinstance(held, (int, float))
        else f"[server-cell] SERVE-HOLD {args.server} done: rc={code}",
        flush=True,
    )
    if hold.get("error"):
        print(f"[server-cell] hold error: {hold['error']}", flush=True)
    return code


def hold_the_server(
    *,
    command: Sequence[str] | None,
    hold_seconds: float,
    process: subprocess.Popen[Any],
    server: str,
) -> dict[str, Any]:
    """Run ``command`` while the server is up, or sleep for the hold window.

    With a command, ``--hold-seconds`` is the CAP: the command is killed and
    the hold ends if it outruns the window, so a wedged client can never pin
    the GPU lock past the guarded window that owns it.

    Without one, the mode is a plain sleep for a caller driving the endpoint
    from outside this process; the sleep polls so a server that dies is noticed
    in seconds rather than at the end of the window.
    """

    started = time.monotonic()
    if not command:
        deadline = started + hold_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return {
                    "ran": False,
                    "mode": "sleep",
                    "held_s": time.monotonic() - started,
                    "returncode": 1,
                    "error": (
                        f"server exited with {process.returncode} during the "
                        "hold window"
                    ),
                }
            time.sleep(5.0)
        return {
            "ran": False,
            "mode": "sleep",
            "held_s": time.monotonic() - started,
            "returncode": 0,
        }

    argv = [str(a) for a in command]
    print(
        f"[server-cell] hold command ({server}): " + shlex.join(argv),
        flush=True,
    )
    timed_out = False
    try:
        completed = subprocess.run(argv, timeout=hold_seconds, check=False)
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = 124
    except OSError as error:
        return {
            "ran": False,
            "mode": "command",
            "argv": argv,
            "held_s": time.monotonic() - started,
            "returncode": 127,
            "error": f"could not run the hold command: {error}",
        }
    return {
        "ran": True,
        "mode": "command",
        "argv": argv,
        "held_s": time.monotonic() - started,
        "returncode": returncode,
        "timed_out": timed_out,
    }


def compare_branch_preflights(receipt_dir: Path) -> tuple[int, str]:
    """Both branch arms must arm the SAME four auto-stamped M4 routes.

    Neither arm exports them any more, so they should reach them by one code
    path. Asserting it turns a precedence argument into a measurement: a
    divergence here is a real signal, not something to reason about.
    """

    arms: dict[str, dict[str, Any]] = {}
    for path in sorted(receipt_dir.glob("preflight-*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("stack") != "branch":
            continue
        markers = (payload.get("stack_engagement") or {}).get("markers") or {}
        arms[str(payload.get("server"))] = {
            "m4": {k: bool(markers.get(k)) for k in LOGGER_ONLY},
            "exported": payload.get("auto_armed_env") or {},
            "receipt": path.name,
        }
    if len(arms) < 2:
        return 0, (
            f"only {len(arms)} branch preflight(s) found; nothing to compare"
        )
    names = sorted(arms)
    first = arms[names[0]]["m4"]
    lines = [f"branch arms compared: {', '.join(names)}"]
    for name in names:
        exported = {k: v for k, v in arms[name]["exported"].items() if v}
        lines.append(
            f"  {name}: m4={arms[name]['m4']} exported_auto_keys={exported or 'none'}"
        )
    bad = [n for n in names if arms[n]["m4"] != first]
    if bad:
        return 1, "\n".join(
            lines + [f"MISMATCH: {bad} differ from {names[0]} on the M4 routes"]
        )
    leaked = sorted(
        {k for n in names for k, v in arms[n]["exported"].items() if v}
    )
    if leaked:
        lines.append(
            f"WARNING: auto-stamped keys were EXPORTED ({leaked}); both arms "
            "should reach them by the server's setdefault path"
        )
    return 0, "\n".join(lines)


def launch_env(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """The environment a served engine is launched with: ``(env, overrides, cli_env)``.

    ONE definition, shared by :func:`run_battery` and :func:`run_serve_hold`,
    so a server held up for an external caller is launched with byte-identical
    environment to the one the battery times. Two constructions would let a
    quality eval and a speed battery disagree about which lane they measured.

    The memory-knob ceiling is enforced here rather than at the call site for
    the same reason: it is a property of launching this engine on this box, not
    of the mode that launches it.
    """

    overrides = branch_env_overrides(
        stack=args.stack,
        full_stack=bool(args.full_stack),
        fable_flags=(args.fable_flags["resolved"]),
        server_defaults=bool(getattr(args, "server_defaults", False)),
    )
    cli_env = cli_env_overrides(args.env)
    env = dict(os.environ)
    env.update(overrides)
    env.update(cli_env)
    for key in ("MTPLX_MEMORY_LIMIT_BYTES", "MTPLX_WIRED_LIMIT_BYTES"):
        if int(env.get(key, 0)) > MAX_SAFE_LIMIT_BYTES:
            raise SystemExit(
                f"{key}={env[key]} exceeds the documented {MAX_SAFE_LIMIT_BYTES} "
                "byte ceiling for this 128 GiB machine; raising it leaves too "
                "little for macOS (documented kernel-panic regime)"
            )
    return env, overrides, cli_env


def run_battery(args: argparse.Namespace) -> int:
    cache_path = Path(args.prompt_cache)
    payload = json.loads(cache_path.read_text())
    prompts = payload["prompts"]

    contexts = resolve_contexts(args)

    env, overrides, cli_env = launch_env(args)

    probed = probe_engine_version(args)
    version_block = engine_version_block(args)
    version_reason = None if version_block else probed.get("reason")
    print(
        "[server-cell] engine version: "
        + (json.dumps(version_block, sort_keys=True) if version_block
           else f"UNRECORDED -- {version_reason}"),
        flush=True,
    )

    base_url = f"http://127.0.0.1:{int(args.port)}"
    argv = server_command(args)
    if args.stack == "mlx-serve":
        model_id = MLXSERVE_MODEL_ID
        model_dir = MLXSERVE_MODEL
        metrics_url: str | None = f"{base_url}/metrics.json"
        metric_keys: tuple[str, ...] = MLXSERVE_METRIC_KEYS
        props_url: str | None = f"{base_url}{MLXSERVE_PROPS_PATH}"
    else:
        model_id = MODEL_ID
        model_dir = MODEL
        metrics_url = None
        metric_keys = ()
        props_url = None
    print(f"[server-cell] starting {args.server}: {shlex.join(str(a) for a in argv)}", flush=True)
    print(
        "[server-cell] fable flags:\n"
        + textwrap.indent(render_fable_flags(args.fable_flags), "    "),
        flush=True,
    )
    for line in render_field_caveats(args.stack):
        print(f"[server-cell] {line}", flush=True)
    env_report = report_unvalidated_env(env)

    log_path = Path(args.server_log) if args.server_log else None
    log_handle = log_path.open("w") if log_path else subprocess.DEVNULL
    started_epoch = time.time()
    session_fans_start = fan_state()
    session_power_start = power_source()
    process = subprocess.Popen(
        argv,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(Path(args.server_cwd)) if args.server_cwd else None,
    )
    records: list[dict[str, Any]] = []
    health: dict[str, Any] = {}
    warmup: dict[str, Any] = {}
    engagement: dict[str, Any] = {"available": False, "reason": "not reached"}
    prewarm: list[dict[str, Any]] = []
    try:
        health = wait_for_health(base_url, process)
        warmup = wait_for_background_warmup(base_url)
        print(
            f"[server-cell] healthy: model={health.get('model')} "
            f"profile={(health.get('profile') or {}).get('name')} "
            f"mtp={health.get('mtp_enabled')} depth={health.get('depth')} "
            f"warmup={warmup.get('state')}",
            flush=True,
        )

        # Proof of what actually installed, read from the server's own log
        # before a single timed request. Env is not evidence: most restack
        # keys are bare os.environ.get with defaults, so a wrong one is
        # silently ignored rather than refused.
        engagement = scan_engagement(log_path)
        print(
            "[server-cell] engagement " + json.dumps(engagement, sort_keys=True),
            flush=True,
        )
        if args.require_full_stack:
            problems = require_full_stack(engagement)
            if problems:
                raise SystemExit(
                    "[server-cell] REFUSING to time this cell: "
                    "--require-full-stack was asked for but the server did "
                    "not install it:\n  - " + "\n  - ".join(problems)
                )
            print("[server-cell] full-stack engagement VERIFIED", flush=True)

        # Pre-read the n-gram/PLE table so every arm starts from the same
        # page-cache regime as the in-process runs.
        prewarm = prewarm_model_tables(model_dir, args.stack)

        # One untimed warm request so the first TIMED point is not paying for
        # graph compilation. Its result is discarded on purpose.
        stream_chat(
            base_url=base_url, model_id=model_id, prompt="ping",
            max_tokens=8, temperature=0.0, top_p=1.0, top_k=None, seed=None,
            reasoning_effort=VANITY_REASONING_EFFORT,
            enable_thinking=VANITY_ENABLE_THINKING, timeout_s=600.0,
        )

        plan = cell_plan(
            prompts=prompts,
            contexts=contexts,
            repeats=int(args.repeats),
            cells=args.cells,
            only_seed=args.only_seed,
        )
        for index, item in enumerate(plan):
            prompt = item["prompt"]
            is_vanity = prompt.get("cell") == "vanity"
            thermal = wait_for_temperature(float(args.thermal_gate_max_c))
            label = (
                "vanity" if is_vanity else point_name(("sweep", int(prompt["target_tokens"])))
            )
            print(
                f"[server-cell] {index + 1}/{len(plan)} {args.server} {label} "
                f"seed={item['seed']} repeat={item['repeat']} "
                f"(ready {thermal['ready_c']:.1f}C)",
                flush=True,
            )
            power = power_source()
            fans = fan_state()
            if args.require_fan_max and fans.get("available") and not fans.get("at_max"):
                raise SystemExit(
                    "[server-cell] REFUSING to time: fans are not at max "
                    f"({fans.get('modes')}, rpm "
                    f"{[f.get('actual_rpm') for f in fans.get('fans') or []]}). "
                    "Every cell of every engine must share one cooling regime; "
                    "run `sudo -n /usr/local/bin/thermalforge max` first."
                )
            if power.get("available") and not power.get("on_ac"):
                raise SystemExit(
                    "[server-cell] REFUSING to time on BATTERY power: the SoC "
                    "is throttled and every number in this cell would be "
                    f"invalid (pmset: {power.get('raw')})"
                )
            wake = wake_request(base_url, model_id)
            if not wake["ok"]:
                print(f"    wake request failed: {wake['error']}", flush=True)
            vm_before = vm_stat_pages()
            props_peak: int | None = None
            props_before: int | None = None
            if props_url:
                try:
                    props_before = _dig(
                        http_get_json(props_url, timeout=5.0), ("peak_bytes",)
                    )
                except Exception:  # noqa: BLE001
                    props_before = None
            sampling = cell_sampling(args, prompt, item["seed"])
            log_offset = log_path.stat().st_size if (log_path and log_path.exists()) else 0
            with RssSampler(process.pid) as sampler, MacmonTrace() as trace, \
                    MetricsPoller(metrics_url, metric_keys) as metrics, \
                    ProcSampler(process.pid) as proc, VmStatSampler() as vmstat:
                try:
                    call = stream_chat(
                        base_url=base_url,
                        model_id=model_id,
                        prompt=prompt["text"],
                        timeout_s=float(args.timeout_s),
                        **sampling,
                    )
                except Exception as error:  # noqa: BLE001 - recorded, not raised
                    call = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                if props_url:
                    try:
                        props = http_get_json(props_url, timeout=5.0)
                        props_peak = _dig(props, ("peak_bytes",))
                    except Exception:  # noqa: BLE001
                        props_peak = None
            vm_after = vm_stat_pages()
            page = vm_after.get("_page_bytes") or 16384
            residency = {
                key.replace("Pages ", "").replace(" ", "_"): (
                    (vm_after.get(key, 0) - vm_before.get(key, 0)) * page
                )
                for key in (
                    "Pages free", "Pages active", "Pages inactive",
                    "Pages speculative", "Pages wired down", "Pageins",
                )
                if key in vm_after and key in vm_before
            }
            record = build_record(
                server=args.server,
                cell="vanity" if is_vanity else "sweep",
                target_tokens=int(prompt["target_tokens"]),
                seed=item["seed"],
                repeat=int(item["repeat"]),
                call=call,
                rss_bytes=sampler.max_rss_bytes,
                thermal=thermal,
                prompt_sha256=str(prompt["text_sha256"]),
                prefer_server_timings=not args.client_timings_only,
            )
            record["templated_tokens_expected"] = prompt.get("templated_tokens")
            # Two-sided parity. request_parity/its digest say what this cell
            # ASKED for, canonicalised so the three engines are comparable;
            # response_parity says what the engine DID with it. Both are
            # derived from the real call, not from the plan.
            # A request that raised before the server answered records no
            # wire body, and hashing {} would give every failed cell the same
            # meaningless digest. Fall back to the PLANNED body and say so,
            # so a failed row still states what it was going to ask for.
            wire = record.get("request_body")
            canonical = parity_body(
                wire if wire else chat_body(model_id=model_id, **sampling),
                prompt_sha256=str(prompt["text_sha256"]),
                prompt_chars=len(str(prompt.get("text") or "")),
            )
            record["request_parity"] = canonical
            record["request_parity_source"] = "wire" if wire else "planned"
            record["request_body_sha256"] = parity_digest(canonical)
            record["response_parity"] = response_parity(call, sampling)
            record["rss_samples"] = sampler.samples
            record["wake_request"] = wake
            record["power"] = power
            record["fans"] = fans
            record["thermal_trace"] = trace.summarize(
                split_at=call.get("first_delta_at")
            )
            record["server_metrics_max"] = dict(metrics.maxima)
            record["server_metrics_samples"] = metrics.samples
            # Memory precedence, best comparable source first. ps-rss is LAST
            # because it does not count MLX unified-memory buffers at all
            # (it read 49-53 GB against a server reporting 77 GB).
            record["props_peak_bytes_before"] = props_before
            record["props_peak_bytes_after"] = props_peak
            if props_peak and props_before:
                # peak is monotone on both engines; the delta is the honest
                # per-request growth.
                record["props_peak_growth_bytes"] = int(props_peak) - int(props_before)
            if props_peak:
                record["peak_memory_bytes"] = int(props_peak)
                record["peak_memory_source"] = "server:/props peak_bytes"
            elif record.get("peak_memory_source") == "client:ps_rss" and (
                proc.max_footprint
            ):
                record["peak_memory_bytes"] = int(proc.max_footprint)
                record["peak_memory_source"] = "client:phys_footprint"
            elif not record.get("peak_memory_bytes") and metrics.maxima.get(
                "mlx_serve:memory_mb"
            ):
                # Prometheus reports MEBIbytes, not megabytes.
                record["peak_memory_bytes"] = int(
                    metrics.maxima["mlx_serve:memory_mb"] * 1024 * 1024
                )
                record["peak_memory_source"] = "server:/metrics memory_mb (MiB)"
            if record.get("peak_memory_bytes"):
                record["peak_memory_gb"] = record["peak_memory_bytes"] / 1e9
            record["proc_footprint"] = proc.summary()
            record["peak_footprint_bytes"] = proc.max_footprint
            record["peak_footprint_gb"] = (
                proc.max_footprint / 1e9 if proc.max_footprint else None
            )
            record["residency"] = residency
            record["vm_stat"] = vmstat.summary()
            pressure_events = scan_pressure_events(log_path, log_offset)
            record["memory_pressure_event"] = bool(
                pressure_events.get("event_count")
            )
            record["memory_pressure_events"] = pressure_events
            record["memory_pressure"] = memory_pressure_level()
            records.append(record)
            if record["ok"]:
                print(
                    f"    prefill {_fmt(record['prefill_time_s'])}s "
                    f"({_fmt(record['prefill_tok_s'], 1)} tok/s) "
                    f"ttft {_fmt(record['ttft_s'])}s "
                    f"decode {_fmt(record['decode_tok_s'])} tok/s "
                    f"peak {_fmt(record['peak_memory_gb'])} GB "
                    f"[{record['peak_memory_source']}] "
                    f"gen {record['completion_tokens']} "
                    f"({record['finish_reason']})",
                    flush=True,
                )
            else:
                print(f"    FAILED: {record['error']}", flush=True)
    finally:
        returncode = terminate_server(process)
        if log_handle is not subprocess.DEVNULL:
            log_handle.close()

    receipt_dir = Path(args.receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "mtplx-server-cell-v1",
        "server": args.server,
        "stack": args.stack,
        # The three that decide whether two receipts may be POOLED. Only the
        # preflight carried them before, so a full-stack run and a plain
        # branch run were indistinguishable once the logs were gone.
        "full_stack": bool(args.full_stack),
        "profile": args.profile,
        "server_cwd": str(args.server_cwd) if args.server_cwd else None,
        "engine_version": version_block,
        "engine_version_reason": version_reason,
        "server_python": str(args.server_python),
        "server_argv": argv,
        "server_env_overrides": overrides,
        "cli_env_overrides": cli_env,
        "fable_flags": args.fable_flags,
        "engine_field_caveats": [dict(c) for c in engine_field_caveats(args.stack)],
        "mlxserve_source_root": str(MLXSERVE_SOURCE_ROOT),
        "ngram_prewarm": prewarm,
        "server_log": str(log_path) if log_path else None,
        # session header: the cooling/power regime this whole cell ran under
        "session_fans_start": session_fans_start,
        "session_fans_end": fan_state(),
        "session_power_start": session_power_start,
        "stack_engagement": engagement,
        "env_validation": env_report,
        "model_dir": str(model_dir),
        "model_id": model_id,
        "server_returncode": returncode,
        "started_epoch_s": started_epoch,
        "elapsed_s": time.time() - started_epoch,
        "health": health,
        "warmup": warmup,
        "prompt_cache": str(cache_path),
        "contexts": contexts,
        "records": records,
    }
    out = receipt_dir / f"{args.server}-{int(started_epoch)}.json"
    out.write_text(json.dumps(receipt, indent=1, default=str))
    print(f"[server-cell] wrote {out}", flush=True)
    return 0 if all(r.get("ok") for r in records) else 1


# ---------------------------------------------------------------------------
# Receipts / CLI
# ---------------------------------------------------------------------------


def load_records(receipt_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(receipt_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, Mapping) and "records" in payload:
            records.extend(dict(r) for r in payload["records"])
    return records


def summarize_fastest_of_seeds(
    records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per (cell, target): the FASTEST seed and the min-max spread.

    Per the ledger rule "report fastest of seeds": tables report the fastest
    seed (max tok/s, min TTFT) with the min-max range, never the mean.
    """

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for record in records:
        if not record.get("ok"):
            continue
        key = (str(record.get("cell")), int(record.get("target_tokens") or 0))
        groups.setdefault(key, []).append(record)

    summaries: list[dict[str, Any]] = []
    for (cell, target), rows in sorted(
        groups.items(), key=lambda kv: (kv[0][0] != "vanity", kv[0][1])
    ):
        def _vals(field: str) -> list[tuple[float, Any]]:
            out = []
            for r in rows:
                v = r.get(field)
                if v is not None:
                    out.append((float(v), r.get("seed")))
            return out

        decode = _vals("decode_tok_s")
        prefill = _vals("prefill_tok_s")
        ttft = _vals("ttft_s")
        best_decode = max(decode, default=(None, None))
        best_prefill = max(prefill, default=(None, None))
        best_ttft = min(ttft, default=(None, None))
        summaries.append(
            {
                "cell": cell,
                "target_tokens": target,
                "point": point_name((cell, target)),
                "seeds": [r.get("seed") for r in rows],
                "decode_tok_s_max": best_decode[0],
                "decode_tok_s_max_seed": best_decode[1],
                "decode_tok_s_range": [
                    min(v for v, _ in decode) if decode else None,
                    max(v for v, _ in decode) if decode else None,
                ],
                "prefill_tok_s_max": best_prefill[0],
                "prefill_tok_s_max_seed": best_prefill[1],
                "prefill_tok_s_range": [
                    min(v for v, _ in prefill) if prefill else None,
                    max(v for v, _ in prefill) if prefill else None,
                ],
                "ttft_s_min": best_ttft[0],
                "ttft_s_min_seed": best_ttft[1],
                "ttft_s_range": [
                    min(v for v, _ in ttft) if ttft else None,
                    max(v for v, _ in ttft) if ttft else None,
                ],
            }
        )
    return summaries


def run_cells_remote(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    """Client-only sweep against an ALREADY-RUNNING endpoint (--base-url).

    Starts no server and holds no GPU lock: the gpu_window STEP that calls this
    owns the serve lifecycle. Reuses the exact request/parse/receipt machinery
    the Qwen battery uses (stream_chat / build_record / parity), so the wire
    body and the receipt schema are identical.
    """

    if not args.base_url:
        parser.error("--mode cells requires --base-url")
    if not args.served_model_id:
        parser.error("--mode cells requires --served-model-id")

    cache_path = Path(args.prompt_cache)
    payload = json.loads(cache_path.read_text())
    prompts = payload["prompts"]
    contexts = resolve_contexts(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    base_url = str(args.base_url)
    model_id = str(args.served_model_id)
    family = args.model_family_resolved
    _dsv41_thinking = deepseek_v41_thinking_settings(args)
    started_epoch = time.time()

    print(
        f"[server-cell] cells (client-only) base_url={base_url} "
        f"model_id={model_id} family={family}",
        flush=True,
    )

    if not args.no_warm:
        try:
            stream_chat(
                base_url=base_url,
                model_id=model_id,
                prompt="ping",
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                top_k=None,
                seed=None,
                reasoning_effort=None,
                enable_thinking=None,
                timeout_s=120.0,
            )
            print("[server-cell] warm ping ok", flush=True)
        except Exception as error:  # noqa: BLE001 - warm-up is best effort
            print(f"[server-cell] warm ping skipped: {error}", flush=True)

    plan = cell_plan(
        prompts=prompts,
        contexts=contexts,
        repeats=int(args.repeats),
        cells=args.cells,
        only_seed=args.only_seed,
        seeds=seeds,
    )
    records: list[dict[str, Any]] = []
    for index, item in enumerate(plan):
        prompt = item["prompt"]
        is_vanity = prompt.get("cell") == "vanity"
        label = "vanity" if is_vanity else point_name(
            ("sweep", int(prompt["target_tokens"]))
        )
        print(
            f"[server-cell] {index + 1}/{len(plan)} {label} seed={item['seed']} "
            f"repeat={item['repeat']}",
            flush=True,
        )
        sampling = cell_sampling(args, prompt, item["seed"])
        try:
            call = stream_chat(
                base_url=base_url,
                model_id=model_id,
                prompt=prompt["text"],
                timeout_s=float(args.timeout_s),
                **sampling,
            )
        except Exception as error:  # noqa: BLE001 - recorded, not raised
            call = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        record = build_record(
            server=args.server,
            cell="vanity" if is_vanity else "sweep",
            target_tokens=int(prompt["target_tokens"]),
            seed=item["seed"],
            repeat=int(item["repeat"]),
            call=call,
            rss_bytes=None,
            thermal=None,
            prompt_sha256=str(prompt["text_sha256"]),
            prefer_server_timings=not args.client_timings_only,
        )
        record["templated_tokens_expected"] = prompt.get("templated_tokens")
        wire = record.get("request_body")
        canonical = parity_body(
            wire if wire else chat_body(model_id=model_id, **sampling),
            prompt_sha256=str(prompt["text_sha256"]),
            prompt_chars=len(str(prompt.get("text") or "")),
        )
        record["request_parity"] = canonical
        record["request_parity_source"] = "wire" if wire else "planned"
        record["request_body_sha256"] = parity_digest(canonical)
        record["response_parity"] = response_parity(call, sampling)
        records.append(record)
        if record["ok"]:
            reasoning_tokens = record["response_parity"].get("reasoning_tokens")
            reasoning_chars = record.get("reasoning_chars") or 0
            think = (
                f"reasoning {reasoning_tokens} tok "
                if reasoning_tokens
                else (f"reasoning {reasoning_chars} chars " if reasoning_chars else "")
            )
            print(
                f"    prefill {_fmt(record['prefill_time_s'])}s "
                f"({_fmt(record['prefill_tok_s'], 1)} tok/s) "
                f"ttft {_fmt(record['ttft_s'])}s "
                f"decode {_fmt(record['decode_tok_s'])} tok/s "
                f"wall {_fmt(record['wall_s'])}s "
                f"gen {record['completion_tokens']} tok {think}"
                f"({record['finish_reason']})",
                flush=True,
            )
        else:
            print(f"    FAILED: {record['error']}", flush=True)

    receipt_dir = Path(args.receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_fastest_of_seeds(records)
    receipt = {
        "schema": "mtplx-server-cell-v1",
        "server": args.server,
        "stack": "remote",
        "mode": "cells",
        "model_family": family,
        "served_model_id": model_id,
        "template_settings": template_settings_for_family(
            family,
            reasoning_effort=args.reasoning,
            dsv41_enable_thinking=_dsv41_thinking[0],
            dsv41_reasoning_effort=_dsv41_thinking[1],
        ),
        "base_url": base_url,
        "prompt_cache": str(cache_path),
        "prompt_cache_context_sha256": payload.get("context_sha256"),
        "contexts": contexts,
        "seeds": seeds,
        "max_tokens": int(args.max_tokens),
        "sampler": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
        },
        "started_epoch_s": started_epoch,
        "elapsed_s": time.time() - started_epoch,
        "fastest_of_seeds": summary,
        "records": records,
    }
    # Append-only: stamp-suffixed, never overwrites a prior measurement.
    out = receipt_dir / f"{args.server}-cells-{int(started_epoch)}.json"
    out.write_text(json.dumps(receipt, indent=1, default=str))
    print(f"[server-cell] wrote {out}", flush=True)

    print("[server-cell] fastest-of-seeds (max tok/s, min TTFT; range in []):", flush=True)
    for entry in summary:
        print(
            f"    {entry['point']:>7}: decode {_fmt(entry['decode_tok_s_max'])} tok/s "
            f"(seed {entry['decode_tok_s_max_seed']}, range "
            f"{_fmt(entry['decode_tok_s_range'][0])}-{_fmt(entry['decode_tok_s_range'][1])}) "
            f"prefill {_fmt(entry['prefill_tok_s_max'], 1)} tok/s "
            f"ttft {_fmt(entry['ttft_s_min'])}s "
            f"(range {_fmt(entry['ttft_s_range'][0])}-{_fmt(entry['ttft_s_range'][1])})",
            flush=True,
        )
    return 0 if records and all(r.get("ok") for r in records) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "build-prompts", "build-prompt-ids", "cells", "preflight",
            "preflight-compare", "run", "serve-hold", "report"
        ),
        default="run",
    )
    parser.add_argument("--server", default="branch")
    parser.add_argument(
        "--stack", choices=("branch", "upstream", "mlx-serve"), default="branch"
    )
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument(
        "--server-defaults",
        action="store_true",
        help=(
            "launch the MTPLX server on ITS OWN defaults: no --profile in the "
            "argv at all, and no harness env beyond the two memory caps (and "
            "MTPLX_SESSION_BANK_MAX_BYTES=auto, which is the server's own "
            "sizing default). No BRANCH_BASE_ENV, no FULL_STACK_ENV, no "
            "MTPLX_FABLE_* set. This is the only arm that answers 'what does "
            "a user who types `mtplx serve` actually get', so it refuses "
            "--profile/--full-stack/--fable-flags-file and requires a "
            "distinct --server label. --env is still honoured: it is the "
            "operator's explicit word and is recorded separately."
        ),
    )
    parser.add_argument(
        "--require-fan-max",
        action="store_true",
        help=(
            "refuse to time a cell unless every fan is within 5%% of its own "
            "max RPM. Cooling regime must be constant across arms and "
            "sessions or a thermal difference reads as an engine difference."
        ),
    )
    parser.add_argument(
        "--mlxserve-no-mtp",
        action="store_true",
        help=(
            "run mlx-serve as pure AR (--no-mtp). Control cell for the "
            "runtime_disabled finding: its controller switches speculation "
            "off mid-cell, so a measured AR baseline says what that costs."
        ),
    )
    parser.add_argument(
        "--profile",
        default="turbo",
        help=(
            "server profile. W61 adds 'turbo-full-stack' (alias 'full-stack') "
            "= turbo union the 20-key block with the same driver-wins "
            "conflicts; use it once merged instead of exporting the block."
        ),
    )
    parser.add_argument(
        "--server-log-level",
        default=None,
        help=(
            "install a stderr logging handler in the server so the "
            "[qwen4-fixed-M4-verify] / [qwen4-M4-stage3] / "
            "[qwen4-compiled-MTP-prepare] install reports become visible. "
            "The server configures no handler of its own, so these are "
            "otherwise unobservable. Use for a PREFLIGHT launch, not for "
            "timed cells: it also enables per-request logging whose I/O "
            "would perturb the measurement."
        ),
    )
    parser.add_argument(
        "--full-stack",
        action="store_true",
        help=(
            "apply FULL_STACK_ENV (the restack the ABBA driver sets via "
            "flags) and require proof it installed"
        ),
    )
    parser.add_argument(
        "--require-full-stack",
        action="store_true",
        help=(
            "abort before the first timed request unless the server log "
            "proves frspec installed at n=65536, an M4 route installed, and "
            "an all-ok warmup ladder"
        ),
    )
    parser.add_argument("--server-python", default=None)
    parser.add_argument("--server-cwd", default=None)
    parser.add_argument("--server-log", default=None)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=900.0,
        help=(
            "--mode serve-hold only. How long the server stays up. With "
            "--hold-command this is the CAP on the command, so a wedged "
            "client cannot pin the GPU lock past the guarded window."
        ),
    )
    parser.add_argument(
        "--ready-file",
        default=None,
        help=(
            "--mode serve-hold only. JSON file written (atomically) once the "
            "endpoint is healthy, warm and -- with --require-full-stack -- "
            "proven, and removed when it goes away. Carries base_url and the "
            "engine's own model_id so a caller never has to guess either."
        ),
    )
    parser.add_argument(
        "--hold-command",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="ARG",
        help=(
            "--mode serve-hold only. Run this argv while the server is up and "
            "return its exit code; the server is torn down when it finishes. "
            "Consumes the REST of the command line, so it must come LAST."
        ),
    )
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--contexts", default=None)
    parser.add_argument("--seeds", default=",".join(str(s) for s in PRODUCTION_SEEDS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cells", choices=("both", "vanity", "sweep"), default="both")
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--reasoning", default="xhigh")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument(
        "--fable-flags-file",
        action="extend",
        nargs="+",
        metavar="PATH",
        default=[],
        dest="fable_flags_file",
        help=(
            "file(s) of KEY=VALUE lines holding the branch arm's retained "
            "MTPLX_FABLE_* set. Repeatable and multi-valued; later files win "
            "and the shadowed entry is recorded. The set merges OVER the "
            "derived family env and UNDER --env. Given at all, it REPLACES "
            "the harness's hard-coded DEFAULT_FABLE_FLAGS -- so a flag the "
            "files omit is off, which is the point. Server-owned keys and "
            "MTPLX_FABLE_* names the served tree never reads are refused "
            "before anything boots."
        ),
    )
    parser.add_argument(
        "--flag-registry-root",
        default=None,
        help=(
            "tree whose mtplx/ package defines the known MTPLX_FABLE_* "
            "names. Defaults to --server-cwd on a branch arm (validate "
            "against the code that will actually serve), else "
            f"{MTPLX_SOURCE_ROOT}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print the exact argv, env and per-cell plan this invocation "
            "would run, then exit 0 without starting a server or sending a "
            "request. Output is deterministic and diffable."
        ),
    )
    parser.add_argument("--prompt-cache", default=None)
    parser.add_argument("--receipt-dir", default=None)
    parser.add_argument("--thermal-gate-max-c", type=float, default=DEFAULT_THERMAL_MAX_C)
    parser.add_argument("--stop-after-context", type=int, default=DEFAULT_STOP_AFTER_CONTEXT)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--client-timings-only", action="store_true")
    parser.add_argument("--png", default=None)
    parser.add_argument("--notes", action="append", default=[])
    # --- W49 model-family + client-only ("cells") mode ---
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES,
        default=None,
        help=(
            "prompt-template family. Default: auto-detect from --served-model-id "
            "(a 'deepseek'/'v4.1' id => deepseek-v41), else qwen38. deepseek-v41 "
            "counts/renders via the artifact chat_template.jinja (BOS id 0 first) "
            "and sends enable_thinking (default OFF; see --dsv41-enable-thinking)."
        ),
    )
    parser.add_argument(
        "--dsv41-enable-thinking",
        dest="dsv41_enable_thinking",
        action="store_true",
        default=None,
        help=(
            "deepseek-v41 only: run the sweep cell in THINKING mode. Default is "
            "the official thinking-OFF (reference generate.py --thinking-mode "
            "'chat'). Also settable via DSV41_ENABLE_THINKING=1. DeepSeek's own "
            "instruct evals use thinking + reasoning_effort=100."
        ),
    )
    parser.add_argument(
        "--dsv41-reasoning-effort",
        dest="dsv41_reasoning_effort",
        default=None,
        help=(
            "deepseek-v41 only: reasoning effort for the sweep cell when thinking "
            "is on (int 1-100 or low/high/max; default high=75). Inert in chat "
            "mode. Also settable via DSV41_REASONING_EFFORT."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "tokenizer dir for --mode build-prompts / build-prompt-ids. Default: "
            "the pinned Qwen MODEL. Point at the DSV4.1 artifact to size DSV4.1 "
            "prompts."
        ),
    )
    parser.add_argument(
        "--served-model-id",
        default=None,
        help=(
            "the model id the running server advertises (/v1/models). Sent as "
            "'model' on --mode cells and used for family auto-detection."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "--mode cells only: an ALREADY-RUNNING OpenAI-compatible endpoint "
            "(e.g. http://127.0.0.1:PORT). This mode is a pure HTTP client: it "
            "starts no server and holds no GPU lock -- the caller (a gpu_window "
            "STEP such as served_cell_bench.sh) owns the serve lifecycle."
        ),
    )
    parser.add_argument(
        "--prompt-ids-out",
        default=None,
        help=(
            "--mode build-prompts / build-prompt-ids: also write the exact "
            "per-cell/per-seed server token-id lists to this path."
        ),
    )
    parser.add_argument(
        "--no-warm",
        action="store_true",
        help="--mode cells: skip the untimed warm ping before the first cell.",
    )
    args = parser.parse_args(argv)
    args.model_family_resolved = resolve_model_family(
        args.model_family, args.served_model_id
    )
    _dsv41_thinking, _dsv41_effort = deepseek_v41_thinking_settings(args)
    if args.dry_run and args.mode not in {"run", "preflight", "serve-hold"}:
        parser.error(f"--dry-run has nothing to plan for --mode {args.mode}")

    default_out = ROOT / ".benchmark-artifacts" / "fable" / "server-cell"
    args.receipt_dir = args.receipt_dir or str(default_out)
    args.prompt_cache = args.prompt_cache or str(default_out / "prompts.json")

    if args.mode == "build-prompts":
        contexts = [
            int(c) for c in (args.contexts or "").split(",") if c.strip()
        ] or list(CONTEXT_BATTERY)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        built = build_prompt_cache(
            contexts=contexts,
            seeds=seeds,
            path=Path(args.prompt_cache),
            tokenizer_path=args.tokenizer,
            model_id=args.served_model_id,
            model_family=args.model_family_resolved,
            reasoning_effort=args.reasoning,
            dsv41_enable_thinking=_dsv41_thinking,
            dsv41_reasoning_effort=_dsv41_effort,
            ids_path=args.prompt_ids_out,
        )
        print(
            f"[server-cell] family={built.get('model_family')} "
            f"tokenizer={built.get('model')}"
        )
        for entry in built["prompts"]:
            print(
                f"[server-cell] {entry['cell']:>6} target "
                f"{entry['target_tokens']:>7} seed {entry['seed']}: "
                f"{entry['templated_tokens']} templated tokens "
                f"sha={entry['text_sha256'][:12]}"
            )
        print(f"[server-cell] wrote {args.prompt_cache}")
        if args.prompt_ids_out:
            print(f"[server-cell] wrote token ids {args.prompt_ids_out}")
        return 0

    if args.mode == "build-prompt-ids":
        if not args.prompt_ids_out:
            parser.error("--mode build-prompt-ids requires --prompt-ids-out")
        from transformers import AutoTokenizer

        payload = json.loads(Path(args.prompt_cache).read_text())
        family = resolve_model_family(
            args.model_family or payload.get("model_family"),
            args.served_model_id or payload.get("served_model_id"),
        )
        tok_path = str(args.tokenizer or payload.get("model") or MODEL)
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=False)
        written = write_prompt_ids(
            payload["prompts"],
            tokenizer=tokenizer,
            family=family,
            reasoning_effort=args.reasoning,
            dsv41_enable_thinking=_dsv41_thinking,
            dsv41_reasoning_effort=_dsv41_effort,
            path=Path(args.prompt_ids_out),
            tokenizer_path=tok_path,
            model_id=args.served_model_id or payload.get("served_model_id"),
        )
        for entry in written["prompts"]:
            print(
                f"[server-cell] ids {entry['cell']:>6} target "
                f"{entry['target_tokens']:>7} seed {entry['seed']}: "
                f"{entry['input_tokens']} ids bos={entry['bos_id_prepended']} "
                f"sha={entry['token_ids_sha256'][:12]}"
            )
        print(f"[server-cell] wrote token ids {args.prompt_ids_out}")
        return 0

    if args.mode == "cells":
        return run_cells_remote(args, parser)

    if args.mode == "report":
        records = load_records(Path(args.receipt_dir))
        if not records:
            print(f"no receipts under {args.receipt_dir}", file=sys.stderr)
            return 1
        summary = summarize_records(records)
        print(render_report(summary, notes=args.notes))
        if args.png:
            written = render_png(summary, Path(args.png))
            print(f"\nPNG: {written}" if written else "\nPNG: matplotlib unavailable")
        return 0

    if args.full_stack:
        args.require_full_stack = True
    if args.stack != "mlx-serve" and not args.server_python:
        parser.error(
            f"--server-python is required for --mode {args.mode}"
        )

    # argparse cannot tell "--profile turbo" from the default, and
    # --server-defaults has to refuse the former while ignoring the latter.
    # The raw argv is the only place that distinction exists.
    source_argv = list(sys.argv[1:] if argv is None else argv)
    # --hold-command is argparse.REMAINDER: everything after it belongs to the
    # held client, not to this parser. A --profile inside the eval's own argv
    # is not the harness configuring the server, so the scan stops there --
    # otherwise a defaults hold would be refused for a flag it never passed.
    if "--hold-command" in source_argv:
        source_argv = source_argv[: source_argv.index("--hold-command")]
    args.profile_explicit = any(
        a == "--profile" or a.startswith("--profile=") for a in source_argv
    )

    for problem in (
        check_arm_label(args),
        check_server_cwd(args),
        check_server_defaults(args),
    ):
        if problem:
            parser.error(problem)

    if args.fable_flags_file and args.stack != "branch":
        parser.error(
            "--fable-flags-file applies to the branch arm only; the control "
            f"and mlx-serve arms take no MTPLX_FABLE_* set (stack={args.stack})"
        )
    registry_root = args.flag_registry_root
    if registry_root is None and args.stack == "branch" and args.server_cwd:
        if (Path(args.server_cwd) / "mtplx").is_dir():
            registry_root = args.server_cwd
    try:
        args.fable_flags = resolve_fable_flags(
            args.fable_flags_file,
            stack=args.stack,
            server_defaults=bool(args.server_defaults),
            registry_root=registry_root,
        )
    except FlagFileError as error:
        parser.error(str(error))

    if args.dry_run:
        sys.stdout.write(render_dry_run(args))
        return 0

    if args.mode == "preflight-compare":
        code, text = compare_branch_preflights(Path(args.receipt_dir))
        print(text, flush=True)
        return code
    if args.mode == "preflight":
        if args.stack == "mlx-serve":
            parser.error("preflight is an MTPLX-only mode")
        return run_preflight(args)
    if args.mode == "serve-hold":
        if float(args.hold_seconds) <= 0:
            parser.error("--hold-seconds must be positive")
        return run_serve_hold(args)
    return run_battery(args)


if __name__ == "__main__":
    raise SystemExit(main())
