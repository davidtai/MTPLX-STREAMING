"""Bounded single-layer DeepSeek attention M6/M7 census at native dimensions."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import threading

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

signal.alarm(300)
source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
    raise RuntimeError("tracked source must be clean before GPU measurement")
baseline = float(os.environ["MTPLX_DSV41_BOX_BASELINE_GB"]) * 1e9
if not 0 <= baseline <= 20e9:
    raise RuntimeError("invalid measured baseline")

import mlx.core as mx

GIB = 1024 ** 3
mx.set_memory_limit(8 * GIB)
mx.set_wired_limit(8 * GIB)
mx.set_cache_limit(GIB // 2)
if baseline + 8 * GIB + 2 * GIB > 110e9:
    raise RuntimeError("candidate lacks bounded physical headroom")

prefix = Path("/tmp/dsv41-110-preflight/verify-attn-real-m7")
probe = Path("scripts/deepseek_v41/verify_attn_rows_census.py")
args = [str(probe), "--gpu", "--rows", "6", "7", "--modes",
        "swa_only,full,reindex,reuse", "--codec", "mxfp8", "--T", "16384",
        "--repeats", "3", "--warmup", "1", "--chain", "2",
        "--out", str(prefix.with_suffix(".json"))]
samples = [host_memory_snapshot()]
stop = threading.Event()

def monitor():
    while not stop.wait(0.25):
        samples.append(host_memory_snapshot())

thread = threading.Thread(target=monitor, daemon=True)
thread.start()
complete = False
try:
    sys.argv = args
    try:
        runpy.run_path(str(probe), run_name="__main__")
    except SystemExit as completed:
        # The probe's CLI ends with raise SystemExit(main()). Treat zero as
        # normal completion, then validate its actual receipt below.
        if completed.code not in (0, None):
            raise
    receipt = json.loads(prefix.with_suffix(".json").read_text())
    assert receipt["rows"] == [6, 7]
    assert receipt["codec"] == "mxfp8" and receipt["dims"]["T"] == 16384
    assert set(receipt["modes"]) == {"swa_only", "full", "reindex", "reuse"}
    complete = True
finally:
    stop.set()
    thread.join(2)
    samples.append(host_memory_snapshot())
    report = {
        "source_commit": source,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        "scope": "one synthetic native-shape mxfp8 attention layer, 16K KV, M6/M7; no model, target experts or MTP",
        "baseline_bytes": baseline,
        "active_scope_bound_bytes": 8 * GIB,
        "mlx_active_peak_bytes": mx.get_peak_memory(),
        "physical_peak_sampled_bytes": max(s["box"]["used_bytes"] for s in samples),
        "swapouts_first": samples[0]["box"]["swapouts_pages"],
        "swapouts_last": samples[-1]["box"]["swapouts_pages"],
        "sample_count": len(samples),
        "complete": complete,
    }
    prefix.with_suffix(".bounds.json").write_text(json.dumps(report, indent=2) + "\n")
    print("M7_SCOPE", json.dumps(report), flush=True)
