"""Guard-owned, one-layer real-shape attention census with explicit caps."""
import hashlib
import json
import pathlib
import runpy
import signal
import subprocess
import threading

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

signal.alarm(360)
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
    raise RuntimeError('tracked source must be clean before measurement')
import mlx.core as mx

limit = 8 * 1024**3
cache_limit = 512 * 1024**2
mx.set_memory_limit(limit)
mx.set_wired_limit(limit)
mx.set_cache_limit(cache_limit)
stop = threading.Event()
samples = []

def monitor():
    while not stop.wait(.25):
        samples.append(host_memory_snapshot())

samples.append(host_memory_snapshot())
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
try:
    runpy.run_path('scripts/deepseek_v41/verify_attn_rows_census.py', run_name='__main__')
finally:
    stop.set()
    thread.join(2)
    samples.append(host_memory_snapshot())
    report = {
        'source_commit': source,
        'wrapper_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'memory_limit_bytes': limit, 'cache_limit_bytes': cache_limit,
        'mlx_active_peak_bytes': mx.get_peak_memory(),
        'static_scope': 'one attention layer, no experts/model: <=6 query rows, <=4 queued copies, 16K KV; construction and bounded temporary arrays budgeted below8 GiB',
        'os_samples': samples,
    }
    pathlib.Path('/tmp/dsv41-110-preflight/verify-attn-real-shapes.bounds.json').write_text(json.dumps(report, indent=2) + '\n')
