"""Profile native D5/M6 using the already-bounded cap91 prefill wrapper."""
import cProfile
import hashlib
import inspect
import json
import os
from pathlib import Path
import pstats
import runpy
import sys

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('the parent GPU guard must hold the lock before MLX import')

ROOT = Path('/tmp/dsv41-online-cache-20260917')
GROWTH_ROOT = Path('/tmp/dsv41-depth7-full-20260917')
PROFILE_RESERVE = 128 * 1024**2
for name in ('decode.prof', 'decode-profile.json'):
    if (ROOT / name).exists():
        raise RuntimeError(f'refusing to overwrite {name}')

sys.path.insert(0, str(GROWTH_ROOT))
import admission

original_admission = admission.resolve_admission


def profile_admission(base, wired, **kwargs):
    # Charge bounded profiler bookkeeping to the host allowance before selecting
    # storage. Keep measured baseline distinct from this additional reservation.
    result = original_admission(
        base + PROFILE_RESERVE, wired + PROFILE_RESERVE, **kwargs
    )
    result['admission_baseline_with_profiler_reserve_bytes'] = result['baseline_bytes']
    result['baseline_bytes'] = base
    result['wired_before_bytes'] = wired
    result['profiler_host_reserve_bytes'] = PROFILE_RESERVE
    result['host_reserve_bytes'] += PROFILE_RESERVE
    # The runner derives its MLX cap from the actual machine baseline and its
    # existing 2 GiB host policy. Preserve that cap; the stricter static physical
    # bound and slot selection above separately price this profiler's extra host
    # storage. A policy limit is not an estimate of allocated bytes.
    result['allocator_limit_bytes'] += PROFILE_RESERVE
    result['allocator_policy_host_reserve_bytes'] = 2 * 1024**3
    return result


admission.resolve_admission = profile_admission

from mtplx.models import deepseek_v41_dspark_decode as decode

original_cycles = decode._decode_cycles
original_cycles_sha256 = hashlib.sha256(
    inspect.getsource(original_cycles).encode()
).hexdigest()


def profiled_cycles(**kwargs):
    profile = cProfile.Profile()
    print('DECODE_PROFILE_START', flush=True)
    profile.enable()
    try:
        return original_cycles(**kwargs)
    finally:
        profile.disable()
        profile.dump_stats(str(ROOT / 'decode.prof'))
        stats = pstats.Stats(profile)
        rows = []
        for (filename, line, name), (primitive, total, self_s, cumulative_s, callers) in stats.stats.items():
            rows.append(dict(file=filename, line=line, name=name,
                primitive_calls=primitive, total_calls=total,
                self_s=self_s, cumulative_s=cumulative_s))
        rows.sort(key=lambda row: row['self_s'], reverse=True)
        receipt = dict(
            scope='Diagnostic main-thread cProfile of native D5/M6 decode; timings are ineligible for throughput promotion',
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            decode_cycles_source_sha256=original_cycles_sha256,
            growth_wrapper_sha256=hashlib.sha256((GROWTH_ROOT / 'run_full.py').read_bytes()).hexdigest(),
            profiler_host_reserve_bytes=PROFILE_RESERVE,
            thread_scope='main thread only; asynchronous I/O worker execution is not traced',
            functions=len(rows), total_calls=stats.total_calls,
            total_self_s=stats.total_tt, rows=rows,
        )
        (ROOT / 'decode-profile.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('DECODE_PROFILE_FINISHED', json.dumps({k: receipt[k] for k in (
            'functions', 'total_calls', 'total_self_s')}), flush=True)


decode._decode_cycles = profiled_cycles
runpy.run_path(str(GROWTH_ROOT / 'run_full.py'), run_name='__main__')
