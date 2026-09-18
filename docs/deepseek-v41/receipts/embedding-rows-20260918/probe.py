"""Bounded embedding ownership and lookup-cost screen; never loads the model."""
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU/service guard required before MLX import')
signal.alarm(180)
ROOT = Path(__file__).resolve().parent
proof = json.loads((ROOT / 'installation.json').read_text())
if (ROOT / 'probe.json').exists():
    raise RuntimeError('refusing prior evidence overwrite')
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
bound = proof['static_incremental_bound_bytes']
if (not before['box']['ok'] or before['box']['used_bytes'] + bound > 110000000000
        or before['box']['wired_bytes'] + bound > 100 * 1024**3):
    raise RuntimeError('complete embedding bound does not fit')

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from library_identity import identify
from file_embedding import FileRowEmbedding, ARENA_BYTES

library = identify(proof['strict_allocator'])
mx.set_memory_limit(4 * 1024**3)
mx.set_cache_limit(256 * 1024**2)
report = {'source_commit': proof['source_commit'], 'scope': proof['scope'],
          'construction': proof, 'before': before, 'library': library,
          'complete': False, 'cases': {}}
native = cache = raw = words = ids = output = None
try:
    cache = FileRowEmbedding(proof['embedding'])
    # Read only this tensor, with no file-backed mapping or unrelated shard tensors.
    raw = bytearray(proof['embedding']['nbytes'])
    view = memoryview(raw)
    for start in range(0, len(raw), 32 * 1024**2):
        count = min(32 * 1024**2, len(raw) - start)
        payload = os.pread(cache._fd, count, cache._offset + start)
        if len(payload) != count:
            raise RuntimeError('short input table read')
        view[start:start + count] = payload
    del payload, view
    words = np.frombuffer(raw, dtype=np.uint16).reshape(proof['embedding']['shape'])
    native = nn.Embedding.__new__(nn.Embedding)
    nn.Module.__init__(native)
    native.weight = mx.array(words).view(mx.bfloat16)
    mx.eval(native.weight)
    report['table_sha256'] = hashlib.sha256(raw).hexdigest()
    words = raw = None
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    report['active_with_native_bytes'] = int(mx.get_active_memory())
    rng = np.random.default_rng(20260918)
    references = {}
    # Model shapes: AR1, draft block, verify6, including fixed noise and duplicates.
    shapes = (1, proof['draft_block_size'], 6)
    rows_by_case = {}
    for width in shapes:
        cases = []
        for index in range(32):
            values = rng.integers(0, 129280, size=(1, width), dtype=np.int32)
            if width == proof['draft_block_size']:
                values[:, 1:] = proof['noise_token_id']
            if index == 0:
                values[:, 0] = 129279
            cases.append(values)
        rows_by_case[str(width)] = cases
    for mode in ('native', 'file_cold', 'file_warm', 'native_repeat'):
        timings = {}
        for width, cases in rows_by_case.items():
            elapsed = []
            for index, values in enumerate(cases):
                ids = mx.array(values)
                mx.eval(ids)
                start = time.perf_counter_ns()
                output = native(ids) if mode.startswith('native') else cache(ids)
                mx.eval(output)
                elapsed.append(time.perf_counter_ns() - start)
                digest = hashlib.sha256(np.array(output.view(mx.uint16)).tobytes()).hexdigest()
                key = (width, index)
                if mode == 'native':
                    references[key] = digest
                if digest != references[key]:
                    raise RuntimeError(f'embedding output differs: {mode}/{key}')
            timings[width] = {'total_ns': sum(elapsed), 'median_ns': statistics.median(elapsed),
                              'max_ns': max(elapsed), 'calls': len(elapsed)}
        report['cases'][mode] = timings
    output = ids = None
    native = None
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    report['active_after_retirement_bytes'] = int(mx.get_active_memory())
    report['released_native_bytes'] = report['active_with_native_bytes'] - report['active_after_retirement_bytes']
    if report['released_native_bytes'] < proof['embedding']['nbytes']:
        raise RuntimeError('native table ownership was not released')
    # Ensure a lazy returned result survives more than an arena of cache evictions.
    ids = mx.array(rows_by_case['6'][0])
    mx.eval(ids)
    held = cache(ids)
    for start in range(0, cache._capacity + 32, 8):
        output = cache(mx.array(np.arange(start, start + 8, dtype=np.int32).reshape(1, 8)))
        mx.eval(output)
    mx.eval(held)
    if hashlib.sha256(np.array(held.view(mx.uint16)).tobytes()).hexdigest() != references[('6', 0)]:
        raise RuntimeError('pending returned rows alias the cache arena')
    report['arena_bytes'] = ARENA_BYTES
    report['resident_rows'] = len(cache._lru)
    report['arena_capacity_rows'] = cache._capacity
    report['eviction_ownership_exact'] = True
    report['all_outputs_exact'] = True
    report['peak_mlx_bytes'] = int(mx.get_peak_memory())
    report['after'] = host_memory_snapshot()
    held = output = ids = None
    report['complete'] = True
finally:
    mx.synchronize()
    if cache is not None:
        cache.close()
    native = cache = raw = words = output = ids = None
    gc.collect()
    mx.clear_cache()
    report['active_after_close_bytes'] = int(mx.get_active_memory())
    (ROOT / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')
print('EMBEDDING_ROWS_SCREEN', json.dumps({k:v for k,v in report.items()
      if k not in ('construction','before','after')}), flush=True)
