"""Native/Q8 cache owners under the real layer-major prefix-view lifetime."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('exclusive parent GPU guard required before MLX import')
signal.alarm(180)
ROOT = Path(__file__).resolve().parent
proof = json.loads((ROOT / 'construction.json').read_text())
for name, digest in proof['source_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('source differs from lifetime proof: ' + name)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
if (not before['box']['ok']
        or before['box']['used_bytes'] + proof['incremental_bound_bytes'] > 110000000000
        or before['box']['wired_bytes'] + proof['incremental_bound_bytes'] > 100 * 1024**3):
    raise RuntimeError('bounded lifetime probe does not fit current memory')
os.environ.update(proof['cache_env'])

import mlx.core as mx
from mtplx.models.deepseek_v41_cache import DeepseekV41Cache
from mtplx.models.deepseek_v41_dspark import DSparkStageCache
from mtplx.models.deepseek_v41_fixed_q8_cache import (
    FixedQ8CacheConfig, FixedQ8DraftCache, make_fixed_q8_cache)

mx.set_memory_limit(3 * 1024**3)
mx.set_cache_limit(256 * 1024**2)
config = FixedQ8CacheConfig(**proof['config'])
report = dict(before=before, construction=proof, arms=[])
output = ROOT / 'probe.json'
if output.exists():
    raise RuntimeError('refusing to overwrite a lifetime receipt')


def run_arm(q8):
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.monotonic()
    if q8:
        cache = make_fixed_q8_cache(config)
        drafts = [FixedQ8DraftCache(config) for _ in range(3)]
    else:
        cache = DeepseekV41Cache(40, window_size=128,
            compress_ratios=config.ratios, kv_source_layer_ids=config.sources)
        drafts = [DSparkStageCache(128, 512) for _ in range(3)]
    chunks = [953] * 17 + [183]
    # A shared runtime per prefill chunk retains that chunk's source KV/index
    # view until the next source layer overwrites it, as _forward_layer_major does.
    shared = [(None, None) for _ in chunks]
    row = dict(q8=q8, layer_samples=[])
    for layer_index, layer in enumerate(cache):
        for chunk_index, n in enumerate(chunks):
            x = mx.sin(mx.arange(n * 512, dtype=mx.float32).reshape(1, n, 512) * 0.001)
            window = x.astype(mx.bfloat16)
            layer.append_window(window)
            if layer.is_kv_source:
                if layer.compress_ratio > 1:
                    compressed = layer.comp_state.push(x, x * 0.125)
                else:
                    compressed = window
                layer.append_compress(compressed)
                layer.append_index_k(compressed[:, :, :128])
                shared[chunk_index] = (layer.compress_kv, layer.index_k)
            mx.eval(layer.eval_backing(), [a for a in shared[chunk_index] if a is not None])
        if layer.is_kv_source:
            row['layer_samples'].append(dict(layer=layer_index,
                active_bytes=int(mx.get_active_memory()), peak_bytes=int(mx.get_peak_memory()),
                shared_view_logical_bytes=sum(a.nbytes for pair in shared for a in pair if a is not None),
                compressed_dtype=str(layer.compress_kv.dtype)))
    cache.advance(16384)
    # Draft seeding uses the full prompt; its implementation retains only a tail.
    seed = mx.sin(mx.arange(16384 * 512, dtype=mx.float32).reshape(1, 16384, 512) * 0.001).astype(mx.bfloat16)
    for draft in drafts:
        draft.append_main(seed)
        mx.eval(draft.detach_prefill_backings())
    row['prefill_active_bytes'] = int(mx.get_active_memory())
    row['prefill_peak_bytes'] = int(mx.get_peak_memory())
    shared.clear()
    del seed, x, window, compressed
    mx.synchronize()
    mx.clear_cache()
    row['after_prefill_views_released_bytes'] = int(mx.get_active_memory())
    for layer in cache:
        x = mx.sin(mx.arange(6 * 512, dtype=mx.float32).reshape(1, 6, 512) * 0.001)
        layer.append_window(x.astype(mx.bfloat16))
        if layer.is_kv_source:
            compressed = (layer.comp_state.push(x, x * 0.125)
                          if layer.compress_ratio > 1 else x.astype(mx.bfloat16))
            layer.append_compress(compressed)
            layer.append_index_k(compressed[:, :, :128])
        mx.eval(layer.eval_backing())
    cache.advance(6)
    if cache.trim(5) != 5 or cache.offset != 16385:
        raise RuntimeError('verify/trim changed the context contract')
    row['verify_peak_bytes'] = int(mx.get_peak_memory())
    row['elapsed_s'] = time.monotonic() - start
    return row


try:
    for q8 in (False, True, False):
        row = run_arm(q8)
        mx.synchronize()
        mx.clear_cache()
        row['active_after_close_bytes'] = int(mx.get_active_memory())
        if row['active_after_close_bytes'] > 2 * 1024**2:
            raise RuntimeError('cache storage retained after the arm returned')
        report['arms'].append(row)
        print('LIFETIME_ARM', json.dumps(row), flush=True)
    report['complete'] = True
finally:
    report['after'] = host_memory_snapshot()
    output.write_text(json.dumps(report, indent=2) + '\n')
