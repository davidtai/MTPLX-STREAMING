import hashlib
import json
import os
from pathlib import Path
import signal
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('exclusive GPU guard required before MLX import')
signal.alarm(180)
ROOT = Path(__file__).resolve().parent
proof = json.loads((ROOT / 'construction.json').read_text())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
if (not before['box']['ok']
        or before['box']['used_bytes'] + proof['incremental_bound_bytes'] > 110000000000
        or before['box']['wired_bytes'] + proof['incremental_bound_bytes'] > 100 * 1024**3):
    raise RuntimeError('fixed Q8 bounded probe does not fit the live baseline')
source = Path('mtplx/models/deepseek_v41_fixed_q8_cache.py')
if hashlib.sha256(source.read_bytes()).hexdigest() != proof['source_sha256']:
    raise RuntimeError('fixed Q8 source differs from its allocation proof')
output = ROOT / 'probe.json'
if output.exists():
    raise RuntimeError('refusing to overwrite probe receipt')

import numpy as np
import mlx.core as mx
from mtplx.models.deepseek_v41_fixed_q8_cache import (
    FixedQ8CacheConfig, FixedQ8DraftCache, make_fixed_q8_cache)
from mtplx.models.deepseek_v41_cache import CompressorState

mx.set_memory_limit(proof['allocator_limit_bytes'])
mx.set_cache_limit(proof['cache_limit_bytes'])
config = FixedQ8CacheConfig(**proof['config'])
report = dict(construction=proof, before=before, checks=[])


def run():
    cache = make_fixed_q8_cache(config)
    drafts = [FixedQ8DraftCache(config) for _ in range(config.draft_layers)]
    backings = [x for c in cache for x in c.eval_backing()]
    backings += [x for c in drafts for x in c.detach_prefill_backings()]
    mx.eval(backings)
    allocated = sum(x.nbytes for x in backings)
    if allocated != proof['storage_bytes']['total']:
        raise RuntimeError(f'physical storage {allocated} differs from byte plan')
    del backings
    report['allocated_storage_bytes'] = allocated
    report['active_after_cache_construction_bytes'] = int(mx.get_active_memory())
    # One real-width source layer, including odd prefill chunk/group crossings.
    layer = cache.layers[2]
    native = CompressorState(2)
    rng = np.random.default_rng(20260917)
    source_rows = []
    compressed_reference = []
    index_reference = []
    for chunk in [953] * 17 + [183]:
        x = mx.array(rng.standard_normal((1, chunk, 512)).astype(np.float32))
        s = mx.array(rng.standard_normal((1, chunk, 512)).astype(np.float32))
        k = layer.comp_state.push(x, s)
        reference = native.push(x, s)
        layer.append_window(x)
        layer.append_compress(k)
        layer.append_index_k(k[:, :, :128])
        layer.advance(chunk)
        mx.eval(k, reference, layer.eval_backing())
        if not np.array_equal(np.array(k), np.array(reference)):
            raise RuntimeError('fixed native compressor pooling changed values')
        source_rows.append(np.array(x))
        compressed_reference.append(np.array(k))
        index_reference.append(np.array(k[:, :, :128]))
    if layer.offset != 16384:
        raise RuntimeError('incorrect prefill length')
    window_reference = mx.array(np.concatenate(source_rows, axis=1))
    def quantized_reference(value):
        return mx.dequantize(*mx.quantize(value, group_size=64, bits=8),
                             group_size=64, bits=8)
    expected = quantized_reference(window_reference[:, layer.window_drop_offset:])
    actual = layer.window
    mx.eval(expected, actual)
    if not np.array_equal(np.array(expected), np.array(actual)):
        raise RuntimeError('window quantization or compaction changed Q8 bytes')
    for stored, arrays in ((layer.compress_kv, compressed_reference),
                            (layer.index_k, index_reference)):
        expected = quantized_reference(mx.array(np.concatenate(arrays, axis=1)))
        mx.eval(expected, stored)
        if not np.array_equal(np.array(expected), np.array(stored)):
            raise RuntimeError('compressed/index quantization differs from Q8 reference')
    report['checks'].append('native compressor pooling and Q8 storage match independent references at 16384 rows')
    # Rejected verification rows must not change the recovered packed prefix.
    mark = layer.mark()
    old_window = np.array(layer.window)
    old_compress = np.array(layer.compress_kv)
    old_index = np.array(layer.index_k)
    x = mx.array(rng.standard_normal((1, 6, 512)).astype(np.float32))
    s = mx.array(rng.standard_normal((1, 6, 512)).astype(np.float32))
    k = layer.comp_state.push(x, s)
    layer.append_window(x); layer.append_compress(k); layer.append_index_k(k[:, :, :128]); layer.advance(6)
    mx.eval(layer.eval_backing())
    layer.rollback(mark)
    for actual, expected in ((layer.window, old_window), (layer.compress_kv, old_compress), (layer.index_k, old_index)):
        mx.eval(actual)
        if not np.array_equal(np.array(actual), expected):
            raise RuntimeError('rollback changed a previously stored Q8 prefix')
    report['checks'].append('six-row verify and rollback preserve Q8 prefix bytes')
    # Draft prefill stores only its reachable final window, without re-quantization
    # when the seed helper materializes independent backing arrays.
    draft = drafts[0]
    draft.append_main(window_reference)
    expected = quantized_reference(window_reference[:, -128:])
    mx.eval(draft.detach_prefill_backings(), expected)
    if not np.array_equal(np.array(draft.window), np.array(expected)):
        raise RuntimeError('draft seed differs from independently quantized tail')
    draft_mark = draft.mark()
    draft.append_main(x)
    mx.eval(draft.detach_prefill_backings())
    draft.rollback(draft_mark)
    if not np.array_equal(np.array(draft.window), np.array(expected)):
        raise RuntimeError('draft rollback changed its window')
    report['checks'].append('full draft seed, detach and rollback preserve packed state')
    after = sum(x.nbytes for c in cache for x in c.eval_backing())
    after += sum(x.nbytes for c in drafts for x in c.detach_prefill_backings())
    if after != allocated:
        raise RuntimeError('KV backing storage grew during prefill or decode')
    report['constant_storage_bytes_after_decode'] = after
    # Quantization error, separate from exact packed-storage correctness.
    original = window_reference[:, -128:]
    quantized = quantized_reference(original)
    mx.eval(original, quantized)
    a, b = np.array(original), np.array(quantized)
    report['q8_error'] = dict(max_abs=float(np.max(np.abs(a-b))),
                              relative_rms=float(np.linalg.norm(a-b)/np.linalg.norm(a)))
    report['checks'].append('fixed capacity is unchanged after prefill, quantization and rollback')


start = time.monotonic()
try:
    run()
    report['complete'] = True
finally:
    mx.synchronize(); mx.clear_cache()
    report['mlx_peak_bytes'] = int(mx.get_peak_memory())
    report['mlx_active_after_close_bytes'] = int(mx.get_active_memory())
    report['after'] = host_memory_snapshot()
    report['elapsed_s'] = time.monotonic() - start
    output.write_text(json.dumps(report, indent=2) + '\n')
print('FIXED_Q8_PROBE', json.dumps({k:v for k,v in report.items() if k not in ('construction','before','after')}), flush=True)
