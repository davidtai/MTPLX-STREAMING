"""Bounded conversion-cost screen; native grouped BF16 reduction is unchanged."""
import gc, hashlib, json, os, signal, statistics, time
from pathlib import Path

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU guard required before MLX')
signal.alarm(240)
ROOT = Path(__file__).resolve().parent
proof = json.loads((ROOT / 'installation.json').read_text())
if (ROOT / 'probe.json').exists():
    raise RuntimeError('refusing prior evidence overwrite')
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
if (not before['box']['ok']
        or before['box']['used_bytes'] + proof['incremental_bound_bytes'] > 110000000000
        or before['box']['wired_bytes'] + proof['incremental_bound_bytes'] > 100 * 1024**3):
    raise RuntimeError('complete operator bound does not fit')
model = Path(proof['model_path'])
for name, key in [('expert-manifest.json', 'model_manifest_sha256'), ('config.json', 'config_sha256')]:
    if hashlib.sha256((model / name).read_bytes()).hexdigest() != proof[key]:
        raise RuntimeError('native artifact changed')

import mlx.core as mx
import numpy as np
from mtplx.expert_manifest import load_expert_manifest
from attention_reader import load_attention_tensors
from fused_transpose import make_transpose

mx.set_memory_limit(proof['allocator_limit_bytes'])
mx.set_cache_limit(proof['allocator_cache_limit_bytes'])
report = {'source_commit': proof['source_commit'], 'construction': proof,
          'before': before, 'complete': False, 'weight_parity': [], 'cases': []}


def digest(value):
    return hashlib.sha256(np.array(value.view(mx.uint8)).tobytes()).hexdigest()


def run():
    manifest = load_expert_manifest(model / 'expert-manifest.json')
    kept = tuple(t for t in manifest.resident_tensors if t.tensor in proof['resident_names'])
    raw = load_attention_tensors(model, manifest, kept, mx=mx)
    pairs = []
    transpose = make_transpose()
    for layer in proof['layers']:
        prefix = f'layers.{layer}.attn.'
        w, s, bw, bs = (raw[prefix + name] for name in
                       ['wo_a.weight', 'wo_a.scales', 'wo_b.weight', 'wo_b.scales'])
        if (w.dtype != mx.uint32 or tuple(w.shape) != (8192, 1024)
                or s.dtype != mx.uint8 or tuple(s.shape) != (8192, 128)
                or bw.dtype != mx.uint32 or tuple(bw.shape) != (5120, 2048)
                or bs.dtype != mx.uint8 or tuple(bs.shape) != (5120, 256)):
            raise RuntimeError('native projection codec or geometry differs')
        native = mx.contiguous(mx.dequantize(w, s, None, group_size=32, bits=8,
                                            mode='mxfp8').astype(mx.bfloat16)
                               .reshape(8, 1024, 4096).swapaxes(1, 2))
        fused = transpose(w, s)
        mx.eval(native, fused)
        a, b = digest(native), digest(fused)
        report['weight_parity'].append({'layer': layer, 'shape': list(native.shape),
                                       'bytes': native.nbytes, 'native_sha256': a,
                                       'fused_sha256': b, 'exact': a == b})
        if a != b:
            raise RuntimeError('fused transpose changed native BF16 weight bytes')
        pairs.append((w, s, bw, bs, native))
        del fused

    def native_fresh(w, s):
        return mx.contiguous(mx.dequantize(w, s, None, group_size=32, bits=8,
                                            mode='mxfp8').astype(mx.bfloat16)
                             .reshape(8, 1024, 4096).swapaxes(1, 2))

    rng = np.random.default_rng(20260918)
    for rows in (1, 6):
        xs = [mx.array(rng.normal(0, .15, (rows, 8, 4096)).astype(np.float32),
                       dtype=mx.bfloat16).swapaxes(0, 1) for _ in pairs]
        mx.eval(xs)

        def execute(mode):
            outputs = []
            for x, (w, s, bw, bs, native) in zip(xs, pairs):
                weight = native if mode == 'cached' else (
                    native_fresh(w, s) if mode == 'native_fresh' else transpose(w, s))
                projected = mx.matmul(x, weight).swapaxes(0, 1).reshape(rows, 8192)
                output = mx.quantized_matmul(projected, bw, bs, None,
                                            transpose=True, group_size=32,
                                            bits=8, mode='mxfp8')
                mx.eval(output)
                outputs.append(output)
            return outputs

        reference = execute('cached')
        parity = {}
        for mode in ('native_fresh', 'fused_fresh'):
            candidate = execute(mode)
            equal = [digest(a) == digest(b) for a, b in zip(reference, candidate)]
            parity[mode] = equal
            if not all(equal):
                raise RuntimeError('projection output differs despite exact weight layout')
        del reference, candidate

        blocks = []
        for mode in ('cached', 'fused_fresh', 'native_fresh', 'cached',
                     'native_fresh', 'fused_fresh', 'cached'):
            for _ in range(2):
                execute(mode)
            samples = []
            for _ in range(9):
                start = time.perf_counter_ns()
                execute(mode)
                samples.append(time.perf_counter_ns() - start)
            blocks.append({'mode': mode, 'samples_five_layers_ns': samples,
                           'median_ns': statistics.median(samples)})
        medians = {mode: statistics.median(b['median_ns'] for b in blocks if b['mode'] == mode)
                   for mode in ('cached', 'native_fresh', 'fused_fresh')}
        controls = [b['median_ns'] for b in blocks if b['mode'] == 'cached']
        report['cases'].append({'rows': rows, 'output_exact': parity, 'blocks': blocks,
                                'five_layer_medians_ns': medians,
                                'fused_vs_cached_ratio': medians['fused_fresh'] / medians['cached'],
                                'fused_vs_native_fresh_ratio': medians['fused_fresh'] / medians['native_fresh'],
                                'added_ns_per_layer': (medians['fused_fresh'] - medians['cached']) / len(pairs),
                                'control_spread_fraction': (max(controls) - min(controls)) / medians['cached']})
    report['mlx_peak_bytes'] = int(mx.get_peak_memory())
    report['complete'] = True


try:
    run()
finally:
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    report['active_after_close_bytes'] = int(mx.get_active_memory())
    report['after'] = host_memory_snapshot()
    (ROOT / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['construction', 'before', 'after']}, indent=2))
