"""Bounded native-shape HC post screen; no full model and no TPS claim."""
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent must hold the GPU lock before MLX import')
signal.alarm(120)
import mlx.core as mx
import numpy as np
from mtplx.models.deepseek_v4 import _hc_post_impl

mx.set_memory_limit(512 * 1024**2)
mx.set_cache_limit(64 * 1024**2)
out_path = Path('/tmp/dsv41-hc-post-prefill-20260917/gpu-results.json')
if out_path.exists():
    raise RuntimeError('refusing to overwrite evidence')
config_path = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/config.json')
config_bytes = config_path.read_bytes()
config = json.loads(config_bytes)['text_config']
if (config['hc_mult'], config['hidden_size']) != (4, 5120):
    raise RuntimeError('native HC geometry changed')

# Preserve the existing einsum. Only the multiply/add tail is fused, with a
# volatile rounded product to prevent contraction into a fused multiply-add.
tail_source = '''
    const uint i = thread_position_in_grid.x;
    if (i >= uint(total_size)) return;
    const uint row = i / (4 * 5120);
    const uint d = i % 5120;
    volatile float scaled = float(post[i / 5120]) * float(x[row * 5120 + d]);
    out[i] = scaled + float(mixed[i]);
'''
tail_kernel = mx.fast.metal_kernel(
    name='dsv41_hc_post_rounded_tail_prefill',
    input_names=['x', 'post', 'mixed', 'total_size'], output_names=['out'],
    source=tail_source,
)

def rounded_tail(x, residual, post, comb):
    mixed = mx.einsum('...jk,...jd->...kd', comb, residual.astype(mx.float32))
    return tail_kernel(
        inputs=[x.astype(mx.float32), post, mixed, int(residual.size)],
        grid=(int(residual.size), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[residual.shape], output_dtypes=[x.dtype],
    )[0]

rng = np.random.default_rng(91703)
x = mx.array(rng.normal(0, .3, (1, 512, 5120)).astype(np.float32))
residual = mx.array(rng.normal(0, .3, (1, 512, 4, 5120)).astype(np.float32))
post = mx.array(rng.uniform(.01, 1.8, (1, 512, 4)).astype(np.float32))
comb_np = rng.uniform(.05, 1, (1, 512, 4, 4)).astype(np.float32)
comb_np /= comb_np.sum(axis=-1, keepdims=True)
comb = mx.array(comb_np)
del comb_np
args = (x, residual, post, comb)
mx.eval(*args)
routes = {'eager': _hc_post_impl, 'compiled': mx.compile(_hc_post_impl),
          'rounded_tail': rounded_tail}
reference = np.array(_hc_post_impl(*args))
reference_digest = hashlib.sha256(reference.tobytes()).hexdigest()
results = {}
for name, route in routes.items():
    value = route(*args)
    mx.eval(value)
    observed = np.array(value)
    equality = bool(np.array_equal(reference.view(np.uint32), observed.view(np.uint32)))
    error = float(np.max(np.abs(reference - observed)))
    digest = hashlib.sha256(observed.tobytes()).hexdigest()
    del observed, value
    mx.clear_cache()
    active = int(mx.get_active_memory())
    mx.reset_peak_memory()
    value = route(*args)
    mx.eval(value)
    peak = int(mx.get_peak_memory())
    del value
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(10):
            value = route(*args)
            mx.eval(value)
            del value
        samples.append((time.perf_counter() - started) / 10)
    result = {'median_s': statistics.median(samples), 'samples_s': samples,
              'input_active_bytes': active, 'peak_active_bytes': peak,
              'temporary_and_output_peak_bytes': peak - active,
              'bit_exact': equality, 'max_abs': error, 'sha256': digest}
    results[name] = result
    print(name, json.dumps(result), flush=True)

report = {
    'scope': 'one native [1,512,4,5120] float32 HC post, synthetic activations; '
             'original einsum and rounded multiply/add; no model loading, '
             'generation, or production throughput claim',
    'allocator_limit_bytes': 512 * 1024**2,
    'allocator_cache_limit_bytes': 64 * 1024**2,
    'static_resident_input_bytes': sum(int(v.nbytes) for v in args),
    'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'hc_post_source_sha256': hashlib.sha256(
        __import__('inspect').getsource(_hc_post_impl).encode()).hexdigest(),
    'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'reference_sha256': reference_digest, 'results': results,
}
out_path.parent.mkdir(exist_ok=True)
with out_path.open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
