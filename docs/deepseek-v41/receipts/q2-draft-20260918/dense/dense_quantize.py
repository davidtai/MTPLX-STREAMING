"""Requantize only draft dense matrices which do not build committed KV."""
import gc
import time


def convert_dense(weights, mx):
    result = dict(weights)
    records = []
    for name in sorted(weights):
        if not name.startswith('mtp.layers.') or not name.endswith('.weight'):
            continue
        if '.switch_mlp.' in name or '.main_proj.' in name or '.attn.wkv.' in name:
            continue
        prefix = name[:-len('.weight')]
        if prefix + '.scales' not in weights:
            continue  # construction inventory: BF16 routers, norms and Markov stay native
        packed, scales = weights[name], weights[prefix + '.scales']
        if packed.ndim != 2 or packed.dtype != mx.uint32 or scales.dtype != mx.uint8:
            raise RuntimeError('expected native MXFP8 dense matrix')
        if packed.shape[-1]*4 != scales.shape[-1]*32 or scales.shape[-1] % 2:
            raise RuntimeError('Q2/64 does not divide native dense geometry')
        if not ('.attn.' in prefix or '.mlp.shared_experts.' in prefix):
            raise RuntimeError('unexpected dense projection in conversion allowlist')
        started = time.perf_counter()
        dense = mx.dequantize(packed, scales, group_size=32, bits=8,
                              mode='mxfp8', dtype=mx.bfloat16)
        arrays = mx.quantize(dense, group_size=64, bits=2, mode='affine')
        mx.eval(arrays)
        if tuple(v.dtype for v in arrays) != (mx.uint32, mx.bfloat16, mx.bfloat16):
            raise RuntimeError('dense Q2 storage differs from priced geometry')
        for suffix, value in zip(('.weight', '.scales', '.biases'), arrays):
            result[prefix+suffix] = value
        records.append({'projection': prefix, 'source_bytes': packed.nbytes+scales.nbytes,
                        'converted_bytes': sum(v.nbytes for v in arrays),
                        'shapes': [list(v.shape) for v in arrays],
                        'elapsed_s': time.perf_counter()-started})
        del dense, arrays
        gc.collect(); mx.synchronize(); mx.clear_cache()
    if len(records) != 21:
        raise RuntimeError(f'expected21 non-seeding dense projections, got{len(records)}')
    for name, value in weights.items():
        if '.main_proj.' in name or '.attn.wkv.' in name:
            if result[name] is not value:
                raise RuntimeError('draft KV seed weights changed')
    source = sum(r['source_bytes'] for r in records)
    converted = sum(r['converted_bytes'] for r in records)
    return result, {'bits': 2, 'group_size': 64, 'mode': 'affine',
                    'source_bytes': source, 'converted_bytes': converted,
                    'retired_bytes_if_control_released': source-converted,
                    'seed_weights_unchanged_by_identity': True, 'projections': records}
