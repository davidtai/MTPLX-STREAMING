"""Construction-only Q2/64 conversion of an existing compact MXFP4 draft."""
import gc
import hashlib
import time


def convert_experts(weights, mx):
    # Only the draft bank changes. Non-expert arrays are shared with control.
    result = {name: value for name, value in weights.items() if '.switch_mlp.' not in name}
    records = []
    for name in sorted(weights):
        if '.switch_mlp.' not in name or not name.endswith('.weight'):
            continue
        prefix = name[:-len('.weight')]
        packed, scales = weights[name], weights[prefix + '.scales']
        if packed.ndim != 3 or scales.dtype != mx.uint8 or packed.dtype != mx.uint32:
            raise RuntimeError('expected native stacked MXFP4 expert layout')
        if packed.shape[-1] * 8 != scales.shape[-1] * 32 or scales.shape[-1] % 2:
            raise RuntimeError('Q2/64 does not divide source expert columns')
        planes = [[], [], []]
        started = time.perf_counter()
        for expert in range(packed.shape[0]):
            dense = mx.dequantize(packed[expert], scales[expert], group_size=32,
                                  bits=4, mode='mxfp4', dtype=mx.bfloat16)
            converted = mx.quantize(dense, group_size=64, bits=2, mode='affine')
            mx.eval(converted)
            if tuple(v.dtype for v in converted) != (mx.uint32, mx.bfloat16, mx.bfloat16):
                raise RuntimeError('Q2 conversion storage differs from priced geometry')
            for plane, value in zip(planes, converted):
                plane.append(value)
            del dense, converted
        arrays = tuple(mx.stack(p, axis=0) for p in planes)
        mx.eval(arrays)
        for suffix, value in zip(('.weight', '.scales', '.biases'), arrays):
            result[prefix + suffix] = value
        records.append({'projection': prefix, 'experts': packed.shape[0],
                        'source_bytes': packed.nbytes + scales.nbytes,
                        'converted_bytes': sum(v.nbytes for v in arrays),
                        'elapsed_s': time.perf_counter() - started,
                        'shapes': [list(v.shape) for v in arrays],
                        'dtypes': [str(v.dtype) for v in arrays]})
        del planes, arrays
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    if len(records) != 9:
        raise RuntimeError('conversion did not cover all three draft expert projections/stage')
    source_bytes = sum(r['source_bytes'] for r in records)
    converted_bytes = sum(r['converted_bytes'] for r in records)
    if source_bytes != 3440517120 or converted_bytes != 2023833600:
        raise RuntimeError('compact expert payload inventory differs')
    return result, {'mode': 'affine', 'bits': 2, 'group_size': 64,
                    'source_mode': 'mxfp4', 'source_bits': 4, 'source_group_size': 32,
                    'dequantize_dtype': 'bfloat16', 'source_bytes': source_bytes,
                    'converted_bytes': converted_bytes,
                    'retired_bytes_if_control_released': source_bytes - converted_bytes,
                    'non_expert_arrays_shared': True,
                    'existing_compact_route_table_unchanged': True, 'projections': records}
