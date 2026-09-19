"""One real-shape native wo_a, no backbone/expert/Engram allocation."""
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

signal.alarm(120)
os.environ['MTPLX_DSV41_ATTN_WO_A_CACHE'] = '1'
os.environ['MTPLX_DSV41_ATTN_FUSED_PROJ'] = '1'
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

GIB = 1024**3
assert 0 <= float(os.environ['MTPLX_DSV41_BOX_BASELINE_GB']) <= 20
source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
diff = subprocess.check_output(['git','diff'],text=True)
result = {'source_commit': source, 'source_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'one native mxfp8 wo_a [8192,4096], group32; no model or expert bank',
          'bound': '134MiB initializer+quant/dequant/transpose/comparison temporaries <1GiB; 3GiB MLX limit,128MiB cache,4GiB process guard',
          'snapshots': [], 'phases': []}
OUT = Path('/tmp/dsv41-110-preflight/projection-cache-probe.json')

import mlx.core as mx
import mlx.nn as nn
from mtplx.models.deepseek_v41 import Attention

mx.set_default_device(mx.gpu)
mx.set_memory_limit(3*GIB)
mx.set_cache_limit(128*1024**2)
mx.set_wired_limit(3*GIB)
mx.random.seed(20260913)

class Projection(nn.Module):
    _o_lora_dense_weight = Attention._o_lora_dense_weight
    _o_lora_fused_weight = Attention._o_lora_fused_weight

def settle(name):
    mx.synchronize(); gc.collect(); mx.clear_cache()
    snap = host_memory_snapshot()
    result['snapshots'].append(snap)
    dense = projection.get('_wo_a_dense_cache')
    fused = projection.get('_wo_a_bf16T_cache')
    row = {'phase': name, 'active_bytes': mx.get_active_memory(),
           'peak_bytes': mx.get_peak_memory(),
           'dense_cache_bytes': 0 if dense is None else dense[3].nbytes,
           'fused_cache_bytes': 0 if fused is None else fused[3].nbytes}
    result['phases'].append(row)
    print(json.dumps(row),flush=True)
    return row

try:
    projection = Projection()
    projection.n_groups, projection.o_lora_rank = 8, 1024
    projection.wo_a = nn.QuantizedLinear(4096,8192,bias=False,group_size=32,bits=8,mode='mxfp8')
    mx.eval(projection.parameters())
    base = settle('packed_weights')['active_bytes']
    for cycle in range(2):
        dense = projection._o_lora_dense_weight()
        assert projection._o_lora_dense_weight() is dense
        ref = mx.dequantize(projection.wo_a.weight, projection.wo_a.scales, None,
                            group_size=32,bits=8,mode='mxfp8').reshape(8,1024,4096).astype(mx.float32)
        assert bool(mx.array_equal(dense,ref).item())
        del dense,ref
        row = settle(f'prefill_{cycle}')
        assert row['dense_cache_bytes']==134217728 and row['fused_cache_bytes']==0
        assert 134217728 <= row['active_bytes']-base < 134217728 + 16*1024**2
        fused = projection._o_lora_fused_weight()
        assert projection._o_lora_fused_weight() is fused
        ref = mx.contiguous(mx.dequantize(projection.wo_a.weight,projection.wo_a.scales,None,
                                          group_size=32,bits=8,mode='mxfp8').astype(mx.bfloat16)
                            .reshape(8,1024,4096).swapaxes(1,2))
        assert bool(mx.array_equal(fused,ref).item())
        del fused,ref
        row = settle(f'decode_{cycle}')
        assert row['dense_cache_bytes']==0 and row['fused_cache_bytes']==67108864
        assert 67108864 <= row['active_bytes']-base < 67108864 + 16*1024**2
    before = projection._o_lora_fused_weight()
    projection.wo_a.scales = projection.wo_a.scales + mx.array(1,dtype=mx.uint8)
    after = projection._o_lora_fused_weight()
    assert after is not before and not bool(mx.array_equal(before,after).item())
    ref = mx.contiguous(mx.dequantize(projection.wo_a.weight,projection.wo_a.scales,None,
                                      group_size=32,bits=8,mode='mxfp8').astype(mx.bfloat16)
                        .reshape(8,1024,4096).swapaxes(1,2))
    assert bool(mx.array_equal(after,ref).item())
    del before,after,ref
    settle('scales_reloaded')
    result['ok'] = True
    result['repeated_phase_weights_exact'] = True
    result['native_scales_reload_exact'] = True
finally:
    result['peak_mlx_bytes']=mx.get_peak_memory()
    result['snapshots'].append(host_memory_snapshot())
    OUT.write_text(json.dumps(result,indent=2)+'\n')
