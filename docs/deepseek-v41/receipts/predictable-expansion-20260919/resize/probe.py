"""Price exact packed-bank110->111 growth after the expansion component win."""
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import struct
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent-held GPU/service guard required before MLX')
signal.alarm(120)
ROOT = Path(__file__).resolve().parent
proof = json.loads((ROOT/'installation.json').read_text())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
bound = proof['static_incremental_bound_bytes']
if (not before['box']['ok'] or before['box']['used_bytes']+bound > 110000000000
        or before['box']['wired_bytes']+bound > 100*1024**3):
    raise RuntimeError('packed-growth bound does not fit')
import mlx.core as mx
from library_identity import identify
from mtplx.expert_manifest import load_expert_manifest
from mtplx.models.expert_mlx import MlxComponentBank
from packed_storage import remove_raw_scales
from bank_growth_final import grow_bank

library = identify(proof['strict_allocator'])
mx.set_memory_limit(4*1024**3)
mx.set_cache_limit(256*1024**2)
manifest = load_expert_manifest(Path(proof['model_path'])/'expert-manifest.json')
record = next(r for r in manifest.records if r.layer == 34 and r.expert == 0)
assert record.logical_bytes == 18800640
report = {'complete':False,'construction':proof,'before':before,'library':library,'trials':[]}
bank = None

def fill_markers(bank):
    # Initial arrays have completed their zero fill; only CPU owners write here.
    for component,view in bank._views.items():
        stride = bank._segment_bytes[component]
        tag = int.from_bytes(hashlib.sha256(component.encode()).digest()[:4],'little')
        for row in range(bank.capacity):
            marker = struct.pack('<4I',tag,row,0x31415926,0xDEADBEEF)
            view[row*stride:row*stride+16] = marker
            view[(row+1)*stride-16:(row+1)*stride] = marker[::-1]

def zero_sha(size):
    block = bytes(1024**2)
    h = hashlib.sha256()
    while size:
        count = min(size,len(block));h.update(memoryview(block)[:count]);size -= count
    return h.hexdigest()

try:
    # One untimed existing2->3-row warmup, then three actual110->111 samples.
    for trial,old,new in ((-1,2,3),(0,110,111),(1,110,111),(2,110,111)):
        bank = MlxComponentBank(capacity=old,record=record,label='packed-growth')
        remove_raw_scales(bank,mx=mx)
        assert set(bank.arrays) == {p+'.weight' for p in ('gate_proj','up_proj','down_proj')}
        fill_markers(bank)
        source = {name:hashlib.sha256(view).hexdigest() for name,view in bank._views.items()}
        mx.synchronize();mx.clear_cache();mx.reset_peak_memory()
        initial = int(mx.get_active_memory())
        started = time.perf_counter_ns()
        added = grow_bank(bank,new,mx=mx)
        elapsed = time.perf_counter_ns()-started
        active,peak = int(mx.get_active_memory()),int(mx.get_peak_memory())
        assert added == (new-old)*17694720
        assert active <= initial+added+1024**2
        copy_allowance = (2*new-old)*5898240
        assert peak <= initial+added+copy_allowance+1024**2
        output = []
        for name,view in bank._views.items():
            stride = bank._segment_bytes[name]
            old_sha = hashlib.sha256(view[:old*stride]).hexdigest()
            tail_sha = hashlib.sha256(view[old*stride:]).hexdigest()
            assert old_sha == source[name]
            assert tail_sha == zero_sha((new-old)*stride)
            output.append({'component':name,'old_prefix_sha256':old_sha,'zero_tail_sha256':tail_sha})
        report['trials'].append({'trial':trial,'old_capacity':old,'capacity':new,'elapsed_ns':elapsed,
            'initial_active_bytes':initial,'active_bytes':active,'peak_bytes':peak,
            'added_payload_bytes':added,'copy_allowance_bytes':copy_allowance,
            'old_payload_exact':True,'new_rows_zero':True,'digests':output})
        bank.close();bank=None
        gc.collect();mx.synchronize();mx.clear_cache()
        assert mx.get_active_memory() <= 1024**2
    samples = [r['elapsed_ns'] for r in report['trials'] if r['trial'] >= 0]
    report.update(complete=True,median_one_layer_growth_ns=statistics.median(samples),
        projected_40_layer_growth_s=statistics.median(samples)*40/1e9,
        scope='Real packed bank geometry and preserved prefix/zero tail; no full model.40-layer time is a projection, not a measured full transition.',
        after=host_memory_snapshot())
finally:
    mx.synchronize()
    if bank is not None:bank.close()
    bank=None;gc.collect();mx.clear_cache()
    report['active_after_close_bytes']=int(mx.get_active_memory())
    (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
print('PACKED_GROWTH_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('construction','before','after','library','trials')}),flush=True)
