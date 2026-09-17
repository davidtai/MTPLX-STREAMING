import hashlib,json,os,signal,time
from pathlib import Path
from types import SimpleNamespace
if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('exclusive guard required before MLX import')
signal.alarm(120)
import mlx.core as mx
from mtplx.models.expert_mlx import MlxComponentBank
from bank_growth_sync import grow_bank
mx.set_memory_limit(512*1024**2)
mx.set_cache_limit(64*1024**2)
root=Path('/tmp/dsv41-cache-growth-20260917')
meta=json.loads((root/'native-record.json').read_text())
r=meta['record'];record=SimpleNamespace(logical_bytes=r['logical_bytes'],segments=[SimpleNamespace(**s) for s in r['segments']])
payload=Path('/tmp/dsv41-expert-record-0.bin').read_bytes()
assert hashlib.sha256(payload).hexdigest()==r['sha256']
second=payload.translate(bytes(v ^ 85 for v in range(256)))
bank=MlxComponentBank(capacity=2,record=record,label='bounded-growth-probe')
for row,blob in enumerate((payload,second)):
    offset=0
    for seg in record.segments:
        view=bank.component_view(row,seg.component)
        view[:]=memoryview(blob)[offset:offset+seg.length]
        view.release();del view
        offset+=seg.length
mx.synchronize();mx.clear_cache()
initial=mx.get_active_memory();mx.reset_peak_memory();start=time.perf_counter()
added=grow_bank(bank,3,mx=mx)
elapsed=time.perf_counter()-start
active=mx.get_active_memory();peak=mx.get_peak_memory()
hashes=[]
for row in range(3):
    digest=hashlib.sha256()
    for seg in record.segments:
        view=bank.component_view(row,seg.component);digest.update(view);view.release();del view
    hashes.append(digest.hexdigest())
assert hashes[:2]==[hashlib.sha256(x).hexdigest() for x in (payload,second)]
assert hashes[2]==hashlib.sha256(bytes(len(payload))).hexdigest()
assert active-initial==added==18800640
bank.close();mx.clear_cache()
report={'scope':'single native six-component bank, 2 to 3 rows; byte copies and release accounting, no model generation','initial_active_bytes':initial,'grown_active_bytes':active,'peak_active_bytes':peak,'active_after_close_bytes':mx.get_active_memory(),'added_bytes':added,'growth_seconds':elapsed,'row_sha256':hashes,'allocator_limit_bytes':512*1024**2,'allocator_cache_request_bytes':64*1024**2,'helper_sha256':hashlib.sha256((root/'bank_growth_sync.py').read_bytes()).hexdigest(),'manifest_sha256':meta['manifest_sha256']}
(root/'probe-sync-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
