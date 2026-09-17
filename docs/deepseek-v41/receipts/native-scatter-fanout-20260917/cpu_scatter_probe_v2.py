"""Bounded I/O-only screen of native component-scatter fanout; no MLX import."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent-held GPU/service guard required')
signal.alarm(120)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
from mtplx.expert_io import ExpertIOMetrics, PositionalExpertReader
from mtplx.expert_manifest import load_expert_manifest
ROOT=Path('/tmp/dsv41-io-fanout8-20260917')
OUT=ROOT/'cpu-scatter-results-v2.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite an earlier screen')
ARTIFACT=Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
MANIFEST=ARTIFACT/'expert-manifest.json'
MANIFEST_SHA='44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9'
if hashlib.sha256(MANIFEST.read_bytes()).hexdigest()!=MANIFEST_SHA:
    raise RuntimeError('native artifact identity changed')
if host_memory_snapshot()['box']['used_bytes']+2*1024**3>110000000000:
    raise RuntimeError('2GiB bounded probe does not fit current physical memory')
manifest=load_expert_manifest(MANIFEST)
records={(r.layer,r.expert):r for r in manifest.records}
layers=tuple(sorted({r.layer for r in manifest.records}))
if len(layers)!=40 or len(records)!=15360:
    raise RuntimeError('native record inventory changed')
first=records[layers[0],0]
lengths=tuple(s.length for s in first.segments)
if len(lengths)!=6 or sum(lengths)!=18800640:
    raise RuntimeError('native component layout changed')
batches=tuple(tuple(records[layers[i%40],(i*11+offset)%384] for offset in (0,137,277)) for i in range(512))
for batch in batches:
    for record in batch:
        if tuple(s.length for s in record.segments)!=lengths or not record.sha256:
            raise RuntimeError('native record geometry or digest absent')
class ComponentSlot:
    def __init__(self):
        self.arrays=tuple(bytearray(n) for n in lengths)
    def record_views(self,record):
        return tuple(memoryview(a) for a in self.arrays)
    def digest(self):
        h=hashlib.sha256()
        for a in self.arrays: h.update(a)
        return h.hexdigest()
slots=tuple(ComponentSlot() for _ in range(3))
report={'scope':'I/O-only screening, three concurrent records, real six-component native byte layout; CPU bytearrays, no GPU compute/ownership/cache pressure; not a decode-TPS measurement',
    'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('mtplx/expert_io.py','mtplx/expert_manifest.py')},
    'manifest_sha256':MANIFEST_SHA,'record_bytes':18800640,'segment_bytes':lengths,
    'component_buffer_bytes':3*18800640,'physical_allowance_bytes':2*1024**3,
    'initial_memory':host_memory_snapshot(),'batches':len(batches),'arms':[]}
with concurrent.futures.ThreadPoolExecutor(max_workers=3,thread_name_prefix='scatter-callers') as callers:
    for arm,fanout in (('control_before',4),('candidate',8),('control_after',4)):
        reader=PositionalExpertReader(ARTIFACT,bypass_page_cache=True,use_native=False,io_read_fanout=fanout)
        try:
            def batch_read(batch):
                futures=[callers.submit(reader.read_record_into,manifest,record,slot,verify_hash=False) for record,slot in zip(batch,slots)]
                error=None
                for future in futures:
                    try: future.result()
                    except BaseException as exc:
                        if error is None: error=exc
                if error is not None: raise error
            for batch in batches[:4]: batch_read(batch)
            reader.metrics=ExpertIOMetrics()
            started=time.perf_counter()
            for batch in batches: batch_read(batch)
            wall=time.perf_counter()-started
            metrics=reader.metrics.as_dict()
            hashes=[slot.digest() for slot in slots]
            if hashes!=[record.sha256 for record in batches[-1]]:
                raise RuntimeError('landed native component bytes differ')
            if (metrics['records_read']!=len(batches)*3 or metrics['read_bytes']!=len(batches)*3*18800640
                or metrics['native_positional_calls']!=0 or any(metrics[k] for k in ('short_reads','integrity_errors','io_errors','cancellations','deadline_errors'))):
                raise RuntimeError('I/O coverage or failure counters differ')
            row={'arm':arm,'fanout':fanout,'fanout_pool_workers':reader._fanout_pool_workers,
                'wall_s':wall,'gb_per_s':metrics['read_bytes']/wall/1e9,
                'metrics':metrics,'final_record_sha256':hashes,'memory':host_memory_snapshot()}
            report['arms'].append(row)
            print(json.dumps(row),flush=True)
        finally:
            reader.close()
if any(name=='mlx' or name.startswith('mlx.') for name in sys.modules):
    raise RuntimeError('CPU I/O screen unexpectedly imported MLX')
report['mlx_imported']=False
report['final_memory']=host_memory_snapshot()
OUT.write_text(json.dumps(report,indent=2)+'\n')
