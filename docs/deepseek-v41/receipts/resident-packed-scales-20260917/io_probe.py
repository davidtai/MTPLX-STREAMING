"""CPU positional-I/O screen: complete records vs native weight ranges only."""
import concurrent.futures
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'mlx','mlx_lm'}:raise RuntimeError('MLX forbidden in CPU I/O screen')
sys.meta_path.insert(0,NoMLX())
if os.environ.get('_GPU_WINDOW_LOCKED')!='1':raise RuntimeError('parent guard required')
signal.alarm(180)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
from mtplx.expert_io import ExpertIOMetrics,PositionalExpertReader
from mtplx.expert_manifest import load_expert_manifest
root=Path('/tmp/dsv41-resident-scales-20260917');out=root/'io-probe.json'
if out.exists():raise RuntimeError('refusing to overwrite evidence')
before=host_memory_snapshot();bound=2*1024**3
if not before['box']['ok'] or before['box']['used_bytes']+bound>109500000000:
    raise RuntimeError('insufficient bounded headroom')
artifact=Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
mp=artifact/'expert-manifest.json'
if hashlib.sha256(mp.read_bytes()).hexdigest()!='44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9':
    raise RuntimeError('native manifest identity changed')
manifest=load_expert_manifest(mp);records={(r.layer,r.expert):r for r in manifest.records}
lengths=tuple(s.length for s in records[0,0].segments)
if lengths!=(5898240,368640,5898240,368640,5898240,368640):raise RuntimeError('native record geometry changed')
if manifest.sidecar.file!='experts.bin':raise RuntimeError('one native sidecar required')
batches=tuple(tuple(records[i%40,(i*11+off)%384] for off in (0,137,277)) for i in range(512))
for b in batches:
 for r in b:
  if tuple(s.length for s in r.segments)!=lengths or r.sidecar_length!=18800640:
   raise RuntimeError('record geometry changed')
class Slot:
 def __init__(self):self.arrays=tuple(bytearray(n) for n in lengths)
 def record_views(self,record):return tuple(memoryview(a) for a in self.arrays)
 def weights_sha(self):
  h=hashlib.sha256()
  for i in (0,2,4):h.update(self.arrays[i])
  return h.hexdigest()
 def raw_sha(self):
  h=hashlib.sha256()
  for a in self.arrays:h.update(a)
  return h.hexdigest()
slots=tuple(Slot() for _ in range(3))
# Bind disjoint destination slices once. Each native weight plane is separated
# by a368640-byte scale gap in the original, unchanged sidecar.
plans={}
for mode in ('weights3','weights4'):
 per_slot=[]
 for slot in slots:
  if mode=='weights3':
   groups=[[(i*6266880,(memoryview(slot.arrays[2*i]),))] for i in range(3)]
  else:
   groups=[]
   for lo,hi in PositionalExpertReader._fanout_byte_ranges(3*5898240,4):
    jobs=[]
    for i in range(3):
     a=max(lo,i*5898240);b=min(hi,(i+1)*5898240)
     if b>a:
      start=a-i*5898240
      jobs.append((i*6266880+start,(memoryview(slot.arrays[2*i])[start:start+b-a],)))
    groups.append(jobs)
  per_slot.append(groups)
 plans[mode]=per_slot
report={'scope':'I/O-only, three concurrent records and real sidecar gaps; CPU bytearrays, not full-model overlap or decode TPS',
 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'initial_memory':before,'static_host_bound_bytes':bound,'batches':len(batches),'arms':[]}
expected=None
with concurrent.futures.ThreadPoolExecutor(max_workers=3,thread_name_prefix='scale-io-caller') as callers:
 for label,mode in [('control_before','stock'),('weights3_before','weights3'),('weights4','weights4'),('weights3_after','weights3'),('control_after','stock')]:
  reader=PositionalExpertReader(artifact,bypass_page_cache=True,use_native=False,io_read_fanout=4)
  try:
   def run_group(base,jobs):
    for off,views in jobs:
     reader._readv_range_into('experts.bin',base+off,views,cancel_event=None,deadline_ns=None)
   def read_weight_record(record,index):
    groups=plans[mode][index]
    reader.metrics.update(record_requests=1,records_read=1,sidecar_record_requests=1)
    futures=[reader._fanout_executor.submit(run_group,record.sidecar_offset,g) for g in groups[1:]]
    error=None
    try:run_group(record.sidecar_offset,groups[0])
    except BaseException as e:error=e
    for future in futures:
     try:future.result()
     except BaseException as e:
      if error is None:error=e
    if error is not None:raise error
   def batch_read(batch):
    if mode=='stock':
     fs=[callers.submit(reader.read_record_into,manifest,r,s,verify_hash=False) for r,s in zip(batch,slots)]
    else:fs=[callers.submit(read_weight_record,r,i) for i,r in enumerate(batch)]
    error=None
    for f in fs:
     try:f.result()
     except BaseException as e:
      if error is None:error=e
    if error is not None:raise error
   for b in batches[:4]:batch_read(b)
   reader.metrics=ExpertIOMetrics();started=time.perf_counter()
   for b in batches:batch_read(b)
   wall=time.perf_counter()-started;metrics=reader.metrics.as_dict()
   hashes=[s.weights_sha() for s in slots]
   if mode=='stock':
    if [s.raw_sha() for s in slots]!=[r.sha256 for r in batches[-1]]:raise RuntimeError('full-record hash mismatch')
    if expected is None:expected=hashes
   if hashes!=expected:raise RuntimeError('weight-only reads changed a byte')
   record_bytes=18800640 if mode=='stock' else 17694720
   if (metrics['records_read']!=len(batches)*3 or metrics['read_bytes']!=len(batches)*3*record_bytes
       or metrics['native_positional_calls']!=0 or any(metrics[k] for k in ['short_reads','integrity_errors','io_errors','cancellations','deadline_errors'])):
    raise RuntimeError('I/O accounting or coverage differs')
   row={'arm':label,'mode':mode,'wall_s':wall,'record_bytes':record_bytes,'metrics':metrics,
        'landed_weights_sha256':hashes,'memory_after':host_memory_snapshot()}
   report['arms'].append(row)
   print('SCALE_IO_ARM',json.dumps({'arm':label,'wall_s':wall,'read_bytes':metrics['read_bytes'],'preadv_calls':metrics['python_preadv_invocations']}),flush=True)
  finally:reader.close()
report['complete']=True;out.write_text(json.dumps(report,indent=2)+'\n')
