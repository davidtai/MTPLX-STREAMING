"""Bounded CPU inventory for globally resident, exact row-packed scales."""
import fcntl
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import signal
import sys
import time

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx','mlx_lm'}:
            raise RuntimeError('CPU scale screen must not import MLX')
sys.meta_path.insert(0,NoMLX())
if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent guard required for bounded memory and lane ownership')
signal.alarm(180)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before=host_memory_snapshot()
BOUND=2*1024**3
if not before['box']['ok'] or before['box']['used_bytes']+BOUND>109500000000:
    raise RuntimeError('insufficient bounded host headroom')
import numpy as np
from codec import pack_rows,unpack_rows
root=Path('/tmp/dsv41-resident-scales-20260917');out=root/'screen.json'
if out.exists():raise RuntimeError('refusing to overwrite evidence')
model=Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
blob=(model/'expert-manifest.json').read_bytes()
manifest_hash=hashlib.sha256(blob).hexdigest()
if manifest_hash!='44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9':
    raise RuntimeError('native manifest changed')
manifest=json.loads(blob);del blob
records={(r['layer'],r['expert']):r for r in manifest['records']}
if len(records)!=40*384:raise RuntimeError('not the native target inventory')
report={'kind':'CPU exact row-packed scale sample; no Metal or inference timing',
        'source_commit':os.environ['DSV41_SOURCE_COMMIT'],'manifest_sha256':manifest_hash,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'codec_sha256':hashlib.sha256((root/'codec.py').read_bytes()).hexdigest(),
        'static_host_bound_bytes':BOUND,'memory_before':before,'experts_per_layer':[0,127,255,383],
        'rows':[],'read_bytes':0}
fd=os.open(model/'experts.bin',os.O_RDONLY);fcntl.fcntl(fd,fcntl.F_NOCACHE,1)
start=time.perf_counter()
try:
 for layer in range(40):
  for expert in report['experts_per_layer']:
   r=records[layer,expert]
   data=os.pread(fd,r['sidecar_length'],r['sidecar_offset'])
   digest=hashlib.sha256(data).hexdigest()
   if len(data)!=18800640 or digest!=r['sha256']:raise RuntimeError('native record integrity failure')
   report['read_bytes']+=len(data)
   row={'layer':layer,'expert':expert,'record_sha256':digest,'components':[]}
   offset=0
   for segment in r['segments']:
    if segment['component'].endswith('.scales'):
     columns=72 if segment['component'].startswith('down_') else 160
     a=np.frombuffer(data,np.uint8,count=segment['length'],offset=offset).reshape(-1,columns)
     descriptors,payload=pack_rows(a)
     decoded=unpack_rows(descriptors,payload,columns)
     if not np.array_equal(a,decoded):raise RuntimeError('scale codec changed a byte')
     widths=(descriptors>>8)&15
     row['components'].append({'name':segment['component'],'raw_bytes':a.nbytes,
       'descriptor_bytes':descriptors.nbytes,'payload_bytes':payload.nbytes,
       'width_row_counts':{str(b):int(np.count_nonzero(widths==b)) for b in [0,1,2,4,8]},
       'exact_bytes':True})
    offset+=segment['length']
   if offset!=len(data):raise RuntimeError('record geometry changed')
   row['packed_bytes']=sum(c['descriptor_bytes']+c['payload_bytes'] for c in row['components'])+3*4
   report['rows'].append(row)
  print('SCALE_LAYER',layer,'packed_mean',sum(x['packed_bytes'] for x in report['rows'][-4:])//4,flush=True)
finally:os.close(fd)
report['elapsed_s']=time.perf_counter()-start
sizes=[r['packed_bytes'] for r in report['rows']]
report['sample_record_packed_mean_bytes']=sum(sizes)/len(sizes)
report['sample_record_packed_min_bytes']=min(sizes)
report['sample_record_packed_max_bytes']=max(sizes)
report['projected_all_target_scale_bytes']=report['sample_record_packed_mean_bytes']*40*384
report['all_target_bound_using_sample_max_bytes']=max(sizes)*40*384
report['projection_is_not_full_inventory_bound']=True
report['raw_scale_bytes_per_record']=1105920
report['existing_cap100_persistent_scale_bytes']=1105920*40*100
report['existing_transient48_scale_bytes']=1105920*48
report['memory_after']=host_memory_snapshot()
report['complete']=True
out.write_text(json.dumps(report,indent=2)+'\n')
print('SCALE_RESULT',json.dumps({k:report[k] for k in ['elapsed_s','read_bytes','sample_record_packed_mean_bytes','sample_record_packed_max_bytes','projected_all_target_scale_bytes','all_target_bound_using_sample_max_bytes','existing_cap100_persistent_scale_bytes']}),flush=True)
