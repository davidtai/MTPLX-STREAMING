"""CPU-only chronological label adaptation using the existing exact capture."""
import os
os.environ['VECLIB_MAXIMUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import fcntl
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX is forbidden in the CPU screen')

sys.meta_path.insert(0,NoMLX())
if os.environ.get('_GPU_WINDOW_LOCKED')!='1':
    raise RuntimeError('parent-held guard required for the CPU envelope')
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'screen.json'
if OUT.exists():raise RuntimeError('refusing evidence overwrite')
proof=json.loads((ROOT/'installation.json').read_text())
if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=proof['source_commit']:
    raise RuntimeError('source commit differs')
for name,field in (('screen.py','script_sha256'),('analysis.py','analysis_sha256')):
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=proof[field]:
        raise RuntimeError('CPU helper identity differs')
before=host_memory_snapshot()
if not before['box']['ok'] or before['box']['used_bytes']+1024**3>110000000000:
    raise RuntimeError('complete1GiB CPU envelope does not fit')
signal.alarm(120)
import numpy as np
from analysis import analyze

with open(proof['capture_path'],'rb',buffering=0) as f:
    fcntl.fcntl(f.fileno(),48,1)
    data=f.read(proof['capture_bytes']+1)
if len(data)!=proof['capture_bytes'] or hashlib.sha256(data).hexdigest()!=proof['capture_sha256']:
    raise RuntimeError('native capture changed')
with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
    raw=arrays['scores']
    actual=arrays['actual']
    nrows=arrays['nrows']
    persistent=arrays['persistent']
    physical=arrays['physical']
    reads=arrays['reads']
del data
if raw.shape!=(2,64,36,6,384) or not np.all(nrows[:,3:]==6):
    raise RuntimeError('native M6 geometry differs')
scores=np.empty((3,64,36,6,384),dtype=np.float32)
scores[0]=raw[1]
del raw
started=time.perf_counter()
layers=[]
for layer in range(36):
    z=scores[0,:16,layer].reshape(-1,384)
    labels=actual[:16,layer+4].reshape(-1,6)
    truth=np.zeros_like(z,dtype=bool)
    np.put_along_axis(truth,labels,True,axis=1)
    order=np.sort(z,axis=1)
    cut=(order[:,-6:-5]+order[:,-7:-6])*0.5
    target=np.where(truth,np.maximum(z,cut+proof['ranking_margin']),
                          np.minimum(z,cut-proof['ranking_margin']))
    residual=target-z
    mean=z.mean(axis=0)
    scale=np.maximum(z.std(axis=0),np.float32(0.1))
    offset=residual.mean(axis=0)
    design=(z-mean)/scale
    gram=design.T@design
    gram.flat[::385]+=np.float32(len(z)*proof['ridge'])
    coef=np.linalg.solve(gram,design.T@(residual-offset))
    all_z=scores[0,:,layer].reshape(-1,384)
    scores[1,:,layer]=(all_z+offset).reshape(64,6,384)
    scores[2,:,layer]=(all_z+((all_z-mean)/scale)@coef+offset).reshape(64,6,384)
    layers.append({'layer':layer+4,'adapter_bytes':sum(a.nbytes for a in (mean,scale,offset,coef)),
        'adapter_sha256':hashlib.sha256(b''.join(a.tobytes() for a in (mean,scale,offset,coef))).hexdigest()})
result=analyze(scores,actual,nrows,persistent,physical,reads)
report={'complete':True,'cpu_only':True,'source_commit':proof['source_commit'],
    'scope':proof['scope'],'training':proof['construction'],'capture_sha256':proof['capture_sha256'],
    'static_incremental_bound_bytes':1024**3,'before':before,'after':host_memory_snapshot(),
    'elapsed_s':time.perf_counter()-started,'layers':layers,'analysis':result}
OUT.write_text(json.dumps(report,indent=2)+'\n')
print('EXACT_ROUTE_ADAPTER',json.dumps({'complete':True,'elapsed_s':report['elapsed_s'],
    'families':{name:{'layers':len(f['selected_layers']),'totals':f['totals']} for name,f in result['families'].items()}}),flush=True)
