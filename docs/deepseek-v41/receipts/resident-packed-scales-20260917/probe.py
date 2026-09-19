"""Small real-weight MLP comparison for lossless resident packed scales."""
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import time
if os.environ.get('_GPU_WINDOW_LOCKED')!='1':raise RuntimeError('parent GPU guard required')
signal.alarm(180)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before=host_memory_snapshot();BOUND=6*1024**3
if not before['box']['ok'] or before['box']['used_bytes']+BOUND>109500000000:
    raise RuntimeError('static CPU/Metal/cache/compiler bound does not fit')
root=Path('/tmp/dsv41-resident-scales-20260917');out=root/'probe.json'
if out.exists():raise RuntimeError('refusing to overwrite evidence')
import numpy as np
import mlx.core as mx
from codec import pack_rows
from kernels import make_projection
from mtplx.models.expert_mlx import _clamped_swiglu
mx.set_memory_limit(2*1024**3);mx.set_cache_limit(256*1024**2)
model=Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
blob=(model/'expert-manifest.json').read_bytes();mh=hashlib.sha256(blob).hexdigest()
if mh!='44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9':
    raise RuntimeError('native manifest changed')
records={r['expert']:r for r in json.loads(blob)['records'] if r['layer']==20};del blob
geometry={'gate_proj':(2304,5120),'up_proj':(2304,5120),'down_proj':(5120,2304)}
cpu={}
for proj,(n,k) in geometry.items():
    cpu[proj+'.weight']=np.empty((36,n,k//8),np.uint32)
    cpu[proj+'.scales']=np.empty((36,n,k//32),np.uint8)
fd=os.open(model/'experts.bin',os.O_RDONLY);fcntl.fcntl(fd,fcntl.F_NOCACHE,1)
hashes={}
try:
    for e in range(36):
        r=records[e];data=os.pread(fd,r['sidecar_length'],r['sidecar_offset'])
        h=hashlib.sha256(data).hexdigest()
        if len(data)!=18800640 or h!=r['sha256']:raise RuntimeError('native record integrity failure')
        hashes[str(e)]=h;offset=0
        for s in r['segments']:
            a=cpu[s['component']]
            a[e]=np.frombuffer(data,dtype=a.dtype,count=a[e].size,offset=offset).reshape(a[e].shape)
            offset+=s['length']
        if offset!=len(data):raise RuntimeError('record geometry differs')
    del data,a
finally:os.close(fd)
compressed={}
for proj in geometry:
    desc=[];payload=[];bases=[];offset=0
    for e in range(36):
        d,p=pack_rows(cpu[proj+'.scales'][e])
        desc.append(d);payload.append(p);bases.append(offset);offset+=len(p)
    compressed[proj]=(mx.array(np.stack(desc)),mx.array(np.concatenate(payload)),mx.array(bases,dtype=mx.uint32))
bank={k:mx.array(v) for k,v in cpu.items()};mx.eval(bank,compressed)
del cpu,desc,payload,bases,d,p
kernels={(proj,mode):make_projection(n,k,mode=='packed') for proj,(n,k) in geometry.items() for mode in ['raw_fixed','packed']}
report={'kind':'exact native expert-MLP scale codec comparison',
 'scope':'real layer20 records; deterministic synthetic BF16 inputs; no target inference or full-resident-scale-bank timing',
 'source_commit':os.environ['DSV41_SOURCE_COMMIT'],'manifest_sha256':mh,'record_hashes':hashes,
 'mlx_version':importlib.metadata.version('mlx'),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'helper_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['codec.py','kernels.py']},
 'static_increment_bound_bytes':BOUND,'memory_before':before,
 'raw_scale_bytes':sum(bank[p+'.scales'].nbytes for p in geometry),
 'packed_scale_bytes':sum(a.nbytes for group in compressed.values() for a in group),'cases':[]}
rng=np.random.default_rng(419)
for rows,unique in [(18,3),(36,12),(36,36)]:
    ids=mx.array(np.arange(rows,dtype=np.uint32)%unique)
    x=mx.array(rng.standard_normal((rows,5120)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(x,ids)
    def mlp(mode):
        def project(values,proj):
            n,k=geometry[proj]
            if mode=='stock':
                return mx.gather_qmm(values,bank[proj+'.weight'],bank[proj+'.scales'],
                    rhs_indices=ids.reshape(rows,1),transpose=True,group_size=32,bits=4,mode='mxfp4')
            desc,payload,bases=compressed[proj]
            return kernels[proj,mode](inputs=[values,ids,bank[proj+'.weight'],bank[proj+'.scales'],desc,payload,bases],
                template=[('T',mx.bfloat16)],grid=(32,n//4,rows),threadgroup=(32,2,1),
                output_shapes=[(rows,1,1,n)],output_dtypes=[mx.bfloat16])[0]
        values=x.reshape(rows,1,1,5120)
        hidden=_clamped_swiglu(project(values,'gate_proj'),project(values,'up_proj'),10.0)
        return project(hidden,'down_proj').reshape(rows,5120)
    ref=mlp('stock');mx.eval(ref);ref_bits=np.array(ref.view(mx.uint16))
    case={'rows':rows,'unique_experts':unique,'parity':{},'timings':[]}
    passing=[]
    for mode in ['raw_fixed','packed']:
        y=mlp(mode);mx.eval(y);bits=np.array(y.view(mx.uint16));exact=bool(np.array_equal(bits,ref_bits))
        case['parity'][mode]={'exact_bytes':exact,'different_elements':int(np.count_nonzero(bits!=ref_bits))}
        if exact:passing.append(mode)
    for _ in range(30):mx.eval(mlp('stock'))
    for mode in ['stock',*passing,'stock',*reversed(passing),'stock']:
        mx.eval(mlp(mode));samples=[]
        for _ in range(15):
            t=time.perf_counter_ns();mx.eval(mlp(mode));samples.append((time.perf_counter_ns()-t)/1e9)
        case['timings'].append({'arm':mode,'samples_s':samples,'median_s':statistics.median(samples)})
    report['cases'].append(case)
    print('SCALE_KERNEL_CASE',json.dumps({'rows':rows,'unique':unique,'parity':case['parity'],
      'arm_medians':{mode:statistics.median(t['median_s'] for t in case['timings'] if t['arm']==mode) for mode in ['stock',*passing]}}),flush=True)
report['mlx_allocator_peak_bytes']=mx.get_peak_memory()
report['memory_after']=host_memory_snapshot();report['complete']=True
out.write_text(json.dumps(report,indent=2)+'\n')
