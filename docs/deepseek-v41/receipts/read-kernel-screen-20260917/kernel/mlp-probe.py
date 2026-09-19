"""Bounded native MXFP4 down-projection specialization screen, not a model run."""
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import time

ROOT = Path('/tmp/dsv41-down-tail-20260917')
OUT = ROOT/'mlp-probe.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite a measured receipt')
if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU window must own the exclusive lock')
signal.alarm(180)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
RESERVE = 6*1024**3
if not before['box']['ok'] or before['box']['used_bytes'] + RESERVE > 109500000000:
    raise RuntimeError('insufficient static CPU/Metal/cache/compiler headroom')
# Read-only model access; no model load. At most 36 full expert records, each
# 18,800,640 bytes, plus CPU copies and an MLX cache. The generous six-GiB
# allowance covers <1.5GiB tensor/copy peak, 256MiB allocator cache and 4GiB host/compiler.
import numpy as np
import mlx.core as mx
mx.set_memory_limit(2*1024**3)
mx.set_cache_limit(256*1024**2)
ARTIFACT=Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
blob=(ARTIFACT/'expert-manifest.json').read_bytes()
MANIFEST='44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9'
if hashlib.sha256(blob).hexdigest()!=MANIFEST:
    raise RuntimeError('native manifest identity changed')
records={r['expert']:r for r in json.loads(blob)['records'] if r['layer']==20}
del blob
cpu_arrays={}
for proj,n,k in [('gate_proj',2304,5120),('up_proj',2304,5120),('down_proj',5120,2304)]:
    cpu_arrays[proj+'.weight']=np.empty((36,n,k//8),np.uint32)
    cpu_arrays[proj+'.scales']=np.empty((36,n,k//32),np.uint8)
record_hashes={}
fd=os.open(ARTIFACT/'experts.bin',os.O_RDONLY)
fcntl.fcntl(fd,fcntl.F_NOCACHE,1)
try:
    for e in range(36):
        r=records[e]
        data=os.pread(fd,r['sidecar_length'],r['sidecar_offset'])
        digest=hashlib.sha256(data).hexdigest()
        if len(data)!=18800640 or digest!=r['sha256']:
            raise RuntimeError('record integrity failure')
        record_hashes[str(e)]=digest
        offset=0
        for s in r['segments']:
            target=cpu_arrays[s['component']]
            target[e]=np.frombuffer(data,dtype=target.dtype,count=target[e].size,offset=offset).reshape(target[e].shape)
            offset+=s['length']
        if offset!=len(data): raise RuntimeError('component geometry changed')
    del data
finally:
    os.close(fd)
bank={k:mx.array(v) for k,v in cpu_arrays.items()}
mx.eval(list(bank.values()))
del cpu_arrays,target
w,s=bank['down_proj.weight'],bank['down_proj.scales']
from mtplx.models.expert_mlx import _clamped_swiglu
HEADER='''
// Arithmetic derived from MLX v0.32.2 fp_quantized.h, Apache-2.0.
// Preserve eight values per lane, two left-associated four-value sums,
// nine 256-wide K steps, four output rows per SIMD group and SIMD sum.
inline float dsv_fp4(uchar bits) {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return float(bits & 8 ? -converted : converted);
}
inline float dsv_dot8(const device ushort* w, thread const float* x, float s) {
    float accum=0;
    for (int i=0;i<2;i++) {
        accum += (x[4*i]*dsv_fp4(uchar(w[i])) +
                  x[4*i+1]*dsv_fp4(uchar(w[i] >> 4)) +
                  x[4*i+2]*dsv_fp4(uchar(w[i] >> 8)) +
                  x[4*i+3]*dsv_fp4(uchar(w[i] >> 12)));
    }
    return s*accum;
}
'''
SOURCE='''
const uint lane=thread_index_in_simdgroup;
const uint outrow=threadgroup_position_in_grid.y*8 + simdgroup_index_in_threadgroup*4;
const uint assignment=threadgroup_position_in_grid.z;
const uint expert=ids[assignment];
const device ushort* wp=(const device ushort*)packed + size_t(expert)*5120*576 + outrow*576 + lane*2;
const device uchar* sp=scale_bytes + size_t(expert)*5120*72 + outrow*72 + lane/4;
const device T* xp=x + assignment*2304 + lane*8;
float result[4]={0,0,0,0};
UNROLL
for(int k=0;k<9;k++) {
    float xv[8];
    for(int i=0;i<8;i++) xv[i]=float(xp[k*256+i]);
    for(int r=0;r<4;r++) {
        uchar sb=sp[r*72+k*8];
        float scale=as_type<float>(sb==0 ? uint(0x400000) : (uint(sb)<<23));
        result[r] += dsv_dot8(wp+r*576+k*64,xv,scale);
    }
}
for(int r=0;r<4;r++) {
    result[r]=simd_sum(result[r]);
    if(lane==0) out[assignment*5120+outrow+r]=T(result[r]);
}
'''
kernels={name:mx.fast.metal_kernel(name='dsv41_down_'+name,
    input_names=['x','ids','packed','scale_bytes'],output_names=['out'],
    header=HEADER,source=SOURCE.replace('UNROLL',pragma))
    for name,pragma in [('fixed_loop','#pragma clang loop unroll(disable)'),('unroll9','#pragma unroll')]}
report={'kind':'bounded exact full expert-MLP comparison',
        'scope':'real layer20 weights, deterministic synthetic activations; no full-model throughput claim',
        'source_commit':os.environ['DSV41_SOURCE_COMMIT'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'manifest_sha256':MANIFEST,'record_hashes':record_hashes,
        'mlx_version':importlib.metadata.version('mlx'),
        'static_increment_bound_bytes':RESERVE,'memory_before':before,
        'source_url':'https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/kernels/fp_quantized.h',
        'upstream_sha256':hashlib.sha256((ROOT/'fp_quantized.h').read_bytes()).hexdigest(),
        'cases':[]}
rng=np.random.default_rng(419)
for rows,unique in [(18,3),(36,12),(36,36)]:
    ids=mx.array(np.arange(rows,dtype=np.uint32)%unique)
    # Small, broad finite activation values resemble a BF16 MLP output range.
    x=mx.array(rng.standard_normal((rows,5120)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(x,ids)
    def gu():
        values=x.reshape(rows,1,1,5120)
        def qmm(proj):
            return mx.gather_qmm(values,bank[proj+'.weight'],bank[proj+'.scales'],
                                rhs_indices=ids.reshape(rows,1),transpose=True,
                                group_size=32,bits=4,mode='mxfp4')
        return _clamped_swiglu(qmm('gate_proj'),qmm('up_proj'),10.0)
    def stock():
        return mx.gather_qmm(gu(),w,s,rhs_indices=ids.reshape(rows,1),
                            transpose=True,group_size=32,bits=4,mode='mxfp4').reshape(rows,5120)
    def custom(name):
        return kernels[name](inputs=[gu(),ids,w,s],template=[('T',mx.bfloat16)],
            grid=(32,1280,rows),threadgroup=(32,2,1),
            output_shapes=[(rows,5120)],output_dtypes=[mx.bfloat16])[0]
    ref=stock(); mx.eval(ref)
    ref_bits=np.array(ref.view(mx.uint16))
    case={'rows':rows,'unique_experts':unique,'parity':{},'timings':[]}
    passing=[]
    for name in kernels:
        y=custom(name); mx.eval(y)
        bits=np.array(y.view(mx.uint16))
        same=bool(np.array_equal(ref_bits,bits))
        case['parity'][name]={'exact_bytes':same,'different_elements':int(np.count_nonzero(ref_bits!=bits))}
        if same: passing.append(name)
    # Only successful exact candidates get performance timing. Interleave
    # unchanged controls to expose thermal/clock drift without a long suite.
    for _ in range(30): mx.eval(stock())
    for name in ['stock',*passing,'stock',*reversed(passing),'stock']:
        run=stock if name=='stock' else lambda name=name: custom(name)
        mx.eval(run())
        samples=[]
        for _ in range(15):
            t=time.perf_counter_ns(); mx.eval(run()); samples.append((time.perf_counter_ns()-t)/1e9)
        case['timings'].append({'arm':name,'samples_s':samples,'median_s':statistics.median(samples)})
    report['cases'].append(case)
    print('CASE',json.dumps(case),flush=True)
report['mlx_allocator_peak_bytes']=mx.get_peak_memory()
report['memory_after']=host_memory_snapshot()
report['complete']=True
OUT.write_text(json.dumps(report,indent=2)+'\n')
