"""Bounded read-only MXFP4 entropy screen, not a serving codec benchmark."""
import importlib.abc
import sys
class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mlx','mlx_lm'):
            raise ImportError('No MLX in entropy screen')
sys.meta_path.insert(0,NoMLX())
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zlib
import numpy as np
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

signal.alarm(120)
ROOT=Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
OUT=Path('/tmp/dsv41-110-preflight/expert-entropy-probe.json')
assert 0 <= float(os.environ['MTPLX_DSV41_BOX_BASELINE_GB']) <=20
manifest_path=ROOT/'expert-manifest.json'
assert manifest_path.stat().st_size < 40 * 1024**2
manifest=json.loads(manifest_path.read_text())
assert manifest['model_key']=='deepseek-v41-flash-expert-mxfp4'
records={(r['layer'],r['expert']):r for r in manifest['records']}
del manifest
bank=ROOT/'experts.bin';before=bank.stat()
assert before.st_size==288777830400
result={'kind':'read-only sampled entropy/compression screen; no serving speed or full-bank ratio claim',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'bound':'one18.8MB raw record,onecompressed+roundtrip record,<=151MB histogram scratch,metadata<512MiB;2GiB child cap; no MLX',
        'records':[],'snapshots':[host_memory_snapshot()]}
def entropy(hist):
    p=hist[hist>0]/hist.sum()
    return float(-np.sum(p*np.log2(p)))
fd=os.open(bank,os.O_RDONLY)
try:
    fcntl.fcntl(fd,48,1)  # Darwin F_NOCACHE, same bank mode as serving.
    for layer in (0,8,16,24,32,39):
        for expert in (0,191):
            r=records[layer,expert]
            assert r['logical_bytes']==r['sidecar_length']==18800640
            raw=os.pread(fd,r['sidecar_length'],r['sidecar_offset'])
            assert len(raw)==r['logical_bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
            view=np.frombuffer(raw,dtype=np.uint8)
            h=np.bincount(view,minlength=256)
            offset=0;parts=[]
            for s in r['segments']:
                n=s['length'];hist=np.bincount(view[offset:offset+n],minlength=256)
                parts.append({'component':s['component'],'bytes':n,'entropy_bits_per_byte':entropy(hist)})
                offset+=n
            assert offset==len(raw)
            t=time.perf_counter();compressed=zlib.compress(raw,level=1);compress_s=time.perf_counter()-t
            t=time.perf_counter();decoded=zlib.decompress(compressed);decompress_s=time.perf_counter()-t
            assert decoded==raw
            row={'layer':layer,'expert':expert,'raw_bytes':len(raw),'sha256':r['sha256'],
                 'whole_record_entropy_bits_per_byte':entropy(h),'components':parts,
                 'zlib1_bytes':len(compressed),'zlib1_ratio':len(raw)/len(compressed),
                 'zlib1_compress_s':compress_s,'zlib1_decompress_s':decompress_s,'roundtrip_exact':True}
            result['records'].append(row)
            result['snapshots'].append(host_memory_snapshot())
            print(json.dumps({k:row[k] for k in ('layer','expert','whole_record_entropy_bits_per_byte','zlib1_ratio','zlib1_decompress_s')}),flush=True)
            del raw,view,compressed,decoded
    after=bank.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    result['bank_identity_unchanged']=True
    result['ok']=True
finally:
    os.close(fd)
    result['snapshots'].append(host_memory_snapshot())
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
