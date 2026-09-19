"""Bounded CPU-only screen for exact cross-expert delta compression."""
import fcntl
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import signal
import sys

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm'}:
            raise RuntimeError('MLX is forbidden in this CPU entropy screen')

sys.meta_path.insert(0, NoMLX())
if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('use the parent guard to avoid contending with other jobs')
import numpy as np
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

ROOT = Path('/tmp/dsv41-expert-delta-20260917')
ARTIFACT = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
OUT = ROOT/'screen.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite the screen')
signal.alarm(180)
before = host_memory_snapshot()
RESERVE = 2 * 1024**3
if before['box']['used_bytes'] + RESERVE > 109500000000:
    raise RuntimeError('CPU sample buffers and manifest lack bounded headroom')
manifest_blob = (ARTIFACT/'expert-manifest.json').read_bytes()
if hashlib.sha256(manifest_blob).hexdigest() != '44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9':
    raise RuntimeError('model manifest changed')
manifest = json.loads(manifest_blob)
del manifest_blob
records = {(r['layer'], r['expert']):r for r in manifest['records']}
if manifest['sidecar']['file'] != 'experts.bin' or len(records) != 40 * 384:
    raise RuntimeError('not the native target inventory')

def entropy(counts):
    counts = counts[counts > 0].astype(np.float64)
    p = counts/counts.sum()
    return float(-(p*np.log2(p)).sum())

def weight_stats(base, current):
    raw = np.bincount(current, minlength=256)
    xor = np.bincount(base ^ current, minlength=256)
    # A 16-context exact-symbol coder could condition each weight nibble on
    # the same-position reference nibble, without assuming XOR is optimal.
    joint = np.zeros(256, dtype=np.int64)
    for shift in (0, 4):
        joint += np.bincount((((base >> shift) & 15) << 4) |
                             ((current >> shift) & 15), minlength=256)
    table = joint.reshape(16, 16)
    h_cond = entropy(joint) - entropy(table.sum(axis=1))
    return {'raw_byte_entropy':entropy(raw), 'xor_byte_entropy':entropy(xor),
            'conditional_nibble_bits_per_byte':2*h_cond,
            'equal_byte_fraction':float(np.mean(base == current))}

fd = os.open(ARTIFACT/'experts.bin', os.O_RDONLY)
fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
report = {'scope':'Representative CPU entropy bounds only; no codec size, decoder speed or inference claim',
          'source_commit':os.environ.get('DSV41_SOURCE_COMMIT'),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'manifest_sha256':'44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9',
          'cpu_host_reserve_bytes':RESERVE, 'baseline':before, 'layers':[0,10,20,30,39],
          'reference_expert':0,'candidate_experts':[1,128],'record_hashes':{},'rows':[]}

def read_record(layer, expert):
    record = records[layer, expert]
    if record['logical_bytes'] != 18800640 or record['sidecar_length'] != 18800640:
        raise RuntimeError('native record geometry changed')
    data = os.pread(fd, record['sidecar_length'], record['sidecar_offset'])
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != record['logical_bytes'] or digest != record['sha256']:
        raise RuntimeError('short read or record identity mismatch')
    report['record_hashes'][f'{layer}:{expert}'] = digest
    return data

try:
    for layer in report['layers']:
        base_blob = read_record(layer, 0)
        for expert in report['candidate_experts']:
            current_blob = read_record(layer, expert)
            offset = 0
            components = []
            for segment in records[layer,expert]['segments']:
                length = segment['length']
                a = np.frombuffer(base_blob, dtype=np.uint8, count=length, offset=offset)
                b = np.frombuffer(current_blob, dtype=np.uint8, count=length, offset=offset)
                if segment['component'].endswith('.weight'):
                    stats = weight_stats(a,b)
                else:
                    joint = np.bincount(a.astype(np.int64)*256+b, minlength=65536).reshape(256,256)
                    stats = {'raw_byte_entropy':entropy(np.bincount(b,minlength=256)),
                             'xor_byte_entropy':entropy(np.bincount(a ^ b,minlength=256)),
                             'conditional_nibble_bits_per_byte':entropy(joint.ravel())-entropy(joint.sum(axis=1)),
                             'equal_byte_fraction':float(np.mean(a == b))}
                components.append({'component':segment['component'],'bytes':length,**stats})
                offset += length
            assert offset == len(current_blob)
            estimates = {name:sum(c['bytes']*c[name]/8 for c in components)
                         for name in ('raw_byte_entropy','xor_byte_entropy','conditional_nibble_bits_per_byte')}
            row = {'layer':layer,'expert':expert,'components':components,'ideal_payload_bytes':estimates}
            report['rows'].append(row)
            print('DELTA_SCREEN',json.dumps({'layer':layer,'expert':expert,**estimates}),flush=True)
            del current_blob,a,b
        del base_blob
finally:
    os.close(fd)
report['after'] = host_memory_snapshot()
report['complete'] = True
OUT.write_text(json.dumps(report,indent=2)+'\n')
