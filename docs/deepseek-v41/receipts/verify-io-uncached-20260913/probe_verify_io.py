"""Bounded real-record CPU I/O probe; no MLX or model execution."""
import importlib.abc
import sys

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mlx', 'mlx_lm'):
            raise ImportError('No MLX permitted in the CPU I/O probe')
sys.meta_path.insert(0, NoMLX())

import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_manifest import load_expert_manifest
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

signal.alarm(120)
ROOT = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
OUT = Path('/tmp/dsv41-110-preflight/verify-io-probe.json')
RECORD_BYTES = 18800640
BATCH = 8
BATCHES = 256
assert 0 <= float(os.environ['MTPLX_DSV41_BOX_BASELINE_GB']) <= 20
assert (ROOT / 'expert-manifest.json').stat().st_size < 40 * 1024**2
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
bank = ROOT / 'experts.bin'
identity = bank.stat()
assert identity.st_size == 288777830400
snapshots = []
stop = threading.Event()
def sample_loop():
    while not stop.wait(.25):
        snapshots.append(host_memory_snapshot())
snapshots.append(host_memory_snapshot())
monitor = threading.Thread(target=sample_loop, daemon=True)
monitor.start()
result = {'source_commit':source, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'kind':'uncached CPU I/O proxy, not model TPS', 'record_bytes':RECORD_BYTES,
          'records_per_batch':BATCH, 'batches_per_sample':BATCHES,
          'bound':'8 records=150405120B; manifest<40MiB, <=512MiB parsed metadata, <=1GiB other CPU overhead; 2GiB child cap, no MLX',
          'samples':[]}
try:
    manifest = load_expert_manifest(ROOT / 'expert-manifest.json')
    assert manifest.model_key == 'deepseek-v41-flash-expert-mxfp4'
    records = {(r.layer,r.expert):r for r in manifest.records}
    assert len(records) == 40 * 384
    exemplar = records[0,0]
    lengths = tuple(s.length for s in exemplar.segments)
    assert sum(lengths) == RECORD_BYTES
    assert all(r.logical_bytes == RECORD_BYTES and tuple(s.length for s in r.segments)==lengths for r in records.values())
    class Destination:
        def __init__(self):
            self.views = tuple(memoryview(bytearray(n)) for n in lengths)
        def record_views(self, record):
            # The reader owns/releases the views returned for this call.
            return tuple(memoryview(v) for v in self.views)
        def digest(self):
            h=hashlib.sha256()
            for v in self.views: h.update(v)
            return h.hexdigest()
    destinations = [Destination() for _ in range(BATCH)]
    rng = random.Random(20260913)
    batches = [[records[i % 40, e] for e in rng.sample(range(384), BATCH)] for i in range(BATCHES)]
    result['batch_ids_sha256'] = hashlib.sha256(json.dumps([[(r.layer,r.expert) for r in batch] for batch in batches]).encode()).hexdigest()
    # Interleaved controls, identical locations, fixed caller concurrency.
    for fanout in (4,1,2,8,8,2,1,4):
        with PositionalExpertReader(ROOT, use_native=False, bypass_page_cache=True,
                                    io_read_fanout=fanout) as reader, ThreadPoolExecutor(max_workers=BATCH) as pool:
            def read_batch(batch):
                futures = [pool.submit(reader.read_record_into, manifest, record, dst,
                                       verify_hash=False, pipeline_phase='decode')
                           for record,dst in zip(batch,destinations)]
                for f in futures: f.result()
            for batch in batches[:8]: read_batch(batch)
            before = reader.metrics.as_dict()
            t0=time.perf_counter()
            for batch in batches: read_batch(batch)
            wall=time.perf_counter()-t0
            after=reader.metrics.as_dict()
            delta={k:after[k]-before[k] for k in ('read_bytes','read_wall_ns','read_ns','read_operations','python_preadv_invocations','short_reads','io_errors')}
            assert delta['read_bytes'] == BATCHES * BATCH * RECORD_BYTES
            assert delta['short_reads']==delta['io_errors']==0
            digests=[d.digest() for d in destinations]
            assert all(d==r.sha256 for d,r in zip(digests,batches[-1]))
            row={'fanout':fanout,'wall_s':wall,'aggregate_gb_s':delta['read_bytes']/wall/1e9,
                 'io_window_gb_s':delta['read_bytes']/(delta['read_wall_ns']/1e9)/1e9,
                 'realized_qd':delta['read_ns']/delta['read_wall_ns'], 'metrics_delta':delta,
                 'last_batch_digests':digests, 'last_batch_matches_manifest':True}
            result['samples'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('metrics_delta','last_batch_digests')}),flush=True)
    after=bank.stat()
    assert (identity.st_dev,identity.st_ino,identity.st_size,identity.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    result['bank_identity_unchanged']=True
    result['ok']=True
finally:
    stop.set(); monitor.join(2)
    snapshots.append(host_memory_snapshot())
    result['physical_peak_bytes']=max(s['box']['used_bytes'] for s in snapshots)
    result['process_footprint_peak_bytes']=max(s['process']['phys_footprint_bytes'] for s in snapshots)
    result['swapouts_first']=snapshots[0]['box']['swapouts_pages']
    result['swapouts_last']=snapshots[-1]['box']['swapouts_pages']
    result['snapshots']=snapshots
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    assert not any(k=='mlx' or k.startswith('mlx.') for k in sys.modules)
