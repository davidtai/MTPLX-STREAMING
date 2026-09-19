"""Prepare a two-layer scheduling screen without importing MLX or NumPy."""
from pathlib import Path
import array,ast,fcntl,gzip,hashlib,json,shutil,struct,subprocess,zipfile

ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
BASE=Path('/tmp/dsv41-packed-row-order-20260918')
CAPTURE=Path('/tmp/dsv41-110-stage/full-router-feature-20260918-v2.router-capture.npz')
LAYERS=(30,31)
RAW,WEIGHTS=18800640,17694720
for name in ('packed_storage.py','paired_kernels.py','kernels.py','library_identity.py'):
    shutil.copyfile(BASE/name,ROOT/name)
(ROOT/'artifact').symlink_to(BASE/'artifact',target_is_directory=True)
restore=(BASE/'restore_bank.py').read_text()
restore=restore.replace('single_pool=None):','single_pool=None, layer_id=0, prefetch_ring=None, prefetch_slots=0):')
restore=restore.replace('bank = LayerExpertSlotBank(**kwargs)',
    'bank = LayerExpertSlotBank(**kwargs, layer_id=layer_id, prefetch_ring=prefetch_ring, prefetch_slots=prefetch_slots)')
(ROOT/'restore_bank.py').write_text(restore)
lane=(BASE/'plane_lane.py').read_text()
start=lane.index('    def run(self, x, indices, *, shared_work):')
end=lane.index('\ndef install(',start)
method=lane[start:end]
method=method.replace('        parts = tuple(self.executor.parts)',
    '        parts = tuple(self.executor.parts)\n        remaining = len(parts)')
method=method.replace('        def submit_gu(part):',
    '        def submit_gu(part):\n            nonlocal remaining')
method=method.replace("            part.phase = 'submitted'", """            part.phase = 'submitted'
            remaining -= 1
            if remaining == 0:
                # Every current demand part has enqueued its down jobs before
                # publishing GU readiness. Speculative jobs follow those jobs.
                self.issue()""")
method=method.replace('            ready_iter = pending.iter_ready_misses()',
    '            if not parts:\n                self.issue()\n            ready_iter = pending.iter_ready_misses()')
extra='''
class IgnoreGUPublication:
    def publish_read_components(self, items):
        pass


class SpeculativeExecutor:
    def __init__(self, executor, local):
        self.executor, self.local = executor, local
        self.witness = IgnoreGUPublication()

    def submit(self, fn, *args, **kwargs):
        def execute():
            self.local.part = self.witness
            try:
                return fn(*args, **kwargs)
            finally:
                del self.local.part
        return self.executor.submit(execute)

    def shutdown(self, *args, **kwargs):
        return self.executor.shutdown(*args, **kwargs)


class PrefetchDecode(PackedDecode):
    def __init__(self, *args, issue, **kwargs):
        super().__init__(*args, **kwargs)
        self.issue = issue

'''+method+'\n'
lane=lane[:end]+extra+lane[end:]
lane=lane.replace('def install(runtime, switches, scales_by_layer, *, early=True):',
    'def install(runtime, switches, scales_by_layer, *, early=True, prefetch_source=None):')
lane=lane.replace('or c.decode_miss_records_per_part!=3 or c.prefetch_slots or c.resource_telemetry',
    'or c.decode_miss_records_per_part!=3 or c.prefetch_slots!=16 or c.resource_telemetry')
lane=lane.replace('    bind_reader(runtime.reader,local)',
    '    runtime._prefetch_executor = SpeculativeExecutor(runtime._prefetch_executor,local)\n    bind_reader(runtime.reader,local)')
old='        runner = PackedDecode(runtime,layer,PackedOps(scales_by_layer[layer]),executor,early=early)'
new='''        if prefetch_source is not None and layer == prefetch_source[0]:
            runner = PrefetchDecode(runtime,layer,PackedOps(scales_by_layer[layer]),executor,early=early,issue=prefetch_source[1])
        else:
            runner = PackedDecode(runtime,layer,PackedOps(scales_by_layer[layer]),executor,early=early)'''
assert lane.count(old)==1
lane=lane.replace(old,new)
(ROOT/'plane_lane.py').write_text(lane)

def header(f):
    assert f.read(6)==b'\x93NUMPY'
    v=f.read(2);n=struct.unpack('<H' if v==b'\x01\x00' else '<I',f.read(2 if v==b'\x01\x00' else 4))[0]
    return ast.literal_eval(f.read(n).decode())

predictions=[]
with CAPTURE.open('rb',buffering=0) as source:
    fcntl.fcntl(source.fileno(),48,1)
    with zipfile.ZipFile(source) as z:
        with z.open('scores.npy') as f:
            h=header(f);base=f.tell()
            assert h['shape']==(2,64,36,6,384) and h['descr']=='<f4'
            for cycle in range(64):
                candidates={}
                for row in range(6):
                    offset=(((64+cycle)*36+31-4)*6+row)*384*4
                    f.seek(base+offset)
                    values=array.array('f');values.frombytes(f.read(384*4))
                    top=sorted(range(384),key=lambda e:(-values[e],e))[:6]
                    for e in top[:4]:
                        gap=struct.unpack('<f',struct.pack('<f',values[e]-values[top[5]]))[0]
                        if gap>=struct.unpack('<f',struct.pack('<f',.1))[0]:
                            candidates[e]=max(candidates.get(e,float('-inf')),gap)
                predictions.append(sorted(candidates,key=lambda e:(-candidates[e],e)))
trace_path=REPO/'docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz'
with gzip.open(trace_path,'rt') as f:trace=json.load(f)
payload={'layers':LAYERS,'cycles':64,'routes':{str(l):trace['target_routes_by_layer'][str(l)][:64] for l in LAYERS},
    'initial_banks':{str(l):trace['initial_banks'][str(l)] for l in LAYERS},
    'predictions':predictions,'predictor_config':{'width':4,'margin':.1,'max_records':8},
    'capture_sha256':'5d8dd85c412f0c8e332733843e6eb9ed36ac6c8a8e5a7c71c8c2615e71edca3e',
    'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    'selection':'Target31 has the most useful predictions in the TRAINING half; evaluate timing only on cycles32..63.'}
(ROOT/'routes.json').write_text(json.dumps(payload,indent=2)+'\n')
proof=json.loads((BASE/'installation.json').read_text())
proof['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
proof['layers']=LAYERS;proof.pop('layer',None)
proof['capacity']=105;proof['bank_capacity']=2*105+48+16
proof['static_incremental_bound_bytes']=11*1024**3
proof['scope']='Two real packed expert layers, native105 persistent+48 shared transient+16 shared prefetch slots; first64 exact native routes and recorded causal predictions. Cycles32..63 decide timing. Excludes live router evaluation and attention; an optimistic I/O screen only.'
proof['arms']=['native','prefetch','native','prefetch','native']
proof['spec'].update(routed_layer_start=30,routed_layer_count=2,total_tensor_bytes=2*384*RAW+4096)
persist=2*105*RAW;pref=16*RAW;limit=7*1024**3
proof['plan'].update(total_limit_bytes=limit,expert_cache_limit_bytes=persist,persistent_budget_bytes=persist,
    persistent_slots=210,slots_per_layer=105,persistent_cache_bytes=persist,prefetch_ring_slots=16,prefetch_bytes=pref,
    unallocated_bytes=limit-persist-48*RAW-pref-4096)
proof['config'].update(memory_limit_bytes=limit,expert_cache_limit_bytes=persist,prefetch_slots=16)
proof['helper_sha256']={}
for name,digest in proof['runtime_source_sha256'].items():
    assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest,name
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'root':str(ROOT),'raw_bank_bound':proof['bank_capacity']*RAW,'packed_bank_bytes':proof['bank_capacity']*WEIGHTS,
                  'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],'source':proof['source_commit']}))
