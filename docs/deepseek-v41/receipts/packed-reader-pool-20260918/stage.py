"""Screen a six-worker packed-plane pool without changing physical read jobs."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = Path('/tmp/dsv41-prelaunch-hits-20260918').resolve()
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof = json.loads((BASE/'installation.json').read_text())
for name, expected in proof['runtime_source_sha256'].items():
    assert sha(REPO/name) == expected,name
for name, expected in proof['helper_sha256'].items():
    assert sha(BASE/name) == expected,name
    if name not in ('prelaunch.py','prelaunch_kernels.py'):
        shutil.copyfile(BASE/name,ROOT/name)
(ROOT/'artifact').symlink_to((BASE/'artifact').resolve(),target_is_directory=True)
p = ROOT/'probe.py';s = p.read_text()
s = s.replace('import prelaunch','from concurrent.futures import ThreadPoolExecutor')
old = '''        if mode == 'native':
            runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)
        else:
            runners = prelaunch.install(runtime,{LAYER:switch},{LAYER:scales},
                runtime_sha256=proof['runtime_source_sha256'])'''
new = '''        if reader._fanout_executor._max_workers != 15:
            raise RuntimeError('native auxiliary reader worker geometry differs')
        if mode == 'six-workers':
            # Construction boundary: seeding is complete, no read is in flight.
            reader._fanout_executor.shutdown(wait=True)
            reader._fanout_executor = ThreadPoolExecutor(max_workers=6,
                thread_name_prefix='dsv41-packed-six')
        result['reader_auxiliary_workers'] = reader._fanout_executor._max_workers
        runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)'''
assert s.count(old) == 1
s = s.replace(old,new)
s = s.replace("a['mode']=='prelaunch'", "a['mode']=='six-workers'")
s = s.replace('PRELAUNCH_HIT','PACKED_READER_POOL')
s = s.replace('One-layer native route replay for early persistent-hit computation.',
              'One-layer native route replay for six auxiliary plane-reader workers.')
p.write_text(s)
proof['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
proof['arms'] = ['native','six-workers','native','six-workers','native']
proof['comparison'] = 'Original15 auxiliary FIFO reader workers versus6 workers. Native callers, three whole-plane positional reads, bytes, buffers, all arithmetic, cache policy and readiness remain unchanged.'
proof['scope'] = 'Layer34 native110/48 M6, actual73 initial residents plus37 empty slots, all206 native routes. Synthetic BF16 inputs and pre-evaluated routes. Sum of call times with outside-timer hashing; not continuous/full-model TPS.'
proof['bound_scope'] = '9GiB incremental:5GiB Metal/cache/compile plus4GiB host/reader/compiler.158 raw rows, one packed-scale layer,16MiB scratch. Native15 workers stop before the replacement6-worker pool starts. No additional buffers or GPU roots.'
proof['allocator_sources']['rule'] = 'Pinned strict allocator unchanged. Native reader and all PackedDecode/kernel source unchanged; only construction-time FIFO executor capacity differs.'
proof['helper_sha256'] = {p.name:sha(p) for p in ROOT.iterdir() if p.is_file()
    and p.suffix in ('.py','.json') and p.name not in ('installation.json','stage.py')}
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
command = (BASE/'command.sh').read_text().replace(str(BASE),str(ROOT)).replace('/tmp/dsv41-prelaunch-hits-20260918',str(ROOT))
(ROOT/'command.sh').write_text(command)
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
assert (ROOT/'plane_lane.py').read_bytes() == (BASE/'plane_lane.py').read_bytes()
audit = {'source_commit':proof['source_commit'],'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],
    'native_reader_source_sha256':sha(REPO/'mtplx/expert_io.py'),
    'original_plane_source_sha256':sha(ROOT/'plane_lane.py'),
    'worker_count_before':15,'worker_count_after':6,
    'native_pool_formula':'min(64,max(4,(1+4)*(4-1))) =15; it reserves concurrency for four speculative records even though this installed lane has no prefetch.',
    'read_geometry':'Packed bind_reader sends three5898240B whole-plane reads per physical record. Pool-size change does not change range geometry or positional-call count. This differs from the earlier native fanout4/8 split-range experiment.',
    'ownership':'Existing ThreadPoolExecutor and shutdown(wait=True) semantics remain. All writers drain before full READY, and native GU publication, failure cleanup, policies, pins and deferred consumer lifetimes remain unchanged.',
    'full_model_promotion':False}
(ROOT/'construction.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
