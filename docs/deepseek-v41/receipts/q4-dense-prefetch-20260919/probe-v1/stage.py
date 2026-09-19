"""CPU-only construction inventory and immutable component run manifest."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
BASE = Path('/tmp/dsv41-packed-reader-pool-20260918')
sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
proof = json.loads((BASE/'installation.json').read_text())
for name in ('plane_lane.py','packed_storage.py','paired_kernels.py','kernels.py',
             'restore_bank.py','library_identity.py','routes.json'):
    assert sha(BASE/name) == proof['helper_sha256'][name],name
    shutil.copyfile(BASE/name,ROOT/name)
artifact = (BASE/'artifact').resolve()
if not (ROOT/'artifact').exists():
    (ROOT/'artifact').symlink_to(artifact,target_is_directory=True)
meta = json.loads((artifact/'manifest.json').read_text())
assert sha(artifact/'manifest.json') == proof['artifact_manifest_sha256']
model = Path(proof['model_path'])
assert sha(model/'expert-manifest.json') == meta['source_manifest_sha256']
inventory = json.loads((ROOT.parent/'inventory-result.json').read_text())
query = inventory['candidates']['wq_b']
assert query['uniform_layers'] and query['resident_bytes'] == 1730150400
proof['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
proof['arms'] = ['resident-110','stream-112','resident-110','stream-112','resident-110']
proof['query_tensors'] = query['sources']
proof['static_incremental_bound_bytes'] = 24*1024**3
proof['bound_components'] = {
    'mlx_active_cache_and_compile_envelope_bytes':8*1024**3,
    'host_python_reader_compiler_reserve_bytes':4*1024**3,
    'source_file_cache_conservative_reserve_bytes':12*1024**3,
    'raw_expert_bank_max_bytes':160*18800640,
    'packed_expert_bank_max_bytes':160*17694720,
    'query_control_payload_bytes':1730150400,
    'query_candidate_payload_bytes':86507520,
    'packed_scales_bound_bytes':100*1024**2,
    'retained_component_outputs_and_inputs_bound_bytes':256*1024**2,
    'inactive_allocator_cache_inside_envelope_bytes':256*1024**2,
    'unique_expert_source_bytes':384*18800640,
}
proof['scope'] = ('Component only: all 40 real native MXFP8 query layers rotate across 206 calls, '
    'combined with layer34 actual M6 expert routes and real packed Q4 expert I/O. Synthetic query '
    'inputs and saved route IDs with a query-dependent barrier; no attention or complete model. '
    'Continuous loop timing, output hashing outside timing. First four calls warm kernels. '
    'Resident110 versus two-buffer streamed112 redeems the whole-model payload saving into two '
    'extra slots in this one-layer component. This is not full-model throughput evidence.')
proof['bound_scope'] = ('24GiB incremental. Control retains 1.73GB query weights plus at most160 '
    'raw expert rows, then strips scales in place before steady execution. No weight copies on '
    'stream reads. One dense worker, original15 expert auxiliary workers. Only evaluated small '
    'query/output arrays retained. 8GiB strict Metal envelope includes inactive cache and compiler '
    'workspace;4GiB host/compiler plus12GiB conservative cache reserve covers all touched payload.')
proof['comparison'] = ('Query storage resident versus two rotating buffers, native MXFP8 qmm '
    'unchanged. Dense issue follows current demand submission. Ring retirement follows the '
    'query-dependent existing router eval. Original packed expert kernels, read pool, parts and '
    'policy unchanged except the112-slot memory-funded capacity.')
proof['touched_files'] = sorted({str(model/t['shard']) for t in query['sources']} | {str(model/'experts.bin')} |
    {str(artifact/component[field]['file']) for component in meta['layers'][proof['layer']]['components'].values()
     for field in ('descriptors','payload','bases')})
proof['sha256'] = {str(ROOT/name):sha(ROOT/name) for name in (
    'dense_io.py','probe.py','run_screen.py','stage.py','plane_lane.py','packed_storage.py',
    'paired_kernels.py','kernels.py','restore_bank.py','library_identity.py','routes.json')}
proof['sha256'].update({str(REPO/name):sha(REPO/name) for name in proof['runtime_source_sha256']})
proof['sha256'][str(model/'expert-manifest.json')] = sha(model/'expert-manifest.json')
proof['sha256'][str(artifact/'manifest.json')] = sha(artifact/'manifest.json')
proof['sha256'][proof['strict_allocator']['path']] = proof['strict_allocator']['sha256']
# Drop stale fields inherited from the previous, different experiment.
for name in ('helper_sha256','runtime_source_sha256','layer_selection','native_slot_bytes',
             'packed_slot_bytes','allocator_sources','bank_capacity'):
    proof.pop(name,None)
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text(),filename=str(path))
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
argv = ['env','GPU_WINDOW_CHILD_RSS_CAP_BYTES='+str(12*1024**3),
    'GPU_WINDOW_LOCK_TIMEOUT=120','GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000',
    'GPU_WINDOW_MIN_AVAIL_GB=24','GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'PYTHONHASHSEED=0','PYTHONUNBUFFERED=1','PYTHONPATH='+str(REPO)+':'+str(ROOT),
    'scripts/deepseek_v41/gpu_window.sh',
    '/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(ROOT/'run_screen.py')]
(ROOT/'command.sh').write_text(shlex.join(argv)+' > '+shlex.quote(str(ROOT/'guard.log'))+' 2>&1\n')
print(json.dumps({'source_commit':proof['source_commit'],'static_bound_bytes':proof['static_incremental_bound_bytes'],
                  'query_control_bytes':query['resident_bytes'],'query_candidate_bytes':query['two_buffer_bytes'],
                  'scope':proof['scope'],'mlx_imported':False}))
