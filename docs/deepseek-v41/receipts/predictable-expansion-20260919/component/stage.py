"""Stage a source-pinned, bounded component without importing MLX."""
import ast
import hashlib
import importlib.abc
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX import forbidden while staging')

sys.meta_path.insert(0,NoMLX())
from mtplx.expert_manifest import load_expert_manifest

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
BASE = Path('/tmp/dsv41-q4-dense-prefetch-20260919/probe-v1')
CONVERTER = REPO/'docs/deepseek-v41/receipts/woa-fused-transpose-20260918'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((BASE/'installation.json').read_text())
assert not (ROOT/'probe.json').exists()
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True)
for path,digest in prior['sha256'].items():
    assert sha(Path(path)) == digest,path
for name in ('plane_lane.py','packed_storage.py','paired_kernels.py','kernels.py',
             'restore_bank.py','library_identity.py','routes.json'):
    shutil.copyfile(BASE/name,ROOT/name)
shutil.copyfile(CONVERTER/'fused_transpose.py',ROOT/'fused_transpose.py')
shutil.copyfile('/tmp/dsv41-ridge-prefetch-20260919/v2/wait_and_run.py',ROOT/'wait_and_run.py')
artifact = (BASE/'artifact').resolve()
if not (ROOT/'artifact').exists():
    (ROOT/'artifact').symlink_to(artifact,target_is_directory=True)
model = Path(prior['model_path'])
manifest = load_expert_manifest(model/'expert-manifest.json')
names = {f'layers.{layer}.attn.{part}.{field}' for layer in range(40)
         for part in ('wo_a','wo_b') for field in ('weight','scales')}
kept = tuple(t for t in manifest.resident_tensors if t.tensor in names)
assert len(kept) == 160 and sum(t.length for t in kept) == 3_114_270_720
shard_names = {t.shard for t in kept}
shards = {s.name:s.size for s in manifest.shards if s.name in shard_names}
assert max(shards.values()) <= 200_000_000
budget = {
    'metal_active_cache_compiler_envelope_bytes':10*1024**3,
    'host_reader_compiler_reserve_bytes':4*1024**3,
    'source_file_cache_reserve_bytes':12*1024**3,
    'raw_expert_bank_max_bytes':159*18_800_640,
    'packed_expert_bank_max_bytes':159*17_694_720,
    'control_bf16_wo_a_bytes':40*67_108_864,
    'candidate_packed_wo_a_bytes':40*34_603_008,
    'wo_b_bytes':40*43_253_760,
    'candidate_two_live_expansions_bytes':2*67_108_864,
    'candidate_expansion_replacement_peak_bytes':3*67_108_864,
    'conservative_projection_payload_credit_bytes':40*67_108_864-40*34_603_008-3*67_108_864,
    'one_uniform_packed_slot_bytes':40*17_694_720,
    'one_scale_layer_bound_bytes':100*1024**2,
    'inputs_retained_outputs_and_small_scratch_bytes':256*1024**2,
    'cache_inside_metal_envelope_bytes':256*1024**2,
    'construction_native_conversion_scratch_bytes':3*67_108_864,
    'compiler_and_extra_metal_headroom_bytes':2*1024**3,
    'unique_expert_source_bytes':384*18_800_640,
}
assert budget['conservative_projection_payload_credit_bytes'] > budget['one_uniform_packed_slot_bytes']
payload_peak = sum(budget[k] for k in ('raw_expert_bank_max_bytes','control_bf16_wo_a_bytes',
    'wo_b_bytes','one_scale_layer_bound_bytes','inputs_retained_outputs_and_small_scratch_bytes',
    'cache_inside_metal_envelope_bytes','construction_native_conversion_scratch_bytes',
    'compiler_and_extra_metal_headroom_bytes'))
assert payload_peak < budget['metal_active_cache_compiler_envelope_bytes']
assert budget['unique_expert_source_bytes']+sum(t.length for t in kept)+budget['one_scale_layer_bound_bytes'] < budget['source_file_cache_reserve_bytes']

reader = (CONVERTER/'attention_reader.py').read_text()
old = "if len(kept) != 20 or sum(t.length for t in kept) != 389283840 or {t.tensor for t in kept} != set(proof['resident_names']):"
assert reader.count(old) == 1
reader = reader.replace(old,"if len(kept) != 160 or sum(t.length for t in kept) != 3114270720 or {t.tensor for t in kept} != set(proof['resident_names']):")
(ROOT/'attention_reader.py').write_text(reader)

code = (BASE/'probe.py').read_text()
def replace(old,new):
    global code
    assert code.count(old) == 1,old
    code = code.replace(old,new)
replace('Continuous native Q4 query/expert component replay with bounded prefetch.',
        'Continuous Q4 output-projection expansion during expert I/O; no dense decode reads.')
replace('from dense_io import QueryStore','from output_store import OutputStore')
code = code.replace('8*1024**3','10*1024**3')
replace('del full_manifest','')
replace('capacity = 110 if resident else 112','capacity = 110 if resident else 111')
replace("store = QueryStore(model, proof['query_tensors'], mx=mx, resident=resident)\n        result['query_backing_bytes'] = sum(a.nbytes for b in store.buffers for a in b.values())",
        "store = OutputStore(model, full_manifest, proof, resident=resident)\n        result['projection_storage_mode'] = 'cached-bf16' if resident else 'resident-packed-next-bf16'")
replace('(1,6,1280)','(1,6,32768)')
replace("acquire = (lambda step: store.buffers[step%40]) if resident else store.acquire",
        "acquire = store.resident_acquire if resident else store.acquire")
replace('store.start()','store.issue(0)')
replace("q = mx.quantized_matmul(qi, weights['weight'], weights['scales'],\n                                    transpose=True, group_size=32, bits=8, mode='mxfp8')\n            x = q[...,:5120]",
        "q = store.project(qi,weights,call)\n            x = q")
replace('dense_read_bytes=0 if resident else len(routes)*43253760,','dense_read_bytes=0,')
replace("if a['mode']=='stream-112'","if a['mode']=='expand-111'")
replace('streamed query or expert output differs from resident control',
        'predictably expanded output projection or expert output differs from resident control')
code = code.replace('DENSE_PREFETCH_','PREDICTABLE_EXPANSION_')
(ROOT/'probe.py').write_text(code)

supervisor = (BASE/'run_screen.py').read_text()
needle = "proof = json.loads((ROOT/'installation.json').read_text())\n"
assert supervisor.count(needle) == 1
supervisor = supervisor.replace(needle,needle+"if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() != proof['source_commit']:\n    raise RuntimeError('source commit differs from pinned candidate')\nif subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True):\n    raise RuntimeError('tracked source is dirty')\n")
(ROOT/'run_screen.py').write_text(supervisor)
source_hashes = {path:digest for path,digest in prior['sha256'].items() if path.startswith(str(REPO))}
proof = {key:prior[key] for key in ('model_path','spec','plan','config','layer','strict_allocator',
                                    'artifact_manifest_sha256','box_budget_bytes') if key in prior}
proof.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    arms=['resident-110','expand-111','resident-110'],
    static_incremental_bound_bytes=26*1024**3,bound_components=budget,
    resident_names=sorted(names),shards=shards,
    scope='One real expert layer34 replay and all40 real output projections. Fixed M6 synthetic BF16 inputs and saved206 routes, no attention. Native cached BF16 output path versus exact one-layer-ahead expansion from resident MXFP8. Candidate111 slots redeems payload credit only; full-model admission and TPS remain unproved.',
    bound_scope='26GiB incremental:10GiB Metal/cache/compiler,4GiB host/reader/compiler,12GiB conservative touched source cache. Payload entries are inside these envelopes. One runtime at a time; no recurring dense SSD reads.',
    predecessor={'path':str(BASE/'probe.json'),'sha256':sha(BASE/'probe.json'),
                 'difference':'Keep packed output weights resident and overlap exact expansion; no dense SSD worker.'},
    converter={'source':str(CONVERTER/'fused_transpose.py'),'sha256':sha(CONVERTER/'fused_transpose.py'),
               'prior_exact_layers':[0,2,3,20,24]})
meta = json.loads((artifact/'manifest.json').read_text())
proof['touched_files'] = sorted({str(model/name) for name in shards}|{str(model/'experts.bin')}|
    {str(artifact/component[field]['file']) for component in meta['layers'][proof['layer']]['components'].values()
     for field in ('descriptors','payload','bases')})
for p in ROOT.glob('*.py'):
    ast.parse(p.read_text(),filename=str(p))
proof['sha256'] = {str(p):sha(p) for p in ROOT.iterdir() if p.is_file() and p.name!='installation.json'}
proof['sha256'].update(source_hashes)
for path in (model/'expert-manifest.json',artifact/'manifest.json',Path(proof['strict_allocator']['path'])):
    proof['sha256'][str(path)] = sha(path)
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
argv = ['env','GPU_WINDOW_CHILD_RSS_CAP_BYTES='+str(14*1024**3),
    'GPU_WINDOW_LOCK_TIMEOUT=600','GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000',
    'GPU_WINDOW_MIN_AVAIL_GB=26','GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'PYTHONHASHSEED=0','PYTHONUNBUFFERED=1','PYTHONPATH='+str(REPO)+':'+str(ROOT),
    'scripts/deepseek_v41/gpu_window.sh',
    '/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(ROOT/'run_screen.py')]
(ROOT/'command.sh').write_text(shlex.join(argv)+' > '+shlex.quote(str(ROOT/'guard.log'))+' 2>&1\n')
print(json.dumps({'source_commit':proof['source_commit'],'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],
    'payload_peak_bound_bytes':payload_peak,'payload_credit_bytes':budget['conservative_projection_payload_credit_bytes'],
    'projection_source_bytes':sum(t.length for t in kept),'cpu_only':True,'mlx_imported':False}))
