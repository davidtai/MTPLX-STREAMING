"""Stage one native/packed/native head-only comparison with unchanged weights."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
base = Path('/tmp/dsv41-draft-surrogates-20260918/v2').resolve()
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((base/'installation.json').read_text())
for p,h in prior['runtime_source_sha256'].items():
    assert sha(Path(p)) == h,p
for name in ('head_screen.py','run_head.py','lookup.py','library_identity.py'):
    assert sha(base/name) == prior['helper_sha256'][name],name
for name in ('run_head.py','lookup.py','library_identity.py','selection.json','control-commit-lengths.json'):
    shutil.copy2(base/name,root/name)
s = (base/'head_screen.py').read_text().replace(str(base),str(root))
start = s.index('from pruning import alias_slots')
end = s.index('    caches = [ds.DSparkStageCache',start)
s = s[:start] + '''from projection import Installation
control_lengths = json.loads((CURRENT_ROOT/'control-commit-lengths.json').read_text())
del weights['full']
gc.collect();mx.synchronize();mx.clear_cache()
projection = Installation(owner)
report['projection_installations'] = []
report['target_execution'] = False
report['physically_compacted'] = False
report['training_cycles'] = 0
report['heldout_start_position'] = 0
for sequence, mode in enumerate(('hybrid_m8','packed_f32','hybrid_m8')):
    report['projection_installations'].append(projection.select(mode=='packed_f32'))
''' + s[end:]
needle = '    started=time.perf_counter()'
assert s.count(needle)==1
s = s.replace(needle, '''    # Draft attention does not commit KV. Warm this fixed shape before timing.
    warm = owner.mtp.draft_block(initial_main,mx.array([ids[0]]),caches,owner.model.embed_tokens,owner.head)
    mx.eval(warm)
    del warm
    started=time.perf_counter()''')
s = s.replace("    row={'mode':mode", "    row={'mode':mode,'sequence':sequence,'active_after_replay_bytes':int(mx.get_active_memory())")
s = s.replace("print('DRAFT_SURROGATE_ARM'", "print('DRAFT_PACKED_ARM'")
(root/'head_screen.py').write_text(s)
scope = 'Native/packed-FP32/native draft-only trajectories; same93/58/32 experts, native inverse RoPE, packedMXFP8 output projection withFP32 input/output, nativewo_b. Target trunk never loads. Future target IDs only score proposals; full parity/TPS unmeasured.'
proof = dict(prior)
proof.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    scope=scope, training_cycles=0, selection_rule='No selection change; all183 native compact experts and original LUTs retained.',
    preceding_script_sha256=sha(base/'head_screen.py'), physically_compacted=False,
    removable_dense_cache_bytes=3*134217728, previous_head_quality_reused=False)
proof['helper_sha256'] = {p.name:sha(p) for p in root.glob('*.py')}
for p in root.glob('*.py'):ast.parse(p.read_text())
(root/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
command = (base/'command.sh').read_text().replace(str(base),str(root)).replace('GPU_WINDOW_LOCK_TIMEOUT=120','GPU_WINDOW_LOCK_TIMEOUT=600')
(root/'command.sh').write_text(command)
waiter = Path('/tmp/dsv41-extension-bank-20260919/wait_and_run.py')
shutil.copy2(waiter,root/'wait_and_run.py')
print(json.dumps({'source':proof['source_commit'],'incremental_bound':proof['static_incremental_bound_bytes'],
    'removable_dense_cache_bytes':proof['removable_dense_cache_bytes'],'scope':scope}))
