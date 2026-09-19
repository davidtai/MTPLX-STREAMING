"""Pin a native draft-state replay, adding only the causal lookup extension."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base=repo/'docs/deepseek-v41/receipts/decode-read-attribution-20260917/confidence-screen/screen.py'
prior=json.loads(gzip.decompress(base.with_name('screen.json.gz').read_bytes()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
# Audit native draft arithmetic before reusing its complete allocation proof.
path='mtplx/models/deepseek_v41_dspark.py'
old=subprocess.check_output(['git','show',prior['source_commit']+':'+path],cwd=repo,text=True)
new=(repo/path).read_text()
old_ast,new_ast=ast.parse(old),ast.parse(new)
for node in new_ast.body:
    if isinstance(node,ast.ClassDef) and node.name=='DSparkStageCache':
        node.body=[f for f in node.body if not (isinstance(f,ast.FunctionDef) and f.name=='detach_prefill_backings')]
assert ast.dump(old_ast,include_attributes=False)==ast.dump(new_ast,include_attributes=False)
assert sha(repo/'mtplx/models/deepseek_v41.py')==prior['runtime_source_sha256']['mtplx/models/deepseek_v41.py']
proof={'source_commit':head,'scope':'Draft-head-only native D5 plus causal two-token lookup, maximum target proposal M8. Saved target states; not target execution or TPS.',
       'static_incremental_bound_bytes':49*1024**3,
       'bound_components':{'active_bytes':41*1024**3,'cache_bytes':4*1024**3,'host_bytes':4*1024**3,
                           'lookup_host_bytes_inside_host_reserve':16*1024**2},
       'native_draft_ast_matches_prior_except_unused_detach_method':True,
       'prior_receipt_sha256':sha(base.with_name('screen.json.gz')),
       'lookup_config':{'minimum_context':2,'extra_tokens':2,'selection':'earliest_longest_context'},
       'model_path':'/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4',
       'strict_allocator':json.loads(Path('/tmp/dsv41-embedding-rows-20260918/installation.json').read_text())['strict_allocator']}
s=base.read_text()
s=s.replace("OUT=Path('/tmp/dsv41-confidence-20260917/screen.json')",f"OUT=Path({str(root/'head-screen.json')!r})")
start=s.index("installation=json.loads((ROOT/'installation.json').read_text())")
end=s.index('for key,value in reference[\'arm_env\'].items():',start)
s=s[:start]+f'''CURRENT_ROOT=Path({str(root)!r})
installation=json.loads((CURRENT_ROOT/'installation.json').read_text())
if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=installation['source_commit']:
    raise RuntimeError('source commit differs from pinned head screen')
for path,expected in installation['runtime_source_sha256'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('native source differs from head-screen audit')
''' + s[end:]
s=s.replace('>109500000000','>110000000000')
s=s.replace('import mlx.nn as nn','import mlx.nn as nn\nfrom library_identity import identify\nlibrary_identity=identify(installation["strict_allocator"])')
s=s.replace('from mtplx.models.deepseek_v41_dspark_decode import _effective_draft_len', '''from lookup import LookupExtension
prompt_payload=json.loads(Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json').read_text())
prompt=next(p['token_ids'] for p in prompt_payload['prompts'] if len(p['token_ids'])==16384)''')
s=s.replace("'scope': 'Native D5 confidence trimming on saved exact M6 target states. Futures score proposals but do not select widths. No target execution or throughput claim.'", "'scope': installation['scope'], 'library_identity':library_identity, 'lookup_configuration':installation['lookup_config']")
s=s.replace("for threshold in (None, 0.25, 0.5, 0.75, 0.9):", "for mode in ('native','hybrid_m8'):")
s=s.replace('    pos=0;main_h=initial_main;commit_lengths=[];width_counts={};rows=[]', '    pos=0;main_h=initial_main;commit_lengths=[];width_counts={};rows=[]\n    lookup=LookupExtension(prompt,minimum_context=2,extra_tokens=2)\n    history_count=0')
s=s.replace('        keep=_effective_draft_len(conf,5,threshold)', '''        native_proposed=[int(v) for v in proposed]
        if mode=='hybrid_m8':
            lookup.append_committed(ids[history_count:pos+1])
            history_count=pos+1
            proposed=lookup.extend(native_proposed)
        keep=len(proposed)''')
s=s.replace("'proposed_ids':[int(v) for v in proposed], 'teacher_matched_prefix':native_accepted,", "'proposed_ids':[int(v) for v in proposed], 'native_proposed_ids':native_proposed, 'teacher_matched_prefix':native_accepted,")
s=s.replace('    if threshold is None:', "    if mode=='native':")
s=s.replace("row={'threshold':threshold,", "row={'mode':mode,")
s=s.replace("('threshold','cycles','verify_rows','verify_width_counts','head_replay_wall_s','mlx_peak_bytes')", "('mode','cycles','verify_rows','verify_width_counts','head_replay_wall_s','mlx_peak_bytes')")
s=s.replace("'Native D5 control does not reproduce captured boundaries'", "'Native D5 control does not reproduce captured boundaries'")
s=s.replace("print('CONFIDENCE_SCREEN_ARM'", "print('HYBRID_HEAD_ARM'")
(root/'head_screen.py').write_text(s)
(root/'library_identity.py').write_text(Path('/tmp/dsv41-embedding-rows-20260918/library_identity.py').read_text())
screen=Path('/tmp/dsv41-embedding-rows-20260918/run_screen.py').read_text()
screen=screen.replace("files = [Path(installation['embedding']['path'])]", "files = reclaim.model_files(Path(installation['model_path']))")
screen=screen.replace("('probe.json', 'child.json', 'reclamation.json')", "('head-screen.json', 'child.json', 'reclamation.json')")
screen=screen.replace("str(ROOT / 'probe.py')", "str(ROOT / 'head_screen.py')")
screen=screen.replace('EMBEDDING_SOURCE_CACHE_RECLAIMED','HEAD_SOURCE_CACHE_RECLAIMED')
(root/'run_head.py').write_text(screen)
sources=['mtplx/models/deepseek_v41.py','mtplx/models/deepseek_v41_dspark.py',
    'mtplx/models/deepseek_v41_loader.py','mtplx/expert_runtime.py','mtplx/resident_io.py',
    'mtplx/deepseek_v41_memory_profile.py','scripts/deepseek_v41/reclaim_file_cache.py']
proof['runtime_source_sha256']={str(repo/p):sha(repo/p) for p in sources}
proof['helper_sha256']={p.name:sha(p) for p in root.glob('*.py')}
for p in root.glob('*.py'):ast.parse(p.read_text())
(root/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
command=shlex.join(['env','GPU_WINDOW_LOCK_TIMEOUT=120','GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000',
    f'GPU_WINDOW_CHILD_RSS_CAP_BYTES={49*1024**3}','GPU_WINDOW_MIN_AVAIL_GB=52','GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'GPU_WINDOW_CANDIDATE_MODEL_DIR='+proof['model_path'],'PYTHONHASHSEED=0','PYTHONUNBUFFERED=1',
    f'PYTHONPATH={repo}:{root}','scripts/deepseek_v41/gpu_window.sh',
    '/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(root/'run_head.py')])
(root/'command.sh').write_text(command+' > '+shlex.quote(str(root/'head.guard.log'))+' 2>&1\n')
print(json.dumps({'source':head,'incremental_bound_bytes':proof['static_incremental_bound_bytes'],
                  'native_draft_arithmetic_unchanged':True,'root':str(root)}))
