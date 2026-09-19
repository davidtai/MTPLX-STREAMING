"""Compose the causal D5-plus-lookup candidate with the current memory winner."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parent
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base=Path('/tmp/dsv41-embedding-rows-20260918/full-v2').resolve()
r=root/'full-v1';r.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
screen=json.loads((root/'head-screen.json').read_text())
assert screen['complete']
assert [(a['cycles'],a['verify_rows']) for a in screen['arms']]==[(206,1236),(198,1242)]
for sub in ('native','compat','packed'):
    d=json.loads((base/sub/'installation.json').read_text())
    for name,digest in d.get('runtime_source_sha256',{}).items():assert sha(repo/name)==digest,name
    for name,digest in d['helper_sha256'].items():assert sha(base/sub/name)==digest,name
    shutil.copytree(base/sub,r/sub,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
for p in r.rglob('*.py'):p.write_text(p.read_text().replace(str(base),str(r)))
for name in ('lookup.py','hybrid_install.py'):shutil.copyfile(root/name,r/'packed'/name)
p=r/'packed/packed_admission.py';s=p.read_text()
s=s.replace('    embedding_host = 32 * 1024**2', '    embedding_host = 32 * 1024**2\n    lookup_host = 16 * 1024**2')
s=s.replace("allocator_limit = original['allocator_limit_bytes'] - embedding_host", "allocator_limit = original['allocator_limit_bytes'] - embedding_host - lookup_host")
s=s.replace("original['host_reserve_bytes'] + embedding_host", "original['host_reserve_bytes'] + embedding_host + lookup_host")
s=s.replace("original['prefill_physical_bound_bytes'] + embedding_host", "original['prefill_physical_bound_bytes'] + embedding_host + lookup_host")
s=s.replace('        embedding_host_allowance_bytes=embedding_host,', '        embedding_host_allowance_bytes=embedding_host,\n        lookup_host_allowance_bytes=lookup_host,')
s=s.replace("bound_scope='Native input", "bound_scope='Adds16MiB fixed-workload lookup metadata within the retained native M8 tensor envelope. Native input")
p.write_text(s)
p=r/'packed/run_full.py';s=p.read_text()
old="with rt.admit_kv_tokens(len(kw['prompt_ids']) + int(kw['steps']) + int(kw['depth']) + 1):"
assert s.count(old)==1
s=s.replace(old,"with rt.admit_kv_tokens(len(kw['prompt_ids']) + int(kw['steps']) + 7 + 1):")
anchor='    original_dspark_generate = decode_module.dspark_generate'
assert s.count(anchor)==1;s=s.replace(anchor,'    hybrid_report = {}\n'+anchor)
old="    def dspark_with_boundary_observation(*a, **kw):\n        callback = kw['prefill_callback']"
new="""    def dspark_with_boundary_observation(*a, **kw):
        from hybrid_install import install as install_hybrid
        target = a[0] if a else kw['model']
        prompt = a[1] if len(a)>1 else kw['prompt_ids']
        if float(getattr(kw['sampler'],'temperature',0.0)) != 0.0:
            raise RuntimeError('hybrid benchmark requires greedy sampling')
        hybrid_report.update(install_hybrid(decode_module,target,prompt,
            requested_depth=kw.get('speculative_depth'),verify_chunks=kw.get('verify_chunks'),
            confidence_threshold=decode_module._confidence_threshold_from_env(kw.get('confidence_threshold'))))
        callback = kw['prefill_callback']"""
assert s.count(old)==1;s=s.replace(old,new)
old="        receipt['input_embedding_ownership'] = dict(embedding_report)"
assert s.count(old)==1;s=s.replace(old,old+"\n        receipt['hybrid_lookup'] = dict(hybrid_report)")
p.write_text(s)
for sub in ('native','compat','packed'):
    p=r/sub/'installation.json';d=json.loads(p.read_text());d['source_commit']=head
    d['scope']='Exact16K/1024 nativeKV16, native D5 head plus causal two-token lookup, target M6/M8. Existing packed arithmetic and exact input-row cache.16MiB added host reserve; target M8 native memory bound retained.'
    if sub=='packed':
        d['native_admission_sha256']=sha(r/'native/admission.py')
        for name in ('lookup.py','hybrid_install.py'):d['helper_sha256'][name]='pending'
        d['hybrid_lookup']={'head_screen_path':str(root/'head-screen.json'),'head_screen_sha256':sha(root/'head-screen.json'),
            'maximum_proposal_depth':7,'native_head_depth':5,'maximum_verify_rows':8,
            'host_allowance_bytes':16*1024**2,'target_arithmetic':'Unchanged native accept/commit and M<=8 packed target; only proposal selection differs.'}
    d['helper_sha256']={name:sha(r/sub/name) for name in d['helper_sha256']}
    p.write_text(json.dumps(d,indent=2)+'\n')
(r/'launch_full.py').write_text((base/'launch_full.py').read_text().replace(str(base),str(r)))
s=(base/'command.sh').read_text().replace(str(base),str(r))
s=s.replace('full-embedding-rows-20260918-v2','full-hybrid-lookup-20260918-v1')
s=s.replace('--host-overhead-gib 1.3087120056152344','--host-overhead-gib 1.3243370056152344')
(r/'command.sh').write_text(s)
s=(base/'preflight.py').read_text()
s=s.replace("assert a['embedding_host_allowance_bytes']==32*1024**2", "assert a['embedding_host_allowance_bytes']==32*1024**2 and a['lookup_host_allowance_bytes']==16*1024**2")
s=s.replace('embedding-admission.json','hybrid-admission.json')
s=s.replace("'embedding_host_allowance_bytes','embedding_post_prefill_credit_bytes'", "'lookup_host_allowance_bytes','embedding_post_prefill_credit_bytes'")
(r/'preflight.py').write_text(s)
for p in r.rglob('*.py'):ast.parse(p.read_text())
audit={'source_commit':head,'base':str(base),'extra_host_bytes':16*1024**2,
    'helper_sha256':{str(p.relative_to(r)):sha(p) for p in r.rglob('*')
        if p.is_file() and p.suffix in ('.py','.json') and 'artifact' not in p.parts}}
(r/'source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'root':str(r),'source':head,'extra_host_bytes':16*1024**2}))
