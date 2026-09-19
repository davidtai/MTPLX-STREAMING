"""Compose the successful bounded input-row ownership with the exact full lane."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base = Path('/tmp/dsv41-strict-cache-20260918/full-v1')
r = root / 'full-v2'
r.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
probe = json.loads((root / 'probe.json').read_text())
assert probe['complete'] and probe['all_outputs_exact'] and probe['eviction_ownership_exact']
assert probe['released_native_bytes'] == 1323827200 and probe['active_after_close_bytes'] == 0
for sub in ('native','compat','packed'):
    old = json.loads((base / sub / 'installation.json').read_text())
    for name, digest in old.get('runtime_source_sha256', {}).items():
        assert sha(repo / name) == digest, name
    for name, digest in old['helper_sha256'].items():
        assert sha(base / sub / name) == digest, name
    shutil.copytree(base / sub, r / sub, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
for p in r.rglob('*.py'):
    p.write_text(p.read_text().replace(str(base), str(r)))
for name in ('file_embedding.py','embedding_install.py'):
    shutil.copyfile(root / name, r / 'packed' / name)
shutil.copyfile(root / 'probe.json', r / 'packed/embedding-probe.json')

p = r / 'packed/packed_admission.py'
s = p.read_text()
anchor = '    projection_credit = projection_source_bytes - projection_cold_overlap\n'
insertion = '''    embedding = installation['embedding_optimization']
    blob = (ROOT / 'embedding-probe.json').read_bytes()
    operator = json.loads(blob)
    if (hashlib.sha256(blob).hexdigest() != embedding['probe_sha256']
            or not operator['complete'] or not operator['all_outputs_exact']
            or not operator['eviction_ownership_exact']
            or operator['released_native_bytes'] != 1323827200
            or operator['active_after_close_bytes'] != 0):
        raise RuntimeError('bounded input-row ownership proof changed')
    embedding_credit = 1323827200
    embedding_host = 32 * 1024**2
    allocator_limit = original['allocator_limit_bytes'] - embedding_host
    if (original['prefill_active_bound_bytes'] + original['prefill_cache_allowance_bytes'] > allocator_limit
            or original['prefill_physical_bound_bytes'] + embedding_host > DEFAULT_BOX_BUDGET_BYTES):
        raise RuntimeError('input-row host allowance does not fit unchanged prefill')
'''
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + insertion)
anchor = '        active = max(steady, resize, seed)\n'
s = s.replace(anchor, '''        steady -= embedding_credit
        resize -= embedding_credit
        seed -= embedding_credit
''' + anchor)
s = s.replace("physical = base + original['host_reserve_bytes'] + active", "physical = base + original['host_reserve_bytes'] + embedding_host + active")
s = s.replace("<= original['allocator_limit_bytes']", '<= allocator_limit')
s = s.replace("        physical_ceiling_bytes=DEFAULT_BOX_BUDGET_BYTES,", """        physical_ceiling_bytes=DEFAULT_BOX_BUDGET_BYTES,
        allocator_limit_bytes=allocator_limit,
        native_host_reserve_bytes=original['host_reserve_bytes'],
        host_reserve_bytes=original['host_reserve_bytes'] + embedding_host,
        prefill_physical_bound_bytes=original['prefill_physical_bound_bytes'] + embedding_host,
        embedding_host_allowance_bytes=embedding_host,
        embedding_post_prefill_credit_bytes=embedding_credit,
        embedding_prefill_credit_bytes=0,
        transition_start_active_bound_bytes=original['transition_start_active_bound_bytes'] - embedding_credit,""")
s = s.replace("physical_bound_bytes=max(original['prefill_physical_bound_bytes'], physical)",
              "physical_bound_bytes=max(original['prefill_physical_bound_bytes'] + embedding_host, physical)")
s = s.replace("bound_scope='Original cap84", "bound_scope='Native input table is retired and measured before growth; exact1323827200B credit applies to resize/seed/steady only;32MiB host reserve covers the fixed16MiB row arena plus metadata/copies. Original cap84")
p.write_text(s)

p = r / 'packed/run_full.py'
s = p.read_text()
s = s.replace('    bounded_engram_report = {}\n', '    bounded_engram_report = {}\n    embedding_report = {}\n    embedding_completed = None\n', 1)
old = 'global growth_transition, growth_report, prefill_resolved_plan, projection_owner_report, tail_prefill_report, bounded_engram_report'
s = s.replace(old, old + ', embedding_report, embedding_completed')
anchor = '        native_growth = growth_transition\n'
s = s.replace(anchor, '''        from embedding_install import prepare as prepare_embedding
        embedding_retire, embedding_completed, embedding_report = prepare_embedding(
            resident.model, PACKED_INSTALLATION['embedding_optimization']['source'], growth_admission)
''' + anchor)
s = s.replace('        def grow_and_install_projection_owners():\n            native_growth()',
              '        def grow_and_install_projection_owners():\n            embedding_retire()\n            native_growth()')
s = s.replace("                result['projection_ownership'] = dict(projection_owner_report)",
              "                result['projection_ownership'] = dict(projection_owner_report)\n                result['input_embedding_ownership'] = embedding_completed()")
s = s.replace("        receipt['projection_ownership'] = dict(projection_owner_report)",
              "        receipt['projection_ownership'] = dict(projection_owner_report)\n        receipt['input_embedding_ownership'] = dict(embedding_report)")
s = s.replace("    _cleanup = [_cleanup_module.reclaim_file(_source_path)]",
              "    _cleanup = [_cleanup_module.reclaim_file(_source_path),\n                _cleanup_module.reclaim_file(Path(PACKED_INSTALLATION['embedding_optimization']['source']['path']))]")
p.write_text(s)

for sub in ('native','compat','packed'):
    p = r / sub / 'installation.json'
    d = json.loads(p.read_text())
    d['source_commit'] = head
    d['scope'] = 'Exact16K/1024 nativeKV16 D5/M6. Fixed16MiB exact input-row arena after prefill,32MiB host allowance; retire native1.324GB table before growth. Retained native packed math and cache policy.'
    if sub == 'packed':
        d['native_admission_sha256'] = sha(r / 'native/admission.py')
        d['embedding_optimization'] = {'source':probe['construction']['embedding'],
            'probe_sha256':sha(root / 'probe.json'), 'arena_bytes':16*1024**2,
            'host_allowance_bytes':32*1024**2, 'post_prefill_credit_bytes':1323827200,
            'prefill_credit_bytes':0, 'source_table_sha256':probe['table_sha256']}
        for name in ('file_embedding.py','embedding_install.py','embedding-probe.json'):
            d['helper_sha256'][name] = 'pending'
    d['helper_sha256'] = {name:sha(r / sub / name) for name in d['helper_sha256']}
    p.write_text(json.dumps(d, indent=2) + '\n')
(r / 'launch_full.py').write_text((base / 'launch_full.py').read_text().replace(str(base),str(r)))
s = (base / 'command.sh').read_text().replace(str(base),str(r))
s = s.replace('full-strict-cache-20260918-v1','full-embedding-rows-20260918-v2')
s = s.replace('--host-overhead-gib 1.2774620056152344','--host-overhead-gib 1.3087120056152344')
(r / 'command.sh').write_text(s)

pre = Path('/tmp/dsv41-cpu-attribution-20260918/full-v1/preflight.py').read_text()
pre = pre.replace("a['decode_slots_per_layer']==109", "a['decode_slots_per_layer']==110")
pre = pre.replace("assert a['diagnostic_host_allowance_bytes']==16*1024**2", "assert a['embedding_host_allowance_bytes']==32*1024**2")
pre = pre.replace("assert a['diagnostic_active_allowance_bytes']==0", "assert a['embedding_post_prefill_credit_bytes']==1323827200 and a['embedding_prefill_credit_bytes']==0")
pre = pre.replace('diagnostic-admission.json','embedding-admission.json')
pre = pre.replace("'diagnostic_host_allowance_bytes','diagnostic_active_allowance_bytes'", "'embedding_host_allowance_bytes','embedding_post_prefill_credit_bytes'")
(r / 'preflight.py').write_text(pre)
for p in r.rglob('*.py'):
    ast.parse(p.read_text())
audit = {'source_commit':head, 'base':str(base), 'embedding_probe_sha256':sha(root / 'probe.json'),
    'helper_sha256':{str(p.relative_to(r)):sha(p) for p in r.rglob('*')
                    if p.is_file() and p.suffix in ('.py','.json') and 'artifact' not in p.parts}}
(r / 'source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'root':str(r),'source':head,'helpers':len(audit['helper_sha256'])}))
