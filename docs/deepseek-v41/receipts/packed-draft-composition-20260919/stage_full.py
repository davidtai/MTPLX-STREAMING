"""Compose the measured native packed draft with projection priming before growth."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

root=Path(__file__).resolve().parent
out=root/'full-v1'
base=Path('/tmp/dsv41-extension-bank-20260919/full-v1').resolve()
draft=Path('/tmp/dsv41-draft-packed-projection-20260919').resolve()
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert head=='b5f41c51d74a770069048dddd0a644f07886a999'
screen=json.loads((draft/'head-screen.json').read_text())
assert screen['complete'] and not screen['target_execution']
assert all(a['cycles']==198 and a['verify_rows']==1242 for a in screen['arms'])
assert screen['arms'][0]['commit_lengths']==screen['arms'][1]['commit_lengths']==screen['arms'][2]['commit_lengths']
retirement=screen['projection_installations'][1]
assert retirement['dense_cache_bytes_removed']==402653184
assert retirement['active_before_bytes']-retirement['active_after_bytes']==402653184
out.mkdir()
for sub in ('packed','native','compat'):
    p=json.loads((base/sub/'installation.json').read_text())
    for name,h in p['helper_sha256'].items():assert sha(base/sub/name)==h,(sub,name)
    shutil.copytree(base/sub,out/sub,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
for p in out.rglob('*.py'):p.write_text(p.read_text().replace(str(base),str(out)))
shutil.copy2(draft/'projection.py',out/'packed/draft_projection.py')

p=out/'packed/run_full.py';s=p.read_text()
def replace_once(old,new):
    global s
    assert s.count(old)==1,old
    s=s.replace(old,new)
replace_once('    projection_owner_report = {}','    projection_owner_report = {}\n    draft_projection_report = {}')
replace_once('        resident = original_load(*a, **kw)',
'''        resident = original_load(*a, **kw)
        from draft_projection import Installation
        installed_draft = Installation(resident.model)
        draft_projection_report.update(installed_draft.select(True))
        if draft_projection_report['dense_cache_bytes_removed'] != 0:
            raise RuntimeError('draft dense caches unexpectedly existed before first prefill')
        draft_projection_report.update(native_dense_cache_bytes_avoided=402653184,
            scope='Native93/58/32 experts; packed FP32 draft output only. No target arithmetic change.')''')
replace_once("        overflow_report = grow_rows(runtime, capacity=growth_admission['decode_slots_per_layer'], layout='extension', mx=mx)",
'''        # Cold projection compilation runs with only84 target rows present.
        # Keep this first67MiB output during extension; later decode still prices
        # the complete three-buffer replacement envelope.
        projection_owner_report.update(prime_model(target))
        seed_boundary_memory['after_early_prime'] = {
            'mlx_active_bytes': int(mx.get_active_memory()),
            'mlx_peak_bytes': int(mx.get_peak_memory()),
        }
        overflow_report = grow_rows(runtime, capacity=growth_admission['decode_slots_per_layer'], layout='extension', mx=mx)''')
needle="        projection_owner_report.update(prime_model(target))\n        seed_boundary_memory['after_overflow_and_prime']"
replace_once(needle,"        seed_boundary_memory['after_overflow_and_prime']")
replace_once("ALLOCATOR_LIMIT_BYTES = growth_admission['allocator_limit_bytes']",
"if growth_admission['decode_slots_per_layer'] < 111:\n    raise RuntimeError('packed draft comparison requires at least111 target rows within the live budget')\nALLOCATOR_LIMIT_BYTES = growth_admission['allocator_limit_bytes']")
replace_once("                result['projection_ownership'] = dict(projection_owner_report)",
'''                result['projection_ownership'] = dict(projection_owner_report)
                from draft_projection import PackedDraftOutput
                if any(getattr(layer.attn, '_wo_a_dense_cache', None) is not None
                       or not isinstance(layer.attn._draft_packed_out, PackedDraftOutput)
                       for layer in kw['model'].mtp.layers):
                    raise RuntimeError('packed draft ownership changed during the request')
                draft_projection_report['post_request_ownership_verified'] = True
                result['draft_projection'] = dict(draft_projection_report)''')
replace_once("        receipt['projection_ownership'] = dict(projection_owner_report)",
"        receipt['projection_ownership'] = dict(projection_owner_report)\n        receipt['draft_projection'] = dict(draft_projection_report)")
s=s.replace('separate extension allocation, projection installation and priming are inside decode wall time',
            'early projection priming and separate extension allocation are inside decode wall time')
p.write_text(s)

p=out/'packed/projection_install.py';s=p.read_text()
assert s.count("'primed_after_native_seed_and_overflow': True")==1
p.write_text(s.replace("'primed_after_native_seed_and_overflow': True", "'primed_after_native_seed_before_extension': True"))

p=out/'packed/packed_admission.py';s=p.read_text()
s=s.replace('    expansion_host = 16 * 1024**2',
'''    expansion_host = 16 * 1024**2
    draft_host = 16 * 1024**2
    background_growth = 256 * 1024**2
    draft_credit = 3 * 134217728
    draft_workspace = 128 * 1024**2
    # Three packed weight/scale copies (if needed), inverse-RoPE/input/layout
    # copies and gather outputs fit below128MiB at nativeT6. GatherQMM's packed
    # Metal path allocates output/contiguity copies, not dense weight expansions.
    draft_proof = installation['draft_projection']
    blob = Path(draft_proof['component_receipt']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != draft_proof['component_sha256']:
        raise RuntimeError('packed draft component identity changed')
    component = json.loads(blob)
    retired = component['projection_installations'][1]
    if (not component['complete'] or component['target_execution']
        or retired['active_before_bytes'] - retired['active_after_bytes'] != draft_credit):
        raise RuntimeError('packed draft physical retirement proof differs')''')
s=s.replace(' - embedding_host - lookup_host - expansion_host',
            ' - embedding_host - lookup_host - expansion_host - draft_host - background_growth')
s=s.replace('embedding_host + lookup_host + expansion_host',
            'embedding_host + lookup_host + expansion_host + draft_host + background_growth')
s=s.replace('        steady -= expansion_credit',
'''        steady -= expansion_credit
        steady += draft_workspace - draft_credit
        early_prime_peak = seed + 3*67108864''')
old='append_peak = seed + overflow_payload + (capacity-old_capacity)*(RAW-WEIGHTS) + 3*67108864 + original[\'page_padding_allowance_bytes\']'
new='append_peak = seed + overflow_payload + (capacity-old_capacity)*(RAW-WEIGHTS) + 67108864 + original[\'page_padding_allowance_bytes\']'
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('active = max(steady, resize, seed, append_peak)', 'active = max(steady, resize, seed, early_prime_peak, append_peak)')
s=s.replace('        embedding_host_allowance_bytes=embedding_host,',
'''        draft_dense_cache_steady_credit_bytes=draft_credit,
        draft_projection_workspace_allowance_bytes=draft_workspace,
        draft_projection_host_allowance_bytes=draft_host,
        draft_prefill_seed_and_extension_credit_bytes=0,
        background_variation_allowance_bytes=background_growth,
        process_host_reserve_bytes=original['host_reserve_bytes'] + embedding_host + lookup_host + expansion_host + draft_host,
        early_prime_active_bound_bytes=early_prime_peak,
        retained_projection_during_extension_bytes=67108864,
        extension_cold_overlap_removed_bytes=2*67108864,
        embedding_host_allowance_bytes=embedding_host,''')
s=s.replace('# Price all new packed rows, one temporary native scale owner, three\n        # BF16 expansions including cold compilation, and extra page padding.',
'''# Prime at84 rows before extension. Cold compilation still prices three
        # BF16 arrays in that earlier phase. Extension retains one initialized
        # array and one temporary native scale owner. Decode keeps all three.''')
s=s.replace('Extension peak includes all final added packed rows, one full extension-bank raw-scale temporary, three BF16 projection arrays and page padding.',
            'Cold projection prime occurs at84 rows and prices three BF16 arrays. Extension owns all final added rows, one raw-scale temporary, one already initialized BF16 projection and page padding. Draft-only403MB credit applies to steady state;128MiB new GPU workspace,16MiB host metadata and256MiB background allowance are explicit. Native93/58/32 draft experts remain.')
p.write_text(s)

scope='Native93/58/32 Q4 draft with measured packed FP32 output projection; original target arithmetic/KV16/D5+lookup/M8. Prime first target projection at84 rows before independent bank extension. Phase-specific128MiB draft workspace,16MiB host and256MiB background allowances;110GB unchanged.'
backend=Path('/tmp/dsv41-strict-cache-20260918/mlx-0.32.2/mlx/backend/metal/quantized.cpp').resolve()
for sub in ('packed','native','compat'):
    p=out/sub/'installation.json';d=json.loads(p.read_text());d.update(source_commit=head,scope=scope)
    if sub=='packed':
        d['native_admission_sha256']=sha(out/'native/admission.py')
        d['helper_sha256']['draft_projection.py']='pending'
        d['draft_projection']={'component_receipt':str(draft/'head-screen.json'),
            'component_sha256':sha(draft/'head-screen.json'),'source_sha256':sha(out/'packed/draft_projection.py'),
            'steady_credit_bytes':402653184,'additional_gpu_workspace_bytes':128*1024**2,
            'additional_host_bytes':16*1024**2,'minimum_decode_rows':111,
            'backend_source':str(backend),'backend_source_sha256':sha(backend),
            'workspace_inventory_bytes':3*(34603008 + 4*6*32768*4 + 4*6*8192*4 + 4096),
            'scope':'Retirement credit only against native steady dense caches; none for prefill/seed/extension. Three complete packed-copy allowances plus FP32 temporaries fit128MiB; target operators unchanged.'}
        assert d['draft_projection']['workspace_inventory_bytes']<128*1024**2
        d['projection_ownership']['prime_phase']='after native seed at84 rows, before extension'
        d['background_variation']={'allowance_bytes':256*1024**2,'physical_ceiling_bytes':110000000000}
    d['helper_sha256']={name:sha(out/sub/name) for name in d['helper_sha256']}
    p.write_text(json.dumps(d,indent=2)+'\n')
for name in ('launch_full.py','wait_and_run.py'):
    (out/name).write_text((base/name).read_text().replace(str(base),str(out)))
command=(base/'command.sh').read_text().replace(str(base),str(out)).replace('full-extension-bank-20260919-v1','full-packed-draft-20260919-v1')
parts=shlex.split(command);old_host=parts[parts.index('--host-overhead-gib')+1]
command=command.replace('--host-overhead-gib '+old_host,'--host-overhead-gib '+format((1438773248+16*1024**2+256*1024**2)/1024**3,'.17g'))
(out/'command.sh').write_text(command)
for p in out.rglob('*.py'):compile(p.read_text(),str(p),'exec')
audit={'cpu_only':True,'source_commit':head,'scope':scope,
       'target_kernels_unchanged':all(sha(out/'packed'/n)==sha(base/'packed'/n) for n in ('plane_lane.py','paired_kernels.py','kernels.py','fused_transpose.py','hybrid_install.py','extension.py')),
       'native_draft_experts':[93,58,32],'stage_sha256':sha(Path(__file__)),
       'full_result':'unmeasured'}
(out/'source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
