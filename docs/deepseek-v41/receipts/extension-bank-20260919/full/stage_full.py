"""Compose the measured separate extension with the exact full Q4 runner."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parent
base=Path('/tmp/dsv41-predictable-expansion-20260919/full-v1').resolve()
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
r=root/'full-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert head=='d5f15e7a02d225115debf6ce317e26dc4a935b60'
probe=json.loads((root/'probe.json').read_text())
assert probe['complete'] and all(a['all_outputs_exact'] for a in probe['arms'])
assert probe['candidate_over_control']<1-probe['control_spread_fraction']
assert len({a['expert_records'] for a in probe['arms']})==1
r.mkdir()
for sub in ('native','compat','packed'):
    d=json.loads((base/sub/'installation.json').read_text())
    for name,h in d.get('runtime_source_sha256',{}).items():assert sha(repo/name)==h,name
    for name,h in d['helper_sha256'].items():assert sha(base/sub/name)==h,name
    shutil.copytree(base/sub,r/sub,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
for p in r.rglob('*.py'):p.write_text(p.read_text().replace(str(base),str(r)))
shutil.copyfile(root/'extension.py',r/'packed/extension.py')

def edit(path,edits):
    source=path.read_text()
    for old,new in edits:
        assert source.count(old)==1,(str(path),old)
        source=source.replace(old,new)
    path.write_text(source)

edit(r/'packed/packed_phase.py',[
 ('        or not old_capacity < capacity <= admission[\'capacity_search_ceiling\']',
  '        or capacity != old_capacity'),
 ('                    grow_bank(bank, capacity, mx=mx)',
  '                    # Existing84-row backing stays in place; extension follows native seed.'),
 ("timing_scope='scale installation, raw backing release and one weight resize are inside decode wall time'",
  "timing_scope='Packed scale installation and raw-scale retirement at unchanged84 rows; no expert-weight copy; inside decode wall time'"),
])
edit(r/'packed/run_full.py',[
 ('        from overflow import append_rows','        from extension import grow_rows'),
 ('        overflow_report = append_rows(runtime, mx=mx)',
  "        overflow_report = grow_rows(runtime, capacity=growth_admission['decode_slots_per_layer'], layout='extension', mx=mx)"),
 ("timing_scope='Original first growth plus independent overflow allocation, projection installation and priming are all inside decode wall time; growth_seconds sums the two bank phases only.'",
  "timing_scope='Scale installation at84 rows, native MTP seed, separate extension allocation, projection installation and priming are inside decode wall time; growth_seconds sums scale and extension phases only.'"),
])
admission=r/'packed/packed_admission.py'
edit(admission,[
 ('    overflow_payload = 40 * WEIGHTS',
  '    initial_net_growth = (40*old_capacity+48)*WEIGHTS + PACKED - raw_before'),
 ("    for capacity in range(strict['capacity_search_ceiling'], old_capacity, -1):",
  '    for capacity in range(112, old_capacity, -1):'),
 ('        packed_copy = (2 * capacity - old_capacity) * 5898240',
  '        overflow_payload = (capacity-old_capacity)*40*WEIGHTS'),
 ("        resize = (original['transition_start_active_bound_bytes'] + net_growth + packed_peak\n                  + packed_copy + original['page_padding_allowance_bytes'] + extra - tail_credit)",
  "        # Negative final scale delta cannot be credited at the start. Price\n"
  "        # the entire packed-scale inventory plus a layer staging allowance\n"
  "        # on top of the unchanged84-row transition start. No weight copies.\n"
  "        resize = (original['transition_start_active_bound_bytes'] + PACKED + packed_peak\n"
  "                  + original['page_padding_allowance_bytes'] + extra - tail_credit)"),
 ('        seed = seed_peak + (capacity - 101) * 40 * WEIGHTS',
  '        seed = seed_peak + (old_capacity - 101) * 40 * WEIGHTS'),
 ('        steady += overflow_payload - expansion_credit','        steady -= expansion_credit'),
 ("        append_peak = seed + overflow_payload + (RAW-WEIGHTS) + 3*67108864 + original['page_padding_allowance_bytes']",
  "        append_peak = seed + overflow_payload + (capacity-old_capacity)*(RAW-WEIGHTS) + 3*67108864 + original['page_padding_allowance_bytes']"),
 ("        capacity_search_ceiling=strict['capacity_search_ceiling'],",
  "        capacity_search_ceiling=112,\n        existing_bank_rows=84, maximum_extension_bank_rows=28,"),
 ('        initial_decode_slots_per_layer=capacity, initial_growth_payload_bytes=net_growth,',
  '        initial_decode_slots_per_layer=old_capacity, initial_growth_payload_bytes=initial_net_growth,'),
 ('        decode_slots_per_layer=capacity+1, growth_payload_bytes=net_growth+overflow_payload,',
  '        decode_slots_per_layer=capacity, growth_payload_bytes=net_growth,'),
 ('        packed_transition_payload_and_copy_bound_bytes=net_growth + packed_peak + packed_copy + extra,',
  '        packed_transition_payload_and_copy_bound_bytes=PACKED + packed_peak + extra,'),
 ("        capacity_selection='largest admitted initial packed bank through110, then one independent row per layer after seed; no second resize',",
  "        capacity_selection='largest admitted85..112 final slots;84 original rows plus1..28 extension rows after native seed, no expert-bank copies',"),
])
s=admission.read_text()
line=next(line for line in s.splitlines() if line.startswith('        bound_scope='))
scope=('Unchanged84-row prefill. First transition retains84 expert rows, retires raw scales and installs packed scales; '
       'price the entire packed inventory plus one packed layer above transition-start instead of crediting its negative final delta. '
       'Native MTP seed completes at84 rows. Extension peak includes all final added packed rows, one full extension-bank raw-scale temporary, '
       'three BF16 projection arrays and page padding. Existing256MiB allocation margin, nativeM8/KV envelope, '
       '256MiB strict allocator cache,100GiB wired ceiling and110GB whole-machine ceiling retained. '
       'Packed projection credit only in steady state. Extra16MiB schedule/bank metadata reserve remains in every phase. '
       'No existing expert backing array is resized or copied.')
s=s.replace(line,'        bound_scope='+repr(scope)+',')
admission.write_text(s)

scope='Exact16K/1024 Q4 nativeKV16, D5 plus original lookup and packed projection scheduling. Keep84 original expert rows, install packed scales without resize, seed native MTP, append1..28 independent rows per layer. No target arithmetic change. All phases timed and separately admitted.'
for sub in ('native','compat','packed'):
    p=r/sub/'installation.json';d=json.loads(p.read_text());d['source_commit']=head;d['scope']=scope
    if sub=='packed':
        d['native_admission_sha256']=sha(r/'native/admission.py')
        d['helper_sha256']['extension.py']='pending'
        d['projection_ownership'].update(initial_capacity_ceiling=84,final_capacity_ceiling=112,
            scope='Projection arithmetic and bounded owners unchanged; first scale conversion at84 rows, extension to at most112 rows after native seed.')
        d['bank_extension']={'component_receipt':str(root/'probe.json'),
            'component_receipt_sha256':sha(root/'probe.json'),'initial_rows':84,'maximum_final_rows':112,
            'measured_extension_rows':26,'maximum_extension_rows':28,
            'geometry':'Existing packed kernels group by bank; original bank index stays0..83, extension0..27, both below already proved native bank/index envelope.',
            'source_helper_sha256':sha(root/'extension.py')}
    d['helper_sha256']={name:sha(r/sub/name) for name in d['helper_sha256']}
    p.write_text(json.dumps(d,indent=2)+'\n')
for name in ('launch_full.py','wait_and_run.py'):
    (r/name).write_text((base/name).read_text().replace(str(base),str(r)))
command=(base/'command.sh').read_text().replace(str(base),str(r))
command=command.replace('full-predictable-expansion-20260919-v1','full-extension-bank-20260919-v1')
(r/'command.sh').write_text(command)
preflight=(base/'preflight.py').read_text()
preflight=preflight.replace("a['decode_slots_per_layer']==111 and a['initial_decode_slots_per_layer']==110", "a['decode_slots_per_layer']==112 and a['initial_decode_slots_per_layer']==84")
preflight=preflight.replace('expansion-admission.json','extension-admission.json')
(r/'preflight.py').write_text(preflight)
audit=(base/'composition_audit.py').read_text().replace(str(base),str(r))
audit=audit.replace("assert a['decode_slots_per_layer']==a['initial_decode_slots_per_layer']+1", "assert a['initial_decode_slots_per_layer']==84 and 85<=a['decode_slots_per_layer']<=112")
audit=audit.replace("r.parent/'overflow/fused_transpose.py'", "r.parent/'fused_transpose.py'")
(r/'composition_audit.py').write_text(audit)
for p in r.rglob('*.py'):compile(p.read_text(),str(p),'exec')
record={'source_commit':head,'base':str(base),'scope':scope,
    'helper_sha256':{str(p.relative_to(r)):sha(p) for p in r.rglob('*') if p.is_file()
        and p.suffix in ('.py','.json') and 'artifact' not in p.parts}}
(r/'source-audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'source_commit':head,'root':str(r),'cpu_only':True}))
