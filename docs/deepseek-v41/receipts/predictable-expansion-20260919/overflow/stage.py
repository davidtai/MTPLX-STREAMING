"""Stage one independent-row comparison after the full-size resize lost."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
proof=json.loads((BASE/'installation.json').read_text())
result=json.loads((BASE/'probe.json').read_text())
growth=json.loads((BASE/'growth/probe.json').read_text())
assert result['complete'] and growth['complete']
assert not (ROOT/'probe.json').exists()
for name,digest in proof['sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for p in BASE.iterdir():
    if p.is_file() and p.name in ('plane_lane.py','packed_storage.py','paired_kernels.py','kernels.py',
        'restore_bank.py','library_identity.py','routes.json','fused_transpose.py','attention_reader.py',
        'output_store.py','wait_and_run.py','run_screen.py'):
        shutil.copyfile(p,ROOT/p.name)
if not (ROOT/'artifact').exists():(ROOT/'artifact').symlink_to((BASE/'artifact').resolve(),target_is_directory=True)
code=(BASE/'probe.py').read_text()
def replace(old,new):
    global code
    assert code.count(old)==1,old
    code=code.replace(old,new)
replace('capacity = 110 if resident else 111','capacity = 110')
replace("result = {'mode':mode, 'sequence':sequence, 'capacity':capacity}",
        "result = {'mode':mode, 'sequence':sequence, 'capacity':110 if resident else 111}\n    allocation_ns = 0")
needle="        result['projection_storage_mode'] = 'cached-bf16' if resident else 'resident-packed-next-bf16'\n"
replace(needle,needle+"""        # Canonicalize the component's stripped-weight physical plan before
        # the phase extension; this also runs for the unchanged control.
        plan = dataclasses.replace(plan,persistent_cache_bytes=capacity*WEIGHT_BYTES,
            persistent_budget_bytes=capacity*WEIGHT_BYTES,expert_cache_limit_bytes=capacity*WEIGHT_BYTES,
            transient_bytes=48*WEIGHT_BYTES)
        runtime.plan = slots.plan = allocator.plan = plan
        slots.allocated_bytes = (capacity+48)*WEIGHT_BYTES
        if not resident:
            from overflow import append_rows
            result['overflow_installation'] = append_rows(runtime,mx=mx)
            allocation_ns = result['overflow_installation']['elapsed_ns']
""")
replace("expert_records=after_reads['records_read']-before_reads['records_read'],",
        "charged_warm_wall_ns=ended-warm_started+allocation_ns,\n                      charged_wall_ns=ended-started+allocation_ns,\n                      expert_records=after_reads['records_read']-before_reads['records_read'],")
replace("        digests = []", """        if not resident:
            physical = slots._persistent[LAYER,110]
            if physical.expert is None or physical.buffer.bank_index != 0:
                raise RuntimeError('overflow row was not used by the measured routes')
            result['overflow_final_expert'] = physical.expert
        digests = []""")
code=code.replace("a['warm_wall_ns'] for a in report['arms']","a['charged_warm_wall_ns'] for a in report['arms']")
(ROOT/'probe.py').write_text(code)
proof['scope'] += ' This variant appends an independent one-row bank after initial110-slot construction; existing row owners remain fixed. Reported comparison charges append allocation time.'
proof['bound_components']['overflow_native_allocation_peak_bytes']=18800640
proof['bound_components']['raw_expert_bank_max_bytes']=159*18800640
proof['predecessor']={'component_sha256':hashlib.sha256((BASE/'probe.json').read_bytes()).hexdigest(),
    'growth_sha256':hashlib.sha256((BASE/'growth/probe.json').read_bytes()).hexdigest(),
    'reason':'Native second resize projects1.798s; avoid copying the existing bank.'}
proof['sha256']={p:d for p,d in proof['sha256'].items() if not p.startswith(str(BASE))}
for p in ROOT.iterdir():
    if p.is_file() and p.name!='installation.json':
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        proof['sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
command=(BASE/'command.sh').read_text().replace(str(BASE),str(ROOT))
(ROOT/'command.sh').write_text(command)
print(json.dumps({'source_commit':proof['source_commit'],'bound_bytes':proof['static_incremental_bound_bytes'],'cpu_only':True}))
