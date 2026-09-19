"""Stage an equal-capacity, allocation-charged first-bank-copy comparison."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
BASE=Path('/tmp/dsv41-predictable-expansion-20260919/overflow').resolve()
FULL=BASE.parent/'full-v1'
REPO=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'probe.json').exists()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
assert head=='d5f15e7a02d225115debf6ce317e26dc4a935b60'
proof=json.loads((BASE/'installation.json').read_text())
result=json.loads((BASE/'probe.json').read_text())
assert result['complete'] and all(a['all_outputs_exact'] for a in result['arms'])
for path,digest in proof['sha256'].items():assert sha(Path(path))==digest,path
full=json.loads((FULL/'packed/installation.json').read_text())
for name,digest in full['helper_sha256'].items():assert sha(FULL/'packed'/name)==digest,name
for name in ('plane_lane.py','packed_storage.py','paired_kernels.py','kernels.py',
             'restore_bank.py','library_identity.py','routes.json','fused_transpose.py',
             'attention_reader.py','output_store.py','wait_and_run.py','run_screen.py'):
    shutil.copyfile(BASE/name,ROOT/name)
shutil.copyfile(FULL/'packed/overflow.py',ROOT/'overflow.py')
shutil.copyfile(FULL/'compat/bank_growth_final.py',ROOT/'bank_growth_final.py')
if not (ROOT/'artifact').exists():
    (ROOT/'artifact').symlink_to((BASE/'artifact').resolve(),target_is_directory=True)

def rewrite(source,edits):
    for old,new in edits:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source

# All new branching is at a quiescent installation boundary. The native
# steady packed expert runner and its bank grouping remain byte-identical.
extension=rewrite((ROOT/'overflow.py').read_text(),[
 ('Append one independent packed bank row per layer at a quiescent boundary.',
  'Grow the original bank or add a separate extension at a quiescent boundary.'),
 ('def append_rows(runtime, *, mx):','def grow_rows(runtime, *, capacity, layout, mx):'),
 ('    old=old_plan.slots_per_layer',
  "    old=old_plan.slots_per_layer\n    if old!=84 or not old<capacity<=112 or layout not in ('resize','extension'):\n        raise RuntimeError('exact first-growth geometry required')\n    added=capacity-old"),
 ('    delta=len(layers)*WEIGHT_BYTES','    delta=len(layers)*added*WEIGHT_BYTES'),
 ('            or pool.allocated_bytes!=expected_bytes):',
  '            or pool.allocated_bytes!=expected_bytes\n            or old_plan.allocated_bytes+old_plan.unallocated_bytes!=old_plan.total_limit_bytes):'),
 ('persistent_slots=old_plan.persistent_slots+len(layers)',
  'persistent_slots=old_plan.persistent_slots+len(layers)*added'),
 ('slots_per_layer=old+1,','slots_per_layer=capacity,'),
 ("        for layer in layers:\n            label=f'layer-{layer}-overflow-{old}'\n            bank=MlxComponentBank(capacity=1,record=pool._record_map[layer,0],label=label)\n            # Register ownership before any subsequent operation can fail.\n            allocator.banks['overflow',layer]=bank\n            remove_raw_scales(bank,mx=mx)\n            buffer=MlxComponentSlot(bank,0,label=label)\n            allocator.slots[label]=buffer\n            pool._persistent[layer,old]=_PhysicalSlot(label,buffer)",
  """        for layer in layers:
            if layout=='resize':
                from bank_growth_final import grow_bank
                bank=allocator.banks['persistent',layer]
                grow_bank(bank,capacity,mx=mx)
                first_index=old
            else:
                bank=MlxComponentBank(capacity=added,record=pool._record_map[layer,0],
                                      label=f'layer-{layer}-extension')
                allocator.banks['overflow',layer]=bank
                remove_raw_scales(bank,mx=mx)
                first_index=0
            for offset in range(added):
                label=f'layer-{layer}-added-{old+offset}'
                buffer=MlxComponentSlot(bank,first_index+offset,label=label)
                allocator.slots[label]=buffer
                pool._persistent[layer,old+offset]=_PhysicalSlot(label,buffer)"""),
 ('            policy._slot_to_expert.append(None)',
  '            policy._slot_to_expert.extend([None]*added)'),
 ('policy.persistent_slots=policy._persistent_capacity=old+1',
  'policy.persistent_slots=policy._persistent_capacity=capacity'),
 ('policy.slot_count=old+1+48','policy.slot_count=capacity+48'),
 ("policy._protected_cap=max(1,int((old+1)*.8))", "policy._protected_cap=max(1,int(capacity*.8))"),
 ('pool._persistent_route_capacity=old+1','pool._persistent_route_capacity=capacity'),
 ('pool._persistent_route_capacities={layer:old+1 for layer in layers}',
  'pool._persistent_route_capacities={layer:capacity for layer in layers}'),
 ("return {'initial_capacity':old,'capacity':old+1,'layers':len(layers),",
  "return {'initial_capacity':old,'capacity':capacity,'layout':layout,'layers':len(layers),"),
 ("'scope':'One fixed added row; no existing bank resize or copy.'",
  "'scope':'Construction-only allocation; original row objects/indices preserved. Extension leaves all original bank backing arrays in place.'"),
])
(ROOT/'extension.py').write_text(extension)

probe=rewrite((BASE/'probe.py').read_text(),[
 ("    resident = mode == 'resident-110'\n    capacity = 110",
  "    resident = False\n    capacity = 84"),
 ("result = {'mode':mode, 'sequence':sequence, 'capacity':110 if resident else 111}",
  "result = {'mode':mode, 'sequence':sequence, 'capacity':110, 'initial_capacity':84}"),
 ('        runtime.plan = slots.plan = allocator.plan = plan',
  '        plan = dataclasses.replace(plan,unallocated_bytes=plan.total_limit_bytes-plan.allocated_bytes)\n        runtime.plan = slots.plan = allocator.plan = plan'),
 ("        if not resident:\n            from overflow import append_rows\n            result['overflow_installation'] = append_rows(runtime,mx=mx)\n            allocation_ns = result['overflow_installation']['elapsed_ns']",
  """        from extension import grow_rows
        from overflow import append_rows
        initial_arrays={name:id(value) for name,value in allocator.banks['persistent',LAYER].arrays.items()}
        if mode=='resize109-plus1':
            first=grow_rows(runtime,capacity=109,layout='resize',mx=mx)
            second=append_rows(runtime,mx=mx)
            phases=[first,second]
        else:
            phases=[grow_rows(runtime,capacity=110,layout='extension',mx=mx)]
        allocation_ns=sum(p['elapsed_ns'] for p in phases)
        result['allocation_phases']=phases
        result['allocation_ns']=allocation_ns
        result['existing_backings_unchanged']=all(id(allocator.banks['persistent',LAYER].arrays[name])==value
                                                for name,value in initial_arrays.items())
        if result['existing_backings_unchanged'] != (mode=='extension84-plus26'):
            raise RuntimeError('bank backing replacement does not match selected layout')
        result['layout']={str(key):bank.capacity for key,bank in allocator.banks.items()}
        if runtime.plan.slots_per_layer!=110 or runtime.slots.allocated_bytes!=(110+48)*WEIGHT_BYTES:
            raise RuntimeError('final physical capacity differs')"""),
 ("        if not resident:\n            physical = slots._persistent[LAYER,110]\n            if physical.expert is None or physical.buffer.bank_index != 0:\n                raise RuntimeError('overflow row was not used by the measured routes')\n            result['overflow_final_expert'] = physical.expert",
  """        extra_start=109 if mode=='resize109-plus1' else 84
        extra=[slots._persistent[LAYER,i] for i in range(extra_start,110)]
        if not any(p.expert is not None for p in extra):
            raise RuntimeError('extension bank was not used by the measured routes')
        result['extension_final_experts']=[p.expert for p in extra]"""),
 ("if a['mode']=='resident-110'", "if a['mode']=='resize109-plus1'"),
 ("if a['mode']=='expand-111'", "if a['mode']=='extension84-plus26'"),
])
(ROOT/'probe.py').write_text(probe)
proof['source_commit']=head
proof['arms']=['resize109-plus1','extension84-plus26','resize109-plus1']
proof['scope']='Equal final110-slot comparison. Both retain packed projections and issue the next exact BF16 transpose. Control reproduces first84->109 resize plus one overflow row; candidate adds26 independent rows to the unchanged84-row bank. All206 expert/projection outputs compared; warmed replay plus allocation is the primary metric, with full replay also retained. Synthetic inputs and saved layer34 routes, no attention.'
proof['predecessor']={'full_result_path':'/tmp/dsv41-110-stage/full-predictable-expansion-20260919-v1.jsonl',
    'full_result_sha256':sha(Path('/tmp/dsv41-110-stage/full-predictable-expansion-20260919-v1.jsonl')),
    'full_growth_seconds':3.6737118340040418,'reason':'Remove first bank copies while pricing extra steady bank grouping.'}
proof['bound_components']['control_bf16_wo_a_bytes']=0
proof['bound_components']['native_control_component_copy_allowance_bytes']=(2*109-84)*5898240
proof['bound_components']['raw_extension_native_scale_peak_bytes']=28*(18800640-17694720)
proof['bound_components']['raw_expert_bank_max_bytes']=160*18800640
proof['bound_components']['packed_expert_bank_max_bytes']=160*17694720
proof['bound_components']['overflow_native_allocation_peak_bytes']=28*18800640
proof['plan'].update(persistent_slots=84,slots_per_layer=84,persistent_cache_bytes=84*18800640,
    persistent_budget_bytes=84*18800640,expert_cache_limit_bytes=84*18800640,
    total_limit_bytes=10*1024**3,unallocated_bytes=10*1024**3-4096-(84+48)*18800640)
proof['config'].update(memory_limit_bytes=10*1024**3,expert_cache_limit_bytes=84*18800640)
proof['sha256']={p:d for p,d in proof['sha256'].items() if not p.startswith(str(BASE))}
for p in ROOT.iterdir():
    if p.is_file() and p.name!='installation.json':
        if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
        proof['sha256'][str(p)]=sha(p)
command=(BASE/'command.sh').read_text().replace(str(BASE),str(ROOT))
(ROOT/'command.sh').write_text(command)
proof['sha256'][str(ROOT/'command.sh')]=sha(ROOT/'command.sh')
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'root':str(ROOT),'source_commit':head,'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],'cpu_only':True}))
