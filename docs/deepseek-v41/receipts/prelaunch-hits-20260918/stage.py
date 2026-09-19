"""Construct the fixed-geometry prelaunch replay without importing MLX."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parent
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base=Path('/tmp/dsv41-reader-hop-20260918')
proof=json.loads((base/'installation.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in proof['helper_sha256'].items():
    assert sha(base/name)==digest,name
    if name!='reader_hop.py':shutil.copyfile(base/name,root/name)
for name,digest in proof['runtime_source_sha256'].items():assert sha(repo/name)==digest,name
(root/'artifact').symlink_to((base/'artifact').resolve(),target_is_directory=True)
p=root/'paired_kernels.py';native=p.read_text()
updated=native.replace('const uint slot=pairs[2*assignment];','const int slot=pairs[2*assignment];')
needle='const device ushort* wp='
assert updated.count(needle)==1
guard='''if(slot<0) {{
    if(lane==0) for(int r=0;r<4;r++) out[assignment*N+outrow+r]=T(0);
    return;
}}
'''
updated=updated.replace(needle,guard+needle)
updated=updated.replace("name=f'dsv41_packed_scales_{n}_{k}_paired'","name=f'dsv41_packed_scales_{n}_{k}_masked_hit'")
(root/'prelaunch_kernels.py').write_text(updated)
p=root/'probe.py';s=p.read_text().replace('import reader_hop','import prelaunch')
old="""        runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)
        if mode == 'miss-worker':
            reader_hop.install(runtime,source_sha256=proof['slot_source_sha256'])"""
new="""        if mode == 'native':
            runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)
        else:
            runners = prelaunch.install(runtime,{LAYER:switch},{LAYER:scales},
                runtime_sha256=proof['runtime_source_sha256'])"""
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("a['mode']=='miss-worker'","a['mode']=='prelaunch'")
s=s.replace('READER_HOP','PRELAUNCH_HIT')
s=s.replace('One-layer native route replay for a miss-worker reader.',
            'One-layer native route replay for early persistent-hit computation.')
p.write_text(s)
proof['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
proof['arms']=['native','prelaunch','native','prelaunch','native']
proof['comparison']='Native plane scheduler versus masked persistent-hit compute submitted after router async event and before CPU route planning. Same native miss path and arithmetic.'
proof['scope']='Layer34 native110/48 M6, actual73 initial residents plus37 empty slots, all206 saved routes. Inputs/router IDs are pre-evaluated for the operator; this measures overlap with CPU route planning, not earlier attention or full-model TPS.'
proof['bound_scope']='9GiB incremental:5GiB Metal/cache/compile plus4GiB host/reader/compiler.158 raw rows, one packed-scale layer,16MiB input/output/scratch including full masked48-assignment graphs and lookup. No extra workers, model tensors or I/O staging.'
proof['allocator_sources']['rule']='Pinned strict allocator unchanged. Masked kernels retain native dot loops, BF16 casts and geometry; all miss kernels and record readers are unchanged.'
proof['runtime_source_sha256']['mtplx/expert_streaming.py']=sha(repo/'mtplx/expert_streaming.py')
proof.pop('slot_source_sha256',None)
proof['helper_sha256']={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.json') and p.name not in ('stage.py','installation.json')}
(root/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
(root/'command.sh').write_text((base/'command.sh').read_text().replace(str(base.resolve()),str(root)).replace(str(base),str(root)))
for p in root.glob('*.py'):ast.parse(p.read_text())
audit={'source_commit':proof['source_commit'],'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],
 'native_kernel_sha256':sha(root/'paired_kernels.py'),'masked_kernel_sha256':sha(root/'prelaunch_kernels.py'),
 'arithmetic':'Only negative-slot early return and signed slot declaration added before unchanged dot body; miss kernels remain original.',
 'ownership':'Fixed single-request layer cache; no ring or background policy mutation. Persistent map contains READY records. Native planner protects every requested resident hit from victims before issuing writes. Native full-record publication and deferred pins cover downstream results; outer failure handler drains early consumers even before a pending route exists.',
 'synchronization':'MLX0.32.2 async_eval attaches an event to existing indices and finalizes that submission. Later early-hit async work does not replace the original array event. CPU reads original indices via buffer protocol, avoiding a new Metal reshape fence.',
 'full_model_promotion':False}
(root/'construction.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'root':str(root),'source':proof['source_commit'],'helper_hashes':len(proof['helper_sha256']),'bound':proof['static_incremental_bound_bytes']}))
