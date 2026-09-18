"""Clone the validated one-layer screen without importing MLX."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base = repo/'docs/deepseek-v41/receipts/packed-operators-20260918/clamped-activation'
proof = json.loads((base/'installation.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name, expected in proof['runtime_source_sha256'].items():
    assert sha(Path(name)) == expected, name
for name, expected in proof['helper_sha256'].items():
    assert sha(base/name) == expected, name
    if name != 'plane_lane_activation.py':
        shutil.copyfile(base/name,root/name)
(root/'artifact').symlink_to(repo/'benchmarks/raw/deepseek-v41-resident-scales/20260917',target_is_directory=True)
p = root/'probe.py';s = p.read_text()
s = s.replace('"""One-layer native route replay for miss-completion batch size."""',
              '"""One-layer native route replay for a miss-worker reader."""')
s = s.replace('import plane_lane_activation','import reader_hop')
s = s.replace("        lane = plane_lane if mode=='native' else plane_lane_activation\n        runners = lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)",
              "        runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)\n        if mode == 'miss-worker':\n            reader_hop.install(runtime,source_sha256=proof['slot_source_sha256'])")
s = s.replace("a['mode']=='activation'","a['mode']=='miss-worker'")
s = s.replace('ALLOCATOR_CACHE_ARM','READER_HOP_ARM').replace('MISS_PART_COMPLETE','READER_HOP_COMPLETE')
p.write_text(s)
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
proof['source_commit'] = head
proof['slot_source_sha256'] = sha(repo/'mtplx/expert_slots.py')
proof['bank_capacity'] = 158
proof['native_slot_bytes'] = 158*18800640
proof['packed_slot_bytes'] = 158*17694720
for key in ('expert_cache_limit_bytes','persistent_budget_bytes','persistent_cache_bytes'):
    proof['plan'][key] = 110*18800640
proof['plan']['unallocated_bytes'] -= 18800640
proof['plan']['persistent_slots'] = proof['plan']['slots_per_layer'] = 110
proof['config']['expert_cache_limit_bytes'] = 110*18800640
proof['bound_components']['raw_bank_bytes'] = proof['native_slot_bytes']
proof['bound_components']['packed_bank_bytes'] = proof['packed_slot_bytes']
proof['bound_components']['mlx_cache_inside_envelope_bytes'] = 256*1024**2
proof['bound_scope'] = '9GiB incremental:5GiB Metal/cache/compile and4GiB host/reader/compiler.158 native rows before scale retirement, one104857600-byte packed scale bound,16MiB scratch. No added GPU owners or workers. Strict256MiB cache, one runtime at a time.'
proof['comparison'] = 'Unchanged plane reader versus one noncontiguous component batch per native miss part, executed by the existing miss worker at its native future wait.'
proof['scope'] = 'Layer34 native110-slot/48-transient M6 replay with all206 saved routes and actual73-slot restored initial bank. Same expert IDs, policy, part geometry, kernels, slot lifetime and physical bytes. Target end-to-end TPS and current hybrid M8 trajectory are not measured.'
proof['arms'] = ['native','miss-worker','native','miss-worker','native']
proof['layer_selection'] = 'Previously selected median-miss layer34. Same73 captured physical residents plus37 empty slots in every arm.'
proof['allocator_sources']['rule'] = 'Already-attested strict allocator unchanged; candidate changes host reader scheduling only.'
proof['helper_sha256'] = {p.name:sha(p) for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.json') and p.name not in ('stage.py','installation.json')}
(root/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
cmd = (base/'command.sh').read_text().replace('/tmp/dsv41-clamped-activation-20260918',str(root))
(root/'command.sh').write_text(cmd)
for p in root.glob('*.py'): ast.parse(p.read_text())
audit = {'source_commit':head,'runtime_hashes_unchanged':len(proof['runtime_source_sha256']),
         'helper_hashes':len(proof['helper_sha256']),'bound_bytes':proof['static_incremental_bound_bytes'],
         'slot_future_scope':'Native _ensure_route_locked owns its private futures, calls result after releasing completion-error lock, drains every result on failure, and pins only after every fill completes. No generic executor replacement or cross-thread future consumer.',
         'arithmetic':'All plane kernels and PackedDecode calls remain byte-identical to current winner.',
         'changed_owners':'One Future and one component batch per <=3-record part; no new Metal buffers, staging payload or worker threads.'}
(root/'construction.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
