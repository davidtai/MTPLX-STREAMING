"""Stage a new Q8 reference from the reviewed packed-capacity harness."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
WT = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
ORIGINAL = Path('/tmp/dsv41-prefill84-pair-20260917/packed')
packed = ROOT / 'packed'
compat = ROOT / 'compat'
packed.mkdir(exist_ok=True)
compat.mkdir(exist_ok=True)
for name in ('kernels.py', 'paired_kernels.py', 'packed_storage.py', 'packed_phase.py',
             'packed_admission.py', 'storage-probe.json', 'storage_probe.py'):
    shutil.copyfile(ORIGINAL / name, packed / name)
if not (packed / 'artifact').exists():
    (packed / 'artifact').symlink_to((ORIGINAL / 'artifact').resolve(), target_is_directory=True)
for name in ('admission.py', 'phase_growth.py', 'bank_growth_final.py'):
    shutil.copyfile(Path('/tmp/dsv41-cache-growth-20260917') / name, compat / name)
c = json.loads(Path('/tmp/dsv41-cache-growth-20260917/installation.json').read_text())
c['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()
for name in list(c['runtime_source_sha256']) + ['mtplx/expert_runtime.py',
        'mtplx/models/deepseek_v41_loader.py','mtplx/models/deepseek_v41_cache.py',
        'mtplx/models/deepseek_v41_fixed_q8_cache.py',
        'mtplx/models/deepseek_v41_dspark.py','mtplx/models/deepseek_v41_dspark_decode.py']:
    c['runtime_source_sha256'][name] = hashlib.sha256((WT/name).read_bytes()).hexdigest()
c['scope'] = 'Fresh Q8 target KV reference and D5/M6 candidate, fixed max17664/append953; native target/head weights and packed expert decode; preserve native memory allowances and add full Q8 reserve.'
c['q8_change'] = {'cache_bits':8,'max_kv':17664,'max_append':953,
    'additional_allocator_reserve_bytes':503316480,
    'ar_reference_host_logit_reserve_bytes':640*1024**2,
    'lifetime_probe_sha256':hashlib.sha256(Path('/tmp/dsv41-q8-lifetimes-20260917/probe.json').read_bytes()).hexdigest()}
(compat/'installation.json').write_text(json.dumps(c,indent=2)+'\n')
p = json.loads((ORIGINAL/'installation.json').read_text())
p.update(source_commit=c['source_commit'],scope=c['scope'],q8_change=c['q8_change'])
# Lower capacity uses the same additive byte formula and the measured cap84
# prefill. This removes a native-predecessor nomination restriction; it does
# not relax the physical, allocator, wired, or copy bounds.
native=(ORIGINAL.parent/'native/admission.py').read_text()
native=native.replace('range(100, 95, -1)','range(100, 83, -1)')
native=native.replace('range(96, 101)','range(84, 101)')
native=native.replace('96..100','84..100')
(ROOT/'native_admission.py').write_text(native)
pa=(packed/'packed_admission.py').read_text().replace(
    '/tmp/dsv41-prefill84-pair-20260917/native/admission.py',str(ROOT/'native_admission.py'))
(packed/'packed_admission.py').write_text(pa)
p['native_admission_sha256']=hashlib.sha256(native.encode()).hexdigest()
p['helper_sha256']['packed_admission.py']=hashlib.sha256(pa.encode()).hexdigest()
p['helper_sha256'].pop('run_full.py')
(packed/'installation.json').write_text(json.dumps(p,indent=2)+'\n')

source=(ORIGINAL/'run_full.py').read_text()
def change(old,new):
    global source
    if source.count(old)!=1:
        raise RuntimeError('unexpected source substitution count: '+old[:100])
    source=source.replace(old,new)

change("PACKED_ROOT = Path('/tmp/dsv41-prefill84-pair-20260917/packed')",
       f"PACKED_ROOT = Path({str(packed)!r})")
change("GROWTH_ROOT = Path('/tmp/dsv41-cache-growth-20260917')",
       f"GROWTH_ROOT = Path({str(compat)!r})")
change("from packed_admission import resolve_admission", "from q8_admission import resolve_admission")
change("FIXED_MTP, TARGET_SLOTS, TARGET_REMAINDER = RESERVE_PLANS[runtime_reserve_gib]",
       "FIXED_MTP, TARGET_SLOTS, TARGET_REMAINDER = RESERVE_PLANS[runtime_reserve_gib]\nFIXED_MTP += 503316480")
change("if engine != 90194844488 + (slots - 93) * SLOT_BAND_BYTES:",
       "if engine != 90194844488 + 503316480 + (slots - 93) * SLOT_BAND_BYTES:")
change("or args.arms != ['cell16k_ring_v2_draft_attn_pf0'] or args.decode_mode != 'dspark'",
       "or args.arms != ['cell16k_ring_v2_draft_attn_pf0'] or args.decode_mode != 'ar' or not args.with_mtp")
change("or args.max_kv != 17664 or args.box_target_gb != 110",
       "or args.max_kv != 17664 or args.box_target_gb != 110 or args.kv_cache_bits != 8 or args.kv_max_append != 953")
change("or args.allocator_cache_gib != 1 or args.host_overhead_gib != 2",
       "or args.allocator_cache_gib != 1 or args.host_overhead_gib != 2.625")
change("'fixed_footprint_bytes':FIXED_MTP, 'plan_remainder_bytes':TARGET_REMAINDER,",
       "'fixed_footprint_bytes':FIXED_MTP, 'plan_remainder_bytes':TARGET_REMAINDER,\n 'fixed_q8_cache':COMPATIBILITY['q8_change'],")

prefix=source[:source.index('# A prior complete AR run may supply only the comparison token stream.')]
prefix += "PREFIX.with_suffix('.bounds.json').write_text(json.dumps(bounds, indent=2) + '\\n')\n\n"
middle=source[source.index('def record_pass(kind, result):'):source.index('    original_ar = ab._generate')]
footer=source[source.index('finally:\n    stop.set()'):]
footer=footer.replace("    PREFIX.with_suffix('.reclamation.json').write_text(",
    "    _owned_logits = PREFIX.with_suffix('.ar-logits.f32')\n"
    "    if _owned_logits.exists():\n"
    "        _cleanup.append(_cleanup_module.reclaim_file(_owned_logits))\n"
    "    PREFIX.with_suffix('.reclamation.json').write_text(")
custom=(ROOT/'ar_body.txt').read_text()
result=prefix+middle+custom+footer
ast.parse(result)
(ROOT/'ar_reference.py').write_text(result)
p['helper_sha256']['../ar_reference.py']=hashlib.sha256(result.encode()).hexdigest()
p['helper_sha256']['../q8_admission.py']=hashlib.sha256((ROOT/'q8_admission.py').read_bytes()).hexdigest()
p['helper_sha256']['../native_admission.py']=hashlib.sha256(native.encode()).hexdigest()
(packed/'installation.json').write_text(json.dumps(p,indent=2)+'\n')
print('Staged Q8 reference wrapper',hashlib.sha256(result.encode()).hexdigest())
