"""Compose one physically compact draft candidate with the retained full runner."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = Path('/tmp/dsv41-hybrid-lookup-20260918/full-v1').resolve()
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
OUT = ROOT/'full-v1'
OUT.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
screen = json.loads((ROOT/'v2/head-screen.json').read_text())
artifact = json.loads((ROOT/'artifact/receipt.json').read_text())
candidate = screen['draft_pruning']['one_band']
CREDIT = 733_224_960
assert screen['complete'] and artifact['complete'] and artifact['removed_payload_bytes'] == CREDIT
assert tuple(map(len,candidate['selected_experts_by_stage'])) == (80,40,24)
assert artifact['selected_experts_by_stage'] == candidate['selected_experts_by_stage']
for sub in ('native','compat','packed'):
    d = json.loads((BASE/sub/'installation.json').read_text())
    for name, digest in d.get('runtime_source_sha256',{}).items():
        assert sha(REPO/name) == digest, name
    for name, digest in d['helper_sha256'].items():
        assert sha(BASE/sub/name) == digest, name
    shutil.copytree(BASE/sub,OUT/sub,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
for p in OUT.rglob('*.py'):
    p.write_text(p.read_text().replace(str(BASE),str(OUT)))
(OUT/'packed/draft-config.json').write_text(json.dumps(candidate,indent=2)+'\n')

p = OUT/'packed/run_full.py'
s = p.read_text()
start = s.index('# Candidate-only installation.')
end = s.index('\n\nclass _CompactMTPExpertSwitch',start)
s = s[:start] + '''# Draft-only approximate aliases, fixed from a completed head quality screen.
# All target verification, acceptance and commit arithmetic remains native.
TRACE_SHA256 = "6a90006c9c4829dac2a548298c1d237bfd1beb793c0bc6eaf99315fad8c9c3dd"
DRAFT_PROOF = PACKED_INSTALLATION['draft_surrogates']
config_blob = (PACKED_ROOT/'draft-config.json').read_bytes()
if hashlib.sha256(config_blob).hexdigest() != DRAFT_PROOF['config_sha256']:
    raise RuntimeError('pinned draft alias configuration changed')
draft_config = json.loads(config_blob)
SELECTED_MTP_EXPERTS = tuple(tuple(ids) for ids in draft_config['selected_experts_by_stage'])
if tuple(map(len,SELECTED_MTP_EXPERTS)) != (80,40,24):
    raise RuntimeError('compact draft geometry differs')
_COMPACT_LUTS = []
for selected, aliases in zip(SELECTED_MTP_EXPERTS,draft_config['alias_experts_by_stage']):
    positions = {expert:slot for slot,expert in enumerate(selected)}
    if len(aliases) != 128 or any(e not in positions for e in aliases):
        raise RuntimeError('draft alias points outside installed compact storage')
    if any(aliases[e] != e for e in selected):
        raise RuntimeError('retained draft expert must map to itself')
    _COMPACT_LUTS.append(mx.array([positions[e] for e in aliases],dtype=mx.int32))
''' + s[end:]
s = s.replace("Path('/tmp/dsv41-compact-residents')",f"Path({str(ROOT/'artifact')!r})")
s = s.replace("_COMPACT_RESIDENT_RECEIPT = json.loads(", "if hashlib.sha256((_COMPACT_RESIDENT_ROOT/'receipt.json').read_bytes()).hexdigest() != DRAFT_PROOF['artifact_receipt_sha256']:\n    raise RuntimeError('compact draft artifact receipt changed')\n_COMPACT_RESIDENT_RECEIPT = json.loads(")
s = s.replace('(201, 3_778_928_640)', '(240, 4_512_153_600)')
s = s.replace('2: (20_166_777_672, 94, 89_686_016)', f'2: ({20_166_777_672-CREDIT}, 94, 89_686_016)')
s = s.replace('engine != 90194844488 +', f'engine != {90194844488-CREDIT} +')
needle = "ALLOCATOR_LIMIT_BYTES = growth_admission['allocator_limit_bytes']"
assert s.count(needle) == 1
s = s.replace(needle,"if growth_admission['decode_slots_per_layer'] != 111:\n    raise RuntimeError('smaller draft bank cannot fund the intended extra target band at this baseline')\n"+needle)
needle = " 'mtp_pruned_experts':MTP_PRUNED_EXPERTS, 'mtp_pruned_bytes':MTP_PRUNED_BYTES,"
assert s.count(needle) == 1
s = s.replace(needle,needle+"\n 'draft_surrogates':DRAFT_PROOF,")
p.write_text(s)

p = OUT/'packed/packed_admission.py'
s = p.read_text()
needle = '    embedding_credit = 1323827200'
assert s.count(needle) == 1
s = s.replace(needle,"""    draft_proof = installation['draft_surrogates']
    artifact_path = Path(draft_proof['artifact_receipt_path'])
    blob = artifact_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != draft_proof['artifact_receipt_sha256']:
        raise RuntimeError('true draft subset allocation proof changed')
    draft_artifact = json.loads(blob)
    draft_credit = draft_artifact['removed_payload_bytes']
    if (not draft_artifact['complete'] or draft_credit != 733224960
        or tuple(map(len,draft_artifact['selected_experts_by_stage'])) != (80,40,24)
        or sum(f['payload_bytes'] for f in draft_artifact['files']) != 2707292160):
        raise RuntimeError('true draft subset geometry differs')
    # True subset files and smaller installed expert axes remove this payload
    # throughout execution. Keep the larger original prefill bound anyway.
"""+needle)
for expr in ('steady','resize','seed'):
    needle = f'        {expr} -= embedding_credit'
    assert s.count(needle) == 1
    s = s.replace(needle,needle+' + draft_credit')
s = s.replace("transition_start_active_bound_bytes=original['transition_start_active_bound_bytes'] - embedding_credit,",
              "transition_start_active_bound_bytes=original['transition_start_active_bound_bytes'] - embedding_credit - draft_credit,\n        draft_resident_payload_credit_bytes=draft_credit, draft_prefill_bound_credit_bytes=0,")
s = s.replace('through110;', 'through111;')
s = s.replace("bound_scope='Adds16MiB", "bound_scope='True144-expert draft subset removes733224960B from resize/seed/steady; original prefill upper bound retained. Adds16MiB")
p.write_text(s)

for sub in ('native','compat','packed'):
    p = OUT/sub/'installation.json'
    d = json.loads(p.read_text())
    d.update(source_commit=source,scope='16K/1024 native target D5 plus causal lookup; true144-expert draft bank with fixed router-nearest aliases;84->111 packed slots under110GB. Target arithmetic unchanged; draft quality screen198 calls but full parity and throughput remain unmeasured.')
    if sub == 'packed':
        d['native_admission_sha256'] = sha(OUT/'native/admission.py')
        d['strict_allocator']['capacity_search_ceiling'] = 111
        d['strict_allocator']['capacity_geometry'] = ('Page-aligned5898240B components,111 persistent rows plus48 shared transients; per-component654704640B fits prior signed32-bit offsets. Existing native M<=8 route/kernel geometry unchanged. All copy/seed/steady/wired inequalities still apply.')
        d['draft_surrogates'] = {'head_screen_path':str(ROOT/'v2/head-screen.json'),
            'head_screen_sha256':sha(ROOT/'v2/head-screen.json'),
            'config_sha256':sha(OUT/'packed/draft-config.json'),
            'artifact_receipt_path':str(ROOT/'artifact/receipt.json'),
            'artifact_receipt_sha256':sha(ROOT/'artifact/receipt.json'),
            'retired_payload_bytes':CREDIT,'target_band_bytes':707788800,
            'minimum_decode_capacity':111,'target_verification':'unchanged native',
            'host_metadata':'Fixed384-entry alias mapping within existing32MiB helper allowance.'}
    d['helper_sha256'] = {name:sha(OUT/sub/name) for name in d['helper_sha256']}
    p.write_text(json.dumps(d,indent=2)+'\n')
(OUT/'launch_full.py').write_text((BASE/'launch_full.py').read_text())
s = (BASE/'command.sh').read_text().replace(str(BASE),str(OUT))
s = s.replace('full-hybrid-lookup-20260918-v1','full-draft-surrogates-20260918-v1')
s = s.replace('GPU_WINDOW_CANDIDATE_AUX_DIR=/tmp/dsv41-compact-residents',f'GPU_WINDOW_CANDIDATE_AUX_DIR={ROOT}/artifact')
(OUT/'command.sh').write_text(s)
s = (BASE/'preflight.py').read_text()
s = s.replace("a['decode_slots_per_layer']==110", "a['decode_slots_per_layer']==111")
s = s.replace('engine=83426614088',f'engine={83426614088-CREDIT}')
s = s.replace("(r/'hybrid-admission.json')", "(r/'draft-admission.json')")
s = s.replace("assert a['embedding_post_prefill_credit_bytes']==1323827200", "assert a['draft_resident_payload_credit_bytes']==733224960 and a['draft_prefill_bound_credit_bytes']==0\nassert a['embedding_post_prefill_credit_bytes']==1323827200")
(OUT/'preflight.py').write_text(s)
for p in OUT.rglob('*.py'):
    ast.parse(p.read_text())
    if str(BASE) in p.read_text(): raise RuntimeError('cloned helper retains old root')
audit = {'source_commit':source,'base':str(BASE),'draft_payload_credit_bytes':CREDIT,
    'prefill_bound_credit_bytes':0,'target_arithmetic_unchanged':True,
    'native_helpers_changed':['run_full.py: draft construction, physical subset, fixed plan','packed_admission.py: true subset credit and ceiling111'],
    'helper_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ('.py','.json') and 'artifact' not in p.parts}}
(OUT/'source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'root':str(OUT),'source':source,'draft_payload_credit_bytes':CREDIT}))
