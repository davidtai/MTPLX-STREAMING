"""Compose the measured extension layout with the screened native Q4 draft subset."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

root = Path(__file__).resolve().parent
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base = Path('/tmp/dsv41-extension-bank-20260919/full-v1').resolve()
subset = Path('/tmp/dsv41-draft-surrogates-20260918').resolve()
out = root / 'full-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert head == 'a15ba7c40b6e59c7e3c3fdae852112085bddf40e'
screen = json.loads((subset / 'v2/head-screen.json').read_text())
artifact = json.loads((subset / 'artifact/receipt.json').read_text())
draft = screen['draft_pruning']['one_band']
arm = next(a for a in screen['arms'] if a['mode'] == 'one_band')
assert screen['complete'] and artifact['complete'] and arm['cycles'] == 198 and arm['verify_rows'] == 1242
assert artifact['selected_experts_by_stage'] == draft['selected_experts_by_stage']
assert artifact['removed_payload_bytes'] == 733224960
assert tuple(map(len, artifact['selected_experts_by_stage'])) == (80, 40, 24)
out.mkdir()
for sub in ('native', 'compat', 'packed'):
    install = json.loads((base / sub / 'installation.json').read_text())
    for name, digest in install.get('runtime_source_sha256', {}).items():
        assert sha(repo / name) == digest, name
    for name, digest in install['helper_sha256'].items():
        assert sha(base / sub / name) == digest, name
    shutil.copytree(base / sub, out / sub, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
for p in out.rglob('*.py'):
    p.write_text(p.read_text().replace(str(base), str(out)))
(out / 'packed/draft-config.json').write_text(json.dumps(draft, indent=2) + '\n')

runner = out / 'packed/run_full.py'
s = runner.read_text()
old_runner = (subset / 'full-v1/packed/run_full.py').read_text()
start = s.index('# Candidate-only installation.')
end = s.index('\n\nclass _CompactMTPExpertSwitch', start)
a = old_runner.index('# Draft-only approximate aliases')
b = old_runner.index('\n\nclass _CompactMTPExpertSwitch', a)
s = s[:start] + old_runner[a:b] + s[end:]
s = s.replace("Path('/tmp/dsv41-compact-residents')", f"Path({str(subset / 'artifact')!r})")
s = s.replace('_COMPACT_RESIDENT_RECEIPT = json.loads(',
    "if hashlib.sha256((_COMPACT_RESIDENT_ROOT/'receipt.json').read_bytes()).hexdigest() != DRAFT_PROOF['artifact_receipt_sha256']:\n    raise RuntimeError('compact draft artifact receipt changed')\n_COMPACT_RESIDENT_RECEIPT = json.loads(")
for old, new in [
    ('(201, 3_778_928_640)', '(240, 4_512_153_600)'),
    ('2: (20_166_777_672, 94, 89_686_016)', '2: (19_433_552_712, 94, 89_686_016)'),
    ('engine != 90194844488 +', 'engine != 89461619528 +'),
    ("ALLOCATOR_LIMIT_BYTES = growth_admission['allocator_limit_bytes']",
     "if growth_admission['decode_slots_per_layer'] < 112:\n    raise RuntimeError('smaller draft cannot fund the intended additional target rows with background reserve')\nALLOCATOR_LIMIT_BYTES = growth_admission['allocator_limit_bytes']"),
    (" 'mtp_pruned_experts':MTP_PRUNED_EXPERTS, 'mtp_pruned_bytes':MTP_PRUNED_BYTES,",
     " 'mtp_pruned_experts':MTP_PRUNED_EXPERTS, 'mtp_pruned_bytes':MTP_PRUNED_BYTES,\n 'draft_surrogates':DRAFT_PROOF,"),
]:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
runner.write_text(s)

p = out / 'packed/packed_admission.py'
s = p.read_text()
old_admission = (subset / 'full-v1/packed/packed_admission.py').read_text()
a = old_admission.index("    draft_proof = installation['draft_surrogates']")
b = old_admission.index('    embedding_credit = 1323827200', a)
s = s.replace('    embedding_credit = 1323827200', old_admission[a:b] + '    embedding_credit = 1323827200')
s = s.replace('    expansion_host = 16 * 1024**2',
    '    expansion_host = 16 * 1024**2\n    background_growth = 256 * 1024**2')
s = s.replace(' - embedding_host - lookup_host - expansion_host',
    ' - embedding_host - lookup_host - expansion_host - background_growth')
s = s.replace('embedding_host + lookup_host + expansion_host',
    'embedding_host + lookup_host + expansion_host + background_growth')
for name in ('steady', 'resize', 'seed'):
    needle = f'        {name} -= embedding_credit'
    assert s.count(needle) == 1
    s = s.replace(needle, needle + ' + draft_credit')
s = s.replace('range(112, old_capacity, -1)', 'range(113, old_capacity, -1)')
s = s.replace("transition_start_active_bound_bytes=original['transition_start_active_bound_bytes'] - embedding_credit,",
    "transition_start_active_bound_bytes=original['transition_start_active_bound_bytes'] - embedding_credit - draft_credit,\n        draft_resident_payload_credit_bytes=draft_credit, draft_prefill_bound_credit_bytes=0,\n        background_variation_allowance_bytes=background_growth,\n        process_host_reserve_bytes=original['host_reserve_bytes'] + embedding_host + lookup_host + expansion_host,\n        cli_host_reserve_scope='Aggregate non-MLX allowance including separately itemized background variation; not a measured Python allocation.',")
s = s.replace('capacity_search_ceiling=112', 'capacity_search_ceiling=113').replace('maximum_extension_bank_rows=28', 'maximum_extension_bank_rows=29')
s = s.replace('admitted85..112', 'admitted85..113').replace('plus1..28 extension', 'plus1..29 extension')
s = s.replace("bound_scope='Unchanged84-row", "bound_scope='Native144-expert draft removes733224960B from seed/transition/steady, no prefill credit. Explicit256MiB background allowance prices the prior107409300B launch-estimate miss with additional headroom. Unchanged84-row")
p.write_text(s)
p = out / 'packed/extension.py'
s = p.read_text(); assert s.count('old<capacity<=112') == 1
p.write_text(s.replace('old<capacity<=112', 'old<capacity<=113'))

scope = 'Exact16K/1024 Q4 target with native80/40/24 draft experts and screened router-nearest aliases; unchanged D5/lookup/M8 verification and packed projection schedule. Original84 rows plus up to29 extension rows. Explicit256MiB background allowance. No target arithmetic or quantization change.'
for sub in ('native', 'compat', 'packed'):
    p = out / sub / 'installation.json'
    d = json.loads(p.read_text()); d.update(source_commit=head, scope=scope)
    if sub == 'packed':
        d['native_admission_sha256'] = sha(out / 'native/admission.py')
        d['projection_ownership']['final_capacity_ceiling'] = 113
        d['projection_ownership']['scope'] = d['projection_ownership']['scope'].replace('112 rows', '113 rows')
        d['bank_extension'].update(maximum_final_rows=113, maximum_extension_rows=29,
            source_helper_sha256=sha(out / 'packed/extension.py'),
            geometry='Original indices0..83, extension0..28, both below proved native bank/index envelope. No resize/copy; complete allocation and one-bank raw-scale temporary priced.')
        d['draft_surrogates'] = {
            'head_screen_path': str(subset / 'v2/head-screen.json'),
            'head_screen_sha256': sha(subset / 'v2/head-screen.json'),
            'config_sha256': sha(out / 'packed/draft-config.json'),
            'artifact_receipt_path': str(subset / 'artifact/receipt.json'),
            'artifact_receipt_sha256': sha(subset / 'artifact/receipt.json'),
            'retired_payload_bytes': 733224960, 'target_band_bytes': 707788800,
            'minimum_decode_capacity': 112, 'target_verification': 'unchanged native',
            'host_metadata': 'Fixed384-entry alias mapping in existing helper allowance.',
        }
        d['background_variation'] = {'allowance_bytes': 256 * 1024**2,
            'reference': '/tmp/dsv41-110-stage/full-extension-bank-20260919-v1.jsonl',
            'reference_sha256': sha(Path('/tmp/dsv41-110-stage/full-extension-bank-20260919-v1.jsonl')),
            'observed_peak_above_estimate_bytes': 107409300,
            'scope': 'Reserve, not usage. Physical ceiling remains110000000000; fresh live admission and guard remain required.'}
        d['helper_sha256']['draft-config.json'] = 'pending'
    d['helper_sha256'] = {name: sha(out / sub / name) for name in d['helper_sha256']}
    p.write_text(json.dumps(d, indent=2) + '\n')

shutil.copy2(root/'verify_subset.py', out/'verify_subset.py')
for name in ('launch_full.py', 'wait_and_run.py'):
    (out / name).write_text((base / name).read_text().replace(str(base), str(out)))
launch = out/'launch_full.py'
launch.write_text(launch.read_text().replace('env=os.environ.copy();', "from verify_subset import verify\nverify(proof['draft_surrogates'], root/'artifact-verification.json')\nenv=os.environ.copy();"))
command = (base / 'command.sh').read_text().replace(str(base), str(out))
command = command.replace('full-extension-bank-20260919-v1', 'full-extension-draft-20260919-v1')
command = command.replace('GPU_WINDOW_CANDIDATE_AUX_DIR=/tmp/dsv41-compact-residents', f'GPU_WINDOW_CANDIDATE_AUX_DIR={subset}/artifact')
parts = shlex.split(command)
idx = parts.index('--host-overhead-gib') + 1
old_host = parts[idx]
new_host = format(1707208704 / 1024**3, '.17g')
command = command.replace('--host-overhead-gib ' + old_host, '--host-overhead-gib ' + new_host)
(out / 'command.sh').write_text(command)
s = (base / 'preflight.py').read_text().replace("a['decode_slots_per_layer']==112", "a['decode_slots_per_layer']==113")
s = s.replace('engine=83426614088', 'engine=82693389128').replace('extension-admission.json', 'draft-extension-admission.json')
s = s.replace('import sys\n', 'import sys\nimport os\n').replace('sys.meta_path.insert(0,NoMLX())', "sys.meta_path.insert(0,NoMLX())\nos.environ['MTPLX_ENGRAM_CACHE_LIMIT']='67108864'")
(out / 'preflight.py').write_text(s)
for p in out.rglob('*.py'):
    ast.parse(p.read_text())
    assert str(base) not in p.read_text(), p
audit = {'source_commit': head, 'scope': scope, 'draft_subset_payload_bytes': 2707292160,
    'credit_bytes': 733224960, 'background_allowance_bytes': 256 * 1024**2,
    'maximum_final_rows': 113, 'maximum_extension_component_bytes': 29 * 5898240,
    'prior_head_cycles': 198, 'prior_head_verify_rows': 1242,
    'target_kernels_unchanged': all(sha(out / 'packed' / n) == sha(base / 'packed' / n) for n in ('plane_lane.py', 'packed_storage.py', 'paired_kernels.py', 'kernels.py', 'hybrid_install.py', 'lookup.py', 'projection_install.py')),
    'full_result': 'unmeasured'}
(out / 'source-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit))
