"""Clone the retained full Q4 lane; price and bind consensus independently."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base = Path('/tmp/dsv41-hybrid-lookup-20260918/full-v1').resolve()
r = root / 'full-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert head == '4b5591e8245addba2c96105cc37c34e8a63233a5'
screen = json.loads((root / 'head/head-screen.json').read_text())
assert screen['complete'] and screen['source_commit'] == head
assert [(a['cycles'], a['verify_rows']) for a in screen['arms']] == [(198, 1242), (195, 1240)]
inventory = json.loads((root / 'host-inventory.json').read_text())
assert inventory['complete'] and inventory['extra_host_allowance_bytes'] == 32 * 1024**2
confidence_margin = min(abs(min(row['confidence']) - 0.9)
                        for arm in screen['arms'] for row in arm['rows'])
assert confidence_margin > 0.002
r.mkdir()
for sub in ('native', 'compat', 'packed'):
    d = json.loads((base / sub / 'installation.json').read_text())
    for name, digest in d.get('runtime_source_sha256', {}).items():
        assert sha(repo / name) == digest, name
    for name, digest in d['helper_sha256'].items():
        assert sha(base / sub / name) == digest, name
    shutil.copytree(base / sub, r / sub, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
for p in r.rglob('*.py'):
    p.write_text(p.read_text().replace(str(base), str(r)))
shutil.copyfile(root / 'consensus.py', r / 'packed/consensus.py')
shutil.copyfile(root / 'consensus_install.py', r / 'packed/hybrid_install.py')
p = r / 'packed/packed_admission.py'
s = p.read_text()
assert s.count('lookup_host = 16 * 1024**2') == 1
s = s.replace('lookup_host = 16 * 1024**2', 'lookup_host = 48 * 1024**2')
s = s.replace('lookup_host_allowance_bytes=lookup_host,',
              'lookup_host_allowance_bytes=lookup_host,\n        consensus_extra_host_allowance_bytes=32 * 1024**2,')
s = s.replace('Adds16MiB fixed-workload lookup metadata',
              'Adds48MiB fixed-workload proposal metadata (16MiB original lookup plus32MiB consensus)')
p.write_text(s)
p = r / 'packed/projection_install.py'
s = p.read_text()
assert s.count('rows in (1, 6, 8)') == 1
s = s.replace('rows in (1, 6, 8)', 'rows in range(1, 9)')
p.write_text(s)
for sub in ('native', 'compat', 'packed'):
    p = r / sub / 'installation.json'
    d = json.loads(p.read_text())
    d['source_commit'] = head
    d['scope'] = ('Exact16K/1024 nativeKV16, native D5 and original lookup plus causal consensus suffix. '
                  'Target M6/M7/M8, all within the retained M8 tensor and48-slot envelope. '
                  'Original target arithmetic; additional32MiB host reserve.')
    if sub == 'packed':
        d['native_admission_sha256'] = sha(r / 'native/admission.py')
        d['helper_sha256']['consensus.py'] = 'pending'
        d['hybrid_lookup'] = {
            'head_screen_path': str(root / 'head/head-screen.json'),
            'head_screen_sha256': sha(root / 'head/head-screen.json'),
            'host_inventory_path': str(root / 'host-inventory.json'),
            'host_inventory_sha256': sha(root / 'host-inventory.json'),
            'maximum_proposal_depth': 7, 'native_head_depth': 5, 'maximum_verify_rows': 8,
            'host_allowance_bytes': 48 * 1024**2, 'consensus_extra_host_bytes': 32 * 1024**2,
            'head_replay_minimum_confidence_margin': confidence_margin,
            'confidence_threshold_scope': 'Compare already evaluated native logits to log(9); every saved decision is more than0.002 away from sigmoid0.9.',
            'target_arithmetic': 'Unchanged native accept/commit and row-dependent M<=8 packed target.'}
    d['helper_sha256'] = {name: sha(r / sub / name) for name in d['helper_sha256']}
    p.write_text(json.dumps(d, indent=2) + '\n')
(r / 'launch_full.py').write_text((base / 'launch_full.py').read_text().replace(str(base), str(r)))
s = (base / 'command.sh').read_text().replace(str(base), str(r))
s = s.replace('full-hybrid-lookup-20260918-v1', 'full-suffix-consensus-20260919-v1')
s = s.replace('--host-overhead-gib 1.3243370056152344', '--host-overhead-gib 1.3555870056152344')
s = s.replace('GPU_WINDOW_LOCK_TIMEOUT=120', 'GPU_WINDOW_LOCK_TIMEOUT=600')
(r / 'command.sh').write_text(s)
s = (base / 'preflight.py').read_text()
s = s.replace("a['lookup_host_allowance_bytes']==16*1024**2", "a['lookup_host_allowance_bytes']==48*1024**2")
s = s.replace('hybrid-admission.json', 'consensus-admission.json')
(r / 'preflight.py').write_text(s)
for p in r.rglob('*.py'):
    ast.parse(p.read_text())
audit = {'source_commit': head, 'base': str(base), 'extra_host_bytes': 32 * 1024**2,
         'helper_sha256': {str(p.relative_to(r)): sha(p) for p in r.rglob('*')
                           if p.is_file() and p.suffix in ('.py', '.json') and 'artifact' not in p.parts}}
(r / 'source-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({'root': str(r), 'source': head, 'extra_host_bytes': 32 * 1024**2}))
