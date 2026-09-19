"""Advance the fixed predictor's issue point; preserve the completed v1 receipt."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

root = Path(__file__).resolve().parent
b = root / 'v1'
r = root / 'v2'
r.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
proof = json.loads((b / 'installation.json').read_text())
previous = json.loads((b / 'probe.json').read_text())
assert previous['complete'] and all(a['all_outputs_exact'] for a in previous['arms'])
for name, digest in proof['helper_sha256'].items():
    assert sha(b / name) == digest, name
    shutil.copyfile(b / name, r / name)
for name in ('preflight.py', 'preparation.json', 'ridge-parameters.npz', 'wait_and_run.py'):
    shutil.copyfile(b / name, r / name)
(r / 'artifact').symlink_to(b / 'artifact', target_is_directory=True)
p = r / 'plane_lane.py'
s = p.read_text()
begin = s.index('class PrefetchDecode(')
prefix, body = s[:begin], s[begin:]
old = '''            if issue_pending:
                issue_pending = False
                # The first GU witness creates earlier lead. All later demand
                # jobs outrank queued speculation; active reads still finish.
                self.issue()
'''
assert body.count(old) == 1
body = body.replace(old, '')
body = body.replace('        issue_pending = True\n', '')
body = body.replace('            nonlocal issue_pending\n', '')
old = '''            if not parts:
                self.issue()
'''
new = '''            # Demand parts already exist and resident/shared GPU roots
            # are enqueued. Start speculative reads before a miss GU completes.
            self.issue()
'''
assert body.count(old) == 1
body = body.replace(old, new)
assert 'issue_pending' not in body
p.write_text(prefix + body)
proof['scope'] = proof['scope'].replace('First-GU issue', 'Issue after demand submission and resident/shared GPU submission')
proof['predecessor'] = {'path': str(b / 'probe.json'), 'sha256': sha(b / 'probe.json'),
                        'latency_ratio': previous['latency_ratio'], 'only_change': 'earlier prefetch issue point'}
proof['parameter_reuse'] = {'path': str(b / 'preparation.json'), 'sha256': sha(b / 'preparation.json'),
                            'parameters_sha256': sha(b / 'ridge-parameters.npz'), 'refitted': False}
proof['helper_sha256'] = {}
(r / 'installation.json').write_text(json.dumps(proof, indent=2) + '\n')
p = r / 'run_screen.py'
s = p.read_text()
anchor = "installation = json.loads((ROOT / 'installation.json').read_text())"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''
reuse = installation['parameter_reuse']
for name, expected in (('preparation.json', reuse['sha256']),
                       ('ridge-parameters.npz', reuse['parameters_sha256'])):
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('reused predictor artifact changed: ' + name)
''')
p.write_text(s)
command = (b / 'command.sh').read_text().replace(str(b), str(r))
command = command.replace(str(r / 'prepare_and_run.py'), str(r / 'run_screen.py'))
(r / 'command.sh').write_text(command)
for p in r.glob('*.py'):
    ast.parse(p.read_text())
print(json.dumps({'root':str(r),'change':'prefetch issue point only','parameters_reused':True,'bound':proof['static_incremental_bound_bytes']}))
