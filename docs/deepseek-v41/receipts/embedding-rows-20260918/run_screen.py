"""Own a projection subprocess; reclaim its source pages only after process exit."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent-held GPU/service guard required')
ROOT = Path(__file__).resolve().parent
installation = json.loads((ROOT / 'installation.json').read_text())
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != installation['source_commit']:
    raise RuntimeError('source commit differs from pinned screen')
if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True):
    raise RuntimeError('tracked source is dirty')
for name, expected in installation['helper_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('screen helper identity changed')
for name, expected in installation['runtime_source_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('runtime identity changed')
for name in ('probe.json', 'child.json', 'reclamation.json'):
    if (ROOT / name).exists():
        raise RuntimeError('refusing to overwrite prior screen evidence')
spec = importlib.util.spec_from_file_location('draft_reclaim', 'scripts/deepseek_v41/reclaim_file_cache.py')
reclaim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reclaim)
files = [Path(installation['embedding']['path'])]

def now():
    return datetime.now(timezone.utc).isoformat()

def interrupted(signum, frame):
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
env = os.environ.copy()
env['DYLD_LIBRARY_PATH'] = str(Path(installation['strict_allocator']['path']).parent)
child = subprocess.Popen([sys.executable, str(ROOT / 'probe.py')], env=env)
life = {'pid': child.pid, 'started_at': now(), 'returncode': None}
(ROOT / 'child.json').write_text(json.dumps(life, indent=2) + '\n')
try:
    code = child.wait()
finally:
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    life.update(returncode=child.returncode, exited_at=now())
    (ROOT / 'child.json').write_text(json.dumps(life, indent=2) + '\n')
    rows = [reclaim.reclaim_file(path) for path in files]
    report = {'child_terminal': True, 'child_returncode': child.returncode,
              'completed_at': now(), 'files': rows,
              'cached_page_bytes_before': sum(r['cached_page_bytes_before'] for r in rows),
              'cached_page_bytes_after': sum(r['cached_page_bytes_after'] for r in rows)}
    (ROOT / 'reclamation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('EMBEDDING_SOURCE_CACHE_RECLAIMED', json.dumps({k:v for k,v in report.items() if k != 'files'}), flush=True)
    if report['cached_page_bytes_after']:
        raise RuntimeError('draft source pages remained cached after child exit')
sys.exit(code if code >= 0 else 128 - code)
