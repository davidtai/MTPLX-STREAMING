"""Own the bounded child and reclaim touched files after it has exited."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent-held GPU/service guard required')
ROOT = Path(__file__).resolve().parent
proof = json.loads((ROOT/'installation.json').read_text())
if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() != proof['source_commit']:
    raise RuntimeError('source commit differs from pinned candidate')
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True):
    raise RuntimeError('tracked source is dirty')
for name,expected in proof['sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('pinned source changed: '+name)
for name in ('probe.json','child.json','reclamation.json'):
    if (ROOT/name).exists():
        raise RuntimeError('refusing to overwrite component evidence')
spec = importlib.util.spec_from_file_location('reclaim','scripts/deepseek_v41/reclaim_file_cache.py')
reclaim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reclaim)
def interrupted(signum,frame):
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
env = os.environ.copy()
env['DYLD_LIBRARY_PATH'] = str(Path(proof['strict_allocator']['path']).parent)
child = subprocess.Popen([sys.executable,str(ROOT/'probe.py')],env=env)
life = {'pid':child.pid,'started_at':time.time(),'returncode':None}
(ROOT/'child.json').write_text(json.dumps(life,indent=2)+'\n')
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
    life.update(returncode=child.returncode,exited_at=time.time())
    (ROOT/'child.json').write_text(json.dumps(life,indent=2)+'\n')
    files = [Path(name) for name in proof['touched_files']]
    rows = [reclaim.reclaim_file(path) for path in files]
    result = {'child_terminal':True,'child_returncode':child.returncode,'files':rows,
              'cached_page_bytes_before':sum(r['cached_page_bytes_before'] for r in rows),
              'cached_page_bytes_after':sum(r['cached_page_bytes_after'] for r in rows)}
    (ROOT/'reclamation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SOURCE_CACHE_RECLAIMED',json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
    if result['cached_page_bytes_after']:
        raise RuntimeError('source pages remained cached after child exit')
sys.exit(code if code >= 0 else 128-code)
