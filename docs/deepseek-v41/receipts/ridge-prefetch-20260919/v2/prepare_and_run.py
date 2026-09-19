"""Prepare CPU parameters, reap that child, then use the existing GPU supervisor."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('canonical parent guard required')
root = Path(__file__).resolve().parent
proof = json.loads((root / 'installation.json').read_text())
for name, digest in proof['helper_sha256'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('preparation helper identity changed: ' + name)
if (root / 'preparation-child.json').exists():
    raise RuntimeError('refusing prior lifecycle overwrite')


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
child = subprocess.Popen([sys.executable, str(root / 'prepare.py')])
life = {'pid': child.pid, 'started_at': datetime.now(timezone.utc).isoformat(), 'returncode': None}
(root / 'preparation-child.json').write_text(json.dumps(life, indent=2) + '\n')
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
    life.update(returncode=child.returncode, exited_at=datetime.now(timezone.utc).isoformat())
    (root / 'preparation-child.json').write_text(json.dumps(life, indent=2) + '\n')
if code != 0:
    raise SystemExit(code if code >= 0 else 128 - code)
os.execv(sys.executable, [sys.executable, str(root / 'run_screen.py')])
