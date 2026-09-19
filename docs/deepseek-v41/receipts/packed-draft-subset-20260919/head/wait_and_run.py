"""Wait read-only for prior GPU windows; the canonical guard acquires anew."""
import fcntl
import json
import os
from pathlib import Path
import time
import urllib.request

r = Path(__file__).resolve().parent
deadline = time.monotonic() + 600
announced = False
while time.monotonic() < deadline:
    try:
        with open('/tmp/mtplx-gpu-exclusive.lock', 'r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        with urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2) as f:
            health = json.load(f)
        with urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=2) as f:
            models = json.load(f)
        if (health['ok'] and health['scheduler']['active_requests'] == 0
                and health['startup']['warmup']['background']['state'] == 'done'
                and [d['id'] for d in models['data']] == ['mtplx-flash-next-optimized-speed']):
            result = {'checked_at': time.time(), 'health': health, 'models': models,
                      'lock_free_at_probe': True,
                      'scope': 'Readiness only; canonical guard must acquire and retain lock before GPU work.'}
            (r / 'pre-run-health.json').write_text(json.dumps(result, indent=2) + '\n')
            print('GPU lane ready; invoking canonical guard with fresh admission.', flush=True)
            os.execv('/bin/bash', ['/bin/bash', str(r / 'command.sh')])
    except (BlockingIOError, OSError, KeyError, ValueError):
        pass
    if not announced:
        print('Waiting read-only for prior GPU owner and exact warmed Qwen service.', flush=True)
        announced = True
    time.sleep(0.5)
raise SystemExit('readiness timeout; no child launched and no service changed')
