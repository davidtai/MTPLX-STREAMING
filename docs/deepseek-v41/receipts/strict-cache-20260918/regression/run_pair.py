import hashlib,json,os,signal,subprocess,sys
from pathlib import Path
if os.environ.get('_GPU_WINDOW_LOCKED')!='1':raise RuntimeError('parent GPU guard required')
root=Path(__file__).resolve().parent
proof=json.loads((root/'installation.json').read_text())
for name,digest in proof['helper_sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('regression helper changed')
def interrupt(signum,frame):raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
for binary in ('stock_source','strict'):
    if (root/(binary+'.json')).exists():raise RuntimeError('refusing completed evidence overwrite')
    env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(Path(proof['binary_choices'][binary]['path']).parent);env['DSV41_OPERATOR_BINARY']=binary
    child=subprocess.Popen([sys.executable,str(root/'run_checks.py')],env=env)
    try:code=child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    if code:raise RuntimeError('regression child failed')
print('CACHE_REGRESSION_COMPLETE stock3expected_failures strict3passed',flush=True)
