"""Compare isolated host libraries under one parent-held GPU/service window."""
import hashlib, importlib.util, json, os, signal, statistics, subprocess, sys, time
from pathlib import Path

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent-held GPU/service guard required')
ROOT = Path(__file__).resolve().parent
proof = json.loads((ROOT / 'installation.json').read_text())
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != proof['source_commit']:
    raise RuntimeError('source pin changed')
if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True):
    raise RuntimeError('tracked repository source is dirty')
for name, digest in proof['helper_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('operator helper changed')
for name, digest in proof['runtime_source_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('runtime source changed')
if (ROOT / 'comparison.json').exists():
    raise RuntimeError('refusing completed comparison overwrite')
spec = importlib.util.spec_from_file_location('owned_reclaim', 'scripts/deepseek_v41/reclaim_file_cache.py')
reclaim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reclaim)
inventory=json.loads((ROOT/'artifact/manifest.json').read_text())
files=[Path(proof['model_path'])/'experts.bin']
files += [ROOT/'artifact'/component[field]['file']
          for component in inventory['layers'][proof['layer']]['components'].values()
          for field in ('descriptors','payload','bases')]
records = []
report = {'source_commit': proof['source_commit'], 'complete': False, 'children': records}


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    for index, binary in enumerate(('wheel', 'stock_source', 'strict', 'stock_source')):
        tag = f'{index:02d}-{binary}'
        case = ROOT / 'cases' / tag
        case.mkdir(parents=True)
        env = os.environ.copy()
        env.pop('DYLD_LIBRARY_PATH', None)
        if binary != 'wheel':
            env['DYLD_LIBRARY_PATH'] = str(Path(proof['binary_choices'][binary]['path']).parent)
        env['DSV41_OPERATOR_CASE'] = tag
        env['DSV41_OPERATOR_BINARY'] = binary
        print('OPERATOR_START', tag, flush=True)
        with (case / 'output.log').open('w') as output:
            child = subprocess.Popen([sys.executable, str(ROOT / 'probe.py')], env=env,
                                     stdout=output, stderr=subprocess.STDOUT)
            life = {'case': tag, 'binary': binary, 'pid': child.pid, 'started_at': time.time()}
            records.append(life)
            try:
                code = child.wait(timeout=240)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                life.update(returncode=child.returncode, exited_at=time.time())
                (case / 'child.json').write_text(json.dumps(life, indent=2) + '\n')
        if code != 0:
            raise RuntimeError('operator child failed: ' + tag)
        row = json.loads((case / 'probe.json').read_text())
        if not row['complete'] or row['active_after_close_bytes'] > 2*1024**2:
            raise RuntimeError('operator did not finish or release its arrays')
        life['warm_total_ns']=row['arms'][0]['warm_total_ns']
        print('OPERATOR_RESULT', json.dumps(life), flush=True)
    report['complete'] = True
finally:
    rows = [reclaim.reclaim_file(path) for path in files]
    cleanup = {'files': rows, 'cached_page_bytes_before': sum(r['cached_page_bytes_before'] for r in rows),
               'cached_page_bytes_after': sum(r['cached_page_bytes_after'] for r in rows)}
    (ROOT / 'reclamation.json').write_text(json.dumps(cleanup, indent=2) + '\n')
    (ROOT / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('OPERATOR_CLEANUP', json.dumps({k: v for k, v in cleanup.items() if k != 'files'}), flush=True)
    if cleanup['cached_page_bytes_after']:
        raise RuntimeError('source-file cache remained after child exit')
