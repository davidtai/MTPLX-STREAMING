"""Build matched stock/strict host libraries while the service guard owns the lane."""
import hashlib, json, os, shutil, signal, subprocess, time
from pathlib import Path

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('guard must own the lane before configure or compilation')
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'mlx-0.32.2'
LIB = ROOT / 'strict-lib'
CONTROL = ROOT / 'control-lib'
design = json.loads((ROOT / 'design.json').read_text())
proof = json.loads((ROOT / 'build-installation.json').read_text())
if (ROOT / 'build-result.json').exists():
    raise RuntimeError('refusing to overwrite completed build evidence')
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != proof['source_commit']:
    raise RuntimeError('repository source pin changed')
for name, digest in proof['helper_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('build input changed: ' + name)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
if (not before['box']['ok'] or before['box']['used_bytes'] + design['incremental_build_budget_bytes'] > 110000000000):
    raise RuntimeError('compiler envelope does not fit the current baseline')
if shutil.disk_usage(ROOT).free < 2 * design['disk_budget_bytes']:
    raise RuntimeError('insufficient disk headroom for isolated build')
LIB.mkdir(exist_ok=True)
CONTROL.mkdir(exist_ok=True)
allocator = SRC / 'mlx/backend/metal/allocator.cpp'
strict = allocator.read_bytes()
child = None
report = {'source_commit': proof['source_commit'], 'before': before,
          'started_at': time.time(), 'commands': [], 'complete': False}


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def run(argv):
    global child
    print('BUILD_COMMAND', json.dumps(argv), flush=True)
    start = time.time()
    child = subprocess.Popen(argv, start_new_session=True)
    try:
        code = child.wait(timeout=600)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    report['commands'].append({'argv': argv, 'returncode': code,
                               'wall_s': time.time() - start})
    if code:
        raise RuntimeError('build command failed')
    if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) > design['disk_budget_bytes']:
        raise RuntimeError('isolated build exceeded its disk budget')


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    # Only the allocator object differs between these two builds.
    allocator.write_bytes((ROOT / 'allocator.stock.cpp').read_bytes())
    run(proof['configure_argv'])
    run(proof['build_argv'])
    for name in ('libmlx.dylib', 'libjaccl.dylib', 'mlx.metallib'):
        if (LIB / name).exists():
            shutil.copy2(LIB / name, CONTROL / name)
    allocator.write_bytes(strict)
    run(proof['build_argv'])
    report['libraries'] = {}
    for mode, directory in [('stock_source', CONTROL), ('strict', LIB)]:
        report['libraries'][mode] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in directory.iterdir() if p.is_file()}
        if report['libraries'][mode].get('mlx.metallib') != design['stock_metallib_sha256']:
            raise RuntimeError('Metal shader bytes changed')
    report['complete'] = True
finally:
    allocator.write_bytes(strict)
    report['finished_at'] = time.time()
    report['after'] = host_memory_snapshot()
    (ROOT / 'build-result.json').write_text(json.dumps(report, indent=2) + '\n')
print('BUILD_COMPLETE', json.dumps(report['libraries']), flush=True)
