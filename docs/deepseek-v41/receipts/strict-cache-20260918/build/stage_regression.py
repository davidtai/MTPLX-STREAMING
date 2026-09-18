import ast,difflib,hashlib,json,shutil
from pathlib import Path
r=Path('/tmp/dsv41-strict-cache-20260918');root=r/'regression';root.mkdir()
p=r/'mlx-0.32.2/python/tests/test_memory.py';original=p.read_text()
(r/'test_memory.stock.py').write_text(original)
s=original.replace('import unittest','import gc\nimport unittest')
insert='''    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_cache_limit_releases_oversized_free(self):
        limit = 8 * 1024**2
        previous = mx.set_cache_limit(limit)
        self.addCleanup(mx.set_cache_limit, previous)
        self.addCleanup(mx.clear_cache)
        mx.clear_cache()
        value = mx.zeros((2 * limit,), dtype=mx.uint8)
        mx.eval(value)
        del value
        gc.collect()
        mx.synchronize()
        self.assertLessEqual(mx.get_cache_memory(), limit)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_cache_limit_bounds_multiple_frees(self):
        limit = 8 * 1024**2
        previous = mx.set_cache_limit(limit)
        self.addCleanup(mx.set_cache_limit, previous)
        self.addCleanup(mx.clear_cache)
        mx.clear_cache()
        first = mx.zeros((6 * 1024**2,), dtype=mx.uint8)
        second = mx.zeros((4 * 1024**2,), dtype=mx.uint8)
        mx.eval(first, second)
        mx.synchronize()
        mx.clear_cache()
        del first
        gc.collect()
        mx.synchronize()
        self.assertGreaterEqual(mx.get_cache_memory(), 6 * 1024**2)
        del second
        gc.collect()
        mx.synchronize()
        self.assertLessEqual(mx.get_cache_memory(), limit)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_lower_cache_limit_trims_immediately(self):
        previous = mx.set_cache_limit(32 * 1024**2)
        self.addCleanup(mx.set_cache_limit, previous)
        self.addCleanup(mx.clear_cache)
        mx.clear_cache()
        value = mx.zeros((16 * 1024**2,), dtype=mx.uint8)
        mx.eval(value)
        del value
        gc.collect()
        mx.synchronize()
        self.assertGreaterEqual(mx.get_cache_memory(), 16 * 1024**2)
        self.assertEqual(mx.set_cache_limit(8 * 1024**2), 32 * 1024**2)
        self.assertLessEqual(mx.get_cache_memory(), 8 * 1024**2)

'''
s=s.replace('    def test_memory_info(self):',insert+'    def test_memory_info(self):')
ast.parse(s);p.write_text(s)
(r/'test-memory.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/python/tests/test_memory.py',tofile='b/python/tests/test_memory.py')))
shutil.copy2(r/'library_identity.py',root/'library_identity.py')
(root/'run_checks.py').write_text('''import hashlib,json,os,sys,unittest
from pathlib import Path
if os.environ.get('_GPU_WINDOW_LOCKED')!='1':raise RuntimeError('parent GPU guard required')
root=Path(__file__).resolve().parent
proof=json.loads((root/'installation.json').read_text())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
box=host_memory_snapshot()['box']
if not box['ok'] or box['used_bytes']+proof['incremental_bound_bytes']>110000000000:raise RuntimeError('regression bound does not fit')
source=Path(proof['test_source'])
if hashlib.sha256(source.read_bytes()).hexdigest()!=proof['test_sha256']:raise RuntimeError('regression source changed')
import mlx.core as mx
from library_identity import identify
binary=os.environ['DSV41_OPERATOR_BINARY']
identity=identify(proof['binary_choices'][binary])
mx.set_default_device(mx.gpu)
mx.set_memory_limit(512*1024**2)
sys.path.insert(0,str(source.parent))
import test_memory
suite=unittest.TestSuite(test_memory.TestMemory(name) for name in proof['tests'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
record={'binary':binary,'library':identity,'tests_run':result.testsRun,'failures':[(t.id(),tb) for t,tb in result.failures],'errors':[(t.id(),tb) for t,tb in result.errors],'skipped':result.skipped,'mlx_peak_bytes':int(mx.get_peak_memory())}
mx.synchronize();mx.clear_cache();record['active_after_close_bytes']=int(mx.get_active_memory())
(root/(binary+'.json')).write_text(json.dumps(record,indent=2)+'\\n')
expected=3 if binary=='stock_source' else 0
if result.testsRun!=3 or result.errors or result.skipped or len(result.failures)!=expected:raise RuntimeError('cache regression outcome differs')
''')
(root/'run_pair.py').write_text('''import hashlib,json,os,signal,subprocess,sys
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
''')
for f in root.glob('*.py'):ast.parse(f.read_text())
proof={'incremental_bound_bytes':2*1024**3,'bound_scope':'512MiB allocator envelope including at most32MiB free cache and16MiB arrays;1.5GiB host/import/shader/compiler reserve. One child at a time.','test_source':str(p),'test_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tests':['test_cache_limit_releases_oversized_free','test_cache_limit_bounds_multiple_frees','test_lower_cache_limit_trims_immediately'],'binary_choices':json.loads((r/'expert-screen/installation.json').read_text())['binary_choices'],'helper_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}}
(root/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
(root/'command.sh').write_text('env GPU_WINDOW_LOCK_TIMEOUT=120 GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000 GPU_WINDOW_MIN_AVAIL_GB=4 GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 PYTHONUNBUFFERED=1 PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41:'+str(root)+' scripts/deepseek_v41/gpu_window.sh /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python '+str(root/'run_pair.py')+' > '+str(root/'guard.log')+' 2>&1\n')
print('STAGED3_TARGETED_CACHE_REGRESSIONS_AFTER_FULL_WIN')
