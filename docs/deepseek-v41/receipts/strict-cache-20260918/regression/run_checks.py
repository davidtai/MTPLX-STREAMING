import hashlib,json,os,sys,unittest
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
(root/(binary+'.json')).write_text(json.dumps(record,indent=2)+'\n')
expected=3 if binary=='stock_source' else 0
if result.testsRun!=3 or result.errors or result.skipped or len(result.failures)!=expected:raise RuntimeError('cache regression outcome differs')
