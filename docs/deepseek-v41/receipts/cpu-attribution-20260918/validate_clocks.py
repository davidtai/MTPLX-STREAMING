"""Check main/worker clock attribution and its small diagnostic overhead."""
import importlib.abc,json,sys,threading,time
from pathlib import Path

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('clock validation cannot import MLX')
sys.meta_path.insert(0,NoMLX())
from clock_probe import Span

def spin(duration):
    start=time.thread_time_ns()
    while time.thread_time_ns()-start<duration:
        pass

main=Span();started=main.start();spin(30000000);main.finish(started)
worker=Span();started=worker.start()
t=threading.Thread(target=spin,args=(60000000,));t.start();t.join(timeout=2)
assert not t.is_alive()
worker.finish(started)
assert main.thread_ns>=30000000
assert worker.thread_ns<10000000 and worker.process_ns-worker.thread_ns>=50000000
empty=Span();started=time.perf_counter_ns()
for _ in range(20000):begin=empty.start();empty.finish(begin)
cost=(time.perf_counter_ns()-started)/20000
assert cost<100000
result={'cpu_only':True,'python_executable':sys.executable,'python_version':sys.version,'main_spin':main.snapshot(),'worker_spin_and_main_join':worker.snapshot(),
        'empty_span_mean_ns':cost,'all_clock_attribution_checks_pass':True,
        'expected_full_span_count':206*(2*40+6)+1,
        'estimated_empty_instrumentation_s':cost*(206*(2*40+6)+1)/1e9}
Path(__file__).with_name('clock-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
