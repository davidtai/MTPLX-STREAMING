"""CPU proof of scope attribution, generator boundaries and measurement cost."""
import hashlib
import json
from pathlib import Path
import threading
import time
from boundary_timer import BoundaryTimer

root = Path(__file__).resolve().parent
timer = BoundaryTimer()
start = threading.Event()
stop = threading.Event()


def worker():
    start.wait()
    while not stop.is_set():
        timer.call('foreign_thread', time.sleep, .001)


def source():
    time.sleep(.005)
    yield 'a'
    time.sleep(.005)
    yield 'b'


def main():
    start.set()
    values = []
    for value in timer.wrap_iterator('next_miss', source)():
        values.append(value)
        timer.call('compute_between_yields', time.sleep, .01)
    try:
        timer.call('throws', lambda: 1/0)
    except ZeroDivisionError:
        pass
    stop.set()
    return values


thread = threading.Thread(target=worker)
thread.start()
assert timer.call('decode', main) == ['a', 'b']
thread.join()
proof = timer.snapshot()
by_name = {r['name']:r for r in proof['rows']}
assert 'foreign_thread' not in by_name
assert .007 < by_name['next_miss']['inclusive_s'] < by_name['compute_between_yields']['inclusive_s']
assert by_name['next_miss']['observations'] == 3
assert by_name['throws']['observations'] == 1

noop = lambda: None
reps = 100000
t = time.perf_counter_ns()
for _ in range(reps):
    noop()
bare = time.perf_counter_ns() - t
probe = BoundaryTimer()
t = time.perf_counter_ns()
for _ in range(reps):
    probe.call('noop', noop)
wrapped = time.perf_counter_ns() - t
overhead = (wrapped-bare)/reps
assert overhead < 10000
proof.update(scope='CPU-only explicit-boundary validation; no model or MLX import',
             foreign_thread_excluded=True, iterator_consumer_time_excluded=True,
             exceptions_balance_scopes=True, overhead_ns_per_flat_call=overhead,
             timer_sha256=hashlib.sha256((root/'boundary_timer.py').read_bytes()).hexdigest())
(root/'cpu-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:proof[k] for k in ('root_wall_ns','exclusive_sum_ns',
    'foreign_thread_excluded','iterator_consumer_time_excluded','overhead_ns_per_flat_call')}))
