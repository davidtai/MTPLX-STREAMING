"""CPU-only process-footprint reader screen; leaves whole-machine vm_stat alone."""
import ctypes as ct
import json
from pathlib import Path
import statistics
import time
from mtplx import deepseek_v41_memory_profile as original

info_type = original._TaskVMInfo.struct()
words = ct.sizeof(info_type) // 4
lib = ct.CDLL('/usr/lib/libSystem.B.dylib')
task_self = lib.mach_task_self
task_self.argtypes, task_self.restype = [], ct.c_uint32
task_info = lib.task_info
task_info.argtypes = [ct.c_uint32, ct.c_int, ct.POINTER(info_type), ct.POINTER(ct.c_uint32)]
task_info.restype = ct.c_int


def fast():
    info = info_type()
    count = ct.c_uint32(words)
    kr = task_info(task_self(), 22, ct.byref(info), ct.byref(count))
    if kr != 0 or count.value < words:
        return None
    return {'resident_bytes': int(info.resident_size),
            'phys_footprint_bytes': int(info.phys_footprint),
            'compressed_bytes': int(info.compressed)}


if __name__ == '__main__':
    timing = {}
    for label, reader in (('control_before', original._mach_task_vm_info),
                          ('candidate', fast),
                          ('control_after', original._mach_task_vm_info)):
        samples = []
        for _ in range(100):
            started = time.perf_counter_ns()
            value = reader()
            samples.append(time.perf_counter_ns() - started)
            assert value and value['phys_footprint_bytes'] > 0
        timing[label] = {'median_ns': statistics.median(samples), 'samples_ns': samples}
    before = fast()
    payload = bytearray(16 * 1024**2)
    after = fast()
    control_after = original._mach_task_vm_info()
    report = dict(scope='Own-process TASK_VM_INFO binding-cache CPU screen only',
        timing=timing, allocation_bytes=len(payload), before=before, after=after,
        control_after=control_after,
        footprint_delta_bytes=after['phys_footprint_bytes']-before['phys_footprint_bytes'])
    Path('/tmp/dsv41-online-cache-20260917/task-reader-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['median_ns'] for k,v in timing.items()}))
    print('immediate_footprint_growth', report['footprint_delta_bytes'])
