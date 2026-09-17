"""CPU-only candidate: same VM counters as vm_stat without a child process."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from mtplx import deepseek_v41_memory_profile as original


class VMStats(C.Structure):
    # HOST_VM_INFO64_REV1_COUNT, through compressor counters. Newer SDKs append
    # fields; this prefix is sufficient and reports the returned word count.
    _fields_ = [
        ('free_count', C.c_uint32), ('active_count', C.c_uint32),
        ('inactive_count', C.c_uint32), ('wire_count', C.c_uint32),
        ('zero_fill_count', C.c_uint64), ('reactivations', C.c_uint64),
        ('pageins', C.c_uint64), ('pageouts', C.c_uint64),
        ('faults', C.c_uint64), ('cow_faults', C.c_uint64),
        ('lookups', C.c_uint64), ('hits', C.c_uint64), ('purges', C.c_uint64),
        ('purgeable_count', C.c_uint32), ('speculative_count', C.c_uint32),
        ('decompressions', C.c_uint64), ('compressions', C.c_uint64),
        ('swapins', C.c_uint64), ('swapouts', C.c_uint64),
        ('compressor_page_count', C.c_uint32), ('throttled_count', C.c_uint32),
        ('external_page_count', C.c_uint32), ('internal_page_count', C.c_uint32),
        ('total_uncompressed_pages_in_compressor', C.c_uint64),
    ]


lib = C.CDLL('/usr/lib/libSystem.B.dylib')
host_self = lib.mach_host_self
host_self.argtypes = []
host_self.restype = C.c_uint32
task_self = lib.mach_task_self
task_self.argtypes = []
task_self.restype = C.c_uint32
read = lib.host_statistics64
read.argtypes = [C.c_uint32, C.c_int, C.POINTER(VMStats), C.POINTER(C.c_uint32)]
read.restype = C.c_int
release = lib.mach_port_deallocate
release.argtypes = [C.c_uint32, C.c_uint32]
release.restype = C.c_int
page_size = os.sysconf('SC_PAGE_SIZE')


def native():
    host = host_self()
    if host in (0, 0xffffffff):
        raise RuntimeError('invalid host port')
    try:
        s = VMStats()
        count = C.c_uint32(C.sizeof(s) // 4)
        kr = read(host, 4, C.byref(s), C.byref(count))
        if kr or count.value < C.sizeof(s) // 4:
            raise RuntimeError(f'host_statistics64 failed: {kr}, words={count.value}')
        return {
            'page_size': page_size,
            'free_bytes': (s.free_count - s.speculative_count) * page_size,
            'wired_bytes': s.wire_count * page_size,
            'active_bytes': s.active_count * page_size,
            'inactive_bytes': s.inactive_count * page_size,
            'speculative_bytes': s.speculative_count * page_size,
            'anonymous_bytes': s.internal_page_count * page_size,
            'file_backed_bytes': s.external_page_count * page_size,
            'compressor_bytes': s.compressor_page_count * page_size,
            'compressed_bytes': s.total_uncompressed_pages_in_compressor * page_size,
            'used_bytes': (s.wire_count+s.active_count+s.inactive_count+s.compressor_page_count) * page_size,
            'non_file_used_bytes': (s.wire_count+s.internal_page_count+s.compressor_page_count) * page_size,
            'swapins_pages': s.swapins, 'swapouts_pages': s.swapouts,
            'ok': True, 'source': 'host_statistics64', 'used_includes_file_cache': True,
        }
    finally:
        kr = release(task_self(), host)
        if kr:
            raise RuntimeError(f'host port release failed: {kr}')


if __name__ == '__main__':
    comparisons = []
    for _ in range(5):
        before = native()
        text = original.box_memory_snapshot()
        after = native()
        comparisons.append(dict(native_before=before, vm_stat=text, native_after=after))
    timing = {}
    for label, fn in (('vm_stat', original.box_memory_snapshot), ('native', native), ('vm_stat_after', original.box_memory_snapshot)):
        samples = []
        for _ in range(30):
            t = time.perf_counter_ns()
            snap = fn()
            samples.append(time.perf_counter_ns() - t)
            assert snap['ok']
        timing[label] = dict(median_ns=statistics.median(samples), samples_ns=samples)
    report = dict(scope='OS memory sampling cost only; no model speedup claim',
        structure_bytes=C.sizeof(VMStats), word_count=C.sizeof(VMStats)//4,
        prototype_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparisons=comparisons, timing=timing)
    Path('/tmp/dsv41-online-cache-20260917/mach-box-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['median_ns'] for k,v in timing.items()}))
    print('first paired difference', {k:comparisons[0]['vm_stat'][k]-comparisons[0]['native_before'][k] for k in comparisons[0]['native_before'] if k.endswith('_bytes')})
