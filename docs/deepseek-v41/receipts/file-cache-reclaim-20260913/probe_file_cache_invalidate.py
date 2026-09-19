"""Test read-only mapped cache invalidation on a new 64 MiB scratch file only."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.mmap.restype = ctypes.c_void_p
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_ubyte)]
libc.mincore.restype = ctypes.c_int
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.msync.restype = ctypes.c_int
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

def check(rc, operation):
    if rc != 0:
        error = ctypes.get_errno()
        raise OSError(error, f'{operation}: {os.strerror(error)}')

size = 64 * 1024**2
page_size = os.sysconf('SC_PAGE_SIZE')
payload = os.urandom(1024**2)
expected = hashlib.sha256(payload * 64).hexdigest()
with tempfile.TemporaryDirectory(prefix='dsv41-cache-probe-') as temp:
    path = Path(temp) / 'owned-scratch.bin'
    with path.open('wb', buffering=0) as handle:
        for _ in range(64):
            if handle.write(payload) != len(payload):
                raise OSError('short scratch write')
        os.fsync(handle.fileno())
    fd = os.open(path, os.O_RDONLY)
    address = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
    if address == ctypes.c_void_p(-1).value:
        os.close(fd)
        raise OSError(ctypes.get_errno(), 'mmap failed')
    try:
        def residency():
            vec = (ctypes.c_ubyte * (size // page_size))()
            check(libc.mincore(address, size, vec), 'mincore')
            return sum(bool(value & 1) for value in vec) * page_size

        before = residency()
        os_before = host_memory_snapshot()
        start = time.monotonic()
        check(libc.msync(address, size, 0x10 | 0x2), 'msync')
        elapsed = time.monotonic() - start
        after = residency()
        os_after = host_memory_snapshot()
        digest = hashlib.sha256()
        for offset in range(0, size, len(payload)):
            block = os.pread(fd, len(payload), offset)
            if len(block) != len(payload):
                raise OSError('short verification read')
            digest.update(block)
        assert digest.hexdigest() == expected
        result = {'scope': 'new owned64 MiB scratch file; read-only mapping, no model/artifact',
                  'bytes': size, 'resident_before_bytes': before,
                  'resident_after_bytes': after, 'content_sha256_unchanged': True,
                  'msync_seconds': elapsed, 'os_before': os_before, 'os_after': os_after}
        Path('/tmp/dsv41-110-preflight/file-cache-invalidate-probe.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k:v for k,v in result.items() if not k.startswith('os_')}, indent=2))
    finally:
        try:
            check(libc.munmap(address, size), 'munmap')
        finally:
            os.close(fd)
