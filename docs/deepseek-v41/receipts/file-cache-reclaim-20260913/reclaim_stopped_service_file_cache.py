"""Inspect/reclaim clean cached pages of the exact stopped Qwen service artifact, no MLX.

Uses read-only descriptors and PROT_READ mappings. Files are never truncated or
written. One 1 GiB virtual mapping and a <=64 KiB mincore vector exist at a time.
The guard owns the service transition and exclusive GPU lane throughout.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import stat
import time

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

ROOT = Path('/Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed')
CHUNK = 1024**3
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

def identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)

def inspect_file(path, invalidate):
    root = ROOT.resolve()
    if path.is_symlink() or path.resolve().parent not in (root, root / 'engram'):
        raise ValueError(f'file is outside the explicit artifact scope: {path}')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        original = os.fstat(fd)
        if not stat.S_ISREG(original.st_mode) or original.st_uid != os.getuid():
            raise ValueError('expected a regular current-user-owned artifact file')
        size = original.st_size
        page_size = os.sysconf('SC_PAGE_SIZE')
        before = after = 0
        for offset in range(0, size, CHUNK):
            length = min(CHUNK, size - offset)
            address = libc.mmap(None, length, 1, 1, fd, offset)  # PROT_READ, MAP_SHARED
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), 'mmap failed')
            try:
                pages = (length + page_size - 1) // page_size
                vector = (ctypes.c_ubyte * pages)()
                check(libc.mincore(address, length, vector), 'mincore before')
                count = sum(bool(value & 1) for value in vector)
                before += count * page_size
                if invalidate and count:
                    if identity(os.fstat(fd)) != identity(original):
                        raise RuntimeError('artifact changed during cache inspection')
                    check(libc.msync(address, length, 0x10 | 0x2), 'msync invalidate')
                    check(libc.mincore(address, length, vector), 'mincore after')
                after += sum(bool(value & 1) for value in vector) * page_size
            finally:
                check(libc.munmap(address, length), 'munmap')
        if identity(os.fstat(fd)) != identity(original) or identity(path.stat()) != identity(original):
            raise RuntimeError('artifact identity/size/mtime changed during inspection')
        return {'file': str(path.relative_to(ROOT)), 'file_bytes': size,
                'cached_page_bytes_before': before, 'cached_page_bytes_after': after,
                'identity_size_mtime_unchanged': True}
    finally:
        os.close(fd)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--invalidate', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(180)
    # Create the report before doing anything to cached pages. Never allow this
    # diagnostic's only write to overwrite an artifact or an alias to one.
    root = ROOT.resolve()
    output = args.out.resolve()
    if root == output or root in output.parents:
        raise ValueError('report must be outside the artifact directory')
    provenance = json.loads((ROOT / '.mtplx-source.json').read_text())
    if provenance['resolved_sha'] != '29ba90f82124961d0d902a9ea9bbb1034972af2f':
        raise ValueError('unexpected service artifact revision')
    paths = [*sorted(ROOT.glob('model-*.safetensors')),
             ROOT / 'mtp.safetensors', ROOT / 'ngram-table.safetensors']
    report_fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(report_fd, 'w', encoding='utf-8') as report:
        result = {'root': str(ROOT), 'invalidate': args.invalidate,
                  'semantics': 'mincore counts cached physical pages, including speculative pages; only vm_stat used_bytes measures physical-used reduction',
                  'before': host_memory_snapshot(), 'files': []}
        started = time.monotonic()
        try:
            for path in paths:
                row = inspect_file(path, args.invalidate)
                result['files'].append(row)
                if row['cached_page_bytes_before']:
                    print(json.dumps(row), flush=True)
        finally:
            result['after'] = host_memory_snapshot()
            result['elapsed_s'] = time.monotonic() - started
            report.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'physical_used_before': result['before']['box']['used_bytes'],
                      'physical_used_after': result['after']['box']['used_bytes'],
                      'cached_before': sum(r['cached_page_bytes_before'] for r in result['files']),
                      'cached_after': sum(r['cached_page_bytes_after'] for r in result['files']),
                      'elapsed_s': result['elapsed_s']}), flush=True)

if __name__ == '__main__':
    main()
