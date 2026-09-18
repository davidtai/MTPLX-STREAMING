"""Read-only mincore inventory of Qwen cold-session blobs; no payload reads."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import time

spec = importlib.util.spec_from_file_location('reclaimer', 'scripts/deepseek_v41/reclaim_file_cache.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
lib = helper._libc()
page = os.sysconf('SC_PAGE_SIZE')
root = Path('/Users/davidtai/.mtplx/session-bank')
start = time.monotonic()
report = {'scope': 'read-only page residency; no payload reads, invalidations or database writes',
          'root': str(root), 'files': 0, 'file_bytes': 0, 'cached_bytes': 0,
          'cached_files': [], 'vanished': 0, 'complete': False}
for shard in sorted((root/'blobs').iterdir()):
    if shard.name == '.DS_Store':
        continue
    if not re.fullmatch('[0-9a-f]{2}', shard.name) or shard.is_symlink() or not shard.is_dir():
        raise RuntimeError('unexpected blob shard layout')
    for entry in os.scandir(shard):
        if not re.fullmatch(shard.name+'[0-9a-f]{62}\\.bin', entry.name):
            continue
        info = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError('non-owned or non-regular blob')
        if report['files'] >= 100000 or report['file_bytes'] + info.st_size > 128*1024**3:
            raise RuntimeError('bounded file inventory exceeded')
        try:
            fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            report['vanished'] += 1
            continue
        cached = 0
        try:
            size = os.fstat(fd).st_size
            for offset in range(0, size, helper.CHUNK_BYTES):
                length = min(helper.CHUNK_BYTES, size-offset)
                address = lib.mmap(None, length, 1, 1, fd, offset)
                if address == ctypes.c_void_p(-1).value:
                    raise OSError(ctypes.get_errno(), 'read-only mmap')
                try:
                    flags = (ctypes.c_ubyte * ((length+page-1)//page))()
                    helper.check(lib.mincore(address, length, flags), 'read-only mincore')
                    cached += sum(bool(value & 1) for value in flags) * page
                finally:
                    helper.check(lib.munmap(address, length), 'munmap')
        finally:
            os.close(fd)
        report['files'] += 1
        report['file_bytes'] += size
        report['cached_bytes'] += cached
        if cached:
            report['cached_files'].append({'path': entry.path, 'bytes': size, 'cached_bytes': cached})
    if report['files'] // 10000 > (report.get('progress_at', 0) // 10000):
        report['progress_at'] = report['files']
        print(json.dumps({k: report[k] for k in ('files','file_bytes','cached_bytes')}), flush=True)
report['complete'] = True
report['elapsed_s'] = time.monotonic() - start
Path('/tmp/dsv41-session-cache-20260917/inventory.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'cached_files'}), flush=True)
