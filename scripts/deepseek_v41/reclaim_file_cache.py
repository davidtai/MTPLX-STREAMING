"""Discard clean model-file cache after the guard has stopped its service.

No MLX, payload reads, file writes, or privilege escalation. Darwin mappings are
read-only; one 1 GiB virtual window and <=64 KiB page flags exist at a time.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

CHUNK_BYTES = 1024**3


def model_files(root: Path, *, max_files: int = 512) -> list[Path]:
    root = root.resolve(strict=True)
    if not root.is_dir() or root.stat().st_uid != os.getuid():
        raise ValueError('expected a current-user-owned model directory')
    for name in ('config.json', 'model.safetensors.index.json'):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'model directory is missing regular {name}')
    paths = []
    for path in root.glob('*.safetensors'):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError(f'expected a regular current-user-owned model file: {path}')
        paths.append(path)
        if len(paths) > max_files:
            raise ValueError('model file count exceeds bounded reclamation scope')
    if not paths:
        raise ValueError('no model safetensors files found')
    return sorted(paths)


def receipt_safetensor_files(root: Path, *, max_files: int = 16) -> list[Path]:
    """Resolve a small current-user-owned safetensor set from ``receipt.json``.

    Benchmark-owned compact resident banks live outside the model artifact.  Their
    receipt binds each file by absolute path and size, which lets the GPU guard
    invalidate stale clean pages without accepting an arbitrary directory scan.
    """

    root = root.resolve(strict=True)
    root_info = root.stat()
    if not root.is_dir() or root_info.st_uid != os.getuid():
        raise ValueError('expected a current-user-owned receipt directory')
    receipt_path = root / 'receipt.json'
    receipt_info = receipt_path.lstat()
    if (
        receipt_path.is_symlink()
        or not stat.S_ISREG(receipt_info.st_mode)
        or receipt_info.st_uid != os.getuid()
    ):
        raise ValueError('receipt directory is missing regular current-user-owned receipt.json')
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('receipt.json is unreadable or malformed') from exc
    records = receipt.get('files') if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get('schema') != 1
        or not isinstance(records, list)
    ):
        raise ValueError('receipt.json has an unsupported schema')
    if not 0 < len(records) <= max_files:
        raise ValueError('receipt file count is outside bounded reclamation scope')

    paths: list[Path] = []
    seen: set[Path] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('receipt file entry must be an object')
        try:
            declared = Path(record['path'])
            declared_size = int(record['file_bytes'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('receipt file entry is malformed') from exc
        if not declared.is_absolute() or declared.suffix != '.safetensors':
            raise ValueError('receipt must name absolute safetensors paths')
        if declared.parent.resolve(strict=True) != root:
            raise ValueError(f'receipt file escapes its directory: {declared}')
        info = declared.lstat()
        if (
            declared.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_size != declared_size
        ):
            raise ValueError(f'receipt file identity changed: {declared}')
        path = declared.resolve(strict=True)
        if path in seen:
            raise ValueError(f'receipt file is duplicated: {declared}')
        seen.add(path)
        paths.append(path)
    return sorted(paths)


def check(rc: int, operation: str) -> None:
    if rc != 0:
        error = ctypes.get_errno()
        raise OSError(error, f'{operation}: {os.strerror(error)}')


def _libc():
    if sys.platform != 'darwin':
        raise RuntimeError('file cache reclamation requires Darwin')
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
    return libc


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def reclaim_file(path: Path, *, chunk_bytes: int = CHUNK_BYTES) -> dict:
    page_size = os.sysconf('SC_PAGE_SIZE')
    if not 0 < chunk_bytes <= CHUNK_BYTES or chunk_bytes % page_size:
        raise ValueError('mapping window must be page aligned and at most 1 GiB')
    libc = _libc()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        original = os.fstat(fd)
        if not stat.S_ISREG(original.st_mode) or original.st_uid != os.getuid():
            raise ValueError('expected a regular current-user-owned model file')
        before = after = 0
        for offset in range(0, original.st_size, chunk_bytes):
            length = min(chunk_bytes, original.st_size - offset)
            address = libc.mmap(None, length, 1, 1, fd, offset)  # PROT_READ, MAP_SHARED
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), 'mmap failed')
            try:
                vector = (ctypes.c_ubyte * ((length + page_size - 1) // page_size))()
                check(libc.mincore(address, length, vector), 'mincore before')
                count = sum(bool(value & 1) for value in vector)
                before += count * page_size
                if count:
                    if _identity(os.fstat(fd)) != _identity(original):
                        raise RuntimeError('model file changed during reclamation')
                    check(libc.msync(address, length, 0x10 | 0x2), 'msync invalidate')
                    check(libc.mincore(address, length, vector), 'mincore after')
                after += sum(bool(value & 1) for value in vector) * page_size
            finally:
                check(libc.munmap(address, length), 'munmap')
        if (_identity(os.fstat(fd)) != _identity(original)
                or _identity(path.stat()) != _identity(original)):
            raise RuntimeError('model file identity/size/mtime changed during reclamation')
        return {'file': path.name, 'file_bytes': original.st_size,
                'cached_page_bytes_before': before, 'cached_page_bytes_after': after,
                'identity_size_mtime_unchanged': True}
    finally:
        os.close(fd)


def reclaim_model(root: Path) -> list[dict]:
    # Validate the complete file set before the first invalidation.
    paths = model_files(root)
    return [reclaim_file(path) for path in paths]


def reclaim_receipt_safetensors(root: Path) -> list[dict]:
    # Validate the complete receipt-bound set before the first invalidation.
    paths = receipt_safetensor_files(root)
    return [reclaim_file(path) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--operation',
        choices=(
            'stopped_service_file_cache_reclamation',
            'candidate_file_cache_reclamation',
            'candidate_aux_file_cache_reclamation',
        ),
        default='stopped_service_file_cache_reclamation',
    )
    parser.add_argument(
        '--layout',
        choices=('model', 'receipt-safetensors'),
        default='model',
    )
    parser.add_argument('model_dir', type=Path)
    args = parser.parse_args()
    signal.alarm(30)
    # Direct script execution also works without a caller-provided PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

    before = host_memory_snapshot()
    started = time.monotonic()
    rows = (
        reclaim_model(args.model_dir)
        if args.layout == 'model'
        else reclaim_receipt_safetensors(args.model_dir)
    )
    after = host_memory_snapshot()
    print(json.dumps({
        'operation': args.operation,
        'layout': args.layout,
        'model_dir': str(args.model_dir.resolve()), 'files': rows,
        'before': before, 'after': after, 'elapsed_s': time.monotonic() - started,
        'physical_used_reduction_bytes': before['box']['used_bytes'] - after['box']['used_bytes'],
        'cached_page_bytes_before': sum(row['cached_page_bytes_before'] for row in rows),
        'cached_page_bytes_after': sum(row['cached_page_bytes_after'] for row in rows),
        'semantics': 'cached pages include speculative free pages; physical-used reduction is measured separately',
    }), flush=True)


if __name__ == '__main__':
    main()
