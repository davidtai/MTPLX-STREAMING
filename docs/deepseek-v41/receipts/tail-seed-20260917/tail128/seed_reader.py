"""Read only the twelve admitted seed tensors into their final MLX owners."""
from contextlib import ExitStack
import fcntl
import os
import stat

from mtplx.expert_manifest import _pread_exact, _read_safetensors_header, _readonly_flags


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def load_seed_tensors(root, manifest, kept, *, mx):
    if len(kept) != 12 or sum(t.length for t in kept) != 89224192:
        raise RuntimeError('seed tensor inventory differs from the bounded native layout')
    shards = {s.name: s for s in manifest.shards}
    grouped = {}
    for tensor in kept:
        grouped.setdefault(tensor.shard, []).append(tensor)
    if set(grouped) != {'model-00044.safetensors', 'model-00045.safetensors', 'model-00046.safetensors'}:
        raise RuntimeError('native seed shard inventory changed')
    handles = {}
    originals = {}
    # Validate all source headers/ranges before any tensor allocation or read.
    with ExitStack() as stack:
        for name, tensors in grouped.items():
            path = root / name
            fd = os.open(path, _readonly_flags())
            stack.callback(os.close, fd)
            original = os.fstat(fd)
            if not stat.S_ISREG(original.st_mode) or original.st_size > 305000000:
                raise RuntimeError('seed source shard exceeds its admitted size')
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            fcntl.fcntl(fd, 45, 0)
            if int.from_bytes(_pread_exact(fd, 0, 8), 'little') > 1024**2:
                raise RuntimeError('seed source header exceeds1MiB')
            current, actual = _read_safetensors_header(path, relative_name=name, fd=fd)
            expected = shards[name]
            if any(getattr(current, k) != getattr(expected, k) for k in ('size', 'header_bytes', 'header_sha256')):
                raise RuntimeError('seed source header differs from the native manifest')
            by_name = {t.name: t for t in actual}
            for tensor in tensors:
                value = by_name.get(tensor.tensor)
                if value is None or any(getattr(value, k) != getattr(tensor, k) for k in ('offset', 'length', 'dtype', 'shape')):
                    raise RuntimeError('seed source tensor metadata changed')
            handles[name] = fd
            originals[name] = identity(original)
        selected = {}
        for tensor in kept:
            dtype = {'U32': mx.uint32, 'U8': mx.uint8, 'BF16': mx.bfloat16}[tensor.dtype]
            value = mx.zeros(tensor.shape, dtype=dtype)
            mx.eval(value)
            raw = memoryview(value).cast('B')
            try:
                if raw.readonly or not raw.c_contiguous or raw.nbytes != tensor.length:
                    raise RuntimeError('seed output buffer does not match its admitted tensor')
                done = 0
                while done < raw.nbytes:
                    with raw[done:min(done + 8 * 1024**2, raw.nbytes)] as part:
                        count = os.preadv(handles[tensor.shard], [part], tensor.offset + done)
                    if count <= 0:
                        raise RuntimeError('short native seed tensor read')
                    done += count
            finally:
                raw.release()
            selected[tensor.tensor] = value
        for name, fd in handles.items():
            if identity(os.fstat(fd)) != originals[name] or identity((root / name).stat()) != originals[name]:
                raise RuntimeError('seed source identity changed during loading')
    return selected
