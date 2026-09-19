"""Construction-time identity and inactive-cache policy observations."""
import ctypes
import gc
import hashlib
from pathlib import Path


def identify(expected):
    process = ctypes.CDLL(None)
    count = process._dyld_image_count
    count.restype = ctypes.c_uint32
    name = process._dyld_get_image_name
    name.argtypes = [ctypes.c_uint32]
    name.restype = ctypes.c_char_p
    paths = [Path(name(i).decode()).resolve() for i in range(count())]
    libraries = [p for p in paths if p.name == 'libmlx.dylib']
    if len(libraries) != 1 or libraries[0] != Path(expected['path']).resolve():
        raise RuntimeError('unexpected MLX host library: ' + str(libraries))
    digest = hashlib.sha256(libraries[0].read_bytes()).hexdigest()
    if digest != expected['sha256']:
        raise RuntimeError('loaded MLX host library hash changed')
    return {'path': str(libraries[0]), 'sha256': digest}


def observe_cache_policy(mx, *, strict):
    mib = 1024**2
    mx.synchronize()
    mx.clear_cache()
    mx.set_cache_limit(8*mib)
    value = mx.zeros((16*mib,), dtype=mx.uint8)
    mx.eval(value)
    del value
    gc.collect()
    mx.synchronize()
    oversized_free = int(mx.get_cache_memory())
    mx.clear_cache()
    mx.set_cache_limit(32*mib)
    value = mx.zeros((16*mib,), dtype=mx.uint8)
    mx.eval(value)
    del value
    gc.collect()
    mx.synchronize()
    before_lowering = int(mx.get_cache_memory())
    previous = int(mx.set_cache_limit(8*mib))
    after_lowering = int(mx.get_cache_memory())
    result = {'limit_bytes': 8*mib, 'freed_buffer_bytes': 16*mib,
              'cached_after_oversized_free_bytes': oversized_free,
              'cached_before_lowering_limit_bytes': before_lowering,
              'cached_after_lowering_limit_bytes': after_lowering,
              'previous_limit_bytes': previous}
    mx.clear_cache()
    mx.set_cache_limit(256*mib)
    if strict and (oversized_free > 8*mib or after_lowering > 8*mib):
        raise RuntimeError('strict allocator violated configured inactive-cache capacity')
    return result
