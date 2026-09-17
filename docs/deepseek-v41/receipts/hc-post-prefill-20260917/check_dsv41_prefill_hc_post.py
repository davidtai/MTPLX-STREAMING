import os, signal
if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('exclusive guard required before MLX import')
signal.alarm(120)
import mlx.core as mx
mx.set_memory_limit(512 * 1024**2)
mx.set_cache_limit(64 * 1024**2)
import pytest
raise SystemExit(pytest.main(['-q', 'tests/models/test_deepseek_v41_layer_major_prefill.py::test_layer_major_is_byte_identical_to_chunk_major']))
