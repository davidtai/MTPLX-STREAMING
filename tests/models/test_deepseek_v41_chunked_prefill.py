"""W20: token-chunked prefill exactness + transient-bound tests (CPU, synthetic).

Skeleton committed first (loss-recovery rule). Filled in incrementally.
All tests pin MLX to CPU and use tiny random configs / synthetic switches;
no real-artifact load.
"""
from __future__ import annotations

import mlx.core as mx

mx.set_default_device(mx.cpu)


def test_placeholder_skeleton():
    # Replaced by the real chunked-prefill exactness + transient-bound tests.
    assert True
