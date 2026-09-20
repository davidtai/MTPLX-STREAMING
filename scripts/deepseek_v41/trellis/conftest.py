"""Test bootstrap for the F34 trellis package: put the module dir on sys.path and pin MLX to CPU.

'no GPU' means CPU: MLX defaults to Metal, so force the CPU device before any array op.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
except Exception:
    pass
