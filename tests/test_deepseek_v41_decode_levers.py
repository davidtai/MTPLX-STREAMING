"""Byte-identity + behaviour tests for the W24 DSV4.1 decode-streaming levers.

Every lever added by W24 (IO fanout depth, cache eviction policy/scope, read
chunk/coalescing, ...) is gated behind an explicit switch that defaults OFF and
must NOT change the bytes the runtime returns -- it only changes WHEN/HOW records
are read or WHICH records are cached.  These tests run on a TINY synthetic expert
bank (no real artifact, no GPU) and assert that, for a fixed routing sequence, the
gathered expert outputs are bitwise identical with the lever ON vs OFF, and that
the lever actually changed the IO/cache behaviour it claims to.

MLX is pinned to the CPU device at import (worker-tests-must-pin-mlx-cpu.md: a
"no GPU" instruction is not enough -- MLX defaults to Metal).
"""

from __future__ import annotations

import mlx.core as mx

mx.set_default_device(mx.cpu)


def test_placeholder_skeleton():
    # Real tests land with the lever implementations (next commit).
    assert True
