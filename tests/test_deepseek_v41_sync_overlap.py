"""W28 — host-sync census + byte-identity for the DSV4.1 shared-overlap lever.

SKELETON. CPU-pinned, tiny synthetic streamed switch + fake bank (reuses the
``_BankOverlap*`` doubles from tests/test_streamed_models.py). No GPU, no
artifact. Fills in:

  1. host-sync census per streamed layer (mx.eval / mx.async_eval / .tolist),
     control (shared serialized after routed) vs overlap (shared hoisted);
  2. byte-identity of routed + shared + combined MoE output, switch off vs on.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def test_skeleton_placeholder():
    assert True
