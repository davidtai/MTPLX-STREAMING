"""W75: the DeepSeek-V4.1 chunked prefill ticks the model-owner progress
heartbeat at each settled per-chunk / per-layer eval fence.

Before W75 the chunk-major (``_eval_cache_state``) and layer-major
(``_eval_layer_transients``) loops used a bare ``mx.eval`` that never ticked the
heartbeat. A 16K prefill emits no token for ~200 s, so the heartbeat was frozen
for the whole prefill; the stream stall watchdog (#86) compares successive
heartbeat readings, and a prefill slowed past the 300 s deadline (e.g. by the
spurious CRITICAL clear_cache W75 also fixes) was killed with 0 tokens. Ticking
at each settled prefill fence makes a long-but-alive prefill a moving heartbeat.

CPU-only: MLX pinned to CPU, tiny arrays, no model load.
"""

from __future__ import annotations

import mlx.core as mx

mx.set_default_device(mx.cpu)

from types import SimpleNamespace  # noqa: E402

from mtplx import progress_heartbeat  # noqa: E402
from mtplx.models.deepseek_v41 import DeepseekV41Backbone  # noqa: E402


def test_eval_cache_state_ticks_on_a_settled_chunk():
    # A chunk-major span whose output/cache stores are forced: exactly one
    # settled forward => exactly one heartbeat tick.
    before = progress_heartbeat.value()
    fake_cache = SimpleNamespace(layers=[])  # the extra array drives the eval
    DeepseekV41Backbone._eval_cache_state(fake_cache, mx.zeros((2, 2)))
    assert progress_heartbeat.value() == before + 1


def test_eval_cache_state_no_settled_eval_no_tick():
    # Nothing to force (no output, empty cache): no eval, so no phantom tick.
    before = progress_heartbeat.value()
    fake_cache = SimpleNamespace(layers=[])
    DeepseekV41Backbone._eval_cache_state(fake_cache)
    assert progress_heartbeat.value() == before


def test_eval_layer_transients_ticks_on_a_settled_layer_chunk():
    before = progress_heartbeat.value()
    fake_lc = SimpleNamespace(
        window=None, compress_kv=None, index_k=None, comp_state=None
    )
    DeepseekV41Backbone._eval_layer_transients(fake_lc, mx.zeros((2, 2)))
    assert progress_heartbeat.value() == before + 1


def test_eval_layer_transients_no_settled_eval_no_tick():
    before = progress_heartbeat.value()
    fake_lc = SimpleNamespace(
        window=None, compress_kv=None, index_k=None, comp_state=None
    )
    DeepseekV41Backbone._eval_layer_transients(fake_lc)
    assert progress_heartbeat.value() == before


def test_many_chunks_advance_the_heartbeat_monotonically():
    # A long chunked prefill => many ticks => a watchdog that resets its frozen
    # window on every chunk, so time alone never breaches during a live prefill.
    before = progress_heartbeat.value()
    fake_cache = SimpleNamespace(layers=[])
    for _ in range(8):
        DeepseekV41Backbone._eval_cache_state(fake_cache, mx.zeros((1, 1)))
    assert progress_heartbeat.value() == before + 8
