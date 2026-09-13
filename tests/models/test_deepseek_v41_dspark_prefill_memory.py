"""Guarded, bounded MLX checks: <64 MiB of prompt arrays, no model weights."""
import gc
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.models import deepseek_v41_dspark_decode as decode
from mtplx.models.deepseek_v41_dspark import DSparkStageCache
from mtplx.sampling import SamplerConfig


@pytest.mark.parametrize("lane", ["direct", "served", "custom_forward"])
def test_prefill_releases_full_buffers_before_decode(monkeypatch, lane):
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    gc.collect()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    rows, width, vocab = 2048, 4096, 2048
    calls, memory = [], {}

    class Model:
        def __init__(self):
            self.mtp = SimpleNamespace(block_size=5, seed_main=self.seed_main)

        def make_cache(self):
            return []

        def make_mtp_cache(self):
            return [DSparkStageCache(8, 16) for _ in range(3)]

        def __call__(self, ids, *, cache, return_hidden, logits_keep=None):
            n = ids.shape[1]
            calls.append((n, logits_keep))
            logits = mx.arange((logits_keep or n) * vocab, dtype=mx.float32)
            hidden = mx.arange(n * width, dtype=mx.float32)
            return logits.reshape(1, -1, vocab), hidden.reshape(1, n, width)

        def seed_main(self, hidden, caches):
            for cache in caches:
                cache.append_main(hidden[:, :, :16] * 1.0)

    model = Model()

    def cycles(**kwargs):
        mx.eval(kwargs["main_h"], [c.window for c in kwargs["mtp_caches"]])
        mx.synchronize()
        gc.collect()
        memory["decode_active_bytes"] = mx.get_active_memory() - baseline
        print(f"{lane}: {memory}")
        # A slice can retain its full parent allocation even after Python locals
        # are deleted. Check allocator bytes, not just tensor shape/refcounts.
        assert memory["decode_active_bytes"] < 1024**2
        main_h = kwargs["main_h"]
        assert main_h.shape == (1, 1, width)
        assert main_h[0, 0, 0].item() == (rows - 1) * width
        assert main_h[0, 0, -1].item() == rows * width - 1
        for cache in kwargs["mtp_caches"]:
            assert cache.offset == rows
            assert cache.window.shape == (1, 8, 16)
            assert cache.window[0, 0, 0].item() == (rows - 8) * width
        # Verify must still evaluate every candidate row.
        logits, _ = kwargs["forward"](mx.array([[1, 2, 3]]), kwargs["cache"])
        assert logits.shape == (1, 3, vocab)
        return [7], "length"

    def prefill(_):
        memory["prefill_active_bytes"] = mx.get_active_memory() - baseline

    monkeypatch.setattr(decode, "_decode_cycles", cycles)
    kwargs = dict(max_tokens=2, sampler=SamplerConfig(temperature=0.0),
                  prefill_callback=prefill)
    try:
        if lane == "served":
            rt = SimpleNamespace(model=model, mtp_enabled=True,
                                 forward_ar=model, tokenizer=SimpleNamespace(decode=str))
            result = decode.generate_dspark(rt, [1] * rows, stop_token_ids=set(), **kwargs)
            tokens = result.tokens
        else:
            if lane == "custom_forward":
                kwargs["forward"] = lambda ids, cache: model(
                    ids, cache=cache, return_hidden=True)
            tokens = decode.dspark_generate(model, [1] * rows, **kwargs)
        assert tokens == [vocab - 1, 7]
        assert calls == [(rows, None if lane == "custom_forward" else 1), (3, None)]
    finally:
        mx.set_default_device(previous)
