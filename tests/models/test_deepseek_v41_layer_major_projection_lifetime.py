"""Guarded tiny native-quantized model checks for prefill projection ownership."""
import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41 as dsv41

_path = Path(__file__).with_name("test_deepseek_v41_layer_major_prefill.py")
_spec = importlib.util.spec_from_file_location("layer_major_fixtures", _path)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


@pytest.mark.parametrize("rows,chunk,release", [(25, 7, True), (6, 2, False)])
def test_projection_lifetime_and_state_across_requests(monkeypatch, rows, chunk, release):
    monkeypatch.setenv("MTPLX_DSV41_ATTN_FUSED_PROJ", "0")
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", False)
    args = _fixture._csa_args(dspark_target_layer_ids=[5, 6, 7])
    model = dsv41.Model(args)
    _fixture._randomize(model, seed=71)
    for layer in model.model.layers:
        layer.attn.wo_a = nn.QuantizedLinear.from_linear(
            layer.attn.wo_a, group_size=32, bits=8, mode="mxfp8")
    mx.eval(model.parameters())
    original = dsv41.Attention._o_lora_dense_weight
    observed = {"active": False, "previous": None, "ids": {}}

    def weight(attn):
        if not observed["active"]:
            return original(attn)
        previous = observed["previous"]
        if release and previous is not None and previous is not attn:
            assert getattr(previous, "_wo_a_dense_cache", None) is None, (
                "completed prefill layer still owns its fp32 projection cache")
        w = original(attn)
        ids = observed["ids"].setdefault(attn.layer_id, [])
        ids.append(id(w))
        assert len(set(ids)) == 1, "all chunks in a layer must reuse the same weight"
        observed["previous"] = attn
        return w

    monkeypatch.setattr(dsv41.Attention, "_o_lora_dense_weight", weight)
    for request in range(2):
        ids = mx.array((np.arange(rows)[None, :] + request) % args.vocab_size)
        ref_cache, cache = model.make_cache(), model.make_cache()
        monkeypatch.setenv("MTPLX_DSV41_ATTN_WO_A_CACHE", "0")
        reference = model(ids, cache=ref_cache, prefill_chunk=chunk,
                          prefill_layer_major=True, return_hidden=True)
        mx.eval(reference)
        observed.update(active=True, previous=None, ids={})
        monkeypatch.setenv("MTPLX_DSV41_ATTN_WO_A_CACHE", "1")
        candidate = model(ids, cache=cache, prefill_chunk=chunk,
                          prefill_layer_major=True, return_hidden=True)
        mx.eval(candidate)
        observed["active"] = False
        for a, b in zip(reference, candidate):
            assert a is not None and bool(mx.array_equal(a, b).item())
        assert len(observed["ids"]) == args.num_hidden_layers
        assert all(len(v) == (rows + chunk - 1) // chunk for v in observed["ids"].values())
        saved = [getattr(layer.attn, "_wo_a_dense_cache", None)
                 for layer in model.model.layers]
        assert all((entry is None) == release for entry in saved)
        ref_state, ref_offset = _fixture._cache_snapshot(ref_cache)
        state, offset = _fixture._cache_snapshot(cache)
        assert ref_offset == offset == rows and ref_state.keys() == state.keys()
        for key in state:
            np.testing.assert_array_equal(ref_state[key], state[key])

        token = mx.array([[3]])
        monkeypatch.setenv("MTPLX_DSV41_ATTN_WO_A_CACHE", "0")
        reference = model(token, cache=ref_cache)
        mx.eval(reference)
        monkeypatch.setenv("MTPLX_DSV41_ATTN_WO_A_CACHE", "1")
        candidate = model(token, cache=cache)
        mx.eval(candidate)
        assert bool(mx.array_equal(reference, candidate).item())
        for layer, old in zip(model.model.layers, saved):
            cached = layer.attn._wo_a_dense_cache
            assert cached is not None and layer.attn._o_lora_dense_weight() is cached[3]
            if not release:
                assert cached is old, "small verification must retain its cache"
