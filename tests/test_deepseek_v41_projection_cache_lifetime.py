"""CPU ownership/accounting checks; real-shape MLX evidence lives in receipts."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace


class Packed(dict):
    def __init__(self):
        super().__init__(weight=np.arange(24).reshape(4, 6),
                         scales=np.array(1.), biases=np.array(0.))
        self.group_size, self.bits, self.mode = 32, 8, "affine"

    def __getattr__(self, name):
        return self[name]


@pytest.fixture
def attention():
    flags = SimpleNamespace(cache=True)
    methods = definitions("mtplx/models/deepseek_v41.py",
                          {"_o_lora_dense_weight", "_o_lora_fused_weight"}, {
        "nn": SimpleNamespace(QuantizedLinear=Packed),
        "mx": SimpleNamespace(dequantize=lambda w, s, b, **kw: (w * s + b).astype(np.float16),
                              float32=np.float32, bfloat16=np.float16,
                              eval=lambda *a: None, swapaxes=np.swapaxes,
                              contiguous=np.ascontiguousarray),
        "_resolve_wo_a_cache": lambda: flags.cache,
    }, class_name="Attention")
    cls = type("Attention", (), {n: methods[n] for n in ("_o_lora_dense_weight", "_o_lora_fused_weight")})
    obj = cls()
    obj.wo_a, obj.n_groups, obj.o_lora_rank = Packed(), 2, 2
    return obj, flags


@pytest.mark.parametrize("cache", [False, True])
def test_repeated_prefill_decode_owns_only_current_representation(attention, cache):
    obj, flags = attention
    flags.cache = cache
    previous = None
    for _ in range(3):
        dense = obj._o_lora_dense_weight()
        if previous is not None:
            assert previous() is None, "prefill retained the previous decode representation"
        if cache:
            assert obj._o_lora_dense_weight() is dense
        expected = dense.astype(np.float16).swapaxes(1, 2).copy()
        previous = weakref.ref(dense)
        del dense
        fused = obj._o_lora_fused_weight()
        assert previous() is None, "decode retained the previous prefill representation"
        assert obj._o_lora_fused_weight() is fused
        np.testing.assert_array_equal(fused, expected)
        previous = weakref.ref(fused)
        del fused


@pytest.mark.parametrize("member", ["weight", "scales", "biases"])
def test_fused_cache_reloads_each_packed_input(attention, member):
    obj, _ = attention
    original = obj._o_lora_fused_weight()
    obj.wo_a[member] = obj.wo_a[member] + 1
    rebuilt = obj._o_lora_fused_weight()
    assert rebuilt is not original
    expected = (obj.wo_a.weight * obj.wo_a.scales + obj.wo_a.biases).astype(np.float16)
    np.testing.assert_array_equal(rebuilt, expected.reshape(2, 2, 6).swapaxes(1, 2))


@pytest.mark.parametrize("cache,fused", [(False, False), (False, True), (True, False), (True, True)])
def test_plan_prices_largest_target_representation_and_native_mtp(monkeypatch, cache, fused):
    monkeypatch.setitem(sys.modules, "mtplx.models.deepseek_v41", SimpleNamespace(
        _resolve_wo_a_cache=lambda: cache, _resolve_attn_fused_proj=lambda: fused))
    ns = definitions("mtplx/models/deepseek_v41_loader.py",
                     {"deepseek_v41_additional_resident_bytes"}, {
        "__package__": "mtplx.models", "NUM_TEXT_LAYERS": 40,
        "WO_A_DENSE_F32_BYTES": 134217728, "WO_A_DENSE_BF16_BYTES": 67108864,
        "SWA_WINDOW_BYTES": 5242880, "SLIDING_WINDOW": 128, "KV_LATENT_DIM": 512,
    })
    target = 40 * (134217728 if cache else 67108864 if fused else 0)
    stage = 134217728 if cache else 0
    assert ns["deepseek_v41_additional_resident_bytes"](mtp_layers=3) == (
        5242880 + 3 * 128 * 512 * 4 + target + 3 * stage)
