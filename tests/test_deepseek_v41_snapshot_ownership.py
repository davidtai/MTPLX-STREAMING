"""Session snapshot ownership; AST doubles never import MLX.

The tests named ``test_real_*`` must run through the exclusive GPU guard.
"""
from __future__ import annotations

import ast
import gc
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Snapshot:
    states: tuple
    meta_states: tuple = ()


class Array:
    def __init__(self, value, dtype=None):
        self.data = np.asarray(value, dtype=dtype)
        self.evaluated = False

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    def __array__(self, dtype=None, copy=None):
        return np.array(self.data, dtype=dtype, copy=True) if copy else np.asarray(self.data, dtype=dtype)

    def __getitem__(self, index):
        return Array(self.data[index])

    def __add__(self, value):
        return Array(self.data + value.data)

    def reshape(self, *shape):
        return Array(self.data.reshape(*shape))

    def view(self, dtype):
        return Array(self.data.view(dtype))


def _scope():
    """Execute the actual admission methods with tiny ndarray-backed doubles."""
    source = ROOT / "mtplx/session_bank.py"
    parsed = ast.parse(source.read_text())
    wanted = {"_tree_nbytes", "_snapshot_nbytes", "_compact_snapshot_tree",
              "_snapshot_owner_for_runtime"}
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    bank = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "SessionBank")
    nodes.append(ast.ClassDef(name="Bank", bases=[], keywords=[], decorator_list=[],
        body=[n for n in bank.body if isinstance(n, ast.FunctionDef) and n.name in {"put", "put_snapshot"}]))
    mx = SimpleNamespace(array=Array, uint8=np.uint8, cpu="cpu",
        default_device=lambda: SimpleNamespace(type="cpu"),
        zeros=lambda shape, dtype: Array(np.zeros(shape, dtype=dtype)))
    def evaluate(*values):
        for value in values:
            if isinstance(value, Array):
                value.evaluated = True
    mx.eval = evaluate
    def clone(value):
        if isinstance(value, Array):
            return value + mx.zeros((), dtype=value.dtype)
        if isinstance(value, (list, tuple)):
            return type(value)(clone(x) for x in value)
        if isinstance(value, dict):
            return {key: clone(x) for key, x in value.items()}
        return value
    def snapshot(cache):
        return Snapshot(tuple(item.state for item in cache))
    scope = {"mx": mx, "np": np, "os": os, "time": time,
        "CacheSnapshot": Snapshot, "SessionBankEntry": SimpleNamespace,
        "_clone_tree": clone, "_is_trimmable": lambda value: True,
        "snapshot_cache_lazy_hybrid": snapshot, "snapshot_cache": snapshot,
        "_lazy_snapshot_enabled": lambda: True,
        "_validate_none_policy_auxiliary_state": lambda **kwargs: None,
        "_mtp_history_policy_is_none": lambda policy: False,
        "_canonicalize_none_policy_entry": lambda entry: False,
        "token_prefix_hash": lambda tokens: str(tokens)}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    bank = scope["Bank"]()
    bank._entries = {}
    bank.per_session_max_bytes = 1024
    bank.max_bytes = 1024
    bank.eviction_log = deque()
    bank.shed_gdn_boundaries_to_fit = False
    bank._touch_session = lambda *args: None
    bank.longest_prefix = lambda *args: None
    bank._schedule_snapshot_settle = lambda *args, **kwargs: None
    bank._enqueue_cold_entry = lambda *args, **kwargs: None
    bank._supersede_contained_prefixes = lambda *args: None
    bank._evict_if_needed = lambda **kwargs: None
    bank.warn_oversized_snapshot_skip = lambda *args, **kwargs: None
    bank._schedule_live_ref_spill = lambda *args, **kwargs: None
    return scope, bank


def _runtime(deepseek=True):
    key = "deepseek-v41-flash-expert-mxfp4" if deepseek else "hy3-expert-oq2e"
    return SimpleNamespace(model_path="/tiny", mtp_enabled=False,
        expert_streaming=SimpleNamespace(config=SimpleNamespace(model_key=key)))


@pytest.mark.parametrize("method", ["put", "put_snapshot"])
def test_deepseek_admission_retains_compact_owned_bits(method):
    _, bank = _scope()
    backing = np.arange(4096, dtype=np.uint32)
    bits = np.array([0x80000000, 0x7fc12345, 0x00000001, 0x7f800000], dtype=np.uint32)
    backing[:4] = bits
    source = Array(backing[:4].view(np.float32))
    kwargs = dict(runtime=_runtime(), token_ids=[1, 2], logits=source, hidden=source,
                  mtp_history_snapshot=Snapshot(((source,),)), keep_live_ref=True)
    if method == "put":
        kwargs.update(cache=[SimpleNamespace(state=(source,))], extra_state={"leaf": source},
                      gdn_boundaries=[(1, Snapshot(((source,),)), source)])
    else:
        kwargs["cache_snapshot"] = Snapshot(((source,),))
    entry = getattr(bank, method)(**kwargs)
    leaves = [entry.cache_snapshot.states[0][0], entry.logits, entry.hidden,
              entry.mtp_history_snapshot.states[0][0]]
    if method == "put":
        leaves.append(entry.extra_state["leaf"])
        leaves.extend([entry.gdn_boundaries[0][1].states[0][0], entry.gdn_boundaries[0][2]])
    for leaf in leaves:
        assert not np.shares_memory(leaf.data, backing), "snapshot retains oversized backing"
        assert leaf.evaluated, "snapshot copy is still a lazy graph"
        np.testing.assert_array_equal(leaf.data.view(np.uint32), bits)
    assert entry.nbytes == len(leaves) * source.nbytes
    assert entry.cache_ref is None
    assert entry.lazy_kv, "warm restore must retain the bit-preserving view route"


@pytest.mark.parametrize("method", ["put", "put_snapshot"])
def test_deepseek_rejects_underpriced_override_before_copy(method):
    scope, bank = _scope()
    bank.per_session_max_bytes = bank.max_bytes = 8
    source = Array(np.arange(16, dtype=np.uint32))
    scope["_compact_snapshot_tree"] = lambda value: pytest.fail("oversized entry was copied")
    kwargs = dict(runtime=_runtime(), token_ids=[1], logits=source, hidden=None, nbytes_override=1)
    if method == "put":
        kwargs.update(cache=[SimpleNamespace(state=(source,))], keep_live_ref=True)
    else:
        kwargs.update(cache_snapshot=Snapshot(((source,),)), keep_live_ref=True,
                      cache_ref=[SimpleNamespace(state=(source,))])
    assert getattr(bank, method)(**kwargs) is None
    assert bank._entries == {}


def test_other_models_keep_lazy_snapshots():
    _, bank = _scope()
    backing = np.arange(4096, dtype=np.uint32)
    source = Array(backing[:4])
    entry = bank.put(runtime=_runtime(False), token_ids=[1],
                     cache=[SimpleNamespace(state=(source,))], logits=source, hidden=None)
    assert np.shares_memory(entry.cache_snapshot.states[0][0].data, backing)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires guarded MLX execution")
@pytest.mark.parametrize("device_name", ["cpu", "gpu"])
def test_real_compact_copy_preserves_bits_and_releases_large_backing(device_name):
    """Run only through run_guarded.py, even for the CPU device."""
    import mlx.core as mx
    from mtplx.session_bank import _compact_snapshot_tree

    old_device = mx.default_device()
    try:
        mx.set_default_device(getattr(mx, device_name))
        bits = np.array([0x80000000, 0x7fc12345, 0x00000001, 0x7f800000], dtype=np.uint32)
        large = mx.zeros((4 * 1024 * 1024,), dtype=mx.uint32)
        large = mx.slice_update(large, mx.array(bits), mx.array([0]), axes=[0])
        mx.eval(large)
        view = large[:4].view(mx.float32)
        baseline = mx.get_active_memory()
        owned = _compact_snapshot_tree(view)
        np.testing.assert_array_equal(np.array(owned.view(mx.uint32)), bits)
        source_address = np.frombuffer(memoryview(view), dtype=np.uint8).__array_interface__["data"][0]
        owned_address = np.frombuffer(memoryview(owned), dtype=np.uint8).__array_interface__["data"][0]
        assert source_address != owned_address
        del large, view
        gc.collect()
        mx.clear_cache()
        if device_name == "gpu":
            assert mx.get_active_memory() < baseline - 8 * 1024**2
        np.testing.assert_array_equal(np.array(owned.view(mx.uint32)), bits)
        strided = mx.array(np.stack((bits, bits), axis=1)).view(mx.float32)[:, 0]
        compact = _compact_snapshot_tree(strided)
        assert compact.shape == strided.shape
        np.testing.assert_array_equal(np.array(compact.view(mx.uint32)), bits)
    finally:
        mx.set_default_device(old_device)
