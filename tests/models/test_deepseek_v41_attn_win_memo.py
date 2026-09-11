"""W45 / kernel-ledger K24 -- sliding-window attend-mask memoization (CPU, synthetic).

``MTPLX_DSV41_ATTN_WIN_MEMO`` computes the causal sliding-window attend mask
``(wp <= qp) & (wp > qp - window_size)`` -> ``[b, s, T]`` ONCE per forward and
reuses the identical array across the backbone's other layers, instead of every
layer rebuilding it.  The mask is a pure function of ``(positions, window length
T, window_size)``, all invariant across the layers of one ``_forward_span`` (the
census found it the largest remaining mode-invariant per-layer dispatch chunk
after K22).  Reuse fires only when ``positions`` is the *same object* and ``(T,
window_size, b, s)`` match, so it is byte-identical (the reused array is the same
object) and degrades to per-layer recompute where they differ -- never wrong.

Gates (all pin MLX to CPU; tiny random config; no artifact load):
  * flag on vs off ``mx.array_equal`` over decode (n=1), a K+1 verify batch, and
    chunked + layer-major prefill -- both with and without the K22 attention-tape
    compile (the memo composes with K22 and is independent of it);
  * the dedup actually fires and REDUCES the per-token attention primitive count
    (asserted from the W41/W45 census tool: memo-on attention primitives/token <
    memo-off), while the stages K24 does not touch are unchanged;
  * the memo is per-forward: a fresh forward (new ``shared`` runtime) recomputes;
  * env default OFF.

Uses ``q_lora_rank=16`` for the same compile-stable-reduction reason as the K22
suite (the memo itself is exact at any dim; this only matters when composed with
the K22 tapes).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
from pathlib import Path

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402

_TEST_CAP = 7


def _csa_args(**over) -> ModelArgs:
    base = dict(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5,
        sliding_window=8, window_size=8, swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )
    base.update(over)
    return ModelArgs(**base)


def _randomize(model, seed=0, scale=0.1):
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = scale * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


@contextlib.contextmanager
def _memo(flag: bool, *, compile_on: bool = False):
    """Flip the K24 memo knob (and optionally the K22 compile knob) for a block."""
    om, oc = dv41._ATTN_WIN_MEMO, dv41._ATTN_COMPILE
    dv41._ATTN_WIN_MEMO = flag
    dv41._ATTN_COMPILE = compile_on
    dv41._ATTN_COMPILE_MAX_ROWS = _TEST_CAP
    dv41._ATTN_COMPILED.clear()
    try:
        yield
    finally:
        dv41._ATTN_WIN_MEMO, dv41._ATTN_COMPILE = om, oc
        dv41._ATTN_COMPILED.clear()


def _new_model(seed=1, **over):
    args = _csa_args(**over)
    model = Model(args)
    _randomize(model, seed=seed)
    return model, args


def _prefill_one_shot(model, args, s, seed):
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    cache = model.make_cache()
    mx.eval(model(ids, cache=cache, prefill_chunk=0))
    return cache


# ---------------------------------------------------------------------------
# 1. byte-identity: decode / verify / chunked / layer-major, memo composed with
#    and without the K22 tapes
# ---------------------------------------------------------------------------
def _decode(flag, compile_on, seed):
    model, args = _new_model(seed=seed)
    with _memo(flag, compile_on=compile_on):
        cache = _prefill_one_shot(model, args, s=12, seed=seed)
        outs = []
        for t in [3, 17, 5, 29]:
            lo = model(mx.array([[t]]), cache=cache)
            mx.eval(lo)
            outs.append(np.array(lo))
    return outs


def _prefill(flag, compile_on, *, chunk, layer_major, s=20, seed=4):
    model, args = _new_model(seed=seed)
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    with _memo(flag, compile_on=compile_on):
        cache = model.make_cache()
        lo = model(ids, cache=cache, prefill_chunk=chunk,
                   prefill_layer_major=(True if layer_major else None))
        mx.eval(lo)
        return np.array(lo)


def _verify(flag, compile_on, seed):
    model, args = _new_model(seed=seed)
    with _memo(flag, compile_on=compile_on):
        cache = _prefill_one_shot(model, args, s=12, seed=seed)
        logits = model(mx.array([[7, 2, 41, 13]]), cache=cache)
        mx.eval(logits)
        return np.array(logits)


import pytest  # noqa: E402


@pytest.mark.parametrize("compile_on", [False, True])
def test_decode_memo_on_off_identical(compile_on):
    off, on = _decode(False, compile_on, 1), _decode(True, compile_on, 1)
    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), f"decode {i} differs (max {np.max(np.abs(a-b))})"


@pytest.mark.parametrize("compile_on", [False, True])
def test_verify_memo_on_off_identical(compile_on):
    assert np.array_equal(_verify(False, compile_on, 2), _verify(True, compile_on, 2))


@pytest.mark.parametrize("compile_on", [False, True])
@pytest.mark.parametrize("layer_major", [False, True])
def test_prefill_memo_on_off_identical(compile_on, layer_major):
    off = _prefill(False, compile_on, chunk=5, layer_major=layer_major)
    on = _prefill(True, compile_on, chunk=5, layer_major=layer_major)
    assert np.array_equal(off, on), \
        f"prefill lm={layer_major} compile={compile_on} differs (max {np.max(np.abs(off-on))})"


def test_one_shot_memo_on_off_identical():
    model, args = _new_model(seed=5)
    ids = mx.array(np.random.RandomState(6).randint(0, args.vocab_size, size=(1, 12)))
    with _memo(False):
        off = np.array(model(ids, cache=model.make_cache(), prefill_chunk=0))
    model, _ = _new_model(seed=5)
    with _memo(True):
        on = np.array(model(ids, cache=model.make_cache(), prefill_chunk=0))
    assert np.array_equal(off, on)


# ---------------------------------------------------------------------------
# 2. the dedup fires and reduces the per-token attention primitive count
#    (asserted straight from the census tool)
# ---------------------------------------------------------------------------
def _load_census():
    path = Path(__file__).resolve().parents[2] / "scripts" / "deepseek_v41" / "dispatch_census.py"
    spec = importlib.util.spec_from_file_location("dsv41_dispatch_census_w45", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _attn_prims(report):
    return sum(v["primitives_per_token"] for k, v in report["stages"].items()
               if k.startswith("attn."))


def test_census_memo_reduces_attention_dispatches():
    census = _load_census()
    # isolate the memo: K22 compile off in both, memo off vs on.
    off = census._run_full_census(False, _TEST_CAP, seed=1, win_memo=False)
    on = census._run_full_census(False, _TEST_CAP, seed=1, win_memo=True)
    a_off, a_on = _attn_prims(off), _attn_prims(on)
    assert a_on < a_off, (a_off, a_on)
    # the 8-layer tiny model has 7 layers reusing the ~11-node window mask
    assert a_off - a_on >= 7 * 5, (a_off, a_on)  # >= 5 real dispatches x 7 layers
    # the stages K24 does not touch are unchanged.
    for untouched in ("hc.premix_sinkhorn", "hc.combine", "moe.routed_switch",
                      "moe.shared_expert", "moe.gate_topk", "moe.combine"):
        assert on["stages"][untouched]["primitives_per_token"] == \
            off["stages"][untouched]["primitives_per_token"], untouched


def test_census_memo_composes_with_k22():
    census = _load_census()
    base = census._run_full_census(False, _TEST_CAP, seed=1, win_memo=False)
    both = census._run_full_census(True, _TEST_CAP, seed=1, win_memo=True)
    assert both["total_primitives_per_token"] < base["total_primitives_per_token"]
    assert _attn_prims(both) < _attn_prims(base)


# ---------------------------------------------------------------------------
# 3. the memo is per-forward + reuse actually populated
# ---------------------------------------------------------------------------
def test_window_attend_reuses_same_object_and_recomputes_on_change():
    """Direct unit test of the memo semantics: same (positions, T, ws, b, s) ->
    the SAME array object (reuse); a different positions object or T -> a fresh
    array; and reuse is only enabled under the flag."""
    model, args = _new_model(seed=7)
    attn = model.model.layers[0].attn  # window_size = 8
    shared = model.make_cache().new_shared_runtime()
    positions = mx.array([12])

    with _memo(True):
        a1 = attn._window_attend(positions, 13, 1, 1, shared)
        a2 = attn._window_attend(positions, 13, 1, 1, shared)  # same key -> reuse
        assert a1 is a2, "memo did not reuse the identical array for a repeat key"
        a3 = attn._window_attend(mx.array([12]), 13, 1, 1, shared)  # new positions obj
        assert a3 is not a1, "memo reused across a different positions object"
        a4 = attn._window_attend(positions, 14, 1, 1, shared)  # different T
        assert a4 is not a1
    # flag off: never memoized (fresh array each call), but byte-identical values
    with _memo(False):
        b1 = attn._window_attend(positions, 13, 1, 1, shared)
        b2 = attn._window_attend(positions, 13, 1, 1, shared)
        assert b1 is not b2
        mx.eval(a1, b1)
        assert np.array_equal(np.array(a1), np.array(b1))  # memo value == eager value


def test_env_default_off():
    assert dv41._ATTN_WIN_MEMO_ENV == "MTPLX_DSV41_ATTN_WIN_MEMO"
    truthy = (os.environ.get(dv41._ATTN_WIN_MEMO_ENV) or "").strip().lower() not in (
        "", "0", "false", "no", "off", "auto")
    assert dv41._ATTN_WIN_MEMO == truthy
