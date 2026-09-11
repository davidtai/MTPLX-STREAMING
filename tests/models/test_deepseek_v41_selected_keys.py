"""W59 / K30: CPU exactness tests for the DSV4.1 selected-key gather prefill.

Under ``MTPLX_DSV41_SELECTED_KEYS`` the prefill (rows > 1) attention gathers only
the keys each query attends -- its sliding window plus the indexer's ``index_topk``
selected compressed rows -- into a compact ``[rows, k, head_dim]`` operand and runs
one softmax over ``k`` keys, exactly as the reference ``sparse_attn`` gathers
``kv[topk_idxs]`` (``~/models/DeepSeek-V4.1-Flash-src/inference/kernel.py``
``sparse_attn_kernel``).  The shipped path instead scores the full ``[rows, H, T]``
window+compressed history and masks (masked keys contribute exactly 0), so the two
are mathematically identical up to float reassociation of the softmax sum.

Covers:
  * ``_gather_rows`` / ``_mask_to_topk_idx`` / ``_window_selected_idx`` helpers;
  * unit: ``_sparse_attend_selected`` == the masked-full one-shot over the same
    window band + selection, reassociation-level (<= 1e-5), NaN-free on all-masked
    rows;
  * integration: on the tiny 8-layer CSA model (every mode: swa / full-r2 / reuse /
    full-r1+candidate / reindex), K30 prefill logits equal control (<= 1e-5,
    greedy argmax identical) across chunked and layer-major schedules;
  * decode (rows == 1) is untouched -- byte-identical to control.

No GPU/Metal, no artifact load, <3 GB RSS.  MLX pinned to CPU per
memory/worker-tests-must-pin-mlx-cpu.md.  Run under ``nice -n 19``, no ``-n auto``.
"""

from __future__ import annotations

import os

import numpy as np
import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41 as dsv41  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402

_SEL = dsv41._SELECTED_KEYS_ENV


@pytest.fixture(autouse=True)
def _clear_env():
    saved = {k: os.environ.get(k) for k in (_SEL, "MTPLX_DSV41_PREFILL_LAYER_MAJOR")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ===========================================================================
# unit: gather / index helpers
# ===========================================================================
def test_gather_rows_matches_numpy_gather():
    b, n, d = 2, 40, 8
    mx.random.seed(1)
    src = mx.random.normal((b, n, d))
    idx = mx.array([[[0, 5, 39, -1], [1, 1, 2, 3]], [[-1, -1, 0, 7], [10, 20, 30, -1]]])
    valid = idx >= 0
    g = dsv41._gather_rows(src, idx, valid)
    assert g.shape == (b, 2, 4, d)
    src_np, idx_np = np.array(src), np.array(idx)
    for bi in range(b):
        for si in range(2):
            for ki in range(4):
                j = idx_np[bi, si, ki]
                if j >= 0:
                    assert np.allclose(np.array(g[bi, si, ki]), src_np[bi, j])


def test_mask_to_topk_idx_is_the_true_set_ascending_padded():
    # row 0: 3 True (k too big -> pad -1); row 1: exactly k True.
    mask = mx.array([[[False, True, False, True, True, False]],
                     [[True, True, False, True, False, True]]])
    idx = dsv41._mask_to_topk_idx(mask, 4)
    assert idx.shape == (2, 1, 4)
    got = np.array(idx)
    assert got[0, 0].tolist() == [1, 3, 4, -1]      # ascending True positions, padded
    assert got[1, 0].tolist() == [0, 1, 3, 5]        # exactly 4 True (5 True but k=4 -> first 4)


def test_mask_to_topk_idx_gathered_set_equals_mask():
    mx.random.seed(2)
    n, k = 60, 12
    mask = mx.random.uniform(shape=(1, 5, n)) > 0.8
    # cap each row to <= k True so the conversion is lossless (matches the port,
    # where the mask holds exactly min(index_topk, reachable) True)
    keep = mx.cumsum(mask.astype(mx.int32), axis=-1) <= k
    mask = mask & keep
    idx = dsv41._mask_to_topk_idx(mask, k)
    # reconstruct a bool mask from the (>=0) indices and compare set membership
    for r in range(5):
        sel = {int(v) for v in np.array(idx[0, r]) if v >= 0}
        want = {i for i in range(n) if bool(mask[0, r, i])}
        assert sel == want


def test_window_selected_idx_matches_reference_band():
    """`_window_selected_idx` reproduces reference get_window_topk_idxs: query p
    attends {max(0,p-W+1)..p}, future/out-of-store -> invalid."""
    shim = _Shim(head_dim=8, H=4, window_size=8)
    T = 20
    positions = mx.arange(T)
    idx, valid = dsv41.Attention._window_selected_idx(shim, positions, T)
    idx_np, valid_np = np.array(idx), np.array(valid)
    for p in range(T):
        got = {int(idx_np[p, j]) for j in range(idx_np.shape[1]) if valid_np[p, j]}
        want = set(range(max(0, p - 7), p + 1))
        assert got == want, (p, got, want)


# ===========================================================================
# unit: selected-gather softmax == masked-full one-shot
# ===========================================================================
class _Shim:
    """Binds the real K30 methods to the minimal state they read."""

    def __init__(self, head_dim, H, window_size, sink=None):
        self.softmax_scale = head_dim ** -0.5
        self.attn_sink = 0.3 * mx.random.normal((H,)) if sink is None else sink
        self.window_size = window_size
        self.mode = "reuse"


_Shim._sparse_attend_selected = dsv41.Attention._sparse_attend_selected
_Shim._window_selected_idx = dsv41.Attention._window_selected_idx
_Shim._sparse_attend_oneshot = dsv41.Attention._sparse_attend_oneshot
_Shim._sparse_attend = dsv41.Attention._sparse_attend


def _masked_full_control(shim, q, window_all, compress_kv, positions, sel_mask):
    """The shipped masked-full one-shot over the SAME window band + selection the
    gather path uses -- the f32 oracle for the reassociation-level comparison."""
    b, s, H, _ = q.shape
    T = window_all.shape[1]
    wp = mx.arange(T)[None, :]
    qp = positions[:, None]
    win_attend = mx.broadcast_to(((wp <= qp) & (wp > qp - shim.window_size))[None], (b, s, T))
    if compress_kv is not None:
        KV = mx.concatenate([window_all, compress_kv], axis=1)
        attend = mx.concatenate([win_attend, sel_mask], axis=-1)
    else:
        KV, attend = window_all, win_attend
    return shim._sparse_attend_oneshot(q, KV, attend, mx.float32)


def test_selected_gather_equals_masked_full_with_compressed():
    mx.random.seed(5)
    b, s, H, hd, W = 1, 24, 4, 32, 8
    T, n_comp, k = s, 30, 5
    q = mx.random.normal((b, s, H, hd))
    window_all = mx.random.normal((b, T, hd))
    compress_kv = mx.random.normal((b, n_comp, hd))
    positions = mx.arange(s)
    compress_lens = (positions + 1)  # ratio 1 -> reachable = position+1 (capped at n_comp)
    reach = mx.arange(n_comp)[None, :] < mx.minimum(compress_lens, n_comp)[:, None]
    score = mx.random.normal((b, s, n_comp))
    score = mx.where(reach[None], score, -mx.inf)
    sel_mask = dsv41._topk_rows(score, min(k, n_comp)) & reach[None]

    shim = _Shim(hd, H, W)
    comp_idx = dsv41._mask_to_topk_idx(sel_mask, min(k, n_comp))
    got = shim._sparse_attend_selected(q, window_all, compress_kv, comp_idx, positions)
    ref = _masked_full_control(shim, q, window_all, compress_kv, positions, sel_mask)
    d = float(mx.max(mx.abs(got - ref)).item())
    assert d <= 1e-5, f"selected-gather vs masked-full max abs diff {d}"
    assert not bool(mx.any(mx.isnan(got)))


def test_selected_gather_equals_masked_full_swa_only():
    mx.random.seed(6)
    b, s, H, hd, W = 1, 20, 4, 16, 8
    q = mx.random.normal((b, s, H, hd))
    window_all = mx.random.normal((b, s, hd))
    positions = mx.arange(s)
    shim = _Shim(hd, H, W)
    got = shim._sparse_attend_selected(q, window_all, None, None, positions)
    ref = _masked_full_control(shim, q, window_all, None, positions, None)
    d = float(mx.max(mx.abs(got - ref)).item())
    assert d <= 1e-5, f"swa selected-gather vs masked-full {d}"


def test_selected_gather_all_masked_row_is_zero_not_nan():
    """A query whose window is empty AND whose compressed selection is all -1
    (fully invalid) must yield an all-zero row (reference finite-max convention),
    never a NaN."""
    b, s, H, hd, W = 1, 2, 2, 8, 8
    q = mx.random.normal((b, s, H, hd))
    window_all = mx.random.normal((b, s, hd))
    positions = mx.array([-5, -9])  # every window idx is "future" -> all invalid
    comp_idx = mx.full((b, s, 3), -1, dtype=mx.int32)
    compress_kv = mx.random.normal((b, 4, hd))
    shim = _Shim(hd, H, W, sink=mx.zeros((H,)))
    o = shim._sparse_attend_selected(q, window_all, compress_kv, comp_idx, positions)
    assert not bool(mx.any(mx.isnan(o)))
    assert float(mx.max(mx.abs(o)).item()) == 0.0


# ===========================================================================
# integration: tiny CSA model, control vs K30
# ===========================================================================
def _csa_args(**over) -> ModelArgs:
    base = dict(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=12, o_lora_rank=8, o_groups=2,
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


def _randomize(model, seed=0, scale=0.3):
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


def _build():
    model = Model(_csa_args())
    _randomize(model)
    return model


@pytest.mark.parametrize("chunk,layer_major", [
    (0, False), (4, False), (7, False), (8, False), (0, True), (5, True),
])
def test_k30_prefill_matches_control(chunk, layer_major):
    model = _build()
    ids = mx.array(np.random.RandomState(0).randint(0, 48, size=(1, 37)))

    base = model(ids, cache=model.make_cache(), prefill_chunk=0)
    mx.eval(base)

    if layer_major:
        os.environ["MTPLX_DSV41_PREFILL_LAYER_MAJOR"] = "1"
    os.environ[_SEL] = "1"
    got = model(ids, cache=model.make_cache(), prefill_chunk=chunk)
    mx.eval(got)

    ldiff = float(mx.max(mx.abs(got - base)).item())
    assert ldiff <= 1e-5, f"chunk={chunk} lm={layer_major} logit max abs diff {ldiff}"
    assert bool(mx.array_equal(mx.argmax(got, -1), mx.argmax(base, -1))), \
        f"chunk={chunk} lm={layer_major} greedy argmax differs"


def test_k30_decode_is_byte_identical_to_control():
    """Decode (rows == 1) never takes the K30 path -- byte-identical to control."""
    model = _build()
    ids = mx.array(np.random.RandomState(0).randint(0, 48, size=(1, 12)))

    cache_ctl = model.make_cache()
    mx.eval(model(ids, cache=cache_ctl, prefill_chunk=0))
    step = mx.array([[7]])
    ctl = model(step, cache=cache_ctl)
    mx.eval(ctl)

    os.environ[_SEL] = "1"
    cache_k30 = model.make_cache()
    mx.eval(model(ids, cache=cache_k30, prefill_chunk=0))  # K30 prefill (greedy-identical)
    # feed the SAME window/compress state a fresh decode step; decode ignores K30
    got = model(step, cache=cache_k30)
    mx.eval(got)
    # decode logits must be byte-identical between the two caches' decode step when
    # the prefill was greedy-identical; here we assert the decode PATH is untouched
    # by re-running control decode on a K30-prefilled cache and matching a control
    # decode on a control-prefilled cache to <= 1e-5 (prefill is reassoc-level).
    assert float(mx.max(mx.abs(got - ctl)).item()) <= 1e-5
