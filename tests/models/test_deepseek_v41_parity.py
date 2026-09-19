"""P1.1-P1.3 parity + P1.3 rollback for the DeepSeek-V4.1 text forward.

torch is not importable in this environment, so the oracle is a float64 numpy
transcription of ``inference/model.py`` (the DeepSeek MIT reference), written here
independently of the MLX port and reading the port's own randomised weights so
only the arithmetic is compared.  The tests assert next-token argmax parity (the
spec bar) plus a tight numeric tolerance on the sub-module outputs.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

# DeepSeek-V4.1 parity is device-dependent: MLX's CPU and GPU matmul use different
# accumulation precision (see _mm), so the oracle must match the device the tests
# run on. Force the CPU stream at import (like the engram tests) so the suite is
# device-independent whether these tests run alone or alongside the engram tests.
mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten

from mtplx.models.deepseek_v41 import (
    DeepseekV41Cache,
    Model,
    ModelArgs,
    _LayerCache,
    _SharedRuntime,
)

# ---------------------------------------------------------------------------
# numpy oracle (float64), transcribed from inference/model.py
# ---------------------------------------------------------------------------
def _np(a) -> np.ndarray:
    return np.array(a, dtype=np.float64)


def _mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """fp32-accumulate matmul, matching MLX's CPU GEMM.

    Direct micro-test (2026-09-10, mlx 0.32.2, this box): ``mx.matmul`` on the CPU
    stream is *bit-identical* to numpy float32-accumulate (max abs diff 0.0 over
    shapes up to K=6144), and its error vs a float64 reference is pure fp32 rounding
    (~7e-5 rel at K=32, ~9e-4 at K=5120). The CPU backend rounds operands to fp32 and
    accumulates in fp32 — it does **not** round to tf32. (The tf32/10-bit-mantissa
    rounding W1 described is MLX's *GPU/Metal* matmul; that is why the parity tests
    passed only when the default device was the GPU. On CPU the tf32 oracle injected
    ~1e-2 of spurious error, failing swa/block/csa2 by a device artifact, not a port
    bug.) These tests force ``mx.set_default_device(mx.cpu)``, so the oracle
    accumulates matmuls in fp32 too; the result is widened to float64 so the
    surrounding elementwise math stays a high-precision reference.
    """
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float64)


def _einsum(subscripts: str, *ops) -> np.ndarray:
    """fp32-accumulate einsum contraction (MLX lowers these to the CPU GEMM path)."""
    return np.einsum(
        subscripts, *[o.astype(np.float32) for o in ops]
    ).astype(np.float64)


def _rmsnorm(x, w, eps):
    x = x.astype(np.float64)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return w.astype(np.float64) * (x / np.sqrt(var + eps))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _lin(x, w):  # nn.Linear: y = x @ w.T, w is [out, in]
    return _mm(x, w.T)


def _inv_freq_swa(rd, theta):
    return 1.0 / (theta ** (np.arange(0, rd, 2) / rd))


def _inv_freq_yarn(rd, theta, orig, factor, bf, bs):
    freqs = 1.0 / (theta ** (np.arange(0, rd, 2) / rd))
    if orig and orig > 0:
        def cd(nr):
            return rd * np.log(orig / (nr * 2 * np.pi)) / (2 * np.log(theta))

        low = max(np.floor(cd(bf)), 0)
        high = min(np.ceil(cd(bs)), rd - 1)
        ramp = np.clip((np.arange(rd // 2) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def _cos_sin(inv_freq, positions):
    ang = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    return np.cos(ang), np.sin(ang)


def _rope(x, cos, sin, inverse=False):
    rd = cos.shape[-1] * 2
    head, tail = x[..., :-rd], x[..., -rd:]
    shp = tail.shape
    t = tail.reshape(*shp[:-1], shp[-1] // 2, 2)
    x0, x1 = t[..., 0], t[..., 1]
    extra = tail.ndim - 3
    c = cos.reshape((cos.shape[0],) + (1,) * extra + (cos.shape[-1],))
    s = sin.reshape((cos.shape[0],) + (1,) * extra + (cos.shape[-1],))
    if inverse:
        s = -s
    r0 = x0 * c - x1 * s
    r1 = x0 * s + x1 * c
    out = np.stack([r0, r1], axis=-1).reshape(shp)
    return out if head.shape[-1] == 0 else np.concatenate([head, out], axis=-1)


def _topk_rows(score, k):
    n = score.shape[-1]
    if k >= n:
        return score > -np.inf
    ranked = np.sort(score, axis=-1)[..., ::-1]
    thr = ranked[..., k - 1:k]
    gt = score > thr
    eq = score == thr
    n_gt = np.sum(gt, axis=-1, keepdims=True)
    tie_rank = np.cumsum(eq.astype(np.int64), axis=-1) - 1
    return gt | (eq & (tie_rank < (k - n_gt)))


def _select_candidates(logits, compress_lens, topk_blocks, block_size):
    b, s, width = logits.shape
    pad = (-width) % block_size
    if pad:
        logits = np.concatenate(
            [logits, np.full((b, s, pad), -np.inf)], axis=-1
        )
    num_blocks = logits.shape[-1] // block_size
    scores = np.max(logits.reshape(b, s, num_blocks, block_size), axis=-1)
    last = (compress_lens - 1) // block_size  # [s]
    pin = np.arange(num_blocks)[None, None, :] == last[None, :, None]
    scores = np.where(pin, np.inf, scores)
    kb = min(topk_blocks, num_blocks)
    keep = _topk_rows(scores, kb) & (scores > -np.inf)
    return np.repeat(keep, block_size, axis=-1)[..., :width]


class _RefAttn:
    """Pull one Attention module's weights + geometry into numpy."""

    def __init__(self, attn, args):
        self.a = attn
        self.args = args
        self.H = attn.n_heads
        self.hd = attn.head_dim
        self.rd = attn.rope_head_dim
        self.ng = attn.n_groups
        self.olr = attn.o_lora_rank
        self.win = attn.window_size
        self.eps = attn.eps
        self.ratio = attn.compress_ratio
        self.scale = attn.softmax_scale
        self.inv_freq = _np(attn.inv_freq)
        self.wq_a = _np(attn.wq_a.weight)
        self.q_norm = _np(attn.q_norm_weight)
        self.wq_b = _np(attn.wq_b.weight)
        self.wkv = _np(attn.wkv.weight)
        self.kv_norm = _np(attn.kv_norm_weight)
        self.wo_a = _np(attn.wo_a.weight)
        self.wo_b = _np(attn.wo_b.weight)
        self.sink = _np(attn.attn_sink)

    def qr(self, x):
        return _rmsnorm(_lin(x, self.wq_a), self.q_norm, self.eps)

    def window_kv(self, x, cos, sin):
        kv = _rmsnorm(_lin(x, self.wkv), self.kv_norm, self.eps)
        return _rope(kv, cos, sin)

    def attend(self, q, KV, mask):
        b, s, H, _ = q.shape
        scores = _einsum("bshd,btd->bsht", q, KV) * self.scale
        scores = np.where(mask[:, :, None, :], scores, -np.inf)
        sink = np.broadcast_to(self.sink.reshape(1, 1, H, 1), (b, s, H, 1))
        full = np.concatenate([scores, sink], axis=-1)
        w = _softmax(full, axis=-1)[..., : KV.shape[1]]
        return _einsum("bsht,btd->bshd", w, KV)

    def out(self, x, KV, mask, positions, cos, sin):
        b, s, _ = x.shape
        q = _lin(self.qr(x), self.wq_b).reshape(b, s, self.H, self.hd)
        q = _rope(q, cos, sin)
        o = self.attend(q, KV, mask)
        o = _rope(o, cos, sin, inverse=True)
        o = o.reshape(b, s, self.ng, -1)
        wo_a = self.wo_a.reshape(self.ng, self.olr, -1)
        o = _einsum("bsgd,grd->bsgr", o, wo_a)
        return _lin(o.reshape(b, s, -1), self.wo_b)


def _ref_indexer(attn, args, x, qr, index_k, positions, cos, sin,
                 compress_lens, n_comp, candidates, set_c):
    idx = attn.indexer
    H = idx.n_heads
    D = idx.index_head_dim
    wq_b = _np(idx.wq_b.weight)
    wproj = _np(idx.weights_proj.weight)
    b, s, _ = x.shape
    q = _lin(qr, wq_b).reshape(b, s, H, D)
    q = _rope(q, cos, sin)
    weights = _lin(x, wproj) * (idx.softmax_scale * H ** -0.5)
    score = _einsum("bshd,btd->bsht", q, index_k)
    score = np.maximum(score, 0.0) * weights[..., None]
    score = np.sum(score, axis=2)  # [b,s,n_comp]
    reach = np.arange(n_comp)[None, None, :] < compress_lens[None, :, None]
    score = np.where(reach, score, -np.inf)
    cand_out = None
    if set_c:
        cand_out = _select_candidates(
            score, compress_lens, args.candidate_topk_blocks, args.candidate_block_size
        )
    elif candidates is not None:
        score = np.where(candidates, score, -np.inf)
    topk = min(idx.index_topk, n_comp)
    mask = _topk_rows(score, topk) & reach
    return mask, cand_out


def _ref_index_keys(attn, latent_pre, cos, sin, eps):
    idx = attn.indexer
    wk = _np(idx.wk.weight)
    k_norm = _np(idx.k_norm_weight)
    k = _rmsnorm(_lin(latent_pre, wk), k_norm, eps)
    return _rope(k, cos, sin)


def _ref_compress_prefill(attn, x, eps):
    comp = attn.compressor
    ratio = comp.ratio
    norm_w = _np(comp.norm_weight)
    wkv = _np(comp.wkv.weight)
    if ratio == 1:
        return _rmsnorm(_lin(x, wkv), norm_w, eps)
    b, s, _ = x.shape
    wgate = _np(comp.wgate.weight)
    kv = _lin(x, wkv)
    score = _lin(x, wgate)
    cutoff = s - (s % ratio)
    kv = kv[:, :cutoff].reshape(b, -1, ratio, kv.shape[-1])
    score = score[:, :cutoff].reshape(b, -1, ratio, score.shape[-1])
    pooled = np.sum(kv * _softmax(score, axis=2), axis=2)
    return _rmsnorm(pooled, norm_w, eps)


def _ref_attention(attn, args, x, positions, shared):
    """Full attention with the CSA2 shared-runtime dispatch, in numpy."""
    r = _RefAttn(attn, args)
    cos, sin = _cos_sin(r.inv_freq, positions)
    b, s, _ = x.shape
    win_kv = r.window_kv(x, cos, sin)  # [b, s, hd]  (prefill: window == all rows)
    wpos = np.arange(win_kv.shape[1])
    qp = positions[:, None]
    wp = wpos[None, :]
    win_mask = (wp <= qp) & (wp > qp - r.win)
    mask = np.broadcast_to(win_mask[None], (b, s, win_kv.shape[1]))
    KV = win_kv

    if r.ratio:
        qr = r.qr(x)
        if attn.is_kv_source:
            latent_pre = _ref_compress_prefill(attn, x, r.eps)
            group_pos = np.arange(latent_pre.shape[1]) * r.ratio
            gcos, gsin = _cos_sin(r.inv_freq, group_pos)
            shared["compress_kv"] = _rope(latent_pre, gcos, gsin)
            shared["index_k"] = _ref_index_keys(attn, latent_pre, gcos, gsin, r.eps)
        compress_kv = shared["compress_kv"]
        index_k = shared["index_k"]
        n_comp = compress_kv.shape[1]
        compress_lens = (positions + 1) // r.ratio
        if attn.is_index_source:
            set_c = attn.is_candidate_source
            cand = None if set_c else shared.get("candidates")
            cmask, cand_out = _ref_indexer(
                attn, args, x, qr, index_k, positions, cos, sin,
                compress_lens, n_comp, cand, set_c,
            )
            shared["topk"] = cmask
            if set_c:
                shared["candidates"] = cand_out
        else:
            cmask = shared["topk"]
        KV = np.concatenate([win_kv, compress_kv], axis=1)
        mask = np.concatenate([mask, cmask], axis=-1)

    return r.out(x, KV, mask, positions, cos, sin)


def _ref_moe(mlp, args, x):
    xf = x.reshape(-1, x.shape[-1])
    w = _np(mlp.gate.weight)
    scores = np.sqrt(np.logaddexp(_mm(xf, w.T), 0.0))  # sqrt(softplus), overflow-safe
    bias = _np(mlp.gate.e_score_correction_bias)
    biased = scores + bias
    topk = mlp.gate.topk
    idx = np.argsort(-biased, axis=-1)[:, :topk]
    weights = np.take_along_axis(scores, idx, axis=-1)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    weights = weights * args.routed_scaling_factor
    # routed experts
    gate_w = _np(mlp.switch_mlp.gate_proj.weight)   # [E, inter, dim]
    up_w = _np(mlp.switch_mlp.up_proj.weight)
    down_w = _np(mlp.switch_mlp.down_proj.weight)
    limit = args.swiglu_limit
    n = xf.shape[0]
    routed = np.zeros((n, xf.shape[-1]))
    for t in range(topk):
        e = idx[:, t]  # [n]
        g = _einsum("nd,ndi->ni", xf, gate_w[e].transpose(0, 2, 1))
        u = _einsum("nd,ndi->ni", xf, up_w[e].transpose(0, 2, 1))
        if limit > 0:
            g = np.minimum(g, limit)
            u = np.clip(u, -limit, limit)
        h = _silu(g) * u
        o = _einsum("ni,nid->nd", h, down_w[e].transpose(0, 2, 1))
        routed += weights[:, t:t + 1] * o
    # shared expert
    se = mlp.shared_experts
    g = _mm(xf, _np(se.w1.weight).T)
    u = _mm(xf, _np(se.w3.weight).T)
    if limit > 0:
        g = np.minimum(g, limit)
        u = np.clip(u, -limit, limit)
    shared = _mm(_silu(g) * u, _np(se.w2.weight).T)
    return (routed + shared).reshape(x.shape)


def _hc_split_sinkhorn(mixes, scale, base, hc, iters, eps):
    pre = 1.0 / (1.0 + np.exp(-(mixes[..., :hc] * scale[0] + base[:hc]))) + eps
    post = 2.0 / (1.0 + np.exp(-(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])))
    comb = mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)
    comb = _softmax(comb, axis=-1) + eps
    comb = comb / (np.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (np.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (np.sum(comb, axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _ref_mixes(layer, h, fn, base, scale):
    hc = layer.hc_mult
    flat = h.reshape(*h.shape[:-2], -1)
    rsqrt = 1.0 / np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + layer.norm_eps)
    mixes = _mm(flat, _np(fn).T) * rsqrt
    return _hc_split_sinkhorn(mixes, _np(scale), _np(base), hc, layer.hc_iters, layer.hc_eps)


def _hc_pre(h, pre_mix):
    return np.sum(pre_mix[..., None] * h, axis=2)


def _hc_post(x, residual, post, comb):
    term = post[..., None] * x[..., None, :]
    mixed = _einsum("...jk,...jd->...kd", comb, residual)
    return term + mixed


def _ref_block(layer, args, h, pre_mix, positions, shared):
    residual = h
    a_pre, a_post, a_comb = _ref_mixes(layer, h, layer.hc_attn_fn, layer.hc_attn_base, layer.hc_attn_scale)
    x = _hc_pre(h, pre_mix)
    x = _rmsnorm(x, _np(layer.attn_norm_weight), layer.norm_eps)
    x = _ref_attention(layer.attn, args, x, positions, shared)
    h = _hc_post(x, residual, a_post, a_comb)

    residual = h
    f_pre, f_post, f_comb = _ref_mixes(layer, h, layer.hc_ffn_fn, layer.hc_ffn_base, layer.hc_ffn_scale)
    x = _hc_pre(h, a_pre)  # ffn collapses with the attn-computed pre (threaded)
    x = _rmsnorm(x, _np(layer.ffn_norm_weight), layer.norm_eps)
    x = _ref_moe(layer.mlp, args, x)
    h = _hc_post(x, residual, f_post, f_comb)
    return h, f_pre


def _ref_model(model, ids, positions=None):
    args = model.args
    b, s = ids.shape
    if positions is None:
        positions = np.arange(s)
    emb = _np(model.model.embed_tokens.weight)
    h = emb[np.asarray(ids)]  # [b, s, dim]
    h = np.broadcast_to(h[:, :, None, :], (b, s, args.hc_mult, h.shape[-1])).copy()
    pre_mix = np.zeros((b, s, args.hc_mult))
    pre_mix[:, :, 0] = 1.0
    shared = {}
    for layer in model.model.layers:
        h, pre_mix = _ref_block(layer, args, h, pre_mix, positions, shared)
    h = np.sum(pre_mix[..., None] * h, axis=2)
    h = _rmsnorm(h, _np(model.model.norm_weight), args.rms_norm_eps)
    return _lin(h, _np(model.head.weight))


# ---------------------------------------------------------------------------
# test fixtures
# ---------------------------------------------------------------------------
def _randomize(model, seed=0):
    mx.random.seed(seed)
    flat = tree_flatten(model.parameters())
    new = []
    for name, arr in flat:
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = 0.3 * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


def _swa_args(**over):
    base = dict(
        vocab_size=48, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=12, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        sliding_window=6, window_size=6, swiglu_limit=0.5,
        compress_ratios=[0, 0],
        kv_source_layer_ids=[], index_source_layer_ids=[],
        candidate_source_layer_id=-1,
    )
    base.update(over)
    return ModelArgs(**base)


def _csa_args(**over):
    # 8 layers: swa, swa, full-r2, reuse, reuse, full-r1(cand), reindex, reuse
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


# ---------------------------------------------------------------------------
# P1.2: SWA-only attention
# ---------------------------------------------------------------------------
def test_attention_swa_only():
    args = _swa_args()
    model = Model(args)
    _randomize(model, seed=1)
    attn = model.model.layers[0].attn
    assert attn.mode == "swa_only" and attn.compressor is None

    s = 32
    x = mx.random.normal((1, s, args.hidden_size)) * 0.5
    positions = mx.arange(s)
    out = attn(x, positions, _LayerCache(), _SharedRuntime())
    mx.eval(out)

    ref = _ref_attention(attn, args, _np(x), np.arange(s), {})
    diff = np.max(np.abs(_np(out) - ref))
    assert diff < 1e-3, f"swa attention max abs diff {diff}"


# ---------------------------------------------------------------------------
# P1.1: dense block (HC threading + MoE + norms), SWA attention
# ---------------------------------------------------------------------------
def test_dense_block_parity():
    args = _swa_args()
    model = Model(args)
    _randomize(model, seed=2)
    layer = model.model.layers[0]

    s = 24
    hc = args.hc_mult
    h = mx.random.normal((1, s, hc, args.hidden_size)) * 0.4
    pre = mx.concatenate(
        [mx.ones((1, s, 1)), mx.zeros((1, s, hc - 1))], axis=-1
    ).astype(mx.float32)
    positions = mx.arange(s)
    out, out_pre = layer(h, pre, positions, _LayerCache(), _SharedRuntime())
    mx.eval(out, out_pre)

    ref_h, ref_pre = _ref_block(layer, args, _np(h), _np(pre), np.arange(s), {})
    dh = np.max(np.abs(_np(out) - ref_h))
    dp = np.max(np.abs(_np(out_pre) - ref_pre))
    assert dh < 1e-2, f"block hidden max abs diff {dh}"
    assert dp < 1e-4, f"block pre_mix max abs diff {dp}"


# ---------------------------------------------------------------------------
# P1.3: full CSA2 forward across the ratio-2->1 boundary + candidate prefilter
# ---------------------------------------------------------------------------
def test_csa2_modes():
    args = _csa_args()
    model = Model(args)
    _randomize(model, seed=3)
    for layer in model.model.layers:
        layer.attn.capture_selection = True

    s = 288  # >= 256, crosses the ratio-2 (128 rows) -> ratio-1 (288 rows) boundary
    ids = mx.array([[(i * 7 + 3) % args.vocab_size for i in range(s)]])
    logits = model(ids)
    mx.eval(logits)

    ref = _ref_model(model, np.asarray(ids))
    mx_arg = np.asarray(mx.argmax(logits, axis=-1)[0])
    ref_arg = np.argmax(ref[0], axis=-1)
    # argmax must match wherever the decision is well separated; the fp32-accumulate
    # oracle carries only ~1e-5 residual (elementwise fp32-vs-fp64) at this depth, so
    # a token whose reference top-2 logits are within that band may pick the near-tie.
    TIE = 5e-3  # >> the residual oracle error at this depth; << any real margin
    mism = np.where(mx_arg != ref_arg)[0]
    for p in mism:
        top2 = np.sort(ref[0, p])[-2:]
        margin = float(top2[1] - top2[0])
        assert margin < TIE, f"pos {p}: real argmax divergence, ref top-2 margin {margin}"
    assert len(mism) <= 2, f"{len(mism)}/{s} argmax mismatches (only tie flips allowed)"

    # A Reuse layer's selected compressed rows equal its source layer's.
    L = model.model.layers
    sel = {i: np.asarray(L[i].attn.last_selection) for i in (2, 3, 4, 5, 6, 7)}
    assert np.array_equal(sel[3], sel[2]), "reuse L3 must reuse L2 (full r2) top-k"
    assert np.array_equal(sel[4], sel[2]), "reuse L4 must reuse L2 (full r2) top-k"
    assert np.array_equal(sel[7], sel[6]), "reuse L7 must reuse L6 (reindex) top-k"
    # Reindex L6 owns its queries: its selection generally differs from source L5.
    assert not np.array_equal(sel[6], sel[5])
    # the candidate prefilter is non-trivial: at the final (fully reachable) query
    # it keeps far fewer than all compressed rows.
    cand = np.asarray(L[6].attn.last_candidates)  # [1, s, n_comp] from L5
    n_comp = cand.shape[-1]
    assert cand is not None and cand[0, -1].sum() < n_comp
    # index top-k then trims further: selected rows <= candidate rows at that query
    assert sel[6][0, -1].sum() <= cand[0, -1].sum()


# ---------------------------------------------------------------------------
# P1.3: trim/rollback restores identical logits
# ---------------------------------------------------------------------------
def test_rollback():
    args = _csa_args()  # includes ratio-2 (partial-group) and ratio-1 compress layers
    model = Model(args)
    _randomize(model, seed=4)
    prompt = mx.array([[(i * 5 + 1) % args.vocab_size for i in range(40)]])
    step_ids = [mx.array([[11]]), mx.array([[23]]), mx.array([[7]])]

    # (0) incremental decode matches a single full prefill of the same tokens,
    #     so the decode path (ring window + partial compressor groups) reproduces
    #     the oracle-validated prefill path.
    whole = mx.concatenate([prompt] + step_ids, axis=1)
    full = model(whole)
    mx.eval(full)
    cache0 = model.make_cache()
    model(prompt, cache0)
    for k, tok in enumerate(step_ids):
        lg = model(tok, cache0)
        mx.eval(lg)
        pos = prompt.shape[1] + k
        a = np.asarray(mx.argmax(lg[0, -1]))
        b = np.asarray(mx.argmax(full[0, pos]))
        assert a == b, f"decode step {k} argmax != full prefill at pos {pos}"

    # (1) decode is deterministic across two independent caches
    def decode_run():
        cache = model.make_cache()
        model(prompt, cache)
        outs = []
        for tok in step_ids:
            lg = model(tok, cache)
            mx.eval(lg)
            outs.append(np.asarray(lg[0, -1]))
        return outs

    first = decode_run()
    replay = decode_run()
    for a, b in zip(first, replay):
        assert np.array_equal(a, b), "decode is not deterministic"

    # (2) rollback restores identical logits: mark, decode a divergent tail,
    #     roll back, then decode the real continuation and match `first`.
    cache = model.make_cache()
    model(prompt, cache)
    mark = cache.mark()
    model(mx.array([[41]]), cache)  # divergent throwaway steps past the mark
    model(mx.array([[9]]), cache)
    cache.rollback(mark)
    assert cache.offset == 40

    rolled = []
    for tok in step_ids:
        lg = model(tok, cache)
        mx.eval(lg)
        rolled.append(np.asarray(lg[0, -1]))
    for a, b in zip(first, rolled):
        assert np.array_equal(a, b), "rollback did not restore identical logits"
