#!/usr/bin/env python3
"""K22 (W41) CPU dispatch census for DeepSeek-V4.1-Flash decode.

Window-13 stage timing (docs/deepseek-v41/W37_STAGE_TIMING_PROBE.md) attributed
*wall time* per decode stage but could not say how many graph PRIMITIVES each
stage dispatches -- the lever that whole-chain ``mx.compile`` moves (a single M=1
attention step costing 1.7-4.9 ms is a dispatch-chain problem, not a compute one:
[[b1-decode-dispatch-removal-hides]]).  This census counts primitives per decode
stage, before/after ``MTPLX_DSV41_ATTN_COMPILE``, so the K22 collapse is a number.

How it counts (reliable on CPU, no GPU, no artifact)
----------------------------------------------------
``mx.export_to_dot`` writes the lazy MLX graph of a set of output arrays; every
``[label ="Op", shape=rectangle]`` node is one primitive.  ``mx.compile`` FUSES
elementwise chains into single ``Compiled*`` nodes, so the dot node count drops
exactly as the tape collapses (validated here on the Sinkhorn: 198 eager -> 80
compiled).  The K3 worker's "~119 primitives per Sinkhorn call" is this same
count of the ``_sinkhorn_ops`` graph.

Two censuses are produced:

* **Full-model per-stage** -- a ``_CensusProbe`` is installed into the W37 stage
  singleton (``deepseek_v41_stage_timing._ACTIVE``); it reuses the model's own
  ``with _stime.stage(name)`` brackets but, instead of timing, counts the graph
  primitives of each stage's output arrays and then ``mx.eval``s them (so the
  next stage's subgraph is disjoint and the per-stage counts tile the token).
  Run flag-off (eager) and flag-on (K22) on the tiny real-structure model (all
  four CSA modes -- swa_only/full/reindex/reuse -- HC, gate, combine, and a
  synthetic engram hook on the engram layers).
* **Per-chain micro-census** -- each compiled chain (attention QKV-prep,
  attention output-prep, gate prefix, MoE combine) counted eager vs compiled in
  isolation with the op-type breakdown, the cleanest before/after evidence.

Run:
    PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \
        scripts/deepseek_v41/dispatch_census.py --out receipt.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter
from contextlib import contextmanager

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
from mtplx.models import deepseek_v41_stage_timing as _stime  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402

_DOT_RECT = re.compile(r'\[label ="([^"]+)", shape=rectangle\]')


def count_prims(*outs):
    """(#primitives, Counter by op label) in the lazy graph rooted at ``outs``.

    Inputs that are already ``mx.eval``'d appear as leaf/source nodes, not
    rectangles, so only the ops between the eval'd leaves and ``outs`` are
    counted."""
    arrs = [a for a in outs if isinstance(a, mx.array)]
    if not arrs:
        return 0, Counter()
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrs)
    labels = _DOT_RECT.findall(buf.getvalue())
    return len(labels), Counter(labels)


# ---------------------------------------------------------------------------
# tiny real-structure model (all 4 CSA modes + HC + gate + combine + engram hook)
# ---------------------------------------------------------------------------
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


# The engram artifact is 104 GiB (not loadable here); a synthetic hook keeps the
# engram-layer STRUCTURE in the census (K22 does not touch engram, so its real
# primitive count is out of scope -- this exercises the engram-hooked layer path
# and books a representative additive-residual ``engram.apply`` bracket).
class _FakeEngramState:
    def advance(self, token_ids):
        with _stime.stage("engram.advance"):
            pass

    def current_row_ids(self, idx):
        return None


class _FakeEngramHash:
    def fresh(self):
        return _FakeEngramState()


def _make_engram_hook(dim):
    gate_w = 0.05 * mx.random.normal((dim,))
    val_w = 0.05 * mx.random.normal((dim,))
    mx.eval(gate_w, val_w)

    def hook(h, input_ids, state):
        with _stime.stage("engram.apply") as _st:
            g = mx.sigmoid(h * gate_w)
            out = h + g * val_w
            _st.add(out)
        return out

    return hook


def _attach_synthetic_engram(model, layer_ids=(1, 6)):
    model.model.engram_hash = _FakeEngramHash()
    for lid in layer_ids:
        model.model.layers[lid].engram_hook = _make_engram_hook(model.args.hidden_size)


def _build(seed=1, engram=True):
    args = _csa_args()
    model = Model(args)
    _randomize(model, seed=seed)
    if engram:
        _attach_synthetic_engram(model)
    return model, args


@contextmanager
def _attn_flag(flag, cap, win_memo=False):
    of, orr = dv41._ATTN_COMPILE, dv41._ATTN_COMPILE_MAX_ROWS
    om = dv41._ATTN_WIN_MEMO
    dv41._ATTN_COMPILE = flag
    dv41._ATTN_COMPILE_MAX_ROWS = cap
    dv41._ATTN_WIN_MEMO = win_memo
    dv41._ATTN_COMPILED.clear()
    try:
        yield
    finally:
        dv41._ATTN_COMPILE, dv41._ATTN_COMPILE_MAX_ROWS = of, orr
        dv41._ATTN_WIN_MEMO = om
        dv41._ATTN_COMPILED.clear()


# ---------------------------------------------------------------------------
# census probe: same W37 stage tiling, counts primitives instead of timing
# ---------------------------------------------------------------------------
class _CensusFence:
    __slots__ = ("_arrays",)

    def __init__(self):
        self._arrays = []

    def add(self, *arrays):
        for a in arrays:
            if isinstance(a, mx.array):
                self._arrays.append(a)

    fence = add


class _CensusProbe:
    """Duck-types the ``_stime._Probe`` surface the model reads, but at each stage
    boundary counts the graph primitives of the stage's outputs (before eval)
    rather than timing them."""

    def __init__(self):
        self._recording_now = False
        self.prims = Counter()
        self.counts = Counter()
        self.optypes = {}
        self.tokens = 0

    def enter_forward(self, seq_len):
        self._recording_now = seq_len == 1

    @contextmanager
    def _stage(self, name):
        fence = _CensusFence()
        try:
            yield fence
        finally:
            if fence._arrays:
                n, ops = count_prims(*fence._arrays)
                self.prims[name] += n
                self.counts[name] += 1
                self.optypes.setdefault(name, Counter()).update(ops)
                mx.eval(fence._arrays)

    @contextmanager
    def _frame(self):
        try:
            yield
        finally:
            self.tokens += 1

    def snapshot(self):
        tokens = self.tokens or 1
        stages = {}
        for name in sorted(set(self.prims) | set(self.counts)):
            c = self.counts.get(name, 0)
            p = self.prims.get(name, 0)
            stages[name] = {
                "primitives_per_token": p / tokens,
                "count_per_token": c / tokens,
                "primitives_per_call": (p / c) if c else 0.0,
                "op_types": dict(self.optypes.get(name, {})),
            }
        return {
            "tokens": self.tokens,
            "total_primitives_per_token": sum(self.prims.values()) / tokens,
            "stages": stages,
        }


@contextmanager
def _census_session():
    probe = _CensusProbe()
    old = _stime._ACTIVE
    _stime._ACTIVE = probe
    try:
        yield probe
    finally:
        _stime._ACTIVE = old


def _run_full_census(flag, cap, seed=1, decode_tokens=(3, 17, 5), prefill_s=12,
                     win_memo=False):
    """Decode a few tokens under the census probe; return the snapshot.

    ``flag`` = K22 attention-tape compile; ``win_memo`` = K24 window-mask memo.
    Prefill is one-shot at ``s = prefill_s > cap`` so it is eager (and not
    censused -- ``_recording_now`` gates on s==1), matching the W37 probe."""
    model, args = _build(seed=seed)
    with _attn_flag(flag, cap, win_memo=win_memo):
        ids = mx.array(np.random.RandomState(0).randint(0, args.vocab_size, size=(1, prefill_s)))
        cache = model.make_cache()
        mx.eval(model(ids, cache=cache, prefill_chunk=0))
        with _census_session() as probe:
            for t in decode_tokens:
                with _stime.frame():
                    logits = model(mx.array([[t]]), cache=cache)
                    with _stime.stage("sample") as _st:
                        tok = mx.argmax(logits[:, -1, :], axis=-1)
                        _st.add(tok)
                    mx.eval(tok)
        return probe.snapshot()


# ---------------------------------------------------------------------------
# per-chain micro-census: each compiled chain eager vs compiled, in isolation
# ---------------------------------------------------------------------------
def _micro_census(seed=1):
    """Count each K22-compiled chain's primitives eager vs compiled (rows=1)."""
    from mtplx.models.deepseek_v41 import (
        _rmsnorm, _rope_last, _cos_sin, _lin_arrays, _attn_qkv_prep, _attn_out_prep,
    )
    import mtplx.models.deepseek_v41_moe as moe

    model, args = _build(seed=seed, engram=False)
    # a FULL (CSA-index-source) attention layer exercises the whole prep chain
    attn = model.model.layers[2].attn
    dim = args.hidden_size
    x = 0.3 * mx.random.normal((1, 1, dim))
    positions = mx.array([5])
    mx.eval(x)
    qcos, qsin = _cos_sin(attn.inv_freq, positions)
    mx.eval(qcos, qsin)

    out = {}

    # QKV-prep chain
    def qkv_eager():
        b, s, _ = x.shape
        qr = _rmsnorm(attn.wq_a(x), attn.q_norm_weight, attn.eps)
        q = _rope_last(attn.wq_b(qr).reshape(b, s, attn.n_heads, attn.head_dim), qcos, qsin)
        kv = _rope_last(_rmsnorm(attn.wkv(x), attn.kv_norm_weight, attn.eps), qcos, qsin)
        return q, qr, kv

    qe = qkv_eager()
    qc = _attn_qkv_prep(attn)(
        x, qcos, qsin, attn.q_norm_weight, attn.kv_norm_weight,
        *_lin_arrays(attn.wq_a), *_lin_arrays(attn.wq_b), *_lin_arrays(attn.wkv),
    )
    out["attn.qkv_prep"] = _pair(qe, qc)

    # output-prep chain (o is the SDPA output [b,s,H,hd])
    o = 0.3 * mx.random.normal((1, 1, attn.n_heads, attn.head_dim))
    mx.eval(o)

    def out_eager():
        b, s = 1, 1
        oo = _rope_last(o, qcos, qsin, inverse=True)
        oo = oo.reshape(b, s, attn.n_groups, -1)
        oo = attn._o_lora_down(oo)
        return attn.wo_b(oo.reshape(b, s, -1))

    oe = out_eager()
    oc = _attn_out_prep(attn)(o, qcos, qsin, attn._o_lora_dense_weight(), *_lin_arrays(attn.wo_b))
    out["attn.out_prep"] = _pair((oe,), (oc,))

    # gate prefix
    gate = model.model.layers[2].mlp.gate
    xf = 0.3 * mx.random.normal((1, dim))
    mx.eval(xf)

    ge = _gate_eager(gate, xf)
    gc = moe._gate_prefix(gate)(xf, gate.weight, gate.e_score_correction_bias)
    out["moe.gate_prefix"] = _pair(ge, gc)

    # moe combine
    tk = args.num_experts_per_tok
    routed = 0.3 * mx.random.normal((1, tk, dim))
    weights = 0.3 * mx.random.normal((1, tk))
    shared = 0.3 * mx.random.normal((1, dim))
    mx.eval(routed, weights, shared)
    ce = (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2) + shared
    cc = moe._moe_combine(routed, weights, shared)
    out["moe.combine"] = _pair((ce,), (cc,))
    return out


def _gate_eager(gate, xf):
    import mlx.nn as nn
    s = (xf.astype(mx.float32) @ gate.weight.astype(mx.float32).T) / gate.gate_temp
    s = mx.sqrt(nn.softplus(s))
    return s, s + gate.e_score_correction_bias


def _pair(eager_outs, compiled_outs):
    ne, oe = count_prims(*eager_outs)
    nc, oc = count_prims(*compiled_outs)
    mx.eval([a for a in eager_outs if isinstance(a, mx.array)]
            + [a for a in compiled_outs if isinstance(a, mx.array)])
    return {
        "eager": {"primitives": ne, "op_types": dict(oe)},
        "compiled": {"primitives": nc, "op_types": dict(oc)},
        "reduction": ne - nc,
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _render_table(before, after):
    bs, as_ = before["stages"], after["stages"]
    names = sorted(set(bs) | set(as_),
                   key=lambda n: -bs.get(n, as_.get(n, {})).get("primitives_per_token", 0))
    w = 24
    lines = [
        f"{'stage':<{w}} {'calls/tok':>9} {'prim/call(before)':>18} "
        f"{'prim/tok(before)':>17} {'prim/tok(after)':>16} {'delta':>7}",
        "-" * (w + 9 + 18 + 17 + 16 + 7 + 5),
    ]
    tb = ta = 0.0
    for n in names:
        b = bs.get(n, {})
        a = as_.get(n, {})
        pb = b.get("primitives_per_token", 0.0)
        pa = a.get("primitives_per_token", 0.0)
        cc = b.get("count_per_token", a.get("count_per_token", 0.0))
        pc = b.get("primitives_per_call", 0.0)
        tb += pb
        ta += pa
        lines.append(
            f"{n:<{w}} {cc:>9.1f} {pc:>18.1f} {pb:>17.1f} {pa:>16.1f} "
            f"{pb - pa:>7.1f}"
        )
    lines.append("-" * (w + 9 + 18 + 17 + 16 + 7 + 5))
    lines.append(f"{'TOTAL':<{w}} {'':>9} {'':>18} {tb:>17.1f} {ta:>16.1f} {tb - ta:>7.1f}")
    return "\n".join(lines)


def _render_micro(micro):
    lines = [f"{'chain':<20} {'eager':>7} {'compiled':>9} {'reduction':>10}"]
    lines.append("-" * 50)
    for name, d in micro.items():
        lines.append(f"{name:<20} {d['eager']['primitives']:>7} "
                     f"{d['compiled']['primitives']:>9} {d['reduction']:>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# K33 (W65): draft-block census -- the DSpark-DIRECT draft block on the tiny double
# ---------------------------------------------------------------------------
# The draft block's own stage brackets (deepseek_v41_dspark, no-op unless a probe
# is armed) tile it: forward_embed, the 3 stages' attention prep (main_kv / qkv /
# sdpa / out) + Hyper-Connection prep (attn_prep / ffn_prep / moe_combine) + the
# reused MoE (gate_topk / routed_switch / shared_expert / combine), then the last
# stage's head + markov + confidence.  We arm the census probe, run ONE draft_block
# (one cycle == one "token" here), and count primitives per stage before/after
# MTPLX_DSV41_DRAFT_COMPILE -- the same _CensusProbe / count_prims machinery as the
# full-model backbone census, so the reduction is a number.
def _dspark_args():
    return ModelArgs(
        vocab_size=64, hidden_size=32, num_hidden_layers=5, num_attention_heads=4,
        head_dim=16, qk_rope_head_dim=8, q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        sliding_window=8, window_size=8, hc_mult=4, hc_sinkhorn_iters=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0,
        n_mtp_layers=3, dspark_block_size=4, dspark_noise_token_id=63,
        dspark_target_layer_ids=[2, 3, 4], dspark_markov_rank=12,
        dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )


def _build_dspark(seed=1):
    """Tiny real-structure DSpark head (3 stages, block_size 4, resident 8-expert
    top-2 MoE == the 128-expert top-3 structure at small scale).  Power-of-2
    reductions (hidden 32, hc*dim 128, q_lora 16, head_dim 16) so every compiled
    chain is bit-exact vs eager (the K22 tiny-config RMSNorm caveat)."""
    mx.random.seed(seed)
    args = _dspark_args()
    model = Model(args, quantize=False, mtp=True)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if value.ndim == 1:
            centre = 1.0 if leaf.endswith("norm_weight") or leaf == "scale" else 0.0
            new = mx.random.normal(value.shape) * 0.1 + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    mx.eval(model.parameters())
    return model, args


@contextmanager
def _draft_flag(flag):
    import mtplx.models.deepseek_v41_dspark as dsp
    prev = dsp._DRAFT_COMPILE
    dsp._DRAFT_COMPILE = flag
    dsp._DRAFT_COMPILED.clear()
    dv41._ATTN_COMPILED.clear()
    dv41._HC_COMPILED.clear()
    try:
        yield
    finally:
        dsp._DRAFT_COMPILE = prev
        dsp._DRAFT_COMPILED.clear()
        dv41._ATTN_COMPILED.clear()
        dv41._HC_COMPILED.clear()


def _run_draft_census(flag, seed=1):
    """One draft_block under the census probe; return the per-stage snapshot.

    ``flag`` = K33 draft-block compile (the module global, flipped like the K22
    census flips ``deepseek_v41._ATTN_COMPILE``)."""
    model, args = _build_dspark(seed=seed)
    # a real prompt forward to produce main_hidden + seed the DSpark windows, then
    # draft one block from the last prompt hidden (exactly the decode-lane cycle).
    ids = mx.array(np.random.RandomState(0).randint(0, args.vocab_size, size=(1, 17)))
    logits, main_hidden = model(ids, return_hidden=True)
    mx.eval(logits, main_hidden)
    caches = model.make_mtp_cache()
    model.mtp.seed_main(main_hidden, caches)
    primary = mx.array([int(mx.argmax(logits[0, -1]))])
    main_h = main_hidden[:, -1:, :]
    embed, head = model.model.embed_tokens, model.head
    with _draft_flag(flag):
        with _census_session() as probe:
            probe._recording_now = True  # the draft block is a block-row chain, not a 1-row model forward
            with _stime.frame():
                out_ids, dlogits, conf = model.mtp.draft_block(
                    main_h, primary, caches, embed, head
                )
                with probe._stage("sample") as _st:
                    _st.add(out_ids, conf)
                mx.eval(out_ids, dlogits, conf)
        return probe.snapshot()


# ---------------------------------------------------------------------------
# K35 (W91): small-stages fusion census -- the per-layer AR-decode dispatch count
# for everything that is NOT attention and NOT the routed switch, before/after
# MTPLX_DSV41_SMALL_STAGES_FUSED, plus the GPU projection with the K3 Sinkhorn
# kernel collapsing each 80-op compiled Sinkhorn to one dispatch.
# ---------------------------------------------------------------------------
#: One compiled Sinkhorn recurrence (hc=4, iters=20) is 80 graph primitives
#: (198 eager); the K3 Metal kernel (MTPLX_DSV41_SINKHORN_METAL, GPU only)
#: collapses it to a single dispatch.  Measured by
#: ``mx.compile(_sinkhorn_ops)`` + ``count_prims`` (tests assert it).
_SINK_COMPILED_HC4_IT20 = 80


def _small_seg_micro(seed=1):
    """Per-segment (seg1/seg2/seg3) eager-vs-compiled primitive counts for the
    small-stages fusion on ONE full CSA layer at M=1 -- the same ``count_prims``
    machinery as :func:`_micro_census`, counting each fused graph in isolation
    (bypassing the ``_stime.recording`` eager-guard that keeps the *timing* census
    on the eager path).  Returns ``{seg: {eager, compiled, gpu_kernel, n_sinkhorn,
    op_types}}`` where ``gpu_kernel`` projects each compiled Sinkhorn (80 prims)
    down to the one K3-kernel dispatch."""
    from mtplx.models.deepseek_v41 import (
        _hc_attn_prep_impl, _hc_ffn_prep_impl, _gate_topk_impl, _shared_expert_impl,
        _hc_post_impl, _lin_desc, _lin_n, _lin_arrays, _small_compiled,
    )
    model, args = _build(seed=seed, engram=False)
    dv41._SMALL_STAGES_COMPILED.clear()
    # a FULL layer (owns compressed KV + indexer) exercises every small stage
    L = next(l for l in model.model.layers if l.attn.mode == dv41.MODE_FULL)
    hc, dim = L.hc_mult, args.hidden_size
    g, se = L.mlp.gate, L.mlp.shared_experts
    consts = (hc, L.hc_iters, L.norm_eps, L.hc_eps)
    mx.random.seed(0)
    h = (0.3 * mx.random.normal((1, 1, hc, dim))).astype(mx.float32)
    pre_mix = mx.concatenate(
        [mx.ones((1, 1, 1)), mx.zeros((1, 1, hc - 1))], axis=-1).astype(mx.float32)
    attn_out = (0.3 * mx.random.normal((1, 1, dim))).astype(mx.float32)
    mx.eval(h, pre_mix, attn_out)
    ai, apre, apost, acomb = _hc_attn_prep_impl(
        h, pre_mix, L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, L.attn_norm_weight, *consts)
    mx.eval(ai, apre, apost, acomb)
    w1d, w3d, w2d = _lin_desc(se.w1), _lin_desc(se.w3), _lin_desc(se.w2)
    n1, n3 = _lin_n(w1d), _lin_n(w3d)
    warrs = _lin_arrays(se.w1) + _lin_arrays(se.w3) + _lin_arrays(se.w2)
    gt, sf, tk = float(g.gate_temp), str(g.score_func), int(g.topk)
    ntp, rsf, swl = bool(g.norm_topk_prob), float(g.route_scale), float(se.swiglu_limit)
    # ffn-prep outputs (feed seg3); recomputed eager so seg3's inputs are eval'd leaves
    mi, h2, fpost, fcomb, fpre = _hc_ffn_prep_impl(
        attn_out, h, apre, apost, acomb,
        L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale, L.ffn_norm_weight, *consts)
    xf0 = mi.reshape(-1, dim)
    wt0, ix0 = _gate_topk_impl(xf0, g.weight, g.e_score_correction_bias, gt, sf, tk, ntp, rsf)
    sh0 = _shared_expert_impl(xf0, warrs[:n1], warrs[n1:n1 + n3], warrs[n1 + n3:],
                              w1d, w3d, w2d, swl).astype(mx.float32)
    routed = (0.3 * mx.random.normal((1, tk, dim))).astype(mx.float32)
    mx.eval(h2, fpost, fcomb, wt0, sh0, routed)

    def seg1_e():
        return _hc_attn_prep_impl(h, pre_mix, L.hc_attn_fn, L.hc_attn_base,
                                  L.hc_attn_scale, L.attn_norm_weight, *consts)

    def seg1_c():
        return _small_compiled("seg1", L)(h, pre_mix, L.hc_attn_fn, L.hc_attn_base,
                                          L.hc_attn_scale, L.attn_norm_weight)

    def seg2_e():
        m2, h2b, fp, fc, fpr = _hc_ffn_prep_impl(
            attn_out, h, apre, apost, acomb,
            L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale, L.ffn_norm_weight, *consts)
        xf = m2.reshape(-1, dim)
        w, ix = _gate_topk_impl(xf, g.weight, g.e_score_correction_bias, gt, sf, tk, ntp, rsf)
        sh = _shared_expert_impl(xf, warrs[:n1], warrs[n1:n1 + n3], warrs[n1 + n3:],
                                 w1d, w3d, w2d, swl).astype(mx.float32)
        return xf, w, ix, sh, h2b, fp, fc, fpr

    def seg2_c():
        return _small_compiled("seg2", L)(
            attn_out, h, apre, apost, acomb,
            L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale, L.ffn_norm_weight,
            g.weight, g.e_score_correction_bias, *warrs)

    def seg3_e():
        y = (routed.astype(mx.float32) * wt0[..., None]).sum(-2) + sh0
        y = y.astype(h2.dtype).reshape(*h2.shape[:-2], dim)
        return (_hc_post_impl(y, h2, fpost, fcomb),)

    def seg3_c():
        return (_small_compiled("seg3", L)(routed, wt0, sh0, h2, fpost, fcomb),)

    n_sink = {"seg1": 1, "seg2": 1, "seg3": 0}
    out = {}
    for name, ef, cf in (("seg1", seg1_e, seg1_c), ("seg2", seg2_e, seg2_c),
                         ("seg3", seg3_e, seg3_c)):
        e = ef()
        ne, _ = count_prims(*[a for a in e if isinstance(a, mx.array)])
        c = cf()
        nc, oc = count_prims(*[a for a in c if isinstance(a, mx.array)])
        mx.eval([a for a in e if isinstance(a, mx.array)]
                + [a for a in c if isinstance(a, mx.array)])
        ns = n_sink[name]
        out[name] = {
            "eager": ne, "compiled": nc,
            "gpu_kernel": nc - ns * (_SINK_COMPILED_HC4_IT20 - 1),
            "n_sinkhorn": ns, "op_types": dict(oc),
        }
    return out, args.num_hidden_layers


_SMALL_STAGE_NAMES = (
    "hc.premix_sinkhorn", "hc.combine", "moe.gate_topk",
    "moe.shared_expert", "moe.combine",
)


def _run_small_census(seed=1, cap=7):
    eager_full = _run_full_census(False, cap, seed=seed)
    segs, n_layers = _small_seg_micro(seed=seed)
    return {"eager_full": eager_full, "segments": segs, "n_layers": n_layers}


def _small_main(args):
    """K35/W91 small-stages census: the per-layer AR-decode dispatch count for the
    small stages, before (eager) / after (fixed-shape ``mx.compile``) / GPU
    projection (+ K3 Sinkhorn kernel), and the eager per-stage baseline."""
    r = _run_small_census(seed=args.seed, cap=args.cap)
    ef, segs, nL = r["eager_full"], r["segments"], r["n_layers"]
    print("=" * 82)
    print("W91 small-stages dispatch census -- AR-decode (M=1) per layer, tiny "
          "real-structure")
    print("=" * 82)
    print("eager per-stage baseline (primitives / token, summed over all layers):")
    small_tot = 0.0
    for n in _SMALL_STAGE_NAMES:
        p = ef["stages"].get(n, {}).get("primitives_per_token", 0.0)
        small_tot += p
        print(f"  {n:<24} {p:>9.1f}")
    print(f"  {'SMALL-STAGE TOTAL':<24} {small_tot:>9.1f}   (attention + routed switch "
          "excluded -- the two un-fused calls)")
    print()
    print("fused segments (per LAYER, M=1): input norm+HC premix (seg1) | attention |")
    print("  HC combine+ffn premix+gate/top-k+shared (seg2) | routed switch | "
          "MoE combine+HC combine (seg3)")
    hdr = f"{'segment':<8}{'eager':>8}{'mx.compile':>12}{'+K3 kernel':>12}{'sinkhorns':>10}"
    print(hdr)
    print("-" * len(hdr))
    te = tc = tg = 0
    for s in ("seg1", "seg2", "seg3"):
        d = segs[s]
        te += d["eager"]; tc += d["compiled"]; tg += d["gpu_kernel"]
        print(f"{s:<8}{d['eager']:>8}{d['compiled']:>12}{d['gpu_kernel']:>12}"
              f"{d['n_sinkhorn']:>10}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<8}{te:>8}{tc:>12}{tg:>12}{'':>10}")
    pc = 100.0 * (te - tc) / te if te else 0.0
    pg = 100.0 * (te - tg) / te if te else 0.0
    print(f"per-layer reduction:  mx.compile -{te - tc} ({pc:.0f}%)   "
          f"+K3 kernel -{te - tg} ({pg:.0f}%)")
    print(f"per-token (x{nL} layers):  eager {te * nL}  ->  mx.compile {tc * nL}"
          f"  ->  +K3 kernel {tg * nL}")
    print()
    print(f"M=1 latency estimate (spec: 0.02-0.05 ms per removed dispatch; a linear "
          f"upper bound --")
    print(f"  most fused elementwise pipeline, only host-sync-adjacent dispatches are "
          f"fully critical):")
    for label, removed in (("mx.compile", (te - tc) * nL), ("+K3 kernel", (te - tg) * nL)):
        print(f"  {label:<12} -{removed:>5}/tok  ->  {removed * 0.02:.1f}-{removed * 0.05:.1f} "
              f"ms/tok (this tiny {nL}-layer model)")
    receipt = {
        "flag": {"K35": "MTPLX_DSV41_SMALL_STAGES_FUSED"},
        "census": "small_stages_fused",
        "seed": args.seed, "row_cap": args.cap,
        "mlx_version": mx.__version__,
        "n_layers": nL,
        "eager_per_stage_per_token": {
            n: ef["stages"].get(n, {}).get("primitives_per_token", 0.0)
            for n in _SMALL_STAGE_NAMES
        },
        "segments": segs,
        "per_layer_total": {"eager": te, "compiled": tc, "gpu_kernel": tg},
        "sinkhorn_compiled_prims": _SINK_COMPILED_HC4_IT20,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"\nreceipt -> {args.out}")
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default=None, help="JSON receipt path")
    ap.add_argument("--cap", type=int, default=7,
                    help="row cap for the census decode (default 7: prefill s=12 stays eager)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--draft", action="store_true",
                    help="census the DSpark-DIRECT draft block (K33, W65) instead of "
                         "the backbone decode: primitives per draft cycle per stage, "
                         "before/after MTPLX_DSV41_DRAFT_COMPILE")
    ap.add_argument("--small-stages", action="store_true", dest="small_stages",
                    help="census the K35/W91 small-stages fusion: per-layer AR-decode "
                         "dispatch count before/after MTPLX_DSV41_SMALL_STAGES_FUSED "
                         "(+ the GPU K3-Sinkhorn-kernel projection)")
    args = ap.parse_args()

    if args.draft:
        return _draft_main(args)
    if args.small_stages:
        return _small_main(args)

    # before = all off (eager); k22 = attention-tape compile only; after = K22 +
    # K24 window-mask memo (the full W45 attention-compile mode).
    before = _run_full_census(False, args.cap, seed=args.seed)
    k22 = _run_full_census(True, args.cap, seed=args.seed)
    after = _run_full_census(True, args.cap, seed=args.seed, win_memo=True)
    micro = _micro_census(seed=args.seed)

    table = _render_table(before, after)
    micro_table = _render_micro(micro)

    print("=" * 82)
    print("W45 dispatch census -- primitives/token per decode stage (tiny real-structure)")
    print("before=OFF (eager)  after=ON (K22 attn-tape compile + K24 window-mask memo)")
    print("=" * 82)
    print(table)
    print()
    print("Per-chain micro-census (rows=1, eager vs compiled, isolated):")
    print(micro_table)
    print()
    attn = lambda r: sum(v["primitives_per_token"] for k, v in r["stages"].items()
                         if k.startswith("attn."))
    print(f"attention primitives/token:  eager {attn(before):.1f}  ->  K22 {attn(k22):.1f}"
          f"  ->  K22+K24 {attn(after):.1f}")
    print(f"total primitives/token:      eager {before['total_primitives_per_token']:.1f}"
          f"  ->  K22 {k22['total_primitives_per_token']:.1f}"
          f"  ->  K22+K24 {after['total_primitives_per_token']:.1f}")
    print(f"(K24 memoizes the ~11-node window mask, computing it once/forward instead of"
          f" once/layer: -{attn(k22) - attn(after):.0f} attn prim/tok on 8 layers,"
          f" ~11 x (n_layers-1) at scale)")

    receipt = {
        "flags": {"K22": "MTPLX_DSV41_ATTN_COMPILE", "K24": "MTPLX_DSV41_ATTN_WIN_MEMO"},
        "row_cap": args.cap,
        "seed": args.seed,
        "mlx_version": mx.__version__,
        "before_off": before,
        "k22_compile_only": k22,
        "after_k22_k24": after,
        "micro_census": micro,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"\nreceipt -> {args.out}")
    return receipt


def _draft_main(args):
    """K33/W65 draft-block census: primitives per draft cycle per stage, before /
    after ``MTPLX_DSV41_DRAFT_COMPILE`` (the DSpark-DIRECT drafter tape collapse)."""
    before = _run_draft_census(False, seed=args.seed)
    after = _run_draft_census(True, seed=args.seed)

    table = _render_table(before, after)
    print("=" * 82)
    print("W65 DSpark draft-block dispatch census -- primitives per draft cycle per stage")
    print("before=OFF (eager)  after=ON (MTPLX_DSV41_DRAFT_COMPILE: K33 tape collapse)")
    print("(tiny real-structure DSpark head: 3 stages, block_size 4, resident 8-expert top-2 MoE)")
    print("=" * 82)
    print(table)
    print()

    def _grp(r, pred):
        return sum(v["primitives_per_token"] for k, v in r["stages"].items() if pred(k))

    attn = lambda r: _grp(r, lambda k: k.startswith("dspark.attn."))
    hc = lambda r: _grp(r, lambda k: k.startswith("dspark.hc."))
    moe = lambda r: _grp(r, lambda k: k.startswith("moe."))
    mark = lambda r: _grp(r, lambda k: k in ("dspark.markov", "dspark.confidence"))
    tot = lambda r: r["total_primitives_per_token"]
    print(f"attention prep primitives/cycle:  eager {attn(before):.1f}  ->  K33 {attn(after):.1f}")
    print(f"Hyper-Connection primitives/cycle: eager {hc(before):.1f}  ->  K33 {hc(after):.1f}")
    print(f"MoE (gate/switch/combine)/cycle:   eager {moe(before):.1f}  ->  K33 {moe(after):.1f}")
    print(f"markov + confidence /cycle:        eager {mark(before):.1f}  ->  K33 {mark(after):.1f}")
    print(f"TOTAL primitives/draft cycle:      eager {tot(before):.1f}  ->  K33 {tot(after):.1f}"
          f"  (delta {tot(before) - tot(after):.1f})")

    receipt = {
        "flag": {"K33": "MTPLX_DSV41_DRAFT_COMPILE"},
        "census": "dspark_draft_block",
        "seed": args.seed,
        "mlx_version": mx.__version__,
        "before_off": before,
        "after_on": after,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"\nreceipt -> {args.out}")
    return receipt


if __name__ == "__main__":
    main()
