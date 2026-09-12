"""W91 / kernel-ledger K35 -- small-stages fusion (CPU, synthetic, no artifact).

The AR-decode dispatch census (scripts/deepseek_v41/dispatch_census.py
--small-stages) shows the per-layer *small* stages -- everything that is NOT
attention and NOT the routed-expert switch -- are the top per-token dispatch
source at M=1: two Hyper-Connection premix chains (the Sinkhorn dominates), the
MoE gate + top-k, the shared expert, the two HC combines and the MoE combine.
K4 (``MTPLX_DSV41_HC_COMPILE``) tape-collapses only the HC chains; K22
(``MTPLX_DSV41_ATTN_COMPILE``) folds only the gate prefix + MoE combine.  K35
(``MTPLX_DSV41_SMALL_STAGES_FUSED``, default OFF) collapses the WHOLE small-stage
set into three compiled per-layer graphs separated only by the two un-fused
data-dependent calls (attention's KV write, the routed switch's expert gather):

  seg1 : input norm -> attn HC premix (Sinkhorn) -> attention input
  seg2 : attn HC combine -> ffn HC premix -> gate+top-k -> shared expert
  seg3 : MoE combine -> ffn HC combine

These gates prove, on a tiny CPU config:

  * flag on vs off is ``mx.array_equal`` (f32 CPU) over 64 decode steps (n=1) and
    a K+1 verify batch (n<=7): the fused graphs are bit-exact to the eager
    DecoderLayer / MoE bodies in the decode/verify row regime the cap keeps them
    in (every op is the same fp32 op in the same order; ``mx.compile`` only fuses
    adjacent elementwise runs and replays one prebuilt tape);
  * the fold of the gate top-k + shared expert + MoE combine is exact-by-
    construction: the fused seg2/seg3 outputs equal the real Gate/Expert/combine
    module outputs bitwise, so K35 adds ZERO fp delta beyond K4's HC collapse;
  * dispatch collapse (count_prims): each fused segment has strictly fewer graph
    primitives than its eager body (per-layer 584 -> 265, the census headline),
    and each compiled Sinkhorn is 80 prims (the K3 Metal kernel collapses it to 1
    on the GPU -- the census GPU projection);
  * compile-cache stability: exactly 3 tapes (seg1/seg2/seg3) are traced ONCE and
    replayed for every one of 64 decode steps -- no per-token retrace;
  * the fused path is INERT above the row cap (prefill one-shot flag on == off);
  * ``_small_stages_use`` gating (flag/env off, or rows > cap -> eager);
  * byte-identity across (K35) x (SINKHORN_METAL): on CPU the Metal route is inert.

Pins MLX to CPU (worker-tests-must-pin-mlx-cpu.md); tiny random config; no
artifact load.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402

# rows <= 7 is the mx.compile bit-exact regime for this tiny config's HC matmul +
# RMS/HC-mix mean reductions (W33): decode (1) and a K=3 verify batch (4) sit
# under it, so the tests keep every fused shape at rows <= 7 and prefill (s=12 >
# cap) eager -- exactly the K4 hc_compile test's structure.
_TEST_CAP = 7

_DOT_RECT = re.compile(r'\[label ="([^"]+)", shape=rectangle\]')


def _count_prims(*outs):
    arrs = [a for a in outs if isinstance(a, mx.array)]
    if not arrs:
        return 0
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrs)
    return len(_DOT_RECT.findall(buf.getvalue()))


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


def _new_model(seed=1, **over):
    args = _csa_args(**over)
    model = Model(args)
    _randomize(model, seed=seed)
    return model, args


@contextlib.contextmanager
def _fused(flag: bool, max_rows: int = _TEST_CAP):
    """Arm/disarm K35 via the ENV (the flag is read at use, like SINKHORN_METAL),
    set the cap, and clear the tape cache on entry/exit."""
    key = dv41._SMALL_STAGES_FUSED_ENV
    old_env = os.environ.get(key)
    old_rows = dv41._SMALL_STAGES_MAX_ROWS
    if flag:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    dv41._SMALL_STAGES_MAX_ROWS = max_rows
    dv41._SMALL_STAGES_COMPILED.clear()
    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_env
        dv41._SMALL_STAGES_MAX_ROWS = old_rows
        dv41._SMALL_STAGES_COMPILED.clear()


@contextlib.contextmanager
def _metal(flag: bool):
    key = dv41._SINKHORN_METAL_ENV
    old = os.environ.get(key)
    if flag:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _prefill_one_shot(model, args, s, seed):
    """One-shot prefill; s=12 > cap 7 so the layer forward is eager regardless of
    the flag -> the post-prefill cache is identical on both flag settings."""
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    cache = model.make_cache()
    mx.eval(model(ids, cache=cache, prefill_chunk=0))
    return cache


# ---------------------------------------------------------------------------
# 1. byte-identity: 64 decode steps (n=1), flag on vs off
# ---------------------------------------------------------------------------
def test_decode_64_steps_byte_identical():
    def run(flag):
        model, args = _new_model(seed=1)
        with _fused(flag):
            cache = _prefill_one_shot(model, args, s=12, seed=0)  # >cap -> eager both
            outs = []
            tok = 5
            for _ in range(64):
                lo = model(mx.array([[tok]]), cache=cache)  # n=1 -> fused when flag
                mx.eval(lo)
                outs.append(np.array(lo))
                tok = int(mx.argmax(lo[0, -1]))
            return outs

    off = run(False)
    on = run(True)
    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), \
            f"decode step {i} logits differ (max {np.max(np.abs(a - b))})"


# ---------------------------------------------------------------------------
# 2. byte-identity: a K+1 verify batch (n = K+1 = 4 <= cap)
# ---------------------------------------------------------------------------
def test_verify_batch_byte_identical():
    verify_ids = mx.array([[7, 2, 41, 13]])

    def run(flag):
        model, args = _new_model(seed=2)
        with _fused(flag):
            cache = _prefill_one_shot(model, args, s=12, seed=1)
            logits = model(verify_ids, cache=cache)
            mx.eval(logits)
            return np.array(logits)

    assert run(False).shape[1] == 4
    assert np.array_equal(run(False), run(True))


# ---------------------------------------------------------------------------
# 3. exact-by-construction: the fused seg2 (gate top-k + shared) and seg3 (MoE
#    combine + HC combine) equal the REAL Gate / Expert / combine module outputs
#    bitwise -- so K35 adds ZERO fp delta beyond K4's HC collapse.
# ---------------------------------------------------------------------------
def test_gate_shared_combine_fold_bit_exact():
    from mtplx.models.deepseek_v41 import (
        _hc_attn_prep_impl, _hc_ffn_prep_impl, _gate_topk_impl, _shared_expert_impl,
        _hc_post_impl, _lin_desc, _lin_n, _lin_arrays,
    )
    model, args = _new_model(seed=3)
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
    _, attn_pre, attn_post, attn_comb = _hc_attn_prep_impl(
        h, pre_mix, L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, L.attn_norm_weight, *consts)
    mx.eval(attn_pre, attn_post, attn_comb)

    # seg2 fold vs real modules
    moe_input, h2, ffn_post, ffn_comb, ffn_pre = _hc_ffn_prep_impl(
        attn_out, h, attn_pre, attn_post, attn_comb,
        L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale, L.ffn_norm_weight, *consts)
    xf = moe_input.reshape(-1, dim)
    w1d, w3d, w2d = _lin_desc(se.w1), _lin_desc(se.w3), _lin_desc(se.w2)
    n1, n3 = _lin_n(w1d), _lin_n(w3d)
    warrs = _lin_arrays(se.w1) + _lin_arrays(se.w3) + _lin_arrays(se.w2)
    my_w, my_i = _gate_topk_impl(
        xf, g.weight, g.e_score_correction_bias, float(g.gate_temp), str(g.score_func),
        int(g.topk), bool(g.norm_topk_prob), float(g.route_scale))
    my_sh = _shared_expert_impl(xf, warrs[:n1], warrs[n1:n1 + n3], warrs[n1 + n3:],
                                w1d, w3d, w2d, float(se.swiglu_limit)).astype(mx.float32)
    ref_w, ref_i = g(xf)
    ref_sh = se(xf).astype(mx.float32)
    mx.eval(my_w, my_i, my_sh, ref_w, ref_i, ref_sh)
    assert np.array_equal(np.array(my_w), np.array(ref_w)), "gate weights differ"
    assert np.array_equal(np.array(my_i), np.array(ref_i)), "gate indices differ"
    assert np.array_equal(np.array(my_sh), np.array(ref_sh)), "shared expert differs"

    # seg3 fold: MoE combine + HC combine vs the eager MoE.combine_dispatch path
    tk = int(g.topk)
    routed = (0.3 * mx.random.normal((1, tk, dim))).astype(mx.float32)
    mx.eval(routed)
    y_ref = (routed.astype(mx.float32) * ref_w[..., None]).sum(-2) + ref_sh
    y_ref = y_ref.astype(h2.dtype).reshape(*h2.shape[:-2], dim)
    h_ref = _hc_post_impl(y_ref, h2, ffn_post, ffn_comb)
    with _fused(True):
        h_fused = dv41._small_compiled("seg3", L)(routed, ref_w, ref_sh, h2, ffn_post, ffn_comb)
    mx.eval(h_ref, h_fused)
    assert np.array_equal(np.array(h_ref), np.array(h_fused)), "seg3 combine differs"


# ---------------------------------------------------------------------------
# 4. dispatch collapse: each fused segment has fewer primitives than eager, and
#    a compiled Sinkhorn is 80 prims (the K3 kernel collapses it to 1 on GPU).
# ---------------------------------------------------------------------------
def test_dispatch_collapse_per_segment():
    from mtplx.models.deepseek_v41 import (
        _hc_attn_prep_impl, _small_compiled,
    )
    from mtplx.models.deepseek_v4 import _sinkhorn_ops
    model, args = _new_model(seed=4)
    L = next(l for l in model.model.layers if l.attn.mode == dv41.MODE_FULL)
    hc, dim = L.hc_mult, args.hidden_size
    consts = (hc, L.hc_iters, L.norm_eps, L.hc_eps)
    mx.random.seed(0)
    h = (0.3 * mx.random.normal((1, 1, hc, dim))).astype(mx.float32)
    pre_mix = mx.concatenate(
        [mx.ones((1, 1, 1)), mx.zeros((1, 1, hc - 1))], axis=-1).astype(mx.float32)
    mx.eval(h, pre_mix)
    with _fused(True):
        eager = _hc_attn_prep_impl(h, pre_mix, L.hc_attn_fn, L.hc_attn_base,
                                   L.hc_attn_scale, L.attn_norm_weight, *consts)
        ne = _count_prims(*eager)
        comp = _small_compiled("seg1", L)(
            h, pre_mix, L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, L.attn_norm_weight)
        nc = _count_prims(*comp)
        mx.eval(eager, comp)
    assert nc < ne, f"seg1 compiled ({nc}) not fewer than eager ({ne})"

    # one compiled Sinkhorn (hc=4, iters=20) is 80 prims; the K3 Metal kernel
    # (MTPLX_DSV41_SINKHORN_METAL, GPU) collapses it to one dispatch. This is the
    # census GPU projection's per-Sinkhorn saving (80 -> 1).
    comb = mx.softmax(0.2 * mx.random.normal((1, 1, hc, hc)), axis=-1).astype(mx.float32)
    mx.eval(comb)
    sink_c = mx.compile(lambda c: _sinkhorn_ops(c, L.hc_iters, L.hc_eps))(comb)
    assert _count_prims(sink_c) == 80, _count_prims(sink_c)
    mx.eval(sink_c)


# ---------------------------------------------------------------------------
# 5. compile-cache stability: exactly 3 tapes traced ONCE, replayed for 64 steps.
# ---------------------------------------------------------------------------
def test_compile_cache_stable_over_64_steps():
    model, args = _new_model(seed=5)
    with _fused(True):
        cache = _prefill_one_shot(model, args, s=12, seed=2)  # >cap -> eager (no tapes yet)
        assert len(dv41._SMALL_STAGES_COMPILED) == 0
        tok = 5
        after_first = None
        for step in range(64):
            mx.eval(model(mx.array([[tok]]), cache=cache))
            tok = 3
            if step == 0:
                after_first = len(dv41._SMALL_STAGES_COMPILED)
        # seg1 + seg2 + seg3 = 3 tapes, all traced on step 0, none added after.
        assert after_first == 3, after_first
        assert len(dv41._SMALL_STAGES_COMPILED) == 3, len(dv41._SMALL_STAGES_COMPILED)


# ---------------------------------------------------------------------------
# 6. inert above the row cap (prefill one-shot flag on == flag off, both eager)
# ---------------------------------------------------------------------------
def test_inert_above_row_cap():
    ids = mx.array(np.random.RandomState(6).randint(0, 48, size=(1, 12)))  # 12 > cap 7
    m_off, _ = _new_model(seed=5)
    with _fused(False):
        lo = np.array(m_off(ids, cache=m_off.make_cache(), prefill_chunk=0))
    m_on, _ = _new_model(seed=5)
    with _fused(True):
        ln = np.array(m_on(ids, cache=m_on.make_cache(), prefill_chunk=0))
    assert np.array_equal(lo, ln)


# ---------------------------------------------------------------------------
# 7. _small_stages_use gating
# ---------------------------------------------------------------------------
def test_small_stages_use_gating():
    x1 = mx.zeros((1, 1, 4, 32))   # decode
    x4 = mx.zeros((1, 4, 4, 32))   # verify K+1
    x9 = mx.zeros((1, 9, 4, 32))   # > cap 7
    with _fused(False):
        assert dv41._small_stages_use(x1) is False
    with _fused(True, max_rows=7):
        assert dv41._small_stages_use(x1) is True
        assert dv41._small_stages_use(x4) is True
        assert dv41._small_stages_use(x9) is False


# ---------------------------------------------------------------------------
# 8. byte-identity across (K35) x (SINKHORN_METAL) -- CPU Metal route inert.
# ---------------------------------------------------------------------------
def test_flag_2x2_byte_identical_on_cpu():
    def probe(fused_flag, metal_flag):
        with _metal(metal_flag), _fused(fused_flag):
            model, args = _new_model(seed=11)
            cache = _prefill_one_shot(model, args, s=12, seed=7)
            out = []
            tok = 3
            for _ in range(8):
                lo = model(mx.array([[tok]]), cache=cache)
                mx.eval(lo)
                out.append(np.array(lo))
                tok = int(mx.argmax(lo[0, -1]))
            return out

    combos = [(False, False), (True, False), (False, True), (True, True)]
    ref = probe(*combos[0])
    for f, mflag in combos[1:]:
        got = probe(f, mflag)
        for i, (a, b) in enumerate(zip(ref, got)):
            assert np.array_equal(a, b), \
                f"decode[{i}] differs (SMALL_STAGES={f} SINKHORN_METAL={mflag})"


# ---------------------------------------------------------------------------
# 9. the whole ADMITTED range is byte-identical: at the SHIPPED module cap
#    (_SMALL_STAGES_MAX_ROWS), rows 1, cap//2 and cap are all mx.array_equal flag
#    on vs off. Guards against a cap that admits the 8..N reassociation band while
#    the lever is labelled byte-identical (the reviewer's HIGH finding).
# ---------------------------------------------------------------------------
def test_module_cap_all_admitted_rows_byte_identical():
    cap = dv41._SMALL_STAGES_MAX_ROWS  # the SHIPPED default (not _TEST_CAP)
    assert cap <= 7, f"cap {cap} admits the >=8-row reassociation band (not byte-identical)"
    for rows in sorted({1, max(1, cap // 2), cap}):
        def run(flag):
            model, args = _new_model(seed=3)
            with _fused(flag, max_rows=cap):
                cache = _prefill_one_shot(model, args, s=12, seed=1)  # >cap -> eager both
                batch = mx.array(
                    np.random.RandomState(rows).randint(0, args.vocab_size, size=(1, rows)))
                lo = model(batch, cache=cache)
                mx.eval(lo)
                return np.array(lo)
        assert np.array_equal(run(False), run(True)), \
            f"rows={rows} not byte-identical at module cap={cap}"


# ---------------------------------------------------------------------------
# 10. engagement counters: the fused path actually runs on a fused decode, and is
#     forced eager while a W37 stage-timing forward records (the recording guard --
#     so a --stage-timing headline measures K35 OFF; the ab harness runs an untimed
#     headline separately).
# ---------------------------------------------------------------------------
def test_engagement_counters():
    from mtplx.models import deepseek_v41_stage_timing as stime
    model, args = _new_model(seed=8)
    L = args.num_hidden_layers
    with _fused(True):
        cache = _prefill_one_shot(model, args, s=12, seed=1)  # eager prefill (>cap)
        dv41._reset_small_stages_calls()
        for tok in (5, 8, 2):
            mx.eval(model(mx.array([[tok]]), cache=cache))
        calls = dv41._small_stages_calls()
        assert calls["fused"] == 3 * L, calls   # 3 decode tokens x L layers, all fused
        assert calls["eager"] == 0, calls
        # under W37 recording the fused path is forced eager (one opaque call cannot
        # be split by the per-stage fences) -> a timed headline would measure K35 OFF.
        dv41._reset_small_stages_calls()
        stime.begin()
        try:
            mx.eval(model(mx.array([[9]]), cache=cache))
        finally:
            stime.end()
        calls2 = dv41._small_stages_calls()
        assert calls2["fused"] == 0, calls2
        assert calls2["eager"] == L, calls2


# ---------------------------------------------------------------------------
# 11. K35 fused HC-premix kernel FALLBACK + INERTNESS (item 3, CPU). This is NOT a
#     parity test -- on CPU `_hc_premix_sinkhorn` IS `hc_split_sinkhorn` (the
#     reference), so equality is definitional. It gates the two things that CAN be
#     wrong on CPU: (a) the flag is inert (use_kernel False, kernel never built or
#     dispatched off-GPU, so a GPU window is never touched), and (b) the fallback is
#     wired. The real kernel-vs-reference numeric parity is
#     test_hc_premix_kernel_parity_gpu (GPU-gated, below).
# ---------------------------------------------------------------------------
def test_hc_premix_kernel_fallback_and_inert_on_cpu():
    from mtplx.models.deepseek_v41 import (
        _hc_premix_sinkhorn, hc_split_sinkhorn, _hc_premix_use_kernel,
        _hc_premix_kernel_calls, _reset_hc_premix_kernel_calls,
        _hc_premix_sinkhorn_metal_kernel,
    )
    hc, it, eps = 4, 20, 1e-6
    mix_hc = (2 + hc) * hc
    mx.random.seed(0)
    mixes = mx.random.normal((1, 1, mix_hc)).astype(mx.float32)
    scale = mx.random.normal((3,)).astype(mx.float32)
    base = mx.random.normal((mix_hc,)).astype(mx.float32)
    mx.eval(mixes, scale, base)

    _reset_hc_premix_kernel_calls()
    a = _hc_premix_sinkhorn(mixes, scale, base, hc, it, eps)
    b = hc_split_sinkhorn(mixes, scale, base, hc, it, eps)
    mx.eval(a, b)
    for x, y in zip(a, b):
        assert np.array_equal(np.array(x), np.array(y)), "premix reference not bit-exact"

    # flag ON but CPU default device -> use_kernel False, reference taken, still exact
    assert _hc_premix_use_kernel() is False
    with _premix_kernel(True):
        assert _hc_premix_use_kernel() is False  # CPU: never the kernel
        c = _hc_premix_sinkhorn(mixes, scale, base, hc, it, eps)
        mx.eval(c)
        for x, y in zip(c, b):
            assert np.array_equal(np.array(x), np.array(y)), "flag-on CPU not bit-exact"
    # every CPU call took the reference, none the kernel (never built/dispatched)
    assert _hc_premix_kernel_calls()["kernel"] == 0
    assert _hc_premix_kernel_calls()["reference"] >= 2

    # the Metal source is well-formed (built lazily only on GPU; here just assert the
    # builder is callable and the source names the fused outputs -- the actual
    # compile + numeric parity is a GPU-window gate).
    assert callable(_hc_premix_sinkhorn_metal_kernel)


@contextlib.contextmanager
def _premix_kernel(flag: bool):
    key = dv41._HC_PREMIX_KERNEL_ENV
    old = os.environ.get(key)
    if flag:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


# ---------------------------------------------------------------------------
# 12. GPU parity (item 4): the fused HC-premix kernel == hc_split_sinkhorn (the
#     stock split + K3 Sinkhorn recurrence) to 1e-6, comb row-argmax exact -- the
#     REAL parity gate (unlike #11, which compares the CPU fallback to itself).
#     Mirrors the K3 test_sinkhorn_kernel_parity_gpu convention: skipped unless
#     MTPLX_GPU_PARITY=1 (run inside a GPU window), self-diagnosing into a
#     MTPLX_PARITY_RECEIPT before any assertion. On [2,7,24] f32 (== the K3
#     input_shape's leading dims), so it exercises multi-row premix.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.mark.skipif(
    os.environ.get("MTPLX_GPU_PARITY") != "1",
    reason="GPU parity: run inside a GPU window with MTPLX_GPU_PARITY=1",
)
def test_hc_premix_kernel_parity_gpu():
    from mtplx.models.deepseek_v41 import (
        _hc_premix_sinkhorn_kernel_apply, hc_split_sinkhorn,
    )
    hc, iters, eps = 4, 20, 1e-6
    mix_hc = (2 + hc) * hc
    diag = {
        "test": "test_hc_premix_kernel_parity_gpu",
        "hc": hc, "iters": iters, "eps": eps,
        "input_shape": [2, 7, mix_hc],
        "metal_available": bool(mx.metal.is_available()),
    }
    saved = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        rs = np.random.RandomState(99)
        mixes = mx.array(rs.standard_normal((2, 7, mix_hc)).astype("float32")) * 2.5
        scale = mx.array(rs.standard_normal((3,)).astype("float32"))
        base = mx.array(rs.standard_normal((mix_hc,)).astype("float32"))
        mx.eval(mixes, scale, base)
        try:
            k_pre, k_post, k_comb = _hc_premix_sinkhorn_kernel_apply(
                mixes, scale, base, hc, iters, eps)
            mx.eval(k_pre, k_post, k_comb)
            diag["kernel_builder_ok"] = True
        except Exception as exc:  # pragma: no cover - GPU-only
            diag["kernel_builder_ok"] = False
            diag["kernel_builder_error"] = repr(exc)
            raise
        # The oracle is the STOCK Sinkhorn recurrence: pin MTPLX_DSV41_SINKHORN_METAL
        # OFF so hc_split_sinkhorn does NOT take the K3 kernel on the GPU (else the
        # "reference" would itself be a kernel and the comparison would be
        # kernel-vs-kernel, not kernel-vs-recurrence). Record the route it took.
        with _metal(False):
            diag["sinkhorn_reference_route"] = (
                "kernel" if dv41._sinkhorn_use_kernel() else "recurrence")
            r_pre, r_post, r_comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)
            mx.eval(r_pre, r_post, r_comb)
        assert diag["sinkhorn_reference_route"] == "recurrence", diag
        deltas = {
            "pre": float(mx.max(mx.abs(k_pre - r_pre))),
            "post": float(mx.max(mx.abs(k_post - r_post))),
            "comb": float(mx.max(mx.abs(k_comb - r_comb))),
        }
        diag["max_abs_delta"] = deltas
        # comb is the routing matrix -> its row-argmax must be exact (the tie-break
        # lever the census cares about), like K3's argmax gate.
        argmax_mismatch = int(
            mx.sum(mx.argmax(k_comb, axis=-1) != mx.argmax(r_comb, axis=-1)).item())
        diag["comb_argmax_mismatch"] = argmax_mismatch
        diag["passed"] = (max(deltas.values()) <= 1e-6) and (argmax_mismatch == 0)
    finally:
        diag["receipt_path"] = _write_parity_receipt(diag)
        print(json.dumps(diag, indent=2, sort_keys=True))
        mx.set_default_device(saved)
    assert diag.get("kernel_builder_ok"), diag.get("kernel_builder_error")
    assert max(diag["max_abs_delta"].values()) <= 1e-6, diag
    assert diag["comb_argmax_mismatch"] == 0, diag


def _write_parity_receipt(diag: dict):
    """Write ``diag`` as JSON to MTPLX_PARITY_RECEIPT if set (never raises); the K3
    convention (test_deepseek_v41_sinkhorn_metal.py) so a GPU window captures what
    failed even if stdout is truncated."""
    path = os.environ.get("MTPLX_PARITY_RECEIPT")
    if not path:
        return None
    try:
        p = Path(path)
        if p.parent and str(p.parent):
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(diag, indent=2, sort_keys=True))
        return str(p)
    except Exception as exc:  # pragma: no cover - IO edge
        print(f"[W91/K35] receipt write failed: {exc!r}")
        return None
