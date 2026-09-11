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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default=None, help="JSON receipt path")
    ap.add_argument("--cap", type=int, default=7,
                    help="row cap for the census decode (default 7: prefill s=12 stays eager)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
