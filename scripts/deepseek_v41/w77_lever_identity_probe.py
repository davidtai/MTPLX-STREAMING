#!/usr/bin/env python3
"""W77 -- CPU fake-model probe: which cell16k lever breaks byte-identity between
the 1-row AR (M=1) decode forward and the K+1-row (M=4) verify forward.

The DSpark-DIRECT greedy stream diverged from AR at index 228 under the ``cell16k``
arm (window 29b, real 16K model).  The greedy verify is authoritative, so a
divergence can only be a greedy argmax *flip* driven by the target forward
returning slightly different logits at the same committed context depending on the
row count (M=1 AR decode vs M=K+1 verify).  This probe reproduces the M=1-vs-M=4
delta *in exact CPU fp32 arithmetic* on a shrunk seeded model (the real V4.1 Model
class, tiny dims, random weights -- NOT the 376 GB artifact), toggling each cell16k
lever, so a LOGIC bug (a wrong row/position, a mis-unsorted gather) shows as a LARGE
delta even without Metal, while a pure Metal/bf16 kernel-dispatch reassociation
(rounding-class -> tie flip) shows as ~0 on CPU.

Run (CPU, no GPU, tiny):
    PYTHONPATH=<worktree> nice -n 19 <venv>/bin/python3 \
        scripts/deepseek_v41/w77_lever_identity_probe.py

Comparison geometry (mirrors the AR-vs-verify block): a shared prefill of
``ctx[:-K-1]`` seeds an identical cache; then the M=1 lane feeds the last K+1
tokens as sequential 1-row decode steps while the M=K+1 lane feeds them as one
verify block.  Both predict the SAME final position; ``max |Δlogit|`` between the
two is the divergence magnitude that lever contributes.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os

import numpy as np

# --- lever env map (cell16k = every one of these) ---------------------------
# name -> (env dict applied for the run, head must be bf16, note on M=4 engagement)
BOOL_ENVS = {
    "MTPLX_DSV41_PREFILL_LAYER_MAJOR",
    "MTPLX_DSV41_PREFILL_DENSE_EXPERTS",
    "MTPLX_DSV41_SELECTED_KEYS",
    "MTPLX_DSV41_KV_CHUNK_GROW",
    "MTPLX_DSV41_LAYOUT_FIX",
    "MTPLX_DSV41_SINKHORN_METAL",
    "MTPLX_DSV41_ATTN_COMPILE",
    "MTPLX_DSV41_ATTN_WIN_MEMO",
}

# Levers whose flags are frozen at import as module globals in deepseek_v41 (read
# through the global at call time); set the global, not just the env.
MODULE_GLOBAL_LEVERS = {
    "MTPLX_DSV41_ATTN_COMPILE": "_ATTN_COMPILE",
    "MTPLX_DSV41_ATTN_WIN_MEMO": "_ATTN_WIN_MEMO",
}

# (label, env, head_bf16, engages_at_m4_verify, note)
LEVERS = [
    ("layer_major", {"MTPLX_DSV41_PREFILL_LAYER_MAJOR": "1"}, False, False,
     "prefill schedule only; the verify is one forward, no chunk schedule"),
    ("prefill_dense", {"MTPLX_DSV41_PREFILL_DENSE_EXPERTS": "1",
                        "MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS": "1"}, False, False,
     "streamed-switch PREFILL-phase path; verify routes DECODE-phase and the "
     "tiny double uses resident experts -> inert here (and gated off on the box)"),
    ("score_path=lean", {"MTPLX_DSV41_PREFILL_SCORE_PATH": "lean"}, False, True,
     "rows>1 attention score path; the M=4 verify is rows>1, M=1 decode is not"),
    ("selected_keys", {"MTPLX_DSV41_SELECTED_KEYS": "1"}, False, True,
     "rows>1 selected-key gather+softmax; engages on the M=4 verify, not M=1 "
     "decode -- 'never bit-identical' (float reassociation) per K30 note"),
    ("kv_chunk_grow", {"MTPLX_DSV41_KV_CHUNK_GROW": "1"}, False, True,
     "chunk-grown KV append backing; the M=4 verify appends 4 rows at once"),
    ("layout_fix", {"MTPLX_DSV41_LAYOUT_FIX": "1",
                     "MTPLX_DSV41_LAYOUT_FIX_MIN_ROWS": "1"}, False, False,
     "streamed-switch sorted routed gather (min_rows 2048; verify ~8 rows); "
     "inert on the resident double, gated off on the box"),
    ("head=bf16", {"MTPLX_DSV41_HEAD_MODE": "bf16"}, True, True,
     "bf16 head GEMV; the M=1 and M=4 head matmuls take different kernels"),
    ("sinkhorn_metal", {"MTPLX_DSV41_SINKHORN_METAL": "1"}, False, False,
     "Metal-only kernel; falls back to the byte-identical recurrence on CPU"),
    ("attn_compile", {"MTPLX_DSV41_ATTN_COMPILE": "1"}, False, True,
     "mx.compile attention tapes fire at rows<=32 (M=1 and M=4); one tape per "
     "shape, each bit-identical to eager"),
    ("attn_win_memo", {"MTPLX_DSV41_ATTN_WIN_MEMO": "1"}, False, True,
     "memoized window-attend mask; pure host-dispatch reuse"),
]

CELL16K_ENV = {
    "MTPLX_DSV41_PREFILL_LAYER_MAJOR": "1",
    "MTPLX_DSV41_PREFILL_DENSE_EXPERTS": "1",
    "MTPLX_DSV41_PREFILL_SCORE_PATH": "lean",
    "MTPLX_DSV41_SELECTED_KEYS": "1",
    "MTPLX_DSV41_KV_CHUNK_GROW": "1",
    "MTPLX_DSV41_LAYOUT_FIX": "1",
    "MTPLX_DSV41_HEAD_MODE": "bf16",
    "MTPLX_DSV41_SINKHORN_METAL": "1",
    "MTPLX_DSV41_ATTN_COMPILE": "1",
    "MTPLX_DSV41_ATTN_WIN_MEMO": "1",
}

DIM = 32
N_LAYERS = 4
VOCAB = 48


def _tiny_args():
    from mtplx.models.deepseek_v41 import ModelArgs

    return ModelArgs(
        vocab_size=VOCAB, hidden_size=DIM, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=8,
        q_lora_rank=16, o_lora_rank=8, o_groups=2, moe_intermediate_size=16,
        n_routed_experts=8, num_experts_per_tok=2, sliding_window=8, window_size=8,
        hc_mult=4, hc_sinkhorn_iters=2, scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5, swiglu_limit=0.0, n_mtp_layers=3,
        dspark_block_size=4, dspark_noise_token_id=VOCAB - 1,
        dspark_target_layer_ids=[1, 2, 3], dspark_markov_rank=12,
        dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )


def build_seeded_model(seed=0, head_bf16=False, dtype="fp32"):
    """The real V4.1 Model on tiny dims with seeded random weights.

    ``dtype="fp32"`` is the bit-exact CPU lane that catches LOGIC bugs (a genuine
    wrong-row/gather bug shows a large delta even in exact arithmetic).
    ``dtype="bf16"`` casts every weight to bf16 so the whole forward runs in the
    bf16 rounding class the real box uses -- this exhibits the *bf16-class* delta
    each lever contributes (the envelope that flips a near-tie greedy argmax),
    which the fp32 lane rounds away.

    ``head_bf16`` casts the output-head weight to bf16 even in the fp32 lane -- the
    native artifact keeps the head bf16, so HEAD_MODE=bf16's
    ``source.astype(head.weight.dtype)`` only rounds when the weight is bf16.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mtplx.models.deepseek_v41 import Model

    want = mx.bfloat16 if dtype == "bf16" else None
    mx.random.seed(seed)
    model = Model(_tiny_args(), quantize=False, mtp=False)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if value.ndim == 1:
            noise = mx.random.normal(value.shape) * 0.1
            centre = 1.0 if leaf.endswith("norm_weight") or leaf == "scale" else 0.0
            new = noise + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        cast = want if want is not None else value.dtype
        filled.append((name, new.astype(cast)))
    model.update(tree_unflatten(filled))
    if (head_bf16 or want is mx.bfloat16) and getattr(model.head, "weight", None) is not None:
        model.head.weight = model.head.weight.astype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


@contextlib.contextmanager
def lever_env(env: dict):
    """Set lever env keys (+ the two import-frozen module globals) for the block,
    restore exactly afterwards."""
    from mtplx.models import deepseek_v41 as dsv41

    prev_env = {k: os.environ.get(k) for k in env}
    prev_glob = {}
    for k, v in env.items():
        os.environ[k] = v
        g = MODULE_GLOBAL_LEVERS.get(k)
        if g is not None:
            prev_glob[g] = getattr(dsv41, g)
            truthy = str(v).strip().lower() not in ("", "0", "false", "no", "off", "auto")
            setattr(dsv41, g, truthy)
    try:
        yield
    finally:
        for k, old in prev_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        for g, old in prev_glob.items():
            setattr(dsv41, g, old)


def m1_vs_m4(model, ctx, block=4):
    """max |Δlogit| at the final position between the M=1 decode lane and the
    M=(block) verify lane, over an identical shared prefill."""
    import mlx.core as mx

    L = len(ctx)
    assert L > block, "need a prefill prefix"
    prefix = ctx[: L - block]
    tail = ctx[L - block:]

    c1 = model.make_cache()
    model(mx.array([prefix]), cache=c1)          # shared prefill, M=len(prefix)
    logits1 = None
    for t in tail:
        logits1 = model(mx.array([[int(t)]]), cache=c1)   # M=1 decode step
    row1 = logits1[0, -1].astype(mx.float32)

    c4 = model.make_cache()
    model(mx.array([prefix]), cache=c4)          # identical shared prefill
    logits4 = model(mx.array([tail]), cache=c4)  # one M=block verify forward
    row4 = logits4[0, -1].astype(mx.float32)

    mx.eval(row1, row4)
    a = np.asarray(row1)
    b = np.asarray(row4)
    return {
        "max_abs_logit_delta": float(np.max(np.abs(a - b))),
        "argmax_m1": int(np.argmax(a)),
        "argmax_m4": int(np.argmax(b)),
        "argmax_flip": int(np.argmax(a)) != int(np.argmax(b)),
    }


def run_lever(label, env, head_bf16, ctx, seed=0, block=4, dtype="fp32"):
    with lever_env(env):
        model = build_seeded_model(seed=seed, head_bf16=head_bf16, dtype=dtype)
        if os.environ.get("MTPLX_DSV41_HEAD_MODE"):
            model.apply_head_mode()
        out = m1_vs_m4(model, ctx, block=block)
    out["label"] = label
    return out


def sweep(ctx, seed=0, block=4, dtype="fp32"):
    rows = [run_lever("baseline (no levers)", {}, False, ctx, seed, block, dtype)]
    for label, env, head_bf16, engages, _note in LEVERS:
        r = run_lever(label, env, head_bf16, ctx, seed, block, dtype)
        r["engages_at_m4"] = engages
        rows.append(r)
    rows.append(run_lever("cell16k (full stack)", CELL16K_ENV, True, ctx, seed, block, dtype))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ctx-len", type=int, default=20)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--json", action="store_true", help="emit the rows as JSON too")
    p.add_argument("--dtype", choices=("fp32", "bf16", "both"), default="both",
                   help="fp32 = bit-exact bug hunt; bf16 = the box's rounding class")
    args = p.parse_args(argv)

    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(args.seed)
    ctx = [int(x) for x in rng.integers(1, VOCAB - 1, size=args.ctx_len)]

    dtypes = ("fp32", "bf16") if args.dtype == "both" else (args.dtype,)
    all_rows = {}
    for dt in dtypes:
        rows = sweep(ctx, seed=args.seed, block=args.block, dtype=dt)
        all_rows[dt] = rows
        w = max(len(r["label"]) for r in rows)
        lane = ("bit-exact CPU (catches logic bugs)" if dt == "fp32"
                else "bf16 rounding class (the box's envelope)")
        print(f"\nW77 per-lever M=1 vs M=4 logit identity -- {dt} lane: {lane}")
        print(f"(tiny double, ctx={args.ctx_len}, block={args.block}, seed={args.seed})")
        print(f"{'lever':<{w}}  {'max|Δlogit|':>13}  {'argmax_flip':>11}  "
              f"{'m1->m4':>10}  engages_M4")
        print("-" * (w + 52))
        for r in rows:
            flip = "FLIP" if r["argmax_flip"] else "-"
            eng = r.get("engages_at_m4", "-")
            print(f"{r['label']:<{w}}  {r['max_abs_logit_delta']:>13.3e}  {flip:>11}  "
                  f"{r['argmax_m1']:>4}->{r['argmax_m4']:<4}  {eng}")
        print()
    if args.json:
        print(json.dumps(all_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
