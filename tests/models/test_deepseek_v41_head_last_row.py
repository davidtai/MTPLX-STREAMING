"""W29 / K19: head only the last row at prefill (CPU, tiny synthetic configs).

``Model.__call__`` heads the whole prompt by default (``[1, s, vocab]`` logits);
at a 16,384-token prefill that transient is 8.47 GB in f32 and the output GEMM
runs over 16,383 rows the AR path never reads -- decode seeds only from the last
token. ``logits_rows="last"`` (equivalently the runtime's ``logits_keep=1``)
slices the already-final-normed / hyper-connection-merged hidden to the trailing
row *before* the head, dropping both.

These tests pin MLX to CPU, build tiny random models, and prove:

  (a) the last-row path is bit-identical between the two selectors
      (``logits_rows="last"`` == ``logits_keep=1``, ``mx.array_equal``), and the
      surviving row matches the last row of the all-rows logits to ~2e-7 with an
      identical argmax, for one-shot AND W20 chunked prefill.  It is NOT
      bit-identical to the all-rows tail on purpose: the head is a matmul whose
      rounding depends on its row count M, so heading M=1 row (the K19 win) vs
      M=s rows differs by ~1.8e-7 (measured) -- decode's argmax is unaffected.
      The exact guarantee is that the narrowed head equals ``head(h[:, -1:])``;
  (b) the head is invoked exactly ONCE per prefill in chunked mode (not once per
      span), and on the last-row path it projects a single row rather than ``s``;
  (c) ``logits_rows="last"`` leaves the DSpark ``main_hidden`` and the drafted
      logits bit-identical -- the draft reads ``main_hidden``, never the lm head.

No real artifact is loaded; peak RSS stays far under the worker cap.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _csa_args(**over) -> ModelArgs:
    """8-layer CSA2 backbone exercising swa / full-r2 / reuse / reindex modes,
    with a small window (8) and ratio-2/1 groups so tiny chunks straddle both a
    window boundary and a compressor group boundary (mirrors the chunked-prefill
    suite fixture)."""
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


def _mtp_args(vocab: int = 64, **over) -> ModelArgs:
    """5-layer all-SWA backbone plus a 3-stage DSpark head over the last three
    layers (mirrors the DSpark suite fixture)."""
    base = dict(
        vocab_size=vocab, hidden_size=32, num_hidden_layers=5,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=8,
        q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        sliding_window=8, window_size=8, hc_mult=4, hc_sinkhorn_iters=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0,
        n_mtp_layers=3, dspark_block_size=4, dspark_noise_token_id=vocab - 1,
        dspark_target_layer_ids=[2, 3, 4], dspark_markov_rank=12,
        dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )
    base.update(over)
    return ModelArgs(**base)


def _randomize(model, seed=0, scale=0.1):
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")
                              or leaf == "scale"):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = scale * mx.random.normal(arr.shape)
        new.append((name, v.astype(arr.dtype)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


def _ids(vocab, s, seed=0):
    return mx.array(np.random.RandomState(seed).randint(0, vocab, size=(1, s)))


class _CountingHead:
    """Wraps the lm head so ``self.head(x)`` records call count + the row width
    it is asked to project.  A plain callable (not an ``nn.Module``): the model
    only ever calls it, and it must not be swept up by ``parameters()``."""

    def __init__(self, inner, rec):
        self.inner = inner
        self.rec = rec

    def __call__(self, x):
        self.rec["calls"] = self.rec.get("calls", 0) + 1
        self.rec["max_rows"] = max(self.rec.get("max_rows", 0), int(x.shape[1]))
        return self.inner(x)


# --------------------------------------------------------------------------- #
# (a) exactness: last-row logits == last row of all-rows logits
# --------------------------------------------------------------------------- #
_CHUNKS = (1, 3, 5, 7)


def _model():
    args = _csa_args()
    model = Model(args, quantize=False)  # dense head -> unambiguous bit-exactness
    _randomize(model, seed=1)
    return args, model


def test_last_row_equals_last_row_of_all_rows_one_shot():
    args, model = _model()
    s = 12
    ids = _ids(args.vocab_size, s, seed=0)

    full = model(ids, cache=model.make_cache(), prefill_chunk=0)          # [1,s,V]
    last = model(ids, cache=model.make_cache(), prefill_chunk=0,
                 logits_rows="last")                                      # [1,1,V]
    keep = model(ids, cache=model.make_cache(), prefill_chunk=0,
                 logits_keep=1)                                           # [1,1,V]
    mx.eval(full, last, keep)

    assert full.shape == (1, s, args.vocab_size)
    assert last.shape == (1, 1, args.vocab_size)
    # logits_rows="last" is exactly the runtime's logits_keep=1 (same M=1 head).
    assert mx.array_equal(last, keep)
    # The surviving row matches the all-rows tail up to the head's M-dependent
    # GEMM rounding (~2e-7); the decode-relevant argmax is identical.
    assert mx.allclose(last[:, 0, :], full[:, -1, :], atol=1e-5, rtol=0.0)
    assert int(last[:, 0, :].argmax()) == int(full[:, -1, :].argmax())


def test_last_row_equals_last_row_of_all_rows_chunked():
    args, model = _model()
    s = 25
    ids = _ids(args.vocab_size, s, seed=2)

    for chunk in _CHUNKS:
        full = model(ids, cache=model.make_cache(), prefill_chunk=chunk)
        last = model(ids, cache=model.make_cache(), prefill_chunk=chunk,
                     logits_rows="last")
        mx.eval(full, last)
        assert last.shape == (1, 1, args.vocab_size), f"chunk={chunk}"
        # within the SAME (chunked) forward mode the narrowed head reproduces the
        # surviving row up to the head's M-dependent GEMM rounding (~2e-7);
        # argmax identical (the decode seed is unchanged).
        assert mx.allclose(last[:, 0, :], full[:, -1, :], atol=1e-5, rtol=0.0), f"chunk={chunk}"
        assert int(last[:, 0, :].argmax()) == int(full[:, -1, :].argmax()), f"chunk={chunk}"


def test_logits_rows_all_is_the_default_all_rows():
    args, model = _model()
    ids = _ids(args.vocab_size, 10, seed=3)
    default = model(ids, cache=model.make_cache(), prefill_chunk=0)
    explicit = model(ids, cache=model.make_cache(), prefill_chunk=0,
                     logits_rows="all")
    mx.eval(default, explicit)
    assert mx.array_equal(default, explicit)


# --------------------------------------------------------------------------- #
# (b) the head runs once per prefill (not once per span) and narrows to 1 row
# --------------------------------------------------------------------------- #
def test_head_invoked_once_per_chunked_prefill_and_narrows_rows():
    args, model = _model()
    s = 25
    ids = _ids(args.vocab_size, s, seed=4)
    chunk = 3  # -> ceil(25/3) = 9 backbone spans, but ONE head call

    # all-rows: one head call over all s rows
    rec_all: dict = {}
    model.head = _CountingHead(model.head, rec_all)
    out_all = model(ids, cache=model.make_cache(), prefill_chunk=chunk)
    mx.eval(out_all)
    assert rec_all["calls"] == 1, "head must run once, not once per chunk span"
    assert rec_all["max_rows"] == s

    # last-row: still one head call, but over a single row
    rec_last: dict = {}
    model.head.rec = rec_last
    out_last = model(ids, cache=model.make_cache(), prefill_chunk=chunk,
                     logits_rows="last")
    mx.eval(out_last)
    assert rec_last["calls"] == 1
    assert rec_last["max_rows"] == 1, "last-row mode must head a single row"


def test_head_skipped_entirely_when_emit_logits_false():
    args, model = _model()
    ids = _ids(args.vocab_size, 8, seed=5)
    rec: dict = {}
    model.head = _CountingHead(model.head, rec)
    out = model(ids, cache=model.make_cache(), prefill_chunk=0, emit_logits=False)
    assert out is None
    assert rec.get("calls", 0) == 0, "emit_logits=False must not touch the head"


# --------------------------------------------------------------------------- #
# (c) DSpark draft unchanged: it consumes main_hidden, never the lm head
# --------------------------------------------------------------------------- #
def test_dspark_main_hidden_and_draft_unchanged_under_last_row():
    mx.random.seed(0)
    args = _mtp_args()
    model = Model(args, quantize=False, mtp=True)
    _randomize(model, seed=0)
    ids = _ids(args.vocab_size, 9, seed=7)

    logits_full, mh_full = model(ids, return_hidden=True)                  # all rows
    logits_last, mh_last = model(ids, return_hidden=True, logits_rows="last")
    mx.eval(logits_full, mh_full, logits_last, mh_last)

    assert logits_last.shape == (1, 1, args.vocab_size)
    # narrowing the lm head does not touch the target-layer hiddens the draft reads
    assert mx.array_equal(mh_full, mh_last)
    # and the surviving logits row still matches the all-rows tail
    assert mx.array_equal(logits_last[:, 0, :], logits_full[:, -1, :])

    # drive the actual DSpark draft from each main_hidden: identical draft logits
    dl_full, _ = model.mtp_forward(mh_full, ids[:, -1:], mtp_cache=model.make_mtp_cache(),
                                   return_hidden=True, mtp_depth=1)
    dl_last, _ = model.mtp_forward(mh_last, ids[:, -1:], mtp_cache=model.make_mtp_cache(),
                                   return_hidden=True, mtp_depth=1)
    mx.eval(dl_full, dl_last)
    assert mx.array_equal(dl_full, dl_last)
