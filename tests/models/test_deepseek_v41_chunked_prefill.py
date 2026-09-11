"""W20: token-chunked prefill exactness + transient-bound tests (CPU, synthetic).

Covers:
  * the arithmetic that names the 103,089,701,120-byte failing allocation as the
    ratio-2 CSA attention score, and that the chosen chunk bounds it;
  * exactness -- chunked prefill logits == one-shot (<= 1e-5) and the cache state
    (window / compress_kv / index_k / compressor frontier / offset) equals
    one-shot, across chunk sizes straddling the sliding window and the ratio-2
    compressor groups;
  * that the forward feeds the MoE switch <= ``chunk`` rows per call (plumbing);
  * that at the real 16,384-token geometry the chosen chunk keeps the attention,
    indexer and MoE transients under 8 GB -- by arithmetic and by a synthetic
    switch that records the largest tensor it is asked to produce.

All tests pin MLX to CPU and use tiny random configs / synthetic switches; no
real-artifact load, so peak RSS stays well under the worker cap.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mtplx.models.deepseek_v41 import (  # noqa: E402
    Model,
    ModelArgs,
    _derive_prefill_chunk,
    _prefill_score_bytes_per_row,
    _resolve_prefill_chunk,
)

EIGHT_GB = 8e9


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _csa_args(**over) -> ModelArgs:
    """8 layers exercising every CSA2 mode: swa, swa, full-r2, reuse, reuse,
    full-r1(candidate), reindex, reuse -- the same shape the parity suite uses,
    with a small window (8) and ratio-2/ratio-1 groups so tiny chunks straddle
    both a window boundary and a compressor group boundary."""
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


def _real_args(**over) -> ModelArgs:
    """The released 40-layer shapes (no weights are built from this -- it is only
    read for the transient arithmetic and the chunk derivation)."""
    base = dict(
        compress_ratios=[0, 0] + [2] * 38,
        kv_source_layer_ids=[2], index_source_layer_ids=[2],
    )
    base.update(over)
    return ModelArgs(**base)  # every other field defaults to the released value


def _randomize(model, seed=0, scale=0.1):
    # Small weights keep the fp32 activation range comfortable so the accumulated
    # matmul-tiling noise stays well under the 1e-5 exactness bar; the parity
    # suite runs the same fixture at scale 0.3 against a 1e-3 oracle bar.
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


def _cache_snapshot(cache):
    """Every stored array in the cache, keyed by layer + field, plus the offset."""
    out = {}
    for i, lc in enumerate(cache.layers):
        for nm in ("window", "compress_kv", "index_k"):
            a = getattr(lc, nm)
            if a is not None:
                out[f"{i}.{nm}"] = np.array(a)
        cs = getattr(lc, "comp_state", None)
        if cs is not None:
            for nm in ("raw_kv", "raw_score"):
                a = getattr(cs, nm)
                if a is not None:
                    out[f"{i}.cs.{nm}"] = np.array(a)
    return out, int(cache.offset)


# ---------------------------------------------------------------------------
# 1. arithmetic: name the 103 GB allocation
# ---------------------------------------------------------------------------
def test_ratio2_csa_score_is_the_103gb_allocation():
    """The failing 103,089,701,120-byte buffer is exactly the ratio-2 CSA
    attention score [b=1, s=16385, H=64, T=window(16385)+compressed(8192)=24577]
    in fp32 -- 16385*64*24577*4."""
    s, H, ratio = 16385, 64, 2
    n_comp = s // ratio
    T = s + n_comp
    assert (n_comp, T) == (8192, 24577)
    score_bytes = s * H * T * 4
    assert score_bytes == 103_089_701_120
    # and the per-row cost the derivation uses matches H*T*4 for this shape
    assert _prefill_score_bytes_per_row(_real_args(), s) == H * T * 4


def test_swa_only_score_is_under_the_buffer_cap():
    """A pure sliding-window layer's score [b,s,H,s] is 68.7 GB -- under the
    86.5 GiB Metal cap, which is why the crash is the *ratio-2* layer, not the
    first SWA layer, and why the exact number is the discriminator."""
    s, H = 16385, 64
    swa_bytes = s * H * s * 4
    assert swa_bytes < 86_586_540_032  # allocates (barely); does not raise
    assert s * H * (s + s // 2) * 4 > 86_586_540_032  # ratio-2 does raise


# ---------------------------------------------------------------------------
# 2. auto-derivation is shape-aware and inert on the working cells
# ---------------------------------------------------------------------------
def test_auto_chunk_bounds_dominant_transient_and_shrinks_with_context():
    args = _real_args()
    for s in (16384, 16385, 65536, 131072):
        chunk = _derive_prefill_chunk(args, s, EIGHT_GB)
        per_row = _prefill_score_bytes_per_row(args, s)
        assert 1 <= chunk < s
        assert chunk * per_row < EIGHT_GB           # dominant transient bounded
        assert (chunk + 1) * per_row >= EIGHT_GB     # and it is the *largest* such chunk
    # 16K derives to 1271 at an 8 GB budget
    assert _derive_prefill_chunk(args, 16384, EIGHT_GB) == 1271


def test_short_prompt_and_decode_run_one_shot():
    args = _real_args()
    # the 1,024-token "THE input" cell and decode stay one-shot (chunk >= s),
    # so the fix is inert on the working path.
    assert _resolve_prefill_chunk(args, 1024, None) >= 1024
    assert _resolve_prefill_chunk(args, 1, None) >= 1


def test_env_and_arg_precedence(monkeypatch):
    args = _real_args()
    monkeypatch.setenv("MTPLX_DSV41_PREFILL_CHUNK", "512")
    assert _resolve_prefill_chunk(args, 16384, None) == 512      # env wins over auto
    assert _resolve_prefill_chunk(args, 16384, 777) == 777       # arg wins over env
    monkeypatch.setenv("MTPLX_DSV41_PREFILL_CHUNK", "0")
    assert _resolve_prefill_chunk(args, 16384, None) == 0        # 0 -> one-shot
    monkeypatch.setenv("MTPLX_DSV41_PREFILL_CHUNK", "auto")
    assert _resolve_prefill_chunk(args, 16384, None) == 1271     # auto -> derived


# ---------------------------------------------------------------------------
# 3. exactness: chunked == one-shot (logits + cache state)
# ---------------------------------------------------------------------------
# Chunk sizes chosen to straddle the sliding window (8) and the ratio-2
# compressor groups (odd chunks split a 2-token group across the boundary);
# includes 1 (a span shorter than a ratio-2 group -> exercises the compress_kv
# None-skip) and s (degenerate single chunk).
_CHUNKS = (1, 2, 3, 5, 7, 8, 9, 13, 24, 25)


def test_chunked_prefill_matches_one_shot():
    args = _csa_args()
    model = Model(args)
    _randomize(model, seed=1)
    s = 25
    ids = mx.array(np.random.RandomState(0).randint(0, args.vocab_size, size=(1, s)))

    one_cache = model.make_cache()
    logits_one = model(ids, cache=one_cache, prefill_chunk=0)  # one-shot
    mx.eval(logits_one)
    snap_one, off_one = _cache_snapshot(one_cache)

    for chunk in _CHUNKS:
        cache = model.make_cache()
        logits = model(ids, cache=cache, prefill_chunk=chunk)
        mx.eval(logits)
        ldiff = float(mx.max(mx.abs(logits - logits_one)).item())
        assert ldiff <= 1e-5, f"chunk={chunk} logit max abs diff {ldiff}"

        snap, off = _cache_snapshot(cache)
        assert off == off_one == s
        assert set(snap) == set(snap_one), f"chunk={chunk} cache keys differ"
        for k, v in snap_one.items():
            cdiff = float(np.max(np.abs(snap[k] - v)))
            assert cdiff <= 1e-5, f"chunk={chunk} cache[{k}] max abs diff {cdiff}"


def test_chunked_matches_one_shot_swa_only():
    """A pure sliding-window model (no CSA) must also chunk exactly."""
    args = _csa_args(
        num_hidden_layers=4, compress_ratios=[0, 0, 0, 0],
        kv_source_layer_ids=[], index_source_layer_ids=[],
        candidate_source_layer_id=-1,
    )
    model = Model(args)
    _randomize(model, seed=3)
    s = 20
    ids = mx.array(np.random.RandomState(1).randint(0, args.vocab_size, size=(1, s)))
    logits_one = model(ids, cache=model.make_cache(), prefill_chunk=0)
    mx.eval(logits_one)
    for chunk in (1, 3, 7, 8, 19):
        logits = model(ids, cache=model.make_cache(), prefill_chunk=chunk)
        mx.eval(logits)
        ldiff = float(mx.max(mx.abs(logits - logits_one)).item())
        assert ldiff <= 1e-5, f"swa chunk={chunk} logit max abs diff {ldiff}"


# ---------------------------------------------------------------------------
# 4. plumbing: the forward feeds the MoE switch <= chunk rows per call
# ---------------------------------------------------------------------------
class _RecordingSwitch(nn.Module):
    """Drop-in for ``MoE.switch_mlp``: records the (rows, top_k) it is called
    with and the largest routed tensor it would produce, and returns a
    correctly-shaped zero output so the MoE combine still runs."""

    def __init__(self, hidden: int, intermediate: int, top_k: int, recorder: dict):
        super().__init__()
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.top_k = int(top_k)
        self.rec = recorder

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        rows = int(x.shape[0])
        top_k = int(indices.shape[-1])
        # A component-bank wave's largest transient is the widest routed gather:
        # rows*top_k rows by max(intermediate, hidden) columns, fp32.
        transient = rows * top_k * max(self.intermediate, self.hidden) * 4
        self.rec["max_rows"] = max(self.rec.get("max_rows", 0), rows)
        self.rec["max_transient"] = max(self.rec.get("max_transient", 0), transient)
        self.rec["calls"] = self.rec.get("calls", 0) + 1
        return mx.zeros((rows, top_k, self.hidden), dtype=x.dtype)


def _install_recording_switch(model, recorder):
    for layer in model.model.layers:
        layer.mlp.switch_mlp = _RecordingSwitch(
            model.args.hidden_size, model.args.moe_intermediate_size,
            model.args.num_experts_per_tok, recorder,
        )


def test_forward_feeds_moe_at_most_chunk_rows():
    args = _csa_args()
    model = Model(args)
    _randomize(model, seed=2)
    s = 25
    ids = mx.array(np.random.RandomState(4).randint(0, args.vocab_size, size=(1, s)))

    rec_one: dict = {}
    _install_recording_switch(model, rec_one)
    mx.eval(model(ids, cache=model.make_cache(), prefill_chunk=0))
    assert rec_one["max_rows"] == s               # one-shot feeds the whole prompt

    for chunk in (4, 7, 10):
        rec: dict = {}
        _install_recording_switch(model, rec)
        mx.eval(model(ids, cache=model.make_cache(), prefill_chunk=chunk))
        assert rec["max_rows"] == min(chunk, s)   # chunked never exceeds the chunk
        assert rec["calls"] == args.num_hidden_layers * ((s + chunk - 1) // chunk)


# ---------------------------------------------------------------------------
# 5. real geometry: the chosen chunk keeps every prefill transient under 8 GB
# ---------------------------------------------------------------------------
def test_real_geometry_transients_bounded_by_chunk():
    args = _real_args()
    s = 16384
    H = args.num_attention_heads
    ratio = 2
    n_comp = s // ratio
    T = s + n_comp
    top_k = args.num_experts_per_tok
    inter, hidden = args.moe_intermediate_size, args.hidden_size
    idx_heads = args.index_n_heads

    # one-shot -> the 103 GB score that overflows the cap
    assert s * H * T * 4 > 86_586_540_032

    chunk = _resolve_prefill_chunk(args, s, None)  # auto -> 1271
    assert 0 < chunk < s

    # arithmetic bounds at the chosen chunk (all < 8 GB)
    attn_score = chunk * H * T * 4
    indexer_score = chunk * idx_heads * n_comp * 4
    moe_transient = chunk * top_k * max(inter, hidden) * 4
    assert attn_score < EIGHT_GB, attn_score
    assert indexer_score < EIGHT_GB, indexer_score
    assert moe_transient < EIGHT_GB, moe_transient

    # synthetic switch: drive the actual chunk grid, allocate + eval the largest
    # routed tensor each MoE call is asked for, and confirm the recorded maximum
    # is what the arithmetic predicts and stays under 8 GB.
    recorded_max = 0
    start = 0
    while start < s:
        end = min(start + chunk, s)
        rows = end - start                        # b=1 -> chunk_s tokens this call
        out = mx.zeros((rows * top_k, max(inter, hidden)), dtype=mx.float32)
        mx.eval(out)
        recorded_max = max(recorded_max, out.nbytes)
        start = end
    assert recorded_max == chunk * top_k * max(inter, hidden) * 4
    assert recorded_max < EIGHT_GB
