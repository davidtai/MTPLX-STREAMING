"""W30 / kernel-ledger K16 -- layer-major chunked prefill (CPU, synthetic).

W20 chunked prefill is *chunk-major*: every chunk runs through all 40 layers, so
each chunk's MoE re-streams ~the whole routed bank per layer (up to ~13x the
~20 s bank read at 16 K).  K16 restructures to *layer-major*: iterate every layer
over all chunks before the next layer, so a layer's routed experts stream ONCE
across the whole prompt.  These gates prove, on tiny CPU configs (no artifact):

  * layer-major logits + cache state == one-shot (<= 1e-5), across chunk sizes
    straddling the sliding window and the ratio-2 compressor groups, non-divisors
    included -- attention stays per chunk, the reorder is exact;
  * a counting fake switch proves each (layer, expert) is fetched **once** per
    prefill under layer-major, vs C (one per chunk) under chunk-major -- the K16
    claim, asserted hard;
  * the engram n-gram history and every KV lane end identical to one-shot (the
    per-chunk engram-row replay keeps layers 1/14 correct);
  * the ``switch_mlp`` call is row-capped so the routed-output transient stays
    bounded, and the resident per-chunk Hyper-Connection state is well under 1 GB
    at 16 K -- both by arithmetic; correctness survives a forced cap split.

Pins MLX to CPU; tiny random configs / synthetic switches; no artifact load.
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
    _derive_moe_row_cap,
    _resolve_prefill_layer_major,
)
from mtplx.models.deepseek_v41_cache import make_cache  # noqa: E402
from mtplx.engram_v41 import NgramHashState, n_hash_cols  # noqa: E402

EIGHT_GB = 8e9
ONE_GB = 1e9


# ---------------------------------------------------------------------------
# fixtures (mirror tests/models/test_deepseek_v41_chunked_prefill.py)
# ---------------------------------------------------------------------------
def _csa_args(**over) -> ModelArgs:
    """8 layers exercising every CSA2 mode with a small window (8) and ratio-2/1
    groups, so tiny chunks straddle a window boundary and a compressor group."""
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
    """The released 40-layer shapes -- read only for transient arithmetic."""
    base = dict(compress_ratios=[0, 0] + [2] * 38,
                kv_source_layer_ids=[2], index_source_layer_ids=[2])
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


def _cache_snapshot(cache):
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


# Chunk sizes straddle the sliding window (8) and the ratio-2 groups (odd chunks
# split a 2-token group); 1 exercises the compress_kv None-skip, 25 is one chunk.
_CHUNKS = (1, 2, 3, 5, 7, 8, 9, 13, 24, 25)


# ---------------------------------------------------------------------------
# 1. exactness: layer-major == one-shot (logits + cache state)
# ---------------------------------------------------------------------------
def test_layer_major_matches_one_shot():
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
        logits = model(ids, cache=cache, prefill_chunk=chunk, prefill_layer_major=True)
        mx.eval(logits)
        ldiff = float(mx.max(mx.abs(logits - logits_one)).item())
        assert ldiff <= 1e-5, f"layer-major chunk={chunk} logit max abs diff {ldiff}"

        snap, off = _cache_snapshot(cache)
        assert off == off_one == s
        assert set(snap) == set(snap_one), f"chunk={chunk} cache keys differ"
        for k, v in snap_one.items():
            cdiff = float(np.max(np.abs(snap[k] - v)))
            assert cdiff <= 1e-5, f"chunk={chunk} cache[{k}] max abs diff {cdiff}"


def test_layer_major_matches_one_shot_swa_only():
    """A pure sliding-window model (no CSA) must also layer-major exactly."""
    args = _csa_args(
        num_hidden_layers=4, compress_ratios=[0, 0, 0, 0],
        kv_source_layer_ids=[], index_source_layer_ids=[], candidate_source_layer_id=-1,
    )
    model = Model(args)
    _randomize(model, seed=3)
    s = 20
    ids = mx.array(np.random.RandomState(1).randint(0, args.vocab_size, size=(1, s)))
    logits_one = model(ids, cache=model.make_cache(), prefill_chunk=0)
    mx.eval(logits_one)
    for chunk in (1, 3, 7, 8, 19):
        logits = model(ids, cache=model.make_cache(), prefill_chunk=chunk,
                       prefill_layer_major=True)
        mx.eval(logits)
        ldiff = float(mx.max(mx.abs(logits - logits_one)).item())
        assert ldiff <= 1e-5, f"swa layer-major chunk={chunk} logit max abs diff {ldiff}"


# ---------------------------------------------------------------------------
# 2. THE K16 claim: each (layer, expert) is fetched once under layer-major
# ---------------------------------------------------------------------------
class _CountingSwitch(nn.Module):
    """Drop-in for ``MoE.switch_mlp`` that counts, per call, the UNIQUE expert
    ids it is asked to gather -- exactly what ``partition_route_waves`` streams
    from the bank per switch call (each unique expert = one gather).  Records a
    per-(layer, expert) fetch tally so the read-the-bank-once claim is testable
    without the real bank."""

    def __init__(self, hidden: int, top_k: int, layer_id: int, rec: dict):
        super().__init__()
        self.hidden = int(hidden)
        self.top_k = int(top_k)
        self.layer_id = int(layer_id)
        self.rec = rec

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        rows = int(x.shape[0])
        unique = {int(v) for v in np.array(indices).reshape(-1).tolist()}
        fetches = self.rec.setdefault("fetches", {})
        for e in unique:
            fetches[(self.layer_id, e)] = fetches.get((self.layer_id, e), 0) + 1
        self.rec["calls"] = self.rec.get("calls", 0) + 1
        self.rec["max_rows"] = max(self.rec.get("max_rows", 0), rows)
        return mx.zeros((rows, self.top_k, self.hidden), dtype=x.dtype)


def _install_counting_switch(model, rec):
    for layer in model.model.layers:
        layer.mlp.switch_mlp = _CountingSwitch(
            model.args.hidden_size, model.args.num_experts_per_tok, layer.layer_id, rec,
        )


def test_layer_major_reads_each_expert_once_vs_chunk_major():
    args = _csa_args()
    s = 25
    chunk = 7
    n_chunks = (s + chunk - 1) // chunk  # 4
    ids = mx.array(np.random.RandomState(4).randint(0, args.vocab_size, size=(1, s)))

    def run(layer_major):
        model = Model(args)
        _randomize(model, seed=2)
        rec: dict = {}
        _install_counting_switch(model, rec)
        mx.eval(model(ids, cache=model.make_cache(),
                      prefill_chunk=chunk, prefill_layer_major=layer_major))
        return rec

    rec_chunk = run(False)
    rec_layer = run(True)

    fetch_chunk = rec_chunk["fetches"]
    fetch_layer = rec_layer["fetches"]

    # SAME set of (layer, expert) pairs is exercised either way (no expert lost).
    assert set(fetch_chunk) == set(fetch_layer)
    assert len(fetch_layer) > 0

    # THE claim: layer-major fetches every (layer, expert) exactly once; chunk-
    # major fetches at least one of them once per chunk it routes in (up to C).
    assert max(fetch_layer.values()) == 1, "layer-major refetched an expert"
    assert min(fetch_layer.values()) == 1
    assert max(fetch_chunk.values()) == n_chunks, (
        "chunk-major should refetch a hot expert once per chunk"
    )
    assert max(fetch_chunk.values()) > 1  # premise: chunk-major genuinely refetches

    # call accounting: one switch call per layer (layer-major) vs one per (layer,
    # chunk) (chunk-major); and the single layer-major call carries the whole
    # prompt's rows.
    assert rec_layer["calls"] == args.num_hidden_layers
    assert rec_chunk["calls"] == args.num_hidden_layers * n_chunks
    assert rec_layer["max_rows"] == s
    assert rec_chunk["max_rows"] == chunk


# ---------------------------------------------------------------------------
# 3. engram history + KV lanes end identical to one-shot
# ---------------------------------------------------------------------------
_ENG_LAYERS = (1, 3)
_MAXNG = 3
_NHEADS = 2


def _engram_proto(vocab):
    nl = len(_ENG_LAYERS)
    rs = np.random.RandomState
    return NgramHashState(
        layer_ids=list(_ENG_LAYERS), max_ngram_size=_MAXNG, n_heads=_NHEADS,
        token_map=list(range(vocab)),
        multipliers=rs(0).randint(1, 7, size=(nl, _MAXNG)),
        primes=rs(1).randint(3, 97, size=(nl, _MAXNG - 1, _NHEADS)),
        flat_offsets=rs(2).randint(0, 50, size=(nl, n_hash_cols(_MAXNG, _NHEADS))),
        pad_compressed=0,
    )


class _FakeEngramHook:
    """Reads this chunk's engram row ids off the cache-state view and adds a
    deterministic function of them -- enough to make the hidden state genuinely
    depend on the row ids (so a mis-ordered replay would diverge)."""

    def __init__(self, layer_hash_index: int):
        self.lhi = int(layer_hash_index)

    def __call__(self, h, token_ids, cache_state):
        row_ids = cache_state.current_row_ids(self.lhi)         # np [B, L, cols]
        B, L = int(h.shape[0]), int(h.shape[1])
        assert tuple(row_ids.shape[:2]) == (B, L)
        add = (row_ids.astype(np.float64).sum(-1) % 7).astype(np.float32)  # [B, L]
        return h + 0.01 * mx.array(add)[:, :, None, None]


def _attach_fake_engram(model):
    for layer in model.model.layers:
        layer.engram_hook = (
            _FakeEngramHook(_ENG_LAYERS.index(layer.layer_id))
            if layer.layer_id in _ENG_LAYERS else None
        )


def test_layer_major_engram_and_kv_identical_to_one_shot():
    args = _csa_args()
    model = Model(args)
    _randomize(model, seed=5)
    _attach_fake_engram(model)
    vocab = args.vocab_size
    s = 25
    ids = mx.array(np.random.RandomState(6).randint(0, vocab, size=(1, s)))

    one_cache = make_cache(args, engram_state=_engram_proto(vocab).fresh())
    logits_one = model(ids, cache=one_cache, prefill_chunk=0)
    mx.eval(logits_one)
    snap_one, off_one = _cache_snapshot(one_cache)
    eng_one = np.array(one_cache.engram_state.state)

    for chunk in (1, 4, 7, 13):
        cache = make_cache(args, engram_state=_engram_proto(vocab).fresh())
        logits = model(ids, cache=cache, prefill_chunk=chunk, prefill_layer_major=True)
        mx.eval(logits)

        # engram history buffer: same length AND same contents
        eng = np.array(cache.engram_state.state)
        assert eng.shape == eng_one.shape, f"chunk={chunk} engram length differs"
        assert np.array_equal(eng, eng_one), f"chunk={chunk} engram contents differ"
        assert int(cache.engram_state.length) == int(one_cache.engram_state.length) == s

        # KV lanes: same keys + lengths + values (window / compress / index)
        snap, off = _cache_snapshot(cache)
        assert off == off_one == s
        assert set(snap) == set(snap_one)
        for k, v in snap_one.items():
            assert snap[k].shape == v.shape, f"chunk={chunk} cache[{k}] length differs"
            assert float(np.max(np.abs(snap[k] - v))) <= 1e-5, f"chunk={chunk} cache[{k}]"

        # and the engram-wired logits track one-shot
        assert float(mx.max(mx.abs(logits - logits_one)).item()) <= 1e-5


# ---------------------------------------------------------------------------
# 4. flag resolution
# ---------------------------------------------------------------------------
def test_layer_major_flag_resolution(monkeypatch):
    monkeypatch.delenv("MTPLX_DSV41_PREFILL_LAYER_MAJOR", raising=False)
    assert _resolve_prefill_layer_major(None) is False       # default OFF
    assert _resolve_prefill_layer_major(True) is True         # arg wins
    assert _resolve_prefill_layer_major(False) is False
    for on in ("1", "true", "yes", "on", "TRUE"):
        monkeypatch.setenv("MTPLX_DSV41_PREFILL_LAYER_MAJOR", on)
        assert _resolve_prefill_layer_major(None) is True
    for off in ("0", "false", "no", "off", "", "auto"):
        monkeypatch.setenv("MTPLX_DSV41_PREFILL_LAYER_MAJOR", off)
        assert _resolve_prefill_layer_major(None) is False
    monkeypatch.setenv("MTPLX_DSV41_PREFILL_LAYER_MAJOR", "1")
    assert _resolve_prefill_layer_major(False) is False       # arg still wins


def test_layer_major_is_inert_on_short_prompt_and_decode():
    """The knob only engages when chunking is active (chunk < s); a one-shot
    prompt/decode is byte-identical whether or not it is set."""
    args = _csa_args()
    model = Model(args)
    _randomize(model, seed=7)
    ids = mx.array(np.random.RandomState(8).randint(0, args.vocab_size, size=(1, 6)))
    a = model(ids, cache=model.make_cache(), prefill_chunk=0, prefill_layer_major=True)
    b = model(ids, cache=model.make_cache(), prefill_chunk=0, prefill_layer_major=False)
    mx.eval(a, b)
    assert float(mx.max(mx.abs(a - b)).item()) == 0.0


# ---------------------------------------------------------------------------
# 5. memory: row-capped MoE transient + resident HC state, by arithmetic
# ---------------------------------------------------------------------------
def test_moe_row_cap_bounds_routed_transient_at_real_geometry():
    args = _real_args()
    top_k, hidden = args.num_experts_per_tok, args.hidden_size  # 6, 5120
    cap = _derive_moe_row_cap(args, EIGHT_GB)
    # largest row count whose fp32 routed-output transient stays under 8 GB
    assert cap == int(EIGHT_GB // (top_k * hidden * 4)) == 65104
    assert cap * top_k * hidden * 4 < EIGHT_GB
    assert (cap + 1) * top_k * hidden * 4 >= EIGHT_GB
    # the standard 16,384-token prompt is far under the cap -> ONE MoE call/layer
    # (bank read once); its routed transient is ~2 GB, well under budget.
    s = 16384
    assert s < cap
    assert s * top_k * hidden * 4 == 2_013_265_920  # ~2.01 GB
    assert s * top_k * hidden * 4 < EIGHT_GB


def test_resident_hc_state_is_well_under_1gb_at_16k():
    args = _real_args()
    hidden, streams = args.hidden_size, args.hc_mult  # 5120, 4
    # resident per-chunk Hyper-Connection state across the whole layer loop =
    # every chunk's [b, chunk, streams, hidden] bf16, summed over chunks = the
    # full prompt: hidden * streams * tokens * 2 bytes.
    def hc_bytes(tokens):
        return hidden * streams * tokens * 2
    assert hc_bytes(1024) == 41_943_040          # ~0.042 GB at the 1,024 cell
    assert hc_bytes(16384) == 671_088_640        # ~0.671 GB at 16 K
    assert hc_bytes(16384) < ONE_GB              # the design bound


def test_layer_major_moe_row_cap_splits_and_reassembles_exactly():
    """The layer-major MoE (``_layer_major_moe``) reads the bank once by default
    (one ``switch_mlp`` call over the whole concat) but, when the routed-output
    row cap would overflow, splits into >1 call each within the cap -- and either
    way the per-chunk outputs are the same rows as the single-call result.

    Exercised directly so a sub-1 GB cap can be forced (the env budget floors at
    1 GB, so an end-to-end tiny split is not reachable through the flag)."""
    args = _csa_args()
    model = Model(args)
    _randomize(model, seed=2)
    backbone = model.model
    layer = backbone.layers[0]
    hidden = args.hidden_size

    spans = [(0, 3), (3, 6), (6, 9), (9, 12)]     # 4 chunks of 3 rows
    total = spans[-1][1]
    rng = np.random.RandomState(11)
    moe_inputs = [
        mx.array(rng.randn(1, e - s, hidden).astype(np.float32)) for s, e in spans
    ]

    # oracle: the real switch over the full concat in ONE call.
    cat = mx.concatenate(moe_inputs, axis=1)
    oracle = layer.mlp(cat)
    mx.eval(oracle)

    # huge cap -> a single call over every row (the read-once default).
    big = backbone._layer_major_moe(layer, moe_inputs, spans, row_cap=10 ** 18)
    got_big = mx.concatenate(big, axis=1)
    mx.eval(got_big)
    assert got_big.shape == oracle.shape == (1, total, hidden)
    assert float(mx.max(mx.abs(got_big - oracle)).item()) <= 1e-5

    # cap 7 rows -> packs at most two 3-row chunks per call; still exact.
    capped = backbone._layer_major_moe(layer, moe_inputs, spans, row_cap=7)
    got_cap = mx.concatenate(capped, axis=1)
    mx.eval(got_cap)
    assert got_cap.shape == oracle.shape
    assert float(mx.max(mx.abs(got_cap - oracle)).item()) <= 1e-5

    # the cap actually forced a split, and no call exceeded it.
    rec: dict = {}
    layer.mlp.switch_mlp = _CountingSwitch(
        hidden, args.num_experts_per_tok, layer.layer_id, rec,
    )
    out = backbone._layer_major_moe(layer, moe_inputs, spans, row_cap=7)
    mx.eval(out)
    assert 1 < rec["calls"] < len(spans)   # split, but multiple chunks per call
    assert rec["max_rows"] <= 7            # every call within the transient cap
    assert rec["max_rows"] > 3            # and at least one call packed >1 chunk
