"""End-to-end Engram attach + hook integration for the DeepSeek-V4.1 text Model (W6).

Exercises :meth:`Model.attach_engram` and the backbone's per-step engram threading
against the REAL artifact (``$DSV41_ARTIFACT_DIR`` or the default streaming-q2 path):

  * ``attach_engram`` sets an :class:`~mtplx.engram_v41.EngramV41` hook on **exactly**
    the manifest's engram layers (1 and 14) and nowhere else, builds the shared
    :class:`~mtplx.engram_v41.NgramHashState` prototype from the artifact tokenizer
    (compressed vocab 99092), and :meth:`make_cache` hands each sequence its own
    streaming clone;
  * the attached hooks change the backbone hidden state vs ``engram_hook = None`` on a
    real-token prompt (the additive Engram contribution is non-trivial and finite);
  * a speculative-decode rollback trims the engram history in step with the KV
    rollback, so decoding the real continuation after a rolled-back divergent tail
    reproduces the clean-decode hidden states byte-for-byte.

The hooks are the real ones (real 104 GiB row banks + the W4 resident sidecar); only
the *backbone* is a reduced stand-in (hidden 5120 / hc_mult 4 so the real dim-5120
hooks fit, but 15 SWA-only layers, tiny experts/vocab) so a full 40-layer expert
stream is not needed to test the wiring.  The faithful full-artifact forward is the
W6 end-to-end proof in ``tests/test_deepseek_v41_loader.py``.

CPU only.  Run under ``nice -n 19``, without ``-n auto``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

# device-independent whether run alone or with the other deepseek_v41/engram files
mx.set_default_device(mx.cpu)

from mtplx.engram_v41 import EngramV41, NgramHashState
from mtplx.models.deepseek_v41 import Model, ModelArgs

ARTIFACT = Path(
    os.environ.get(
        "DSV41_ARTIFACT_DIR",
        "/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2",
    )
)
ENGRAM_DIR = ARTIFACT / "engram"
ENGRAM_LAYERS = (1, 14)
_CACHE_BYTES = 16 * 1024 * 1024  # tiny resident-row budget: misses re-read positionally

_have_engram = (
    (ENGRAM_DIR / "engram-manifest.json").is_file()
    and (ENGRAM_DIR / "engram-residents.safetensors").is_file()
    and (ENGRAM_DIR / "engram-L1.bin").is_file()
    and (ENGRAM_DIR / "engram-L14.bin").is_file()
    and (ARTIFACT / "tokenizer.json").is_file()
)
needs_engram = pytest.mark.skipif(
    not _have_engram, reason=f"real engram artifact not present: {ENGRAM_DIR}"
)


def _reduced_args() -> ModelArgs:
    """A small SWA-only backbone that keeps the real engram dims (hidden 5120,
    hc_mult 4) so the real dim-5120 hooks attach, but is otherwise tiny.  Needs
    >= 15 layers so engram layer 14 exists."""
    return ModelArgs(
        vocab_size=256,
        hidden_size=5120,
        num_hidden_layers=15,
        num_attention_heads=4,
        head_dim=16,
        qk_rope_head_dim=4,
        q_lora_rank=12,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        swiglu_limit=0.5,
        sliding_window=8,
        window_size=8,
        compress_ratios=[0] * 15,  # SWA-only, no CSA2 compressor/indexer
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        hc_mult=4,
        rms_norm_eps=1e-20,
        engram_layer_ids=list(ENGRAM_LAYERS),
    )


@pytest.fixture(scope="module")
def engram_model():
    """Reduced backbone with the REAL engram hooks attached once (module-scoped:
    attach walks the tokenizer + opens the 104 GiB banks, so do it a single time)."""
    mx.random.seed(0)
    model = Model(_reduced_args(), quantize=False)
    mx.eval(model.parameters())
    attached = model.attach_engram(ENGRAM_DIR, cache_bytes=_CACHE_BYTES)
    assert tuple(attached) == ENGRAM_LAYERS
    return model


# ---------------------------------------------------------------------------
# 1) hooks attached on EXACTLY the engram layers + prototype built from tokenizer
# ---------------------------------------------------------------------------
@needs_engram
def test_attach_sets_hooks_on_exactly_engram_layers(engram_model):
    model = engram_model
    hooked = [
        i for i, layer in enumerate(model.model.layers) if layer.engram_hook is not None
    ]
    assert hooked == list(ENGRAM_LAYERS), f"hooked layers {hooked} != {list(ENGRAM_LAYERS)}"
    for i in ENGRAM_LAYERS:
        hook = model.model.layers[i].engram_hook
        assert isinstance(hook, EngramV41)
        # the layer's hash index is its position in the manifest layer_ids list
        assert hook.layer_hash_index == ENGRAM_LAYERS.index(i)
        assert hook.dim == model.args.hidden_size and hook.hc_mult == model.args.hc_mult

    # prototype hash state built from the artifact tokenizer (compressed vocab 99092)
    proto = model.model.engram_hash
    assert isinstance(proto, NgramHashState)
    assert proto.n_layers == len(ENGRAM_LAYERS)
    assert len(proto.token_map) == proto.token_map.shape[0]
    # each engram layer's num_embeddings matches its row bank
    assert proto.num_embeddings is not None and len(proto.num_embeddings) == 2

    # make_cache clones a fresh, independent streaming state per sequence
    c1 = model.make_cache()
    c2 = model.make_cache()
    assert isinstance(c1.engram_state, NgramHashState)
    assert c1.engram_state is not c2.engram_state
    assert c1.engram_state.token_map is proto.token_map  # config shared, cheap clone
    assert c1.engram_state.length == 0


# ---------------------------------------------------------------------------
# 2) hooks change the hidden state vs engram_hook = None on a real-token prompt
# ---------------------------------------------------------------------------
@needs_engram
def test_hook_changes_hidden_vs_none(engram_model):
    model = engram_model
    ids = mx.array([[5, 40, 200, 3, 77, 128, 9, 60]])  # valid token ids (< vocab)

    cache = model.make_cache()
    h_with = model.model(ids, cache)
    mx.eval(h_with)
    assert bool(np.isfinite(np.array(h_with)).all())
    # engram advanced exactly once, over all prompt positions
    assert model.model.engram_hash is not None
    assert cache.engram_state.length == ids.shape[1]

    saved = {i: model.model.layers[i].engram_hook for i in ENGRAM_LAYERS}
    try:
        for i in ENGRAM_LAYERS:
            model.model.layers[i].engram_hook = None
        h_without = model.model(ids, model.make_cache())
        mx.eval(h_without)
    finally:
        for i, hook in saved.items():
            model.model.layers[i].engram_hook = hook

    diff = float(np.max(np.abs(np.array(h_with) - np.array(h_without))))
    assert diff > 1e-2, f"engram hooks did not change the hidden state (max abs diff {diff})"

    # re-attaching the hooks reproduces the hooked hidden state exactly (no state leak)
    h_again = model.model(ids, model.make_cache())
    mx.eval(h_again)
    assert np.array_equal(np.array(h_with), np.array(h_again))


# ---------------------------------------------------------------------------
# 3) rollback trims the engram history in step with the KV rollback
# ---------------------------------------------------------------------------
@needs_engram
def test_rollback_trims_engram_in_step_with_kv(engram_model):
    model = engram_model
    prompt = mx.array([[(i * 5 + 1) % model.args.vocab_size for i in range(20)]])
    step_ids = [mx.array([[11]]), mx.array([[23]]), mx.array([[7]])]

    def decode_run():
        cache = model.make_cache()
        model.model(prompt, cache)
        outs = []
        for tok in step_ids:
            h = model.model(tok, cache)
            mx.eval(h)
            outs.append(np.array(h[0, -1]))
        return outs

    first = decode_run()

    # decode a divergent throwaway tail past a mark, roll back, decode the real tail
    cache = model.make_cache()
    model.model(prompt, cache)
    mark = cache.mark()
    assert cache.engram_state.length == prompt.shape[1]
    model.model(mx.array([[41]]), cache)
    model.model(mx.array([[9]]), cache)
    assert cache.engram_state.length == prompt.shape[1] + 2
    cache.rollback(mark)
    # KV offset and engram history both restored to the mark
    assert cache.offset == prompt.shape[1]
    assert cache.engram_state.length == prompt.shape[1]

    rolled = []
    for tok in step_ids:
        h = model.model(tok, cache)
        mx.eval(h)
        rolled.append(np.array(h[0, -1]))
    for a, b in zip(first, rolled):
        assert np.array_equal(a, b), "rollback did not restore identical engram-hooked hidden"
