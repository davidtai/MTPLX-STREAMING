"""W99: the byte-identical decode-attention cast lean (``MTPLX_DSV41_ATTN_LEAN_CASTS``;
docs/deepseek-v41/W97_ATTENTION_291MS.md §8).

The decode attention layer issues 15-25 f32 ``AsType`` kernels; W99 traced each and
found almost all LOAD-BEARING reference f32 (rmsnorm, softmax core, cos/sin, RoPE,
o-LoRA). Two are genuinely REDUNDANT and removed here byte-identically:
  1. the eager selected-key core casts ``KVg`` to f32 twice (QK^T + PV) -> cast once;
     and the per-head value-0 sink is re-cast every token -> cached f32 per layer;
  2. ``_cos_sin`` re-lifts the layer's numpy ``inv_freq`` to a device array every
     token -> lifted once and cached per layer.

Unlike the core-compile / K29 levers (rounding-class), this is BYTE-IDENTICAL by
construction (no fp math reordered). This proves:
  1. bit-for-bit identical logits + greedy ids over 64 decode steps, lever OFF vs ON,
     eager AND under the K22 attention compile tapes;
  2. the eager core issues fewer ``AsType`` kernels with the lever on (KVg dedupe);
  3. the lean stack composes byte-identically with the wo_a cache.

Tiny dims, CPU-pinned (worker-tests-must-pin-mlx-cpu.md); no artifact, no GPU.
"""

from __future__ import annotations

import importlib.util
import io
import re
from collections import Counter
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest

from mtplx.models import deepseek_v41 as dsv41

_REPO = Path(__file__).resolve().parents[1]
_RECT = re.compile(r'\[label ="([^"]+)", shape=rectangle\]')


def _load_bisect():
    spec = importlib.util.spec_from_file_location(
        "w99_bisect", _REPO / "scripts" / "deepseek_v41" / "metal_decode_attn_bisect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _op_counts(*outs):
    buf = io.StringIO()
    mx.export_to_dot(buf, *[a for a in outs if isinstance(a, mx.array)])
    return Counter(_RECT.findall(buf.getvalue()))


def _decode(model, ops, prompt_ids, steps):
    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    tok = ops.argmax_last(logits)
    ids, logs = [tok], [logits[0, -1]]
    for _ in range(steps):
        logits = model(ops.input([[tok]]), cache=cache)
        ops.sync(logits)
        tok = ops.argmax_last(logits)
        ids.append(tok)
        logs.append(logits[0, -1])
    return ids, logs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dsv41._ATTN_LEAN_CASTS_ENV, raising=False)
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    yield


@pytest.mark.parametrize("compile_on", [False, True])
def test_lean_casts_byte_identical_64_steps(monkeypatch, compile_on):
    bis = _load_bisect()
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", bool(compile_on))
    steps = 64
    model, _args = bis._build_tiny_full_model(seed=5)
    ops = bis._TinyOps()
    prompt_ids = list(range(1, 41))

    monkeypatch.setenv(dsv41._ATTN_LEAN_CASTS_ENV, "0")
    ids_off, log_off = _decode(model, ops, prompt_ids, steps)
    monkeypatch.setenv(dsv41._ATTN_LEAN_CASTS_ENV, "1")
    ids_on, log_on = _decode(model, ops, prompt_ids, steps)

    assert ids_on == ids_off, "lean casts must be BYTE-identical -> greedy ids identical"
    for t, (a, b) in enumerate(zip(log_off, log_on)):
        assert a.dtype == b.dtype
        assert bool(mx.all(a == b).item()), f"step {t}: lean-casts logits not byte-identical"


def test_lean_casts_composes_with_wo_a_cache_byte_identical(monkeypatch):
    """The byte-identical stack: wo_a cache + lean casts (cell16k_ring_lean) must be
    bit-for-bit identical to neither-on over 64 steps (both levers are exact)."""
    bis = _load_bisect()
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", True)
    steps = 64
    model, _args = bis._build_tiny_full_model(seed=6)
    ops = bis._TinyOps()
    prompt_ids = list(range(1, 41))

    for k in (dsv41._ATTN_LEAN_CASTS_ENV, dsv41._ATTN_WO_A_CACHE_ENV):
        monkeypatch.setenv(k, "0")
    ids_off, log_off = _decode(model, ops, prompt_ids, steps)
    for k in (dsv41._ATTN_LEAN_CASTS_ENV, dsv41._ATTN_WO_A_CACHE_ENV):
        monkeypatch.setenv(k, "1")
    ids_on, log_on = _decode(model, ops, prompt_ids, steps)

    assert ids_on == ids_off
    for a, b in zip(log_off, log_on):
        assert bool(mx.all(a == b).item()), "wo_a+lean stack not byte-identical"


def test_lean_casts_dedupes_kvg_astype(monkeypatch):
    """One swa_only decode `_attend` issues fewer AsType kernels with lean on (the
    KVg f32 cast, otherwise emitted for both QK^T and PV, is computed once)."""
    bis = _load_bisect()
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", False)   # eager core, so the cast is visible
    monkeypatch.setattr(dsv41, "_ATTN_WIN_MEMO", False)
    args = bis.real_args()
    attn = bis._build_layer(args, "swa_only")

    def _one_attend_ops():
        cache, shared, x, pos = bis.build_case(args, attn, "swa_only", T=1024)
        for _ in range(2):
            o = attn._attend(x, pos, cache, shared); mx.eval(o)
            shared = bis.SharedAttentionRuntime()
        o = attn._attend(x, pos, cache, shared)   # count BEFORE eval
        c = _op_counts(o)
        mx.eval(o)
        return c

    monkeypatch.setenv(dsv41._ATTN_LEAN_CASTS_ENV, "0")
    off = _one_attend_ops()
    monkeypatch.setenv(dsv41._ATTN_LEAN_CASTS_ENV, "1")
    on = _one_attend_ops()
    assert on["AsType"] < off["AsType"], (
        f"lean AsType {on['AsType']} not < eager {off['AsType']}"
    )


def test_lean_inv_freq_and_sink_cached_once(monkeypatch):
    bis = _load_bisect()
    args = bis.real_args()
    attn = bis._build_layer(args, "swa_only")
    a = attn._lean_inv_freq()
    b = attn._lean_inv_freq()
    assert a is b, "inv_freq must be lifted once and cached"
    s1 = attn._lean_sink_f32()
    s2 = attn._lean_sink_f32()
    assert s1 is s2, "f32 sink must be cached"
    # values match the per-token path
    import numpy as np
    assert bool(mx.all(a == mx.array(np.asarray(attn.inv_freq, dtype=np.float32))).item())
    assert bool(mx.all(s1 == attn.attn_sink.astype(mx.float32)).item())


def test_lean_casts_off_is_default():
    assert dsv41._resolve_attn_lean_casts(raw="") is False
    assert dsv41._resolve_attn_lean_casts(raw="0") is False
    assert dsv41._resolve_attn_lean_casts(raw="1") is True
    with pytest.raises(ValueError):
        dsv41._resolve_attn_lean_casts(raw="banana")
