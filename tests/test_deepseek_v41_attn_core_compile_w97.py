"""W97: the fixed-shape decode-attention CORE compile lever
(``MTPLX_DSV41_ATTN_CORE_COMPILE``; docs/deepseek-v41/W97_ATTENTION_291MS.md).

The selected-key core (QK^T + CSA/causal mask + per-head value-0 sink + f32 softmax
+ PV, over the gathered ``[b,s,k,hd]`` operand) has FIXED shapes at decode (T==1) /
small-M verify, so it is wrapped in ONE geometry-keyed ``mx.compile`` tape whose
elementwise runs fuse (~13 -> ~8 kernels). It is ROUNDING-CLASS, not byte-identical
(n=1 compile reassociates the fp32 einsum/reductions -- the K35 lesson), so it is
gated separately and its Δ is labelled.

This proves:
  1. dispatch reduction -- the compiled core has strictly fewer graph primitives
     (and fewer non-view kernels) than the eager core, at the real decode geometry;
  2. compile-cache is BOUNDED -- 64 decode steps build a small fixed number of core
     tapes (per geometry), never one per token;
  3. greedy-argmax identity + labelled Δ -- 64 decode steps through a tiny full
     model produce the same sampled token ids with the lever OFF vs ON, and the
     logit max|Δ| stays within the rounding-class band.

Tiny dims, CPU-pinned (worker-tests-must-pin-mlx-cpu.md); no artifact, no GPU.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
from collections import Counter
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest

from mtplx.models import deepseek_v41 as dsv41

_REPO = Path(__file__).resolve().parents[1]
_RECT = re.compile(r'\[label ="([^"]+)", shape=rectangle\]')
_VIEW = {"Reshape", "Flatten", "Unflatten", "ExpandDims", "Squeeze", "Broadcast",
         "Transpose", "Slice", "StopGradient", "Arange"}


def _count_prims(*outs):
    arrs = [a for a in outs if isinstance(a, mx.array)]
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrs)
    labels = _RECT.findall(buf.getvalue())
    return len(labels), Counter(labels)


def _kernels(counter):
    return sum(v for k, v in counter.items() if k not in _VIEW)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dsv41._ATTN_CORE_COMPILE_ENV, raising=False)
    dsv41._ATTN_CORE_COMPILED.clear()
    yield
    dsv41._ATTN_CORE_COMPILED.clear()


def test_core_compiled_has_fewer_dispatches_than_eager():
    """At the real decode geometry (H=64, hd=512, k=640) the compiled core has
    strictly fewer graph primitives AND fewer non-view kernels than eager -- the
    scattered scale/where/max/exp/sum/exp/add/divide fuse into ~3 Compiled nodes."""
    b, s, H, hd, k = 1, 1, 64, 512, 640
    mx.random.seed(0)
    q = mx.random.normal((b, s, H, hd)) * 0.05
    KVg = mx.random.normal((b, s, k, hd)) * 0.05
    valid = mx.random.uniform(shape=(b, s, k)) > 0.1
    sink = mx.random.normal((H,)) * 0.5
    mx.eval(q, KVg, valid, sink)
    scale = hd ** -0.5

    o_e = dsv41._attn_core_impl(q, KVg, valid, sink, scale)   # eager, count BEFORE eval
    ne, ope = _count_prims(o_e)
    o_c = dsv41._attn_core_compiled(q, KVg, valid, scale)(q, KVg, valid, sink)
    nc, opc = _count_prims(o_c)
    mx.eval(o_e, o_c)

    assert nc < ne, f"compiled core prims {nc} not < eager {ne}"
    assert _kernels(opc) < _kernels(ope), (
        f"compiled kernels {_kernels(opc)} not < eager {_kernels(ope)}"
    )
    # the two matmuls (QK, PV) survive compile; the elementwise chain collapses.
    assert opc.get("Matmul", 0) == 2
    assert any(k.startswith("Compiled") for k in opc), "no fused Compiled node in the core tape"


def _load_bisect():
    spec = importlib.util.spec_from_file_location(
        "w97_bisect", _REPO / "scripts" / "deepseek_v41" / "metal_decode_attn_bisect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _decode_ids(model, ops, prompt_ids, steps):
    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    tok = ops.argmax_last(logits)
    ids, last_logits = [tok], [logits[0, -1]]
    for _ in range(steps):
        logits = model(ops.input([[tok]]), cache=cache)
        ops.sync(logits)
        tok = ops.argmax_last(logits)
        ids.append(tok)
        last_logits.append(logits[0, -1])
    return ids, last_logits


def test_core_compile_cache_bounded_and_greedy_delta_64_steps(monkeypatch):
    """64 decode steps through a tiny full model: greedy token ids OFF vs ON, the
    logit max|Δ| (labelled rounding-class), and the core compile-cache growth."""
    bis = _load_bisect()
    # selected-keys path on (so _sparse_attend_selected -- and thus the core -- runs);
    # attention compile tapes off (isolate the CORE lever); CPU.
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", False)

    steps = 64
    model, _args = bis._build_tiny_full_model(seed=3)
    ops = bis._TinyOps()
    prompt_ids = list(range(1, 41))

    # control: core-compile OFF
    monkeypatch.setenv(dsv41._ATTN_CORE_COMPILE_ENV, "0")
    dsv41._ATTN_CORE_COMPILED.clear()
    ids_off, log_off = _decode_ids(model, ops, prompt_ids, steps)
    assert len(dsv41._ATTN_CORE_COMPILED) == 0, "core tape built with the lever OFF"

    # candidate: core-compile ON
    monkeypatch.setenv(dsv41._ATTN_CORE_COMPILE_ENV, "1")
    dsv41._ATTN_CORE_COMPILED.clear()
    ids_on, log_on = _decode_ids(model, ops, prompt_ids, steps)

    # compile-cache BOUNDED: a handful of geometries (per CSA k), never per-token.
    n_tapes = len(dsv41._ATTN_CORE_COMPILED)
    assert 1 <= n_tapes <= 8, f"core tape count {n_tapes} not in [1, 8] over {steps} steps"

    # labelled Δ: rounding-class (the compiled core reassociates fp32).
    max_abs = 0.0
    for a, bb in zip(log_off, log_on):
        max_abs = max(max_abs, float(mx.max(mx.abs(a - bb)).item()))
    assert max_abs <= 1e-2, f"logit max|Δ| {max_abs:.3e} exceeds the rounding-class band"

    # greedy-argmax identity on this seed (the ship bar is the GPU token sha; here
    # the tiny model must not flip a greedy tie under the rounding-class core).
    assert ids_on == ids_off, (
        f"greedy token ids diverged (max|Δ|={max_abs:.3e}); if a real tie flips on GPU "
        "this is a rounding-class flip to LABEL, not a bug"
    )


def test_core_compile_off_is_default():
    assert dsv41._resolve_attn_core_compile(raw="") is False
    assert dsv41._resolve_attn_core_compile(raw="0") is False
    assert dsv41._resolve_attn_core_compile(raw="1") is True
    with pytest.raises(ValueError):
        dsv41._resolve_attn_core_compile(raw="banana")


def test_attn_core_compile_engagement_counts_compiled_vs_eager(monkeypatch):
    """W97 review item 3: the engagement counter distinguishes 'the compiled tape
    ran' (compiled > 0) from 'fell through to eager' (compiled == 0), the same
    'did it actually run?' proof the ab receipt carries as
    ``attn_core_compile_engagement`` (mirroring K29's ``decode_attn_kernel_engagement``)
    so a GPU window cannot credit a delta to a tape that never engaged."""
    bis = _load_bisect()
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", False)

    steps = 8
    model, _args = bis._build_tiny_full_model(seed=5)
    ops = bis._TinyOps()
    prompt_ids = list(range(1, 41))

    # Lever OFF: every selected-key core call runs eager -- compiled must be 0.
    monkeypatch.setenv(dsv41._ATTN_CORE_COMPILE_ENV, "0")
    dsv41._ATTN_CORE_COMPILED.clear()
    dsv41._reset_attn_core_compile_calls()
    _decode_ids(model, ops, prompt_ids, steps)
    off = dsv41._attn_core_compile_calls()
    assert off["compiled"] == 0, f"lever OFF ran the compiled tape: {off}"
    assert off["eager"] > 0, f"selected path never ran the eager core: {off}"

    # Lever ON: the decode (small-M) forwards run the compiled tape -- compiled > 0.
    monkeypatch.setenv(dsv41._ATTN_CORE_COMPILE_ENV, "1")
    dsv41._ATTN_CORE_COMPILED.clear()
    dsv41._reset_attn_core_compile_calls()
    _decode_ids(model, ops, prompt_ids, steps)
    on = dsv41._attn_core_compile_calls()
    assert on["compiled"] > 0, f"lever ON but the compiled tape never engaged: {on}"
