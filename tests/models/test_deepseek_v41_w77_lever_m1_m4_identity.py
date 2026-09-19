"""W77 -- CPU fake-model per-lever M=1 (AR decode) vs M=K+1 (verify) identity.

The cell16k DSpark stream diverged from AR at index 228 on the real 16K box.  The
greedy verify is authoritative, so a divergence is a greedy argmax flip driven by
the target forward returning different logits at the M=1 vs M=K+1 row count.  Two
possibilities: a genuine LOGIC bug in an M>1 path (a wrong row/position, a
mis-ordered batched append, a mis-unsorted gather) -- which shows a LARGE delta
even in exact fp32 arithmetic -- or a Metal/bf16 kernel-dispatch reassociation
(rounding class) that flips a near-tie -- which the CPU double cannot reproduce and
which David rules acceptable ([[dsv41-inexact-ok-if-tie-flips]]).

Gates:
  1. ``_GrowBuffer`` (kv_chunk_grow ON) window/compress/index views are
     BYTE-identical to the plain ``_grow`` concatenate store after a multi-row
     (M=4 verify) append -- the prime "wrong row/position under kv_chunk_grow"
     suspect, checkable exactly.  Plus a NEGATIVE CONTROL: a deliberately
     row-reversed batched append is detected (the check has power).
  2. Per-lever M=1 vs M=4 forward identity on the tiny fp32 model: every cell16k
     lever's max |Δlogit| stays at fp32 epsilon (no structural break / logic bug).
     Plus a NEGATIVE CONTROL: an injected wrong-order M=4 verify append makes the
     delta explode (so the null result is credible, not a blind spot).

Self-contained: shrunk seeded config, CPU device, no downloads/checkpoint/experts.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _load_probe():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "deepseek_v41" / "w77_lever_identity_probe.py"
    )
    spec = importlib.util.spec_from_file_location("_w77_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_probe()
# fp32-noise ceiling: a genuine logic bug (wrong row/gather) is O(0.1-10); fp32
# reassociation between the decode and verify forwards is ~1e-6.
FP32_NOISE = 1e-4


# ---------------------------------------------------------------------------
# 1. kv_chunk_grow: _GrowBuffer multi-row append is byte-identical to _grow
# ---------------------------------------------------------------------------
from mtplx.models.deepseek_v41_cache import _GrowBuffer, _grow  # noqa: E402


def _rand(b, s, d, seed):
    mx.random.seed(seed)
    return mx.random.normal((b, s, d))


def test_growbuffer_multirow_append_byte_identical():
    # Emulate a verify: seed 16 prefill rows, then a 4-row block append (M=4), the
    # exact shape the K+1 verify writes into the window/compress/index lanes.
    prefill = _rand(1, 16, 8, seed=1)
    block = _rand(1, 4, 8, seed=2)

    # plain concatenate store (chunk_grow OFF, the shipped path)
    plain = _grow(_grow(None, prefill), block)

    # chunk-grown store (chunk_grow ON)
    gb = _GrowBuffer()
    gb.append(prefill)
    gb.append(block)
    grown = gb.view()

    mx.eval(plain, grown)
    assert grown.shape == plain.shape
    assert bool(mx.all(grown == plain)), "chunk-grow multi-row append is not byte-identical"

    # also identical if the block arrives as 4 separate 1-row appends (decode) vs
    # one 4-row append (verify): the buffer must not care.
    gb2 = _GrowBuffer()
    gb2.append(prefill)
    for i in range(4):
        gb2.append(block[:, i : i + 1])
    mx.eval(gb2.view())
    assert bool(mx.all(gb2.view() == plain))


def test_growbuffer_negative_control_detects_reordered_append():
    # NEGATIVE CONTROL: a batched append that writes the block rows reversed is a
    # plausible "wrong row/position under kv_chunk_grow at M=4" bug; the byte
    # compare must catch it (else gate 1 has no power).
    prefill = _rand(1, 16, 8, seed=1)
    block = _rand(1, 4, 8, seed=2)
    plain = _grow(_grow(None, prefill), block)

    gb = _GrowBuffer()
    gb.append(prefill)
    gb.append(block[:, ::-1])  # corrupted order
    mx.eval(gb.view())
    assert not bool(mx.all(gb.view() == plain))


# ---------------------------------------------------------------------------
# 2. per-lever M=1 vs M=4 forward identity on the tiny fp32 model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _ctx():
    rng = np.random.default_rng(0)
    return [int(x) for x in rng.integers(1, PROBE.VOCAB - 1, size=20)]


def test_baseline_m1_vs_m4_is_fp32_noise(_ctx):
    out = PROBE.run_lever("baseline", {}, False, _ctx, seed=0, block=4)
    assert out["max_abs_logit_delta"] < FP32_NOISE
    assert not out["argmax_flip"]


@pytest.mark.parametrize("label,env,head_bf16,engages,note", PROBE.LEVERS)
def test_each_lever_no_structural_break(label, env, head_bf16, engages, note, _ctx):
    # No cell16k lever introduces a LARGE (logic-bug) delta between the M=1 decode
    # forward and the M=4 verify forward in exact fp32 arithmetic.
    out = PROBE.run_lever(label, env, head_bf16, _ctx, seed=0, block=4)
    assert out["max_abs_logit_delta"] < FP32_NOISE, (
        f"{label}: max|Δlogit|={out['max_abs_logit_delta']:.3e} exceeds the fp32-noise "
        f"ceiling -- a structural M=1-vs-M=4 break, investigate ({note})"
    )


def test_full_cell16k_stack_no_structural_break(_ctx):
    out = PROBE.run_lever("cell16k", PROBE.CELL16K_ENV, True, _ctx, seed=0, block=4)
    assert out["max_abs_logit_delta"] < FP32_NOISE


def test_negative_control_injected_m4_append_bug_is_caught(_ctx, monkeypatch):
    # NEGATIVE CONTROL: inject a wrong-order batched (verify-block) append into the
    # KV window lane -- reverse only small multi-row appends (1 < n <= block), so
    # the M=1 decode (n=1) and the shared prefill (n=16) are untouched and only the
    # M=4 verify block is corrupted.  The M=1-vs-M=4 delta must then explode, which
    # proves the null result above is a real "no bug" finding, not a blind harness.
    import mtplx.models.deepseek_v41_cache as cache_mod

    orig_append = cache_mod._GrowBuffer.append

    def buggy_append(self, new):
        if new is not None and new.ndim >= 2 and 1 < int(new.shape[1]) <= 8:
            new = new[:, ::-1]
        return orig_append(self, new)

    monkeypatch.setattr(cache_mod._GrowBuffer, "append", buggy_append)
    out = PROBE.run_lever("kv_chunk_grow(BUGGY)", {"MTPLX_DSV41_KV_CHUNK_GROW": "1"},
                          False, _ctx, seed=0, block=4)
    assert out["max_abs_logit_delta"] > 1e-2, (
        "the negative control did not surface -- the M=1-vs-M=4 probe would miss a "
        "real wrong-row bug"
    )
