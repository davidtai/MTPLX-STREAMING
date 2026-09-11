"""Gate for the DSV4.1 Hyper-Connection Sinkhorn Metal fast path
(``MTPLX_DSV41_SINKHORN_METAL``, kernel-ledger K3 / W32).

DSV4.1's backbone runs the Sinkhorn alternating-normalisation loop twice per
layer per token (the attention HC mix and the ffn HC mix,
:meth:`deepseek_v41.DecoderLayer._mixes` -> :func:`deepseek_v41._hc_split_sinkhorn`);
at 40 layers that is 80 calls per token, each ~119 tiny graph primitives on a
``[..., 4, 4]`` tensor.  This module ports DeepSeek-V4's one-dispatch Sinkhorn
Metal kernel (``deepseek_v4._sinkhorn_metal_kernel``, reached through
``_sinkhorn_kernel_apply``) so each call collapses to a single dispatch, with the
identical fp32 arithmetic in the identical order as the stock recurrence
(``_sinkhorn_ops``) it replaces.

These are the CPU-side dispatch-logic gates: they run entirely on the CPU device
(every worker test pins ``mx.set_default_device(mx.cpu)``) and never dispatch a
Metal kernel — the GPU-routing case is proven by monkeypatching the kernel
callable to a spy.  The numeric kernel-vs-recurrence parity, which *does* need a
Metal GPU, is :func:`test_sinkhorn_kernel_parity_gpu`, skipped unless
``MTPLX_GPU_PARITY=1`` and meant to be run inside a GPU window.

Self-contained: no downloads, no model artifacts, no torch.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

# Metal cannot run on the CPU device; every dispatch-logic test below is CPU-only.
mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41 as V41  # noqa: E402
from mtplx.models.deepseek_v4 import (  # noqa: E402
    _sinkhorn_kernel_apply,
    _sinkhorn_ops,
)

HC, ITERS, EPS = 4, 20, 1e-6
MIX_HC = (2 + HC) * HC  # (2 + hc) * hc mix columns -> pre[hc], post[hc], comb[hc*hc]


def _rand_comb(shape=(2, 7, HC, HC), seed=0, scale=2.5, dtype=mx.float32):
    rng = np.random.default_rng(seed)
    return mx.array((rng.standard_normal(shape) * scale).astype(np.float32)).astype(dtype)


def _rand_mixes(shape=(2, 7), seed=1):
    rng = np.random.default_rng(seed)
    mixes = mx.array((rng.standard_normal((*shape, MIX_HC))).astype(np.float32))
    scale = mx.array((rng.standard_normal((3,))).astype(np.float32))
    base = mx.array((rng.standard_normal((MIX_HC,))).astype(np.float32))
    return mixes, scale, base


@pytest.fixture(autouse=True)
def _hermetic_flag(monkeypatch):
    """No ambient ``MTPLX_DSV41_SINKHORN_METAL`` leaks into a default-OFF assert
    (e.g. running the whole suite under the flag forced on to exercise the CPU
    fallback); the CPU device is restored so nothing leaks into later suites."""
    monkeypatch.delenv(V41._SINKHORN_METAL_ENV, raising=False)
    saved_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(saved_dev)


# ---------------------------------------------------------------------------
# flag parsing (read at use, never frozen at import)
# ---------------------------------------------------------------------------
def test_flag_defaults_off_and_parses_truthy(monkeypatch):
    monkeypatch.delenv(V41._SINKHORN_METAL_ENV, raising=False)
    assert V41._sinkhorn_metal_enabled() is False  # unset -> OFF
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv(V41._SINKHORN_METAL_ENV, v)
        assert V41._sinkhorn_metal_enabled() is True
    for v in ("0", "false", "no", "off", "", "auto"):
        monkeypatch.setenv(V41._SINKHORN_METAL_ENV, v)
        assert V41._sinkhorn_metal_enabled() is False


def test_flag_is_read_at_use_not_frozen(monkeypatch):
    """A stamp set after import (as the serving harness does) is honoured."""
    monkeypatch.delenv(V41._SINKHORN_METAL_ENV, raising=False)
    assert V41._sinkhorn_metal_enabled() is False
    monkeypatch.setenv(V41._SINKHORN_METAL_ENV, "1")
    assert V41._sinkhorn_metal_enabled() is True


# ---------------------------------------------------------------------------
# split is a byte-for-byte port of the stock V4 split (decode numerics unchanged)
# ---------------------------------------------------------------------------
def test_split_flag_off_bit_identical_to_stock():
    """With the flag off, ``_hc_split_sinkhorn`` reproduces the exact V4
    ``hc_split_sinkhorn`` output: the same pre/post plus the stock recurrence."""
    mixes, scale, base = _rand_mixes()
    pre, post, comb = V41._hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)

    r_pre = mx.sigmoid(mixes[..., :HC] * scale[0] + base[:HC]) + EPS
    r_post = 2.0 * mx.sigmoid(mixes[..., HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    r_comb = mixes[..., 2 * HC :] * scale[2] + base[2 * HC :]
    r_comb = r_comb.reshape(*r_comb.shape[:-1], HC, HC)
    r_comb = _sinkhorn_ops(r_comb, ITERS, EPS)
    mx.eval(pre, post, comb, r_pre, r_post, r_comb)

    assert mx.array_equal(pre, r_pre)
    assert mx.array_equal(post, r_post)
    assert mx.array_equal(comb, r_comb)
    assert comb.shape == (2, 7, HC, HC)


def test_split_routes_comb_through_normalise(monkeypatch):
    """The split hands the reshaped ``[..., hc, hc]`` comb to
    ``_sinkhorn_normalise`` (the dispatch point) and returns exactly its result;
    pre/post are untouched by the route."""
    seen = {}

    def spy(comb, hc, iters, eps):
        seen["shape"] = comb.shape
        seen["args"] = (hc, iters, eps)
        return comb + 100.0  # sentinel the split must return verbatim

    monkeypatch.setattr(V41, "_sinkhorn_normalise", spy)
    mixes, scale, base = _rand_mixes()
    pre, post, comb = V41._hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)

    raw = mixes[..., 2 * HC :] * scale[2] + base[2 * HC :]
    raw = raw.reshape(*raw.shape[:-1], HC, HC)
    mx.eval(comb, raw)
    assert seen["shape"] == (2, 7, HC, HC)
    assert seen["args"] == (HC, ITERS, EPS)
    assert mx.array_equal(comb, raw + 100.0)  # exactly the spy's output


# ---------------------------------------------------------------------------
# dispatch: flag off -> recurrence, always
# ---------------------------------------------------------------------------
def test_dispatch_flag_off_uses_recurrence(monkeypatch):
    monkeypatch.delenv(V41._SINKHORN_METAL_ENV, raising=False)
    calls = {"kernel": 0, "ops": 0}

    def kernel_spy(*a, **k):
        calls["kernel"] += 1
        raise AssertionError("kernel path taken with flag off")

    def ops_spy(comb, iters, eps):
        calls["ops"] += 1
        return _sinkhorn_ops(comb, iters, eps)

    monkeypatch.setattr(V41, "_sinkhorn_kernel_apply", kernel_spy)
    monkeypatch.setattr(V41, "_sinkhorn_ops", ops_spy)

    assert V41._sinkhorn_use_kernel() is False
    comb = _rand_comb()
    out = V41._sinkhorn_normalise(comb, HC, ITERS, EPS)
    mx.eval(out)
    assert calls == {"kernel": 0, "ops": 1}


# ---------------------------------------------------------------------------
# dispatch: flag on + CPU device -> recurrence (the flag is inert off-GPU)
# ---------------------------------------------------------------------------
def test_dispatch_flag_on_cpu_uses_recurrence(monkeypatch):
    monkeypatch.setenv(V41._SINKHORN_METAL_ENV, "1")
    mx.set_default_device(mx.cpu)
    calls = {"kernel": 0, "ops": 0}

    def kernel_spy(*a, **k):
        calls["kernel"] += 1
        raise AssertionError("Metal kernel dispatched on the CPU device")

    def ops_spy(comb, iters, eps):
        calls["ops"] += 1
        return _sinkhorn_ops(comb, iters, eps)

    monkeypatch.setattr(V41, "_sinkhorn_kernel_apply", kernel_spy)
    monkeypatch.setattr(V41, "_sinkhorn_ops", ops_spy)

    # Flag is truthy, but the device gate keeps it on the recurrence.
    assert V41._sinkhorn_metal_enabled() is True
    assert V41._sinkhorn_use_kernel() is False

    comb = _rand_comb(seed=3)
    got = V41._sinkhorn_normalise(comb, HC, ITERS, EPS)
    want = _sinkhorn_ops(comb, ITERS, EPS)
    mx.eval(got, want)
    assert calls == {"kernel": 0, "ops": 1}  # kernel never, ops once
    assert mx.array_equal(got, want)


# ---------------------------------------------------------------------------
# dispatch: flag on + GPU device -> kernel path (proven by a spy, no Metal run)
# ---------------------------------------------------------------------------
def test_dispatch_flag_on_gpu_uses_kernel(monkeypatch):
    """Flag on + a GPU default device selects the kernel callable.  Proven with
    the kernel monkeypatched to a spy so no Metal is dispatched on this (possibly
    CPU-pinned, GPU-locked) box; the array is built before the device is faked, so
    nothing real ever leaves the CPU."""
    comb = _rand_comb(seed=5)  # a real CPU array, built before the device is faked

    monkeypatch.setenv(V41._SINKHORN_METAL_ENV, "1")
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)

    seen = {"kernel": 0, "ops": 0, "args": None}

    def kernel_spy(comb_in, hc, iters, eps):
        seen["kernel"] += 1
        seen["args"] = (comb_in.shape, hc, iters, eps)
        return comb_in + 7.0  # sentinel; never touches Metal

    def ops_spy(*a, **k):
        seen["ops"] += 1
        raise AssertionError("recurrence taken though flag on + GPU")

    monkeypatch.setattr(V41, "_sinkhorn_kernel_apply", kernel_spy)
    monkeypatch.setattr(V41, "_sinkhorn_ops", ops_spy)

    assert V41._sinkhorn_use_kernel() is True
    out = V41._sinkhorn_normalise(comb, HC, ITERS, EPS)
    mx.eval(out)
    assert seen["kernel"] == 1 and seen["ops"] == 0
    assert seen["args"] == ((2, 7, HC, HC), HC, ITERS, EPS)
    assert mx.array_equal(out, comb + 7.0)  # exactly the (spied) kernel's output


def test_dispatch_gpu_but_flag_off_uses_recurrence(monkeypatch):
    """The device alone must not arm the kernel: GPU default but flag unset stays
    on the recurrence."""
    monkeypatch.delenv(V41._SINKHORN_METAL_ENV, raising=False)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert V41._sinkhorn_use_kernel() is False


# ---------------------------------------------------------------------------
# numeric parity: the Metal kernel reproduces the recurrence (GPU window only)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("MTPLX_GPU_PARITY") != "1",
    reason="GPU parity: run inside a GPU window with MTPLX_GPU_PARITY=1",
)
def test_sinkhorn_kernel_parity_gpu():
    """One kernel dispatch == the 40-pass stock recurrence, argmax exact.

    fp32 is the production dtype (``_mixes`` casts comb to fp32) and is gated
    bit-identical (1e-6).  bf16 is gated on argmax stability with the observed
    spread recorded (the kernel computes in fp32 registers, the recurrence in
    bf16, so a bf16 value delta is expected; argmax is the invariant)."""
    saved_dev = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        # fp32: bit-identical.
        comb32 = _rand_comb(seed=11, dtype=mx.float32)
        ref32 = _sinkhorn_ops(comb32, ITERS, EPS)
        got32 = _sinkhorn_kernel_apply(comb32, HC, ITERS, EPS)
        mx.eval(ref32, got32)
        assert bool(mx.all(mx.isfinite(got32))), "fp32 kernel produced non-finite comb"
        assert bool(mx.all(mx.argmax(got32, axis=-1) == mx.argmax(ref32, axis=-1))), (
            "fp32 kernel argmax moved"
        )
        d32 = float(mx.max(mx.abs(got32 - ref32)))
        assert d32 <= 1e-6, f"fp32 kernel vs recurrence max|d|={d32:.2e} > 1e-6"

        # bf16: argmax exact is the hard gate; the value spread is recorded.
        comb16 = _rand_comb(seed=11, dtype=mx.bfloat16)
        ref16 = _sinkhorn_ops(comb16, ITERS, EPS)
        got16 = _sinkhorn_kernel_apply(comb16, HC, ITERS, EPS)
        mx.eval(ref16, got16)
        assert bool(mx.all(mx.isfinite(got16))), "bf16 kernel produced non-finite comb"
        assert bool(mx.all(mx.argmax(got16, axis=-1) == mx.argmax(ref16, axis=-1))), (
            "bf16 kernel argmax moved"
        )
        d16 = float(
            mx.max(mx.abs(got16.astype(mx.float32) - ref16.astype(mx.float32)))
        )
        # Recorded tripwire, not the correctness gate (that is argmax, above).
        assert d16 <= 5e-2, f"bf16 kernel vs recurrence spread max|d|={d16:.2e} > 5e-2"
        print(f"[W32/K3 parity] fp32 max|d|={d32:.2e}  bf16 max|d|={d16:.2e}")

        # The armed dispatcher (flag on + GPU) also matches the recurrence in fp32.
        os.environ[V41._SINKHORN_METAL_ENV] = "1"
        try:
            assert V41._sinkhorn_use_kernel() is True
            via = V41._sinkhorn_normalise(comb32, HC, ITERS, EPS)
            mx.eval(via)
            assert float(mx.max(mx.abs(via - ref32))) <= 1e-6, "dispatcher != recurrence"
        finally:
            os.environ.pop(V41._SINKHORN_METAL_ENV, None)
    finally:
        mx.set_default_device(saved_dev)
