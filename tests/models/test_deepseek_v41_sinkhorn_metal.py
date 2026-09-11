"""Gate for the DSV4.1 Hyper-Connection Sinkhorn Metal fast path
(``MTPLX_DSV41_SINKHORN_METAL``, kernel-ledger K3 / W32, diagnostics W38).

DSV4.1's backbone runs the Sinkhorn alternating-normalisation loop twice per layer
per token (the attention HC mix and the ffn HC mix,
``DecoderLayer._mixes`` / the compiled ``_hc_mixes_split`` -> ``hc_split_sinkhorn``);
at 40 layers that is 80 calls per token, each ~119 tiny graph primitives on a
``[..., 4, 4]`` tensor.  This ports DeepSeek-V4's one-dispatch Sinkhorn Metal
kernel so each call collapses to a single dispatch, with the identical fp32
arithmetic in the identical order as the stock recurrence (``_sinkhorn_ops``).

CPU dispatch + engagement gates run entirely on the CPU device (worker tests pin
``mx.set_default_device(mx.cpu)``) and never dispatch a Metal kernel -- the
GPU-routing case is proven by monkeypatching the kernel callable to a spy.  The
numeric kernel-vs-recurrence parity, which needs a Metal GPU, is
:func:`test_sinkhorn_kernel_parity_gpu`: skipped unless ``MTPLX_GPU_PARITY=1``,
self-diagnosing (writes a JSON receipt to ``MTPLX_PARITY_RECEIPT`` and prints the
same diag so a window captures the failure detail even when stdout is truncated).

Self-contained: no downloads, no model artifacts, no torch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

# Metal cannot run on the CPU device; every dispatch/engagement test below is CPU.
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


def _maxabs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _argmax_mismatch(a, b):
    return int((mx.argmax(a, axis=-1) != mx.argmax(b, axis=-1)).sum())


def _write_parity_receipt(diag: dict):
    """Write ``diag`` as JSON to ``MTPLX_PARITY_RECEIPT`` if the env is set.

    Never raises: a receipt-IO error is printed, not propagated, so it cannot
    mask (or manufacture) a test result.  Returns the path written, or None.
    """
    path = os.environ.get("MTPLX_PARITY_RECEIPT")
    if not path:
        return None
    try:
        p = Path(path)
        if p.parent and str(p.parent):
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(diag, indent=2, sort_keys=True))
        return str(p)
    except Exception as exc:  # pragma: no cover - IO edge
        print(f"[W38/K3] receipt write failed: {exc!r}")
        return None


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """No ambient flag leaks into a default-OFF assert; the CPU device is restored
    and the engagement counters are zeroed so nothing leaks between tests."""
    monkeypatch.delenv(V41._SINKHORN_METAL_ENV, raising=False)
    saved_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    V41._reset_sinkhorn_kernel_calls()
    yield
    mx.set_default_device(saved_dev)
    V41._reset_sinkhorn_kernel_calls()


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
    monkeypatch.delenv(V41._SINKHORN_METAL_ENV, raising=False)
    assert V41._sinkhorn_metal_enabled() is False
    monkeypatch.setenv(V41._SINKHORN_METAL_ENV, "1")
    assert V41._sinkhorn_metal_enabled() is True


# ---------------------------------------------------------------------------
# split is a byte-for-byte port of the stock V4 split (decode numerics unchanged)
# ---------------------------------------------------------------------------
def test_split_flag_off_bit_identical_to_stock():
    mixes, scale, base = _rand_mixes()
    pre, post, comb = V41.hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)

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
    seen = {}

    def spy(comb, hc, iters, eps):
        seen["shape"] = comb.shape
        seen["args"] = (hc, iters, eps)
        return comb + 100.0

    monkeypatch.setattr(V41, "_sinkhorn_normalise", spy)
    mixes, scale, base = _rand_mixes()
    pre, post, comb = V41.hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)

    raw = mixes[..., 2 * HC :] * scale[2] + base[2 * HC :]
    raw = raw.reshape(*raw.shape[:-1], HC, HC)
    mx.eval(comb, raw)
    assert seen["shape"] == (2, 7, HC, HC)
    assert seen["args"] == (HC, ITERS, EPS)
    assert mx.array_equal(comb, raw + 100.0)


# ---------------------------------------------------------------------------
# dispatch: flag off -> recurrence, always
# ---------------------------------------------------------------------------
def test_dispatch_flag_off_uses_recurrence(monkeypatch):
    monkeypatch.delenv(V41._SINKHORN_METAL_ENV, raising=False)

    def kernel_spy(*a, **k):
        raise AssertionError("kernel path taken with flag off")

    monkeypatch.setattr(V41, "_sinkhorn_kernel_apply", kernel_spy)
    assert V41._sinkhorn_use_kernel() is False
    comb = _rand_comb()
    out = V41._sinkhorn_normalise(comb, HC, ITERS, EPS)
    mx.eval(out)
    assert V41._sinkhorn_kernel_calls() == {"kernel": 0, "recurrence": 1}
    assert mx.array_equal(out, _sinkhorn_ops(comb, ITERS, EPS))


def test_dispatch_flag_on_cpu_uses_recurrence(monkeypatch):
    monkeypatch.setenv(V41._SINKHORN_METAL_ENV, "1")
    mx.set_default_device(mx.cpu)

    def kernel_spy(*a, **k):
        raise AssertionError("Metal kernel dispatched on the CPU device")

    monkeypatch.setattr(V41, "_sinkhorn_kernel_apply", kernel_spy)
    assert V41._sinkhorn_metal_enabled() is True
    assert V41._sinkhorn_use_kernel() is False

    comb = _rand_comb(seed=3)
    got = V41._sinkhorn_normalise(comb, HC, ITERS, EPS)
    want = _sinkhorn_ops(comb, ITERS, EPS)
    mx.eval(got, want)
    assert V41._sinkhorn_kernel_calls() == {"kernel": 0, "recurrence": 1}
    assert mx.array_equal(got, want)


def test_dispatch_flag_on_gpu_uses_kernel(monkeypatch):
    """Flag on + a GPU default device selects the kernel callable.  Proven with a
    spy so no Metal is dispatched on this (CPU-pinned, GPU-locked) box; the array
    is built before the device is faked, so nothing real leaves the CPU."""
    comb = _rand_comb(seed=5)  # a real CPU array built before the device is faked

    monkeypatch.setenv(V41._SINKHORN_METAL_ENV, "1")
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)

    seen = {"args": None}

    def kernel_spy(comb_in, hc, iters, eps):
        seen["args"] = (comb_in.shape, str(comb_in.dtype), hc, iters, eps)
        return comb_in + 7.0

    def ops_spy(*a, **k):
        raise AssertionError("recurrence taken though flag on + GPU")

    monkeypatch.setattr(V41, "_sinkhorn_kernel_apply", kernel_spy)
    monkeypatch.setattr(V41, "_sinkhorn_ops", ops_spy)

    assert V41._sinkhorn_use_kernel() is True
    out = V41._sinkhorn_normalise(comb, HC, ITERS, EPS)
    mx.eval(out)
    assert seen["args"] == ((2, 7, HC, HC), "mlx.core.float32", HC, ITERS, EPS)
    assert mx.array_equal(out, comb + 7.0)
    assert V41._sinkhorn_kernel_calls() == {"kernel": 1, "recurrence": 0}


def test_dispatch_gpu_but_flag_off_uses_recurrence(monkeypatch):
    monkeypatch.delenv(V41._SINKHORN_METAL_ENV, raising=False)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert V41._sinkhorn_use_kernel() is False


# ---------------------------------------------------------------------------
# engagement counters (W38): confirm the kernel actually ran, not a silent fallback
# ---------------------------------------------------------------------------
def test_engagement_counter_kernel_branch(monkeypatch):
    """The kernel branch increments the kernel counter (and the route-stage probe
    stage) but not the recurrence counter; the recurrence branch is the inverse.
    Proven with a spy kernel, no Metal."""
    comb = _rand_comb(seed=6)
    monkeypatch.setattr(V41, "_sinkhorn_use_kernel", lambda: True)
    monkeypatch.setattr(V41, "_sinkhorn_kernel_apply", lambda c, h, i, e: c + 0.0)
    # Arm the (import-frozen) probe for this call so the stage is emitted too.
    monkeypatch.setattr(V41._route_probe, "ENABLED", True)
    V41._route_probe._COUNTS.clear()

    V41._reset_sinkhorn_kernel_calls()
    out = V41._sinkhorn_normalise(comb, HC, ITERS, EPS)
    mx.eval(out)
    assert V41._sinkhorn_kernel_calls() == {"kernel": 1, "recurrence": 0}
    assert V41._route_probe._COUNTS.get("hc.sinkhorn_kernel", 0) == 1
    assert V41._route_probe._COUNTS.get("hc.sinkhorn_recurrence", 0) == 0


def test_engagement_counter_recurrence_branch(monkeypatch):
    comb = _rand_comb(seed=6)
    monkeypatch.setattr(V41, "_sinkhorn_use_kernel", lambda: False)
    monkeypatch.setattr(V41._route_probe, "ENABLED", True)
    V41._route_probe._COUNTS.clear()

    V41._reset_sinkhorn_kernel_calls()
    out = V41._sinkhorn_normalise(comb, HC, ITERS, EPS)
    mx.eval(out)
    assert V41._sinkhorn_kernel_calls() == {"kernel": 0, "recurrence": 1}
    assert V41._route_probe._COUNTS.get("hc.sinkhorn_kernel", 0) == 0
    assert V41._route_probe._COUNTS.get("hc.sinkhorn_recurrence", 0) == 1


def test_engagement_counter_reset():
    comb = _rand_comb(seed=1)
    V41._sinkhorn_normalise(comb, HC, ITERS, EPS)  # recurrence (flag off default)
    assert V41._sinkhorn_kernel_calls()["recurrence"] >= 1
    V41._reset_sinkhorn_kernel_calls()
    assert V41._sinkhorn_kernel_calls() == {"kernel": 0, "recurrence": 0}


def test_kernel_branch_upcasts_non_fp32(monkeypatch):
    """The kernel path never processes a bf16 buffer: a non-fp32 comb is upcast to
    fp32 for the (fp32-validated) kernel and the result cast back to comb.dtype."""
    seen = {}

    def kernel_spy(comb_in, hc, iters, eps):
        seen["in_dtype"] = str(comb_in.dtype)
        return comb_in + 0.0

    monkeypatch.setattr(V41, "_sinkhorn_use_kernel", lambda: True)
    monkeypatch.setattr(V41, "_sinkhorn_kernel_apply", kernel_spy)

    comb16 = _rand_comb(seed=2, dtype=mx.bfloat16)
    out = V41._sinkhorn_normalise(comb16, HC, ITERS, EPS)
    mx.eval(out)
    assert seen["in_dtype"] == "mlx.core.float32"  # upcast before the kernel
    assert out.dtype == mx.bfloat16                 # cast back to the I/O dtype


# ---------------------------------------------------------------------------
# receipt helper (CPU): the diagnostic JSON is written when the env is set
# ---------------------------------------------------------------------------
def test_receipt_helper_writes_json(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "parity.json"
    monkeypatch.setenv("MTPLX_PARITY_RECEIPT", str(path))
    diag = {"test": "unit", "arms": {"fp32": {"passed": True, "max_abs_d": 1e-7}}}
    written = _write_parity_receipt(diag)
    assert written == str(path)
    assert json.loads(path.read_text()) == diag


def test_receipt_helper_noop_without_env(monkeypatch):
    monkeypatch.delenv("MTPLX_PARITY_RECEIPT", raising=False)
    assert _write_parity_receipt({"x": 1}) is None


# ---------------------------------------------------------------------------
# numeric parity: the Metal kernel reproduces the recurrence (GPU window only)
# ---------------------------------------------------------------------------
def _measure_fp32() -> dict:
    """fp32 (the production dtype): the kernel is bit-identical to the recurrence."""
    d = {"gate": "kernel_vs_recurrence_fp32", "tol": 1e-6, "error": None}
    try:
        comb = _rand_comb(seed=11, dtype=mx.float32)
        ref = _sinkhorn_ops(comb, ITERS, EPS)
        got = _sinkhorn_kernel_apply(comb, HC, ITERS, EPS)
        mx.eval(ref, got)
        d["finite"] = bool(mx.all(mx.isfinite(got)))
        d["max_abs_d"] = _maxabs(got, ref)
        d["argmax_rows"] = int(ref[..., 0].size)
        d["argmax_mismatch"] = _argmax_mismatch(got, ref)
        d["passed"] = bool(
            d["finite"] and d["argmax_mismatch"] == 0 and d["max_abs_d"] <= d["tol"]
        )
    except Exception as exc:  # pragma: no cover - GPU-only
        d["passed"] = False
        d["error"] = repr(exc)
    return d


def _measure_bf16() -> dict:
    """bf16: the kernel computes in fp32 and stores bf16, so the meaningful oracle
    is the fp32 recurrence on the same values rounded to bf16 -- NOT the bf16
    recurrence (which accumulates in bf16 and legitimately diverges; the naive
    kernel-vs-bf16-recurrence comparison was the W32 test's fault, recorded below).
    """
    d = {
        "primary_gate": "dispatch_vs_fp32recurrence_rounded_bf16",
        "tol": 1.6e-2,
        "error": None,
    }
    try:
        comb16 = _rand_comb(seed=11, dtype=mx.bfloat16)
        comb32 = comb16.astype(mx.float32)  # exact upcast of the bf16 values
        ref_fp32 = _sinkhorn_ops(comb32, ITERS, EPS)
        ref = ref_fp32.astype(mx.bfloat16)  # fp32 sinkhorn, rounded to bf16
        # production dispatch path (armed on GPU): upcasts internally, bf16 out
        os.environ[V41._SINKHORN_METAL_ENV] = "1"
        try:
            d["armed"] = bool(V41._sinkhorn_use_kernel())
            got = V41._sinkhorn_normalise(comb16, HC, ITERS, EPS)
        finally:
            os.environ.pop(V41._SINKHORN_METAL_ENV, None)
        mx.eval(ref, got, ref_fp32)
        d["out_dtype"] = str(got.dtype)
        d["finite"] = bool(mx.all(mx.isfinite(got)))
        d["max_abs_d"] = _maxabs(got, ref)
        d["argmax_rows"] = int(ref[..., 0].size)
        d["argmax_mismatch"] = _argmax_mismatch(got, ref)
        d["passed"] = bool(
            d["armed"]
            and d["finite"]
            and d["argmax_mismatch"] == 0
            and d["max_abs_d"] <= d["tol"]
        )
        # informational: why the naive kernel-vs-bf16-recurrence comparison fails
        bf16_rec = _sinkhorn_ops(comb16, ITERS, EPS)
        mx.eval(bf16_rec)
        d["info_bf16recurrence_vs_fp32_max_abs_d"] = _maxabs(bf16_rec, ref_fp32)
        d["info_bf16recurrence_argmax_mismatch_vs_fp32"] = _argmax_mismatch(
            bf16_rec, ref_fp32
        )
        # informational: does mx.fast.metal_kernel accept a raw bf16 buffer at all?
        try:
            raw = _sinkhorn_kernel_apply(comb16, HC, ITERS, EPS)
            mx.eval(raw)
            d["info_raw_bf16_kernel_ok"] = True
            d["info_raw_bf16_kernel_error"] = None
            d["info_raw_bf16_vs_fp32oracle_max_abs_d"] = _maxabs(raw, ref)
        except Exception as exc:
            d["info_raw_bf16_kernel_ok"] = False
            d["info_raw_bf16_kernel_error"] = repr(exc)
            d["info_raw_bf16_vs_fp32oracle_max_abs_d"] = None
    except Exception as exc:  # pragma: no cover - GPU-only
        d["passed"] = False
        d["error"] = repr(exc)
    return d


def _measure_dispatch() -> dict:
    """The armed dispatcher (flag on + GPU) engages the kernel (counter == 1) and
    matches the fp32 recurrence."""
    d = {"error": None}
    try:
        comb = _rand_comb(seed=12, dtype=mx.float32)
        ref = _sinkhorn_ops(comb, ITERS, EPS)
        os.environ[V41._SINKHORN_METAL_ENV] = "1"
        V41._reset_sinkhorn_kernel_calls()
        try:
            d["armed"] = bool(V41._sinkhorn_use_kernel())
            got = V41._sinkhorn_normalise(comb, HC, ITERS, EPS)
            mx.eval(got, ref)
        finally:
            os.environ.pop(V41._SINKHORN_METAL_ENV, None)
        calls = V41._sinkhorn_kernel_calls()
        d["kernel_calls"] = calls["kernel"]
        d["recurrence_calls"] = calls["recurrence"]
        d["max_abs_d_vs_recurrence"] = _maxabs(got, ref)
        d["passed"] = bool(
            d.get("armed")
            and calls["kernel"] == 1
            and calls["recurrence"] == 0
            and d["max_abs_d_vs_recurrence"] <= 1e-6
        )
    except Exception as exc:  # pragma: no cover - GPU-only
        d["passed"] = False
        d["error"] = repr(exc)
    return d


@pytest.mark.skipif(
    os.environ.get("MTPLX_GPU_PARITY") != "1",
    reason="GPU parity: run inside a GPU window with MTPLX_GPU_PARITY=1",
)
def test_sinkhorn_kernel_parity_gpu():
    """One kernel dispatch == the 40-pass stock recurrence, argmax exact.

    Self-diagnosing: every arm's dtype/tolerance, max|d|, argmax mismatch count,
    and metal/kernel-build status are collected into ``diag``, written to
    ``MTPLX_PARITY_RECEIPT`` (if set) and printed, BEFORE any assertion -- so a
    window captures exactly what failed even if stdout is truncated to the tail.
    """
    saved_dev = mx.default_device()
    mx.set_default_device(mx.gpu)
    diag = {
        "test": "test_sinkhorn_kernel_parity_gpu",
        "hc": HC,
        "iters": ITERS,
        "eps": EPS,
        "input_shape": [2, 7, HC, HC],
        "metal_available": bool(mx.metal.is_available()),
        "default_device": str(mx.default_device()),
        "arms": {},
    }
    try:
        try:
            probe = _rand_comb(seed=99, dtype=mx.float32)
            built = _sinkhorn_kernel_apply(probe, HC, ITERS, EPS)
            mx.eval(built)
            diag["kernel_builder_ok"] = True
            diag["kernel_builder_error"] = None
        except Exception as exc:
            diag["kernel_builder_ok"] = False
            diag["kernel_builder_error"] = repr(exc)

        diag["arms"]["fp32"] = _measure_fp32()
        diag["arms"]["bf16"] = _measure_bf16()
        diag["dispatch"] = _measure_dispatch()
        diag["all_passed"] = bool(
            diag.get("kernel_builder_ok")
            and diag["arms"]["fp32"].get("passed")
            and diag["arms"]["bf16"].get("passed")
            and diag["dispatch"].get("passed")
        )
    finally:
        diag["receipt_path"] = _write_parity_receipt(diag)
        print("[W38/K3 parity]\n" + json.dumps(diag, indent=2, sort_keys=True))
        mx.set_default_device(saved_dev)
        V41._reset_sinkhorn_kernel_calls()

    assert diag.get("kernel_builder_ok"), (
        f"kernel failed to build: {diag.get('kernel_builder_error')}"
    )
    assert diag["arms"]["fp32"]["passed"], f"fp32 arm failed: {diag['arms']['fp32']}"
    assert diag["arms"]["bf16"]["passed"], f"bf16 arm failed: {diag['arms']['bf16']}"
    assert diag["dispatch"]["passed"], f"dispatch arm failed: {diag['dispatch']}"
