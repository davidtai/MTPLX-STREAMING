"""W58 / K28: tests for the DSV4.1 fused mask + attention-sink softmax kernel
(``MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL``, ``mtplx/kernels/dsv41_fused_softmax.py``).

CPU tests (the default here) prove, WITHOUT dispatching any Metal:

  * the flag resolver (default off, truthy parse, read-at-use, fail-fast) and the
    GPU-gated ``_prefill_softmax_kernel_use`` (off on a CPU-pinned host);
  * the model dispatch: flag on + a spied GPU routes prefill (``s > 1``)
    ``_sparse_attend_oneshot`` through the kernel, while decode (``s == 1``) never
    does; flag on + CPU falls back to the eager path BYTE-IDENTICAL to control;
  * the wrapper's shape/dtype plumbing (grid, threadgroup, output shapes/dtypes,
    input list, the [rows,T] additive mask + [H] sink), proven with a spy kernel
    (``mx.fast.metal_kernel`` builds/runs on the GPU even with the CPU default
    device, so the real kernel is never built here -- the GPU-locked box).

The numeric kernel-vs-eager parity needs a Metal GPU and is
:func:`test_fused_softmax_parity_gpu`: skipped unless ``MTPLX_GPU_PARITY=1``,
self-diagnosing (writes a JSON receipt to ``MTPLX_PARITY_RECEIPT`` and prints the
same diag before asserting), comparing the kernel to the eager f32 reference on
random ``[64,64,4096]`` and ``[8,64,16384]`` scores with random masks + a per-head
sink -> max|Δ| + argmax parity.

No GPU/Metal execution, no artifact load, <3 GB RSS.  MLX pinned to CPU per
memory/worker-tests-must-pin-mlx-cpu.md.  Run under ``nice -n 19``, no ``-n auto``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41 as V41  # noqa: E402
import mtplx.kernels.dsv41_fused_softmax as K28  # noqa: E402

_ENV = V41._PREFILL_SOFTMAX_KERNEL_ENV


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _AttnShim:
    """Binds the real ``_sparse_attend*`` methods to the minimal state they read
    (a unit harness over the shipped code, no ``Attention.__init__`` needed)."""

    def __init__(self, head_dim, attn_sink, mode="reuse"):
        self.softmax_scale = head_dim ** -0.5
        self.attn_sink = attn_sink
        self.mode = mode


_AttnShim._sparse_attend = V41.Attention._sparse_attend
_AttnShim._sparse_attend_oneshot = V41.Attention._sparse_attend_oneshot
_AttnShim._sparse_attend_chunked = V41.Attention._sparse_attend_chunked


def _rand_inputs(b, s, H, hd, T, *, drop=0.3, seed=3):
    mx.random.seed(seed)
    q = mx.random.normal((b, s, H, hd)).astype(mx.float32)
    KV = mx.random.normal((b, T, hd)).astype(mx.float32)
    attend = mx.random.uniform(shape=(b, s, T)) > drop
    sink = 0.3 * mx.random.normal((H,))
    return q, KV, attend, sink


class _SpyKernel:
    """A fake compiled kernel: records the dispatch kwargs and returns zeros of the
    requested output shapes/dtypes (so the wrapper's reshape/return path runs on
    the CPU with no Metal)."""

    def __init__(self, log):
        self.log = log

    def __call__(self, *, inputs, grid, threadgroup, output_shapes, output_dtypes):
        self.log.update(
            inputs=inputs, grid=grid, threadgroup=threadgroup,
            output_shapes=output_shapes, output_dtypes=output_dtypes,
        )
        return tuple(mx.zeros(sh, dtype=dt) for sh, dt in zip(output_shapes, output_dtypes))


def _install_spy(monkeypatch):
    """Replace the (lru-cached) kernel factory so no real Metal kernel is built."""
    log = {}

    def fake_factory(**kw):
        log["factory_kw"] = kw
        return _SpyKernel(log)

    monkeypatch.setattr(K28, "_k28_kernel", fake_factory)
    return log


@pytest.fixture(autouse=True)
def _clear_env():
    saved = os.environ.get(_ENV)
    os.environ.pop(_ENV, None)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = saved


# ---------------------------------------------------------------------------
# (1) flag resolver + GPU gate
# ---------------------------------------------------------------------------
def test_flag_default_off_and_truthy_parse(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert V41._resolve_prefill_softmax_kernel() is False
    for v in ("1", "true", "on", "yes", "TRUE"):
        monkeypatch.setenv(_ENV, v)
        assert V41._resolve_prefill_softmax_kernel() is True
    for v in ("", "0", "off", "false", "no", "none", "default"):
        monkeypatch.setenv(_ENV, v)
        assert V41._resolve_prefill_softmax_kernel() is False


def test_flag_read_at_use_not_frozen(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert V41._resolve_prefill_softmax_kernel() is False
    monkeypatch.setenv(_ENV, "1")
    assert V41._resolve_prefill_softmax_kernel() is True  # re-read, not import-frozen


def test_flag_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(_ENV, "maybe")
    with pytest.raises(ValueError):
        V41._resolve_prefill_softmax_kernel()


def test_use_off_by_default_and_cpu_pinned(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert V41._prefill_softmax_kernel_use() is False
    # flag on but CPU-pinned default device -> still False (no Metal dispatch)
    monkeypatch.setenv(_ENV, "1")
    assert V41._prefill_softmax_kernel_use() is False


def test_use_true_only_when_flag_on_and_gpu(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert V41._prefill_softmax_kernel_use() is True
    # flag off + GPU -> False
    monkeypatch.delenv(_ENV, raising=False)
    assert V41._prefill_softmax_kernel_use() is False


# ---------------------------------------------------------------------------
# (2) model dispatch: prefill routes to the kernel, decode never does
# ---------------------------------------------------------------------------
def test_oneshot_prefill_routes_through_kernel_on_gpu(monkeypatch):
    q, KV, attend, sink = _rand_inputs(1, 12, 8, 64, 300)
    shim = _AttnShim(64, sink)
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)

    calls = {"n": 0}

    def spy(scores, *, attend, attn_sink, scale, **kw):
        calls["n"] += 1
        calls["scores_shape"] = tuple(scores.shape)
        calls["scale"] = float(scale)
        b, s, H, T = scores.shape
        return mx.zeros((b, s, H, T), dtype=mx.float32)

    monkeypatch.setattr(K28, "fused_prefill_softmax", spy)
    out = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    mx.eval(out)
    assert calls["n"] == 1, "prefill (s>1) must call the fused kernel wrapper"
    assert calls["scores_shape"] == (1, 12, 8, 300)
    assert calls["scale"] == shim.softmax_scale  # not fuse_scale -> kernel applies scale


def test_oneshot_lean_passes_scale_one_to_kernel(monkeypatch):
    q, KV, attend, sink = _rand_inputs(1, 12, 8, 64, 300)
    shim = _AttnShim(64, sink)
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    seen = {}

    def spy(scores, *, attend, attn_sink, scale, **kw):
        seen["scale"] = float(scale)
        b, s, H, T = scores.shape
        return mx.zeros((b, s, H, T), dtype=mx.float32)

    monkeypatch.setattr(K28, "fused_prefill_softmax", spy)
    # lean path folds scale into q, so the kernel must be told scale == 1.0
    out = shim._sparse_attend_oneshot(q, KV, attend, mx.float32, fuse_scale=True, fold_sink=True)
    mx.eval(out)
    assert seen["scale"] == 1.0


def test_decode_m1_never_routes_through_kernel(monkeypatch):
    # decode / M=1 (s == 1) must stay on the eager f32 path, byte-identical.
    q, KV, attend, sink = _rand_inputs(1, 1, 8, 64, 300)
    shim = _AttnShim(64, sink)
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)

    def boom(*a, **k):
        raise AssertionError("decode M=1 must not call the fused softmax kernel")

    monkeypatch.setattr(K28, "fused_prefill_softmax", boom)
    out = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)  # must not raise
    mx.eval(out)


def test_flag_on_cpu_is_byte_identical_eager_fallback(monkeypatch):
    # On a CPU-pinned host the flag armed must run the eager path unchanged.
    q, KV, attend, sink = _rand_inputs(1, 12, 8, 64, 300)
    shim = _AttnShim(64, sink)
    monkeypatch.delenv(_ENV, raising=False)
    off = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    off_lean = shim._sparse_attend_oneshot(q, KV, attend, mx.float32, fuse_scale=True, fold_sink=True)
    monkeypatch.setenv(_ENV, "1")
    assert V41._prefill_softmax_kernel_use() is False  # CPU -> eager
    on = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    on_lean = shim._sparse_attend_oneshot(q, KV, attend, mx.float32, fuse_scale=True, fold_sink=True)
    mx.eval(off, on, off_lean, on_lean)
    assert mx.array_equal(off, on), "flag-on CPU one-shot must equal control"
    assert mx.array_equal(off_lean, on_lean), "flag-on CPU lean must equal control"


# ---------------------------------------------------------------------------
# (3) wrapper plumbing (spy: no real Metal build)
# ---------------------------------------------------------------------------
def test_wrapper_plumbing_bool_mask_and_sink(monkeypatch):
    log = _install_spy(monkeypatch)
    b, s, H, T, tg = 2, 7, 64, 300, 256
    rows = b * s
    scores = mx.random.normal((b, s, H, T)).astype(mx.float32)
    attend = mx.random.uniform(shape=(b, s, T)) > 0.3
    sink = 0.3 * mx.random.normal((H,))
    p = K28.fused_prefill_softmax(scores, attend=attend, attn_sink=sink, scale=0.5, tg=tg)
    mx.eval(p)
    assert tuple(p.shape) == (b, s, H, T) and p.dtype == mx.float32
    assert log["factory_kw"] == dict(
        tg=tg, has_mask=True, sink_on=True, normalize=True, return_stats=False
    )
    assert log["grid"] == (tg * rows * H, 1, 1)
    assert log["threadgroup"] == (tg, 1, 1)
    assert log["output_shapes"] == [(rows, H, T)]
    assert log["output_dtypes"] == [mx.float32]
    inp = log["inputs"]
    assert len(inp) == 6
    assert tuple(inp[0].shape) == (rows, H, T) and inp[0].dtype == mx.float32  # scores 3D
    assert inp[1] == H and inp[2] == T and abs(inp[3] - 0.5) < 1e-9  # H, T, scale scalars
    assert tuple(inp[4].shape) == (rows, T) and inp[4].dtype == mx.float32  # additive mask
    assert tuple(inp[5].shape) == (H,) and inp[5].dtype == mx.float32  # per-head sink


def test_wrapper_bool_mask_becomes_additive_zero_neginf(monkeypatch):
    log = _install_spy(monkeypatch)
    b, s, H, T = 1, 4, 2, 16
    scores = mx.zeros((b, s, H, T), dtype=mx.float32)
    attend = mx.array([[[True, False] * (T // 2)] * s], dtype=mx.bool_)  # [1,s,T]
    K28.fused_prefill_softmax(scores, attend=attend, attn_sink=mx.zeros((H,)))
    mask = log["inputs"][4]
    mx.eval(mask)
    # kept -> 0.0 ; masked -> -inf (the sdpa-style additive form the kernel reads)
    assert float(mask[0, 0]) == 0.0
    assert mask[0, 1].item() == float("-inf")


def test_wrapper_no_mask_fast_path(monkeypatch):
    log = _install_spy(monkeypatch)
    scores = mx.random.normal((3, 5, 40)).astype(mx.float32)  # 3D [rows,H,T]
    p = K28.fused_prefill_softmax(scores, attend=None, attn_sink=mx.zeros((5,)))
    mx.eval(p)
    assert log["factory_kw"]["has_mask"] is False
    assert len(log["inputs"]) == 5  # scores,H,T,scale,sink -- no mask
    assert tuple(p.shape) == (3, 5, 40)


def test_wrapper_no_sink(monkeypatch):
    log = _install_spy(monkeypatch)
    scores = mx.random.normal((3, 5, 40)).astype(mx.float32)
    attend = mx.random.uniform(shape=(3, 40)) > 0.3
    K28.fused_prefill_softmax(scores, attend=attend, attn_sink=None)
    assert log["factory_kw"]["sink_on"] is False
    assert len(log["inputs"]) == 5  # scores,H,T,scale,mask -- no sink


def test_wrapper_additive_mask_passthrough(monkeypatch):
    log = _install_spy(monkeypatch)
    b, s, H, T = 2, 3, 4, 20
    rows = b * s
    scores = mx.random.normal((b, s, H, T)).astype(mx.float32)
    attend = mx.random.uniform(shape=(b, s, T)) > 0.3
    add = mx.where(attend, mx.zeros((b, s, T)), mx.full((b, s, T), -1e4))  # soft additive
    K28.fused_prefill_softmax(scores, attend=add, attn_sink=mx.zeros((H,)))
    m = log["inputs"][4]
    assert tuple(m.shape) == (rows, T) and m.dtype == mx.float32


def test_wrapper_return_stats_outputs(monkeypatch):
    log = _install_spy(monkeypatch)
    b, s, H, T = 2, 3, 4, 20
    rows = b * s
    scores = mx.random.normal((b, s, H, T)).astype(mx.float32)
    attend = mx.random.uniform(shape=(b, s, T)) > 0.3
    p, m, d = K28.fused_prefill_softmax(
        scores, attend=attend, attn_sink=mx.zeros((H,)),
        normalize=False, return_stats=True,
    )
    mx.eval(p, m, d)
    assert log["factory_kw"]["normalize"] is False
    assert log["factory_kw"]["return_stats"] is True
    assert log["output_shapes"] == [(rows, H, T), (rows * H,), (rows * H,)]
    assert tuple(p.shape) == (b, s, H, T)
    assert tuple(m.shape) == (b, s, H) and tuple(d.shape) == (b, s, H)


def test_wrapper_requires_metal(monkeypatch):
    # If Metal is unavailable the wrapper cannot fall back silently -- it raises so
    # the caller (model) is forced to guard with _prefill_softmax_kernel_use.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    K28._k28_kernel.cache_clear()
    scores = mx.zeros((2, 3, 8), dtype=mx.float32)
    with pytest.raises(RuntimeError):
        K28.fused_prefill_softmax(scores, attend=None, attn_sink=None)
    K28._k28_kernel.cache_clear()


def test_source_builder_variants_compile_to_str():
    # The source assembles for every structural variant (no template/preprocessor
    # slip); the mask/sink/normalize/stats blocks are present/absent as asked.
    src_full = K28._build_source(tg=256, has_mask=True, sink_on=True, normalize=True, return_stats=False)
    assert "mask[mbase + t]" in src_full and "sink[head]" in src_full
    assert "ex / D" in src_full and "constexpr uint TG = 256;" in src_full
    assert "float pv;" in src_full and "pv = (D > 0.0f)" in src_full  # declared once, assigned
    src_raw = K28._build_source(tg=128, has_mask=False, sink_on=False, normalize=False, return_stats=True)
    assert "mask[" not in src_raw and "sink[head]" not in src_raw
    assert "pv = ex;" in src_raw and "stats_m[gid]" in src_raw
    assert "constexpr uint TG = 128;" in src_raw
    # pv must be declared exactly once (no shadowing redeclaration inside the if)
    assert src_full.count("float pv;") == 1 and "float pv = " not in src_full


# ---------------------------------------------------------------------------
# receipt helper (CPU)
# ---------------------------------------------------------------------------
def _write_parity_receipt(diag: dict):
    path = os.environ.get("MTPLX_PARITY_RECEIPT")
    if not path:
        return None
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(diag, indent=2, sort_keys=True))
        return str(p)
    except Exception as exc:  # pragma: no cover
        print(f"[W58/K28] receipt write failed: {exc!r}")
        return None


def test_receipt_helper_writes_json(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "parity.json"
    monkeypatch.setenv("MTPLX_PARITY_RECEIPT", str(path))
    diag = {"test": "unit", "arms": {"a": {"passed": True, "max_abs_d": 1e-8}}}
    assert _write_parity_receipt(diag) == str(path)
    assert json.loads(path.read_text()) == diag


def test_receipt_helper_noop_without_env(monkeypatch):
    monkeypatch.delenv("MTPLX_PARITY_RECEIPT", raising=False)
    assert _write_parity_receipt({"x": 1}) is None


# ---------------------------------------------------------------------------
# numeric parity: kernel == eager f32 softmax (GPU window only)
# ---------------------------------------------------------------------------
def _eager_softmax_ref(scores, attend, sink, scale):
    """Normalised softmax weights with the per-head value-0 sink, exactly the
    operation the kernel targets: masked -> -inf; m = max(max(scores·scale), sink);
    p = exp(s - m) / (sum exp + exp(sink - m)); masked -> 0."""
    b, s, H, T = scores.shape
    sc = (scores * mx.array(scale, dtype=mx.float32)).astype(mx.float32)
    sc = mx.where(attend[:, :, None, :], sc, mx.array(float("-inf"), dtype=mx.float32))
    sinkb = sink.astype(mx.float32).reshape(1, 1, H, 1)
    m = mx.maximum(mx.max(sc, axis=-1, keepdims=True), sinkb)
    ex = mx.exp(sc - m)
    denom = mx.sum(ex, axis=-1, keepdims=True) + mx.exp(sinkb - m)
    return ex / denom


def _maxabs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _argmax_mismatch(a, b):
    return int(mx.sum(mx.argmax(a, axis=-1) != mx.argmax(b, axis=-1)).item())


def _measure_parity(b, s, H, T, *, scale, seed) -> dict:
    d = {"shape": [b, s, H, T], "scale": scale, "tol": 1e-6, "error": None}
    try:
        mx.random.seed(seed)
        scores = mx.random.normal((b, s, H, T)).astype(mx.float32)
        attend = mx.random.uniform(shape=(b, s, T)) > 0.3
        sink = 0.3 * mx.random.normal((H,))
        ref = _eager_softmax_ref(scores, attend, sink, scale)
        got = K28.fused_prefill_softmax(scores, attend=attend, attn_sink=sink, scale=scale)
        mx.eval(ref, got)
        d["finite"] = bool(mx.all(mx.isfinite(got)))
        d["max_abs_d"] = _maxabs(got, ref)
        d["argmax_rows"] = int(b * s * H)
        d["argmax_mismatch"] = _argmax_mismatch(got, ref)
        d["passed"] = bool(
            d["finite"] and d["argmax_mismatch"] == 0 and d["max_abs_d"] <= d["tol"]
        )
    except Exception as exc:  # pragma: no cover - GPU-only
        d["passed"] = False
        d["error"] = repr(exc)
    return d


@pytest.mark.skipif(
    os.environ.get("MTPLX_GPU_PARITY") != "1",
    reason="GPU parity: run inside a GPU window with MTPLX_GPU_PARITY=1",
)
def test_fused_softmax_parity_gpu():
    """K28 kernel == eager f32 softmax-with-sink to reassociation level, argmax
    exact, on the two documented shapes.  Self-diagnosing: every arm's max|Δ| /
    argmax mismatch / finiteness collected into ``diag``, written to
    ``MTPLX_PARITY_RECEIPT`` and printed BEFORE any assertion."""
    saved_dev = mx.default_device()
    mx.set_default_device(mx.gpu)
    diag = {
        "test": "test_fused_softmax_parity_gpu",
        "kernel": "K28 dsv41_fused_softmax",
        "metal_available": bool(mx.metal.is_available()),
        "default_device": str(mx.default_device()),
        "arms": {},
    }
    try:
        try:
            probe = mx.random.normal((2, 2, 4, 64)).astype(mx.float32)
            built = K28.fused_prefill_softmax(probe, attend=None, attn_sink=mx.zeros((4,)))
            mx.eval(built)
            diag["kernel_builder_ok"] = True
            diag["kernel_builder_error"] = None
        except Exception as exc:
            diag["kernel_builder_ok"] = False
            diag["kernel_builder_error"] = repr(exc)

        # the two documented parity shapes + a scale=1.0 and a scale!=1.0 case
        diag["arms"]["small_4096"] = _measure_parity(1, 64, 64, 4096, scale=64 ** -0.5, seed=11)
        diag["arms"]["long_16384"] = _measure_parity(1, 8, 64, 16384, scale=64 ** -0.5, seed=17)
        diag["arms"]["scale_one"] = _measure_parity(2, 8, 32, 4096, scale=1.0, seed=23)
        diag["all_passed"] = bool(
            diag.get("kernel_builder_ok")
            and all(a.get("passed") for a in diag["arms"].values())
        )
    finally:
        diag["receipt_path"] = _write_parity_receipt(diag)
        print("[W58/K28 parity]\n" + json.dumps(diag, indent=2, sort_keys=True))
        mx.set_default_device(saved_dev)

    assert diag.get("kernel_builder_ok"), f"kernel build failed: {diag.get('kernel_builder_error')}"
    for name, arm in diag["arms"].items():
        assert arm["passed"], f"{name} arm failed: {arm}"
