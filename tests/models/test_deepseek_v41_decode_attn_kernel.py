"""W60 / K29: tests for the DSV4.1 fused decode / verify MLA attention kernel
(``MTPLX_DSV41_DECODE_ATTN_KERNEL``, ``mtplx/models/deepseek_v41_attn_kernels.py``).

CPU tests (the default here) prove, WITHOUT dispatching any Metal:

  * the flag resolver (default off, truthy parse, read-at-use, fail-fast) and the
    GPU + small-M gate ``_decode_attn_kernel_use`` (off on a CPU-pinned host; off
    above the ``K+1`` verify row cap even with a GPU);
  * the model dispatch: flag on + a spied GPU routes decode (``s==1``) AND the
    small-M verify batch (``s<=cap``) through the kernel for every CSA mode, while a
    large-M prefill (``s>cap``) never does and an unsupported per-head mask falls
    back to eager; flag on + CPU falls back to the eager path BYTE-IDENTICAL to
    control;
  * the wrapper's shape/dtype plumbing (grid, threadgroup, output shape/dtype, the
    ``[q,k,v,H,T,S,scale,(mask),(sink)]`` input list, the ``[rows,T]`` additive mask
    + ``[H]`` sink), proven with a spy kernel (``mx.fast.metal_kernel`` builds/runs
    on the GPU even with the CPU default device, so the real kernel is never built
    here -- the GPU-locked box);
  * the pure-MLX references (one-shot + the exact online-tile algorithm) equal the
    model's eager ``_sparse_attend_oneshot`` to reassociation level (``max|Δ| ≤
    1e-6``, argmax exact), incl. the fully-masked → zero-output edge.

The numeric kernel-vs-eager parity needs a Metal GPU and is
:func:`test_decode_attn_parity_gpu`: skipped unless ``MTPLX_GPU_PARITY=1``,
self-diagnosing (writes a JSON receipt to ``MTPLX_PARITY_RECEIPT`` and prints the
same diag before asserting), comparing the kernel to the eager f32 reference on
random cache states at ``T ∈ {1088, 4096, 16384}`` for each CSA mode (decode M=1 +
a verify M=4 batch) → max|Δ| + argmax parity, and -- only if the streaming artifact
is present -- 32 real-model decode steps (kernel-on argmax vs kernel-off).

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
from mtplx.models import deepseek_v41_attn_kernels as K29  # noqa: E402

_ENV = V41._DECODE_ATTN_KERNEL_ENV
_MODES = ("swa_only", "full", "reindex", "reuse")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _AttnShim:
    """Binds the real ``_sparse_attend*`` / ``_decode_attn_kernel`` methods to the
    minimal state they read (a unit harness over the shipped code, no
    ``Attention.__init__``)."""

    def __init__(self, head_dim, attn_sink, mode="reuse"):
        self.softmax_scale = head_dim ** -0.5
        self.attn_sink = attn_sink
        self.mode = mode


_AttnShim._sparse_attend = V41.Attention._sparse_attend
_AttnShim._sparse_attend_oneshot = V41.Attention._sparse_attend_oneshot
_AttnShim._sparse_attend_chunked = V41.Attention._sparse_attend_chunked
_AttnShim._decode_attn_kernel = V41.Attention._decode_attn_kernel


def _mode_mask(mode, b, s, T, *, seed):
    """A CSA-mode-representative boolean ``[b,s,T]`` attend mask.  swa_only: a
    contiguous causal window; full/reindex/reuse: a window plus a sparse candidate
    set (the kernel is mode-agnostic -- these exercise different mask densities)."""
    mx.random.seed(seed)
    if mode == "swa_only":
        win = min(T, 128)
        keep = mx.arange(T)[None, None, :] >= (T - win)
        return mx.broadcast_to(keep, (b, s, T))
    dense = 0.15 if mode == "full" else (0.08 if mode == "reindex" else 0.12)
    cand = mx.random.uniform(shape=(b, s, T)) < dense
    win = mx.arange(T)[None, None, :] >= (T - 64)
    return cand | mx.broadcast_to(win, (b, s, T))


def _rand_state(b, s, H, hd, T, mode, *, seed=3):
    mx.random.seed(seed)
    q = mx.random.normal((b, s, H, hd)).astype(mx.float32)
    KV = mx.random.normal((b, T, hd)).astype(mx.float32)
    sink = 0.3 * mx.random.normal((H,))
    attend = _mode_mask(mode, b, s, T, seed=seed + 1)
    return q, KV, attend, sink


class _SpyKernel:
    """A fake compiled kernel: records the dispatch kwargs into its log entry and
    returns zeros of the requested output shapes/dtypes (so the wrapper's
    reshape/return path runs on the CPU with no Metal)."""

    def __init__(self, entry):
        self.entry = entry

    def __call__(self, *, inputs, grid, threadgroup, output_shapes, output_dtypes):
        self.entry.update(
            inputs=inputs, grid=grid, threadgroup=threadgroup,
            output_shapes=output_shapes, output_dtypes=output_dtypes,
        )
        return tuple(mx.zeros(sh, dtype=dt) for sh, dt in zip(output_shapes, output_dtypes))


def _install_spy(monkeypatch):
    """Spy all three (lru-cached) kernel factories so no real Metal kernel is built.
    Returns a log with keys ``single`` / ``split`` / ``combine``; a key carries
    ``factory_kw`` + the dispatch kwargs only if that path ran."""
    log = {"single": {}, "split": {}, "combine": {}}

    def factory(key):
        def make(**kw):
            log[key]["factory_kw"] = kw
            return _SpyKernel(log[key])
        return make

    monkeypatch.setattr(K29, "_k29_kernel", factory("single"))
    monkeypatch.setattr(K29, "_k29_split_kernel", factory("split"))
    monkeypatch.setattr(K29, "_k29_combine_kernel", factory("combine"))
    return log


def _fake_gpu(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)


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
# (1) flag resolver + GPU / small-M gate
# ---------------------------------------------------------------------------
def test_flag_default_off_and_truthy_parse(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert V41._resolve_decode_attn_kernel() is False
    for v in ("1", "true", "on", "yes", "TRUE"):
        monkeypatch.setenv(_ENV, v)
        assert V41._resolve_decode_attn_kernel() is True
    for v in ("", "0", "off", "false", "no", "none", "default"):
        monkeypatch.setenv(_ENV, v)
        assert V41._resolve_decode_attn_kernel() is False


def test_flag_read_at_use_not_frozen(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert V41._resolve_decode_attn_kernel() is False
    monkeypatch.setenv(_ENV, "1")
    assert V41._resolve_decode_attn_kernel() is True  # re-read, not import-frozen


def test_flag_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(_ENV, "maybe")
    with pytest.raises(ValueError):
        V41._resolve_decode_attn_kernel()


def test_use_off_by_default_and_cpu_pinned(monkeypatch):
    q = mx.zeros((1, 1, 8, 512))
    monkeypatch.delenv(_ENV, raising=False)
    assert V41._decode_attn_kernel_use(q) is False
    # flag on but CPU-pinned default device -> still False (no Metal dispatch)
    monkeypatch.setenv(_ENV, "1")
    assert V41._decode_attn_kernel_use(q) is False


def test_use_true_only_when_flag_on_gpu_and_small_m(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    _fake_gpu(monkeypatch)
    # decode M=1 and verify K+1 (<= cap) route
    assert V41._decode_attn_kernel_use(mx.zeros((1, 1, 8, 512))) is True
    assert V41._decode_attn_kernel_use(mx.zeros((1, 4, 8, 512))) is True
    assert V41._decode_attn_kernel_use(mx.zeros((1, V41._DECODE_ATTN_KERNEL_MAX_ROWS, 8, 512))) is True
    # above the small-M cap (prefill) -> False (never diverts prefill)
    assert V41._decode_attn_kernel_use(mx.zeros((1, V41._DECODE_ATTN_KERNEL_MAX_ROWS + 1, 8, 512))) is False
    assert V41._decode_attn_kernel_use(mx.zeros((1, 1024, 8, 512))) is False
    # flag off + GPU -> False
    monkeypatch.delenv(_ENV, raising=False)
    assert V41._decode_attn_kernel_use(mx.zeros((1, 1, 8, 512))) is False


# ---------------------------------------------------------------------------
# (2) model dispatch: decode/verify route, prefill never, unsupported falls back
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("s", [1, 4])
def test_decode_and_verify_route_through_kernel(monkeypatch, mode, s):
    q, KV, attend, sink = _rand_state(1, s, 8, 512, 300, mode)
    shim = _AttnShim(512, sink, mode=mode)
    monkeypatch.setenv(_ENV, "1")
    _fake_gpu(monkeypatch)

    calls = {"n": 0}

    def spy(q_, k_, v_, *, attend, attn_sink, scale, T):
        calls["n"] += 1
        calls["shapes"] = (tuple(q_.shape), tuple(k_.shape), tuple(v_.shape))
        calls["T"] = int(T)
        calls["scale"] = float(scale)
        calls["k_is_v"] = k_ is v_
        b_, s_, H_, hd_ = q_.shape
        return mx.zeros((b_, s_, H_, hd_), dtype=mx.float32)

    monkeypatch.setattr(K29, "fused_decode_attention", spy)
    out = shim._sparse_attend(q, KV, attend)
    mx.eval(out)
    assert calls["n"] == 1, f"{mode} s={s} must route through the fused kernel"
    assert calls["shapes"] == ((1, s, 8, 512), (1, 300, 512), (1, 300, 512))
    assert calls["T"] == 300
    assert abs(calls["scale"] - shim.softmax_scale) < 1e-9
    assert calls["k_is_v"], "MLA: key and value must be the one shared latent"
    assert tuple(out.shape) == (1, s, 8, 512)


def test_large_m_prefill_never_routes(monkeypatch):
    # A prefill batch (rows > the small-M cap) must stay on the eager score path.
    s = V41._DECODE_ATTN_KERNEL_MAX_ROWS + 8
    q, KV, attend, sink = _rand_state(1, s, 8, 512, 300, "reuse")
    shim = _AttnShim(512, sink)
    monkeypatch.setenv(_ENV, "1")
    _fake_gpu(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("prefill (large M) must not call the decode kernel")

    monkeypatch.setattr(K29, "fused_decode_attention", boom)
    out = shim._sparse_attend(q, KV, attend)  # must not raise
    mx.eval(out)


def test_unsupported_mask_shape_falls_back_to_eager(monkeypatch):
    # A per-head mask [b,s,H,T] is unsupported -> _decode_attn_kernel returns None
    # -> eager one-shot runs (no kernel call, correct output).
    q, KV, _, sink = _rand_state(1, 1, 8, 512, 300, "reuse")
    per_head = mx.random.uniform(shape=(1, 1, 8, 300)) > 0.3  # 4D, per-head
    shim = _AttnShim(512, sink)
    monkeypatch.setenv(_ENV, "1")
    _fake_gpu(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("unsupported mask shape must fall back to eager")

    monkeypatch.setattr(K29, "fused_decode_attention", boom)
    assert shim._decode_attn_kernel(q, KV, per_head) is None


def test_flag_on_cpu_is_byte_identical_eager_fallback(monkeypatch):
    # On a CPU-pinned host the flag armed must run the eager path unchanged.
    for s in (1, 4):
        q, KV, attend, sink = _rand_state(1, s, 8, 512, 300, "reuse", seed=5)
        shim = _AttnShim(512, sink)
        monkeypatch.delenv(_ENV, raising=False)
        off = shim._sparse_attend(q, KV, attend)
        monkeypatch.setenv(_ENV, "1")
        assert V41._decode_attn_kernel_use(q) is False  # CPU -> eager
        on = shim._sparse_attend(q, KV, attend)
        mx.eval(off, on)
        assert mx.array_equal(off, on), f"flag-on CPU s={s} must equal control"


# ---------------------------------------------------------------------------
# (3) wrapper plumbing (spy: no real Metal build)
# ---------------------------------------------------------------------------
def test_wrapper_single_plumbing_bool_mask_and_sink(monkeypatch):
    # n_splits=1 forces the single-dispatch fused kernel (T here would auto-split).
    log = _install_spy(monkeypatch)
    b, s, H, hd, T, tg = 1, 4, 64, 512, 300, 128
    rows = b * s
    q = mx.random.normal((b, s, H, hd)).astype(mx.float32)
    KV = mx.random.normal((b, T, hd)).astype(mx.float32)
    attend = mx.random.uniform(shape=(b, s, T)) > 0.3
    sink = 0.3 * mx.random.normal((H,))
    out = K29.fused_decode_attention(q, KV, KV, attend=attend, attn_sink=sink,
                                     scale=0.5, T=T, tg=tg, n_splits=1)
    mx.eval(out)
    assert tuple(out.shape) == (b, s, H, hd) and out.dtype == mx.float32
    single = log["single"]
    assert not log["split"] and not log["combine"]  # split path untouched
    assert single["factory_kw"] == dict(tg=tg, hd=hd, has_mask=True, sink_on=True)
    assert single["grid"] == (tg * rows * H, 1, 1)
    assert single["threadgroup"] == (tg, 1, 1)
    assert single["output_shapes"] == [(rows, H, hd)]
    assert single["output_dtypes"] == [mx.float32]
    inp = single["inputs"]
    assert len(inp) == 9  # q,k,v,H,T,S,scale,mask,sink
    assert tuple(inp[0].shape) == (rows, H, hd) and inp[0].dtype == mx.float32  # q 3D
    assert tuple(inp[1].shape) == (b, T, hd) and inp[1].dtype == mx.float32     # k
    assert tuple(inp[2].shape) == (b, T, hd) and inp[2].dtype == mx.float32     # v
    assert inp[3] == H and inp[4] == T and inp[5] == rows // b  # H, T, S(=rows/b)
    assert abs(inp[6] - 0.5) < 1e-9                             # scale
    assert tuple(inp[7].shape) == (rows, T) and inp[7].dtype == mx.float32  # additive mask
    assert tuple(inp[8].shape) == (H,) and inp[8].dtype == mx.float32       # per-head sink


def test_wrapper_splitk_plumbing(monkeypatch):
    # split-K path: pass 1 grid = tg*(rows*H*G) with three partial outputs, pass 2
    # combine grid = tg*(rows*H) reading the partials + sink.
    log = _install_spy(monkeypatch)
    b, s, H, hd, T, tg, G = 1, 1, 64, 512, 4096, 128, 8
    rows = b * s
    q = mx.random.normal((b, s, H, hd)).astype(mx.float32)
    KV = mx.random.normal((b, T, hd)).astype(mx.float32)
    attend = mx.random.uniform(shape=(b, s, T)) > 0.3
    sink = 0.3 * mx.random.normal((H,))
    out = K29.fused_decode_attention(q, KV, KV, attend=attend, attn_sink=sink,
                                     scale=0.5, T=T, tg=tg, n_splits=G)
    mx.eval(out)
    assert tuple(out.shape) == (b, s, H, hd) and out.dtype == mx.float32
    assert not log["single"]  # single path untouched
    sp = log["split"]
    n_part = rows * H * G
    assert sp["factory_kw"] == dict(tg=tg, hd=hd, g=G, has_mask=True)
    assert sp["grid"] == (tg * n_part, 1, 1)
    assert sp["output_shapes"] == [(n_part,), (n_part,), (n_part, hd)]
    assert sp["output_dtypes"] == [mx.float32, mx.float32, mx.float32]
    assert len(sp["inputs"]) == 8  # q,k,v,H,T,S,scale,mask (NO sink in pass 1)
    assert sp["inputs"][4] == T
    cm = log["combine"]
    assert cm["factory_kw"] == dict(tg=tg, hd=hd, g=G, sink_on=True)
    assert cm["grid"] == (tg * rows * H, 1, 1)
    assert cm["output_shapes"] == [(rows, H, hd)]
    ci = cm["inputs"]
    assert len(ci) == 5  # part_m, part_d, part_acc, H, sink
    assert tuple(ci[0].shape) == (n_part,) and tuple(ci[1].shape) == (n_part,)
    assert tuple(ci[2].shape) == (n_part, hd)
    assert ci[3] == H and tuple(ci[4].shape) == (H,)


def test_wrapper_bool_mask_becomes_additive_zero_neginf(monkeypatch):
    log = _install_spy(monkeypatch)
    b, s, H, hd, T = 1, 1, 2, 512, 16  # T<=256 -> single path
    q = mx.zeros((b, s, H, hd), dtype=mx.float32)
    KV = mx.zeros((b, T, hd), dtype=mx.float32)
    attend = mx.array([[[True, False] * (T // 2)]], dtype=mx.bool_)  # [1,1,T]
    K29.fused_decode_attention(q, KV, KV, attend=attend, attn_sink=mx.zeros((H,)))
    mask = log["single"]["inputs"][7]
    mx.eval(mask)
    assert float(mask[0, 0]) == 0.0            # kept -> 0.0
    assert mask[0, 1].item() == float("-inf")  # masked -> -inf


def test_wrapper_no_mask_and_no_sink(monkeypatch):
    log = _install_spy(monkeypatch)
    q = mx.random.normal((1, 1, 4, 512)).astype(mx.float32)
    KV = mx.random.normal((1, 40, 512)).astype(mx.float32)  # T<=256 -> single
    K29.fused_decode_attention(q, KV, KV, attend=None, attn_sink=None)
    assert log["single"]["factory_kw"]["has_mask"] is False
    assert log["single"]["factory_kw"]["sink_on"] is False
    assert len(log["single"]["inputs"]) == 7  # q,k,v,H,T,S,scale -- no mask, no sink


def test_wrapper_3d_q_returns_3d(monkeypatch):
    _install_spy(monkeypatch)
    q = mx.random.normal((3, 4, 512)).astype(mx.float32)  # [rows,H,hd]
    KV = mx.random.normal((3, 20, 512)).astype(mx.float32)  # b == rows (S=1)
    out = K29.fused_decode_attention(q, KV, KV, attn_sink=mx.zeros((4,)))
    mx.eval(out)
    assert tuple(out.shape) == (3, 4, 512)


def test_wrapper_T_smaller_than_cache(monkeypatch):
    log = _install_spy(monkeypatch)
    q = mx.random.normal((1, 1, 4, 512)).astype(mx.float32)
    KV = mx.random.normal((1, 100, 512)).astype(mx.float32)
    attend = mx.random.uniform(shape=(1, 1, 60)) > 0.3
    K29.fused_decode_attention(q, KV, KV, attend=attend, attn_sink=mx.zeros((4,)), T=60)
    assert log["single"]["inputs"][4] == 60                   # T scalar clipped (<=256 single)
    assert tuple(log["single"]["inputs"][7].shape) == (1, 60)  # mask matches T


def test_wrapper_requires_metal(monkeypatch):
    # If Metal is unavailable the wrapper cannot fall back silently -- it raises so
    # the caller (model) is forced to guard with _decode_attn_kernel_use.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    K29._k29_kernel.cache_clear()
    K29._k29_split_kernel.cache_clear()
    K29._k29_combine_kernel.cache_clear()
    q = mx.zeros((1, 1, 4, 512), dtype=mx.float32)
    KV = mx.zeros((1, 8, 512), dtype=mx.float32)  # single path
    with pytest.raises(RuntimeError):
        K29.fused_decode_attention(q, KV, KV, attn_sink=None)
    # split path also raises off-Metal
    KV2 = mx.zeros((1, 4096, 512), dtype=mx.float32)
    with pytest.raises(RuntimeError):
        K29.fused_decode_attention(q, KV2, KV2, attn_sink=None, n_splits=8)
    K29._k29_kernel.cache_clear()
    K29._k29_split_kernel.cache_clear()
    K29._k29_combine_kernel.cache_clear()


def test_choose_splits_occupancy():
    # tiny T stays single; decode M=1 (rows*H=64) picks G>=8 for occupancy; 16K
    # scales up; a larger row count (verify) needs fewer splits.
    assert K29._choose_splits(64, 64) == 1
    assert K29._choose_splits(256, 64) == 1
    assert K29._choose_splits(1088, 64) == 8      # 512/64 occupancy floor
    assert K29._choose_splits(4096, 64) == 8
    assert K29._choose_splits(16384, 64) == 32    # load-bound, capped at _SPLITS_MAX
    assert K29._choose_splits(4096, 256) == 8     # verify M=4: g_occ=2, g_load=8
    assert K29._choose_splits(1088, 256) == 3     # g_occ=2, g_load=ceil(1088/512)=3 -> 3
    # rows*H*G threadgroups clears the occupancy target for M=1 decode
    assert 64 * K29._choose_splits(1088, 64) >= K29._OCC_TARGET_TG


# ---------------------------------------------------------------------------
# (4) source builder variants assemble to str (no template slip)
# ---------------------------------------------------------------------------
def test_source_builder_variants():
    full = K29._build_source(tg=128, hd=512, has_mask=True, sink_on=True)
    assert "constexpr uint TG = 128;" in full and "constexpr uint HD = 512;" in full
    assert "mask[m_base + t]" in full and "precise::exp(sink[head] - m_run)" in full
    assert "threadgroup float q_sh[HD];" in full and "threadgroup float acc_sh[HD];" in full
    # precision fix: every exp is precise (no bare fast-math metal::exp)
    assert "metal::precise::exp" in full
    assert "metal::exp(" not in full and "fast::exp" not in full
    assert "%%" not in full
    bare = K29._build_source(tg=256, hd=576, has_mask=False, sink_on=False)
    assert "mask[m_base + t]" not in bare and "precise::exp(sink[head] - m_run)" not in bare
    assert "constexpr uint TG = 256;" in bare and "constexpr uint HD = 576;" in bare
    assert "%%" not in bare
    # split + combine sources assemble, carry G, and use precise::exp only
    split = K29._build_split_source(tg=128, hd=512, g=8, has_mask=True)
    assert "constexpr uint G  = 8;" in split and "mask[m_base + t]" in split
    assert "part_m[gid]" in split and "part_acc[gid * HD + i]" in split
    assert "metal::precise::exp" in split and "metal::exp(" not in split and "%%" not in split
    comb = K29._build_combine_source(tg=128, hd=512, g=8, sink_on=True)
    assert "constexpr uint G  = 8;" in comb and "part_m[pbase + lane]" in comb
    assert "precise::exp(sink[head] - m)" in comb
    assert "metal::precise::exp" in comb and "metal::exp(" not in comb and "%%" not in comb
    comb_ns = K29._build_combine_source(tg=128, hd=512, g=4, sink_on=False)
    assert "sink[head]" not in comb_ns and "%%" not in comb_ns


# ---------------------------------------------------------------------------
# (5) CPU parity: pure-MLX references == model eager _sparse_attend_oneshot
# ---------------------------------------------------------------------------
def _maxabs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _argmax_mismatch(a, b):
    aa = a.reshape(-1, a.shape[-1])
    bb = b.reshape(-1, b.shape[-1])
    return int(mx.sum(mx.argmax(aa, axis=-1) != mx.argmax(bb, axis=-1)).item())


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("b,s,H,T", [(1, 1, 64, 1088), (1, 1, 8, 300), (1, 4, 8, 512)])
def test_reference_matches_eager_reassoc(mode, b, s, H, T):
    hd = 512
    q, KV, attend, sink = _rand_state(b, s, H, hd, T, mode, seed=11)
    shim = _AttnShim(hd, sink, mode=mode)
    eager = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)  # model eager decode
    ref = K29.decode_attention_reference(q, KV, KV, attend=attend, attn_sink=sink, scale=hd ** -0.5)
    tiled = K29.decode_attention_reference_tiled(q, KV, KV, attend=attend, attn_sink=sink,
                                                 scale=hd ** -0.5, tile=128)
    mx.eval(eager, ref, tiled)
    assert bool(mx.all(mx.isfinite(ref))) and bool(mx.all(mx.isfinite(tiled)))
    assert _maxabs(ref, eager) <= 1e-6, f"{mode} one-shot ref maxΔ"
    assert _maxabs(tiled, eager) <= 1e-6, f"{mode} tiled ref maxΔ"
    assert _argmax_mismatch(ref, eager) == 0
    assert _argmax_mismatch(tiled, eager) == 0


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("b,s,H,T,G", [(1, 1, 64, 1088, 8), (1, 1, 8, 4096, 8),
                                       (1, 4, 8, 4096, 16), (1, 1, 8, 16384, 32)])
def test_splitk_reference_matches_eager_reassoc(mode, b, s, H, T, G):
    # the exact split-K partition+combine (the two-kernel algorithm) equals the
    # eager softmax-with-sink to reassociation level -- the CPU proof the split-K
    # reassociation is greedy-safe before any GPU run.
    hd = 512
    q, KV, attend, sink = _rand_state(b, s, H, hd, T, mode, seed=13)
    shim = _AttnShim(hd, sink, mode=mode)
    eager = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    sk = K29.decode_attention_reference_splitk(q, KV, KV, attend=attend, attn_sink=sink,
                                               scale=hd ** -0.5, n_splits=G)
    mx.eval(eager, sk)
    assert bool(mx.all(mx.isfinite(sk)))
    assert _maxabs(sk, eager) <= 1e-6, f"{mode} split-K(G={G}) maxΔ"
    assert _argmax_mismatch(sk, eager) == 0


def test_splitk_reference_equals_tiled_and_oneshot():
    # partition invariance: split-K (any G), the contiguous tiled reference, and the
    # one-shot reference are the SAME operation up to float reassociation (<=1e-6).
    hd = 512
    q, KV, attend, sink = _rand_state(1, 1, 16, hd, 4096, "reuse", seed=17)
    one = K29.decode_attention_reference(q, KV, KV, attend=attend, attn_sink=sink, scale=hd ** -0.5)
    for G in (1, 4, 8, 32):
        sk = K29.decode_attention_reference_splitk(q, KV, KV, attend=attend, attn_sink=sink,
                                                   scale=hd ** -0.5, n_splits=G)
        mx.eval(one, sk)
        assert _maxabs(sk, one) <= 1e-6, f"G={G}"
        assert _argmax_mismatch(sk, one) == 0


def test_engagement_counters(monkeypatch):
    # the wrapper bumps calls/rows/split_calls on real dispatches; note_fallback
    # bumps fallbacks. Proven with the spy (no Metal).
    _install_spy(monkeypatch)
    K29.reset_engagement()
    assert K29.engagement() == {"calls": 0, "rows": 0, "split_calls": 0, "fallbacks": 0}
    q = mx.random.normal((1, 1, 8, 512)).astype(mx.float32)
    KV = mx.random.normal((1, 100, 512)).astype(mx.float32)  # <=256 -> single
    K29.fused_decode_attention(q, KV, KV, attn_sink=mx.zeros((8,)))
    e = K29.engagement()
    assert e["calls"] == 1 and e["rows"] == 1 and e["split_calls"] == 0
    # a split-K call (T>256) bumps split_calls; verify rows counts b*s
    qv = mx.random.normal((1, 4, 8, 512)).astype(mx.float32)
    KV2 = mx.random.normal((1, 4096, 512)).astype(mx.float32)
    K29.fused_decode_attention(qv, KV2, KV2, attn_sink=mx.zeros((8,)))
    e = K29.engagement()
    assert e["calls"] == 2 and e["rows"] == 1 + 4 and e["split_calls"] == 1
    K29.note_fallback()
    assert K29.engagement()["fallbacks"] == 1
    K29.reset_engagement()
    assert K29.engagement()["calls"] == 0 and K29.engagement()["fallbacks"] == 0


def test_model_fallback_bumps_counter(monkeypatch):
    # an unsupported per-head mask through the model path records a fallback.
    q, KV, _, sink = _rand_state(1, 1, 8, 512, 300, "reuse")
    per_head = mx.random.uniform(shape=(1, 1, 8, 300)) > 0.3  # 4D, unsupported
    shim = _AttnShim(512, sink)
    K29.reset_engagement()
    assert shim._decode_attn_kernel(q, KV, per_head) is None
    assert K29.engagement()["fallbacks"] == 1
    assert K29.engagement()["calls"] == 0


def test_reference_fully_masked_is_zero_and_finite():
    hd = 512
    q = mx.random.normal((1, 1, 4, hd)).astype(mx.float32)
    KV = mx.random.normal((1, 50, hd)).astype(mx.float32)
    attend = mx.zeros((1, 1, 50), dtype=mx.bool_)  # no key reachable
    sink = mx.zeros((4,))
    shim = _AttnShim(hd, sink)
    eager = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    ref = K29.decode_attention_reference(q, KV, KV, attend=attend, attn_sink=sink, scale=hd ** -0.5)
    tiled = K29.decode_attention_reference_tiled(q, KV, KV, attend=attend, attn_sink=sink, scale=hd ** -0.5)
    mx.eval(eager, ref, tiled)
    for name, o in (("eager", eager), ("ref", ref), ("tiled", tiled)):
        assert bool(mx.all(mx.isfinite(o))), name
        assert float(mx.max(mx.abs(o))) == 0.0, name


def test_reference_T_clip_matches_eager():
    # T < cache rows attends only the first T rows -- reference must match eager on
    # the sliced KV.
    hd = 512
    q, KV, attend, sink = _rand_state(1, 1, 8, hd, 200, "full", seed=31)
    shim = _AttnShim(hd, sink)
    eager = shim._sparse_attend_oneshot(q, KV[:, :120, :], attend[:, :, :120], mx.float32)
    ref = K29.decode_attention_reference(q, KV, KV, attend=attend, attn_sink=sink, scale=hd ** -0.5, T=120)
    mx.eval(eager, ref)
    assert _maxabs(ref, eager) <= 1e-6 and _argmax_mismatch(ref, eager) == 0


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
        print(f"[W60/K29] receipt write failed: {exc!r}")
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
# (6) numeric parity: kernel == eager (GPU window only)
# ---------------------------------------------------------------------------
def _eager_ref(q, KV, attend, sink, scale):
    """The exact eager operation the kernel targets (model
    ``_sparse_attend_oneshot`` fold-sink math), computed in pure MLX f32."""
    return K29.decode_attention_reference(q, KV, KV, attend=attend, attn_sink=sink, scale=scale)


def _measure_parity(b, s, H, T, mode, *, scale, seed, n_splits=None) -> dict:
    d = {"shape": [b, s, H, T], "mode": mode, "scale": scale, "tol": 1e-6,
         "n_splits_req": n_splits, "n_splits_auto": K29._choose_splits(T, b * s * H),
         "error": None}
    try:
        q, KV, attend, sink = _rand_state(b, s, H, 512, T, mode, seed=seed)
        ref = _eager_ref(q, KV, attend, sink, scale)
        K29.reset_engagement()
        got = K29.fused_decode_attention(q, KV, KV, attend=attend, attn_sink=sink,
                                         scale=scale, T=T, n_splits=n_splits)
        mx.eval(ref, got)
        d["engagement"] = K29.engagement()
        d["finite"] = bool(mx.all(mx.isfinite(got)))
        d["max_abs_d"] = _maxabs(got, ref)
        d["argmax_rows"] = int(b * s * H)
        d["argmax_mismatch"] = _argmax_mismatch(got, ref)
        d["passed"] = bool(d["finite"] and d["argmax_mismatch"] == 0 and d["max_abs_d"] <= d["tol"])
    except Exception as exc:  # pragma: no cover - GPU-only
        d["passed"] = False
        d["error"] = repr(exc)
    return d


def _real_model_decode_parity(steps=32) -> dict:
    """Optional: 32 real-model decode steps, kernel-on argmax vs kernel-off, IF the
    streaming artifact is present.  Records artifact presence + loader importability;
    the authoritative served comparison is the ``decode_attn_kernel`` A/B arm in
    ``ab_decode_env_levers.py`` (proper streamed load + memory guards + the K29
    engagement counter), so a heavy second load is not reproduced inside pytest --
    the worker box never loads the artifact anyway."""
    d = {"ran": False, "reason": None}
    art = os.environ.get("MTPLX_DSV41_ARTIFACT") or os.path.expanduser(
        "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"
    )
    if not Path(art).exists():
        d["reason"] = f"artifact absent ({art})"
        return d
    try:
        from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming  # noqa: F401
        d["loader_importable"] = True
        d["artifact"] = art
        d["reason"] = (
            "artifact present; the 32-step served kernel-on-vs-off comparison + K29 "
            "engagement counter is the ab_decode_env_levers.py `decode_attn_kernel` arm"
        )
    except Exception as exc:  # pragma: no cover
        d["loader_importable"] = False
        d["reason"] = f"loader import failed: {exc!r}"
    return d


@pytest.mark.skipif(
    os.environ.get("MTPLX_GPU_PARITY") != "1",
    reason="GPU parity: run inside a GPU window with MTPLX_GPU_PARITY=1",
)
def test_decode_attn_parity_gpu():
    """K29 kernel == eager f32 softmax-with-sink attention to reassociation level,
    argmax exact, for each CSA mode at T ∈ {1088, 4096, 16384} (decode M=1 + verify
    M=4).  Self-diagnosing: every arm's max|Δ| / argmax mismatch / finiteness into
    ``diag``, written to ``MTPLX_PARITY_RECEIPT`` and printed BEFORE any assertion."""
    saved_dev = mx.default_device()
    mx.set_default_device(mx.gpu)
    scale = 512 ** -0.5
    diag = {
        "test": "test_decode_attn_parity_gpu",
        "kernel": "K29 deepseek_v41_attn_kernels.fused_decode_attention",
        "metal_available": bool(mx.metal.is_available()),
        "default_device": str(mx.default_device()),
        "arms": {},
    }
    try:
        try:
            probe = mx.random.normal((1, 1, 4, 512)).astype(mx.float32)
            pk = mx.random.normal((1, 64, 512)).astype(mx.float32)
            built = K29.fused_decode_attention(probe, pk, pk, attn_sink=mx.zeros((4,)), scale=scale)
            mx.eval(built)
            diag["kernel_builder_ok"] = True
            diag["kernel_builder_error"] = None
        except Exception as exc:
            diag["kernel_builder_ok"] = False
            diag["kernel_builder_error"] = repr(exc)

        seed = 100
        # default (auto n_splits) path for each mode × T -- what the model ships.
        for T in (1088, 4096, 16384):
            for mode in _MODES:
                diag["arms"][f"decode_{mode}_T{T}"] = _measure_parity(
                    1, 1, 64, T, mode, scale=scale, seed=seed)
                seed += 1
        # a verify M=4 batch at the mid length for each mode
        for mode in _MODES:
            diag["arms"][f"verify4_{mode}_T4096"] = _measure_parity(
                1, 4, 64, 4096, mode, scale=scale, seed=seed)
            seed += 1
        # explicitly exercise BOTH kernel paths at T=4096 (auto picks split-K there),
        # so the single-dispatch fused kernel also gets GPU parity coverage.
        diag["arms"]["single_g1_full_T4096"] = _measure_parity(
            1, 1, 64, 4096, "full", scale=scale, seed=seed, n_splits=1)
        seed += 1
        diag["arms"]["split_g8_full_T4096"] = _measure_parity(
            1, 1, 64, 4096, "full", scale=scale, seed=seed, n_splits=8)
        seed += 1

        diag["real_model_decode"] = _real_model_decode_parity()
        diag["all_passed"] = bool(
            diag.get("kernel_builder_ok")
            and all(a.get("passed") for a in diag["arms"].values())
        )
    finally:
        diag["receipt_path"] = _write_parity_receipt(diag)
        print("[W60/K29 parity]\n" + json.dumps(diag, indent=2, sort_keys=True))
        mx.set_default_device(saved_dev)

    assert diag.get("kernel_builder_ok"), f"kernel build failed: {diag.get('kernel_builder_error')}"
    for name, arm in diag["arms"].items():
        assert arm["passed"], f"{name} arm failed: {arm}"
