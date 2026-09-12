"""W101 / K36 fused projection-chain kernels -- GPU numerics (Metal only).

Runs ONLY under the exclusive GPU lock and only when ``MTPLX_DSV41_GPU_TESTS=1``
(otherwise skipped), via
    GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((8*1024*1024*1024)) \
      bash scripts/deepseek_v41/gpu_window.sh env MTPLX_DSV41_GPU_TESTS=1 \
      nice -n 19 <venv python> -m pytest \
      tests/test_deepseek_v41_attn_fused_proj_w101_gpu.py -x -q

Proves, on Metal, the fused kernels are ROUNDING-CLASS (not byte-identical) vs the
eager chain and that they actually dispatch:
  1. each kernel (rmsnorm, rmsnorm+rope, rope-heads fwd/inverse) vs the pure-MLX
     reference -- max|Δ| in the reassociation band;
  2. the fused qkv-prep / out-prep vs the eager chain at REAL decode geometry
     (max|Δ| per region, bf16-class), on a simple AND a full layer;
  3. a whole fused decode ``_attend`` vs eager at real dims -- the per-layer
     max|Δ| numerics LABEL the task requires;
  4. greedy-argmax identity over 64 decode steps on the tiny full model, lever
     ON vs OFF, flips LABELLED with their top-2 logit margin;
  5. the engagement counter records fused dispatches (proves the kernels ran).

Tiny models only (memory guard); no artifact.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MTPLX_DSV41_GPU_TESTS") != "1",
    reason="GPU numerics: set MTPLX_DSV41_GPU_TESTS=1 and run under the GPU lock",
)

if os.environ.get("MTPLX_DSV41_GPU_TESTS") == "1":
    if not mx.metal.is_available():  # pragma: no cover
        pytest.skip("no Metal GPU", allow_module_level=True)
    mx.set_default_device(mx.gpu)

from mtplx.models import deepseek_v41 as dsv41                       # noqa: E402
from mtplx.models import deepseek_v41_fused_proj_kernels as fp       # noqa: E402
from mtplx.models.deepseek_v41 import _cos_sin, _rmsnorm, _rope_last  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _load_bisect():
    spec = importlib.util.spec_from_file_location(
        "w101_bisect_gpu", _REPO / "scripts" / "deepseek_v41" / "metal_decode_attn_bisect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _maxabs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.delenv(dsv41._ATTN_FUSED_PROJ_ENV, raising=False)
    fp.reset_engagement()
    yield


# --------------------------------------------------------------------------
# 1. each kernel vs its pure-MLX reference (rounding-class)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float32])
def test_rmsnorm_kernel_vs_reference(dt):
    mx.random.seed(0)
    for d in (1280, 512, 20):
        x = (mx.random.normal((3, d)) * 0.5).astype(dt)
        w = (1.0 + 0.2 * mx.random.normal((d,))).astype(dt)
        mx.eval(x, w)
        got = fp.rmsnorm(x, w, 1e-6)
        ref = fp.rmsnorm_reference(x, w, 1e-6)
        mx.eval(got, ref)
        band = 1.5e-2 if dt == mx.bfloat16 else 1e-3
        assert got.dtype == dt
        _d = _maxabs(got, ref)
        print(f"[W101 kernel] rmsnorm d={d} dt={dt} max|Δ|={_d:.3e} (band {band})")
        assert _d <= band, f"rmsnorm d={d} dt={dt} max|Δ|={_d}"


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float32])
def test_rmsnorm_rope_kernel_vs_reference(dt):
    mx.random.seed(1)
    hd, rd = 512, 64
    inv = 1.0 / (10000.0 ** (mx.arange(0, rd, 2, dtype=mx.float32) / rd))
    cos, sin = _cos_sin(inv, mx.array([5, 6, 7]))
    x = (mx.random.normal((3, hd)) * 0.5).astype(dt)
    w = (1.0 + 0.2 * mx.random.normal((hd,))).astype(dt)
    mx.eval(cos, sin, x, w)
    got = fp.rmsnorm_rope(x, w, 1e-6, cos, sin)
    ref = fp.rmsnorm_rope_reference(x, w, 1e-6, cos, sin)
    mx.eval(got, ref)
    band = 1.5e-2 if dt == mx.bfloat16 else 1e-3
    _d = _maxabs(got, ref)
    print(f"[W101 kernel] rmsnorm_rope dt={dt} max|Δ|={_d:.3e} (band {band})")
    assert _d <= band, f"rmsnorm_rope dt={dt} max|Δ|={_d}"


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float32])
def test_rope_heads_kernel_vs_reference(dt, inverse):
    mx.random.seed(2)
    H, hd, rd = 64, 512, 64
    inv = 1.0 / (10000.0 ** (mx.arange(0, rd, 2, dtype=mx.float32) / rd))
    cos, sin = _cos_sin(inv, mx.array([9, 10, 11]))
    # Production layout ``[b, s, H, hd]`` (the real _qkv_prep_fused / _out_prep_fused
    # call rope_heads with a 4D reshape(b, s, H, hd) tensor).  cos/sin are [s, rd/2],
    # so s=3 positions x H heads: this is the layout BOTH the kernel and the pure-MLX
    # oracle (which mirrors the eager ``_rope_last``, [b, s, *head, rope]) are defined
    # for.  A flat 3D [rows, H, hd] is only accepted by the kernel (it collapses
    # leading dims to rows), not by ``_rope_last`` / the oracle.
    q = (mx.random.normal((1, 3, H, hd)) * 0.5).astype(dt)
    mx.eval(cos, sin, q)
    got = fp.rope_heads(q, cos, sin, inverse=inverse)
    ref = fp.rope_heads_reference(q, cos, sin, inverse=inverse)
    mx.eval(got, ref)
    band = 1.5e-2 if dt == mx.bfloat16 else 1e-3
    _d = _maxabs(got, ref)
    print(f"[W101 kernel] rope_heads inv={inverse} dt={dt} max|Δ|={_d:.3e} (band {band})")
    assert _d <= band, f"rope_heads inv={inverse} dt={dt} max|Δ|={_d}"


# --------------------------------------------------------------------------
# 2 + 3. fused qkv/out + whole _attend vs eager at REAL decode geometry
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["swa_only", "reuse", "full"])
def test_fused_attend_vs_eager_real_dims(monkeypatch, mode):
    """Whole fused decode ``_attend`` vs eager at the real 1024-context geometry --
    the per-layer max|Δ| numerics LABEL.  Rounding-class (fused rmsnorm/rope +
    bf16 o-LoRA matmul), so bf16-band, NOT byte-identical."""
    bis = _load_bisect()
    args = bis.real_args()
    attn = bis._build_layer(args, mode)
    T = 1024

    mx.random.seed(100)
    cache_e, shared_e, x, pos = bis.build_case(args, attn, mode, T)
    mx.random.seed(100)
    cache_f, shared_f, xf, posf = bis.build_case(args, attn, mode, T)
    assert bool(mx.all(x == xf).item()), "same-seed cases must give identical decode x"

    monkeypatch.setenv(dsv41._ATTN_FUSED_PROJ_ENV, "0")
    o_e = attn._attend(x, pos, cache_e, shared_e)
    mx.eval(o_e)
    monkeypatch.setenv(dsv41._ATTN_FUSED_PROJ_ENV, "1")
    fp.reset_engagement()
    o_f = attn._attend(xf, posf, cache_f, shared_f)
    mx.eval(o_f)

    d = _maxabs(o_e, o_f)
    eng = fp.engagement()
    print(f"[W101 {mode}] fused _attend vs eager max|Δ|={d:.3e}  engagement={eng}")
    assert eng["qkv_calls"] == 1 and eng["out_calls"] == 1, f"fused kernels must run: {eng}"
    assert bool(mx.all(mx.isfinite(o_f)).item()), f"{mode}: fused output has non-finite values"
    # Numerics band.  swa_only/reuse: the key SET is fixed (no compress / pre-filled
    # selection), so the delta is pure bf16 rounding (rmsnorm/rope + the bf16 o-LoRA
    # matmul, which matches the REFERENCE bf16 einsum, model.py L784-787) -> tight
    # band.  full: this layer's indexer picks the top-k compressed rows FROM the
    # (rounding-class) qr, so a top-k boundary can flip and swap a whole key ->
    # legitimately larger delta ([[dsv41-inexact-ok-if-tie-flips]]); report it, and
    # only guard against a blow-up (a real bug), not a selection flip.
    if mode in ("swa_only", "reuse"):
        assert d <= 5e-2, f"{mode}: fused _attend max|Δ|={d} exceeds the bf16 rounding band"
    else:
        assert d <= 3e-1, f"{mode}: fused _attend max|Δ|={d} is a blow-up, not a top-k flip"


# --------------------------------------------------------------------------
# 4. greedy-argmax identity over 64 decode steps (label flips)
# --------------------------------------------------------------------------
def test_greedy_identity_64_steps_labelled(monkeypatch):
    bis = _load_bisect()
    model, _args = bis._build_tiny_full_model(seed=5)
    ops = bis._TinyOps()
    prompt_ids = list(range(1, 41))
    steps = 64

    def decode(flag):
        monkeypatch.setenv(dsv41._ATTN_FUSED_PROJ_ENV, flag)
        cache = model.make_cache()
        logits = model(ops.input([list(prompt_ids)]), cache=cache)
        mx.eval(logits)
        tok = int(mx.argmax(logits[0, -1]).item())
        ids, logs = [tok], [logits[0, -1]]
        for _ in range(steps):
            logits = model(ops.input([[tok]]), cache=cache)
            mx.eval(logits)
            tok = int(mx.argmax(logits[0, -1]).item())
            ids.append(tok)
            logs.append(logits[0, -1])
        return ids, logs

    ids_off, _ = decode("0")
    fp.reset_engagement()
    ids_on, logs_on = decode("1")
    assert fp.engagement()["qkv_calls"] > 0, "fused kernels must have run on GPU"

    flips = []
    for t, (a, b) in enumerate(zip(ids_off, ids_on)):
        if a != b:
            top2 = mx.sort(logs_on[t])[-2:]
            margin = float((top2[1] - top2[0]).item())
            flips.append((t, a, b, round(margin, 5)))
    n = len(ids_off)
    print(f"[W101] greedy identity over {n} steps: {n - len(flips)}/{n} identical; "
          f"flips (step, off_id, on_id, top2_margin): {flips}")
    # ROUNDING-CLASS (bf16 o-LoRA matmul -- the reference dtype -- vs the port's f32
    # einsum): greedy flips are ALLOWED and LABELLED (the task requirement), not a
    # failure.  Sanity only: a rounding-class lever must not flip the MAJORITY of a
    # 64-step run (that would signal a real numerics bug, not a near-tie), and every
    # flip's top-2 margin is reported for review.
    assert len(flips) <= n // 2, (
        f"greedy flipped {len(flips)}/{n} tokens -- too many for a rounding-class "
        f"lever (a real bug, not near-ties): {flips}")


# --------------------------------------------------------------------------
# 5. engagement records fused dispatches split by phase
# --------------------------------------------------------------------------
def test_engagement_counts_fused_dispatches(monkeypatch):
    bis = _load_bisect()
    model, _args = bis._build_tiny_full_model(seed=8)
    ops = bis._TinyOps()
    monkeypatch.setenv(dsv41._ATTN_FUSED_PROJ_ENV, "1")
    fp.reset_engagement()
    cache = model.make_cache()
    mx.eval(model(ops.input([list(range(1, 21))]), cache=cache))
    for t in [3, 7, 5]:
        mx.eval(model(ops.input([[t]]), cache=cache))
    e = fp.engagement()
    # one qkv + one out call per backbone layer per decode step (3 steps).
    assert e["qkv_calls"] > 0 and e["out_calls"] > 0, e
    assert e["qkv_calls"] == e["out_calls"], f"qkv/out must pair per layer-step: {e}"
    assert e["rows"] == e["qkv_calls"], "each decode step is b*s=1 row"
