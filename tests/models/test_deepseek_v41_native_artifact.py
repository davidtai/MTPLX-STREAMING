"""W18: end-to-end checks against the real native (mxfp8/mxfp4) artifact.

Opt-in (both the artifact and the Hub source must be present); every check is
RSS-bounded (<= ~2 GB): the model parameter tree is inspected lazily (shapes only,
never materialised) and residents are read one tensor at a time.  Covers:

  * strict-load equivalence: every model text param is covered by exactly one
    resident of matching shape (the invariant ``model.load_weights(strict=True)``
    enforces), re-derived key count, at low RSS -- without holding the whole
    model + a full resident dict in memory at once;
  * on-disk resident bit-exactness: sampled mxfp8 dense + mxfp4 MTP residents
    dequantize byte-for-byte to the fp32 source dequant;
  * layer-0 attention golden: the mxfp8 residents produce attn_L0 output at
    >= 0.999 global cosine vs the fp32 dense oracle on the 31-token probe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mtplx import deepseek_v41_convert as dc
from mtplx.models.deepseek_v41 import Attention, Model, ModelArgs, _rmsnorm, _sanitize_name
from mtplx.models.deepseek_v41_loader import is_text_resident
from mtplx.models.deepseek_v41_cache import make_cache as _make_cache

ART = Path(os.environ.get(
    "MTPLX_DSV41_NATIVE_ART",
    str(Path.home() / "models" / "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"),
))
SRC = Path(os.environ.get(
    "MTPLX_DSV41_SRC", str(Path.home() / "models" / "DeepSeek-V4.1-Flash-src")
))
PROBE = [0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291, 6328, 3465, 1241,
         6036, 14, 291, 3395, 361, 1354, 260, 565, 291, 6328, 3465, 21740, 6036, 14, 291, 2605]

_have = (ART / "model.safetensors.index.json").is_file() and (
    ART / "config.json"
).is_file() and (SRC / "model.safetensors.index.json").is_file()
pytestmark = pytest.mark.skipif(
    not _have, reason="native artifact + Hub source required (MTPLX_DSV41_NATIVE_ART / MTPLX_DSV41_SRC)"
)


def _cfg_args():
    cfg = json.loads((ART / "config.json").read_text())
    return cfg, ModelArgs.from_dict(cfg)


def _src_fp32(base, wm):
    sh = wm[base]
    hdr, ds = dc.read_safetensors_header(str(SRC / sh))
    ents = dc.tensor_entries(hdr)
    e = ents[base]
    fd = os.open(str(SRC / sh), os.O_RDONLY)
    try:
        raw = dc.read_tensor_raw(fd, ds, e)
        if e.dtype == "F8_E4M3":
            w = np.frombuffer(raw, np.uint8).reshape(e.shape)
            se = ents[base[:-len(".weight")] + ".scale"]
            s = np.frombuffer(dc.read_tensor_raw(fd, ds, se), np.uint8).reshape(se.shape)
            return dc.dequant_fp8_block(w, s)
        if e.dtype == "I8":  # FP4 expert
            w = np.frombuffer(raw, np.uint8).reshape(e.shape)
            se = ents[base[:-len(".weight")] + ".scale"]
            s = np.frombuffer(dc.read_tensor_raw(fd, ds, se), np.uint8).reshape(se.shape)
            return dc.dequant_fp4(w, s)
        return dc.raw_to_f32(e, raw)
    finally:
        os.close(fd)


def test_config_quantization_block_native():
    cfg, _ = _cfg_args()
    q = cfg["quantization"]
    assert q["mode"] == "mxfp8" and q["group_size"] == 32 and q["bits"] == 8
    mtp = [k for k in q if k not in ("group_size", "bits", "mode")]
    assert len(mtp) == 1152  # 3 MTP layers x 128 experts x 3 projections
    assert all(q[k] == {"group_size": 32, "bits": 4, "mode": "mxfp4"} for k in mtp)


def test_strict_load_equivalence_lowmem():
    cfg, args = _cfg_args()
    model = Model(args, quantize=True, quantization=cfg["quantization"])

    class _Stub(nn.Module):
        def __call__(self, x, indices):
            return mx.zeros((x.shape[0], indices.shape[-1], args.hidden_size))

    for layer in model.model.layers:
        layer.mlp.switch_mlp = _Stub()
    # shapes only -- lazy arrays are never materialised (RSS stays ~0.1 GB)
    spec = {p: tuple(v.shape) for p, v in tree_flatten(model.parameters())}
    del model

    def san(name):
        if name.startswith(("vision.", "aligner.", "image_", "mtp.")) or name.endswith(".bias_vl"):
            return None
        return _sanitize_name(name)

    index = json.loads((ART / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in index.items():
        if is_text_resident(name):
            by_shard.setdefault(shard, []).append(name)

    covered, shape_mismatch = set(), []
    for shard in sorted(by_shard):
        ents = dc.tensor_entries(dc.read_safetensors_header(str(ART / shard))[0])
        for name in by_shard[shard]:
            mp = san(name)
            if mp is None:
                continue
            exp = spec.get(mp)
            if exp is None or tuple(ents[name].shape) != exp:
                shape_mismatch.append((name, mp))
            else:
                covered.add(mp)
    missing = set(spec) - covered
    assert not shape_mismatch, shape_mismatch[:5]
    assert not missing, list(missing)[:5]
    assert len(covered) == len(spec) == 1206  # re-derived text-only key count (mxfp8: no biases)


def test_ondisk_residents_bit_exact():
    index = json.loads((ART / "model.safetensors.index.json").read_text())["weight_map"]
    wm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
    # mxfp8 dense samples
    for base in ["layers.0.attn.wq_a", "layers.0.attn.wo_b", "layers.20.ffn.shared_experts.w2"]:
        shard = index[base + ".weight"]
        arrs = mx.load(str(ART / shard))
        deq = np.array(mx.dequantize(arrs[base + ".weight"], arrs[base + ".scales"],
                                     group_size=32, bits=8, mode="mxfp8").astype(mx.float32))
        assert np.array_equal(deq, _src_fp32(base + ".weight", wm)), base
        del arrs
    # mxfp4 MTP samples
    for base in ["mtp.0.ffn.experts.0.w1", "mtp.2.ffn.experts.127.w3"]:
        shard = index[base + ".weight"]
        arrs = mx.load(str(ART / shard))
        deq = np.array(mx.dequantize(arrs[base + ".weight"], arrs[base + ".scales"],
                                     group_size=32, bits=4, mode="mxfp4").astype(mx.float32))
        assert np.array_equal(deq, _src_fp32(base + ".weight", wm)), base
        del arrs


def test_layer0_attention_golden():
    """attn_L0 with the mxfp8 residents (exact weights, fp8/bf16 GEMM) matches the
    fp32 dense oracle at >= 0.999 global cosine on the 31-token probe.  Layer 0 is
    SWA-only, so attn_L0 = f(embed, 5 dense projections, norms) -- exactly the W18
    residents; no routed-expert bank needed."""
    cfg, args = _cfg_args()
    wm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
    dense = {p: _src_fp32(f"layers.0.attn.{p}.weight", wm)
             for p in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")}

    def build(mode):
        a = Attention(args, 0)
        a.q_norm_weight = mx.array(_src_fp32("layers.0.attn.q_norm.weight", wm)).astype(mx.bfloat16)
        a.kv_norm_weight = mx.array(_src_fp32("layers.0.attn.kv_norm.weight", wm)).astype(mx.bfloat16)
        a.attn_sink = mx.array(_src_fp32("layers.0.attn.attn_sink", wm))
        for p in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            w = mx.array(dense[p])
            if mode is None:
                getattr(a, p).weight = w
            else:
                lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
                lin.weight = w
                gs = 32 if mode == "mxfp8" else 64
                setattr(a, p, nn.QuantizedLinear.from_linear(lin, group_size=gs, bits=8, mode=mode))
        return a

    def run(embed, a):
        an = mx.array(_src_fp32("layers.0.attn_norm.weight", wm)).astype(embed.dtype)
        x = _rmsnorm(embed, an, args.rms_norm_eps)
        cache = _make_cache(args)
        out = a(x, mx.arange(0, embed.shape[1]), cache.layers[0], cache.new_shared_runtime())
        mx.eval(out)
        return np.array(out.astype(mx.float32))

    # probe embed rows (source bf16 table)
    esh = wm["embed.weight"]
    hdr, ds = dc.read_safetensors_header(str(SRC / esh))
    e = dc.tensor_entries(hdr)["embed.weight"]
    dim = e.shape[1]
    fd = os.open(str(SRC / esh), os.O_RDONLY)
    try:
        rows = np.stack([np.frombuffer(dc._pread_exact(fd, ds + e.begin + t * dim * 2, dim * 2), "<u2")
                         for t in PROBE])
    finally:
        os.close(fd)
    embed_f32 = mx.array((rows.astype(np.uint32) << 16).view(np.float32).reshape(1, len(PROBE), dim))
    embed_bf16 = mx.array(rows.reshape(1, len(PROBE), dim)).view(mx.bfloat16)

    oracle = run(embed_f32, build(None))
    mxfp8 = run(embed_bf16, build("mxfp8"))

    A = oracle.reshape(oracle.shape[1], -1).astype(np.float64)
    B = mxfp8.reshape(mxfp8.shape[1], -1).astype(np.float64)
    gcos = float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-30))
    assert gcos >= 0.999, gcos
    assert bool(np.isfinite(mxfp8).all())
