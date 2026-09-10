"""Real-artifact checks for the DeepSeek-V4.1-Flash Engram resident sidecar (W4).

Runs only when ``engram/engram-residents.safetensors`` is present in the artifact
(``$DSV41_ARTIFACT_DIR`` or the default streaming-q2 path); skips otherwise, so the
suite is green on a box without the 376 GiB artifact.

Covers: the manifest ``residents`` entry (file / tensor names / dtypes / shapes /
byte-total / sha256, and a re-derivation of ``manifest_sha256``); the loader
(:func:`mtplx.engram_v41.load_engram_residents`) shapes/dtypes; the affine-q8 ``wkv``
callable == dequantize-then-matmul; and construction of a full :class:`EngramV41` hook
from the sidecar + manifest + the on-disk row bank, with a tiny finite forward.

Run under ``nice -n 19``, without ``-n auto``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mtplx.engram_v41 import EngramResidents, load_engram_residents, _StepState
from mtplx.engram_bank import EngramBank

_DEFAULT_ART = "/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2"
ART = Path(os.environ.get("DSV41_ARTIFACT_DIR", _DEFAULT_ART))
ENGRAM_DIR = ART / "engram"
SIDECAR = ENGRAM_DIR / "engram-residents.safetensors"
MANIFEST = ENGRAM_DIR / "engram-manifest.json"

pytestmark = pytest.mark.skipif(
    not SIDECAR.is_file(),
    reason=f"engram residents sidecar not present: {SIDECAR}",
)

LAYERS = (1, 14)
DIM, HC_MULT = 5120, 4
WKV_OUT = DIM * (HC_MULT + 1)          # 25600
HEAD_DIM, N_HASH_COLS = 256, 24
COLS_HEAD = N_HASH_COLS * HEAD_DIM     # 6144


def test_manifest_residents_entry():
    m = json.loads(MANIFEST.read_text())
    assert "residents" in m, "manifest has no residents entry"
    r = m["residents"]
    assert r["file"] == "engram-residents.safetensors"
    assert r["total_bytes"] == SIDECAR.stat().st_size
    assert len(r["sha256"]) == 64

    names = {t["name"] for t in r["tensors"]}
    dtype_by_name = {t["name"]: t["dtype"] for t in r["tensors"]}
    shape_by_name = {t["name"]: t["shape"] for t in r["tensors"]}
    for L in LAYERS:
        base = f"layers.{L}.engram"
        assert dtype_by_name[f"{base}.wkv.weight"] == "U32"
        assert dtype_by_name[f"{base}.wkv.scales"] == "BF16"
        assert dtype_by_name[f"{base}.wkv.biases"] == "BF16"
        assert dtype_by_name[f"{base}.q_weight"] == "F32"
        assert dtype_by_name[f"{base}.k_weight"] == "F32"
        assert shape_by_name[f"{base}.q_weight"] == [HC_MULT, DIM]
        assert shape_by_name[f"{base}.k_weight"] == [HC_MULT, DIM]
        assert shape_by_name[f"{base}.wkv.weight"] == [WKV_OUT, COLS_HEAD * 8 // 32]
        assert shape_by_name[f"{base}.wkv.scales"] == [WKV_OUT, COLS_HEAD // 64]
        assert f"{base}.wkv.biases" in names

    # manifest_sha256 re-derives with write_manifest's recipe (dict w/o the field, indent=2)
    payload = {k: v for k, v in m.items() if k != "manifest_sha256"}
    assert m["manifest_sha256"] == hashlib.sha256(
        json.dumps(payload, indent=2).encode()).hexdigest()


def test_sidecar_sha256_matches_manifest():
    m = json.loads(MANIFEST.read_text())
    h = hashlib.sha256()
    with open(SIDECAR, "rb") as f:
        for blob in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(blob)
    assert h.hexdigest() == m["residents"]["sha256"]


@pytest.mark.parametrize("layer", LAYERS)
def test_load_residents_shapes_dtypes(layer):
    res = load_engram_residents(ENGRAM_DIR, layer)
    assert isinstance(res, EngramResidents)
    assert res.q_weight.dtype == mx.float32
    assert res.k_weight.dtype == mx.float32
    assert tuple(res.q_weight.shape) == (HC_MULT, DIM)
    assert tuple(res.k_weight.shape) == (HC_MULT, DIM)
    assert res.dim == DIM and res.hc_mult == HC_MULT
    assert res.wkv_packed.dtype == mx.uint32
    assert res.wkv_scales.dtype == mx.bfloat16
    assert res.wkv_biases.dtype == mx.bfloat16
    assert tuple(res.wkv_packed.shape) == (WKV_OUT, COLS_HEAD * 8 // 32)
    assert tuple(res.wkv_scales.shape) == (WKV_OUT, COLS_HEAD // 64)


@pytest.mark.parametrize("layer", LAYERS)
def test_wkv_callable_matches_dequant(layer):
    res = load_engram_residents(ENGRAM_DIR, layer)
    rng = np.random.default_rng(layer)
    x = mx.array(rng.standard_normal((2, 3, COLS_HEAD)).astype(np.float32))
    kv = res.wkv(x)
    assert tuple(kv.shape) == (2, 3, WKV_OUT)
    assert bool(mx.all(mx.isfinite(kv)).item())
    # quantized_matmul == dequantize-then-matmul up to bf16 accumulation; cosine is the
    # robust wiring check (a transposed/misindexed wkv would collapse it toward 0).
    w = mx.dequantize(res.wkv_packed, res.wkv_scales, res.wkv_biases,
                      group_size=res.group_size, bits=res.bits, mode="affine")
    ref = x @ w.T
    a, b = np.array(kv).reshape(-1), np.array(ref).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    assert cos >= 0.999, cos
    rel = float(mx.max(mx.abs(kv - ref)).item()) / (float(mx.max(mx.abs(ref)).item()) + 1e-6)
    assert rel < 5e-2, rel


@pytest.mark.parametrize("layer", LAYERS)
def test_construct_engramv41_from_sidecar(layer):
    bank_file = ENGRAM_DIR / f"engram-L{layer}.bin"
    if not bank_file.is_file():
        pytest.skip(f"engram bank not present: {bank_file}")
    res = load_engram_residents(ENGRAM_DIR, layer)
    bank = EngramBank.open(ENGRAM_DIR, layer, cache_rows=64)
    try:
        # EngramV41 gathers rows through the generic NGramRowCache (EngramBank.cache)
        module = res.build_module(row_cache=bank.cache, layer_hash_index=0,
                                  norm_eps=1e-6, clamp_value=1e-6)
        assert module.head_dim == HEAD_DIM
        B, L = 1, 2
        rng = np.random.default_rng(100 + layer)
        row_ids = rng.integers(0, bank.rows, size=(B, L, 1, N_HASH_COLS)).astype(np.int64)
        x = mx.array(rng.standard_normal((B, L, HC_MULT, DIM)).astype(np.float32) * 0.1)
        out = module(x, np.zeros((B, L), np.int64), _StepState(row_ids=row_ids))
        assert tuple(out.shape) == (B, L, HC_MULT, DIM)
        assert bool(mx.all(mx.isfinite(out)).item())
    finally:
        bank.close()
