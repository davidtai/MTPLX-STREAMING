"""DeepSeek-V4.1 mxfp4 expert-bank reader + fp32 dequantizer  (DSV4.1 F34, CPU-only).

The routed-expert bank is a single ``experts.bin`` (288 GB) indexed by ``expert-manifest.json``.
Each ``(layer, expert)`` record has 6 segments: ``{gate,up,down}_proj.weight`` (U32, mxfp4-packed
``[out, in/8]``) and ``.scales`` (U8, E8M0 ``[out, in/32]``).  mxfp4 = OCP MX E2M1 nibbles + one
E8M0 scale per 32 along the input dim.

Dequantization uses ``mx.dequantize(..., group_size=32, bits=4, mode="mxfp4")`` on the CPU device
(returns bf16, cast to fp32; the mxfp4 value set is exact in bf16).  A pure-numpy E2M1/E8M0 decoder
(:func:`dequantize_numpy`) is cross-checked bit-for-bit against MLX in the tests and in
:func:`load_projection` (first call), per the F34 spec.

CPU-ONLY. Import does not touch the GPU or read ``experts.bin``. Callers must have run
``mx.set_default_device(mx.cpu)``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MODEL_DIR = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4")
MANIFEST_PATH = os.path.join(MODEL_DIR, "expert-manifest.json")
EXPERTS_BIN = os.path.join(MODEL_DIR, "experts.bin")

# OCP MX E2M1 code -> value (index 0..15: low nibble sign at bit 3).
E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
COMPONENTS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Segment:
    component: str
    tensor: str
    offset: int
    length: int
    dtype: str
    shape: tuple


@dataclass(frozen=True)
class ExpertRecord:
    layer: int
    expert: int
    logical_bytes: int
    sha256: str
    segments: dict  # component_key ("gate_proj.weight" etc.) -> Segment


@lru_cache(maxsize=1)
def load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _record_index(path: str = MANIFEST_PATH) -> dict:
    """(layer, expert) -> ExpertRecord, built once from the manifest."""
    m = load_manifest(path)
    q = m["quantization"]
    assert q["bits"] == 4 and q["group_size"] == 32 and q["mode"] == "mxfp4", q
    idx = {}
    for r in m["records"]:
        segs = {
            s["component"]: Segment(
                component=s["component"], tensor=s["tensor"], offset=int(s["offset"]),
                length=int(s["length"]), dtype=s["dtype"], shape=tuple(s["shape"]),
            )
            for s in r["segments"]
        }
        idx[(int(r["layer"]), int(r["expert"]))] = ExpertRecord(
            layer=int(r["layer"]), expert=int(r["expert"]),
            logical_bytes=int(r["logical_bytes"]), sha256=r.get("sha256", ""), segments=segs,
        )
    return idx


def get_record(layer: int, expert: int) -> ExpertRecord:
    idx = _record_index()
    key = (layer, expert)
    if key not in idx:
        raise KeyError(f"no record for layer={layer} expert={expert}")
    return idx[key]


def _read_bytes(offset: int, length: int, path: str = EXPERTS_BIN) -> bytes:
    """Exact pread of one segment; never maps the whole 288 GB file."""
    with open(path, "rb", buffering=0) as f:
        f.seek(offset)
        buf = f.read(length)
    if len(buf) != length:
        raise IOError(f"short read at {offset}: got {len(buf)} want {length}")
    return buf


def read_segment(seg: Segment, path: str = EXPERTS_BIN) -> np.ndarray:
    buf = _read_bytes(seg.offset, seg.length, path)
    if seg.dtype == "U32":
        a = np.frombuffer(buf, dtype="<u4")
    elif seg.dtype == "U8":
        a = np.frombuffer(buf, dtype=np.uint8)
    else:
        raise ValueError(f"unexpected segment dtype {seg.dtype}")
    return a.reshape(seg.shape)


def verify_record_sha256(rec: ExpertRecord, path: str = EXPERTS_BIN) -> str:
    """Hash the record's contiguous bytes and compare with the manifest sha256.

    Segments of a record are contiguous in the bank (verified: offsets are absolute and packed),
    so the record is one span [first_offset, first_offset+logical_bytes).
    """
    first = min(s.offset for s in rec.segments.values())
    buf = _read_bytes(first, rec.logical_bytes, path)
    got = hashlib.sha256(buf).hexdigest()
    if rec.sha256 and got != rec.sha256:
        raise ValueError(f"sha256 mismatch L{rec.layer}E{rec.expert}: {got} != {rec.sha256}")
    return got


def dequantize_numpy(wq: np.ndarray, scales: np.ndarray, group_size: int = 32) -> np.ndarray:
    """Pure-numpy mxfp4 dequantization -> fp32 [out, in].  Bit-exact vs mx.dequantize (see tests).

    wq: uint32 [out, in/8] (8 E2M1 nibbles per u32, low->high). scales: uint8 [out, in/32] (E8M0).
    """
    wq = np.ascontiguousarray(wq).astype(np.uint32)
    out, nu32 = wq.shape
    n_in = nu32 * 8
    nib = np.empty((out, n_in), dtype=np.int64)
    for i in range(8):
        nib[:, i::8] = (wq >> np.uint32(4 * i)) & np.uint32(0xF)
    vals = E2M1[nib]  # [out, in]
    factor = (2.0 ** (scales.astype(np.float64) - 127.0)).astype(np.float32)  # [out, in/32]
    factor = np.repeat(factor, group_size, axis=1)
    if factor.shape[1] != n_in:
        raise ValueError(f"scale width {factor.shape[1]} != in {n_in}")
    return (vals * factor).astype(np.float32)


def dequantize_mlx(wq: np.ndarray, scales: np.ndarray, group_size: int = 32) -> np.ndarray:
    """mx.dequantize mxfp4 on the CPU device -> fp32 [out, in]."""
    import mlx.core as mx

    W = mx.dequantize(
        mx.array(np.ascontiguousarray(wq).astype(np.uint32)),
        mx.array(np.ascontiguousarray(scales).astype(np.uint8)),
        group_size=group_size, bits=4, mode="mxfp4",
    ).astype(mx.float32)
    return np.array(W)


_XCHECKED = {"done": False}


def load_projection(layer: int, expert: int, component: str, *, cross_check: bool = True,
                    verify_sha: bool = False) -> np.ndarray:
    """fp32 reference weight ``W_ref`` in HF orientation ``[out_features, in_features]``.

    On the first call (per process) the numpy and MLX dequantizers are asserted bit-identical,
    unless ``cross_check`` is disabled.  ``verify_sha`` additionally checks the record hash.
    """
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got {component}")
    rec = get_record(layer, expert)
    if verify_sha:
        verify_record_sha256(rec)
    wq = read_segment(rec.segments[f"{component}.weight"])       # uint32 [out, in/8]
    scales = read_segment(rec.segments[f"{component}.scales"])   # uint8  [out, in/32]
    W = dequantize_mlx(wq, scales)
    if cross_check and not _XCHECKED["done"]:
        Wn = dequantize_numpy(wq, scales)
        if not np.array_equal(Wn, W):
            raise AssertionError(
                f"numpy vs MLX mxfp4 mismatch (max |d|={np.abs(Wn - W).max()}) "
                f"L{layer}E{expert} {component}")
        _XCHECKED["done"] = True
    return W


def eschamoe_orientation(W_ref: np.ndarray) -> np.ndarray:
    """HF weight [out, in]  ->  eschamoe weight [in, out] (the forward uses xh[.,in] @ W[in,out])."""
    return np.ascontiguousarray(W_ref.T)
