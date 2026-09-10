"""DeepSeek-V4.1-Flash -> MTPLX SSD-streaming converter primitives.

Pure, testable building blocks used by
``scripts/convert_deepseek_v41_streamed.py``.  Nothing here touches Metal:
``mx.quantize`` runs on the CPU stream so no GPU flock is required.

Source facts (verified from the safetensors headers of the pinned revision
``dba1be0a40aa45a94ad051997016db3960a90277``):

* Routed experts ``layers.{0..39}.ffn.experts.{0..383}.w{1,2,3}`` are packed
  FP4 (E2M1), stored ``I8`` shape ``[out, in/2]`` (two nibbles per byte, low
  nibble first), with an ``F8_E8M0`` ``.scale`` shape ``[out, in/32]`` (one
  scale per 32 input columns; value ``2**(byte-127)``).  ``w1`` = gate_proj
  ``[2304,5120]``, ``w3`` = up_proj ``[2304,5120]``, ``w2`` = down_proj
  ``[5120,2304]``.
* Dense tensors (attention ``wq_a/wq_b/wkv/wo_a/wo_b``, ``shared_experts``,
  ``indexer.wq_b``, ``mtp.*`` dense) are ``F8_E4M3`` with ``F8_E8M0`` 32x32
  block scales (``weight_block_size [32,32]``, ``scale_fmt ue8m0``).
* Norms, ``hc_*`` vectors, ``ffn.gate`` weight/bias, ``attn_sink``, ``embed``,
  ``head``, vision/aligner weights are ``BF16``/``F32``.

The FP4 dequant here is a straight ``FP4_TABLE`` transcription (the reference
``inference/convert.py``'s ``cast_e2m1fn_to_e4m3fn`` is a *lossless FP4->FP8
recast* for its fp8 kernels; the true dequantized value is simply
``FP4_TABLE[nibble] * 2**(scale_exp-127)``, which is what we re-quantize).
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# geometry (DeepSeek-V4.1-Flash text backbone)
# --------------------------------------------------------------------------
HIDDEN_SIZE = 5120
MOE_INTERMEDIATE = 2304
N_ROUTED_EXPERTS = 384
ROUTED_LAYERS = tuple(range(40))  # layers.0 .. layers.39 all carry ffn.experts
TOP_K = 6
GROUP_SIZE = 64
EXPERT_BITS = 2
RESIDENT_BITS = 8
ALIGNMENT = 16 * 1024  # mtplx.expert_manifest.DEFAULT_ALIGNMENT

SOURCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
MODEL_KEY = "deepseek-v41-flash-expert-q2"

# _COMPONENTS order that mtplx.expert_manifest expects for an affine record.
COMPONENTS = (
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
)
# projection -> source ``w{n}`` name (DeepSeek convention: w1 gate, w2 down, w3 up)
PROJ_TO_SOURCE_W = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


def expert_record_bytes(
    hidden: int = HIDDEN_SIZE,
    inter: int = MOE_INTERMEDIATE,
    bits: int = EXPERT_BITS,
    group: int = GROUP_SIZE,
    param_bytes: int = 2,
) -> int:
    """Bytes of one affine expert record (matches spec.expert_record_bytes)."""
    params = 3 * hidden * inter
    packed = params * bits // 8
    scale_bias = (params // group) * 2 * param_bytes
    return packed + scale_bias


EXPERT_RECORD_BYTES = expert_record_bytes()  # 11_059_200 for the pinned geometry


# --------------------------------------------------------------------------
# decode lookup tables
# --------------------------------------------------------------------------
# Reference FP4 (E2M1) table, index = 4-bit code (inference/convert.py).
FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def build_e4m3_lut() -> np.ndarray:
    """256-entry LUT: OCP ``float8_e4m3fn`` byte -> float32 (no inf; 0x7F/0xFF NaN)."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b >> 7) & 1 else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0:
            val = sign * (man / 8.0) * (2.0 ** (1 - 7))  # subnormal, bias 7
        elif exp == 0xF and man == 0x7:
            val = float("nan")  # e4m3fn: only S.1111.111 is NaN
        else:
            val = sign * (1.0 + man / 8.0) * (2.0 ** (exp - 7))
        lut[b] = val
    return lut


def build_e8m0_lut() -> np.ndarray:
    """256-entry LUT: ``float8_e8m0fnu`` byte -> float32 (2**(byte-127); 0xFF NaN)."""
    lut = np.empty(256, dtype=np.float32)
    for b in range(256):
        lut[b] = float("nan") if b == 0xFF else float(2.0 ** (b - 127))
    return lut


E4M3_LUT = build_e4m3_lut()
E8M0_LUT = build_e8m0_lut()


# --------------------------------------------------------------------------
# dequantization (numpy, CPU)
# --------------------------------------------------------------------------
def dequant_fp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize a packed FP4 (E2M1) expert weight to float32.

    ``packed``: uint8 ``[out, in/2]`` (low nibble first -> even columns).
    ``scale``:  uint8 (E8M0) ``[out, in/32]`` -> one scale per 32 input columns.
    Returns float32 ``[out, in]`` == ``FP4_TABLE[nibble] * 2**(scale_exp-127)``.
    """
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    out, half = packed.shape
    in_dim = half * 2
    if scale.shape != (out, in_dim // 32):
        raise ValueError(
            f"scale shape {scale.shape} does not match packed {packed.shape}"
        )
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    weights = np.empty((out, in_dim), dtype=np.float32)
    weights[:, 0::2] = FP4_TABLE[low]
    weights[:, 1::2] = FP4_TABLE[high]
    scale_f = E8M0_LUT[np.ascontiguousarray(scale, dtype=np.uint8)]
    weights *= np.repeat(scale_f, 32, axis=1)
    return weights


def dequant_fp8_block(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize a dense ``F8_E4M3`` weight with ``F8_E8M0`` block scales.

    ``weight``: uint8 ``[O, I]`` (e4m3 bytes).
    ``scale``:  uint8 ``[O/bo, I/bi]`` (e8m0 bytes); block sizes derived from
    the ratio of shapes (32x32 or 128x128 for this checkpoint).
    Returns float32 ``[O, I]``.
    """
    weight = np.ascontiguousarray(weight, dtype=np.uint8)
    scale = np.ascontiguousarray(scale, dtype=np.uint8)
    o, i = weight.shape
    so, si = scale.shape
    if o % so or i % si:
        raise ValueError(f"weight {weight.shape} not divisible by scale {scale.shape}")
    bo, bi = o // so, i // si
    w = E4M3_LUT[weight]
    s = E8M0_LUT[scale]
    s = np.repeat(np.repeat(s, bo, axis=0), bi, axis=1)
    w *= s
    return w


# --------------------------------------------------------------------------
# affine (mx.quantize) requantization -> raw little-endian bytes
# --------------------------------------------------------------------------
def _cpu_mx():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    return mx


def quantize_affine(values_f32: np.ndarray, bits: int, group_size: int = GROUP_SIZE):
    """Affine-quantize a float32 matrix via ``mx.quantize`` (CPU, bf16 params).

    Returns ``(packed_u32, scales_bf16, biases_bf16)`` as mlx arrays, matching
    ``mtplx.expert_manifest._expected_component_shape`` (U32 weight, BF16
    scales/biases).
    """
    mx = _cpu_mx()
    w = mx.array(np.ascontiguousarray(values_f32, dtype=np.float32)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def mx_u32_bytes(arr) -> bytes:
    return np.array(arr).astype("<u4").tobytes()


def mx_bf16_bytes(arr) -> bytes:
    mx = _cpu_mx()
    return np.array(arr.view(mx.uint16)).astype("<u2").tobytes()


def component_bytes(packed, scales, biases) -> tuple[bytes, bytes, bytes]:
    return mx_u32_bytes(packed), mx_bf16_bytes(scales), mx_bf16_bytes(biases)


# --------------------------------------------------------------------------
# safetensors header / raw tensor reading (bounded RSS, one tensor at a time)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int  # offset within the data section
    end: int


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header dict, data_section_start). Does not read tensor bytes."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def tensor_entries(header: dict) -> dict[str, TensorEntry]:
    out: dict[str, TensorEntry] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        out[name] = TensorEntry(name, meta["dtype"], tuple(meta["shape"]), b, e)
    return out


def implied_file_size(header: dict, data_start: int) -> int:
    """File size the header implies: data_start + max tensor end offset."""
    max_end = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        max_end = max(max_end, meta["data_offsets"][1])
    return data_start + max_end


def read_tensor_raw(fd: int, data_start: int, entry: TensorEntry) -> bytes:
    length = entry.end - entry.begin
    return _pread_exact(fd, data_start + entry.begin, length)


def _pread_exact(fd: int, offset: int, length: int) -> bytes:
    chunks = []
    remaining = length
    pos = offset
    while remaining:
        chunk = os.pread(fd, remaining, pos)
        if not chunk:
            raise EOFError(f"short read at {pos}, wanted {remaining} more")
        chunks.append(chunk)
        pos += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# dtype byte width for raw reads / numpy views
_ST_DTYPE_NUMPY = {"F32": np.float32, "F16": np.float16}
_ST_DTYPE_ITEMSIZE = {
    "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E8M0": 1,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4, "F64": 8,
}


def raw_to_f32(entry: TensorEntry, raw: bytes) -> np.ndarray:
    """Decode a resident tensor's raw bytes to float32 (for requantization)."""
    if entry.dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).reshape(entry.shape).copy()
    if entry.dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(entry.shape).copy()
    if entry.dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(entry.shape)
    if entry.dtype == "F8_E4M3":
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(entry.shape)
        return E4M3_LUT[u8].copy()
    raise ValueError(f"cannot decode dtype {entry.dtype} to f32 for {entry.name}")


# --------------------------------------------------------------------------
# resident classification
# --------------------------------------------------------------------------
def is_bank_expert(name: str) -> bool:
    """Backbone routed-expert tensor that belongs in the streamed Q2 bank."""
    import re

    return re.match(r"^layers\.(\d+)\.ffn\.experts\.\d+\.w[123]\.(weight|scale)$", name) is not None


def is_mtp_expert(name: str) -> bool:
    import re

    return re.match(r"^mtp\.\d+\.ffn\.experts\.", name) is not None


def is_engram(name: str) -> bool:
    return ".engram." in name


def is_source_scale(name: str) -> bool:
    """A block-scale sibling (``*.scale``) consumed during dequant, never emitted."""
    return name.endswith(".scale")


# names/shapes kept at exact precision (never affine-quantized) in residents.
def _keep_exact(name: str, shape: tuple[int, ...], dtype: str) -> bool:
    if len(shape) != 2:
        return True  # 1-D: norms, hc vectors, biases, attn_sink, image_* markers
    if name.endswith(".bias") or name.endswith(".bias_vl"):
        return True
    if ".gate." in name:  # ffn router weight/bias stays exact
        return True
    lowered = name
    if "norm" in lowered or "hc_" in lowered or "attn_sink" in lowered:
        return True
    # A quantizable matmul needs its input dim divisible by the group size.
    if shape[1] % GROUP_SIZE != 0:
        return True
    return False


def resident_disposition(entry: TensorEntry) -> str:
    """One of: 'quantize' (q8 affine), 'keep' (verbatim), 'drop' (scale sibling)."""
    if is_source_scale(entry.name):
        return "drop"
    if _keep_exact(entry.name, entry.shape, entry.dtype):
        return "keep"
    return "quantize"


# ==========================================================================
# Engram conditional-memory tables (disk-backed affine-8 row bank)
# --------------------------------------------------------------------------
# Additive helpers for ``scripts/convert_deepseek_v41_engram.py`` and
# ``mtplx/engram_bank.py``.  Nothing above this line is modified.
#
# Source facts (DeepSeek-V4.1-Flash, revision dba1be0a40aa45a94ad051997016db3960a90277):
#   config: engram_layer_ids [1,14]; engram_num_embeddings [384006168,384016682];
#   engram_max_ngram_size 4; engram_vocab_size 16000000; engram_n_heads 8;
#   engram_head_dim 256; engram_compressed_vocab_size 99092; engram_pad_id 2.
# Source tensors (per engram layer L, in shard model-000{47,48}-of-00048.safetensors):
#   layers.L.engram.embed.weight  F8_E4M3  [num_embeddings, head_dim=256]   (the table)
#   layers.L.engram.embed.scale   F8_E8M0  [num_embeddings, head_dim//32=8]  (one scale per
#                                  (row, 32-col group); value 2**(byte-127))
#   layers.L.engram.{k_weight,q_weight,wkv.weight,wkv.scale}  small residents (already q8/exact)
# model.py ParallelEngramEmbedding.forward dequant (mirrored EXACTLY):
#   v = E4M3_LUT[weight]                     # [.,256] f32
#   s = E8M0_LUT[scale]                      # [.,8]   f32
#   out = (v.reshape(.,8,32) * s[.,:,None]).reshape(.,256)   # == dequant_fp8_block(v, s)
# One table row == head_dim (256) values (NOT n_heads*head_dim).  At inference each token
# gathers ``n_hash_cols = (max_ngram_size-1)*n_heads = 24`` rows PER engram layer -- one
# per (n-gram size 2..4, head) -- each hashed to a row index via ``hash % prime + offset``.

ENGRAM_LAYER_IDS = (1, 14)
ENGRAM_NUM_EMBEDDINGS = (384006168, 384016682)
ENGRAM_MAX_NGRAM_SIZE = 4
ENGRAM_VOCAB_SIZE = 16_000_000
ENGRAM_N_HEADS = 8
ENGRAM_HEAD_DIM = 256
ENGRAM_COMPRESSED_VOCAB_SIZE = 99092
ENGRAM_PAD_ID = 2
ENGRAM_FP8_BLOCK = 32  # source embed.scale group width (one scale per 32-col group)
ENGRAM_BITS = 8        # output affine bank bit width
ENGRAM_GROUP_SIZE = 64  # output affine group size


def engram_n_hash_cols(max_ngram_size: int = ENGRAM_MAX_NGRAM_SIZE, n_heads: int = ENGRAM_N_HEADS) -> int:
    """Rows a single token gathers per engram layer (=24)."""
    return (max_ngram_size - 1) * n_heads


def engram_record_bytes(head_dim: int = ENGRAM_HEAD_DIM, bits: int = ENGRAM_BITS,
                        group: int = ENGRAM_GROUP_SIZE, param_bytes: int = 2) -> int:
    """Bytes of one affine bank record: packed weights + bf16 scales + bf16 biases."""
    packed = head_dim * bits // 8
    groups = head_dim // group
    return packed + 2 * groups * param_bytes


def engram_record_layout(head_dim: int = ENGRAM_HEAD_DIM, bits: int = ENGRAM_BITS,
                        group: int = ENGRAM_GROUP_SIZE, param_bytes: int = 2) -> dict:
    """Byte layout of one record (offsets are within the fixed-size record)."""
    packed = head_dim * bits // 8
    groups = head_dim // group
    return {
        "weight": {"offset": 0, "length": packed, "dtype": "U32",
                   "shape": [head_dim * bits // 32]},
        "scales": {"offset": packed, "length": groups * param_bytes, "dtype": "BF16",
                   "shape": [groups]},
        "biases": {"offset": packed + groups * param_bytes, "length": groups * param_bytes,
                   "dtype": "BF16", "shape": [groups]},
        "record_bytes": packed + 2 * groups * param_bytes,
    }


ENGRAM_RECORD_BYTES = engram_record_bytes()  # 272 for head_dim=256, 8-bit, group 64


def _isprime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def engram_find_next_prime(start: int, seen: set) -> int:
    """Smallest prime > ``start`` not already handed out (mirrors engram.find_next_prime)."""
    c = start + 1
    while not _isprime(c) or c in seen:
        c += 1
    return c


def engram_prime_layout(layer_ids: tuple = ENGRAM_LAYER_IDS,
                        max_ngram_size: int = ENGRAM_MAX_NGRAM_SIZE,
                        n_heads: int = ENGRAM_N_HEADS,
                        vocab_size: int = ENGRAM_VOCAB_SIZE) -> tuple:
    """``[layer][n-gram size 2..max][head]`` prime bucket modulus.

    Mirrors ``EngramLayout.from_args``: primes are drawn in order from ``vocab_size-1`` and
    never reused (across n-gram sizes, heads, or layers), so bucket ranges stay disjoint and
    a layer's primes sum to ``engram_num_embeddings``.
    """
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current = engram_find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return tuple(primes)


def engram_flat_offsets(primes_for_layer) -> tuple:
    """``cumsum([0, *flat[:-1]])`` over a layer's flattened (n-gram,head) primes.

    Returns ``(offsets, flat_primes, total_rows)``; ``total_rows`` == the layer's
    ``num_embeddings``.
    """
    flat = [int(p) for per_ngram in primes_for_layer for p in per_ngram]
    offsets, acc = [], 0
    for p in flat:
        offsets.append(acc)
        acc += p
    return offsets, flat, acc


def compute_engram_hash_multipliers(layer_ids: tuple = ENGRAM_LAYER_IDS,
                                    max_ngram_size: int = ENGRAM_MAX_NGRAM_SIZE,
                                    compressed_vocab_size: int = ENGRAM_COMPRESSED_VOCAB_SIZE) -> np.ndarray:
    """``[n_layers, max_ngram_size]`` odd int64 multipliers (mirrors compute_hash_multipliers).

    The reference passes the *compressed* vocab size here (every multiplier derives from it).
    """
    max_long = int(np.iinfo(np.int64).max)
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        vals = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(vals * 2 + 1)
    return np.stack(rows)


def dequant_engram_embed(weight_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """Dequantize engram embed rows to float32, EXACTLY as model.py ParallelEngramEmbedding.

    ``weight_u8``: uint8 ``[rows, head_dim]`` (F8_E4M3 bytes).
    ``scale_u8``:  uint8 ``[rows, head_dim//32]`` (F8_E8M0 bytes).
    Equivalent to :func:`dequant_fp8_block` with 1x32 blocks; kept as a named entry point.
    """
    if weight_u8.ndim != 2 or scale_u8.ndim != 2:
        raise ValueError("engram embed rows must be 2-D [rows, head_dim]")
    rows, hd = weight_u8.shape
    if scale_u8.shape != (rows, hd // ENGRAM_FP8_BLOCK):
        raise ValueError(f"scale shape {scale_u8.shape} != {(rows, hd // ENGRAM_FP8_BLOCK)}")
    return dequant_fp8_block(weight_u8, scale_u8)


def quantize_engram_rows(values_f32: np.ndarray, bits: int = ENGRAM_BITS, group: int = ENGRAM_GROUP_SIZE):
    """Affine-quantize engram rows via ``mx.quantize(mode='affine')`` (CPU, bf16 params).

    Returns ``(packed_u32, scales_bf16, biases_bf16)`` mlx arrays.  ``mode='affine'`` is passed
    explicitly (mxfp8 is not an exact repack in mlx 0.32).
    """
    mx = _cpu_mx()
    w = mx.array(np.ascontiguousarray(values_f32, dtype=np.float32)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(w, group_size=group, bits=bits, mode="affine")
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def engram_chunk_records(values_f32: np.ndarray, bits: int = ENGRAM_BITS,
                        group: int = ENGRAM_GROUP_SIZE) -> np.ndarray:
    """Quantize a ``[rows, head_dim]`` f32 chunk to affine-8 and return a contiguous
    ``[rows, record_bytes]`` uint8 array: ``packed(u32 LE) | scales(bf16 LE) | biases(bf16 LE)``.
    """
    mx = _cpu_mx()
    rows, hd = values_f32.shape
    packed, scales, biases = quantize_engram_rows(values_f32, bits=bits, group=group)
    p = np.ascontiguousarray(np.array(packed).astype("<u4")).reshape(rows, -1)
    s = np.ascontiguousarray(np.array(scales.view(mx.uint16)).astype("<u2")).reshape(rows, -1)
    b = np.ascontiguousarray(np.array(biases.view(mx.uint16)).astype("<u2")).reshape(rows, -1)
    rec = np.concatenate(
        [p.view(np.uint8), s.view(np.uint8), b.view(np.uint8)], axis=1
    )
    exp = engram_record_bytes(hd, bits, group)
    if rec.shape[1] != exp:
        raise RuntimeError(f"engram record width {rec.shape[1]} != expected {exp}")
    return np.ascontiguousarray(rec, dtype=np.uint8)


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """Reinterpret bf16 bit patterns (uint16) as float32 (upper 16 bits of the f32)."""
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def dequant_affine_record(weight_u8: np.ndarray, scales_bf16: np.ndarray, biases_bf16: np.ndarray,
                          head_dim: int = ENGRAM_HEAD_DIM, group: int = ENGRAM_GROUP_SIZE) -> np.ndarray:
    """Pure-numpy affine dequant of a bank record (f32).  ``level*scale + bias`` per group.

    ``weight_u8``: uint8 ``[rows, head_dim]`` quantized levels (8-bit).
    ``scales_bf16``/``biases_bf16``: uint16 ``[rows, head_dim//group]`` bf16 bit patterns.
    Matches ``mx.dequantize`` up to bf16 rounding (mlx does the arithmetic in bf16).
    """
    w = np.asarray(weight_u8, dtype=np.float32)
    rows, hd = w.shape
    s = bf16_bits_to_f32(scales_bf16).reshape(rows, -1)
    b = bf16_bits_to_f32(biases_bf16).reshape(rows, -1)
    grp = np.arange(hd) // group
    return w * s[:, grp] + b[:, grp]
