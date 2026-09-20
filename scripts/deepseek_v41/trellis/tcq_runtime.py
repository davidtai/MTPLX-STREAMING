"""F39: runtime consumers of the tcq3 (eschamoe K=3 trellis) expert bank produced by F38 ``transcode_bank.py``.

This module is the retained DeepSeek-V4.1 runner's *reader* for the lossy 3-bit bank.  It owns, in one place:

  * :class:`TcqGeometry`   — the fixed whole-record layout (13,290,496 B/record; code + routs contiguous), derived
                             from the manifest and asserted against the F38 constants ONCE at construction.
  * :func:`read_tcq_manifest` — accepts ``quantization.mode == "tcq3"`` and the new segments; every model-file /
                             geometry / dtype check happens here, not per token (AGENTS.md "correct by design").
  * :func:`slice_record`   — map ONE whole record's bytes to its three code segments + three routs.
  * :func:`effective_weight_mx` / :func:`forward_decode_verify` — the decode math the verify lane runs:
        y = t128( t128(x) @ W_q ) * rout          (matmul in the trellis domain; W_q = vendor decode of the code)
     and the equivalent effective weight  E = B_in W_q B_out diag(rout)  (rin == 1).  Bit-for-bit the same algebra
     as ``tcq_encode.effective_weight`` / ``mtplx.eschamoe`` (validated to 1e-5 in tests).
  * :func:`load_resident_routs` — the whole-bank rout arrays (15,360 x 19,456 B = 299 MB), loaded resident at
                             construction and indexed by global record index — never read per decode miss.
  * :func:`decode_expert_to_bf16` — the PREFILL/seed path: decode one (expert, layer) to a dense bf16 effective
                             weight ONCE as the prefill streams experts (the per-assignment tile kernel is wrong
                             for prefill — it would re-read every expert per token).
  * :class:`TcqDecodeOps`  — the DECODE-verify ops, mirroring ``plane_lane.PackedOps`` but reading the whole-record
                             bank through the stride-aware tile kernel with ``t128`` before and ``t128 * rout`` after.

The stride-aware kernel and the whole-record geometry constants live in :mod:`tcq_kernels`.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os

import numpy as np

# tcq_kernels holds the whole-record geometry constants and the stride-aware kernel factory.
import tcq_kernels as tk

# ---------------------------------------------------------------------------- fixed model shape (validated, not trusted)
N_LAYERS = 40
N_EXPERTS_PER_LAYER = 384
N_RECORDS = N_LAYERS * N_EXPERTS_PER_LAYER            # 15,360
COMPONENTS = ("gate_proj", "up_proj", "down_proj")
# eschamoe orientation W[in, out]: gate/up decode to [5120, 2304]; down to [2304, 5120].
PROJ_IN_OUT = {"gate_proj": (5120, 2304), "up_proj": (5120, 2304), "down_proj": (2304, 5120)}
K_BITS = 3
NW = 48                                               # int16 words per 16x16 K=3 tile
MXFP4_SLOT_BYTES = 17_694_720                          # retained mxfp4 weight-code bytes per slot (scales resident/separate)


# ---------------------------------------------------------------------------- geometry (derived from manifest, asserted)

@dataclass(frozen=True)
class Segment:
    component: str          # e.g. "gate_proj.code" / "down_proj.rout"
    kind: str               # "code" | "rout"
    proj: str               # "gate_proj" | "up_proj" | "down_proj"
    byte_offset: int        # offset within the record
    byte_length: int
    word_offset: int        # int16-word offset within the record (byte_offset // 2)
    shape: tuple


@dataclass(frozen=True)
class TcqGeometry:
    """Immutable per-record layout for the tcq3 bank; every field validated against the F38 format at read time."""
    record_bytes: int
    record_words: int
    segments: tuple                                    # tuple[Segment] in record order
    n_layers: int = N_LAYERS
    n_experts_per_layer: int = N_EXPERTS_PER_LAYER

    @property
    def n_records(self) -> int:
        return self.n_layers * self.n_experts_per_layer

    def code_segment(self, proj: str) -> Segment:
        return next(s for s in self.segments if s.proj == proj and s.kind == "code")

    def rout_segment(self, proj: str) -> Segment:
        return next(s for s in self.segments if s.proj == proj and s.kind == "rout")

    def code_word_offset(self, proj: str) -> int:
        return self.code_segment(proj).word_offset

    # --- cache-row / residency accounting ---------------------------------------------------------------------
    def slot_bytes(self) -> int:
        """Bytes one decode slot holds: the WHOLE record (code + routs), read contiguously into one cache row."""
        return self.record_bytes

    def cache_row_bytes(self) -> int:
        """A cache 'row' = one slot index present in every one of the 40 MoE layers (== how the retained bank
        counts a dropped row: 40 x slot_bytes).  mxfp4 today: 40 x 17,694,720 = 707,788,800 B."""
        return self.n_layers * self.slot_bytes()

    def resident_rout_bytes(self) -> int:
        """Whole-bank routs loaded resident at construction: 15,360 x (2304 + 2304 + 5120) x 2 B."""
        per_record = sum(self.rout_segment(p).byte_length for p in COMPONENTS)
        return self.n_records * per_record

    def resident_fraction_vs_mxfp4(self) -> float:
        """Slots per unit memory relative to mxfp4 (code bytes only): 17,694,720 / 13,290,496 = 1.331 (~+33%)."""
        return MXFP4_SLOT_BYTES / self.slot_bytes()


def _expected_geometry() -> TcqGeometry:
    """The F38 record layout as constants (transcode_bank.record_layout on the real shapes)."""
    segs = []
    off = 0
    for proj in COMPONENTS:
        in_f, out_f = PROJ_IN_OUT[proj]
        nI, nJ = in_f // 16, out_f // 16
        code_len = nI * nJ * NW * 2
        segs.append(Segment(f"{proj}.code", "code", proj, off, code_len, off // 2, (nI, nJ, NW)))
        off += code_len
        rout_len = out_f * 2
        segs.append(Segment(f"{proj}.rout", "rout", proj, off, rout_len, off // 2, (out_f,)))
        off += rout_len
    return TcqGeometry(record_bytes=off, record_words=off // 2, segments=tuple(segs))


def _geometry_from_manifest_record(rec: dict) -> TcqGeometry:
    """Build the geometry from a manifest record's segments (byte offsets/lengths/dtypes/shapes)."""
    by_name = {s["component"]: s for s in rec["segments"]}
    segs = []
    for proj in COMPONENTS:
        code = by_name.get(f"{proj}.code")
        rout = by_name.get(f"{proj}.rout")
        if code is None or rout is None:
            raise ValueError(f"tcq3 record missing {proj}.code / {proj}.rout segments")
        if code["dtype"] != "I16" or rout["dtype"] != "F16":
            raise ValueError(f"tcq3 {proj} dtypes must be I16 code / F16 rout, got {code['dtype']}/{rout['dtype']}")
        if code["offset"] % 2 or rout["offset"] % 2:
            raise ValueError("tcq3 segment byte offset is not int16-aligned")
        segs.append(Segment(f"{proj}.code", "code", proj, code["offset"], code["length"],
                            code["offset"] // 2, tuple(code["shape"])))
        segs.append(Segment(f"{proj}.rout", "rout", proj, rout["offset"], rout["length"],
                            rout["offset"] // 2, tuple(rout["shape"])))
    record_bytes = rec["logical_bytes"]
    return TcqGeometry(record_bytes=record_bytes, record_words=record_bytes // 2, segments=tuple(segs))


def validate_geometry(geom: TcqGeometry) -> TcqGeometry:
    """Assert the derived geometry equals the fixed F38 layout and the stride constants used by the kernel.

    Raises before any decode if anything differs.  This is the single construction-time gate."""
    expected = _expected_geometry()
    if geom.record_bytes != expected.record_bytes:
        raise ValueError(f"tcq3 record_bytes {geom.record_bytes} != expected {expected.record_bytes}")
    if geom.record_words != tk.TCQ3_RECORD_WORDS:
        raise ValueError(f"tcq3 record_words {geom.record_words} != kernel stride {tk.TCQ3_RECORD_WORDS}")
    if tuple(s.component for s in geom.segments) != tuple(s.component for s in expected.segments):
        raise ValueError("tcq3 segment order differs from the F38 record layout")
    for got, exp in zip(geom.segments, expected.segments):
        if (got.byte_offset, got.byte_length, got.word_offset, got.shape) != \
           (exp.byte_offset, exp.byte_length, exp.word_offset, exp.shape):
            raise ValueError(f"tcq3 segment {got.component} layout {got} != expected {exp}")
    for proj in COMPONENTS:
        if geom.code_word_offset(proj) != tk.TCQ3_CODE_WORD_OFFSETS[proj]:
            raise ValueError(f"tcq3 {proj} code word offset {geom.code_word_offset(proj)} "
                             f"!= kernel constant {tk.TCQ3_CODE_WORD_OFFSETS[proj]}")
    return geom


# ---------------------------------------------------------------------------- manifest reader

@dataclass(frozen=True)
class TcqManifest:
    path: str
    root: str
    raw: dict
    geometry: TcqGeometry
    records: tuple                                     # tuple[dict] in (layer, expert) order

    @property
    def n_records(self) -> int:
        return len(self.records)


def read_tcq_manifest(path: str, *, require_uniform: bool = True) -> TcqManifest:
    """Read a tcq3 expert-manifest.json, accept ``quantization.mode == "tcq3"``, validate geometry once.

    Raises unless the quantization mode is tcq3 with K=3 / rin=1 and the record layout matches the F38 format.
    """
    with open(path) as f:
        raw = json.load(f)
    q = raw.get("quantization", {})
    if q.get("mode") != "tcq3":
        raise ValueError(f"not a tcq3 bank: quantization.mode = {q.get('mode')!r}")
    if q.get("K", q.get("bits")) not in (3,) and q.get("bits") != 3:
        raise ValueError(f"tcq3 bank must be K=3, got K={q.get('K')} bits={q.get('bits')}")
    if q.get("rin", 1) != 1:
        raise ValueError("tcq3 runtime assumes rin == 1 (not stored)")
    records = raw.get("records", [])
    if not records:
        raise ValueError("tcq3 manifest has no records")
    geom = validate_geometry(_geometry_from_manifest_record(records[0]))
    if require_uniform:
        for rec in records:
            if rec.get("logical_bytes") != geom.record_bytes:
                raise ValueError(f"non-uniform record size at L{rec.get('layer')}E{rec.get('expert')}")
    return TcqManifest(path=os.path.abspath(path), root=os.path.dirname(os.path.abspath(path)),
                       raw=raw, geometry=geom, records=tuple(records))


def global_index(layer: int, expert: int, geom: TcqGeometry | None = None) -> int:
    """Global record index into experts.bin / the resident routs = layer * n_experts + expert (F38 order)."""
    n_e = geom.n_experts_per_layer if geom is not None else N_EXPERTS_PER_LAYER
    return layer * n_e + expert


# ---------------------------------------------------------------------------- whole-record slicing

def slice_record(buf, geom: TcqGeometry) -> dict:
    """Slice ONE whole record's bytes (len == geom.record_bytes) into numpy arrays per segment.

    Returns {"gate_proj.code": int16[nI,nJ,48], "gate_proj.rout": f16[out], ...}.  Zero-copy views into ``buf``.
    """
    mv = memoryview(buf)
    if mv.nbytes != geom.record_bytes:
        raise ValueError(f"record buffer is {mv.nbytes} B, expected {geom.record_bytes}")
    out = {}
    for s in geom.segments:
        chunk = mv[s.byte_offset:s.byte_offset + s.byte_length]
        dt = np.int16 if s.kind == "code" else np.float16
        out[s.component] = np.frombuffer(chunk, dtype=dt).reshape(s.shape)
    return out


# ---------------------------------------------------------------------------- decode math (references + runtime)

def _mx():
    import mlx.core as mx
    return mx


def _t128(mx, x):
    """Normalized 128-block Walsh-Hadamard over the last axis (== the vendor mtplx.eschamoe.t128 / tcq_encode t128)."""
    lead = tuple(x.shape[:-1])
    IC = x.shape[-1]
    x = x.astype(mx.float32).reshape(*lead, IC // 128, 128)
    return mx.hadamard_transform(x, scale=128.0 ** -0.5).reshape(*lead, IC)


def decode_wq(code, *, fast: bool = False):
    """Vendor bit-exact decode of one expert's code -> W_q fp16 [IN, OUT].  code int16 [nI,nJ,48] (or [1,nI,nJ,48])."""
    from mtplx import eschamoe
    mx = _mx()
    code = code if isinstance(code, mx.array) else mx.array(np.ascontiguousarray(code))
    if code.ndim == 3:
        code = code.reshape(1, *code.shape)
    if fast:
        return eschamoe.decode_expert_weights_fast(code, K_BITS)[0]   # Metal kernel (GPU)
    return eschamoe.decode_expert_weights(code, K_BITS)[0]


def effective_weight_mx(W_q, rout):
    """E = B_in W_q B_out diag(rout)  (rin == 1), in the ORIGINAL weight space.  W_q [IN,OUT], rout [OUT] -> [IN,OUT].

    == tcq_encode.effective_weight(W_q, ones, rout).  Uses the vendor t128 (hadamard) so it matches the verify lane.
    """
    mx = _mx()
    W_q = W_q.astype(mx.float32)
    rout = rout.astype(mx.float32)
    E = _t128(mx, _t128(mx, W_q.T).T)                  # t128_axis1(t128_axis0(W_q))
    return E * rout[None, :]


def forward_decode_verify(x, W_q, rout):
    """The decode-verify forward: y = t128( t128(x) @ W_q ) * rout.  x [rows,IN], W_q [IN,OUT], rout [OUT] -> [rows,OUT].

    Equals x @ effective_weight_mx(W_q, rout) (B symmetric).  The runtime replaces ``t128(x) @ W_q`` with the
    stride-aware tile kernel on GPU; this pure-MLX version is the CPU reference / fallback for construction checks.
    """
    mx = _mx()
    xh = _t128(mx, x.astype(mx.float32))
    z = xh @ W_q.astype(mx.float32)
    return _t128(mx, z) * rout.astype(mx.float32)[None, :]


def decode_expert_to_bf16(code, rout, *, fast: bool = False):
    """PREFILL/seed path: decode one (expert, layer) tcq3 record to its dense bf16 effective weight, ONCE.

    Returns E bf16 [IN, OUT] = effective_weight_mx(decode(code), rout), fed to the existing grouped prefill matmul.
    Prefill streams experts, so each expert is decoded exactly once (the per-assignment tile kernel would re-read
    every expert per token and is wrong here)."""
    mx = _mx()
    W_q = decode_wq(code, fast=fast)
    return effective_weight_mx(W_q, rout if isinstance(rout, mx.array) else mx.array(np.ascontiguousarray(rout))).astype(mx.bfloat16)


# ---------------------------------------------------------------------------- resident routs (loaded once)

def load_resident_routs(manifest: TcqManifest, *, read_bytes=None):
    """Load the whole-bank routs resident (299 MB), indexed by global record index.

    Returns dict {"gate_proj": mx f16 [n_records, 2304], "up_proj": [...,2304], "down_proj": [...,5120]}.
    ``read_bytes(offset, length) -> bytes`` reads from experts.bin; default opens ``<root>/experts.bin``.  Never
    called per decode miss — the verify lane gathers rows out of these arrays.
    """
    mx = _mx()
    geom = manifest.geometry
    bin_path = os.path.join(manifest.root, "experts.bin")
    if read_bytes is None:
        fh = open(bin_path, "rb", buffering=0)

        def read_bytes(offset, length):
            fh.seek(offset)
            b = fh.read(length)
            if len(b) != length:
                raise RuntimeError("short rout read")
            return b
    routs = {}
    for proj in COMPONENTS:
        seg = geom.rout_segment(proj)
        out_f = seg.shape[0]
        arr = np.empty((manifest.n_records, out_f), dtype=np.float16)
        for rec in manifest.records:
            gi = global_index(rec["layer"], rec["expert"], geom)
            rec_base = record_base_offset(rec, geom)             # record's byte 0 in experts.bin
            b = read_bytes(rec_base + seg.byte_offset, seg.byte_length)
            arr[gi] = np.frombuffer(b, dtype=np.float16)
        routs[proj] = mx.array(arr)
    return routs


def record_base_offset(rec: dict, geom: TcqGeometry) -> int:
    """Absolute byte offset of a record in experts.bin: its gate_proj.code segment offset (the record's byte 0)."""
    seg0 = next(s for s in rec["segments"] if s["component"] == "gate_proj.code")
    return seg0["offset"]


# ---------------------------------------------------------------------------- decode-verify ops (GPU; mirrors PackedOps)

class TcqDecodeOps:
    """Per-layer decode-verify ops over the whole-record tcq3 bank, mirroring ``plane_lane.PackedOps``.

    Swaps the mxfp4 scale-codec kernels for the stride-aware trellis tile kernel, wraps each projection with
    ``t128`` before the matmul and ``t128 * rout`` after.  ``routs`` are the resident whole-bank arrays; a slot's
    global record index selects its rout row.  Constructed once per layer at the post-prefill boundary.
    """

    def __init__(self, routs, *, tables):
        self.mx = _mx()
        self.tables = tables
        self.routs = routs
        # gate/up: OUT=2304, IN=5120 ; down: OUT=5120, IN=2304.
        self.gu_kernel = {p: tk.make_tcq_projection_strided(2304, 5120, p) for p in ("gate_proj", "up_proj")}
        self.down_kernel = tk.make_tcq_projection_strided(5120, 2304, "down_proj")

    def _project(self, x, ids_slot, ids_global, code_bank, proj, out_dim):
        """x [rows, IN] (original space) -> y [rows, out_dim] = t128( t128(x) @ W_q ) * rout, all on GPU."""
        mx = self.mx
        xh = _t128(mx, x)
        kern = self.gu_kernel[proj] if proj in self.gu_kernel else self.down_kernel
        z = tk.run_tcq_projection_strided(kern, xh, ids_slot, code_bank, out_dim, self.tables)
        y = _t128(mx, z)
        rout = mx.take(self.routs[proj], ids_global, axis=0)      # [rows, out_dim]
        return y * rout
