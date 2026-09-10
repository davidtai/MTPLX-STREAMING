"""Disk-backed Engram row bank reader (affine-8, one row == one fixed-size record).

The DeepSeek-V4.1-Flash Engram tables are converted by
``scripts/convert_deepseek_v41_engram.py`` into ``engram/engram-L{1,14}.bin`` +
``engram/engram-manifest.json``.  Each ``.bin`` is a flat array of fixed-size records;
record index == source table row index, so a lookup is a single positional read
``pread(fd, record_bytes, row * record_bytes)`` -- the bank-row==id principle the
dense-island store uses.

``EngramBank`` opens one layer's bank and gathers rows on demand with an LRU cache
bounded by a byte budget (the port plan calls for ~16k-64k resident rows).  Reads use
``os.preadv`` (the pure-Python fallback path of ``mtplx.expert_io``); no mmap.

Pure Python + numpy.  ``gather`` returns the packed affine components exactly as stored
(U32 weights, BF16 scales, BF16 biases) so the MLX runtime can feed them straight into
``mx.dequantize``; ``dequantize_rows`` is a numpy convenience/reference.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

__all__ = ["EngramBank"]


def _bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """Reinterpret bf16 bit patterns (uint16) as float32 (upper 16 bits of the f32)."""
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class EngramBank:
    """Positional-read reader over one engram layer's affine-8 bank."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        record_bytes: int,
        rows: int,
        head_dim: int,
        bits: int,
        group_size: int,
        layout: dict,
        layer_id: int | None = None,
        cache_rows: int = 32768,
        cache_bytes: int | None = None,
    ) -> None:
        self.path = str(path)
        self.record_bytes = int(record_bytes)
        self.rows = int(rows)
        self.head_dim = int(head_dim)
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.layer_id = layer_id

        # record sub-layout (offsets are within one fixed-size record)
        w, s, b = layout["weight"], layout["scales"], layout["biases"]
        self._w_off, self._w_len = int(w["offset"]), int(w["length"])
        self._s_off, self._s_len = int(s["offset"]), int(s["length"])
        self._b_off, self._b_len = int(b["offset"]), int(b["length"])
        self._n_u32 = self._w_len // 4          # packed uint32 per row
        self._n_groups = self._s_len // 2        # bf16 scales/biases per row
        # sanity: derived geometry matches the record width
        assert self._w_off == 0
        assert self._s_off == self._w_len
        assert self._b_off == self._w_len + self._s_len
        assert self._w_len + self._s_len + self._b_len == self.record_bytes
        assert self._n_u32 == self.head_dim * self.bits // 32
        assert self._n_groups == self.head_dim // self.group_size

        budget = cache_bytes if cache_bytes is not None else cache_rows * self.record_bytes
        self._cache_budget = int(max(self.record_bytes, budget))
        self._cache: "OrderedDict[int, bytes]" = OrderedDict()
        self._cache_used = 0

        self._fd = os.open(self.path, os.O_RDONLY)
        st = os.fstat(self._fd)
        expected = self.rows * self.record_bytes
        if st.st_size != expected:
            os.close(self._fd)
            raise ValueError(
                f"{self.path}: size {st.st_size} != rows*record_bytes {expected}"
            )

    # ---- construction from the manifest -----------------------------------
    @classmethod
    def open(cls, directory: str | os.PathLike, layer: int, **kwargs) -> "EngramBank":
        """Open ``<directory>/engram-L{layer}.bin`` using ``engram-manifest.json``."""
        directory = Path(directory)
        manifest = json.loads((directory / "engram-manifest.json").read_text())
        entry = next((e for e in manifest["layers"] if e["layer_id"] == layer), None)
        if entry is None:
            raise KeyError(f"layer {layer} not in manifest ({directory})")
        q = entry["quant"]
        return cls(
            directory / entry["file"],
            record_bytes=entry["record_bytes"],
            rows=entry["rows"],
            head_dim=q["head_dim"],
            bits=q["bits"],
            group_size=q["group_size"],
            layout=entry["record_layout"],
            layer_id=layer,
            **kwargs,
        )

    # ---- context management ----------------------------------------------
    def close(self) -> None:
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "EngramBank":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception:
            pass

    # ---- positional read + LRU -------------------------------------------
    def _read_record(self, row: int) -> bytes:
        """Exact positional read of one record (os.preadv fallback loop), LRU-cached."""
        cached = self._cache.get(row)
        if cached is not None:
            self._cache.move_to_end(row)
            return cached
        if row < 0 or row >= self.rows:
            raise IndexError(f"row {row} out of range [0, {self.rows})")
        buf = bytearray(self.record_bytes)
        view = memoryview(buf)
        base = row * self.record_bytes
        got = 0
        while got < self.record_bytes:
            while True:
                try:
                    n = os.preadv(self._fd, [view[got:]], base + got)
                    break
                except InterruptedError:
                    continue
            if n <= 0:
                raise EOFError(f"short read at row {row}, offset {base + got}")
            got += n
        rec = bytes(buf)
        self._cache[row] = rec
        self._cache_used += self.record_bytes
        while self._cache_used > self._cache_budget and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_used -= len(old)
        return rec

    def _split(self, rec: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = np.frombuffer(rec, dtype="<u4", count=self._n_u32, offset=self._w_off)
        s = np.frombuffer(rec, dtype="<u2", count=self._n_groups, offset=self._s_off)
        b = np.frombuffer(rec, dtype="<u2", count=self._n_groups, offset=self._b_off)
        return w, s, b

    # ---- public API -------------------------------------------------------
    def gather(self, rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gather rows -> (weights, scales, biases).

        ``weights``: uint32 ``[R, head_dim*bits/32]`` packed affine levels.
        ``scales``/``biases``: uint16 ``[R, head_dim/group]`` bf16 bit patterns.
        Feed straight into ``mx.dequantize`` (view uint16 as bfloat16) in the MLX runtime.
        """
        rows = list(rows)
        R = len(rows)
        weights = np.empty((R, self._n_u32), dtype="<u4")
        scales = np.empty((R, self._n_groups), dtype="<u2")
        biases = np.empty((R, self._n_groups), dtype="<u2")
        for i, r in enumerate(rows):
            w, s, b = self._split(self._read_record(int(r)))
            weights[i] = w
            scales[i] = s
            biases[i] = b
        return weights, scales, biases

    def dequantize_rows(self, rows) -> np.ndarray:
        """Numpy affine dequant of gathered rows -> float32 ``[R, head_dim]``.

        ``level * scale + bias`` per group.  Reference path; matches ``mx.dequantize`` up
        to bf16 rounding (mlx does the arithmetic in bf16).
        """
        weights, scales, biases = self.gather(rows)
        R = weights.shape[0]
        # 8-bit affine: each uint32 holds 4 uint8 levels (LE); level index == column index.
        levels = weights.view(np.uint8).reshape(R, -1)[:, : self.head_dim].astype(np.float32)
        s = _bf16_bits_to_f32(scales).reshape(R, -1)
        b = _bf16_bits_to_f32(biases).reshape(R, -1)
        grp = np.arange(self.head_dim) // self.group_size
        return levels * s[:, grp] + b[:, grp]

    @property
    def cache_used_bytes(self) -> int:
        return self._cache_used

    def __len__(self) -> int:
        return self.rows

    def __repr__(self) -> str:
        return (
            f"EngramBank(layer={self.layer_id}, rows={self.rows}, "
            f"record_bytes={self.record_bytes}, path={self.path!r})"
        )
