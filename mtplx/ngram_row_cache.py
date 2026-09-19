"""Generic bounded resident-row cache for on-disk affine row banks.

The abstraction: a byte-budgeted LRU of fixed-width quantized rows kept resident in
unified memory, keyed by global row id, backed by positional (``preadv``) reads from an
on-disk affine row bank -- eviction changes residency only, never row values.  Both the
Qwen3.8 Flash-Next resident n-gram table and the DeepSeek-V4.1 Engram bank sit on it.

This is the reusable core of the Qwen3.8 ``mtplx/qwen4_ngram.py`` streamed cache lifted
to a synchronous, framework-agnostic module (Python + numpy + mlx, no Metal arena, no
worker threads):

  * byte-budget -> slot count                lifted from ``plan_ngram_cache`` (qwen4_ngram.py:2053)
  * row-id-keyed routes + LRU touch/evict    lifted from ``_PackedCacheIndex`` (qwen4_ngram.py:2222-2416)
                                              and ``_choose_slots``/``oldest_unpinned_slots`` (:3391,:2318)
  * positional preadv read loop              lifted from ``_DescriptorReader.read_into`` (qwen4_ngram.py:2690)
  * contiguous-run miss coalescing           lifted from ``NGramRowCache._groups`` (qwen4_ngram.py:3410)
  * uint8 gather -> byte-range view -> mx.dequantize(affine)
                                              lifted from ``AffineQ4NGramRows.__call__`` (qwen4_ngram_mlx.py:46-90)

New here (vs the Qwen module): synchronous single-thread design (no ThreadPoolExecutor /
futures / leases / pins / generations), a generic :class:`RowGeometry` for any
``bits``/``group_size``, a plain numpy CPU arena, support for a gather request larger than
the cache (rows are copied out as their run is read, then may be evicted), and an
``MTPLX_ENGRAM_CACHE_LIMIT`` / ctor byte-budget path in place of the runtime memory planner.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np

import mlx.core as mx

__all__ = [
    "RowGeometry",
    "PositionalRowReader",
    "FileRowReader",
    "NGramRowCache",
    "cache_bytes_from_env",
]


# --------------------------------------------------------------------------
# row geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RowGeometry:
    """Packed layout of one quantized row, for the affine and mxfp8 modes.

    ``values_per_row`` quantized values at ``bits`` bits, grouped by ``group_size``.

    ``mode="affine"`` (default): weights ``U32`` | scales | biases, one ``param_bytes``-wide
    (bf16) scale and bias per group.  Matches ``mx.quantize(mode="affine")`` / ``mx.dequantize``
    packing: engram is ``(256, 8, 64)`` -> 272 bytes; the Qwen table is ``(row_width, 4, 32)``.

    ``mode="mxfp8"``: weights ``U32`` (the source F8_E4M3 code bytes, 4 per word) | scales, one
    ``e8m0`` (single uint8) scale per ``group_size==32`` group and **no bias**.  This is the OCP
    micro-scaling layout ``mx.dequantize(..., mode="mxfp8")`` consumes, and is an exact byte
    repack of the DeepSeek-V4.1 engram source (``embed.weight`` E4M3 + ``embed.scale`` E8M0):
    engram is ``(256, 8, 32)`` -> 256 code bytes + 8 scale bytes == 264 bytes.
    """

    values_per_row: int
    bits: int
    group_size: int
    mode: str = "affine"
    param_bytes: int = 2  # bf16 scale/bias (affine)

    def __post_init__(self) -> None:
        if self.mode not in ("affine", "mxfp8"):
            raise ValueError(f"unsupported mode {self.mode!r}")
        if self.values_per_row <= 0 or self.values_per_row % self.group_size:
            raise ValueError("values_per_row must be a positive multiple of group_size")
        if self.bits not in (2, 3, 4, 6, 8):
            raise ValueError(f"unsupported bits {self.bits}")
        if (self.values_per_row * self.bits) % 32:
            raise ValueError("packed weights must be a whole number of uint32 words")
        if self.mode == "mxfp8" and (self.bits != 8 or self.group_size != 32):
            raise ValueError("mxfp8 requires bits=8 and group_size=32")

    @property
    def has_bias(self) -> bool:
        return self.mode == "affine"

    @property
    def scale_param_bytes(self) -> int:
        """Bytes of one group scale: bf16 (2) for affine, e8m0 (1) for mxfp8."""
        return 1 if self.mode == "mxfp8" else self.param_bytes

    @property
    def weight_bytes(self) -> int:
        return self.values_per_row * self.bits // 8

    @property
    def n_groups(self) -> int:
        return self.values_per_row // self.group_size

    @property
    def scale_block_bytes(self) -> int:
        return self.n_groups * self.scale_param_bytes

    @property
    def bias_block_bytes(self) -> int:
        return self.n_groups * self.param_bytes if self.has_bias else 0

    @property
    def param_block_bytes(self) -> int:
        """Deprecated alias for the affine scale/bias block width (== bias_block_bytes)."""
        return self.n_groups * self.param_bytes

    @property
    def row_bytes(self) -> int:
        return self.weight_bytes + self.scale_block_bytes + self.bias_block_bytes

    def dequantize(self, packed_u8: mx.array) -> mx.array:
        """Dequantize ``[N, row_bytes]`` uint8 rows to ``[N, values_per_row]``.

        The byte-range view + ``mx.dequantize`` path lifted from
        ``AffineQ4NGramRows.__call__`` (qwen4_ngram_mlx.py:46-90); the mxfp8 branch feeds the
        E4M3 code words + E8M0 (uint8) scales straight into ``mx.dequantize(mode="mxfp8")``.
        """
        weight_end = self.weight_bytes
        scale_end = weight_end + self.scale_block_bytes
        weights = packed_u8[:, :weight_end].view(mx.uint32)
        if self.mode == "mxfp8":
            scales = packed_u8[:, weight_end:scale_end]        # e8m0 bytes, uint8
            return mx.dequantize(
                weights, scales, group_size=self.group_size, bits=self.bits, mode="mxfp8",
            )
        scales = packed_u8[:, weight_end:scale_end].view(mx.bfloat16)
        biases = packed_u8[:, scale_end:].view(mx.bfloat16)
        return mx.dequantize(
            weights, scales, biases,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
        )


# --------------------------------------------------------------------------
# positional reader
# --------------------------------------------------------------------------
class PositionalRowReader(Protocol):
    """Reads a contiguous run of packed rows by global row id.  ``read_run`` returns
    exactly ``count * row_bytes`` bytes for rows ``[start_row, start_row + count)``."""

    row_bytes: int
    num_rows: int

    def read_run(self, start_row: int, count: int) -> bytes: ...


class FileRowReader:
    """``os.preadv`` reader over one flat ``.bin`` of fixed-size records (record idx == row id).

    ``data_offset`` lets a bank sit at a byte offset inside a larger file (e.g. a safetensors
    tensor payload); for the engram ``.bin`` it is 0.  The preadv loop is lifted from
    ``_DescriptorReader.read_into`` (qwen4_ngram.py:2690) / ``EngramBank._read_record``.

    macOS defaults to ``F_NOCACHE`` so random misses in the bounded row LRU do not
    accumulate a second, unbounded copy in the OS file cache. Other platforms keep
    buffered reads by default. Explicit ``bypass_page_cache=False`` permits a
    buffered control; requested bypass must succeed before this reader is usable.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        row_bytes: int,
        num_rows: int,
        data_offset: int = 0,
        bypass_page_cache: bool | None = None,
    ) -> None:
        self.path = str(path)
        self.row_bytes = int(row_bytes)
        self.num_rows = int(num_rows)
        self.data_offset = int(data_offset)
        self.bypass_page_cache = (
            sys.platform == "darwin" if bypass_page_cache is None else bool(bypass_page_cache)
        )
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            size = os.fstat(self._fd).st_size
            need = self.data_offset + self.num_rows * self.row_bytes
            if size < need:
                raise ValueError(f"{self.path}: size {size} < required {need}")
            if self.bypass_page_cache:
                if sys.platform != "darwin":
                    raise RuntimeError(f"{self.path}: required F_NOCACHE is only supported on macOS")
                try:
                    import fcntl

                    fcntl.fcntl(self._fd, fcntl.F_NOCACHE, 1)
                except (ImportError, AttributeError, OSError) as exc:
                    raise RuntimeError(f"{self.path}: required F_NOCACHE could not be applied") from exc
            self.io_cache_mode = "f-nocache" if self.bypass_page_cache else "buffered"
        except BaseException:
            self.close()
            raise

    def read_run(self, start_row: int, count: int) -> bytes:
        if count <= 0:
            return b""
        if start_row < 0 or start_row + count > self.num_rows:
            raise IndexError(f"run [{start_row}, {start_row + count}) out of [0, {self.num_rows})")
        length = count * self.row_bytes
        buf = bytearray(length)
        view = memoryview(buf)
        base = self.data_offset + start_row * self.row_bytes
        got = 0
        while got < length:
            while True:
                try:
                    n = os.preadv(self._fd, [view[got:]], base + got)
                    break
                except InterruptedError:
                    continue
            if n <= 0:
                raise EOFError(f"short read at row {start_row}, byte {base + got}")
            got += n
        return bytes(buf)

    def close(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            self._fd = None
            os.close(fd)

    def __enter__(self) -> "FileRowReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------
def _contiguous_runs(sorted_rows: Sequence[int]) -> list[tuple[int, int]]:
    """Collapse a sorted, de-duplicated row list into ``(start, count)`` contiguous runs.

    The read-coalescing rule from ``NGramRowCache._groups`` (qwen4_ngram.py:3410): rows whose
    byte offsets abut become one positional read.
    """
    runs: list[tuple[int, int]] = []
    for r in sorted_rows:
        if runs and r == runs[-1][0] + runs[-1][1]:
            start, count = runs[-1]
            runs[-1] = (start, count + 1)
        else:
            runs.append((r, 1))
    return runs


class NGramRowCache:
    """Byte-budgeted LRU of resident packed rows over a :class:`PositionalRowReader`.

    ``cache_bytes`` (or ``cache_rows``) sets the ceiling; slot count is
    ``cache_bytes // row_bytes`` (>= 1), the same rule as ``plan_ngram_cache``
    (qwen4_ngram.py:2053).  A hit returns the resident bytes and refreshes LRU order; a miss
    reads the row positionally, evicting the oldest resident row only when no slot is free --
    residency changes, values never do.  A gather larger than the cache is served by copying
    each row out as its run is read, so requests are not bounded by the budget.
    """

    def __init__(
        self,
        reader: PositionalRowReader,
        geometry: RowGeometry,
        *,
        num_rows: int | None = None,
        cache_bytes: int | None = None,
        cache_rows: int | None = None,
    ) -> None:
        self.reader = reader
        # Construction metadata, kept separate from resettable numeric counters.
        self.io_cache_mode = getattr(reader, "io_cache_mode", "unknown")
        self.geometry = geometry
        self.row_bytes = geometry.row_bytes
        if int(getattr(reader, "row_bytes")) != self.row_bytes:
            raise ValueError("reader row_bytes does not match geometry")
        self.num_rows = int(num_rows if num_rows is not None else reader.num_rows)

        if cache_bytes is None:
            if cache_rows is None:
                cache_rows = 32768
            cache_bytes = int(cache_rows) * self.row_bytes
        self.budget_bytes = int(max(self.row_bytes, cache_bytes))
        self.slot_count = max(1, self.budget_bytes // self.row_bytes)

        # slot arena: one fixed row per slot; eviction rebinds a slot, never rewrites a live row
        self._arena = np.empty((self.slot_count, self.row_bytes), dtype=np.uint8)
        self._lru: "OrderedDict[int, int]" = OrderedDict()   # row_id -> slot (front = oldest)
        self._free: list[int] = list(range(self.slot_count))
        self.stats = {
            "hits": 0, "misses": 0, "evictions": 0,
            "reads": 0, "rows_read": 0, "gathers": 0,
        }

    # -- residency ----------------------------------------------------------
    def _alloc_slot(self) -> int:
        if self._free:
            return self._free.pop()
        _old_row, slot = self._lru.popitem(last=False)  # evict oldest
        self.stats["evictions"] += 1
        return slot

    def gather_bytes(self, row_ids: Iterable[int]) -> np.ndarray:
        """Return ``[R, row_bytes]`` uint8 for ``row_ids`` (order preserved), reading misses."""
        rows = [int(r) for r in row_ids]
        R = len(rows)
        out = np.empty((R, self.row_bytes), dtype=np.uint8)
        if R == 0:
            self.stats["gathers"] += 1
            return out

        positions: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            if r < 0 or r >= self.num_rows:
                raise IndexError(f"row {r} out of range [0, {self.num_rows})")
            positions.setdefault(r, []).append(i)

        misses: list[int] = []
        for r in positions:
            slot = self._lru.get(r)
            if slot is not None:
                self._lru.move_to_end(r)          # touch
                self.stats["hits"] += 1
                out[positions[r]] = self._arena[slot]
            else:
                misses.append(r)

        if misses:
            misses.sort()
            for start, count in _contiguous_runs(misses):
                off = 0
                while off < count:                 # sub-run bounded by slot count
                    n = min(count - off, self.slot_count)
                    data = self.reader.read_run(start + off, n)
                    self.stats["reads"] += 1
                    self.stats["rows_read"] += n
                    self.stats["misses"] += n
                    block = np.frombuffer(data, dtype=np.uint8).reshape(n, self.row_bytes)
                    for k in range(n):
                        r = start + off + k
                        slot = self._alloc_slot()
                        self._arena[slot] = block[k]
                        self._lru[r] = slot        # newest
                        out[positions[r]] = block[k]
                    off += n
        self.stats["gathers"] += 1
        return out

    def dequantize(self, row_ids) -> mx.array:
        """Gather ``row_ids`` and dequantize to ``mx.array`` of shape ``(*row_ids.shape, values)``."""
        arr = np.asarray(row_ids)
        shape = tuple(int(d) for d in arr.shape)
        flat = arr.reshape(-1)
        raw = self.gather_bytes(flat.tolist())
        packed = mx.array(np.ascontiguousarray(raw))
        out = self.geometry.dequantize(packed)
        return out.reshape((*shape, self.geometry.values_per_row))

    # -- introspection ------------------------------------------------------
    @property
    def resident_rows(self) -> int:
        return len(self._lru)

    @property
    def resident_bytes(self) -> int:
        return len(self._lru) * self.row_bytes

    def is_resident(self, row_id: int) -> bool:
        return int(row_id) in self._lru

    def reset(self) -> None:
        self._lru.clear()
        self._free = list(range(self.slot_count))
        for k in self.stats:
            self.stats[k] = 0

    def close(self) -> None:
        close = getattr(self.reader, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "NGramRowCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# byte-budget from the environment (ctor path; CLI left to the integration worker)
# --------------------------------------------------------------------------
def cache_bytes_from_env(
    default: int | str = "1GiB",
    *,
    env_var: str = "MTPLX_ENGRAM_CACHE_LIMIT",
) -> int:
    """Resolve a resident-row byte budget from ``env_var`` (Pydantic ``ByteSize``), else ``default``.

    Mirrors ``--ngram-cache-limit`` parsing ("1GiB", "1.5 GB", raw bytes...).  This is the
    isolated ctor/env path; wiring an actual ``--engram-cache-limit`` CLI flag is left to the
    model-worker integration.
    """
    from pydantic import ByteSize, TypeAdapter

    raw = os.environ.get(env_var)
    value = raw if raw is not None else default
    return int(TypeAdapter(ByteSize).validate_python(value))
