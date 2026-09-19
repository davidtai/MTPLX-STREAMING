"""F12: parallel packed-scale loader for the DSV4.1 growth transition.

Drop-in replacement for ``packed_storage.load_layer`` that produces byte-for-byte
identical owned Metal arrays and keeps EVERY safety property of the serial loader,
but overlaps the per-file ``preadv`` reads and SHA-256 verification across a
module-level thread pool while the caller's per-layer transition loop continues.

Invariants preserved exactly (see packed_storage.load_layer):

  * The Metal allocation (``mx.zeros`` + ``mx.eval``) and the ``memoryview(...)``
    cast stay on the CALLING thread, in the SAME projection/field order as the
    serial loader, so the MLX allocator sequence and peak are unchanged. Only the
    file open / read / hash / close / release runs on the pool. Pool threads
    allocate nothing in MLX and call no MLX API -- they only write host bytes into
    the already-owned buffer and hash it.
  * Same open flags (``O_RDONLY | O_NOFOLLOW``, ``F_NOCACHE=1``, F_RDAHEAD=45 -> 0),
    same size checks, same 8 MiB chunked ``preadv`` loop, same ``sha256`` compare,
    same error messages, same ``os.close`` + ``raw.release()`` in ``finally``.

``load_layer`` returns the arrays immediately (same dict structure) and records a
future per file in a module-level pending list. ``finish()`` joins every pending
future and re-raises the first error in submission order -- all-or-nothing: it
waits for every worker before raising, so no thread is left writing into a buffer,
and a failed / short / missing / symlinked / size- or digest-mismatched file still
fails the transition loudly. The caller must call ``finish()`` before anything can
consume the scale CONTENTS.

CPU-safe as a module: it imports no MLX; the caller passes ``mx`` for the on-thread
allocation only.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Tuple

# Mirror packed_storage's constants and per-file order so allocation stays in lockstep.
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
_FIELDS = ('descriptors', 'payload', 'bases')
_CHUNK = 8 * 1024 ** 2
_F_RDAHEAD = 45  # Darwin SDK sys/fcntl.h: F_RDAHEAD; Python omits the name.
_ENV_ENABLE = 'MTPLX_DSV41_F12_PARALLEL_SCALES'
_ENV_WORKERS = 'MTPLX_DSV41_F12_WORKERS'

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_pending: List[Future] = []


def _worker_count() -> int:
    raw = os.environ.get(_ENV_WORKERS, '8')
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise RuntimeError(f'{_ENV_WORKERS} must be a positive integer, got {raw!r}')
    if n < 1:
        raise RuntimeError(f'{_ENV_WORKERS} must be >= 1, got {n}')
    return n


def _get_executor() -> ThreadPoolExecutor:
    """Lazily create the shared pool once; worker count read here, one time."""
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=_worker_count(),
                                           thread_name_prefix='dsv41-f12-scale')
        return _executor


def _read_and_verify(path: Path, raw: memoryview, meta: dict) -> None:
    """Pool task: byte-identical to the serial loader's per-file body.

    Opens with the same flags, runs the same size checks, the same chunked
    ``preadv`` loop and the same ``sha256`` compare, then closes the fd and
    releases the caller-owned memoryview. Allocates no MLX and calls no MLX API;
    writes only into ``raw`` (host bytes backing the already-owned array)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            fcntl.fcntl(fd, _F_RDAHEAD, 0)
            if os.fstat(fd).st_size != meta['bytes'] or raw.nbytes != meta['bytes']:
                raise RuntimeError('packed scale size differs from inventory')
            cursor = 0
            while cursor < len(raw):
                end = min(cursor + _CHUNK, len(raw))
                count = os.preadv(fd, [raw[cursor:end]], cursor)
                if count <= 0:
                    raise RuntimeError('short packed scale read')
                cursor += count
            if hashlib.sha256(raw).hexdigest() != meta['sha256']:
                raise RuntimeError('packed scale artifact digest differs')
        finally:
            os.close(fd)
    finally:
        raw.release()


def load_layer(root, entry, *, mx):
    """Overlapped drop-in for packed_storage.load_layer.

    Allocates each owned array on the CALLING thread in the exact serial order,
    submits its (path, raw, meta) read+verify to the shared pool, and returns the
    same ``{projection: (descriptors, payload, bases)}`` dict immediately. Reads
    and hashing complete when ``finish()`` is called."""
    root = Path(root)
    executor = _get_executor()
    arrays = {}
    for projection in PROJECTIONS:
        component = entry['components'][projection]
        values = []
        for field in _FIELDS:
            meta = component[field]
            value = mx.zeros(tuple(meta['shape']), dtype=mx.uint32)
            mx.eval(value)
            raw = memoryview(value).cast('B')
            future = executor.submit(_read_and_verify, root / meta['file'], raw, meta)
            with _lock:
                _pending.append(future)
            values.append(value)
        arrays[projection] = tuple(values)
    return arrays


def finish() -> None:
    """Join every pending read/hash; re-raise the first error in submission order.

    All-or-nothing: even after the first failure we wait for every remaining
    future so no worker is left writing into a buffer, then raise. Draining clears
    the pending list so a later transition starts clean. A no-op when nothing is
    pending."""
    with _lock:
        pending = _pending[:]
        _pending.clear()
    error: BaseException | None = None
    for future in pending:
        try:
            future.result()
        except BaseException as exc:  # noqa: BLE001 -- surface any failure loudly
            if error is None:
                error = exc
    if error is not None:
        raise error


def _noop_finish() -> None:
    """Serial-arm finish(): the serial load_layer has already read every file."""
    return None


def resolve(serial_load_layer: Callable) -> Tuple[Callable, Callable]:
    """Route ONCE at transition construction, gated on ``MTPLX_DSV41_F12_PARALLEL_SCALES``.

    Enabled -> (parallel ``load_layer``, real ``finish``). Unset/``0`` -> the
    caller's ORIGINAL serial ``load_layer`` unchanged and a no-op finish, so one
    staged tree serves both the control and candidate arms. Returns prebound
    callables; the per-layer loop routes nothing."""
    if os.environ.get(_ENV_ENABLE, '0') != '0':
        return load_layer, finish
    return serial_load_layer, _noop_finish
