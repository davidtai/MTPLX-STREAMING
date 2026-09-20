"""ctypes bridge to the C beam step (:file:`tcq_beam.c`) — DSV4.1 F37, CPU only.

Compiles ``tcq_beam.c`` with ``clang -O3`` into a shared library (once, cached by a hash of the
source; guarded by an flock so the 4 encode-pool workers don't race), then exposes
:func:`beam_encode_c`, a drop-in for :func:`tcq_encode.beam_encode_fast` that runs the beam loop in C
and applies the IDENTICAL numpy seam-repair.  On real weight data it produces byte-identical codes to
the numpy beam (distinct float32 costs -> identical surviving set; see :file:`tcq_beam.c` for the tie
rule).  No new Python dependencies (ctypes + the system clang).
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import subprocess

import numpy as np

import tcq_encode as enc

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "tcq_beam.c")
_LIB = None


def _build_lib() -> str:
    src = open(_SRC, "rb").read()
    tag = hashlib.sha1(src).hexdigest()[:16]
    so = os.path.join(_HERE, f"_tcq_beam_{tag}.so")
    lock = os.path.join(_HERE, "_tcq_beam.build.lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)          # only one process compiles; others wait then load
        if not os.path.exists(so):
            tmp = so + f".{os.getpid()}.tmp"
            base = ["clang", "-O3", "-shared", "-fPIC", "-o", tmp, _SRC]
            try:
                subprocess.run(base[:1] + ["-march=native"] + base[1:], check=True,
                               capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                subprocess.run(base, check=True, capture_output=True)   # no -march=native fallback
            os.replace(tmp, so)
    return so


def get_lib():
    """Load (compiling if needed) the C beam library and set argtypes."""
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(_build_lib())
        F = ctypes.POINTER(ctypes.c_float)
        lib.tcq_beam_batch.restype = ctypes.c_int
        lib.tcq_beam_batch.argtypes = [F, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                                       F, F, F, F, ctypes.POINTER(ctypes.c_ubyte)]
        _LIB = lib
    return _LIB


def _fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def beam_encode_c(targets: np.ndarray, dec: np.ndarray, beam: int = 256,
                  batch: int = 512, repair_span: int = 8, on_batch=None) -> tuple:
    """C beam loop + numpy seam-repair.  Same signature/return as :func:`tcq_encode.beam_encode_fast`.

    targets [N,256] (cycle order) -> (new3 [N,256] uint8, s0 [N]).  ``on_batch`` (GPU-window gate) is
    called before each tile batch; the C call itself is uninterruptible, so ``batch`` sets the pause
    granularity and bounds the repair working set.
    """
    lib = get_lib()
    N, S = targets.shape
    assert S == 256
    ct = enc.cycle_tables(3)
    DEC = np.ascontiguousarray(dec, dtype=np.float32)
    DEC2 = np.ascontiguousarray((DEC.astype(np.float64) ** 2).astype(np.float32))
    DECg2 = np.ascontiguousarray(DEC.reshape(8, 8192).T)       # [8192,8] DEC[s + r*8192]
    DEC2g2 = np.ascontiguousarray(DEC2.reshape(8, 8192).T)
    pDEC, pDEC2, pDECg2, pDEC2g2 = _fptr(DEC), _fptr(DEC2), _fptr(DECg2), _fptr(DEC2g2)
    out = np.empty((N, 256), np.uint8)
    sstar = np.empty(N, np.int64)
    for b0 in range(0, N, batch):
        if on_batch is not None:
            on_batch()
        tb = np.ascontiguousarray(targets[b0:b0 + batch], dtype=np.float32)
        B = tb.shape[0]
        nb = np.empty((B, 256), np.uint8)
        rc = lib.tcq_beam_batch(_fptr(tb), ctypes.c_long(B), 256, beam,
                                pDEC, pDEC2, pDECg2, pDEC2g2,
                                nb.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        if rc != 0:
            raise RuntimeError(f"tcq_beam_batch returned {rc}")
        nb = enc.repair_seam(nb, tb, dec, ct, span=repair_span)      # numpy, bit-exact
        out[b0:b0 + B] = nb
        sstar[b0:b0 + B] = enc.decoded_windows(nb, ct)[:, 0] & 0x1FFF
    return out, sstar
