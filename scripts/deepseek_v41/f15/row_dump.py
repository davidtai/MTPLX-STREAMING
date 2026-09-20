"""F15: dump verify-logits rows at requested global positions (monkeypatch install).

Why this exists
---------------
The retained runner can only classify a DSpark divergence against its ONE cached AR
diagnostic row (index 297); a new index raises inside the cache-only override ("no
replay through candidate arithmetic", run_full.py). Three rounding-class levers first
leave the control stream at global position ~480, which needs a tie/no-tie verdict.

This module captures, with zero extra forwards, the exact verify-logits row that
produced each arm's committed token at any requested global position, so an offline
CPU tool (``classify_pair.py``) can feed CONTROL-vs-CANDIDATE rows straight into the
repo's own ``classify_divergence`` rule.

Mechanism
---------
``install_from_env()`` is a NO-OP unless ``MTPLX_DSV41_F15_ROW_DUMP_DIR`` is set. When
set, it wraps ``DivergenceCapture.observe`` at the CLASS level (once per process). Every
wrapped call:

  1. runs the ORIGINAL ``observe`` unchanged (so the runner's own first-AR-divergence
     capture is byte-for-byte what it always was); then
  2. INDEPENDENTLY extracts the verify-logits row of every committed token whose global
     position ``gpos = base_len + m + 1`` is in the requested set and saves it to
     ``<dir>/row-<gpos>.npy``, exactly the way ``observe`` reads it
     (``np.asarray(source.astype(mx.float32)).reshape(-1)`` from ``verify_logits_parts``
     -- ``part[0, m-start]`` -- or ``verify_logits[0, m]``).

Step 2 does its OWN extraction and does NOT rely on the original's early return: the
control/candidate AR divergence at 297 happens before 480, so ``observe`` has already
``found`` and early-returns by the time position 480's cycle runs -- this wrapper still
dumps it.

Hot-path cost: when no requested position falls inside a cycle, the wrapper does only a
cheap integer range test (``cyc_hi < lo or cyc_lo > hi``) and returns. It touches the
disk only for the handful of requested positions. A ``<dir>/rows.json`` index
``{gpos: {"token", "cycle_base_len", "m"}}`` is written at exit (atexit); the ``.npy``
rows themselves are written the moment each position is committed (crash-resilient).

The requested set is ``MTPLX_DSV41_F15_ROW_INDICES`` -- a comma list and/or inclusive
ranges (e.g. ``"297,470-490,500"``).

Runner wiring (a GPU window, owned by Fable): add ``scripts/deepseek_v41/f15`` to
PYTHONPATH so ``import row_dump`` resolves; the staged runner (see
``stage_f15_runner.py``) calls ``install_from_env()`` at the post-prefill boundary.

NB: importing this pulls in ``mlx.core`` transitively (the decode module imports it at
module level). A CPU-only caller sets the default device before importing this module;
nothing here touches Metal.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import mlx.core as mx

from mtplx.models import deepseek_v41_dspark_decode as _dec

__all__ = [
    "install_from_env",
    "install",
    "parse_indices",
    "state",
    "RowDumpState",
    "ENV_DIR",
    "ENV_INDICES",
]

ENV_DIR = "MTPLX_DSV41_F15_ROW_DUMP_DIR"
ENV_INDICES = "MTPLX_DSV41_F15_ROW_INDICES"

# The class-level wrap is installed at most once per process; a second
# install_from_env() reuses the first state (one dir/index-set per process).
_INSTALL_LOCK = threading.Lock()
_STATE: "Optional[RowDumpState]" = None
_ORIG_OBSERVE = None  # the unwrapped DivergenceCapture.observe function
_ATEXIT_REGISTERED = False


def parse_indices(spec: Optional[str]) -> "set[int]":
    """Parse a comma list and/or inclusive ranges into a set of non-negative ints.

    ``"297,470-490, 500"`` -> ``{297, 470..490, 500}``. Whitespace tolerant; ``;`` is
    accepted as a separator too. An empty / all-whitespace spec is an empty set. A
    malformed token (non-digit, or a range with ``hi < lo``) raises ``ValueError`` --
    a silent skip would arm the dump on the wrong positions.
    """
    out: "set[int]" = set()
    if spec is None:
        return out
    for raw in str(spec).replace(";", ",").split(","):
        tok = raw.strip()
        if not tok:
            continue
        if "-" in tok:
            lo_s, _, hi_s = tok.partition("-")
            lo_s, hi_s = lo_s.strip(), hi_s.strip()
            if not (lo_s.isdigit() and hi_s.isdigit()):
                raise ValueError(f"{ENV_INDICES}: bad range token {raw!r}")
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"{ENV_INDICES}: range {raw!r} has hi < lo")
            out.update(range(lo, hi + 1))
        else:
            if not tok.isdigit():
                raise ValueError(f"{ENV_INDICES}: bad index token {raw!r}")
            out.add(int(tok))
    return out


class RowDumpState:
    """Where dumped rows go and which global positions to dump.

    ``indices`` is the requested global-position set; ``lo``/``hi`` are its bounds, used
    for the per-cycle cheap range test. ``index`` accumulates the ``rows.json`` payload.
    """

    def __init__(self, out_dir: str, indices: Iterable[int]):
        self.out_dir = str(out_dir)
        self.indices: "set[int]" = {int(i) for i in indices}
        self.lo: Optional[int] = min(self.indices) if self.indices else None
        self.hi: Optional[int] = max(self.indices) if self.indices else None
        self.index: "Dict[int, Dict[str, int]]" = {}
        self._lock = threading.Lock()
        os.makedirs(self.out_dir, exist_ok=True)

    def maybe_dump(
        self,
        *,
        base_len: int,
        committed: Sequence[int],
        verify_logits=None,
        verify_logits_parts: "Optional[Sequence[tuple]]" = None,
    ) -> None:
        """Dump the verify row of any committed token whose gpos is requested.

        Cheap integer range test first: a cycle whose committed positions cannot contain
        a requested gpos returns immediately, doing no per-token work.
        """
        if self.lo is None:  # nothing requested
            return
        if verify_logits is None and verify_logits_parts is None:
            return
        n = len(committed)
        if n == 0:
            return
        cyc_lo = base_len + 1                 # gpos of the first committed token
        cyc_hi = base_len + n                 # gpos of the last committed token
        if cyc_hi < self.lo or cyc_lo > self.hi:
            return                            # whole cycle outside the requested band
        for m, tok in enumerate(committed):
            gpos = base_len + m + 1
            if gpos not in self.indices or gpos in self.index:
                continue
            # Extract EXACTLY as DivergenceCapture.observe does.
            if verify_logits_parts is not None:
                source = None
                for start, part in verify_logits_parts:
                    width = int(part.shape[1])
                    if int(start) <= m < int(start) + width:
                        source = part[0, m - int(start)]
                        break
                if source is None:
                    raise IndexError(f"missing staged verify logits row {m}")
            else:
                source = verify_logits[0, m]
            row = np.asarray(source.astype(mx.float32)).reshape(-1)
            np.save(os.path.join(self.out_dir, f"row-{gpos}.npy"), row)
            with self._lock:
                self.index[gpos] = {
                    "token": int(tok),
                    "cycle_base_len": int(base_len),
                    "m": int(m),
                }

    def write_index(self) -> str:
        """Write ``<dir>/rows.json`` atomically. Keys are stringified gpos (JSON)."""
        path = os.path.join(self.out_dir, "rows.json")
        with self._lock:
            payload = {str(g): v for g, v in sorted(self.index.items())}
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return path


def state() -> "Optional[RowDumpState]":
    """The installed state, or ``None`` when the dump is not armed."""
    return _STATE


def _wrapped_observe(self, *, base_len, committed, verify_logits=None, verify_logits_parts=None):
    """Class-level replacement for ``DivergenceCapture.observe``.

    Runs the original with identical arguments and return, then dumps requested rows
    independently of the original's ``found`` early return.
    """
    result = _ORIG_OBSERVE(
        self,
        base_len=base_len,
        committed=committed,
        verify_logits=verify_logits,
        verify_logits_parts=verify_logits_parts,
    )
    st = _STATE
    if st is not None:
        st.maybe_dump(
            base_len=base_len,
            committed=committed,
            verify_logits=verify_logits,
            verify_logits_parts=verify_logits_parts,
        )
    return result


def _atexit_write() -> None:
    st = _STATE
    if st is None:
        return
    try:
        st.write_index()
    except Exception as exc:  # never let an exit handler crash the process
        print(f"F15_ROW_DUMP atexit rows.json write failed: {exc!r}", flush=True)


def install(out_dir: str, indices: Iterable[int]) -> "RowDumpState":
    """Bind the row-dump wrap onto ``DivergenceCapture.observe`` (class level, once).

    One state per process: a second call while already installed returns the existing
    state unchanged (the runner boundary can fire more than once per process).
    """
    global _STATE, _ORIG_OBSERVE, _ATEXIT_REGISTERED
    with _INSTALL_LOCK:
        if _STATE is not None:
            return _STATE
        if _ORIG_OBSERVE is None:
            _ORIG_OBSERVE = _dec.DivergenceCapture.observe
            _dec.DivergenceCapture.observe = _wrapped_observe
        if not _ATEXIT_REGISTERED:
            atexit.register(_atexit_write)
            _ATEXIT_REGISTERED = True
        _STATE = RowDumpState(out_dir, indices)
        return _STATE


def install_from_env() -> "Optional[RowDumpState]":
    """Install iff ``MTPLX_DSV41_F15_ROW_DUMP_DIR`` is set; otherwise a NO-OP.

    ``MTPLX_DSV41_F15_ROW_INDICES`` (comma list and/or ranges) selects the global
    positions to dump. Prints a one-line install receipt.
    """
    out_dir = os.environ.get(ENV_DIR)
    if not out_dir:
        return None
    indices = parse_indices(os.environ.get(ENV_INDICES, ""))
    st = install(out_dir, indices)
    print(
        "F15_ROW_DUMP_INSTALL "
        + json.dumps(
            {
                "dir": st.out_dir,
                "n_indices": len(st.indices),
                "indices_preview": sorted(st.indices)[:16],
            }
        ),
        flush=True,
    )
    return st


def _reset_for_tests() -> None:
    """Restore the original method and clear module state. Tests only."""
    global _STATE, _ORIG_OBSERVE
    with _INSTALL_LOCK:
        if _ORIG_OBSERVE is not None:
            _dec.DivergenceCapture.observe = _ORIG_OBSERVE
            _ORIG_OBSERVE = None
        _STATE = None
