"""F27 per-cycle draft log: host-side capture of the retained DSpark decode's confidence
calibration and accepted length, one JSON object per decode cycle.

F25 gates the drafted block on the native draft head's confidence.  Choosing that threshold
from a cost rule (verify a draft iff P(accept) >= verify-row cost / average token time) needs
the per-cycle confidences AND the committed length on EVERY prompt, which the receipts (per-depth
totals ``drafted_by_depth`` / ``accepted_by_depth`` only) do not carry.  This module records, per
DSpark decode cycle, {conf, k_cap, threshold, k_native, k_total, committed}, on the host, and writes
one JSON next to the arm's receipt at process exit.

Seam order (this is the corrected mechanism -- an earlier version had it backwards).  The retained
hybrid install rewrites ``_decode_cycles``, ``exec``s the copy into ``namespace = dict(module.__dict__)``
and reassigns ``module._decode_cycles``.  That install runs at the START of ``dspark_generate`` (from
``run_full.py``'s ``dspark_with_boundary_observation``).  This module's ``install_from_env`` is staged in
packed_phase's growth transition, which runs from the prefill callback DURING ``dspark_generate`` -- AFTER
the hybrid rewrite, and BEFORE ``_decode_cycles`` is called (a module-global lookup at call time).  So by
install time ``_decode_cycles`` is usually the exec'd hybrid copy (``co_filename == '<hybrid_lookup_decode>'``),
and its bare ``_effective_draft_len`` name resolves against its OWN ``__globals__`` (that namespace copy), NOT
the module attribute.  Therefore:

  * ``_effective_draft_len`` -- when the hybrid copy is live, wrap the name inside that function's
    ``__globals__`` (unwrapping an F28 ``__wrapped__`` chain first, so ``+prof+cl`` works in either install
    order).  Otherwise (hybrid not yet live -- tests, or a hypothetical import-time seam) wrap the module
    attribute.  Reported as ``conf_seam``.  Records ``conf`` (sigmoid of the k_cap-float confidence row, 6
    decimals), ``k_cap`` (= k), ``threshold`` (None when off), ``k_native`` (returned value).
  * ``lookup.LookupExtension.extend`` / ``append_committed`` -- wrapped at class level either way; instance
    method resolution finds them regardless of the namespace copy.  ``extend`` records ``k_total`` = len(result);
    the FIRST ``append_committed`` per generate is the rewrite's ``append_committed([primary])`` (not a cycle),
    every later one closes the current cycle with ``committed`` = len(tokens).  A ``k_cap == 0`` cycle calls
    neither of the first two, so ``append_committed`` opens the record lazily and closes it.

Only DSpark generates reach these functions.  A second DSpark generate (new ``LookupExtension`` instance) gets
a ``{"generate": n}`` marker (n = 2, 3, ...) before its cycles.  atexit writes ``{"schema": 1, "cycles": [...]}``
once to the path.

Host-side, diagnostic lane, off unless ``MTPLX_DSV41_F27_CYCLE_LOG`` is set; never a throughput candidate.  The
sigmoid is numpy, but ``conf_row.astype(mx.float32)`` + ``np.asarray`` evaluates a (tiny) MLX cast kernel when the
row is not already float32 -- at most one small cast kernel + host sync per cycle (nothing extra when the F25
threshold is on; it already does the same cast).  Everything is validated once at install; no per-cycle check.
"""
from __future__ import annotations

import atexit
import json
import os

import mlx.core as mx
import numpy as np

ENV = "MTPLX_DSV41_F27_CYCLE_LOG"
_HYBRID_DECODE_FILENAME = "<hybrid_lookup_decode>"   # hybrid_install compile()s the rewrite under this name

_LOG = None   # the installed _CycleLog (one per process); None until install() runs


def _conf_row_to_list(conf_row, k):
    """The native head confidences for the k_cap drafts offered this cycle: sigmoid of the
    (float32) confidence row, first k values, 6 decimals.  ``astype(mx.float32)`` + ``np.asarray`` evaluates a
    tiny MLX cast kernel when the row is not already float32; the sigmoid is host numpy, and the float64 cast keeps
    ``exp`` off the float32 overflow path for saturated logits."""
    logits = np.asarray(conf_row.astype(mx.float32)).reshape(-1)[: int(k)].astype(np.float64)
    conf = 1.0 / (1.0 + np.exp(-logits))
    return [round(float(v), 6) for v in conf]


class _CycleLog:
    """Assembles per-cycle records across the three wraps and writes one JSON at process exit."""

    def __init__(self, path):
        self.path = path
        self.cycles = []
        self._current = None    # the cycle dict being assembled, or None between cycles
        self._active = None     # the LookupExtension instance of the current DSpark generate
        self._generate = 0      # 1-based count of DSpark generates observed

    def _open(self):
        if self._current is None:
            self._current = {}
        return self._current

    def record_draft_len(self, conf_row, k, threshold, k_native):
        cyc = self._open()
        cyc["conf"] = _conf_row_to_list(conf_row, k)
        cyc["k_cap"] = int(k)
        cyc["threshold"] = None if threshold is None else float(threshold)
        cyc["k_native"] = int(k_native)

    def record_extend(self, k_total):
        self._open()["k_total"] = int(k_total)

    def record_committed(self, instance, tokens):
        if instance is not self._active:
            # A new DSpark generate: this is its leading append_committed([primary]) -- not a cycle.
            self._active = instance
            self._current = None
            self._generate += 1
            if self._generate >= 2:
                self.cycles.append({"generate": self._generate})
            return
        cyc = self._open()
        cyc["committed"] = len(tokens)
        self.cycles.append(cyc)
        self._current = None

    def write(self, path=None):
        target = self.path if path is None else path
        with open(target, "w") as fh:
            json.dump({"schema": 1, "cycles": self.cycles}, fh)


def _unwrap(fn):
    """Follow an F28 ``__wrapped__`` chain to the underlying function (cycle-safe)."""
    seen = set()
    while fn is not None and hasattr(fn, "__wrapped__") and id(fn) not in seen:
        seen.add(id(fn))
        fn = fn.__wrapped__
    return fn


def _resolve_effective_draft_len(decode_module):
    """Pick the sink for ``_effective_draft_len``.  When the live ``_decode_cycles`` is the exec'd hybrid copy,
    that is the function's own globals dict (its bare-name lookup resolves there); otherwise the module attribute.
    Returns (seam, getter, setter)."""
    inner = _unwrap(getattr(decode_module, "_decode_cycles", None))
    code = getattr(inner, "__code__", None)
    if (code is not None and code.co_filename == _HYBRID_DECODE_FILENAME
            and isinstance(getattr(inner, "__globals__", None), dict)
            and "_effective_draft_len" in inner.__globals__):
        g = inner.__globals__
        return ("function_globals",
                lambda: g.get("_effective_draft_len"),
                lambda v: g.__setitem__("_effective_draft_len", v))
    return ("module_attribute",
            lambda: getattr(decode_module, "_effective_draft_len", None),
            lambda v: setattr(decode_module, "_effective_draft_len", v))


def _wrap_effective_draft_len(original, log):
    def _effective_draft_len(conf_row, k, threshold):
        k_native = original(conf_row, k, threshold)
        log.record_draft_len(conf_row, k, threshold, k_native)
        return k_native

    _effective_draft_len._f27_cycle_log = True
    return _effective_draft_len


def _wrap_extend(original, log):
    def extend(self, native_draft):
        result = original(self, native_draft)
        log.record_extend(len(result))
        return result

    extend._f27_cycle_log = True
    return extend


def _wrap_append_committed(original, log):
    def append_committed(self, tokens):
        result = original(self, tokens)
        log.record_committed(self, tokens)
        return result

    append_committed._f27_cycle_log = True
    return append_committed


def install(decode_module, lookup_module, *, path):
    """Validate the sinks once, wrap the three functions, register the atexit writer.

    Refuses (RuntimeError) if the path does not end in .json or its parent is missing, if any of the three
    targets is absent, or if any target is already wrapped.  Construction-time only.  The ``_effective_draft_len``
    sink is the live hybrid function's globals when that copy is live, else the module attribute (``conf_seam``).
    """
    path = str(path)
    if not path.endswith(".json"):
        raise RuntimeError(f"F27 cycle log path must end in .json; got {path!r}")
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise RuntimeError(f"F27 cycle log parent directory does not exist: {parent}")
    extension = getattr(lookup_module, "LookupExtension", None)
    if extension is None:
        raise RuntimeError("F27 needs lookup.LookupExtension")
    conf_seam, get_edl, set_edl = _resolve_effective_draft_len(decode_module)
    edl_fn = get_edl()
    ext_fn = getattr(extension, "extend", None)
    apc_fn = getattr(extension, "append_committed", None)
    for name, fn in (("_effective_draft_len", edl_fn), ("extend", ext_fn), ("append_committed", apc_fn)):
        if fn is None:
            raise RuntimeError(f"F27 target missing: {name}")
        if getattr(fn, "_f27_cycle_log", False):
            raise RuntimeError(f"F27 target already wrapped: {name}")
    log = _CycleLog(path)
    set_edl(_wrap_effective_draft_len(edl_fn, log))
    lookup_module.LookupExtension.extend = _wrap_extend(ext_fn, log)
    lookup_module.LookupExtension.append_committed = _wrap_append_committed(apc_fn, log)
    atexit.register(log.write)
    global _LOG
    _LOG = log
    return {"installed": True, "path": path, "conf_seam": conf_seam}


def install_from_env():
    """Read ``MTPLX_DSV41_F27_CYCLE_LOG`` at use.  Unset -> nothing installed; otherwise wrap the retained
    decode module + lookup class and register the exit writer."""
    path = os.environ.get(ENV, "").strip()
    if not path:
        return {"installed": False}
    from mtplx.models import deepseek_v41_dspark_decode as decode_module
    import lookup
    report = install(decode_module, lookup, path=path)
    print("F27_CYCLE_LOG_INSTALL " + json.dumps(report, sort_keys=True), flush=True)
    return report
