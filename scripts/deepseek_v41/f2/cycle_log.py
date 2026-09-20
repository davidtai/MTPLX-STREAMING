"""F27 per-cycle draft log: host-only capture of the retained DSpark decode's confidence
calibration and accepted length, one JSON object per decode cycle.

F25 gates the drafted block on the native draft head's confidence.  Choosing that threshold
from a cost rule (verify a draft iff P(accept) >= verify-row cost / average token time) needs
the per-cycle confidences AND the committed length on EVERY prompt, which the receipts (per-depth
totals ``drafted_by_depth`` / ``accepted_by_depth`` only) do not carry.  This module records, per
DSpark decode cycle, {conf, k_cap, threshold, k_native, k_total, committed}, entirely on the host,
and writes one JSON next to the arm's receipt at process exit.

It wraps three functions already on the retained lane, by attribute replacement, BEFORE the hybrid
install rewrites ``_decode_cycles`` at generate time.  The rewrite runs inside
``dspark_with_boundary_observation`` at generate time and copies the decode module dict into its
exec namespace (``namespace = dict(module.__dict__)``), so a module-attribute wrap installed at the
staged packed_phase growth boundary is exactly what the rewritten cycle calls:

  * ``deepseek_v41_dspark_decode._effective_draft_len(conf_row, k, threshold)`` -- MODULE attr.
    Opens a cycle and records ``conf`` (the sigmoid of the native head's confidence row, k_cap
    floats, 6 decimals), ``k_cap`` (= k), ``threshold`` (None when off) and ``k_native`` (the
    returned value -- drafts kept by the confidence early stop).
  * ``lookup.LookupExtension.extend`` -- class method.  Records ``k_total`` = len(result), the
    drafts after the causal lookup extension (= verify rows - 1).
  * ``lookup.LookupExtension.append_committed`` -- class method.  The FIRST call per generate is the
    rewrite's ``append_committed([primary])`` and opens no cycle; every later call closes the current
    cycle with ``committed`` = len(tokens) (accepted drafts + 1).  A ``k_cap == 0`` cycle calls
    neither of the two above, so ``append_committed`` opens the record lazily and closes it.

Only DSpark generates reach these functions -- the AR-reference pass does not -- so only DSpark
cycles are logged.  A second DSpark generate in the same process (e.g. a timed second pass) is a new
``LookupExtension`` instance; a ``{"generate": n}`` marker (n = 2, 3, ...) is written before its
cycles rather than mixing them silently.

Diagnostic lane, off unless ``MTPLX_DSV41_F27_CYCLE_LOG`` is set; never a throughput candidate.  The sigmoid is
host numpy, but ``conf_row.astype(mx.float32)`` followed by ``np.asarray`` evaluates a (tiny) MLX cast kernel when
the row is not already float32, i.e. at most one small cast kernel + host sync per cycle (when the F25 threshold
is OFF; when it is on, the retained ``_effective_draft_len`` already does the same cast, so nothing extra).  That
is acceptable for a diagnostic arm only.  Everything is validated once at install; nothing here runs a per-cycle check.
"""
from __future__ import annotations

import atexit
import json
import os

import mlx.core as mx
import numpy as np

ENV = "MTPLX_DSV41_F27_CYCLE_LOG"

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
    """Assembles per-cycle records across the three wraps and writes one JSON at process exit.

    A cycle dict is opened lazily by whichever wrap fires first and closed by ``append_committed``.
    Keys land in schema order (conf, k_cap, threshold, k_native, k_total, committed); a k_cap == 0
    cycle carries only ``committed``.
    """

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


def _install_wraps(decode_module, lookup_module, log):
    """Replace the three targets with recording wrappers that call the originals unchanged."""
    original_draft_len = decode_module._effective_draft_len
    original_extend = lookup_module.LookupExtension.extend
    original_append = lookup_module.LookupExtension.append_committed

    def _effective_draft_len(conf_row, k, threshold):
        k_native = original_draft_len(conf_row, k, threshold)
        log.record_draft_len(conf_row, k, threshold, k_native)
        return k_native

    def extend(self, native_draft):
        result = original_extend(self, native_draft)
        log.record_extend(len(result))
        return result

    def append_committed(self, tokens):
        result = original_append(self, tokens)
        log.record_committed(self, tokens)
        return result

    for fn in (_effective_draft_len, extend, append_committed):
        fn._f27_cycle_log = True
    decode_module._effective_draft_len = _effective_draft_len
    lookup_module.LookupExtension.extend = extend
    lookup_module.LookupExtension.append_committed = append_committed


def install(decode_module, lookup_module, *, path):
    """Validate the sinks once, wrap the three functions, register the atexit writer.

    Refuses (RuntimeError) if the path does not end in .json or its parent is missing, if any of the
    three targets is absent, or if any target is already wrapped.  Construction-time only.
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
    targets = {
        "_effective_draft_len": getattr(decode_module, "_effective_draft_len", None),
        "extend": getattr(extension, "extend", None),
        "append_committed": getattr(extension, "append_committed", None),
    }
    for name, fn in targets.items():
        if fn is None:
            raise RuntimeError(f"F27 target missing: {name}")
        if getattr(fn, "_f27_cycle_log", False):
            raise RuntimeError(f"F27 target already wrapped: {name}")
    log = _CycleLog(path)
    _install_wraps(decode_module, lookup_module, log)
    atexit.register(log.write)
    global _LOG
    _LOG = log
    return {"installed": True, "path": path}


def install_from_env():
    """Read ``MTPLX_DSV41_F27_CYCLE_LOG`` at use.  Unset -> nothing installed; otherwise wrap the
    retained decode module + lookup class and register the exit writer."""
    path = os.environ.get(ENV, "").strip()
    if not path:
        return {"installed": False}
    from mtplx.models import deepseek_v41_dspark_decode as decode_module
    import lookup
    report = install(decode_module, lookup, path=path)
    print("F27_CYCLE_LOG_INSTALL " + json.dumps(report, sort_keys=True), flush=True)
    return report
