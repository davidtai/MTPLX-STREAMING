"""Stage patched copies of the retained runner for the F2b prefetch window.

The window runs the EXACT retained runner from a fresh /private/tmp staging dir (never
Codex's pinned original). run_full imports helpers by module name (PYTHONPATH order
decides) while self-checking the pinned originals, so a staged copy runs while the hash
self-checks still pass. Every edit is anchored to a UNIQUE line and round-trip-checked
(the f5_compile/stage_f5_runner.py discipline). CPU-safe, no MLX.

Edits (all to STAGED copies):
  * ``stage_admission`` -- cap the decode capacity-search start (equal-capacity ladder,
    like F5). No ring charge: the F2b ring is HOST memory, not MLX, and is not admitted.
  * ``stage_run_full`` -- (1) install F2b as the LAST step of ``observe_seed_prefill``
    (after ``prime_model``, run_full.py:737), no-op unless MTPLX_DSV41_F2B=1; (2) surface
    the hidden transition/install error in ``observe_prefill_boundary``'s except clause,
    which otherwise raises a bare SystemExit with no traceback (run_full.py:783).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_CAP_ANCHOR = "    for capacity in range(112, old_capacity, -1):"

# Install F2b after prime_model, inside observe_seed_prefill (8-space indent).
_INSTALL_ANCHOR = "        projection_owner_report.update(prime_model(target))"
_INSTALL_INSERT = (
    "        import f2.install as _f2b_install  # F2b\n"
    "        _f2b_install.install_from_env(target)  # F2b (no-op unless MTPLX_DSV41_F2B=1)"
)

# Surface the hidden transition/install error (16-space indent) before the bare SystemExit.
_TRACE_ANCHOR = (
    "                raise SystemExit('critical prefill/cache transition failed; "
    "aborting generation') from error"
)
_TRACE_INSERT = "                import traceback; traceback.print_exc()  # F2b: surface the hidden error"


def _assert_once(text: str, anchor: str, label: str) -> None:
    n = text.count(anchor)
    if n != 1:
        raise RuntimeError(f"{label} anchor is not unique ({n} occurrences)")


def stage_admission(source_text: str, *, max_rows: int) -> str:
    if not 85 <= int(max_rows) <= 112:
        raise RuntimeError("max_rows must be within 85..112")
    _assert_once(source_text, _CAP_ANCHOR, "capacity-search")
    new_cap = f"    for capacity in range({int(max_rows)}, old_capacity, -1):  # F2b equal-capacity cap"
    updated = source_text.replace(_CAP_ANCHOR, new_cap)
    if updated.replace(new_cap, _CAP_ANCHOR) != source_text:
        raise RuntimeError("F2b capacity cap changed more than the search start")
    return updated


def stage_run_full(source_text: str) -> str:
    _assert_once(source_text, _INSTALL_ANCHOR, "observe_seed_prefill prime_model")
    _assert_once(source_text, _TRACE_ANCHOR, "observe_prefill_boundary SystemExit")
    updated = source_text.replace(_INSTALL_ANCHOR, _INSTALL_ANCHOR + "\n" + _INSTALL_INSERT)
    if updated.replace(_INSTALL_ANCHOR + "\n" + _INSTALL_INSERT, _INSTALL_ANCHOR) != source_text:
        raise RuntimeError("F2b install edit changed more than the one insertion")
    step = updated
    updated = updated.replace(_TRACE_ANCHOR, _TRACE_INSERT + "\n" + _TRACE_ANCHOR)
    if updated.replace(_TRACE_INSERT + "\n" + _TRACE_ANCHOR, _TRACE_ANCHOR) != step:
        raise RuntimeError("F2b traceback edit changed more than the one insertion")
    return updated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the F2b-patched runner in place")
    ap.add_argument("--admission", default=None, help="STAGED packed_admission.py to cap in place")
    ap.add_argument("--max-rows", type=int, default=None, help="cap the decode capacity-search start")
    ap.add_argument("--run-full", default=None, help="STAGED run_full.py to patch in place (install + traceback)")
    args = ap.parse_args(argv)
    if (args.admission is None) != (args.max_rows is None):
        raise SystemExit("--admission and --max-rows go together")
    if args.admission is not None:
        p = Path(args.admission)
        out = stage_admission(p.read_text(), max_rows=args.max_rows)
        p.write_text(out)
        print("staged_admission", "max_rows", args.max_rows, "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    if args.run_full is not None:
        p = Path(args.run_full)
        out = stage_run_full(p.read_text())
        p.write_text(out)
        print("staged_run_full", "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
