"""Stage patched copies of the retained packed helpers for the F2 prefetch window.

The window runs the EXACT retained runner (docs/deepseek-v41/receipts/
extension-bank-20260919/full/sources/) from a fresh /private/tmp staging dir, never
the committed receipt and never the pinned original at
/private/tmp/dsv41-extension-bank-20260919/. run_full.py IMPORTS its helpers by module
name, so a staged copy first on PYTHONPATH runs while run_full's own hash self-checks
(against the pinned PACKED_ROOT/GROWTH_ROOT) still see the unchanged originals.

Every edit is anchored to a UNIQUE line and asserted to round-trip (reversing it
recovers the retained source byte-for-byte), exactly like the F5 stager
(.../dsv41-f5-compile/scripts/deepseek_v41/f5_compile/stage_f5_runner.py). CPU-safe, no
MLX. Two edits are well-defined and shipped here:

  * ``stage_admission`` — cap the decode capacity search (equal-capacity ladder, like
    F5) AND, for the candidate arm, charge the R-record speculative ring reserve into
    the admitted ``active`` bound so the row count is derived through the retained
    admission code (packed_admission.py:108 / :126), not a parallel formula.
  * ``stage_packed_phase`` — swap the single ``install_plane_lane(...)`` call in
    ``install_growth`` for ``f2.run_full_install.install_f2_growth(...)`` (the F2 lane).

IMPORTANT — the candidate ring is not yet end-to-end runnable with these two edits
alone: ``packed_phase.install_growth`` refuses a prefetch ring at construction
(packed_phase.py:41-42/54/64-65) and its byte accounting (:78-93) assumes none, and the
runtime must be built with ``prefetch_slots=R`` upstream. See the receipt's "ring-enable
gap" section; the control arms (no ring) ARE runnable with ``stage_admission`` alone.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# One packed mxfp4 weight record (packed_admission.py WEIGHTS; == full_config.EXPERT_WEIGHT_RECORD_BYTES).
WEIGHTS = 17_694_720

_CAP_ANCHOR = "    for capacity in range(112, old_capacity, -1):"
_ACTIVE_ANCHOR = "        active = max(steady, resize, seed, append_peak)"
_INSTALL_ANCHOR = (
    "                from plane_lane import install as install_plane_lane\n"
    "                plane_runners.update(install_plane_lane(rt, dict(zip(layers, switches)), owners))"
)
_INSTALL_F2 = (
    "                from f2.run_full_install import install_f2_growth\n"
    "                plane_runners.update(install_f2_growth(rt, dict(zip(layers, switches)), owners, model=model))"
)


def _assert_once(text: str, anchor: str, label: str) -> None:
    n = text.count(anchor)
    if n != 1:
        raise RuntimeError(f"{label} anchor is not unique ({n} occurrences)")


def stage_admission(source_text: str, *, max_rows: int | None = None, ring_records: int = 0) -> str:
    """Cap the decode capacity-search start at ``max_rows`` and/or charge an
    ``ring_records``-record ring reserve into the admitted ``active`` bound.

    Both edits are anchored + round-trip-checked. With neither argument the source is
    returned unchanged.
    """
    updated = source_text
    if max_rows is not None:
        if not 85 <= int(max_rows) <= 112:
            raise RuntimeError("max_rows must be within 85..112")
        _assert_once(updated, _CAP_ANCHOR, "capacity-search")
        new_cap = f"    for capacity in range({int(max_rows)}, old_capacity, -1):  # F2 equal-capacity cap"
        updated = updated.replace(_CAP_ANCHOR, new_cap)
        if updated.replace(new_cap, _CAP_ANCHOR) != source_text:
            raise RuntimeError("F2 capacity cap changed more than the search start")
    if ring_records:
        if not 0 < int(ring_records) <= 64:
            raise RuntimeError("ring_records must be within 1..64")
        _assert_once(updated, _ACTIVE_ANCHOR, "admission active bound")
        charge = (
            _ACTIVE_ANCHOR
            + f"\n        active += {int(ring_records)} * WEIGHTS  # F2 speculative ring reserve (R records)"
        )
        before = updated
        updated = updated.replace(_ACTIVE_ANCHOR, charge)
        if updated.replace(charge, _ACTIVE_ANCHOR) != before:
            raise RuntimeError("F2 ring charge changed more than the active bound")
    return updated


def stage_packed_phase(source_text: str) -> str:
    """Swap the retained ``install_plane_lane`` call for the F2 lane install."""
    _assert_once(source_text, _INSTALL_ANCHOR, "install_growth plane-lane install")
    updated = source_text.replace(_INSTALL_ANCHOR, _INSTALL_F2)
    if updated.replace(_INSTALL_F2, _INSTALL_ANCHOR) != source_text:
        raise RuntimeError("F2 packed_phase staging changed more than the install call")
    return updated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the F2-patched packed helpers in place")
    ap.add_argument("--admission", default=None, help="STAGED packed_admission.py to patch in place")
    ap.add_argument("--max-rows", type=int, default=None, help="cap the decode capacity-search start")
    ap.add_argument("--ring-records", type=int, default=0, help="charge an R-record ring reserve (candidate)")
    ap.add_argument("--packed-phase", default=None, help="STAGED packed_phase.py to swap the install in place")
    args = ap.parse_args(argv)
    if args.admission is not None:
        p = Path(args.admission)
        src = p.read_text()
        out = stage_admission(src, max_rows=args.max_rows, ring_records=args.ring_records)
        p.write_text(out)
        print("staged_admission", "max_rows", args.max_rows, "ring_records", args.ring_records,
              "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    if args.packed_phase is not None:
        p = Path(args.packed_phase)
        src = p.read_text()
        out = stage_packed_phase(src)
        p.write_text(out)
        print("staged_packed_phase", "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
