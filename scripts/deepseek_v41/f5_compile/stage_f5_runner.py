"""Stage a patched copy of the retained ``run_full.py`` for the F5 compile window.

The window runs the EXACT retained runner (extension-bank-20260919
full/sources/packed/run_full.py) with two surgical, anchored edits -- applied to a
STAGED COPY, never the committed receipt -- so the measured config/admission/installs
are otherwise byte-for-byte the retained ones:

  EDIT 1 (classify any divergence, addendum point b): drop the cache-only
    ``ab._ar_logits_row_at_index = cached_ar_logits_row`` monkeypatch so the AB
    driver's LIVE ``_ar_logits_row_at_index`` (M=1 forwards + one prefill on the AR
    reference prefix, provenance-clean -- it never replays candidate arithmetic)
    computes the AR reference logits at WHATEVER index a compile arm first diverges,
    yielding a W120 tie_flip/divergent verdict instead of the HC-screen's null-at-480.
    The AB classifier still runs by default (it does not require --dspark-require-tie
    -class), so each arm's receipt carries dspark.divergence with the contested
    margins + tie_band the classifier needs.

  EDIT 2 (decode-only lever enable + A2 probe, Task 1 / arms B-F / A2): at the
    post-prefill quiescent boundary (observe_prefill_boundary, AFTER growth_transition
    -> after projection_install), call ``f5_decode_levers.enable_from_env`` (flips the
    import-bound globals HC/ATTN/WIN_MEMO and the read-at-use env for
    SMALL_STAGES/ATTN_CORE/HC_PREMIX for the arm) and, if armed, install the
    TimedPackedDecode probe.  Prefill already completed with the retained (all-off)
    state, so the 84-row / 110e9-budget prefill is unchanged.

Both edits are anchored to unique lines and the transform asserts that reversing
them recovers the retained source byte-for-byte (the hybrid_install.py/projection_
install.py discipline), so the staged runner is a provable minimal delta.  This
script does NO MLX work and is CPU-safe.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


# EDIT 1: a single-line replacement (drop the cache-only AR-logits override).
_E1_OLD = "    ab._ar_logits_row_at_index = cached_ar_logits_row"
_E1_NEW = (
    "    # F5: keep the AB driver's LIVE _ar_logits_row_at_index so a divergence at\n"
    "    # ANY index is classified from the AR reference prefix (not candidate replay).\n"
    "    _ = cached_ar_logits_row  # retained builder kept for provenance; not installed"
)

# EDIT 2: an insertion after the growth transition inside observe_prefill_boundary.
# ``target`` (the model) is captured by that closure (dspark_with_boundary_observation).
_E2_ANCHOR = "                    growth_transition()"
_E2_INSERT = (
    "                    import mtplx.models.deepseek_v41 as _f5_dv\n"
    "                    import f5_decode_levers as _f5\n"
    "                    _f5.enable_from_env(_f5_dv)\n"
    "                    if _f5.timed_probe_requested():\n"
    "                        _f5.install_and_arm_probe(target, scheduled=True)"
)


def stage(source_text: str) -> str:
    if source_text.count(_E1_OLD) != 1:
        raise RuntimeError("EDIT 1 anchor (cached_ar_logits_row install) not unique")
    if source_text.count(_E2_ANCHOR) != 1:
        raise RuntimeError("EDIT 2 anchor (growth_transition call) not unique")
    updated = source_text.replace(_E1_OLD, _E1_NEW)
    updated = updated.replace(_E2_ANCHOR, _E2_ANCHOR + "\n" + _E2_INSERT)
    # round-trip: reversing both edits must recover the retained source exactly.
    recovered = updated.replace(_E2_ANCHOR + "\n" + _E2_INSERT, _E2_ANCHOR)
    recovered = recovered.replace(_E1_NEW, _E1_OLD)
    if recovered != source_text:
        raise RuntimeError("F5 staging changed the retained runner beyond the 2 edits")
    return updated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the F5-patched run_full.py copy")
    ap.add_argument("--retained", required=True, help="tracked retained run_full.py")
    ap.add_argument("--out", required=True, help="staged patched run_full.py path")
    args = ap.parse_args(argv)
    src = Path(args.retained).read_text()
    patched = stage(src)
    Path(args.out).write_text(patched)
    print("retained_sha256", hashlib.sha256(src.encode()).hexdigest())
    print("staged_sha256", hashlib.sha256(patched.encode()).hexdigest())
    print("staged_path", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
