"""Stage a patched copy of the retained ``run_full.py`` for the F5 compile window.

The window runs the EXACT retained runner (extension-bank-20260919
full/sources/packed/run_full.py) with ONE surgical, anchored edit -- applied to a
STAGED COPY, never the committed receipt -- so the measured config/admission/installs
are otherwise byte-for-byte the retained ones:

  (REMOVED by review, 2026-09-19) An earlier draft dropped the retained
    cache-only ``ab._ar_logits_row_at_index = cached_ar_logits_row`` override so the
    AB driver's LIVE replay would classify a divergence at any index.  That is unsafe
    and invalid here: the live replay re-prefills 16,384 tokens INSIDE the candidate
    process after decode (111 expert rows already resident -> the prefill envelope
    no longer fits the 110e9 budget) and it computes the "reference" with the
    candidate's own levers enabled (the retained runner refuses exactly this: "no
    replay through candidate arithmetic").  The retained cache-only override stays.
    A divergence at the known index 297 is classified from the cache as before; a
    divergence at a NEW index raises inside the override, which the AB driver already
    catches (``ar_replay_error``), so the receipt keeps both token streams, timing and
    engagement with the divergence left UNCLASSIFIED.  Classifying a new index needs
    a separate control-arithmetic reference run (teacher-forced M=1 logits along the
    control stream), built only if an arm shows a real gain.

  EDIT 2 (decode-only lever enable + A2 probe, Task 1 / arms B-F / A2): at the
    post-prefill quiescent boundary (observe_prefill_boundary, AFTER growth_transition
    -> after projection_install), call ``f5_decode_levers.enable_from_env`` (flips the
    import-bound globals HC/ATTN/WIN_MEMO and the read-at-use env for
    SMALL_STAGES/ATTN_CORE/HC_PREMIX for the arm) and, if armed, install the
    TimedPackedDecode probe.  Prefill already completed with the retained (all-off)
    state, so the 84-row / 110e9-budget prefill is unchanged.

The edit is anchored to a unique line and the transform asserts that reversing
it recovers the retained source byte-for-byte (the hybrid_install.py/projection_
install.py discipline), so the staged runner is a provable minimal delta.  This
script does NO MLX work and is CPU-safe.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


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
    if source_text.count("    ab._ar_logits_row_at_index = cached_ar_logits_row") != 1:
        raise RuntimeError("retained cache-only AR-logits override not found exactly once")
    if source_text.count(_E2_ANCHOR) != 1:
        raise RuntimeError("EDIT 2 anchor (growth_transition call) not unique")
    updated = source_text.replace(_E2_ANCHOR, _E2_ANCHOR + "\n" + _E2_INSERT)
    # round-trip: reversing the edit must recover the retained source exactly.
    recovered = updated.replace(_E2_ANCHOR + "\n" + _E2_INSERT, _E2_ANCHOR)
    if recovered != source_text:
        raise RuntimeError("F5 staging changed the retained runner beyond the one edit")
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
