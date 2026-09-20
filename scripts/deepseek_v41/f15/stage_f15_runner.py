"""Stage a patched copy of the retained ``run_full.py`` for the F15 row-dump window.

Like the F5 and F6 stagers, this applies an anchored, unique-line insertion to a STAGED
COPY of the retained runner (never the committed receipt) and asserts that reversing the
insertion recovers the input byte-for-byte, so the staged runner is a provable minimal
delta. It does NO MLX work and is CPU-safe.

INSERT: at the post-prefill quiescent boundary, right after ``growth_transition()``
(run_full.py:781) -- the SAME anchor the F5 and F6 stagers use. The two inserted lines

    import row_dump as _f15rd
    _f15rd.install_from_env()

install the F15 row-dump wrap before the decode loop runs its first ``observe`` call.
``install_from_env()`` is a NO-OP unless ``MTPLX_DSV41_F15_ROW_DUMP_DIR`` is set, so an
unarmed staged runner is behaviourally identical to the retained one.

Composition (apply F5, then F6, then this -- any order in fact): F5 and F6 both insert
their own lines AFTER the ``growth_transition()`` line, leaving that anchor line present
exactly once. So this stager's ``replace`` still finds a unique anchor and lands the F15
lines immediately after ``growth_transition()`` (ahead of the F6/F5 blocks); none of the
three inserts shares a line with another, so each stager's own round-trip still holds.
A double-apply is refused via the ``_f15rd.install_from_env`` marker.

The staged runner needs ``scripts/deepseek_v41/f15`` on PYTHONPATH so ``import row_dump``
resolves (the same way F6 needs ``scripts/deepseek_v41/f6`` for ``import engram_parallel``).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Post-prefill quiescent boundary inside observe_prefill_boundary (20-space indent). Same
# anchor as F5's EDIT 2 and F6's INSERT D; those leave the anchor line present once.
_ANCHOR = "                    growth_transition()"
_INSERT = (
    "                    import row_dump as _f15rd\n"
    "                    _f15rd.install_from_env()"
)

# A marker unique to F15's insert, used to refuse a double-apply.
_F15_MARKER = "_f15rd.install_from_env"


def stage(source_text: str) -> str:
    """Insert the F15 row-dump install; assert reversing it recovers ``source_text``.

    Works on the raw retained runner or on an already-F5/F6-staged copy (they share the
    ``growth_transition()`` anchor but leave it present exactly once).
    """
    if _F15_MARKER in source_text:
        raise RuntimeError("F15 staging already applied (install_from_env marker present)")
    if source_text.count(_ANCHOR) != 1:
        raise RuntimeError("F15 anchor (post-prefill growth_transition call) not unique")

    updated = source_text.replace(_ANCHOR, _ANCHOR + "\n" + _INSERT)

    # round-trip: reversing the one F15 edit must recover the input exactly.
    recovered = updated.replace(_ANCHOR + "\n" + _INSERT, _ANCHOR)
    if recovered != source_text:
        raise RuntimeError("F15 staging changed the runner beyond its one insert")
    return updated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the F15-patched run_full.py copy")
    ap.add_argument("--retained", required=True, help="input run_full.py (retained, or F5/F6-staged)")
    ap.add_argument("--out", required=True, help="staged patched run_full.py path")
    args = ap.parse_args(argv)
    src = Path(args.retained).read_text()
    patched = stage(src)
    Path(args.out).write_text(patched)
    print("input_sha256", hashlib.sha256(src.encode()).hexdigest())
    print("staged_sha256", hashlib.sha256(patched.encode()).hexdigest())
    print("staged_path", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
