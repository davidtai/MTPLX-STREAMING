"""Stage a patched copy of the retained ``run_full.py`` for the F6 Engram window.

Like the F5 stager, this applies anchored, unique-line insertions to a STAGED
COPY of the retained runner (never the committed receipt) and asserts that
reversing each insertion recovers the input byte-for-byte, so the staged runner
is a provable minimal delta. It does NO MLX work and is CPU-safe.

Two install points are inserted, both behind the single env value
``MTPLX_DSV41_F6_INSTALL`` (``load`` or ``decode``); each insert calls
``engram_parallel.install_from_env(model, site=...)``, which installs only when
its ``site`` matches the env value AND ``MTPLX_DSV41_F6_ENGRAM_PARALLEL=1``:

  INSERT L (whole-run, prefill + decode): right after Codex's bounded-Engram
    install/describe has run inside ``checked_load`` --
    anchor ``        bounded_engram_report = describe_engram(resident.model)``
    (run_full.py:620). The model is ``resident.model``. Installing here rebinds
    the parallel miss path before prefill, so the ~180k serial Engram reads in
    prefill also fan out (TTFT).

  INSERT D (decode-only): at the post-prefill quiescent boundary, right after
    ``growth_transition()`` -- anchor ``                    growth_transition()``
    (run_full.py:781, the SAME anchor the F5 stager uses; ``target`` is the model
    captured by the ``dspark_with_boundary_observation`` closure). Prefill has
    already run with the retained serial path; only decode uses the parallel one.

Composition with the F5 stager (apply F5 first, then this): INSERT D shares F5's
``growth_transition()`` anchor. Applied after F5, INSERT D lands between
``growth_transition()`` and F5's EDIT-2 block; the anchor is still unique, F5's
block is untouched, and each stager's own round-trip still holds. INSERT L is on
a line F5 never touches. The runner needs ``scripts/deepseek_v41/f6`` on
PYTHONPATH so ``import engram_parallel`` resolves.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


# INSERT L -- whole-run install, inside checked_load (8-space indent).
_L_ANCHOR = "        bounded_engram_report = describe_engram(resident.model)"
_L_INSERT = (
    "        import engram_parallel as _f6ep\n"
    "        _f6ep.install_from_env(resident.model, site='load')"
)

# INSERT D -- decode-only install, inside observe_prefill_boundary (20-space
# indent). Same anchor as F5's EDIT 2 (``growth_transition()``); ``target`` is
# captured by the enclosing dspark_with_boundary_observation closure.
_D_ANCHOR = "                    growth_transition()"
_D_INSERT = (
    "                    import engram_parallel as _f6ep\n"
    "                    _f6ep.install_from_env(target, site='decode')"
)

# A marker unique to F6's inserts, used to refuse a double-apply.
_F6_MARKER = "_f6ep.install_from_env"


def stage(source_text: str) -> str:
    """Insert both F6 install points; assert reversing both recovers ``source_text``.

    Works on the raw retained runner or on an already-F5-staged copy (INSERT D
    shares F5's anchor but F5 leaves it present exactly once)."""
    if _F6_MARKER in source_text:
        raise RuntimeError("F6 staging already applied (install_from_env marker present)")
    if source_text.count(_L_ANCHOR) != 1:
        raise RuntimeError("INSERT L anchor (load-time describe_engram) not unique")
    if source_text.count(_D_ANCHOR) != 1:
        raise RuntimeError("INSERT D anchor (post-prefill growth_transition call) not unique")

    updated = source_text.replace(_L_ANCHOR, _L_ANCHOR + "\n" + _L_INSERT)
    updated = updated.replace(_D_ANCHOR, _D_ANCHOR + "\n" + _D_INSERT)

    # round-trip: reversing both F6 edits must recover the input exactly.
    recovered = updated.replace(_D_ANCHOR + "\n" + _D_INSERT, _D_ANCHOR)
    recovered = recovered.replace(_L_ANCHOR + "\n" + _L_INSERT, _L_ANCHOR)
    if recovered != source_text:
        raise RuntimeError("F6 staging changed the runner beyond its two inserts")
    return updated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the F6-patched run_full.py copy")
    ap.add_argument("--retained", required=True, help="input run_full.py (retained, or F5-staged)")
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
