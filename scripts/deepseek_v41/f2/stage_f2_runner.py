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


# The F2b host ring is anonymous HOST memory the retained admission does not know about.
# Charge it to the physical (whole-machine) bound so a high live baseline lowers the admitted
# capacity instead of letting the measured machine peak cross the 110e9 ceiling (the guard
# killed an arm at a 13.32 GB baseline on 2026-09-19 before this edit existed).
_PHYS_ANCHOR = (
    "        physical = base + original['host_reserve_bytes'] + embedding_host + lookup_host"
    " + expansion_host + active + original['decode_cache_allowance_bytes']"
)


def stage_admission_ring(source_text: str, *, ring_bytes: int) -> str:
    ring_bytes = int(ring_bytes)
    if not 0 < ring_bytes <= 4 * 1024**3:
        raise RuntimeError("ring_bytes must be within (0, 4 GiB]")
    _assert_once(source_text, _PHYS_ANCHOR, "physical-bound")
    new_line = _PHYS_ANCHOR.replace(
        "physical = base + ", f"physical = base + {ring_bytes} + ", 1
    ) + "  # F2b host ring charged"
    updated = source_text.replace(_PHYS_ANCHOR, new_line)
    if updated.replace(new_line, _PHYS_ANCHOR) != source_text:
        raise RuntimeError("F2b ring charge changed more than the physical-bound line")
    return updated


# Row-split exactness probe: the retained hybrid install hardcodes ONE verify chunk of up to
# 8 rows. Staging a two-chunk schedule (e.g. 4+4) makes the target verify the same rows as
# two causal forwards (the second attends to the first's KV) with the unchanged greedy
# accept/commit path. Emitted tokens are identical IF AND ONLY IF the split-row arithmetic
# is bit-stable in practice, so the run's output digest decides whether a two-group verify
# pipeline would be an exact lever. (Sequential chunks are slower; this arm is not a
# throughput candidate.)
_VC_ANCHOR = "            '    verify_chunks = (8,)')"


def stage_verify_chunks(source_text: str, *, chunks) -> str:
    chunks = tuple(int(c) for c in chunks)
    if len(chunks) < 2 or any(c < 1 for c in chunks) or sum(chunks) != 8:
        raise RuntimeError("verify chunks must be >= 2 positive widths summing to 8")
    _assert_once(source_text, _VC_ANCHOR, "hybrid verify_chunks")
    new_line = _VC_ANCHOR.replace("(8,)", "(" + ", ".join(str(c) for c in chunks) + ")")
    updated = source_text.replace(_VC_ANCHOR, new_line)
    if updated.replace(new_line, _VC_ANCHOR) != source_text:
        raise RuntimeError("verify-chunks edit changed more than the one tuple")
    return updated


# Balanced row-split oracle: per cycle, verify ceil(n/2) rows then floor(n/2) rows (single chunk
# when n <= 4) -- the sequential reference for the F16 pipeline's "balanced" split. Two anchored
# insertions into the STAGED hybrid_install.py: one more symmetric replace() inside rewrite()
# (no mx call added, so its AST mx-call check still holds) and one namespace injection.
_VB_REWRITE_ANCHOR = "    restored = updated"
_VB_REWRITE_INSERT = (
    "    replace('        for configured_width in verify_chunks:',\n"
    "            '        for configured_width in _BALANCED_CHUNKS(len(block_ids)):')"
)
_VB_NS_ANCHOR = "    namespace['_LOOKUP_EXTENSION'] = lookup"
_VB_NS_INSERT = (
    "    namespace['_BALANCED_CHUNKS'] = (lambda n: (n,) if n <= 4 else ((n + 1) // 2, n // 2))"
)


def stage_verify_balanced(source_text: str) -> str:
    if "_BALANCED_CHUNKS" in source_text:
        raise RuntimeError("balanced verify schedule already staged")
    _assert_once(source_text, _VB_REWRITE_ANCHOR, "hybrid rewrite restore")
    _assert_once(source_text, _VB_NS_ANCHOR, "hybrid namespace")
    step = source_text.replace(_VB_REWRITE_ANCHOR, _VB_REWRITE_INSERT + "\n" + _VB_REWRITE_ANCHOR)
    if step.replace(_VB_REWRITE_INSERT + "\n" + _VB_REWRITE_ANCHOR, _VB_REWRITE_ANCHOR) != source_text:
        raise RuntimeError("balanced rewrite edit changed more than the one insertion")
    updated = step.replace(_VB_NS_ANCHOR, _VB_NS_ANCHOR + "\n" + _VB_NS_INSERT)
    if updated.replace(_VB_NS_ANCHOR + "\n" + _VB_NS_INSERT, _VB_NS_ANCHOR) != step:
        raise RuntimeError("balanced namespace edit changed more than the one insertion")
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
    ap.add_argument("--hybrid-install", default=None, help="STAGED hybrid_install.py (with --verify-chunks)")
    ap.add_argument("--verify-chunks", default=None, help="e.g. 4,4 : row-split exactness probe")
    ap.add_argument("--ring-bytes", type=int, default=None,
                    help="with --admission: charge the F2b host ring to the physical bound")
    args = ap.parse_args(argv)
    if (args.admission is None) != (args.max_rows is None):
        raise SystemExit("--admission and --max-rows go together")
    if args.admission is not None:
        p = Path(args.admission)
        out = stage_admission(p.read_text(), max_rows=args.max_rows)
        if args.ring_bytes is not None:
            out = stage_admission_ring(out, ring_bytes=args.ring_bytes)
            print("staged_admission_ring_bytes", args.ring_bytes)
        p.write_text(out)
        print("staged_admission", "max_rows", args.max_rows, "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    if (args.hybrid_install is None) != (args.verify_chunks is None):
        raise SystemExit("--hybrid-install and --verify-chunks go together")
    if args.hybrid_install is not None:
        p = Path(args.hybrid_install)
        if args.verify_chunks == "balanced":
            out = stage_verify_balanced(p.read_text())
        else:
            out = stage_verify_chunks(p.read_text(), chunks=[int(x) for x in args.verify_chunks.split(",")])
        p.write_text(out)
        print("staged_verify_chunks", args.verify_chunks, "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    if args.run_full is not None:
        p = Path(args.run_full)
        out = stage_run_full(p.read_text())
        p.write_text(out)
        print("staged_run_full", "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
