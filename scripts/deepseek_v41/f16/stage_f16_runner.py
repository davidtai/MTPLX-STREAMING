"""Stage the three F16 edits onto STAGED copies of the retained runner helpers.

Every edit is anchored to a UNIQUE line and round-trip-checked (removing the edit
recovers the original byte-for-byte -- the f2/f5 staging discipline).  CPU-safe, no
MLX.  The window runs the EXACT retained helpers from a fresh staging dir; a staged
copy runs while the pinned originals' hash self-checks still pass.

Edits (all to STAGED copies):
  * ``stage_run_full``          -- install F16 as the LAST step of ``observe_seed_
    prefill`` (run_full.py:737, after ``prime_model``); no-op-armed unless
    MTPLX_DSV41_F16=1.  Same anchor the F2b stager uses -- F16 stage 1 is mutually
    exclusive with F2b and the F5 stamp probe.
  * ``stage_hybrid_install``    -- (1) inside ``rewrite`` add one ``replace(...)`` that
    routes the ONE verify forward through ``_F16_PIPELINE.pipelined_forward`` (keeping
    ``mx.array([chunk_ids])`` so the retained AST mx-call check still passes);
    (2) inject ``_F16_PIPELINE = model._f16_pipeline`` into the decode namespace.
  * ``stage_projection_install`` -- grow the projection transpose store from 2 to 4
    buffers (index ``layer % 4``; ScheduledOutput.buffer_index; verify_retirement
    expects 4), so a trailing group one layer behind cannot have its buffer clobbered.
    Extra owner 2 x 67,108,864 bytes (reported in the F16_INSTALL provenance line).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# ---- run_full hook (after prime_model, 8-space indent) ---------------------
_RF_ANCHOR = "        projection_owner_report.update(prime_model(target))"
_RF_INSERT = (
    "        import f16.install as _f16_install  # F16\n"
    "        _f16_install.install_from_env(target)  # F16 (armed only if MTPLX_DSV41_F16=1)"
)

# ---- hybrid_install: route the verify forward + inject the pipeline --------
# The verify forward line lives in the pinned _decode_cycles source (16-space indent);
# rewrite() applies this as one more symmetric replace(), so its round trip + AST
# mx-call check still hold (mx.array([chunk_ids]) is preserved).
_HY_REWRITE_ANCHOR = "    restored = updated"
_HY_REWRITE_INSERT = (
    "    replace('                chunk_logits, chunk_hidden = forward(mx.array([chunk_ids]), cache)',\n"
    "            '                chunk_logits, chunk_hidden = _F16_PIPELINE.pipelined_forward(forward, mx.array([chunk_ids]), cache)')"
)
_HY_NS_ANCHOR = "    namespace['_LOOKUP_EXTENSION'] = lookup"
# Resolved lazily at call time: hybrid install precedes prefill, F16's install follows it.
_HY_NS_INSERT = "    from f16.pipeline import LazyPipeline as _F16Lazy; namespace['_F16_PIPELINE'] = _F16Lazy(model)"

# ---- projection_install: 2 -> 4 transpose buffers --------------------------
_PI_EDITS = (
    ("        self.buffers = [None, None]", "        self.buffers = [None, None, None, None]"),
    ("        self.buffers[layer % 2] = value", "        self.buffers[layer % 4] = value"),
    ("        self.buffer_index = layer % 2", "        self.buffer_index = layer % 4"),
    (
        "    if (len(store.buffers) != 2 or any(tuple(v.shape) != (8,4096,1024)",
        "    if (len(store.buffers) != 4 or any(tuple(v.shape) != (8,4096,1024)",
    ),
    (
        "                or lane.buffer_index != index % 2 or lane.wo_b is not attn.wo_b",
        "                or lane.buffer_index != index % 4 or lane.wo_b is not attn.wo_b",
    ),
)


def _apply_once(text: str, old: str, new: str, label: str) -> str:
    """Replace the single occurrence of ``old`` with ``new`` and assert the reverse
    replacement recovers ``text`` byte-for-byte (so nothing else moved)."""
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: anchor is not unique ({n}x): {old!r}")
    updated = text.replace(old, new)
    if updated.replace(new, old) != text:
        raise RuntimeError(f"{label}: edit changed more than the one anchor")
    return updated


def _insert_after(text: str, anchor: str, insert: str, label: str) -> str:
    return _apply_once(text, anchor, anchor + "\n" + insert, label)


def _insert_before(text: str, anchor: str, insert: str, label: str) -> str:
    return _apply_once(text, anchor, insert + "\n" + anchor, label)


def stage_run_full(source: str) -> str:
    if "_f16_install" in source:
        raise RuntimeError("F16 run_full already staged (_f16_install marker present)")
    return _insert_after(source, _RF_ANCHOR, _RF_INSERT, "F16 run_full hook")


def stage_hybrid_install(source: str) -> str:
    if "_F16_PIPELINE" in source:
        raise RuntimeError("F16 hybrid_install already staged (_F16_PIPELINE marker present)")
    step = _insert_before(source, _HY_REWRITE_ANCHOR, _HY_REWRITE_INSERT, "F16 hybrid rewrite")
    return _insert_after(step, _HY_NS_ANCHOR, _HY_NS_INSERT, "F16 hybrid namespace inject")


def stage_projection_install(source: str) -> str:
    if "[None, None, None, None]" in source:
        raise RuntimeError("F16 projection_install already staged (4-buffer store present)")
    updated = source
    for old, new in _PI_EDITS:
        updated = _apply_once(updated, old, new, "F16 projection 4-buffer")
    return updated


def _stage_file(path: str, fn, label: str) -> str:
    p = Path(path)
    out = fn(p.read_text())
    p.write_text(out)
    sha = hashlib.sha256(out.encode()).hexdigest()[:16]
    print(f"staged_{label}", "sha", sha)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the F16-patched runner in place")
    ap.add_argument("--run-full", default=None, help="STAGED run_full.py to patch")
    ap.add_argument("--hybrid-install", default=None, help="STAGED hybrid_install.py to patch")
    ap.add_argument("--projection-install", default=None, help="STAGED projection_install.py to patch")
    args = ap.parse_args(argv)
    did = False
    if args.run_full is not None:
        _stage_file(args.run_full, stage_run_full, "run_full")
        did = True
    if args.hybrid_install is not None:
        _stage_file(args.hybrid_install, stage_hybrid_install, "hybrid_install")
        did = True
    if args.projection_install is not None:
        _stage_file(args.projection_install, stage_projection_install, "projection_install")
        did = True
    if not did:
        raise SystemExit("nothing to stage: pass --run-full / --hybrid-install / --projection-install")
    return 0


if __name__ == "__main__":
    sys.exit(main())
