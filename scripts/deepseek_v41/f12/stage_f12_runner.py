"""Stage the F12 parallel-scales edit into a STAGED copy of ``packed_phase.py``.

Like the F2/F6 stagers, this applies anchored, unique-line insertions to a STAGED
COPY of the retained ``packed_phase.py`` (never the committed receipt, never
anything under ``mtplx/``) and asserts that reversing every insertion recovers the
input byte-for-byte, so the staged runner is a provable minimal delta. It does NO
MLX work and is CPU-safe.

The runner imports helpers by module name (PYTHONPATH order), so the edited staged
``packed_phase.py`` picks up ``parallel_scales`` from ``scripts/deepseek_v41/f12``
placed on PYTHONPATH.

Three additive edits inside ``transition()`` (all within its existing
``try: ... except BaseException: state['phase']='failed'; raise`` block, so a
failure surfaced by ``finish()`` is caught exactly like any other transition
failure):

  EDIT 1  Route ONCE, right before the first ``remove_raw_scales`` call:
            import parallel_scales as _f12
            _f12_load_layer, _f12_finish = _f12.resolve(load_layer)
          ``resolve`` reads ``MTPLX_DSV41_F12_PARALLEL_SCALES`` once and returns
          prebound callables. Unset/``0`` -> the ORIGINAL serial ``load_layer`` and
          a no-op finish, so one staged tree serves both control and candidate arms.

  EDIT 2  The per-layer loop calls the routed loader:
            owners[layer] = load_layer(...)   ->   owners[layer] = _f12_load_layer(...)

  EDIT 3  Join point: ``_f12_finish()`` AFTER the per-layer
          ``for layer, switch in zip(layers, switches):`` loop and BEFORE
          ``for physical in (*pool._persistent.values(), *pool._transient):`` --
          every read+hash completes before any code can consume the scale contents.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# EDIT 1 -- route once, inserted before the transient-bank remove_raw_scales call
# (16-space indent, inside transition()'s try block).
_A1 = "                released = remove_raw_scales(allocator.banks['transient', -1], mx=mx)"
_I1 = (
    "                import parallel_scales as _f12  # F12: parallel growth-transition scales\n"
    "                _f12_load_layer, _f12_finish = _f12.resolve(load_layer)  # route once via MTPLX_DSV41_F12_PARALLEL_SCALES"
)

# EDIT 2 -- the per-layer loop uses the routed loader (20-space indent).
_A2 = "                    owners[layer] = load_layer(ROOT / 'artifact', inventory['layers'][layer], mx=mx)"
_A2_NEW = "                    owners[layer] = _f12_load_layer(ROOT / 'artifact', inventory['layers'][layer], mx=mx)"

# EDIT 3 -- join every read/hash after the per-layer loop and before the first
# post-loop consumer. Two-line anchor pins the exact join point; the loop body ends
# with the 20-space mx.clear_cache(), the 16-space for-physical begins the next step.
_A3 = (
    "                    mx.clear_cache()\n"
    "                for physical in (*pool._persistent.values(), *pool._transient):"
)
_A3_NEW = (
    "                    mx.clear_cache()\n"
    "                _f12_finish()  # F12 join: reads+hashes complete before any scale consumer\n"
    "                for physical in (*pool._persistent.values(), *pool._transient):"
)

# A marker unique to F12's inserts, used to refuse a double-apply.
_MARKER = "_f12_finish"


def stage(source_text: str) -> str:
    """Apply the three F12 edits; assert reversing all three recovers ``source_text``."""
    if _MARKER in source_text:
        raise RuntimeError("F12 staging already applied (_f12_finish marker present)")
    for anchor, label in ((_A1, "EDIT 1 remove_raw_scales(transient)"),
                          (_A2, "EDIT 2 per-layer load_layer call"),
                          (_A3, "EDIT 3 post-loop join point")):
        n = source_text.count(anchor)
        if n != 1:
            raise RuntimeError(f"{label} anchor is not unique ({n} occurrences)")

    updated = source_text.replace(_A1, _I1 + "\n" + _A1)
    updated = updated.replace(_A2, _A2_NEW)
    updated = updated.replace(_A3, _A3_NEW)

    # Round-trip: reversing every edit must recover the input byte-for-byte.
    recovered = updated.replace(_A3_NEW, _A3)
    recovered = recovered.replace(_A2_NEW, _A2)
    recovered = recovered.replace(_I1 + "\n" + _A1, _A1)
    if recovered != source_text:
        raise RuntimeError("F12 staging changed packed_phase.py beyond its three edits")
    return updated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the F12 parallel-scales edit in place")
    ap.add_argument("--packed-phase", required=True,
                    help="STAGED packed_phase.py to patch IN PLACE (never the committed receipt)")
    args = ap.parse_args(argv)
    path = Path(args.packed_phase)
    src = path.read_text()
    out = stage(src)
    path.write_text(out)
    print("input_sha256", hashlib.sha256(src.encode()).hexdigest())
    print("staged_sha256", hashlib.sha256(out.encode()).hexdigest())
    print("staged_path", str(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
