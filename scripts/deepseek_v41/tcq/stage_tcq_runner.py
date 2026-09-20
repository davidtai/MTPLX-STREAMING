"""F39 stager: insert the EXPLICIT tcq3-vs-mxfp4 route at packed_phase.py's plane-lane bind site.

Modelled on hybrid_install.rewrite: exactly one anchored replacement, verified to round-trip (reversing it recovers
the original source byte-for-byte) and to leave the module's direct ``mx.*`` calls unchanged.  The stock mxfp4 path
stays byte-identical when the lane is not armed (route_plane_lane calls the same install_plane_lane(rt, ...)) — so
this is a construction-time route, not a runtime fallback (AGENTS.md "correct by design").

Usage (CPU, no GPU):
    python stage_tcq_runner.py --packed-phase <in packed_phase.py> --out <out packed_phase.py>
"""
from __future__ import annotations

import argparse
import ast

_ANCHOR = (
    "                from plane_lane import install as install_plane_lane\n"
    "                plane_runners.update(install_plane_lane(rt, dict(zip(layers, switches)), owners))"
)
_ROUTED = (
    "                from plane_lane import install as install_plane_lane\n"
    "                import tcq.install as _tcq_route\n"
    "                plane_runners.update(_tcq_route.route_plane_lane(rt, layers, switches, owners, install_plane_lane))"
)


def _mlx_calls(text: str):
    return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "mx"]


def rewrite(source: str) -> str:
    """Insert the tcq3 route; assert single anchor, byte-exact round trip, and unchanged direct mx.* calls."""
    if source.count(_ANCHOR) != 1:
        raise RuntimeError("plane-lane bind anchor changed: expected exactly one occurrence in packed_phase.py")
    updated = source.replace(_ANCHOR, _ROUTED)
    if updated.replace(_ROUTED, _ANCHOR) != source:
        raise RuntimeError("tcq3 route edit does not recover the original packed_phase source")
    if _mlx_calls(source) != _mlx_calls(updated):
        raise RuntimeError("tcq3 route edit changed the module's direct mx.* operations")
    ast.parse(updated)                                    # the edited module still parses
    return updated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-phase", required=True, help="path to the (staged) packed_phase.py to edit")
    ap.add_argument("--out", required=True, help="output path (may equal --packed-phase for in-place)")
    args = ap.parse_args()
    with open(args.packed_phase) as f:
        source = f.read()
    updated = rewrite(source)
    with open(args.out, "w") as f:
        f.write(updated)
    print(f"tcq3 route staged into {args.out} (plane-lane bind site; round-trip + mx-call checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
