"""Derive the F2b intercepted reader from the retained ``plane_lane.bind_reader``.

The replacement is IDENTICAL to the retained reader (same job construction, gate/up
first + down last, same fanout submit, same early gate/up witness publication, same
error joining, metrics updates and view release) EXCEPT the per-plane ``read(job)``
first tries the host ring: a served plane copies from RAM and issues NO pread. To
guarantee the retained reader logic cannot drift, the replacement is not hand-written:
it is DERIVED from ``plane_lane.bind_reader``'s own source by a single anchored line
insertion with a round-trip check (the f5_compile/timed_plane_lane.py discipline), then
exec'd in the retained module's namespace so it closes over the identical helpers.

The inserted line calls ``_f2b_serve(offset, view)`` (the ring's ``try_serve``) and
returns on a hit; it references only that one injected global and issues no ``mx`` op.
CPU-safe: ``derive_bind_reader_source`` is a pure text transform.
"""
from __future__ import annotations

import hashlib

# The retained lane the intercept is pinned to (== timed_plane_lane's pin). A drift in
# plane_lane.py breaks the unique-anchor check below, so this fails loudly.
RETAINED_PLANE_LANE_SHA256 = (
    "1acad9e24c37e5c618b2d8e2e98fb93eb94b5476d5c6de6fa0ee054db468ba54"
)

# Stripped anchor (the per-plane pread) and the stripped injected line inserted BEFORE
# it. The ring lookup is the lane's actual work on the enabled path, not a fallback
# check (AGENTS.md): _f2b_serve is bound once at install and never None on this path.
_ANCHOR = "read_range('experts.bin', offset, (view,), cancel_event=cancel_event,"
# The serve callable is bound once at install and is never None on the enabled path, so
# there is NO eligibility check here (AGENTS.md: no eligible-or-stock branch in the hot path).
_INSERT = "if _f2b_serve(offset, view): return  # F2b ring intercept"


def derive_bind_reader_source(source: str) -> str:
    """Insert the ring intercept before the per-plane pread; assert unique anchor +
    byte-for-byte round trip (removing the injected line recovers the retained source)."""
    if "mx." in _INSERT:
        raise RuntimeError("intercept line would add an mx op")
    orig = source.splitlines()
    hits = [i for i, ln in enumerate(orig) if ln.strip() == _ANCHOR]
    if len(hits) != 1:
        raise RuntimeError(
            f"bind_reader pread anchor is not unique ({len(hits)}x) -- the retained "
            "reader changed; re-pin the F2b intercept"
        )
    i = hits[0]
    indent = orig[i][: len(orig[i]) - len(orig[i].lstrip())]
    lines = list(orig)
    lines.insert(i, indent + _INSERT)
    recovered = [ln for ln in lines if ln.strip() != _INSERT]
    if recovered != orig:
        raise RuntimeError("F2b intercept changed the retained reader beyond the one line")
    return "\n".join(lines)


def build_bind_reader(serve):
    """Compile the derived ``bind_reader`` in ``plane_lane``'s namespace with the ring
    ``serve`` (``HostRing.try_serve``) injected. Returns (bind_reader_fn, base_sha,
    derived_sha). Raises if the retained ``plane_lane.py`` drifted from the pin."""
    import inspect
    import textwrap

    import plane_lane  # retained lane on PYTHONPATH at window time

    lane_src = inspect.getsource(plane_lane)
    if hashlib.sha256(lane_src.encode()).hexdigest() != RETAINED_PLANE_LANE_SHA256:
        raise RuntimeError("plane_lane.py differs from the pinned retained lane")
    base = textwrap.dedent(inspect.getsource(plane_lane.bind_reader))
    derived = derive_bind_reader_source(base)
    namespace = dict(plane_lane.__dict__)
    namespace["_f2b_serve"] = serve
    exec(compile(derived, "<f2b_bind_reader>", "exec"), namespace)  # noqa: S102
    return (
        namespace["bind_reader"],
        hashlib.sha256(base.encode()).hexdigest(),
        hashlib.sha256(derived.encode()).hexdigest(),
    )


def install_intercept(reader, local, ring):
    """Rebind ``reader.read_record_into`` / ``reader.read_component_records_into`` to the
    derived intercept over ``ring``, reusing the installed lane's ``local`` witness."""
    bind_reader_fn, base_sha, derived_sha = build_bind_reader(ring.try_serve)
    bind_reader_fn(reader, local)
    return {"base_bind_reader_sha256": base_sha, "derived_bind_reader_sha256": derived_sha,
            "retained_plane_lane_sha256": RETAINED_PLANE_LANE_SHA256}
