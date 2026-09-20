"""F21 submit yield: hand the GIL to the reader threads right after a route's miss reads are submitted.

F13 measured the cost this removes: the generation thread keeps the GIL for ~0.3 ms of Python after
``begin_split_route`` submits the miss parts, and at the default 5 ms switch interval the freshly woken
part / fanout threads cannot reach ``os.preadv`` until it blocks -- first preadv 332 us after submit vs
31 us, +0.331 ms per 12-plane burst, the SSD idle for that long at every burst start.
``os.sched_yield()`` x8 right after the submit removed the whole tax in that benchmark (first preadv
36 us) without touching the process-wide switch interval (which costs the host phases more than it saves).

This wraps the runtime INSTANCE's ``begin_split_route`` (the packed lane calls it through the instance), and
yields only when the route actually submitted miss parts.  No bytes, offsets, ordering or arithmetic change.
Installed once at the quiescent post-prefill growth boundary (staged packed_phase hook).  No MLX.
"""
from __future__ import annotations

import json
import os

ENV = "MTPLX_DSV41_F21_SUBMIT_YIELDS"


def install(runtime, *, yields: int) -> dict:
    if isinstance(yields, bool) or not isinstance(yields, int) or not 1 <= yields <= 64:
        raise RuntimeError(f"F21 submit yields must be an integer in [1, 64]; got {yields!r}")
    if getattr(runtime, "_f21_submit_yield", None) is not None:
        raise RuntimeError("F21 submit yield already installed on this runtime")
    # The pending route records its submitted miss parts as ``_all_miss_parts`` (ordinal -> plan).  Checked
    # ONCE here against the live class source so the per-call test below can never read a missing name.
    import inspect

    from mtplx.expert_runtime import PendingSplitRoute

    if "self._all_miss_parts" not in inspect.getsource(PendingSplitRoute.__init__):
        raise RuntimeError("F21: PendingSplitRoute no longer records _all_miss_parts; re-derive the submit test")
    original = runtime.begin_split_route            # bound method of the live runtime
    sched_yield = os.sched_yield
    spins = range(yields)

    def begin_split_route(*args, **kwargs):
        pending = original(*args, **kwargs)
        if pending._all_miss_parts:                 # reads were submitted: let their threads reach preadv
            for _ in spins:
                sched_yield()
        return pending

    runtime.begin_split_route = begin_split_route   # instance attribute shadows the class method
    runtime._f21_submit_yield = yields
    return {"installed": True, "yields": yields}


def install_from_env(runtime) -> dict:
    raw = os.environ.get(ENV)
    if not raw:
        return {"installed": False}
    report = install(runtime, yields=int(raw))
    print("F21_SUBMIT_YIELD_INSTALL " + json.dumps(report, sort_keys=True), flush=True)
    return report
