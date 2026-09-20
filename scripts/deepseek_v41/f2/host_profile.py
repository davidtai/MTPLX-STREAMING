"""F28 generation-thread host profile: run the retained DSpark DECODE under cProfile and dump the
raw pstats + a readable text summary into the arm's receipt directory.

In the two-group verify pipeline everything host-side is serialised on ONE generation thread, and a
closed queueing model makes each 0.1 ms of host time per group slice worth ~0.9 s per 1,024-token
run.  Host trims are therefore the most valuable exact lever, but only a profile of the REAL lane
says where the Python time goes.  This arm captures that profile.

Seam (the reassignment problem).  The retained hybrid install ``exec``s a rewritten copy of
``_decode_cycles`` and then does ``module._decode_cycles = namespace['_decode_cycles']``
(hybrid_install.py), so a wrapper placed on ``module._decode_cycles`` BEFORE the install would be
thrown away by that reassignment.  ``dspark_generate`` resolves ``_decode_cycles`` as a module global
at call time, so the wrapper must be applied AFTER the install returns and BEFORE generation calls it.
The install is invoked at generate time from ``run_full.py``'s ``dspark_with_boundary_observation``
via ``from hybrid_install import install as install_hybrid`` (a fresh module-attribute read each
generate).  So this module wraps ``hybrid_install.install`` itself: right after the original install
returns (``_decode_cycles`` now final), the wrapper replaces ``module._decode_cycles`` with a profiled
version.  ``dspark_with_boundary_observation`` then calls ``original_dspark_generate``, whose
``_decode_cycles`` global is now the profiled wrapper.

``profiled`` builds a ``cProfile.Profile()``, ``enable()``s it, calls through, ``disable()``s it in a
``finally`` and dumps the two files there (the decode loop runs once per generate and returns, so
atexit is not needed).  The F16 pipeline runs the two row groups as greenlets ON the generation
thread; cProfile keeps profiling across greenlet switches on the same thread, so a switched-out
frame's time includes the partner's work -- noted at the top of ``hostprof.txt``.  ``strip_dirs()`` is
NOT applied so staged vs pinned paths stay distinguishable.

Files, in ``$MTPLX_DSV41_F28_PROFILE`` (a directory): a single generate writes ``hostprof.pstats`` +
``hostprof.txt``; a second generate renames those to ``hostprof-1.*`` and writes ``hostprof-2.*``
(and so on).  Off unless the variable is set; diagnostic arm, never a throughput candidate.  Validated
once at install; nothing here runs a per-cycle check.
"""
from __future__ import annotations

import cProfile
import json
import os
import pstats
import traceback

ENV = "MTPLX_DSV41_F28_PROFILE"

_CAVEAT = (
    "# F28 host profile of the DSpark decode generation thread (generate {g}).\n"
    "# CAVEAT: the F16 pipeline runs both row groups as greenlets ON this thread; cProfile keeps\n"
    "#   profiling across greenlet switches, so a switched-out frame's time includes the partner's work.\n"
    "# strip_dirs() is NOT applied: full paths distinguish staged vs pinned modules.\n\n"
)


def _write_text(prof, txt_path, g):
    with open(txt_path, "w") as fh:
        fh.write(_CAVEAT.format(g=g))
        stats = pstats.Stats(prof, stream=fh)   # NO strip_dirs(): keep full paths
        fh.write("== top 80 by tottime ==\n")
        stats.sort_stats("tottime").print_stats(80)
        fh.write("\n== top 80 by cumulative ==\n")
        stats.sort_stats("cumulative").print_stats(80)
        fh.write("\n== callers of the 10 largest tottime entries ==\n")
        stats.sort_stats("tottime").print_callers(10)


def _dump(prof, out_dir, g):
    """Write hostprof.{pstats,txt} for a single generate; on the 2nd generate rename the first to
    hostprof-1.* and write hostprof-2.*, and hostprof-<g>.* thereafter."""
    if g == 1:
        base = "hostprof"
    else:
        if g == 2:
            for ext in ("pstats", "txt"):
                first = os.path.join(out_dir, "hostprof." + ext)
                if os.path.exists(first):
                    os.replace(first, os.path.join(out_dir, "hostprof-1." + ext))
        base = "hostprof-%d" % g
    prof.dump_stats(os.path.join(out_dir, base + ".pstats"))
    _write_text(prof, os.path.join(out_dir, base + ".txt"), g)


def _profiled(decode_cycles, out_dir, g):
    """Wrap the final ``_decode_cycles`` so one generate runs under cProfile and dumps on return."""

    def _profiled_decode_cycles(*args, **kwargs):
        prof = cProfile.Profile()
        prof.enable()
        try:
            return decode_cycles(*args, **kwargs)
        finally:
            prof.disable()
            try:
                _dump(prof, out_dir, g)
            except Exception:   # pragma: no cover - telemetry must not mask a decode error
                traceback.print_exc()

    return _profiled_decode_cycles


def install(hybrid_install_module, *, out_dir):
    """Wrap ``hybrid_install_module.install`` so it re-wraps ``module._decode_cycles`` with the profiled
    version right after the original install reassigns it.  Refuses (RuntimeError) if ``out_dir`` is not
    an existing directory, ``install`` is absent, or ``install`` is already wrapped.  Construction-time only."""
    out_dir = str(out_dir)
    if not os.path.isdir(out_dir):
        raise RuntimeError(f"F28 profile directory does not exist: {out_dir}")
    original_install = getattr(hybrid_install_module, "install", None)
    if original_install is None:
        raise RuntimeError("F28 needs hybrid_install.install")
    if getattr(original_install, "_f28_host_profile", False):
        raise RuntimeError("F28 host profile already wrapped hybrid_install.install")
    counter = [0]

    def install(*args, **kwargs):
        report = original_install(*args, **kwargs)
        module = args[0] if args else kwargs["module"]
        counter[0] += 1
        module._decode_cycles = _profiled(module._decode_cycles, out_dir, counter[0])
        return report

    install._f28_host_profile = True
    hybrid_install_module.install = install
    return {"installed": True, "out_dir": out_dir}


def install_from_env():
    """Read ``MTPLX_DSV41_F28_PROFILE`` at use.  Unset -> nothing installed; otherwise wrap the staged
    ``hybrid_install`` module's ``install``."""
    out_dir = os.environ.get(ENV, "").strip()
    if not out_dir:
        return {"installed": False}
    import hybrid_install
    report = install(hybrid_install, out_dir=out_dir)
    print("F28_HOST_PROFILE_INSTALL " + json.dumps(report, sort_keys=True), flush=True)
    return report
