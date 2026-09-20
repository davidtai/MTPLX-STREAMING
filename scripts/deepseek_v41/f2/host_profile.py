"""F28 generation-thread host profile: run the retained DSpark DECODE under cProfile and dump the
raw pstats + a readable text summary into the arm's receipt directory.

In the two-group verify pipeline everything host-side is serialised on ONE generation thread, and a
closed queueing model makes each 0.1 ms of host time per group slice worth ~0.9 s per 1,024-token run.
Host trims are the most valuable exact lever, but only a profile of the REAL lane says where the Python
time goes.  This arm captures that profile.

Seam order (this is the corrected mechanism -- an earlier version had it backwards).  The retained hybrid
install rewrites ``_decode_cycles``, ``exec``s the copy and reassigns ``module._decode_cycles``, and it runs
at the START of ``dspark_generate`` (from ``run_full.py``'s ``dspark_with_boundary_observation``).  This
module's ``install_from_env`` is staged in packed_phase's growth transition, which runs from the prefill
callback DURING ``dspark_generate`` -- AFTER the hybrid rewrite, and BEFORE ``_decode_cycles`` is called (a
module-global lookup at call time).  So by install time ``_decode_cycles`` is usually already the final,
exec'd hybrid copy (``co_filename == '<hybrid_lookup_decode>'``); wrapping ``hybrid_install.install`` then would
never fire for that (only) generate.  Therefore:

  * when the hybrid copy is live, wrap ``module._decode_cycles`` DIRECTLY: ``dspark_generate`` looks the name up
    as a module global after the prefill callback returns, so the profiled wrapper is what runs (``seam=direct``);
  * otherwise (hybrid not yet live -- tests, or an import-time seam), wrap ``hybrid_install.install`` so it
    re-wraps ``_decode_cycles`` right after each rewrite (``seam=hybrid_install_wrap``).

The profiled wrapper exposes ``__wrapped__`` = the real function, so the F27 cycle-log hook can reach the hybrid
copy's globals regardless of which of F27/F28 installs first (``+prof+cl`` in either order records conf AND
profiles).  ``_profiled`` builds a ``cProfile.Profile()``, ``enable()``s it, calls through, ``disable()``s it in a
``finally`` and dumps there (the decode loop runs once per generate and returns, so no atexit); the dump is
guarded so it never masks a decode error.  The F16 pipeline runs both row groups as greenlets ON the generation
thread; cProfile keeps profiling across greenlet switches on the same thread, so a switched-out frame's time
includes the partner's work -- noted atop ``hostprof.txt``.  ``strip_dirs()`` is NOT applied (staged vs pinned
paths matter).  Files, in ``$MTPLX_DSV41_F28_PROFILE`` (a directory): one generate writes ``hostprof.pstats`` +
``hostprof.txt``; a second renames those to ``hostprof-1.*`` and writes ``hostprof-2.*``, etc.  (On the retained
lane the packed growth transition -- hence this install -- runs once per process; the per-generate numbering is
exercised through the ``hybrid_install.install`` fallback.)  Off unless the variable is set; diagnostic arm.
"""
from __future__ import annotations

import cProfile
import json
import os
import pstats
import traceback

ENV = "MTPLX_DSV41_F28_PROFILE"
_HYBRID_DECODE_FILENAME = "<hybrid_lookup_decode>"   # hybrid_install compile()s the rewrite under this name

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
    """Wrap ``_decode_cycles`` so one generate runs under cProfile and dumps on return.  Exposes
    ``__wrapped__`` for the F27 hook, and ``_f28_host_profile`` so a double install is refused."""

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

    _profiled_decode_cycles.__wrapped__ = decode_cycles
    _profiled_decode_cycles._f28_host_profile = True
    return _profiled_decode_cycles


def install(hybrid_install_module, decode_module, *, out_dir):
    """Install the profiler around the DSpark decode.  If ``decode_module._decode_cycles`` is already the exec'd
    hybrid copy, wrap it directly; otherwise wrap ``hybrid_install_module.install`` so it re-wraps the function
    right after each rewrite.  Refuses (RuntimeError) if ``out_dir`` is not an existing directory, the fallback
    needs ``install`` and it is absent, or the chosen sink is already wrapped.  Construction-time only."""
    out_dir = str(out_dir)
    if not os.path.isdir(out_dir):
        raise RuntimeError(f"F28 profile directory does not exist: {out_dir}")
    counter = [0]

    def _next():
        counter[0] += 1
        return counter[0]

    dc = getattr(decode_module, "_decode_cycles", None)
    if dc is not None and getattr(dc, "_f28_host_profile", False):
        raise RuntimeError("F28 host profile already wrapped _decode_cycles")
    code = getattr(dc, "__code__", None)
    hybrid_live = code is not None and code.co_filename == _HYBRID_DECODE_FILENAME
    if hybrid_live:
        decode_module._decode_cycles = _profiled(dc, out_dir, _next())
        seam = "direct"
    else:
        original_install = getattr(hybrid_install_module, "install", None)
        if original_install is None:
            raise RuntimeError("F28 needs hybrid_install.install (hybrid decode not yet live)")
        if getattr(original_install, "_f28_host_profile", False):
            raise RuntimeError("F28 host profile already wrapped hybrid_install.install")

        def install(*args, **kwargs):
            report = original_install(*args, **kwargs)
            module = args[0] if args else kwargs["module"]
            module._decode_cycles = _profiled(module._decode_cycles, out_dir, _next())
            return report

        install._f28_host_profile = True
        hybrid_install_module.install = install
        seam = "hybrid_install_wrap"
    return {"installed": True, "out_dir": out_dir, "seam": seam}


def install_from_env():
    """Read ``MTPLX_DSV41_F28_PROFILE`` at use.  Unset -> nothing installed; otherwise wrap the DSpark decode of
    the retained decode module (via the staged ``hybrid_install`` module for the not-yet-live fallback)."""
    out_dir = os.environ.get(ENV, "").strip()
    if not out_dir:
        return {"installed": False}
    from mtplx.models import deepseek_v41_dspark_decode as decode_module
    import hybrid_install
    report = install(hybrid_install, decode_module, out_dir=out_dir)
    print("F28_HOST_PROFILE_INSTALL " + json.dumps(report, sort_keys=True), flush=True)
    return report
