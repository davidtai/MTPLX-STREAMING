"""F28/F30 generation-thread host profile: WALL-TIME sampling profiler for the retained DSpark decode.

In the two-group verify pipeline everything host-side is serialised on ONE generation thread, and a closed
queueing model makes each 0.1 ms of host time per group slice worth ~0.9 s per 1,024-token run.  Host trims are
the most valuable exact lever, but only a profile of the REAL lane says where the time goes.

Why sampling, not cProfile (F30 supersedes F28's cProfile path).  Python 3.12's ``cProfile`` is built on
``sys.monitoring``, which records events from ALL threads into one call stack; with 3+ I/O worker threads and two
greenlets on the generation thread the caller/cumulative numbers are scrambled (``posix.preadv`` showed 24.9 s
"own" time, ``plan()`` 11.95 s cumulative with 0.5 s of children).  What is needed is WALL-TIME attribution for
ONE thread including time it spends BLOCKED (in ``mx.eval``, on futures, waiting for the GIL).  So this samples the
generation thread's own stack via ``sys._current_frames()[gen_ident]`` from a daemon thread: when the generation
thread blocks it releases the GIL, the sampler runs and records the frame it is blocked in, so blocked time is
attributed to the call that is blocked.

Seam (unchanged from F28; the growth-transition hook runs AFTER the hybrid install has rewritten and reassigned
``_decode_cycles``).  ``install`` picks the seam once: DIRECT (``seam=direct``) when ``decode_module._decode_cycles``
is already the exec'd hybrid copy (``co_filename == '<hybrid_lookup_decode>'``) -- wrap it directly, since
``dspark_generate`` looks the name up as a module global after the prefill callback returns; FALLBACK
(``seam=hybrid_install_wrap``) otherwise -- wrap ``hybrid_install.install`` so each rewrite is re-wrapped.  The
wrapper exposes ``__wrapped__`` so the F27 conf hook reaches the hybrid copy's globals in either install order.

``_profiled`` records ``threading.get_ident()`` of the calling (generation) thread, starts a daemon sampler,
calls through, stops+joins the sampler in ``finally`` (no leaked thread), and writes the outputs (guarded so a
write error never masks a decode error).  The sampler loop (period ``MTPLX_DSV41_F28_SAMPLE_US``, default 500 us,
validated once at install to 100..10000) does minimal per-sample work -- tuples + dict increments, no string
formatting: it snapshots ``sys._current_frames().get(gen_ident)`` and walks ``f_back`` to the root, aggregating
``leaf[(file, func, line)]`` (innermost frame), ``incl[(file, func)]`` (each distinct function on the stack), and
``stack[innermost 6 (func, line)]``, and records the inter-sample ``perf_counter_ns`` deltas so the reader sees how
starved the sampler was by the GIL.  With the F16 greenlets, ``sys._current_frames()`` returns the frame of
whichever greenlet is currently running -- exactly the thread we attribute.

Outputs in ``$MTPLX_DSV41_F28_PROFILE`` (a directory), ``hostprof`` stem (``-2`` for a second generate):
``hostprof.samples.json`` (schema 2: period_us, samples, interval_ns, leaf/incl/stacks sorted desc and truncated
to 400/400/200) and ``hostprof.txt`` (header + top 60 leaf / 60 inclusive / 40 stacks, each with % of samples and a
ms-equivalent ``count * wall / samples``; directories NOT stripped).  Off unless the variable is set; diagnostic arm.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback

ENV = "MTPLX_DSV41_F28_PROFILE"                       # output directory (unchanged)
PERIOD_ENV = "MTPLX_DSV41_F28_SAMPLE_US"              # sampling period in microseconds
_DEFAULT_PERIOD_US = 500
_HYBRID_DECODE_FILENAME = "<hybrid_lookup_decode>"    # hybrid_install compile()s the rewrite under this name
_LEAF_TOP, _INCL_TOP, _STACK_TOP = 400, 400, 200
_STACK_DEPTH = 6


class _Sampler(threading.Thread):
    """Daemon thread that samples ONE thread's Python stack at a fixed period.  Aggregates run in this thread;
    read them only after ``stop()`` + ``join()``."""

    def __init__(self, gen_ident, period_us):
        super().__init__(daemon=True, name="mtplx-f28-sampler")
        self._gen_ident = gen_ident
        self._period_s = period_us / 1e6
        self.period_us = period_us
        self._stop_event = threading.Event()   # NB: Thread._stop is an internal method -- do not shadow it
        self.leaf = {}
        self.incl = {}
        self.stacks = {}
        self.intervals = []
        self.samples = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        current_frames = sys._current_frames
        gen = self._gen_ident
        leaf = self.leaf
        incl = self.incl
        stacks = self.stacks
        intervals = self.intervals
        perf = time.perf_counter_ns
        wait = self._stop_event.wait
        period = self._period_s
        depth = _STACK_DEPTH
        last = perf()
        n = 0
        while not wait(period):
            fr = current_frames().get(gen)
            if fr is None:
                continue
            now = perf()
            intervals.append(now - last)
            last = now
            code = fr.f_code
            lk = (code.co_filename, code.co_name, fr.f_lineno)
            leaf[lk] = leaf.get(lk, 0) + 1
            chain = []
            seen = set()
            f = fr
            while f is not None:
                c = f.f_code
                ik = (c.co_filename, c.co_name)
                if ik not in seen:
                    seen.add(ik)
                    incl[ik] = incl.get(ik, 0) + 1
                if len(chain) < depth:
                    chain.append((c.co_name, f.f_lineno))
                f = f.f_back
            sk = tuple(chain)
            stacks[sk] = stacks.get(sk, 0) + 1
            n += 1
        self.samples = n


def _interval_stats(intervals):
    if not intervals:
        return {"count": 0, "mean_ns": 0, "p50_ns": 0, "p99_ns": 0, "max_ns": 0}
    s = sorted(intervals)
    n = len(s)
    return {
        "count": n,
        "mean_ns": sum(s) // n,
        "p50_ns": s[min(n - 1, n // 2)],
        "p99_ns": s[min(n - 1, (99 * n) // 100)],
        "max_ns": s[-1],
    }


def _write_json(sampler, path):
    leaf = sorted(sampler.leaf.items(), key=lambda kv: kv[1], reverse=True)[:_LEAF_TOP]
    incl = sorted(sampler.incl.items(), key=lambda kv: kv[1], reverse=True)[:_INCL_TOP]
    stacks = sorted(sampler.stacks.items(), key=lambda kv: kv[1], reverse=True)[:_STACK_TOP]
    payload = {
        "schema": 2,
        "period_us": sampler.period_us,
        "samples": sampler.samples,
        "interval_ns": _interval_stats(sampler.intervals),
        "leaf": [{"file": f, "func": fn, "line": ln, "count": c} for (f, fn, ln), c in leaf],
        "incl": [{"file": f, "func": fn, "count": c} for (f, fn), c in incl],
        "stacks": [{"stack": [[nm, ln] for nm, ln in s], "count": c} for s, c in stacks],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh)


def _write_text(sampler, path, g, wall_ns):
    n = sampler.samples
    wall_s = wall_ns / 1e9
    per_ms = (wall_s / n * 1e3) if n else 0.0        # wall ms attributed to one sample
    ist = _interval_stats(sampler.intervals)
    leaf = sorted(sampler.leaf.items(), key=lambda kv: kv[1], reverse=True)
    incl = sorted(sampler.incl.items(), key=lambda kv: kv[1], reverse=True)
    stacks = sorted(sampler.stacks.items(), key=lambda kv: kv[1], reverse=True)
    pct = (lambda c: 100.0 * c / n) if n else (lambda c: 0.0)
    with open(path, "w") as fh:
        fh.write("# F30 sampling profile of the DSpark decode GENERATION THREAD (generate %d).\n" % g)
        fh.write("# Samples are WALL TIME of the generation thread INCLUDING blocked time "
                 "(mx.eval, futures, GIL waits).\n")
        fh.write("# The F16 pipeline runs both row groups as greenlets on this thread; sys._current_frames()\n")
        fh.write("#   returns the frame of whichever greenlet is running -- exactly the thread we attribute.\n")
        fh.write("# strip_dirs() is NOT applied: full paths distinguish staged vs pinned modules.\n")
        fh.write("period_us=%d samples=%d wall_s=%.3f  interval_ns(mean=%d p99=%d max=%d)\n\n"
                 % (sampler.period_us, n, wall_s, ist["mean_ns"], ist["p99_ns"], ist["max_ns"]))
        fh.write("== top 60 leaf (where the thread is / is blocked) ==\n")
        for (f, fn, ln), c in leaf[:60]:
            fh.write("%6.2f%%  %9.1f ms  %s:%d %s\n" % (pct(c), c * per_ms, f, ln, fn))
        fh.write("\n== top 60 inclusive functions ==\n")
        for (f, fn), c in incl[:60]:
            fh.write("%6.2f%%  %9.1f ms  %s %s\n" % (pct(c), c * per_ms, f, fn))
        fh.write("\n== top 40 stacks (innermost %d frames, innermost first) ==\n" % _STACK_DEPTH)
        for s, c in stacks[:40]:
            fh.write("%6.2f%%  %9.1f ms  %s\n"
                     % (pct(c), c * per_ms, " <- ".join("%s:%d" % (nm, ln) for nm, ln in s)))


def _dump(sampler, out_dir, g, wall_ns):
    """Write hostprof.{samples.json,txt} for a single generate; on the 2nd generate rename the first pair to
    hostprof-1.* and write hostprof-2.*, and hostprof-<g>.* thereafter."""
    if g == 1:
        base = "hostprof"
    else:
        if g == 2:
            for suffix in (".samples.json", ".txt"):
                first = os.path.join(out_dir, "hostprof" + suffix)
                if os.path.exists(first):
                    os.replace(first, os.path.join(out_dir, "hostprof-1" + suffix))
        base = "hostprof-%d" % g
    _write_json(sampler, os.path.join(out_dir, base + ".samples.json"))
    _write_text(sampler, os.path.join(out_dir, base + ".txt"), g, wall_ns)


def _profiled(decode_cycles, out_dir, g, period_us):
    """Wrap ``_decode_cycles`` so one generate runs under the per-thread sampler and dumps on return.  Exposes
    ``__wrapped__`` for the F27 hook, and ``_f28_host_profile`` so a double install is refused."""

    def _profiled_decode_cycles(*args, **kwargs):
        sampler = _Sampler(threading.get_ident(), period_us)
        t0 = time.perf_counter_ns()
        sampler.start()
        try:
            return decode_cycles(*args, **kwargs)
        finally:
            sampler.stop()
            sampler.join(timeout=5.0)
            wall_ns = time.perf_counter_ns() - t0
            try:
                _dump(sampler, out_dir, g, wall_ns)
            except Exception:   # pragma: no cover - telemetry must not mask a decode error
                traceback.print_exc()

    _profiled_decode_cycles.__wrapped__ = decode_cycles
    _profiled_decode_cycles._f28_host_profile = True
    return _profiled_decode_cycles


def install(hybrid_install_module, decode_module, *, out_dir, period_us=_DEFAULT_PERIOD_US):
    """Install the sampler around the DSpark decode.  DIRECT-wrap ``_decode_cycles`` when the exec'd hybrid copy is
    already live, else wrap ``hybrid_install_module.install`` so it re-wraps after each rewrite.  Refuses
    (RuntimeError) if ``out_dir`` is not an existing directory, ``period_us`` is not an integer in 100..10000, the
    fallback needs ``install`` and it is absent, or the chosen sink is already wrapped.  Construction-time only."""
    out_dir = str(out_dir)
    if not os.path.isdir(out_dir):
        raise RuntimeError(f"F28 profile directory does not exist: {out_dir}")
    try:
        period_us = int(period_us)
    except (TypeError, ValueError):
        raise RuntimeError(f"F28 sample period must be an integer 100..10000 us; got {period_us!r}")
    if not 100 <= period_us <= 10000:
        raise RuntimeError(f"F28 sample period must be 100..10000 us; got {period_us}")
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
        decode_module._decode_cycles = _profiled(dc, out_dir, _next(), period_us)
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
            module._decode_cycles = _profiled(module._decode_cycles, out_dir, _next(), period_us)
            return report

        install._f28_host_profile = True
        hybrid_install_module.install = install
        seam = "hybrid_install_wrap"
    return {"installed": True, "out_dir": out_dir, "seam": seam, "period_us": period_us}


def install_from_env():
    """Read ``MTPLX_DSV41_F28_PROFILE`` (directory) and ``MTPLX_DSV41_F28_SAMPLE_US`` (period) at use.  Unset dir ->
    nothing installed; otherwise wrap the DSpark decode of the retained decode module (via the staged
    ``hybrid_install`` module for the not-yet-live fallback)."""
    out_dir = os.environ.get(ENV, "").strip()
    if not out_dir:
        return {"installed": False}
    period_us = os.environ.get(PERIOD_ENV, "").strip() or _DEFAULT_PERIOD_US
    from mtplx.models import deepseek_v41_dspark_decode as decode_module
    import hybrid_install
    report = install(hybrid_install, decode_module, out_dir=out_dir, period_us=period_us)
    print("F28_HOST_PROFILE_INSTALL " + json.dumps(report, sort_keys=True), flush=True)
    return report
