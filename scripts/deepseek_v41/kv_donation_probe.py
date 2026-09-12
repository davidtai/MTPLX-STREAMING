#!/usr/bin/env python3
"""W107 (review round-2 finding 1): does ``mx.slice_update`` DONATE into a
preallocated KV buffer on Metal, or copy the whole plane every append?

The GPU census (window 42) showed the per-token ``cache_append`` stage did NOT move
under ``MTPLX_DSV41_KV_BOUNDED`` -- 11.9 ms/tok, same as when the lever did not exist
-- even though the bounded lanes are preallocated and the CPU donation proof (memory
flat across appends) held.  The strong hypothesis: ``mx.slice_update`` on Metal
donates into the buffer ONLY when the buffer is uniquely referenced, and the cache's
read (``window_all = layer_cache.window`` == ``buf[:, :length]``, a live view that
aliases ``_buf``) keeps it non-unique, so every append copies the whole resident plane
(O(T)), defeating the whole lever.

This probe DISCRIMINATES that, at the real lane shapes (window/latent [1, T, 512],
index [1, T, 128], T = 16384..17408), by timing a SINGLE append into a preallocated
buffer and reading the mx active/peak memory delta -- a donating write is ms-flat and
memory-flat across T; a copying write is O(T) in both.  It probes each candidate write
primitive:

  * ``mx.slice_update``           -- the cache's current primitive
  * ``array[:, r:r+1, :] = new``  -- MLX slice ``__setitem__`` (if supported)
  * ``mx.put_along_axis``         -- scatter along the sequence axis
  * ``mx.fast.metal_kernel``      -- a hand-written in-place row write (GPU only)

and -- the load-bearing comparison -- runs each with the previous ``view()`` slice
DROPPED (as the cache reassigns ``self._buf``) vs with a live view HELD (as the model
holds ``window_all`` across the forward).  If a primitive is ms/memory-flat only when
the view is dropped, the fix is architectural (eliminate the read-time alias); if no
primitive is flat on Metal, the fix is a true in-place kernel.

CPU-validatable (default device CPU): every non-metal_kernel primitive runs on CPU and
the mechanism (slice_update flat vs concat growing) is asserted by
``tests/test_deepseek_v41_w107_kv_growth.py``.  ``--gpu`` selects Metal for the real
answer; the orchestrator runs it in a lock gap (this script is the ONLY Metal user in
that window).  MLX does not expose an array's data pointer in Python, so the donation
signal is the active/peak-memory delta + the ms-vs-T slope (documented in W107 §6).

Usage::

    # CPU validation (no GPU):
    MTPLX_TEST_CPU=1 python scripts/deepseek_v41/kv_donation_probe.py \
        --sizes 512,1024,2048 --dim 64 --reps 20
    # real answer (GPU, orchestrator only, under the flock):
    python scripts/deepseek_v41/kv_donation_probe.py --gpu \
        --sizes 16384,16896,17408 --dim 512 --dtype fp32 --reps 50 --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable, Optional

import mlx.core as mx


def _dtype(name: str):
    return {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}[name]


def _active() -> int:
    try:
        return int(mx.get_active_memory())
    except Exception:
        return -1


def _peak() -> int:
    try:
        return int(mx.get_peak_memory())
    except Exception:
        return -1


def _reset_peak() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        pass


def _starts(row: int, ndim: int = 3) -> mx.array:
    return mx.array([0, int(row)] + [0] * (ndim - 2), dtype=mx.int32)


# --- write primitives: each returns a NEW buffer with row ``r`` set to ``new`` ------
def _w_slice_update(buf: mx.array, new: mx.array, r: int) -> mx.array:
    return mx.slice_update(buf, new, _starts(r, buf.ndim), axes=tuple(range(buf.ndim)))


def _w_setitem(buf: mx.array, new: mx.array, r: int) -> mx.array:
    # MLX slice assignment (functional-ish item set); may be unsupported on a build.
    buf[:, r : r + 1, :] = new
    return buf


def _w_put_along_axis(buf: mx.array, new: mx.array, r: int) -> mx.array:
    idx = mx.full((buf.shape[0], 1, buf.shape[2]), r, dtype=mx.int32)
    return mx.put_along_axis(buf, idx, new, axis=1)


def _metal_row_write_kernel():
    """A minimal Metal kernel that copies ``buf`` to the output with row ``r`` replaced
    by ``new`` -- a single fused dispatch (GPU only).  This is NOT a true in-place
    donate (MLX allocates the output); it measures whether one kernel dispatch beats
    slice_update's copy, and is the scaffold for a real ``donate``-annotated write."""
    source = r"""
        uint gid = thread_position_in_grid.x;
        uint total = buf_shape[0] * buf_shape[1] * buf_shape[2];
        if (gid >= total) return;
        uint d = buf_shape[2];
        uint row = (gid / d) % buf_shape[1];
        out[gid] = (row == (uint)r[0]) ? new_row[(gid % d)] : buf[gid];
    """
    return mx.fast.metal_kernel(
        name="kv_row_write",
        input_names=["buf", "new_row", "r"],
        output_names=["out"],
        source=source,
    )


def _make_metal_writer():
    kern = _metal_row_write_kernel()

    def _w(buf: mx.array, new: mx.array, r: int) -> mx.array:
        (out,) = kern(
            inputs=[buf, new.reshape(-1), mx.array([r], dtype=mx.int32)],
            output_shapes=[buf.shape],
            output_dtypes=[buf.dtype],
            grid=(buf.size, 1, 1),
            threadgroup=(256, 1, 1),
        )
        return out

    return _w


def _data_ptr(a) -> Optional[int]:
    """The buffer-protocol data pointer of an mx.array, or None if unavailable
    (bf16 has no numpy dtype; Metal may refuse a zero-copy view)."""
    if a is None:
        return None
    try:
        import numpy as _np
        return int(_np.array(a, copy=False).__array_interface__["data"][0])
    except Exception:
        return None


def _time_primitive(name: str, writer: Callable, N: int, D: int, dtype,
                    reps: int, hold_view: bool) -> dict:
    """Append a single row into a preallocated [1, N, D] buffer at row N-1, in the
    cache's REBIND pattern (``buf = writer(buf, ...)`` -- the previous descriptor is
    dropped, so ``buf`` is uniquely referenced at the write and MLX donates).  Track the
    buffer data-pointer FLIPS across appends: 0 flips == donates (pointer-stable), reps
    flips == copies.  ``hold_view`` keeps a live ``buf[:, :row, :]`` view of the CURRENT
    buffer alive at each write (the cache's read alias), which raises the refcount and
    defeats donation -- so we measure both the primitive AND the alias hazard.

    (Round-4 fix: round 3's ``out = w(buf); buf = out`` kept the PREVIOUS result alive,
    so every input had refcount 2 and slice_update wrongly read as COPY -- the bug the
    re-review found.  The real cache pattern rebinds with no lingering alias.)"""
    B = 1
    row = N - 1
    buf = mx.zeros((B, N, D), dtype=dtype)
    new = mx.ones((B, 1, D), dtype=dtype)
    mx.eval(buf, new)
    try:
        buf = writer(buf, new, row)  # warmup (compile / first dispatch), rebind
        mx.eval(buf)
    except Exception as exc:  # unsupported primitive on this build/device
        return {"primitive": name, "N": N, "supported": False, "error": repr(exc)[:200]}

    _reset_peak()
    base_active = _active()
    p0 = _data_ptr(buf)
    ptr_flips = 0
    t0 = time.perf_counter()
    for _ in range(reps):
        held = buf[:, :row, :] if hold_view else None   # view of the CURRENT buffer
        if held is not None:
            mx.eval(held)
        buf = writer(buf, new, row)                      # REBIND (no lingering `out`)
        mx.eval(buf)
        p1 = _data_ptr(buf)
        if p0 is not None and p1 is not None and p1 != p0:
            ptr_flips += 1
        p0 = p1
        held = None
    t1 = time.perf_counter()
    ms = (t1 - t0) / reps * 1e3
    return {
        "primitive": name,
        "N": N,
        "supported": True,
        "hold_view": hold_view,
        "ms_per_append": round(ms, 4),
        "ptr_flips": ptr_flips,
        "ptr_available": p0 is not None,
        "active_delta_bytes": _active() - base_active,
        "peak_delta_bytes": _peak() - base_active,
        "buffer_bytes": B * N * D * dtype.size,
    }


def _time_concat(N: int, D: int, dtype, reps: int) -> dict:
    """Baseline: concat a [1, N-1, D] store with a [1,1,D] row -> [1,N,D] (O(N))."""
    B = 1
    head = mx.zeros((B, N - 1, D), dtype=dtype)
    new = mx.ones((B, 1, D), dtype=dtype)
    mx.eval(head, new)
    out = mx.concatenate([head, new], axis=1)
    mx.eval(out)  # warmup
    _reset_peak()
    base_active = _active()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = mx.concatenate([head, new], axis=1)
        mx.eval(out)
    t1 = time.perf_counter()
    ms = (t1 - t0) / reps * 1e3
    return {
        "primitive": "concat", "N": N, "supported": True, "hold_view": False,
        "ms_per_append": round(ms, 4),
        "active_delta_bytes": _active() - base_active,
        "peak_delta_bytes": _peak() - base_active,
        "buffer_bytes": B * N * D * dtype.size,
    }


def run(sizes, dim, dtype_name, reps, use_gpu) -> dict:
    dtype = _dtype(dtype_name)
    device = "gpu" if use_gpu else "cpu"
    mx.set_default_device(mx.gpu if use_gpu else mx.cpu)

    writers = {
        "slice_update": _w_slice_update,
        "setitem": _w_setitem,
        "put_along_axis": _w_put_along_axis,
    }
    if use_gpu:
        try:
            writers["metal_kernel"] = _make_metal_writer()
        except Exception as exc:  # pragma: no cover - build without fast kernels
            writers["metal_kernel"] = None
            print(f"[probe] metal_kernel unavailable: {exc!r}")

    rows = []
    for N in sizes:
        rows.append(_time_concat(N, dim, dtype, reps))
        for hold_view in (False, True):
            for name, w in writers.items():
                if w is None:
                    continue
                if name == "metal_kernel" and hold_view:
                    continue  # the kernel allocates its own output; view-alias N/A
                rows.append(_time_primitive(name, w, N, dim, dtype, reps, hold_view))
    out = {
        "device": device, "dtype": dtype_name, "dim": dim, "sizes": list(sizes),
        "reps": reps, "rows": rows,
    }
    out["verdict"] = _verdict(rows)
    return out


def _verdict(rows) -> str:
    """Per primitive/hold_view, classify DONATE vs COPY.  The PRIMARY signal is the
    buffer data-pointer: 0 flips across the rebind appends == DONATE (pointer-stable),
    flips-per-append == COPY.  ms-slope + peak_delta are corroborating (a copy is O(T)
    in ms and allocates ~one buffer plane).  Pointer trumps ms because ms at the cell
    is dominated by fence/dispatch latency, not the copy (window-42 finding)."""
    from collections import defaultdict
    series = defaultdict(list)
    for r in rows:
        if not r.get("supported"):
            continue
        key = (r["primitive"], r.get("hold_view", False))
        series[key].append(r)
    lines = []
    for (prim, hv), rs in sorted(series.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rs.sort(key=lambda r: r["N"])
        r0, r1 = rs[0], rs[-1]
        ms0, ms1 = r0["ms_per_append"], r1["ms_per_append"]
        slope = ms1 / ms0 if ms0 > 0 else float("inf")
        pk1, buf1 = r1["peak_delta_bytes"], r1["buffer_bytes"]
        # pointer flips at the largest N (reps writes)
        flips = r1.get("ptr_flips")
        ptr_ok = r1.get("ptr_available", False)
        if ptr_ok:
            verdict = "DONATE(ptr-stable)" if flips == 0 else f"COPY(ptr-flips={flips})"
        else:  # pointer unavailable -> fall back to ms-slope + peak
            verdict = ("DONATE?" if (slope < 2.0 and pk1 < 0.5 * buf1)
                       else "COPY?(O(T))")
        lines.append(
            f"  {prim:14s} hold_view={hv!s:5s}: ptr_flips@N{r1['N']}="
            f"{flips if ptr_ok else 'n/a'} ms {ms0:.4f}->{ms1:.4f} (x{slope:.1f}) "
            f"peak_delta {pk1} B (buffer {buf1} B) => {verdict}"
        )
    return "\n".join(lines)


def self_test() -> dict:
    """CPU self-test: mx.slice_update is pointer-STABLE in the cache's rebind pattern
    (donates) and FLIPS when a view of the buffer is held at the write.  Returns a dict
    with both counts + a pass flag; raises AssertionError if the pointer is available
    but donation does not hold (the probe would otherwise mis-measure)."""
    mx.set_default_device(mx.cpu)
    reb = _time_primitive("slice_update", _w_slice_update, 4096, 64, mx.float32,
                          reps=16, hold_view=False)
    held = _time_primitive("slice_update", _w_slice_update, 4096, 64, mx.float32,
                           reps=16, hold_view=True)
    out = {"rebind_ptr_flips": reb.get("ptr_flips"),
           "held_view_ptr_flips": held.get("ptr_flips"),
           "ptr_available": reb.get("ptr_available", False)}
    if out["ptr_available"]:
        assert out["rebind_ptr_flips"] == 0, (
            f"slice_update did NOT donate in the rebind pattern "
            f"({out['rebind_ptr_flips']} flips) -- probe would mis-measure")
        assert out["held_view_ptr_flips"] > 0, (
            "holding a view did not defeat donation -- probe cannot discriminate")
        out["pass"] = True
    else:
        out["pass"] = None  # pointer unavailable -> self-test inconclusive (guarded)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", action="store_true",
                    help="run on Metal (default: CPU for validation)")
    ap.add_argument("--sizes", default="2048,8192,16384,17408",
                    help="comma-separated preallocated seq lengths T (8x spread)")
    ap.add_argument("--dim", type=int, default=512, help="head_dim (512 window/latent, 128 index)")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--self-test", action="store_true",
                    help="CPU: assert slice_update is pointer-stable in the rebind pattern")
    ap.add_argument("--json", default=None, help="write the full result table to this JSON path")
    args = ap.parse_args(argv)

    if args.self_test:
        st = self_test()
        print(f"[probe] self-test: {st}")
        return 0 if st.get("pass") is not False else 1

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    res = run(sizes, args.dim, args.dtype, args.reps, args.gpu)

    print(f"[probe] device={res['device']} dtype={res['dtype']} dim={res['dim']} "
          f"reps={res['reps']} sizes={res['sizes']}")
    for r in res["rows"]:
        if not r.get("supported"):
            print(f"  {r['primitive']:14s} N={r['N']}: UNSUPPORTED ({r.get('error')})")
            continue
        print(f"  {r['primitive']:14s} N={r['N']:6d} hv={str(r.get('hold_view')):5s} "
              f"ptr_flips={r.get('ptr_flips')!s:>4s} ms={r['ms_per_append']:.4f} "
              f"peak_d={r['peak_delta_bytes']:>11d} buf={r['buffer_bytes']:>11d}")
    print("[probe] verdict (DONATE = buffer pointer stable across appends; COPY = flips):")
    print(res["verdict"])
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")
        print(f"[probe] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
