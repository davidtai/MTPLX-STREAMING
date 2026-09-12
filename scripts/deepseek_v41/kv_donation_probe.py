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


def _time_primitive(name: str, writer: Callable, N: int, D: int, dtype,
                    reps: int, hold_view: bool) -> dict:
    """Time a single row append into a preallocated [1, N, D] buffer at row N-1.

    ``hold_view`` keeps a live ``buf[:, :N-1, :]`` slice alive across the write (the
    cache's read alias) to test whether it defeats donation."""
    B = 1
    row = N - 1
    buf = mx.zeros((B, N, D), dtype=dtype)
    mx.eval(buf)
    new = mx.ones((B, 1, D), dtype=dtype)
    mx.eval(new)
    # warmup (compile / first dispatch)
    try:
        _b = writer(buf, new, row)
        mx.eval(_b)
        buf = _b
    except Exception as exc:  # unsupported primitive on this build/device
        return {"primitive": name, "N": N, "supported": False, "error": repr(exc)[:200]}

    _reset_peak()
    base_active = _active()
    held = buf[:, :row, :] if hold_view else None
    if held is not None:
        mx.eval(held)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = writer(buf, new, row)
        mx.eval(out)
        buf = out
    t1 = time.perf_counter()
    # keep ``held`` referenced across the loop so the alias is live at each write
    if held is not None:
        _ = int(held.shape[1])
    ms = (t1 - t0) / reps * 1e3
    return {
        "primitive": name,
        "N": N,
        "supported": True,
        "hold_view": hold_view,
        "ms_per_append": round(ms, 4),
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
    """Per primitive/hold_view: does ms scale with N (copy) or stay flat (donate)?
    A donating write's ms at the largest N is < ~2x its ms at the smallest N."""
    from collections import defaultdict
    series = defaultdict(list)
    for r in rows:
        if not r.get("supported"):
            continue
        key = (r["primitive"], r.get("hold_view", False))
        series[key].append(
            (r["N"], r["ms_per_append"], r["peak_delta_bytes"], r["buffer_bytes"]))
    lines = []
    for (prim, hv), pts in sorted(series.items()):
        pts.sort()
        n0, ms0, _, _ = pts[0]
        n1, ms1, pk1, buf1 = pts[-1]
        slope = ms1 / ms0 if ms0 > 0 else float("inf")
        # DONATE == ms roughly flat across T AND the largest-N peak grew far less than
        # one buffer plane (a copy allocates ~buffer_bytes; a donate reuses in place).
        ms_flat = slope < 2.0
        mem_flat = pk1 < 0.5 * buf1
        verdict = "DONATE" if (ms_flat and mem_flat) else "COPY(O(T))"
        lines.append(
            f"  {prim:14s} hold_view={hv!s:5s}: ms {ms0:.4f}->{ms1:.4f} "
            f"(x{slope:.1f} over N {n0}->{n1}), peak_delta@N{n1} {pk1} B "
            f"(buffer {buf1} B) => {verdict}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", action="store_true",
                    help="run on Metal (default: CPU for validation)")
    ap.add_argument("--sizes", default="16384,16896,17408",
                    help="comma-separated preallocated seq lengths T")
    ap.add_argument("--dim", type=int, default=512, help="head_dim (512 window/latent, 128 index)")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--json", default=None, help="write the full result table to this JSON path")
    args = ap.parse_args(argv)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    res = run(sizes, args.dim, args.dtype, args.reps, args.gpu)

    print(f"[probe] device={res['device']} dtype={res['dtype']} dim={res['dim']} "
          f"reps={res['reps']} sizes={res['sizes']}")
    for r in res["rows"]:
        if not r.get("supported"):
            print(f"  {r['primitive']:14s} N={r['N']}: UNSUPPORTED ({r.get('error')})")
            continue
        print(f"  {r['primitive']:14s} N={r['N']:6d} hv={str(r.get('hold_view')):5s} "
              f"ms={r['ms_per_append']:.4f} active_d={r['active_delta_bytes']:>10d} "
              f"peak_d={r['peak_delta_bytes']:>10d} buf={r['buffer_bytes']:>10d}")
    print("[probe] verdict (DONATE = ms flat & memory flat across T; COPY = O(T)):")
    print(res["verdict"])
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")
        print(f"[probe] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
