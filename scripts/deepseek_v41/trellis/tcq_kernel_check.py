"""F35 GPU harness: correctness + microbenchmark of the trellis projection kernels vs the retained mxfp4 kernel.

Runs ONLY inside a guarded GPU window:
    gpu_window.sh <python> tcq_kernel_check.py <out.json>
(`--cpu-smoke` builds the tables, the kernels (no call) and the CPU reference at tiny shapes; no Metal.)

Correctness: random K=3 tile codes for `capacity` experts, random assignments (ids) and activations; the reference
is `mtplx.eschamoe.decode_expert_weights` (bit-exact vs the vendor goldens) followed by an fp32 matmul; the kernels
must match to ~1e-3 relative (fp32 accumulation order differs).
Benchmark: median ms per call at the real per-slice shape (18 assignments = 3 rows x top-6) and at 24, for the mxfp4
packed-lane kernel (`make_projection`, raw-scale variant, same bytes as the lane's packed variant) and the two trellis
kernels, both geometries.  Every timed call is a chained dependency on the previous output so kernels cannot overlap.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SMOKE = "--cpu-smoke" in sys.argv
import mlx.core as mx  # noqa: E402

if SMOKE:
    mx.set_default_device(mx.cpu)

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcq_kernels as tk  # noqa: E402
from mtplx import eschamoe  # noqa: E402

RETAINED_KERNELS = ("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f23-prompts/docs/deepseek-v41/"
                    "receipts/extension-bank-20260919/full/sources/packed/kernels.py")


def warp_tables():
    """lane_a / lane_b / lane_p (warp assembly) and pos_r / pos_c: tile (row, col) of lane L's m-th decoded weight."""
    z = np.load(WT / "mtplx" / "eschamoe_gather.npz")
    perm_lane, perm_m = z["perm_lane_K3"], z["perm_m_K3"]          # tile position p -> (lane, m)
    pos_r = np.zeros(256, np.int32); pos_c = np.zeros(256, np.int32)
    for p in range(256):
        s = int(perm_lane[p]) * 8 + int(perm_m[p])
        pos_r[s], pos_c[s] = p // 16, p % 16
    assert sorted(int(perm_lane[p]) * 8 + int(perm_m[p]) for p in range(256)) == list(range(256))
    arr = lambda a: mx.array(np.asarray(a, dtype=np.int32))
    return tuple(arr(z[f"{nm}_K3"]) for nm in ("lane_a", "lane_b", "lane_p")) + (arr(pos_r), arr(pos_c))


def random_bank(capacity: int, IN: int, OUT: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    codes = rng.integers(-32768, 32767, size=(capacity, IN // 16, OUT // 16, tk.NW), dtype=np.int16)
    return mx.array(codes)


def reference(xh: mx.array, ids: np.ndarray, code: mx.array) -> mx.array:
    outs = []
    for r, e in enumerate(ids):
        W = eschamoe.decode_expert_weights(code[int(e)], tk.K_BITS).astype(mx.float32)   # [IN, OUT]
        outs.append(xh[r : r + 1] @ W)
    return mx.concatenate(outs, axis=0)


def load_retained_projection():
    import importlib.util
    spec = importlib.util.spec_from_file_location("retained_kernels", RETAINED_KERNELS)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.make_projection


def timed(fn, iters=50, warm=8):
    """Chained timing: each call consumes the previous output so calls serialise on the GPU."""
    ts = []
    carry = None
    for i in range(iters + warm):
        mx.synchronize()
        t0 = time.perf_counter()
        carry = fn(carry)
        mx.eval(carry)
        t1 = time.perf_counter()
        if i >= warm:
            ts.append((t1 - t0) * 1e3)
    return statistics.median(ts), sorted(ts)[int(0.9 * len(ts))]


def main():
    out_path = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = out_path[0] if out_path else None
    tables = warp_tables()
    results = {"smoke": SMOKE, "mlx": mx.__version__, "geometries": {}}
    geoms = [(2304, 5120), (5120, 2304)]           # (OUT, IN): gate/up then down
    capacity = 4 if SMOKE else 24
    rows_list = [2] if SMOKE else [18, 24]
    for (OUT, IN) in geoms:
        g = {}
        code = random_bank(capacity, IN, OUT, seed=OUT)
        rng = np.random.default_rng(1)
        rows = rows_list[0]
        ids_np = rng.integers(0, capacity, size=rows).astype(np.uint32)
        xh = mx.array(rng.standard_normal((rows, IN)).astype(np.float32))
        ids = mx.array(ids_np)
        mx.eval(code, xh, ids)
        ref = reference(xh, ids_np, code); mx.eval(ref)
        if SMOKE:
            for variant in ("qmv", "tile"):
                tk.make_tcq_projection(OUT, IN, variant)         # constructs only; no Metal call on the CPU
            g["reference_checksum"] = float(mx.abs(ref).sum())
            results["geometries"][f"{OUT}x{IN}"] = g
            print(f"smoke {OUT}x{IN}: reference ok, kernels constructed", flush=True)
            continue
        for variant in ("qmv", "tile"):
            kern = tk.make_tcq_projection(OUT, IN, variant)
            y = tk.run_tcq_projection(kern, variant, xh, ids, code, OUT, tables); mx.eval(y)
            rel = float(mx.sqrt(mx.sum((y - ref) ** 2) / mx.sum(ref ** 2)))
            maxabs = float(mx.max(mx.abs(y - ref)))
            g[f"{variant}_rel_err"] = rel; g[f"{variant}_max_abs"] = maxabs
            print(f"{OUT}x{IN} {variant}: rel {rel:.3e} max|d| {maxabs:.3e}", flush=True)
        # --- benchmark at the real shapes -------------------------------------------------
        make_projection = load_retained_projection()
        mx_kern = make_projection(OUT, IN, packed=False)
        # Two passes: the first warms the GPU clocks and the kernel caches; only the second is recorded.
        for pass_ in range(2):
          for rows in rows_list:
            ids_np = rng.integers(0, capacity, size=rows).astype(np.uint32)
            ids = mx.array(ids_np)
            xh = mx.array(rng.standard_normal((rows, IN)).astype(np.float32))
            # mxfp4 packed-lane kernel inputs: codes u32 [E, N, K/8], raw E8M0 scales [E, N, K/32] u8 (packed=False path)
            w4 = mx.array(rng.integers(0, 2**32 - 1, size=(capacity, OUT, IN // 8), dtype=np.uint32))
            sc = mx.array(rng.integers(100, 140, size=(capacity, OUT, IN // 32), dtype=np.uint8))
            dummy = mx.zeros((1,), dtype=mx.uint32)
            x4 = xh.astype(mx.bfloat16).reshape(rows, 1, 1, IN)
            ids32 = ids.astype(mx.int32)
            mx.eval(w4, sc, x4, ids32)

            def run_mx(carry):
                xin = x4 if carry is None else (x4 + carry[:, :1, :1, :1].astype(mx.bfloat16) * 0)
                return mx_kern(inputs=[xin, ids32, w4, sc, dummy, dummy, dummy], template=[("T", mx.bfloat16)],
                               grid=(32, OUT // 8, rows), threadgroup=(32, 2, 1),
                               output_shapes=[(rows, 1, 1, OUT)], output_dtypes=[mx.bfloat16])[0]

            m_mx, p_mx = timed(run_mx)
            g[f"rows{rows}_mxfp4_ms"] = m_mx
            print(f"{OUT}x{IN} rows={rows} mxfp4 packed-lane kernel: {m_mx:.3f} ms (p90 {p_mx:.3f})", flush=True)
            for variant in ("qmv", "tile"):
                kern = tk.make_tcq_projection(OUT, IN, variant)

                def run_tcq(carry, kern=kern, variant=variant):
                    xin = xh if carry is None else xh + carry[:, :1] * 0
                    return tk.run_tcq_projection(kern, variant, xin, ids, code, OUT, tables)

                m_t, p_t = timed(run_tcq)
                g[f"rows{rows}_tcq_{variant}_ms"] = m_t
                print(f"{OUT}x{IN} rows={rows} tcq {variant}: {m_t:.3f} ms (p90 {p_t:.3f}) = {m_t / m_mx:.2f}x mxfp4", flush=True)
        results["geometries"][f"{OUT}x{IN}"] = g
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=1)
        print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()
