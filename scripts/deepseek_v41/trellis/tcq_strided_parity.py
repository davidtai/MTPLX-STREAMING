"""F39 GPU parity: the stride-aware tile kernel over a WHOLE-RECORD bank, on the F34 real sample projection.

Runs ONLY inside a guarded GPU window:
    gpu_window.sh <python> tcq_strided_parity.py <out.json>
(`--cpu-smoke` builds tables + constructs the kernels + the CPU reference at tiny shapes; no Metal.)

Two checks on the F34 down_proj sample (escha_code[0] [144,320,48], escha_rout[0] [5120]; IN=2304, OUT=5120):
  1. raw matmul: strided_tile( t128(x) ) == vendor decode_expert_weights -> dense matmul, to <= 1e-3 relative
     (same bar as F35; fp32 accumulation order differs).  The bank is a whole record per slot; the kernel reads the
     down code at word offset 4,428,288 within the 6,645,248-word record.
  2. full effective path: y = t128( t128(x) @ W_q ) * rout  ==  x @ effective_weight(W_q, rout), to <= 1e-3.
Also asserts the stride constants and that strided output == the contiguous-bank F35 tile kernel on the same code.
"""
import json
import sys
from pathlib import Path

import numpy as np

SMOKE = "--cpu-smoke" in sys.argv
import mlx.core as mx  # noqa: E402
if SMOKE:
    mx.set_default_device(mx.cpu)

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcq_kernels as tk           # noqa: E402
import tcq_runtime as R            # noqa: E402
import tcq_encode as enc           # noqa: E402
from tcq_kernel_check import warp_tables  # noqa: E402
from mtplx import eschamoe          # noqa: E402

F34_SAMPLE = "/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f34-trellis/dsv41_f34_L20E3_down_proj_full_beam.npz"


def build_record_bank(down_code: np.ndarray, capacity: int, seed: int) -> mx.array:
    """Whole-record int16 bank [capacity, TCQ3_RECORD_WORDS]; each slot's down segment holds the F34 code."""
    rng = np.random.default_rng(seed)
    bank = rng.integers(-32768, 32767, size=(capacity, tk.TCQ3_RECORD_WORDS), dtype=np.int16)
    off = tk.TCQ3_CODE_WORD_OFFSETS["down_proj"]
    flat = np.ascontiguousarray(down_code).reshape(-1)          # [144*320*48] = 2,211,840 words
    assert flat.size == tk.TCQ3_CODE_WORDS["down_proj"], flat.size
    bank[:, off:off + flat.size] = flat[None, :]
    return mx.array(bank)


def main() -> int:
    out_path = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    z = np.load(F34_SAMPLE)
    down_code = z["escha_code"][0]                              # [144,320,48] int16 (down: IN=2304, OUT=5120)
    rout = z["escha_rout"][0].astype(np.float32)               # [5120]
    OUT, IN = 5120, 2304
    tables = warp_tables()
    results = {"smoke": SMOKE, "mlx": mx.__version__, "sample": Path(F34_SAMPLE).name,
               "stride_words": tk.TCQ3_RECORD_WORDS, "down_offset_words": tk.TCQ3_CODE_WORD_OFFSETS["down_proj"]}

    # W_q reference (vendor bit-exact decode) and effective weight
    W_q = eschamoe.decode_expert_weights(mx.array(down_code).reshape(1, 144, 320, 48), 3)[0].astype(mx.float32)
    mx.eval(W_q)
    E_ref = enc.effective_weight(np.array(W_q), np.ones(IN, np.float32), rout)   # [IN, OUT]

    kern_strided = tk.make_tcq_projection_strided(OUT, IN, "down_proj")
    kern_tile = tk.make_tcq_projection(OUT, IN, "tile")

    if SMOKE:
        results["cpu_smoke"] = {"E_ref_checksum": float(np.abs(E_ref).sum()),
                                "kernels_constructed": True}
        print("smoke: reference + kernels constructed (no Metal)", flush=True)
        if out_path:
            json.dump(results, open(out_path, "w"), indent=1)
        return 0

    capacity, rows = 4, 18
    rng = np.random.default_rng(1)
    ids = mx.array(rng.integers(0, capacity, size=rows).astype(np.uint32))
    x = mx.array(rng.standard_normal((rows, IN)).astype(np.float32))
    bank = build_record_bank(down_code, capacity, seed=OUT)
    mx.eval(ids, x, bank)

    # 1. raw matmul parity: strided_tile(x) == x @ W_q (reference); also == the contiguous F35 tile kernel
    z_strided = tk.run_tcq_projection_strided(kern_strided, x, ids, bank, OUT, tables)
    # contiguous bank of just the down code, for the F35 tile kernel
    contig = mx.array(np.broadcast_to(np.ascontiguousarray(down_code)[None], (capacity, 144, 320, 48)).copy())
    z_tile = tk.run_tcq_projection(kern_tile, "tile", x, ids, contig, OUT, tables)
    ref_raw = mx.stack([x[r] @ W_q for r in range(rows)], axis=0)   # [rows, OUT]
    mx.eval(z_strided, z_tile, ref_raw)
    rel_raw = float(mx.sqrt(mx.sum((z_strided - ref_raw) ** 2) / mx.sum(ref_raw ** 2)))
    rel_vs_tile = float(mx.sqrt(mx.sum((z_strided - z_tile) ** 2) / mx.sum(z_tile ** 2)))
    results["raw_matmul_rel_err"] = rel_raw
    results["strided_vs_contiguous_tile_rel_err"] = rel_vs_tile

    # 2. full effective path: y = t128(t128(x) @ W_q) * rout  ==  x @ E_ref
    ops = R.TcqPackedOps.__new__(R.TcqPackedOps)               # avoid the mtplx swiglu import; use _project only
    ops.mx = mx
    ops.tables = tables
    ops.routs = {"down_proj": mx.array(np.broadcast_to(rout[None], (capacity, OUT)).copy())}
    ops.layer = 0
    ops.n_experts = capacity
    y = ops._project(x, ids, ids.astype(mx.int32), bank, kern_strided, OUT, "down_proj")
    ref_eff = mx.array(np.array(x) @ E_ref)
    mx.eval(y, ref_eff)
    rel_eff = float(mx.sqrt(mx.sum((y - ref_eff) ** 2) / mx.sum(ref_eff ** 2)))
    results["effective_path_rel_err"] = rel_eff

    ok = rel_raw <= 1e-3 and rel_eff <= 1e-3 and rel_vs_tile <= 1e-6
    results["pass"] = ok
    print(f"raw_matmul rel {rel_raw:.3e} | strided-vs-tile rel {rel_vs_tile:.3e} | effective rel {rel_eff:.3e} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    if out_path:
        json.dump(results, open(out_path, "w"), indent=1)
        print("WROTE", out_path, flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
