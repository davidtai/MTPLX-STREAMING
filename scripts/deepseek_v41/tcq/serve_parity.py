"""F39 GPU parity for the tcq3 SERVE decode (tcq/serve_install.py), on REAL records from the (partial) bank.

Runs ONLY inside a guarded GPU window:
    gpu_window.sh <python> serve_parity.py <out.json> [--records 0,1,191]
(`--cpu-smoke` builds tables + kernels + the CPU reference at tiny shapes; no Metal.)

Validates the served decode dispatch the way the server will run it: the F35 CONTIGUOUS tile kernel over the
per-projection component-bank code array (`bank.arrays[f"{proj}.code"]`, int16 [cap, IN/16, OUT/16, 48]) with the
routed slots as ``ids`` — this is what ``serve_install.run_component_bank_tcq3`` calls, NOT the packed-lane strided
kernel.  For a few REAL records read from the partial bank (built from progress.json), per projection:
    strided_tile( t128(x) )  vs  vendor decode_expert_weights -> dense matmul     (rel <= 1e-3, the F35 bar)
and the full effective path  y = t128( t128(x) @ W_q ) * rout  vs  x @ effective_weight(W_q, rout)  (rel <= 1e-3).
The rout is read per-slot from ``bank.arrays[f"{proj}.rout"]`` (real f16), exactly as the serve decode does.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SMOKE = "--cpu-smoke" in sys.argv
import mlx.core as mx  # noqa: E402
if SMOKE:
    mx.set_default_device(mx.cpu)

WT = Path(__file__).resolve().parents[3]                       # worktree root
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts" / "deepseek_v41" / "trellis"))
sys.path.insert(0, str(WT / "scripts" / "deepseek_v41"))
import tcq_kernels as tk            # noqa: E402
import tcq_runtime as R            # noqa: E402
import tcq_encode as enc           # noqa: E402
from tcq import serve_install as S  # noqa: E402
from tcq_kernel_check import warp_tables  # noqa: E402
from mtplx import eschamoe          # noqa: E402

REAL_BANK = "/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-tcq3"
PROJ_INOUT = {"gate_proj": (5120, 2304), "up_proj": (5120, 2304), "down_proj": (2304, 5120)}


def _records(indices):
    """Read real records from the partial bank (built from progress.json); returns per-index {proj: (code, rout)}."""
    import tempfile
    from tcq import build_partial_manifest as B
    out = tempfile.mkdtemp(prefix="tcq3-parity-")
    B.build(REAL_BANK, out, os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/expert-manifest.json"))  # noqa: E501
    man = R.read_tcq_manifest(str(Path(out) / "expert-manifest.json"))
    g = man.geometry
    recs = {}
    with open(Path(out) / "experts.bin", "rb") as fh:
        for idx in indices:
            rec = man.records[idx]
            fh.seek(R.record_base_offset(rec, g))
            seg = R.slice_record(fh.read(g.record_bytes), g)
            recs[idx] = {p: (np.array(seg[f"{p}.code"]), np.array(seg[f"{p}.rout"]).astype(np.float32)) for p in PROJ_INOUT}
    return recs


def main() -> int:
    out_path = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    idx_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--records=")), "0,1")
    indices = [int(i) for i in idx_arg.split(",")]
    tables = warp_tables()
    results = {"smoke": SMOKE, "mlx": mx.__version__, "records": indices, "kernel": "contiguous-tile (serve)"}

    if SMOKE:
        for proj, (IN, OUT) in PROJ_INOUT.items():
            tk.make_tcq_projection(OUT, IN, "tile")             # construct only, no Metal
        results["cpu_smoke"] = {"kernels_constructed": True}
        print("smoke: serve tile kernels constructed (no Metal)", flush=True)
        if out_path:
            json.dump(results, open(out_path, "w"), indent=1)
        return 0

    recs = _records(indices)
    rng = np.random.default_rng(1)
    worst = 0.0
    per = {}
    for idx, projs in recs.items():
        for proj, (code, rout) in projs.items():
            IN, OUT = PROJ_INOUT[proj]
            W_q = eschamoe.decode_expert_weights(mx.array(code).reshape(1, IN // 16, OUT // 16, 48), 3)[0].astype(mx.float32)
            cap = 4
            code_bank = mx.array(np.broadcast_to(np.ascontiguousarray(code)[None], (cap, IN // 16, OUT // 16, 48)).copy())
            rout_bank = mx.array(np.broadcast_to(rout[None], (cap, OUT)).copy())
            rows = 8
            ids = mx.array(rng.integers(0, cap, size=rows).astype(np.uint32))
            x = mx.array(rng.standard_normal((rows, IN)).astype(np.float32))
            mx.eval(W_q, code_bank, rout_bank, ids, x)
            # 1. raw matmul: tile( t128(x) ) vs t128(x) @ W_q
            kern = tk.make_tcq_projection(OUT, IN, "tile")
            xh = R._t128(mx, x)
            z = tk.run_tcq_projection(kern, "tile", xh, ids, code_bank, OUT, tables)
            ref_raw = xh @ W_q
            mx.eval(z, ref_raw)
            rel_raw = float(mx.sqrt(mx.sum((z - ref_raw) ** 2) / mx.sum(ref_raw ** 2)))
            # 2. full effective path via the serve wrapper vs x @ effective_weight
            y = S._tcq3_project(x, code_bank=code_bank, rout_bank=rout_bank, slot_ids=ids, out_dim=OUT, tables=tables)
            E_ref = enc.effective_weight(np.array(W_q), np.ones(IN, np.float32), rout)
            ref_eff = mx.stack([mx.array(np.array(x)[r] @ E_ref) for r in range(rows)], axis=0)
            mx.eval(y, ref_eff)
            rel_eff = float(mx.sqrt(mx.sum((y - ref_eff) ** 2) / mx.sum(ref_eff ** 2)))
            per[f"L?E{idx}.{proj}"] = {"raw_rel": rel_raw, "effective_rel": rel_eff}
            worst = max(worst, rel_raw, rel_eff)
            print(f"rec{idx} {proj}: raw {rel_raw:.3e} effective {rel_eff:.3e}", flush=True)
    results["per_projection"] = per
    results["worst_rel"] = worst
    results["pass"] = worst <= 1e-3
    print(f"WORST rel {worst:.3e} -> {'PASS' if worst <= 1e-3 else 'FAIL'}", flush=True)
    if out_path:
        json.dump(results, open(out_path, "w"), indent=1)
        print("WROTE", out_path, flush=True)
    return 0 if worst <= 1e-3 else 4


if __name__ == "__main__":
    raise SystemExit(main())
