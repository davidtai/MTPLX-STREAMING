#!/usr/bin/env python3
"""Full-depth (all 40 backbone layers) MLX capture on the 31-token probe, CPU.

Records, per layer, the attention output (pre hc_post), the MoE output (pre
hc_post) and the layer residual output -- the tensors ref_ladder_full.py compares
against the torch R2 forward to flag any layer where the port diverges (the
Reindex/Reuse/candidate/second-compressor seams).  Experts stream through the
runtime; captures are ~150 MB total.  Writes mlx_full/*.npy + mlx_layers_full.json.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
mx.random.seed(0)
REPO = Path(__file__).resolve().parents[3]


def _force_worktree_mtplx():
    main_root = "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd"
    wt = str(REPO)
    if wt == main_root:
        return
    for fmod in list(sys.modules.values()):
        name = getattr(fmod, "__name__", "")
        if not (name.startswith("__editable__") and "mtplx" in name):
            continue
        for attr in ("MAPPING", "NAMESPACES"):
            d = getattr(fmod, attr, None)
            if isinstance(d, dict):
                for k, v in list(d.items()):
                    if isinstance(v, str) and v.startswith(main_root + "/mtplx"):
                        d[k] = v.replace(main_root, wt, 1)
                    elif isinstance(v, list):
                        d[k] = [p.replace(main_root, wt, 1) if isinstance(p, str) and p.startswith(main_root + "/mtplx") else p for p in v]
    sys.path.insert(0, wt)


_force_worktree_mtplx()

NPY = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w9" / "mlx_full"
RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"
MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()
GIB = 1024 ** 3
PROBE_IDS = [0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291, 6328, 3465, 1241,
             6036, 14, 291, 3395, 361, 1354, 260, 565, 291, 6328, 3465, 21740, 6036, 14, 291, 2605]


def save(name, arr):
    a = np.array(arr.astype(mx.float32), copy=True)
    NPY.mkdir(parents=True, exist_ok=True)
    np.save(NPY / f"{name}.npy", a)
    return {"shape": list(a.shape), "mean": float(a.mean()), "std": float(a.std()),
            "max_abs": float(np.abs(a).max()), "finite": bool(np.isfinite(a).all())}


def main():
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
    from mtplx.models import deepseek_v41 as M

    t0 = time.time()
    resident = load_deepseek_v41_streaming(
        MODEL, memory_limit_bytes=int(100 * GIB), max_live_kv_tokens=4096, admit=True,
        admission_receipt=None, expert_cache_limit_bytes=int(15 * GIB), apply_memory_cap=False,
        slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime")
    print(f"[mlxfull] loaded {time.time()-t0:.1f}s; layers={len(model.model.layers)}")

    caps = {}
    orig = M.DecoderLayer.__call__

    def cap(self, h, pre_mix, positions, layer_cache, shared):
        residual = h
        ap, apo, ac = self._mixes(h, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        x = M._rmsnorm(self._hc_pre(h, pre_mix), self.attn_norm_weight, self.norm_eps)
        attn_out = self.attn(x, positions, layer_cache, shared)
        h1 = M._hc_post_impl(attn_out, residual, apo, ac)
        residual2 = h1
        fp, fpo, fc = self._mixes(h1, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        moe_in = M._rmsnorm(self._hc_pre(h1, ap), self.ffn_norm_weight, self.norm_eps)
        moe_out = self.mlp(moe_in)
        h2 = M._hc_post_impl(moe_out, residual2, fpo, fc)
        caps[self.layer_id] = {"attn_out": attn_out, "moe_out": moe_out, "layer_out": h2}
        return h2, fp

    M.DecoderLayer.__call__ = cap
    try:
        cache = model.make_cache()
        _ = model.model(mx.array([PROBE_IDS]), cache)
        mx.eval(*[v for c in caps.values() for v in c.values()])
    finally:
        M.DecoderLayer.__call__ = orig

    index = {"meta": {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "mlx",
                      "device": "cpu", "prompt_ids": PROBE_IDS, "n_layers": len(caps),
                      "spec_key": runtime.spec.key}, "layers": {}}
    for L in sorted(caps):
        rec = {}
        for which in ("attn_out", "moe_out", "layer_out"):
            rec[which] = save(f"{which}_L{L}", caps[L][which])
        index["layers"][str(L)] = rec
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    (RECEIPTS / "mlx_layers_full.json").write_text(json.dumps(index, indent=2))
    fin = all(r[w]["finite"] for r in index["layers"].values() for w in ("attn_out", "moe_out", "layer_out"))
    print(f"[mlxfull] wrote {len(caps)} layers; all finite={fin}; {time.time()-t0:.1f}s")
    try:
        runtime.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
