#!/usr/bin/env python3
"""W9 full-depth "no bug" check: run the torch reference for ALL 40 backbone layers
on the 31-token probe as rung **R2** (our q8 dense + our Q2 records + reference
router) and compare each layer's attention output, MoE output and residual output
against the MLX full forward (mlx_dump_full.py).

A per-layer global/min cosine that stays high = the port is faithful at that layer;
a sharp drop -- especially at a structural seam (kv_source 2/8/14/20, Reindex
24/28/32/36, candidate source 20, engram 1/14) -- would localize a port bug in the
Reindex/Reuse/candidate/second-compressor paths.

Memory-bounded: each layer's reference module is built, loaded, run, then dropped
(its small cross-layer compress/index caches survive via the shared runtime);
Q2 experts are dequantized per use and never cached.  CPU/float32.
Writes torchref_ladder_full.json.
"""
from __future__ import annotations
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref_forward as RF
import ref_ladder as RL

torch.set_grad_enabled(False)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)

REPO = RF.REPO
SRC = RF.SRC
NPY = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w9"
DENSE = NPY / "mlx_dense"
MLXFULL = NPY / "mlx_full"
RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"
ARTIFACT = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2").expanduser()
SEAMS = {2: "kv_source+idx+2nd-compressor-start", 8: "kv_source+idx (2nd compressor)",
         14: "kv_source+idx+engram", 20: "kv_source+idx+candidate-source (ratio1)",
         24: "Reindex", 28: "Reindex", 32: "Reindex", 36: "Reindex", 1: "engram"}


class LazyDense:
    """dense[key] -> torch tensor loaded fresh from npy (caller holds it briefly)."""
    def __init__(self):
        self.index = json.loads((DENSE / "index.json").read_text())

    def __contains__(self, k):
        return k in self.index

    def __getitem__(self, k):
        return torch.from_numpy(np.load(DENSE / self.index[k]).astype(np.float32))


def main():
    RF._install_stub_modules()
    import model as refmodel
    import engram as refengram
    refmodel.world_size = 1
    refmodel.rank = 0
    refmodel.default_dtype = torch.float32

    cfg = json.loads((RF.REF_INFERENCE / "config.json").read_text())
    args = refmodel.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = 64          # 31 tokens; keeps the compress/window buffers tiny
    args.temperature = 0.0
    n_layers = args.n_layers

    shards = RF.Shards(SRC)
    ids = torch.tensor([RF.PROBE_IDS], dtype=torch.long)
    hashes, layout = RF.reference_row_ids(args, refengram, ids)
    engram_layer_ids = list(layout.layer_ids)
    dense = LazyDense()
    q2 = RL.Q2Bank(ARTIFACT)
    q2_nocache = True  # never accumulate experts across 40 layers

    hc = args.hc_mult
    embed_w = shards.dequant_weight("embed.weight")
    h = embed_w[ids[0]].unsqueeze(0).unsqueeze(2).repeat(1, 1, hc, 1)
    del embed_w
    pre_mix = refmodel.make_identity_pre_mix(h, hc)
    refmodel.shared_attn = refmodel.SharedAttentionRuntime()  # fresh cross-layer state

    mlx_index = json.loads((RECEIPTS / "mlx_layers_full.json").read_text())["layers"]
    rows = []
    t0 = time.time()
    for L in range(n_layers):
        if L in engram_layer_ids:
            lhi = engram_layer_ids.index(L)
            eng = RF.LazyEngram(shards, L, args, lhi)
            row_ids = np.asarray(hashes[:, :, lhi, :].cpu(), dtype=np.int64)
            h, _ = eng.forward(h, row_ids)
            del eng

        attn = refmodel.Attention(L, args)
        RL.load_attn(attn, L, refmodel, shards, dense)
        moe = RL.LadderMoE(refmodel, shards, dense, q2, L, args, "q2", None)
        if q2_nocache:
            q2._cache.clear()

        hc_w = {k: dense[f"L{L}/{k}"] for k in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                                                "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale")}
        an = refmodel.RMSNorm(args.dim, args.norm_eps); an.weight = torch.nn.Parameter(dense[f"L{L}/attn_norm"], requires_grad=False)
        fn = refmodel.RMSNorm(args.dim, args.norm_eps); fn.weight = torch.nn.Parameter(dense[f"L{L}/ffn_norm"], requires_grad=False)

        residual = h
        ap, apo, ac = RF.hc_mixes(h, hc_w["hc_attn_fn"], hc_w["hc_attn_scale"], hc_w["hc_attn_base"],
                                  args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
        attn_out = attn(an(RF.hc_pre(h, pre_mix)), 0)
        h = RF.hc_post(attn_out, residual, apo, ac)
        residual = h
        fp, fpo, fc = RF.hc_mixes(h, hc_w["hc_ffn_fn"], hc_w["hc_ffn_scale"], hc_w["hc_ffn_base"],
                                  args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
        moe_out, _ = moe.forward(fn(RF.hc_pre(h, ap)))
        h = RF.hc_post(moe_out, residual, fpo, fc)
        pre_mix = fp

        rec = {"layer": L, "seam": SEAMS.get(L)}
        for which, arr in (("attn_out", attn_out), ("moe_out", moe_out), ("layer_out", h)):
            a = arr.detach().to(torch.float32).numpy()
            b = np.load(MLXFULL / f"{which}_L{L}.npy")
            mn, g, mx = RL.cos_maxabs(b, a)
            rec[which] = {"min_cos": mn, "global_cos": g, "max_abs": mx, "finite": bool(np.isfinite(a).all())}
        rows.append(rec)
        flag = ""
        if rec["layer_out"]["global_cos"] < 0.999:
            flag = f"  <<< below 0.999 ({SEAMS.get(L, '')})"
        print(f"[full] L{L:>2} {SEAMS.get(L,''):<34} "
              f"attn g={rec['attn_out']['global_cos']:.5f} moe g={rec['moe_out']['global_cos']:.5f} "
              f"layer g={rec['layer_out']['global_cos']:.5f}/min={rec['layer_out']['min_cos']:.5f}{flag}")
        # drop this layer's heavy weights; shared_attn keeps the small cross-layer caches alive
        del attn, moe, hc_w, an, fn, attn_out, moe_out
        gc.collect()

    below = [r["layer"] for r in rows if r["layer_out"]["global_cos"] < 0.999]
    seam_below = [r["layer"] for r in rows if r["layer_out"]["global_cos"] < 0.999 and r["layer"] in SEAMS]
    report = {
        "probe_ids": RF.PROBE_IDS, "n_layers": n_layers, "rung": "R2 (q8 dense + Q2 experts + ref router)",
        "note": ("R2 is fp32-activation vs MLX bf16, so a gradual global-cos decline with depth is the "
                 "compounding dtype gap (proven ~0.995/op), NOT a bug. A port bug shows as a SHARP drop, "
                 "especially at a seam layer."),
        "seams": SEAMS,
        "layers": rows,
        "layers_below_0999_global": below,
        "seam_layers_below_0999": seam_below,
        "min_global_cos_layer_out": min(r["layer_out"]["global_cos"] for r in rows),
        "all_finite": all(r[w]["finite"] for r in rows for w in ("attn_out", "moe_out", "layer_out")),
    }
    out = RECEIPTS / "torchref_ladder_full.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nlayers with layer_out global_cos < 0.999: {below}")
    print(f"seam layers below 0.999: {seam_below}")
    print(f"min global_cos (layer_out) over all 40 = {report['min_global_cos_layer_out']:.5f}; all finite={report['all_finite']}")
    print(f"[full] DONE {time.time()-t0:.1f}s -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
