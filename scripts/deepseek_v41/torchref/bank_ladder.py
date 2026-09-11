#!/usr/bin/env python3
"""W9 candidate-bank forward ladder: run the torch reference (layers 0..max, 31-token
probe) with the FP4-source routed experts requantized into each candidate format
(via mx.quantize -- authoritative, since numpy affine != mlx affine), and report the
per-layer MoE-output and layer-output cosine vs R0 (source fp4) plus the router top-6
set agreement vs R0.  Dense/engram/gate are held at the source across all formats, so
the ONLY variable is the routed-expert format -- isolating the quantization decision.

Runs in .venv-torchref (torch + mlx 0.32.2).  Formats: q2/q3/q4/q6 gs64, q4 gs32,
mxfp4 gs32.  Writes docs/deepseek-v41/receipts/torchref_bank_ladder.json.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
mx.set_default_device(mx.cpu)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref_forward as RF
import ref_ladder as RL

torch.set_grad_enabled(False)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)

REPO = RF.REPO
RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"
FORMATS = [("source", None, None, None), ("q2_gs64", 2, 64, "affine"), ("q3_gs64", 3, 64, "affine"),
           ("q4_gs64", 4, 64, "affine"), ("q4_gs32", 4, 32, "affine"), ("q6_gs64", 6, 64, "affine"),
           ("mxfp4_gs32", 4, 32, "mxfp4")]


class ExpertBank:
    """(L,eid) -> (w1,w2,w3) torch f32.  'source' = fp4 dequant; else requant via mx.quantize."""
    def __init__(self, shards, bits, gs, mode):
        self.shards = shards
        self.bits, self.gs, self.mode = bits, gs, mode

    def _reformat(self, w):  # w: torch f32 [out,in]
        if self.mode is None:
            return w
        q = mx.quantize(mx.array(w.numpy()), group_size=self.gs, bits=self.bits, mode=self.mode)
        deq = mx.dequantize(*q, group_size=self.gs, bits=self.bits, mode=self.mode)
        return torch.from_numpy(np.array(deq.astype(mx.float32)))

    def expert(self, L, eid):
        base = f"layers.{L}.ffn.experts.{eid}"
        return tuple(self._reformat(self.shards.dequant_weight(f"{base}.{w}.weight")) for w in ("w1", "w2", "w3"))


def run_forward(bank, refmodel, refengram, args, shards, hashes, engram_layer_ids, max_layer):
    hc = args.hc_mult
    limit = args.swiglu_limit
    embed_w = shards.dequant_weight("embed.weight")
    ids = torch.tensor([RF.PROBE_IDS], dtype=torch.long)
    h = embed_w[ids[0]].unsqueeze(0).unsqueeze(2).repeat(1, 1, hc, 1)
    del embed_w
    pre_mix = refmodel.make_identity_pre_mix(h, hc)
    refmodel.shared_attn = refmodel.SharedAttentionRuntime()
    caps = {}

    def swiglu(x, w1, w2, w3, rw=None):
        g = x.to(torch.float32) @ w1.t()
        u = x.to(torch.float32) @ w3.t()
        if limit and limit > 0:
            u = torch.clamp(u, -limit, limit); g = torch.clamp(g, max=limit)
        y = F.silu(g) * u
        if rw is not None:
            y = rw * y
        return y @ w2.t()

    for L in range(max_layer + 1):
        if L in engram_layer_ids:
            lhi = engram_layer_ids.index(L)
            eng = RF.LazyEngram(shards, L, args, lhi)
            h, _ = eng.forward(h, np.asarray(hashes[:, :, lhi, :].cpu(), dtype=np.int64))
            del eng
        attn = refmodel.Attention(L, args)
        RF.load_module_weights(attn, f"layers.{L}.attn", shards, refmodel)
        gate = refmodel.Gate(L, args)
        gate.weight = torch.nn.Parameter(shards.tensor(f"layers.{L}.ffn.gate.weight").to(torch.float32), requires_grad=False)
        gate.bias = torch.nn.Parameter(shards.tensor(f"layers.{L}.ffn.gate.bias").to(torch.float32), requires_grad=False)
        if getattr(gate, "bias_vl", None) is not None:
            gate.bias_vl = torch.nn.Parameter(torch.zeros_like(gate.bias), requires_grad=False)
        shared = tuple(shards.dequant_weight(f"layers.{L}.ffn.shared_experts.{w}.weight") for w in ("w1", "w2", "w3"))
        hc_w = {k: shards.tensor(f"layers.{L}.{k}").to(torch.float32)
                for k in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale")}
        an = refmodel.RMSNorm(args.dim, args.norm_eps); an.weight = torch.nn.Parameter(shards.tensor(f"layers.{L}.attn_norm.weight").to(torch.float32), requires_grad=False)
        fn = refmodel.RMSNorm(args.dim, args.norm_eps); fn.weight = torch.nn.Parameter(shards.tensor(f"layers.{L}.ffn_norm.weight").to(torch.float32), requires_grad=False)

        residual = h
        ap, apo, ac = RF.hc_mixes(h, hc_w["hc_attn_fn"], hc_w["hc_attn_scale"], hc_w["hc_attn_base"], args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
        attn_out = attn(an(RF.hc_pre(h, pre_mix)), 0)
        h = RF.hc_post(attn_out, residual, apo, ac)
        residual = h
        fp, fpo, fc = RF.hc_mixes(h, hc_w["hc_ffn_fn"], hc_w["hc_ffn_scale"], hc_w["hc_ffn_base"], args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
        moe_in = fn(RF.hc_pre(h, ap))
        xf = moe_in.reshape(-1, args.dim)
        weights, indices = gate(xf, None)
        y = torch.zeros_like(xf, dtype=torch.float32)
        for eid in torch.unique(indices).tolist():
            w1, w2, w3 = bank.expert(L, int(eid))
            idx, top = torch.where(indices == eid)
            y[idx] += swiglu(xf[idx], w1, w2, w3, weights[idx, top, None])
        moe_out = (y + swiglu(xf, *shared)).reshape(moe_in.shape)
        h = RF.hc_post(moe_out, residual, fpo, fc)
        pre_mix = fp
        caps[L] = {"moe_out": moe_out.detach().numpy(), "layer_out": h.detach().numpy(),
                   "router_ids": indices.cpu().numpy()}
        del attn, gate
    return caps


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-layer", type=int, default=2)
    a = ap.parse_args()

    RF._install_stub_modules()
    import model as refmodel
    import engram as refengram
    refmodel.world_size = 1; refmodel.rank = 0; refmodel.default_dtype = torch.float32
    cfg = json.loads((RF.REF_INFERENCE / "config.json").read_text())
    args = refmodel.ModelArgs(**cfg)
    args.max_batch_size = 1; args.max_seq_len = 64; args.temperature = 0.0

    shards = RF.Shards(RF.SRC)
    ids = torch.tensor([RF.PROBE_IDS], dtype=torch.long)
    hashes, layout = RF.reference_row_ids(args, refengram, ids)
    eids = list(layout.layer_ids)

    probe = json.loads((RECEIPTS / "bank_mx_probe.json").read_text()) if (RECEIPTS / "bank_mx_probe.json").is_file() else {"formats": {}}
    t0 = time.time()
    runs = {}
    for fname, bits, gs, mode in FORMATS:
        runs[fname] = run_forward(ExpertBank(shards, bits, gs, mode), refmodel, refengram, args, shards, hashes, eids, a.max_layer)
        print(f"[bank] {fname} forward done ({time.time()-t0:.1f}s)")

    R0 = runs["source"]
    report = {"probe_ids": RF.PROBE_IDS, "max_layer": a.max_layer, "expert_quality_vs_source": probe.get("formats", {}), "forward_vs_R0": {}}
    print("\n=== forward cos vs R0 (source fp4) ===")
    for fname, *_ in FORMATS:
        if fname == "source":
            continue
        rec = {}
        for L in range(a.max_layer + 1):
            mo_mn, mo_g, _ = RL.cos_maxabs(R0[L]["moe_out"], runs[fname][L]["moe_out"])
            lo_mn, lo_g, _ = RL.cos_maxabs(R0[L]["layer_out"], runs[fname][L]["layer_out"])
            rec[f"L{L}"] = {"moe_global_cos": mo_g, "moe_min_cos": mo_mn, "layer_global_cos": lo_g, "layer_min_cos": lo_mn}
        # router top-6 set agreement vs R0 at the deepest layer covered
        Lr = a.max_layer
        r0i = R0[Lr]["router_ids"]; fi = runs[fname][Lr]["router_ids"]
        exact = sum(1 for i in range(r0i.shape[0]) if set(r0i[i]) == set(fi[i]))
        ov = float(np.mean([len(set(r0i[i]) & set(fi[i])) for i in range(r0i.shape[0])]))
        rec["router_L%d_vs_R0" % Lr] = {"exact_set_match": exact, "n_tokens": int(r0i.shape[0]), "mean_overlap_of_6": ov}
        report["forward_vs_R0"][fname] = rec
        q = probe.get("formats", {}).get(fname, {})
        print(f"{fname:12} moe_g[L0..{a.max_layer}]=" + ",".join(f"{rec[f'L{L}']['moe_global_cos']:.5f}" for L in range(a.max_layer + 1))
              + f" | layer_g[L{a.max_layer}]={rec[f'L{a.max_layer}']['layer_global_cos']:.5f}"
              + f" | routerL{Lr} {exact}/{r0i.shape[0]} ov{ov:.2f}"
              + f" | q.cos_vs_src={q.get('mean_cos_vs_source','?')} bank={q.get('bank_GiB_40x384','?')}GiB")
    (RECEIPTS / "torchref_bank_ladder.json").write_text(json.dumps(report, indent=2))
    print(f"\n[bank] DONE {time.time()-t0:.1f}s -> torchref_bank_ladder.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
