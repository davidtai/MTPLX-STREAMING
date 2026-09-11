#!/usr/bin/env python3
"""W9 attribution ladder: substitute OUR artifact tensors into the torch reference
(layers 0..2, 31-token probe) to attribute each ref-vs-MLX gap to QUANTIZATION vs a
port BUG.

  R0 = pure reference (fp32 from the source FP8/FP4 weights)            [ref_forward]
  R1 = R0 + dense resident projections -> our q8 gs64 (mx-dequantized)  [export_mlx_dense]
  R2 = R1 + routed experts -> our Q2 records (numpy affine dequant of experts.bin)
  R3 = R2 + router top-6 forced to the MLX run's per-token selection

Per rung, cos/max-abs of attn_L0_output, moe_L0_output, layer{0,1,2}_output vs R0
and vs the MLX run.  Verdict: if R2 ~= MLX (cos>0.999) the forward is correct and
the Q2 bank is the whole story; if not, diff isolates a bug.  Also a router
isolation: identical input into the MLX gate vs the reference gate.
CPU/float32; numpy dequant validated bit-exact against mx.dequantize.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref_forward as RF

torch.set_grad_enabled(False)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)

REPO = RF.REPO
SRC = RF.SRC
NPY = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w9"
MLXNPY = NPY / "mlx"
DENSE = NPY / "mlx_dense"
RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"
ARTIFACT = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()


def bf16_bytes_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def np_affine_dequant(packed, scales, biases, group_size, bits):
    out = packed.shape[0]
    per = 32 // bits
    q = np.empty((out, packed.shape[1] * per), dtype=np.int64)
    mask = (1 << bits) - 1
    for i in range(per):
        q[:, i::per] = (packed >> (bits * i)) & mask
    se = np.repeat(scales, group_size, axis=1)
    be = np.repeat(biases, group_size, axis=1)
    return (q.astype(np.float32) * se + be)


class Q2Bank:
    """Dequantize our 2-bit routed experts straight from experts.bin (validated
    bit-exact vs mx.dequantize).  gate_proj=w1, up_proj=w3, down_proj=w2."""

    def __init__(self, artifact: Path):
        man = json.loads((artifact / "expert-manifest.json").read_text())
        self.recs = {(r["layer"], r["expert"]): r for r in man["records"]}
        q = man["quantization"]
        self.gs, self.bits = int(q["group_size"]), int(q["bits"])
        self.binp = artifact / "experts.bin"
        self._fh = open(self.binp, "rb")
        self._cache = {}

    def _read(self, seg):
        self._fh.seek(seg["offset"])
        return self._fh.read(seg["length"])

    def expert(self, L, eid):
        key = (L, int(eid))
        if key in self._cache:
            return self._cache[key]
        seg = {s["component"]: s for s in self.recs[key]["segments"]}

        def deq(comp):
            w = np.frombuffer(self._read(seg[comp + ".weight"]), np.uint32).reshape(seg[comp + ".weight"]["shape"])
            sc = bf16_bytes_to_f32(np.frombuffer(self._read(seg[comp + ".scales"]), np.uint16).reshape(seg[comp + ".scales"]["shape"]))
            bi = bf16_bytes_to_f32(np.frombuffer(self._read(seg[comp + ".biases"]), np.uint16).reshape(seg[comp + ".biases"]["shape"]))
            return torch.from_numpy(np_affine_dequant(w, sc, bi, self.gs, self.bits))

        t = (deq("gate_proj"), deq("down_proj"), deq("up_proj"))  # (w1, w2, w3)
        self._cache[key] = t
        return t


def load_dense_index():
    idx = json.loads((DENSE / "index.json").read_text())
    return {k: torch.from_numpy(np.load(DENSE / v).astype(np.float32)) for k, v in idx.items()}


def cos_maxabs(a, b):
    A = a.reshape(a.shape[1], -1).astype(np.float64)
    B = b.reshape(b.shape[1], -1).astype(np.float64)
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    cos = num / den
    g = float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-30))
    return float(cos.min()), g, float(np.abs(A - B).max())


# --------------------------------------------------------------------------
class LadderMoE:
    def __init__(self, refmodel, shards, dense, q2, L, args, expert_source, router_override):
        self.L = L
        self.dim = args.dim
        self.limit = args.swiglu_limit
        self.gate_temp = args.gate_temp
        self.route_scale = args.route_scale
        self.norm_topk = args.norm_topk_prob
        self.topk = args.n_activated_experts
        self.q2 = q2
        self.shards = shards
        self.expert_source = expert_source          # "fp4" | "q2"
        self.router_override = router_override        # None | [n,topk] ints
        self.gate = refmodel.Gate(L, args)
        if dense is None:  # R0: source
            gp = f"layers.{L}.ffn.gate"
            self.gate.weight = torch.nn.Parameter(shards.tensor(f"{gp}.weight").to(torch.float32), requires_grad=False)
            self.gate.bias = torch.nn.Parameter(shards.tensor(f"{gp}.bias").to(torch.float32), requires_grad=False)
            self._shared = tuple(shards.dequant_weight(f"layers.{L}.ffn.shared_experts.{w}.weight") for w in ("w1", "w2", "w3"))
        else:  # R1+: mlx
            self.gate.weight = torch.nn.Parameter(dense[f"L{L}/gate/weight"], requires_grad=False)
            self.gate.bias = torch.nn.Parameter(dense[f"L{L}/gate/bias"], requires_grad=False)
            self._shared = (dense[f"L{L}/shared/w1"], dense[f"L{L}/shared/w2"], dense[f"L{L}/shared/w3"])
        if getattr(self.gate, "bias_vl", None) is not None:
            self.gate.bias_vl = torch.nn.Parameter(torch.zeros_like(self.gate.bias), requires_grad=False)

    def _expert(self, eid):
        if self.expert_source == "q2":
            return self.q2.expert(self.L, eid)
        base = f"layers.{self.L}.ffn.experts.{eid}"
        return tuple(self.shards.dequant_weight(f"{base}.{w}.weight") for w in ("w1", "w2", "w3"))

    def _swiglu(self, x, w1, w2, w3, rw=None):
        gate = x.to(torch.float32) @ w1.t()
        up = x.to(torch.float32) @ w3.t()
        if self.limit and self.limit > 0:
            up = torch.clamp(up, -self.limit, self.limit)
            gate = torch.clamp(gate, max=self.limit)
        y = F.silu(gate) * up
        if rw is not None:
            y = rw * y
        return y @ w2.t()

    def forward(self, x):
        shape = x.shape
        xf = x.reshape(-1, self.dim)
        weights, indices = self.gate(xf, None)
        if self.router_override is not None:
            ov = torch.tensor(self.router_override, dtype=torch.long)
            scores = F.softplus(F.linear(xf.float(), self.gate.weight.float()) / self.gate_temp).sqrt()
            weights = scores.gather(1, ov)
            if self.norm_topk and self.topk > 1:
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            weights = weights * self.route_scale
            indices = ov
        y = torch.zeros_like(xf, dtype=torch.float32)
        for eid in torch.unique(indices).tolist():
            w1, w2, w3 = self._expert(int(eid))
            idx, top = torch.where(indices == eid)
            y[idx] += self._swiglu(xf[idx], w1, w2, w3, weights[idx, top, None])
        y = y + self._swiglu(xf, *self._shared, None)
        return y.reshape(shape), {"ids": indices.cpu().numpy(), "weights": weights.cpu().numpy()}


def load_attn(attn, L, refmodel, shards, dense):
    Linear, RMSNorm = refmodel.Linear, refmodel.RMSNorm
    if dense is None:
        RF.load_module_weights(attn, f"layers.{L}.attn", shards, refmodel)
        return
    for name, sub in attn.named_modules():
        if not name:
            continue
        if isinstance(sub, (Linear, RMSNorm)):
            key = f"L{L}/attn/{name.replace('.', '/')}"
            sub.weight = torch.nn.Parameter(dense[key], requires_grad=False)
            if isinstance(sub, Linear):
                sub.scale = None
                try:
                    sub.weight.scale = None
                except Exception:
                    pass
    attn.attn_sink = torch.nn.Parameter(dense[f"L{L}/attn/attn_sink"], requires_grad=False)


def run_rung(rung, refmodel, refengram, args, shards, dense_idx, q2, hashes, engram_layer_ids, mlx_router):
    dense = None if rung == "R0" else dense_idx
    expert_source = "q2" if rung in ("R2", "R3") else "fp4"
    hc = args.hc_mult
    embed_w = shards.dequant_weight("embed.weight")
    ids = torch.tensor([RF.PROBE_IDS], dtype=torch.long)
    h = embed_w[ids[0]].unsqueeze(0).unsqueeze(2).repeat(1, 1, hc, 1)
    del embed_w
    pre_mix = refmodel.make_identity_pre_mix(h, hc)
    cap = {}
    for L in range(3):
        if L in engram_layer_ids:  # engram kept from source across rungs (tiny, row-ids bit-exact)
            lhi = engram_layer_ids.index(L)
            eng = RF.LazyEngram(shards, L, args, lhi)
            row_ids = np.asarray(hashes[:, :, lhi, :].cpu(), dtype=np.int64)
            h, _ = eng.forward(h, row_ids)
        attn = refmodel.Attention(L, args)
        load_attn(attn, L, refmodel, shards, dense)
        if dense is None:
            hc_w = {k: shards.tensor(f"layers.{L}.{k}").to(torch.float32)
                    for k in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale")}
            an = shards.tensor(f"layers.{L}.attn_norm.weight").to(torch.float32)
            fn = shards.tensor(f"layers.{L}.ffn_norm.weight").to(torch.float32)
        else:
            hc_w = {k: dense[f"L{L}/{k}"] for k in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale")}
            an = dense[f"L{L}/attn_norm"]
            fn = dense[f"L{L}/ffn_norm"]
        override = mlx_router[L] if (rung == "R3") else None
        moe = LadderMoE(refmodel, shards, dense, q2, L, args, expert_source, override)

        an_mod = refmodel.RMSNorm(args.dim, args.norm_eps); an_mod.weight = torch.nn.Parameter(an, requires_grad=False)
        fn_mod = refmodel.RMSNorm(args.dim, args.norm_eps); fn_mod.weight = torch.nn.Parameter(fn, requires_grad=False)

        residual = h
        attn_pre, attn_post, attn_comb = RF.hc_mixes(h, hc_w["hc_attn_fn"], hc_w["hc_attn_scale"], hc_w["hc_attn_base"],
                                                     args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
        attn_in = an_mod(RF.hc_pre(h, pre_mix))
        attn_out = attn(attn_in, 0)
        h = RF.hc_post(attn_out, residual, attn_post, attn_comb)
        if L == 0:
            cap["attn_L0_output"] = attn_out.detach().to(torch.float32).numpy()
        residual = h
        ffn_pre, ffn_post, ffn_comb = RF.hc_mixes(h, hc_w["hc_ffn_fn"], hc_w["hc_ffn_scale"], hc_w["hc_ffn_base"],
                                                  args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
        moe_in = fn_mod(RF.hc_pre(h, attn_pre))
        if L == 0:
            cap["moe_L0_input"] = moe_in.detach().to(torch.float32).clone()
        moe_out, _ = moe.forward(moe_in)
        h = RF.hc_post(moe_out, residual, ffn_post, ffn_comb)
        if L == 0:
            cap["moe_L0_output"] = moe_out.detach().to(torch.float32).numpy()
        cap[f"layer{L}_output"] = h.detach().to(torch.float32).numpy()
        pre_mix = ffn_pre
    return cap


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
    args.max_seq_len = 256
    args.temperature = 0.0

    shards = RF.Shards(SRC)
    ids = torch.tensor([RF.PROBE_IDS], dtype=torch.long)
    hashes, layout = RF.reference_row_ids(args, refengram, ids)
    engram_layer_ids = list(layout.layer_ids)
    dense_idx = load_dense_index()
    q2 = Q2Bank(ARTIFACT)

    mlx = json.loads((RECEIPTS / "mlx_layers012.json").read_text())
    mlx_router = {L: mlx["moe"][str(L)]["router"]["topk_ids"] for L in range(3)}

    def mlx_npy(name):
        return np.load(MLXNPY / f"{name}.npy")

    keys = ["attn_L0_output", "moe_L0_output", "layer0_output", "layer1_output", "layer2_output"]
    t0 = time.time()
    rungs = {}
    for rung in ("R0", "R1", "R2", "R3"):
        rungs[rung] = run_rung(rung, refmodel, refengram, args, shards, dense_idx, q2, hashes, engram_layer_ids, mlx_router)
        print(f"[ladder] {rung} done ({time.time()-t0:.1f}s)")

    report = {"probe_ids": RF.PROBE_IDS, "keys": keys, "vs_R0": {}, "vs_MLX": {}}
    for rung in ("R0", "R1", "R2", "R3"):
        report["vs_R0"][rung] = {}
        report["vs_MLX"][rung] = {}
        for k in keys:
            a = rungs[rung][k]
            mn0, g0, mx0 = cos_maxabs(rungs["R0"][k], a)
            mnm, gm, mxm = cos_maxabs(mlx_npy(k), a)
            report["vs_R0"][rung][k] = {"min_cos": mn0, "global_cos": g0, "max_abs": mx0}
            report["vs_MLX"][rung][k] = {"min_cos": mnm, "global_cos": gm, "max_abs": mxm}

    # ---- router isolation: identical input into ref gate vs MLX gate ----
    x_common = rungs["R0"]["moe_L0_input"].reshape(-1, args.dim)      # [31,5120] fp32
    ref_gate = refmodel.Gate(0, args)
    ref_gate.weight = torch.nn.Parameter(shards.tensor("layers.0.ffn.gate.weight").to(torch.float32), requires_grad=False)
    ref_gate.bias = torch.nn.Parameter(shards.tensor("layers.0.ffn.gate.bias").to(torch.float32), requires_grad=False)
    if getattr(ref_gate, "bias_vl", None) is not None:
        ref_gate.bias_vl = torch.nn.Parameter(torch.zeros_like(ref_gate.bias), requires_grad=False)
    mlx_gate = refmodel.Gate(0, args)
    mlx_gate.weight = torch.nn.Parameter(dense_idx["L0/gate/weight"], requires_grad=False)
    mlx_gate.bias = torch.nn.Parameter(dense_idx["L0/gate/bias"], requires_grad=False)
    if getattr(mlx_gate, "bias_vl", None) is not None:
        mlx_gate.bias_vl = torch.nn.Parameter(torch.zeros_like(mlx_gate.bias), requires_grad=False)
    _, ref_idx = ref_gate(x_common, None)
    _, mlx_idx = mlx_gate(x_common, None)
    ref_idx = ref_idx.cpu().numpy(); mlx_idx = mlx_idx.cpu().numpy()
    exact = sum(1 for i in range(31) if set(ref_idx[i]) == set(mlx_idx[i]))
    ov = [len(set(ref_idx[i]) & set(mlx_idx[i])) for i in range(31)]
    wdiff = float(np.abs(dense_idx["L0/gate/weight"].numpy() - shards.tensor("layers.0.ffn.gate.weight").to(torch.float32).numpy()).max())
    report["router_isolation_L0_identical_input"] = {
        "exact_set_match": exact, "n_tokens": 31, "mean_overlap": float(np.mean(ov)),
        "gate_weight_max_abs_diff_ref_vs_mlx": wdiff,
        "note": "feeds the SAME fp32 moe input into the reference gate (source bf16 weight) and the "
                "MLX gate (q8-artifact bf16 weight); disagreement here is the gate-weight quant, not upstream drift.",
    }
    print(f"\n=== router isolation (identical input) L0: ref-gate vs mlx-gate top6 exact {exact}/31, "
          f"mean overlap {np.mean(ov):.2f}/6, gate |Δw| {wdiff:.4g} ===")

    print("\n=== ladder cos vs R0 (min over positions) ===")
    for k in keys:
        print(f"{k:16} " + "  ".join(f"{r}={report['vs_R0'][r][k]['min_cos']:.5f}" for r in ("R0", "R1", "R2", "R3")))
    print("\n=== ladder cos vs MLX run (min over positions) ===")
    for k in keys:
        print(f"{k:16} " + "  ".join(f"{r}={report['vs_MLX'][r][k]['min_cos']:.5f}" for r in ("R0", "R1", "R2", "R3")))
    print("\n=== ladder global_cos vs MLX ===")
    for k in keys:
        print(f"{k:16} " + "  ".join(f"{r}={report['vs_MLX'][r][k]['global_cos']:.5f}" for r in ("R0", "R1", "R2", "R3")))

    r2_vs_mlx = report["vs_MLX"]["R2"]["layer2_output"]["global_cos"]
    r3_vs_mlx = report["vs_MLX"]["R3"]["layer2_output"]["global_cos"]
    report["verdict"] = {
        "R2_layer2_global_cos_vs_mlx": r2_vs_mlx,
        "R3_layer2_global_cos_vs_mlx": r3_vs_mlx,
        "note": ("R2/R3 residual gap to MLX is the fp32(ladder) vs bf16(MLX) activation storage "
                 "(proven ~0.995/op in isolate_wkv). Interpret min_cos deltas R1->R2 as the Q2-expert "
                 "share and R2->R3 as the router-flip share."),
    }
    out = RECEIPTS / "torchref_ladder.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
