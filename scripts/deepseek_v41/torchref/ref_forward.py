#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash TORCH REFERENCE forward for layers 0..2 (extensible), CPU/float32.

Ground truth for the MLX-port parity debug (W9).  Imports the *reference*
``inference/model.py`` + ``inference/engram.py`` classes and replaces ONLY the
CUDA kernels (``kernel.py``: act_quant / fp4_act_quant / fp8_gemm / fp4_gemm /
hc_split_sinkhorn / sparse_attn) with pure-torch float32 equivalents.  All
projection weights are dequantized from the *source* HF checkpoint shards
(``~/models/DeepSeek-V4.1-Flash-src``) to float32 and injected, so the reference
``linear()`` runs plain ``F.linear`` in float32 -- a clean full-precision oracle
(weights are exactly the on-disk fp8/fp4 values; activations carry no quant
noise).  Nothing touches the GPU; ``mx`` is never imported here.

Loads lazily: only the tensors layers 0..2 need, the routed experts the router
actually selects for the 31 probe tokens, and only the engram rows those tokens
hash to -- never a whole 95 GiB shard.

Outputs (see --help):
  * per-submodule goldens  docs/deepseek-v41/receipts/torchref_golden_<sub>_L<n>.json
  * per-layer residual dump docs/deepseek-v41/receipts/torchref_layers012.json
  * full float32 buffers    .benchmark-artifacts/deepseek-v41/w9/*.npy   (git-ignored)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)

REPO = Path(__file__).resolve().parents[3]
SRC = Path(os.path.expanduser("~/models/DeepSeek-V4.1-Flash-src"))
REF_INFERENCE = SRC / "inference"
RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"
NPY_DIR = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w9"

# the 31-token teacher-forced probe (BOS id 0 + the 30 encoded ids), from W8 decode_probe_out.json
PROBE_IDS = [0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291, 6328, 3465, 1241,
             6036, 14, 291, 3395, 361, 1354, 260, 565, 291, 6328, 3465, 21740, 6036, 14, 291, 2605]

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# pure-torch kernel replacements (float32)
# ---------------------------------------------------------------------------
def _k_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
    """No-op quantizer: keep full float32 (clean oracle).  inplace mutates nothing
    numerically; non-inplace returns (x, None) so reference linear() would fall to F.linear."""
    if inplace:
        return x
    return x, None


def _k_fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    if inplace:
        return x
    return x, None


def _dequant_fp8_block_t(w_fp8, scale_e8m0):
    """f32 = e4m3(w) * 2**(e8m0_scale-127) over (out_block, in_block) blocks."""
    W = w_fp8.to(torch.float32)
    S = scale_e8m0.to(torch.float32)  # torch decodes e8m0 -> 2**(byte-127)
    ob = W.shape[0] // S.shape[0]
    ib = W.shape[1] // S.shape[1]
    Sx = S.repeat_interleave(ob, 0).repeat_interleave(ib, 1)
    return W * Sx


def _k_fp8_gemm(a, a_s, b, b_s, scale_dtype=None, block_size=128):
    W = _dequant_fp8_block_t(b, b_s)
    return a.to(torch.float32) @ W.t()


def _dequant_fp4_t(w_i8, scale_e8m0):
    """Unpack fp4_e2m1 (2/byte, low nibble = logical col 2j) via FP4_TABLE, then
    * 2**(e8m0-127) per 32-col group.  w_i8: [out, in//2] (int8/uint8 bytes)."""
    u = w_i8.view(torch.uint8).to(torch.long)              # [out, in//2]
    low = u & 0x0F
    high = (u >> 4) & 0x0F
    vals = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1)  # [out, in//2, 2]
    vals = vals.reshape(u.shape[0], u.shape[1] * 2)               # [out, in]
    S = scale_e8m0.to(torch.float32)                             # [out, in//32]
    Sx = S.repeat_interleave(vals.shape[1] // S.shape[1], dim=1)
    return vals * Sx


def _k_fp4_gemm(a, a_s, b, b_s, scale_dtype=None, act_block_size=128):
    W = _dequant_fp4_t(b, b_s)
    return a.to(torch.float32) @ W.t()


def _k_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    lead = list(mixes.shape[:-1])
    m = mixes.reshape(-1, mixes.shape[-1]).to(torch.float32)
    hc_scale = hc_scale.to(torch.float32)
    hc_base = hc_base.to(torch.float32)
    pre = torch.sigmoid(m[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[:, 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).reshape(-1, hc, hc)
    # comb = softmax(-1) + eps
    comb = comb - comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb)
    comb = comb / comb.sum(dim=-1, keepdim=True)
    comb = comb + eps
    # first col normalization: / (sum over dim -2 + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    pre = pre.reshape(*lead, hc)
    post = post.reshape(*lead, hc)
    comb = comb.reshape(*lead, hc, hc)
    return pre, post, comb


def _k_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Reference sparse_attn semantics, full float32.  q:[b,m,h,d] kv:[b,n,d]
    attn_sink:[h] topk_idxs:[b,m,topk] (int, -1 invalid).  Sink is a value-0 slot
    contributing only to the denominator; all-invalid rows -> zero output."""
    b, m, h, d = q.shape
    n = kv.shape[1]
    idx = topk_idxs.to(torch.long)                         # [b,m,topk]
    valid = idx != -1
    idxc = idx.clamp_min(0)
    kv_e = kv.unsqueeze(1).expand(b, m, n, d)
    kvg = torch.gather(kv_e, 2, idxc.unsqueeze(-1).expand(b, m, idx.shape[-1], d)).to(torch.float32)
    scores = torch.einsum("bmhd,bmtd->bmht", q.to(torch.float32), kvg) * softmax_scale
    vmask = valid.unsqueeze(2)                             # [b,m,1,topk]
    scores = scores.masked_fill(~vmask, float("-inf"))
    mx_ = scores.max(dim=-1, keepdim=True).values          # [b,m,h,1]
    mx_ = torch.clamp_min(mx_, -1e30)
    ex = torch.exp(scores - mx_)
    ex = ex.masked_fill(~vmask, 0.0)
    denom = ex.sum(-1) + torch.exp(attn_sink.to(torch.float32).view(1, 1, h) - mx_.squeeze(-1))
    o = torch.einsum("bmht,bmtd->bmhd", ex, kvg) / denom.unsqueeze(-1)
    return o


def _install_stub_modules():
    kernel = types.ModuleType("kernel")
    kernel.act_quant = _k_act_quant
    kernel.fp4_act_quant = _k_fp4_act_quant
    kernel.fp8_gemm = _k_fp8_gemm
    kernel.fp4_gemm = _k_fp4_gemm
    kernel.hc_split_sinkhorn = _k_hc_split_sinkhorn
    kernel.sparse_attn = _k_sparse_attn
    sys.modules["kernel"] = kernel

    vision = types.ModuleType("vision")
    vision.Aligner = object
    vision.ViT = object
    sys.modules["vision"] = vision

    ip = types.ModuleType("image_processor")
    ip.IMAGE, ip.IMAGE_END, ip.IMAGE_NEW_LINE, ip.IMAGE_START = 0, 1, 2, 3
    ip.TEXT = -1
    ip.prepare_vl_inputs = None
    sys.modules["image_processor"] = ip

    sys.path.insert(0, str(REF_INFERENCE))


# ---------------------------------------------------------------------------
# lazy safetensors loader for the source shards
# ---------------------------------------------------------------------------
class Shards:
    def __init__(self, src: Path):
        self.src = src
        idx = json.loads((src / "model.safetensors.index.json").read_text())
        self.weight_map = idx["weight_map"]
        self._handles = {}
        self._headers = {}

    def _open(self, shard):
        if shard not in self._handles:
            from safetensors import safe_open
            self._handles[shard] = safe_open(str(self.src / shard), framework="pt", device="cpu")
        return self._handles[shard]

    def _header(self, shard):
        if shard not in self._headers:
            with open(self.src / shard, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._headers[shard] = json.loads(fh.read(n))
        return self._headers[shard]

    def has(self, key):
        return key in self.weight_map

    def dtype(self, key):
        shard = self.weight_map[key]
        return self._header(shard)[key]["dtype"]

    def tensor(self, key):
        return self._open(self.weight_map[key]).get_tensor(key)

    def slice(self, key):
        return self._open(self.weight_map[key]).get_slice(key)

    def dequant_weight(self, key):
        """Load ``key`` as float32, dequantizing fp8-block weights via their .scale."""
        dt = self.dtype(key)
        w = self.tensor(key)
        if dt in ("BF16", "F16", "F32"):
            return w.to(torch.float32)
        if dt == "F8_E4M3":
            scale = self.tensor(key.replace(".weight", ".scale"))
            return _dequant_fp8_block_t(w, scale)
        if dt == "I8":  # packed fp4 expert weight
            scale = self.tensor(key.replace(".weight", ".scale"))
            return _dequant_fp4_t(w, scale)
        raise ValueError(f"unhandled dtype {dt} for {key}")


# ---------------------------------------------------------------------------
# capture / summarize helpers
# ---------------------------------------------------------------------------
def _sha_f32(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def summarize(t, name, save_full=True, per_pos=True):
    """Compact JSON-able summary of a tensor + save its full float32 buffer to .npy.

    For [1, S, ...] tensors records per-position first-64 flattened values and L2
    norm; always records mean/std/max-abs/finite and the sha256 of the full f32 buffer."""
    a = t.detach().to(torch.float32).contiguous().cpu().numpy()
    flat = a.reshape(-1)
    d = {
        "name": name,
        "shape": list(a.shape),
        "dtype": "float32",
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "max_abs": float(np.abs(flat).max()) if flat.size else 0.0,
        "finite": bool(np.isfinite(flat).all()),
        "sha256_f32": _sha_f32(a),
    }
    if per_pos and a.ndim >= 2 and a.shape[0] == 1:
        seq = a.shape[1]
        pp = a.reshape(1, seq, -1)[0]
        d["per_pos_feat_len"] = int(pp.shape[1])
        d["per_pos_first64"] = [[float(x) for x in pp[p, :64]] for p in range(seq)]
        d["per_pos_norm"] = [float(np.linalg.norm(pp[p])) for p in range(seq)]
        d["first64_last_pos"] = [float(x) for x in pp[-1, :64]]
    if save_full:
        NPY_DIR.mkdir(parents=True, exist_ok=True)
        np.save(NPY_DIR / f"{name}.npy", a)
        d["npy"] = f"{name}.npy"
    return d


def _stats_only(t):
    a = t.detach().to(torch.float32).contiguous().cpu().numpy().reshape(-1)
    return {
        "mean": float(a.mean()), "std": float(a.std()),
        "max_abs": float(np.abs(a).max()) if a.size else 0.0,
        "finite": bool(np.isfinite(a).all()),
        "first64_last_pos": None,
    }


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"[ref] wrote {path.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# weight injection into constructed reference modules
# ---------------------------------------------------------------------------
def load_module_weights(module, prefix, shards, refmodel):
    """Reassign every Linear/RMSNorm/bare-param under ``module`` to float32
    dequantized tensors from disk under ``prefix`` (e.g. 'layers.0.attn')."""
    Linear = refmodel.Linear
    RMSNorm = refmodel.RMSNorm
    for name, sub in module.named_modules():
        path = f"{prefix}.{name}" if name else prefix
        if isinstance(sub, Linear):
            w = shards.dequant_weight(f"{path}.weight")
            sub.weight = torch.nn.Parameter(w, requires_grad=False)
            try:
                sub.weight.scale = None
            except Exception:
                pass
            sub.scale = None
            if getattr(sub, "bias", None) is not None and shards.has(f"{path}.bias"):
                sub.bias = torch.nn.Parameter(shards.tensor(f"{path}.bias").to(torch.float32), requires_grad=False)
        elif isinstance(sub, RMSNorm):
            sub.weight = torch.nn.Parameter(shards.tensor(f"{path}.weight").to(torch.float32), requires_grad=False)
    # bare parameters directly on the module (attn_sink, k_norm-as-RMSNorm handled above)
    for pname, _ in module.named_parameters(recurse=False):
        key = f"{prefix}.{pname}"
        if shards.has(key):
            setattr(module, pname, torch.nn.Parameter(shards.tensor(key).to(torch.float32), requires_grad=False))


# ---------------------------------------------------------------------------
# block-level hyper-connection ops (reference Block.hc_* transcribed)
# ---------------------------------------------------------------------------
def hc_mixes(x, hc_fn, hc_scale, hc_base, norm_eps, hc_mult, iters, eps):
    x2 = x.flatten(2).to(torch.float32)
    rsqrt = torch.rsqrt(x2.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x2, hc_fn.to(torch.float32)) * rsqrt
    return _k_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)


def hc_pre(x, pre_mix):
    y = torch.sum(pre_mix.unsqueeze(-1) * x.to(torch.float32), dim=2)
    return y.to(x.dtype)


def hc_post(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)


# ---------------------------------------------------------------------------
# lazy engram (layer 1): manual replication of reference Engram.forward
# ---------------------------------------------------------------------------
class LazyEngram:
    def __init__(self, shards, layer_id, args, layer_hash_index):
        self.layer_id = layer_id
        self.layer_hash_index = layer_hash_index
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.head_dim = args.engram_head_dim
        self.eps = args.norm_eps
        self.clamp_value = 1e-6
        base = f"layers.{layer_id}.engram"
        self.wkv_w = shards.dequant_weight(f"{base}.wkv.weight")            # [dim*(hc+1), n_hash_cols*head_dim] f32
        self.q_weight = shards.tensor(f"{base}.q_weight").to(torch.float32)  # [hc, dim]
        self.k_weight = shards.tensor(f"{base}.k_weight").to(torch.float32)
        self.embed_w_slice = shards.slice(f"{base}.embed.weight")           # [rows,256] fp8
        self.embed_s_slice = shards.slice(f"{base}.embed.scale")            # [rows,8] e8m0
        self._row_cache = {}

    def _embed_rows(self, ids_flat):
        out = np.empty((len(ids_flat), self.head_dim), dtype=np.float32)
        for i, rid in enumerate(ids_flat):
            rid = int(rid)
            v = self._row_cache.get(rid)
            if v is None:
                w = self.embed_w_slice[rid].to(torch.float32)              # [256]
                s = self.embed_s_slice[rid].to(torch.float32)              # [8]
                v = (w.reshape(-1, 32) * s.reshape(-1, 1)).reshape(-1)     # per 32-col group
                v = v.numpy()
                self._row_cache[rid] = v
            out[i] = v
        return out

    def forward(self, x, hash_ids):
        """x: [B,L,hc,dim]; hash_ids: numpy [B,L,n_hash_cols].  Returns (updated x, capture dict)."""
        hash_ids = np.asarray(hash_ids, dtype=np.int64)
        B, L, cols = hash_ids.shape
        rows = self._embed_rows(hash_ids.reshape(-1))                       # [B*L*cols, 256]
        embed = torch.from_numpy(rows).reshape(B, L, cols, self.head_dim)
        kv = embed.reshape(B, L, -1).to(torch.float32) @ self.wkv_w.t()     # [B,L,dim*(hc+1)]
        split = self.hc_mult * self.dim
        key = kv[..., :split].reshape(B, L, self.hc_mult, self.dim)
        value = kv[..., split:]
        h = x.to(torch.float32)
        weight = self.q_weight * self.k_weight
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim ** -0.5          # [B,L,hc]
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        out = (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(x.dtype)
        cap = {
            "kv": kv, "key": key, "value": value, "dot": dot, "gate": gate,
            "embed": embed,
        }
        return out, cap


# ---------------------------------------------------------------------------
# lazy MoE (reference MoE.forward + Expert.forward, experts loaded on demand)
# ---------------------------------------------------------------------------
class LazyMoE:
    def __init__(self, shards, layer_id, args, refmodel):
        self.shards = shards
        self.layer_id = layer_id
        self.dim = args.dim
        self.limit = args.swiglu_limit
        self.gate = refmodel.Gate(layer_id, args)
        load_module_weights(self.gate, f"layers.{layer_id}.ffn.gate", shards, refmodel)
        # Gate stores weight/bias/bias_vl as bare params:
        gp = f"layers.{layer_id}.ffn.gate"
        self.gate.weight = torch.nn.Parameter(shards.tensor(f"{gp}.weight").to(torch.float32), requires_grad=False)
        self.gate.bias = torch.nn.Parameter(shards.tensor(f"{gp}.bias").to(torch.float32), requires_grad=False)
        if shards.has(f"{gp}.bias_vl"):
            self.gate.bias_vl = torch.nn.Parameter(shards.tensor(f"{gp}.bias_vl").to(torch.float32), requires_grad=False)
        self._expert_cache = {}

    def _expert(self, eid):
        if eid not in self._expert_cache:
            base = f"layers.{self.layer_id}.ffn.experts.{eid}"
            w1 = self.shards.dequant_weight(f"{base}.w1.weight")
            w2 = self.shards.dequant_weight(f"{base}.w2.weight")
            w3 = self.shards.dequant_weight(f"{base}.w3.weight")
            self._expert_cache[eid] = (w1, w2, w3)
        return self._expert_cache[eid]

    def _shared(self):
        if not hasattr(self, "_shared_w"):
            base = f"layers.{self.layer_id}.ffn.shared_experts"
            self._shared_w = (
                self.shards.dequant_weight(f"{base}.w1.weight"),
                self.shards.dequant_weight(f"{base}.w2.weight"),
                self.shards.dequant_weight(f"{base}.w3.weight"),
            )
        return self._shared_w

    def _swiglu(self, x, w1, w2, w3, route_w=None):
        gate = x.to(torch.float32) @ w1.t()
        up = x.to(torch.float32) @ w3.t()
        if self.limit and self.limit > 0:
            up = torch.clamp(up, -self.limit, self.limit)
            gate = torch.clamp(gate, max=self.limit)
        y = F.silu(gate) * up
        if route_w is not None:
            y = route_w * y
        return y @ w2.t()

    def forward(self, x):
        shape = x.shape
        xf = x.reshape(-1, self.dim)
        weights, indices = self.gate(xf, None)                              # [n,topk] each
        y = torch.zeros_like(xf, dtype=torch.float32)
        sel = torch.unique(indices).tolist()
        for eid in sel:
            w1, w2, w3 = self._expert(int(eid))
            mask = indices == eid
            idx, top = torch.where(mask)
            xe = xf[idx]
            rw = weights[idx, top, None]
            y[idx] += self._swiglu(xe, w1, w2, w3, rw)
        sw1, sw2, sw3 = self._shared()
        shared_out = self._swiglu(xf, sw1, sw2, sw3, None)
        y = y + shared_out
        cap = {
            "router_ids": indices.to(torch.int64).cpu().numpy(),            # [n,topk]
            "router_weights": weights.to(torch.float32).cpu().numpy(),      # [n,topk]
            "shared_out": shared_out.reshape(shape),
        }
        return y.reshape(shape), cap


# ---------------------------------------------------------------------------
# reference engram row ids (ground truth via reference engram.NgramHashState)
# ---------------------------------------------------------------------------
def reference_row_ids(args, refengram, ids_t):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    layout = refengram.EngramLayout.from_args(args)
    hashstate = refengram.NgramHashState(args, layout, tok)
    hashes = hashstate(ids_t, 0, None)             # [B,L,n_engram_layers,n_hash_cols]
    return hashes, layout


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-layer", type=int, default=2, help="run layers 0..max-layer (default 2)")
    ap.add_argument("--ids", type=str, default=None, help="comma-separated token ids (default: 31-token probe)")
    args_cli = ap.parse_args(argv)

    _install_stub_modules()
    import model as refmodel
    import engram as refengram

    refmodel.world_size = 1
    refmodel.rank = 0
    refmodel.default_dtype = torch.float32  # every Linear(dtype=None) => float32 => F.linear

    cfg = json.loads((REF_INFERENCE / "config.json").read_text())
    args = refmodel.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = 256
    args.temperature = 0.0

    ids_list = PROBE_IDS if args_cli.ids is None else [int(x) for x in args_cli.ids.split(",")]
    ids_t = torch.tensor([ids_list], dtype=torch.long)
    B, L = ids_t.shape
    hc = args.hc_mult
    print(f"[ref] {L} tokens, layers 0..{args_cli.max_layer}, hc_mult={hc}, dim={args.dim}")

    shards = Shards(SRC)

    t0 = time.time()
    layout = refengram.EngramLayout.from_args(args)
    engram_layer_ids = list(layout.layer_ids)
    hashes = None
    if any(l <= args_cli.max_layer for l in engram_layer_ids):
        hashes, layout = reference_row_ids(args, refengram, ids_t)
        print(f"[ref] engram row ids computed ({time.time()-t0:.1f}s); engram layers {engram_layer_ids}")
    else:
        print(f"[ref] skipping engram row ids (no engram layer <= {args_cli.max_layer})")

    # embedding
    embed_w = shards.dequant_weight("embed.weight")   # bf16->f32 [vocab, dim]
    h = embed_w[ids_t[0]].unsqueeze(0)                # [1,L,dim]
    del embed_w
    h = h.unsqueeze(2).repeat(1, 1, hc, 1)            # [1,L,hc,dim]
    pre_mix = refmodel.make_identity_pre_mix(h, hc)

    # -- per-submodule + per-layer captures ---------------------------------
    per_layer = []
    embed_summ = summarize(h, "embed_stream", per_pos=True)

    for lid in range(args_cli.max_layer + 1):
        layer_cap = {"layer": lid}
        # engram (reference applies it BEFORE the block on engram layers)
        if lid in engram_layer_ids:
            lhi = engram_layer_ids.index(lid)
            eng = LazyEngram(shards, lid, args, lhi)
            pre_engram = h
            row_ids = np.asarray(hashes[:, :, lhi, :].cpu(), dtype=np.int64)   # [B,L,cols]
            h, ecap = eng.forward(h, row_ids)
            emit_engram_golden(lid, args, pre_engram, h, ecap, row_ids)
            layer_cap["engram"] = {
                "pre": _stats_only(pre_engram), "post": _stats_only(h),
                "gate_mean": float(ecap["gate"].mean().item()),
                "gate_max": float(ecap["gate"].max().item()),
                "dot_max_abs": float(ecap["dot"].abs().max().item()),
                "value_max_abs": float(ecap["value"].abs().max().item()),
            }

        # ---- Block.forward (transcribed), with sublayer captures ----
        attn = refmodel.Attention(lid, args)
        load_module_weights(attn, f"layers.{lid}.attn", shards, refmodel)
        moe = LazyMoE(shards, lid, args, refmodel)
        hc_attn_fn = shards.tensor(f"layers.{lid}.hc_attn_fn").to(torch.float32)
        hc_attn_base = shards.tensor(f"layers.{lid}.hc_attn_base").to(torch.float32)
        hc_attn_scale = shards.tensor(f"layers.{lid}.hc_attn_scale").to(torch.float32)
        hc_ffn_fn = shards.tensor(f"layers.{lid}.hc_ffn_fn").to(torch.float32)
        hc_ffn_base = shards.tensor(f"layers.{lid}.hc_ffn_base").to(torch.float32)
        hc_ffn_scale = shards.tensor(f"layers.{lid}.hc_ffn_scale").to(torch.float32)
        attn_norm_w = shards.tensor(f"layers.{lid}.attn_norm.weight").to(torch.float32)
        ffn_norm_w = shards.tensor(f"layers.{lid}.ffn_norm.weight").to(torch.float32)

        attn_norm_mod = refmodel.RMSNorm(args.dim, args.norm_eps)
        attn_norm_mod.weight = torch.nn.Parameter(attn_norm_w, requires_grad=False)
        ffn_norm_mod = refmodel.RMSNorm(args.dim, args.norm_eps)
        ffn_norm_mod.weight = torch.nn.Parameter(ffn_norm_w, requires_grad=False)

        residual = h
        attn_pre, attn_post, attn_comb = hc_mixes(h, hc_attn_fn, hc_attn_scale, hc_attn_base,
                                                  args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
        attn_in_normed = attn_norm_mod(hc_pre(h, pre_mix))
        attn_out = attn(attn_in_normed, 0)
        h = hc_post(attn_out, residual, attn_post, attn_comb)
        emit_attn_golden(lid, args, attn_in_normed, attn_out, attn)

        residual = h
        ffn_pre, ffn_post, ffn_comb = hc_mixes(h, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                                               args.norm_eps, hc, args.hc_sinkhorn_iters, args.hc_eps)
        moe_in_normed = ffn_norm_mod(hc_pre(h, attn_pre))
        moe_out, mcap = moe.forward(moe_in_normed)
        h = hc_post(moe_out, residual, ffn_post, ffn_comb)
        emit_moe_golden(lid, moe_in_normed, moe_out, mcap)

        pre_mix = ffn_pre
        layer_summ = summarize(h, f"layer{lid}_output", per_pos=True)
        layer_summ["layer"] = lid
        per_layer.append(layer_summ)
        print(f"[ref] layer {lid} done  out mean={layer_summ['mean']:.4g} "
              f"std={layer_summ['std']:.4g} max_abs={layer_summ['max_abs']:.4g} finite={layer_summ['finite']}")

    # aggregate per-layer receipt (compare_hidden_states.py compatible superset)
    dump = {
        "meta": {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "torchref",
            "device": "cpu",
            "dtype": "float32",
            "git_rev": None,
            "engram_ablate": False,
            "prompt_ids": ids_list,
            "n_layers_covered": args_cli.max_layer + 1,
            "engram_layer_ids": engram_layer_ids,
            "ref_src": str(SRC),
        },
        "embed": embed_summ,
        "layers": [
            {"layer": r["layer"], "mean": r["mean"], "std": r["std"], "max_abs": r["max_abs"],
             "finite": r["finite"], "first64_last_pos": r["first64_last_pos"], "npy": r.get("npy")}
            for r in per_layer
        ],
        "engram": {},          # populated by golden files; kept for compare compat
        "final_norm": None,
        "argmax_token": None,
        "logits_top8": [],
    }
    write_json(RECEIPTS / "torchref_layers012.json", dump)
    print(f"[ref] DONE in {time.time()-t0:.1f}s")
    return 0


def emit_attn_golden(lid, args, attn_in_normed, attn_out, attn):
    L = attn_in_normed.shape[1]
    ratio = args.compress_ratios[lid]
    g = {
        "submodule": "attention",
        "layer": lid,
        "mode": "SWA_only" if ratio == 0 else "Full/compress",
        "input_post_hc_norm": summarize(attn_in_normed, f"attn_L{lid}_input"),
        "output": summarize(attn_out, f"attn_L{lid}_output"),
        "window_kv": summarize(attn.window_kv_cache[:1, :L], f"attn_L{lid}_window_kv"),
    }
    if ratio != 0:
        n_comp = L // ratio
        shared = sys.modules["model"].shared_attn
        if shared.compress_kv is not None:
            g["compressed_kv"] = summarize(shared.compress_kv[:1, :n_comp], f"attn_L{lid}_compressed_kv")
        if shared.index_k is not None:
            g["index_k"] = summarize(shared.index_k[:1, :n_comp], f"attn_L{lid}_index_k")
        if shared.topk_idxs is not None:
            ti = shared.topk_idxs[:1].to(torch.int64).cpu().numpy()
            g["topk_idxs"] = {"shape": list(ti.shape), "dtype": "int64", "values": ti.tolist(),
                              "note": "per query: selected compressed-row ids (offset by window length; -1 = none)"}
    write_json(RECEIPTS / f"torchref_golden_attn_L{lid}.json", g)


def emit_moe_golden(lid, moe_in, moe_out, mcap):
    g = {
        "submodule": "moe",
        "layer": lid,
        "input": summarize(moe_in, f"moe_L{lid}_input"),
        "output": summarize(moe_out, f"moe_L{lid}_output"),
        "shared_expert_output": summarize(mcap["shared_out"], f"moe_L{lid}_shared_out"),
        "router": {
            "topk_ids": mcap["router_ids"].tolist(),
            "topk_weights": mcap["router_weights"].tolist(),
            "note": "per token (row) top-6 routed expert ids + normalized*route_scale weights",
        },
    }
    write_json(RECEIPTS / f"torchref_golden_moe_L{lid}.json", g)


def emit_engram_golden(lid, args, pre, post, ecap, row_ids):
    g = {
        "submodule": "engram",
        "layer": lid,
        "input_pre_engram": summarize(pre, f"engram_L{lid}_input"),
        "output_post_engram": summarize(post, f"engram_L{lid}_output"),
        "gate": summarize(ecap["gate"], f"engram_L{lid}_gate"),
        "value": summarize(ecap["value"], f"engram_L{lid}_value"),
        "row_ids": {
            "shape": list(row_ids.shape),
            "dtype": "int64",
            "note": f"[B,L,n_hash_cols] ; n_hash_cols=(max_ngram-1)*n_heads="
                    f"{(args.engram_max_ngram_size-1)*args.engram_n_heads} laid out ngram-major, head-minor",
            "values": row_ids.tolist(),
        },
    }
    write_json(RECEIPTS / f"torchref_golden_engram_L{lid}.json", g)


if __name__ == "__main__":
    raise SystemExit(main())
