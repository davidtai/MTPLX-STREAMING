#!/usr/bin/env python3
"""MLX-port side of the W9 parity dump: run OUR streaming DeepSeek-V4.1 model on
the 31-token probe, CPU/float, capturing the SAME per-submodule tensors as the
torch reference (scripts/deepseek_v41/torchref/ref_forward.py) so
compare_ref_vs_mlx.py can localize the first divergence.

CPU only (mx.set_default_device(mx.cpu)); early-stops after layer 2.  Uses the
same streaming loader (component-banks) the W8 gate/probe uses.  Writes:
  * docs/deepseek-v41/receipts/mlx_layers012.json          (compact, committed)
  * .benchmark-artifacts/deepseek-v41/w9/mlx/*.npy         (full f32, git-ignored)
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

TAG = os.environ.get("W9_TAG", "")   # e.g. "_fp32act" for the fp32-activations A/B arm
SUBDIR = "mlx" + TAG

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
mx.random.seed(0)

REPO = Path(__file__).resolve().parents[3]


def _force_worktree_mtplx():
    """Repoint the editable ``mtplx`` install to THIS worktree.

    The venv's editable finder maps ``mtplx`` to the MAIN checkout (branch ``main``,
    which has no deepseek_v41 model).  Rewrite the finder's MAPPING/NAMESPACES so
    ``import mtplx`` resolves to this worktree's copy (the integration-branch model
    under test), leaving non-mtplx entries (e.g. vllm_metal) untouched."""
    main_root = "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd"
    wt_root = str(REPO)
    if wt_root == main_root:
        return
    for fmod in list(sys.modules.values()):
        name = getattr(fmod, "__name__", "")
        if not (name.startswith("__editable__") and "mtplx" in name):
            continue
        mapping = getattr(fmod, "MAPPING", None)
        if isinstance(mapping, dict):
            for k, v in list(mapping.items()):
                if isinstance(v, str) and v.startswith(main_root + "/mtplx"):
                    mapping[k] = v.replace(main_root, wt_root, 1)
        ns = getattr(fmod, "NAMESPACES", None)
        if isinstance(ns, dict):
            for k, paths in list(ns.items()):
                ns[k] = [p.replace(main_root, wt_root, 1)
                         if isinstance(p, str) and p.startswith(main_root + "/mtplx") else p
                         for p in paths]
    sys.path.insert(0, wt_root)


_force_worktree_mtplx()

REPO = Path(__file__).resolve().parents[3]
RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"
NPY_DIR = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w9" / SUBDIR
MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()
GIB = 1024 ** 3
STOP_AFTER = 2

PROBE_IDS = [0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291, 6328, 3465, 1241,
             6036, 14, 291, 3395, 361, 1354, 260, 565, 291, 6328, 3465, 21740, 6036, 14, 291, 2605]


def _np(a):
    return np.array(a.astype(mx.float32), copy=True)


def _sha_f32(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def summarize(arr, name, save_full=True, per_pos=True):
    a = _np(arr)
    flat = a.reshape(-1)
    d = {
        "name": name, "shape": list(a.shape), "dtype": "float32", "count": int(flat.size),
        "mean": float(flat.mean()), "std": float(flat.std()),
        "max_abs": float(np.abs(flat).max()) if flat.size else 0.0,
        "finite": bool(np.isfinite(flat).all()), "sha256_f32": _sha_f32(a),
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
        d["npy"] = f"{SUBDIR}/{name}.npy"
    return d


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"[mlx] wrote {path.relative_to(REPO)}")


def main():
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
    from mtplx.models import deepseek_v41 as M
    from mlx_lm.utils import load_tokenizer

    t0 = time.time()
    print("[mlx] loading streaming model (CPU) ...")
    resident = load_deepseek_v41_streaming(
        MODEL, memory_limit_bytes=int(100 * GIB), max_live_kv_tokens=4096, admit=True,
        admission_receipt=None, expert_cache_limit_bytes=int(15 * GIB), apply_memory_cap=False,
        slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime")
    print(f"[mlx] loaded {time.time()-t0:.1f}s; engram layers = {getattr(model,'_mtplx_engram_layer_ids',())}")

    ids = mx.array([PROBE_IDS])
    caps = {}

    # -- wrap engram hooks (capture pre/post/gate/value/row_ids) --------------
    for layer in model.model.layers:
        hook = getattr(layer, "engram_hook", None)
        if hook is None:
            continue

        def make(_hook, _lid):
            def wrapper(hidden, token_ids, cache_state):
                pre = hidden
                row_ids = np.asarray(cache_state.current_row_ids(_hook.layer_hash_index), dtype=np.int64)
                # recompute gate/value exactly as EngramV41.__call__ (public attrs only)
                B, L = int(hidden.shape[0]), int(hidden.shape[1])
                embed = _hook.row_cache.dequantize(row_ids)
                kv = _hook.wkv(embed.reshape(B, L, -1))
                split = _hook.hc_mult * _hook.dim
                key = kv[..., :split].astype(mx.float32).reshape(B, L, _hook.hc_mult, _hook.dim)
                value = kv[..., split:].astype(mx.float32)
                hf = hidden.astype(mx.float32)
                weight = (_hook.q_weight * _hook.k_weight).astype(mx.float32)
                eps = _hook.norm_eps
                rstd = mx.rsqrt(mx.mean(hf * hf, axis=-1) + eps) * mx.rsqrt(mx.mean(key * key, axis=-1) + eps)
                dot = mx.sum(hf * weight * key, axis=-1) * rstd * (_hook.dim ** -0.5)
                out = _hook(hidden, token_ids, cache_state)
                caps[f"engram_L{_lid}"] = {
                    "pre": pre, "post": out, "gate_dot": dot, "value": value, "key": key,
                    "row_ids": row_ids,
                }
                return out
            return wrapper

        layer.engram_hook = make(hook, layer.layer_id)

    # capture the compressed-row selection at layer 2
    for layer in model.model.layers[:STOP_AFTER + 1]:
        if getattr(layer.attn, "is_index_source", False) or layer.attn.compress_ratio:
            layer.attn.capture_selection = True

    # -- capturing DecoderLayer.__call__ (transcribed from the real one) ------
    orig_call = M.DecoderLayer.__call__

    class _Stop(Exception):
        pass

    def cap_call(self, h, pre_mix, positions, layer_cache, shared):
        residual = h
        attn_pre, attn_post, attn_comb = self._mixes(h, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        attn_in = self._hc_pre(h, pre_mix)
        attn_in = M._rmsnorm(attn_in, self.attn_norm_weight, self.norm_eps)
        attn_out = self.attn(attn_in, positions, layer_cache, shared)
        h1 = M._hc_post_impl(attn_out, residual, attn_post, attn_comb)

        residual2 = h1
        ffn_pre, ffn_post, ffn_comb = self._mixes(h1, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        moe_in = self._hc_pre(h1, attn_pre)
        moe_in = M._rmsnorm(moe_in, self.ffn_norm_weight, self.norm_eps)
        moe_out = self.mlp(moe_in)
        h2 = M._hc_post_impl(moe_out, residual2, ffn_post, ffn_comb)

        lid = self.layer_id
        if lid <= STOP_AFTER:
            # router top-6 (recompute the gate on the exact MoE input)
            gi, gw = self.mlp.gate(moe_in.reshape(-1, moe_in.shape[-1]))
            shared_out = self.mlp.shared_experts(moe_in.reshape(-1, moe_in.shape[-1])).reshape(moe_in.shape)
            caps[f"attn_L{lid}"] = {"input": attn_in, "output": attn_out}
            caps[f"moe_L{lid}"] = {"input": moe_in, "output": moe_out, "shared_out": shared_out,
                                   "router_ids": np.asarray(gi.astype(mx.int32)),
                                   "router_weights": _np(gw)}
            caps[f"layer_L{lid}"] = {"output": h2}
            # window / compress caches
            win = layer_cache.window
            caps[f"attn_L{lid}"]["window_kv"] = win
            if layer_cache.compress_kv is not None:
                caps[f"attn_L{lid}"]["compressed_kv"] = layer_cache.compress_kv
            if layer_cache.index_k is not None:
                caps[f"attn_L{lid}"]["index_k"] = layer_cache.index_k
            if getattr(self.attn, "last_selection", None) is not None:
                caps[f"attn_L{lid}"]["topk_mask"] = np.asarray(self.attn.last_selection)
        if lid == STOP_AFTER:
            mx.eval(h2)
            raise _Stop()
        return h2, ffn_pre

    M.DecoderLayer.__call__ = cap_call

    # capture the embed stream entering layer 0 by wrapping the backbone embed once
    embed_holder = {}
    orig_layer0 = None

    try:
        cache = model.make_cache()
        try:
            model(ids, cache)
        except _Stop:
            pass
        mx.eval(*[v for c in caps.values() for v in c.values() if isinstance(v, mx.array)])
    finally:
        M.DecoderLayer.__call__ = orig_call

    # embed stream: recompute (embed_tokens + hc broadcast) deterministically
    hb = model.model.embed_tokens(ids)
    hc = model.model.hc_mult
    embed_stream = mx.broadcast_to(hb[:, :, None, :], (hb.shape[0], hb.shape[1], hc, hb.shape[-1]))
    mx.eval(embed_stream)

    # -- assemble receipt -----------------------------------------------------
    out = {
        "meta": {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "mlx", "device": "cpu", "prompt_ids": PROBE_IDS,
            "fp32_activations": os.environ.get("MTPLX_DSV4_FP32_ACTIVATIONS"), "tag": TAG,
            "n_layers_covered": STOP_AFTER + 1, "model": str(MODEL),
            "spec_key": runtime.spec.key,
            "engram_layer_ids": list(getattr(model, "_mtplx_engram_layer_ids", ()) or ()),
        },
        "embed": summarize(embed_stream, "embed_stream"),
        "layers": [], "engram": {}, "attn": {}, "moe": {},
        "final_norm": None, "argmax_token": None, "logits_top8": [],
    }
    for lid in range(STOP_AFTER + 1):
        lc = caps[f"layer_L{lid}"]
        s = summarize(lc["output"], f"layer{lid}_output")
        out["layers"].append({"layer": lid, "mean": s["mean"], "std": s["std"],
                              "max_abs": s["max_abs"], "finite": s["finite"],
                              "first64_last_pos": s["first64_last_pos"], "npy": s.get("npy")})
        ac = caps[f"attn_L{lid}"]
        arec = {"input_post_hc_norm": summarize(ac["input"], f"attn_L{lid}_input"),
                "output": summarize(ac["output"], f"attn_L{lid}_output"),
                "window_kv": summarize(ac["window_kv"], f"attn_L{lid}_window_kv")}
        if "compressed_kv" in ac:
            arec["compressed_kv"] = summarize(ac["compressed_kv"], f"attn_L{lid}_compressed_kv")
        if "index_k" in ac:
            arec["index_k"] = summarize(ac["index_k"], f"attn_L{lid}_index_k")
        if "topk_mask" in ac:
            arec["topk_mask_shape"] = list(ac["topk_mask"].shape)
            np.save(NPY_DIR / f"attn_L{lid}_topk_mask.npy", ac["topk_mask"].astype(np.int8))
            arec["topk_mask_npy"] = f"{SUBDIR}/attn_L{lid}_topk_mask.npy"
        out["attn"][str(lid)] = arec
        mc = caps[f"moe_L{lid}"]
        out["moe"][str(lid)] = {
            "input": summarize(mc["input"], f"moe_L{lid}_input"),
            "output": summarize(mc["output"], f"moe_L{lid}_output"),
            "shared_expert_output": summarize(mc["shared_out"], f"moe_L{lid}_shared_out"),
            "router": {"topk_ids": mc["router_ids"].tolist(), "topk_weights": mc["router_weights"].tolist()},
        }
        ek = f"engram_L{lid}"
        if ek in caps:
            e = caps[ek]
            out["engram"][str(lid)] = {
                "input_pre_engram": summarize(e["pre"], f"engram_L{lid}_input"),
                "output_post_engram": summarize(e["post"], f"engram_L{lid}_output"),
                "value": summarize(e["value"], f"engram_L{lid}_value"),
                "gate_dot": summarize(e["gate_dot"], f"engram_L{lid}_gate_dot"),
                "row_ids": {"shape": list(e["row_ids"].shape), "dtype": "int64",
                            "values": e["row_ids"].tolist()},
            }
        print(f"[mlx] layer {lid}: out mean={s['mean']:.4g} std={s['std']:.4g} "
              f"max_abs={s['max_abs']:.4g} finite={s['finite']}")

    write_json(RECEIPTS / f"mlx_layers012{TAG}.json", out)
    print(f"[mlx] DONE in {time.time()-t0:.1f}s")
    try:
        runtime.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
