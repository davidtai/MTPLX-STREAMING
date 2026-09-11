#!/usr/bin/env python3
"""Export OUR MLX model's q8-dequantized DENSE resident projections (layers 0-2)
to float32 .npy, for the W9 attribution ladder (ref_ladder.py rung R1).

Every quantized dense projection is dequantized with mx.dequantize (authoritative);
bare arrays (norms, attn_sink, hyper-connection vectors, the bf16 gate) are read
directly.  CPU only.  Writes .benchmark-artifacts/deepseek-v41/w9/mlx_dense/*.npy
(git-ignored) + an index json.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx_dump  # noqa: F401  (_force_worktree_mtplx + mx cpu)
import mlx.core as mx
import mlx.nn as nn

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w9" / "mlx_dense"
GIB = 1024 ** 3
MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()


def deq(x):
    if isinstance(x, nn.QuantizedLinear):
        w = mx.dequantize(x.weight, x.scales, x.biases, group_size=x.group_size, bits=x.bits)
        return np.array(w.astype(mx.float32))
    if isinstance(x, nn.Linear):
        return np.array(x.weight.astype(mx.float32))
    if isinstance(x, mx.array):
        return np.array(x.astype(mx.float32))
    raise TypeError(type(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-layer", type=int, default=2, help="export dense for layers 0..max-layer (default 2; 39 = all backbone)")
    a = ap.parse_args()
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
    OUT.mkdir(parents=True, exist_ok=True)
    r = load_deepseek_v41_streaming(
        MODEL, memory_limit_bytes=int(100 * GIB), max_live_kv_tokens=256, admit=True,
        admission_receipt=None, expert_cache_limit_bytes=int(2 * GIB), apply_memory_cap=False,
        slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
    m = r.model
    index = {}

    def save(L, key, arr):
        fn = f"L{L}__{key.replace('/', '__')}.npy"
        np.save(OUT / fn, arr)
        index[f"L{L}/{key}"] = fn

    for L in range(a.max_layer + 1):
        layer = m.model.layers[L]
        attn = layer.attn
        for k in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            save(L, f"attn/{k}", deq(getattr(attn, k)))
        save(L, "attn/q_norm", deq(attn.q_norm_weight))
        save(L, "attn/kv_norm", deq(attn.kv_norm_weight))
        save(L, "attn/attn_sink", deq(attn.attn_sink))
        if attn.compressor is not None:
            c = attn.compressor
            save(L, "attn/compressor/wkv", deq(c.wkv))
            if getattr(c, "wgate", None) is not None:   # ratio>1 only (ratio==1 is a plain projection)
                save(L, "attn/compressor/wgate", deq(c.wgate))
            save(L, "attn/compressor/norm", deq(c.norm_weight))
        if attn.indexer is not None:
            ix = attn.indexer
            save(L, "attn/indexer/wq_b", deq(ix.wq_b))
            save(L, "attn/indexer/weights_proj", deq(ix.weights_proj))
            if getattr(ix, "wk", None) is not None:      # owns_k (kv_source) layers only
                save(L, "attn/indexer/wk", deq(ix.wk))
                save(L, "attn/indexer/k_norm", deq(ix.k_norm_weight))
        save(L, "attn_norm", deq(layer.attn_norm_weight))
        save(L, "ffn_norm", deq(layer.ffn_norm_weight))
        for k in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            save(L, k, deq(getattr(layer, k)))
        se = layer.mlp.shared_experts
        save(L, "shared/w1", deq(se.w1))
        save(L, "shared/w2", deq(se.w2))
        save(L, "shared/w3", deq(se.w3))
        save(L, "gate/weight", deq(layer.mlp.gate.weight))
        save(L, "gate/bias", deq(layer.mlp.gate.e_score_correction_bias))
        print(f"[export] layer {L} dense done ({sum(1 for k in index if k.startswith(f'L{L}/'))} tensors)")

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[export] wrote {len(index)} tensors + index.json to {OUT.relative_to(REPO)}")
    try:
        r._mtplx_expert_runtime.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
