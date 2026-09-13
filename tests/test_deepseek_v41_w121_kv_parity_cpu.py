"""W121 HIGH-4: CPU parity of the bounded/preallocated KV lane vs the growing lane.

Red-team found the real (Metal) windows produce DIFFERENT greedy tokens with
MTPLX_DSV41_KV_BOUNDED=1 (windows 43/45/48 = one token sha, 44/46/47 = another, first
diff ~decode token 33), refuting the W107 "byte-identical by construction" claim ON
METAL.  This test pins down what is and isn't a bug:

  * ON CPU the two lanes are BIT-IDENTICAL per-step (proven across the coordinator's
    suspects -- the sparse indexer, the candidate-block prefilter, the sliding window,
    chunked prefill, and a huge cap with ~1900 garbage rows beyond the logical length).
    So there is NO "reads the cap-length buffer beyond len" correctness bug: the logical
    ``view()`` == the concatenated store and CPU reductions are layout-independent.
  * The Metal divergence is therefore ROUNDING-CLASS: the preallocated sliced-view
    buffers give the attention GEMM a different reduction LAYOUT than the growing lane's
    freshly-concatenated contiguous arrays, so the fp reduction reassociates and a greedy
    near-tie flips ([[dsv41-inexact-ok-if-tie-flips]]) -- but it changes greedy tokens,
    so bounded KV is NOT the default (reverted W121 HIGH-4) until a Metal A/B proves it.

If this CPU test ever FAILS, a real beyond-len / layout bug HAS been introduced on the
selected path -- fix it before shipping.

ISOLATION: each (config, lane) runs in its OWN subprocess.  mx.compile caches compiled
functions across models in one process (a trace built for model A can be reused with
model B's weights), so an in-process A/B of two fresh models is unreliable (it produced
spurious diffs and even NaN); a subprocess per lane is the only faithful comparison.

CPU-only (the worker pins ``mx.set_default_device(mx.cpu)``); no server, no network.
Run under ``nice -n 19`` and WITHOUT ``pytest -n auto`` (host-encode sensitivity).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_THIS = Path(__file__).resolve()
_WT = _THIS.parents[1]


def _worker(mode: str, cfg: dict, out: str) -> None:
    import os
    os.environ["MLX_DEFAULT_DEVICE"] = "cpu"
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx.utils import tree_flatten, tree_unflatten
    from mtplx.models.deepseek_v41 import Model, ModelArgs
    import mtplx.models.deepseek_v41_cache as C

    env = {
        "MTPLX_DSV41_SELECTED_KEYS": "1", "MTPLX_DSV41_WINDOW_RING": "1",
        "MTPLX_DSV41_LAYOUT_FIX": "1", "MTPLX_DSV41_PREFILL_LAYER_MAJOR": "1",
        "MTPLX_DSV41_PREFILL_DENSE": "1", "MTPLX_DSV41_PREFILL_SCORE_PATH": "lean",
        "MTPLX_DSV41_RUNNER": "v2",
    }
    for k, v in env.items():
        os.environ[k] = v
    if mode == "bnd":
        os.environ["MTPLX_DSV41_KV_BOUNDED"] = "1"
        os.environ["MTPLX_DSV41_KV_BOUNDED_MAXKV"] = str(cfg["cap"])

    args = ModelArgs(
        vocab_size=48, hidden_size=32, num_hidden_layers=8, num_attention_heads=4,
        head_dim=16, qk_rope_head_dim=4, q_lora_rank=12, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=cfg["itk"],
        sliding_window=cfg["sld"], window_size=cfg["sld"], swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=cfg["cb"],
        candidate_block_size=cfg["cbs"],
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )
    model = Model(args)
    mx.random.seed(1)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = 0.1 * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new)); mx.eval(model.parameters())

    prompt = list(range(cfg["plen"])); chunk = cfg["chunk"]; steps = cfg["steps"]
    cache = model.make_cache()
    if chunk and chunk < len(prompt):
        i = 0; logits = None
        while i < len(prompt):
            logits = model(mx.array([prompt[i:i + chunk]]), cache=cache); mx.eval(logits); i += chunk
    else:
        logits = model(mx.array([list(prompt)]), cache=cache); mx.eval(logits)
    bounded = bool(getattr(cache.layers[0], "_kv_bounded", False))
    assert bounded == (mode == "bnd"), f"lane engagement wrong: {bounded} for {mode}"
    C.reset_kv_bounded_stats()
    rows = [np.array(logits[0, -1].astype(mx.float32))]
    tok = int(mx.argmax(logits[0, -1]).item()); toks = [tok]
    for _ in range(steps):
        logits = model(mx.array([[tok]]), cache=cache); mx.eval(logits)
        rows.append(np.array(logits[0, -1].astype(mx.float32)))
        tok = int(mx.argmax(logits[0, -1]).item()); toks.append(tok)
    np.save(out, np.stack(rows))
    Path(out + ".toks.json").write_text(json.dumps(toks))


def _run_lane(cfg, mode, tmp_path):
    out = str(tmp_path / f"{mode}.npy")
    r = subprocess.run(
        [sys.executable, str(_THIS), "--worker", mode, json.dumps(cfg), out],
        cwd=str(_WT), env={"PYTHONPATH": str(_WT), "MLX_DEFAULT_DEVICE": "cpu",
                           "PATH": __import__("os").environ.get("PATH", "")},
        capture_output=True, text=True, timeout=180,
    )
    assert r.returncode == 0, f"worker {mode} failed:\n{r.stdout}\n{r.stderr}"
    return np.load(out), json.loads(Path(out + ".toks.json").read_text())


@pytest.mark.parametrize(
    "cfg,label",
    [
        ({"itk": 5, "cb": 8, "cbs": 2, "sld": 8, "plen": 40, "steps": 12, "cap": 60, "chunk": 8},
         "spec 40+12 tight cap"),
        ({"itk": 40, "cb": 64, "cbs": 8, "sld": 8, "plen": 120, "steps": 20, "cap": 160, "chunk": 8},
         "candidate+index p120"),
        ({"itk": 64, "cb": 128, "cbs": 8, "sld": 16, "plen": 200, "steps": 30, "cap": 240, "chunk": 16},
         "large itk/blocks sld16 p200"),
        ({"itk": 40, "cb": 64, "cbs": 8, "sld": 8, "plen": 120, "steps": 20, "cap": 2000, "chunk": 8},
         "huge cap (garbage rows)"),
    ],
)
def test_bounded_bit_identical_to_growing_on_cpu(cfg, label, tmp_path):
    g, gt = _run_lane(cfg, "grow", tmp_path)
    b, bt = _run_lane(cfg, "bnd", tmp_path)
    assert gt == bt, f"[{label}] greedy tokens differ growing vs bounded"
    d = np.abs(g - b)
    assert np.isfinite(d).all(), f"[{label}] non-finite logits"
    assert float(d.max()) == 0.0, (
        f"[{label}] bounded vs growing CPU logits differ max|d|={float(d.max()):.3e}"
    )


if __name__ == "__main__":
    # subprocess worker: --worker <mode> <cfg-json> <out>
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], json.loads(sys.argv[3]), sys.argv[4])
