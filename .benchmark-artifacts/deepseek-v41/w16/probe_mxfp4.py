#!/usr/bin/env python3
"""W16 probe of the native-mxfp4 bank through the REAL streaming serve path.

1. Teacher-forced prefill on the 31-token W9 probe -> next-token argmax matches/30.
2. Capture the MoE module output at layers 0 and 1 (== golden `output`,
   routed+shared sum pre hc_post) and report cosine vs the torch fp32 reference
   golden (docs/deepseek-v41/receipts/torchref_golden_moe_L{0,1}.json):
     - first64-per-position cosine (always; from the committed golden summary)
     - full global cosine if a regenerated reference npy is supplied via
       W16_REF_MOE_L0_NPY / W16_REF_MOE_L1_NPY.

CPU only.  MODEL defaults to the mxfp4 artifact.  Writes probe_mxfp4_out.json +
saves the captured MoE npy to the w16 scratch dir.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
mx.random.seed(0)

REPO = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w16")
import sys
sys.path.insert(0, str(REPO))
W16 = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w16"
RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"
MODEL = Path(os.environ.get("W16_MODEL",
             "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4")).expanduser()
GIB = 1024 ** 3
BOS = 0
A_TEXT = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef mul(a, b):"


def cos_full(a, b):
    a = np.asarray(a, np.float64).reshape(-1); b = np.asarray(b, np.float64).reshape(-1)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def cos_first64(cap, golden):
    """Per-position cosine over the first 64 features vs golden.per_pos_first64."""
    pp = np.asarray(golden["output"]["per_pos_first64"], np.float64)  # [S,64]
    cf = cap.reshape(cap.shape[1], -1)[:, :64].astype(np.float64)      # [S,64]
    per = [(pp[p] @ cf[p]) / (np.linalg.norm(pp[p]) * np.linalg.norm(cf[p]) + 1e-30)
           for p in range(pp.shape[0])]
    return float(np.min(per)), float(np.mean(per))


def main() -> int:
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
    from mtplx.models import deepseek_v41 as M
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(MODEL)
    t0 = time.time()
    print(f"[probe] loading mxfp4 streaming model from {MODEL} ...", flush=True)
    resident = load_deepseek_v41_streaming(
        MODEL, memory_limit_bytes=int(100 * GIB), max_live_kv_tokens=4096, admit=True,
        admission_receipt=None, expert_cache_limit_bytes=int(15 * GIB), apply_memory_cap=False,
        slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime")
    print(f"[probe] loaded {time.time()-t0:.1f}s", flush=True)

    # Capture the MoE module output at L0/L1.  The () operator resolves __call__
    # on the TYPE, so patch the MoE class and mark the target instances (an
    # instance-level __call__ would be ignored).  mlx_dump.py uses the same
    # class-level technique for DecoderLayer.
    caps: dict[int, np.ndarray] = {}
    MoEClass = type(model.model.layers[0].mlp)
    _orig_moe_call = MoEClass.__call__
    for layer in model.model.layers:
        if layer.layer_id in (0, 1):
            object.__setattr__(layer.mlp, "_w16_capture_lid", layer.layer_id)

    def _moe_call(self, x, *a, **k):
        out = _orig_moe_call(self, x, *a, **k)
        lid = getattr(self, "_w16_capture_lid", None)
        if lid is not None:
            caps[lid] = np.array(out.astype(mx.float32), copy=True)
        return out

    MoEClass.__call__ = _moe_call

    a_ids = [BOS] + list(tok.encode(A_TEXT))
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(mx.array([a_ids]), cache=cache)
    mx.eval(logits)
    match = 0
    total = len(a_ids) - 1
    for i in range(total):
        if int(mx.argmax(logits[0, i]).item()) == a_ids[i + 1]:
            match += 1
    print(f"[probe] A matches {match}/{total}", flush=True)

    out = {"model": str(MODEL), "probe_ids": a_ids, "A_match": match, "A_total": total, "moe": {}}
    for lid in (0, 1):
        cap = caps.get(lid)
        if cap is None:
            out["moe"][f"L{lid}"] = {"error": "not captured"}
            continue
        np.save(W16 / f"mxfp4_moe_L{lid}.npy", cap)
        golden = json.loads((RECEIPTS / f"torchref_golden_moe_L{lid}.json").read_text())
        mn, me = cos_first64(cap, golden)
        row = {"shape": list(cap.shape), "first64_min_cos": mn, "first64_mean_cos": me}
        env = os.environ.get(f"W16_REF_MOE_L{lid}_NPY")
        if env and Path(env).is_file():
            row["global_cos_vs_reference"] = cos_full(cap, np.load(env))
        out["moe"][f"L{lid}"] = row
        print(f"[probe] moe_L{lid}: first64 min={mn:.5f} mean={me:.5f}"
              + (f" global_vs_ref={row.get('global_cos_vs_reference'):.5f}" if 'global_cos_vs_reference' in row else ""),
              flush=True)

    (W16 / "probe_mxfp4_out.json").write_text(json.dumps(out, indent=2))
    print("[probe] DONE", flush=True)
    try:
        runtime.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
