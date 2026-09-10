"""W8 §4 layer bisect: capture per-layer hidden states for one decode STEP via the
incremental-KV path (B) and via a full re-prefill of the same prefix (C), then
report the first backbone layer whose last-position hidden diverges.  Names the
decode-path bug.  Usage: python bisect_step.py <K>  (K = decode step index, the
step whose predicting-forward to compare).  CPU, BOS, component-banks.
"""
import sys, time, json
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
mx.random.seed(0)
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2").expanduser()
GIB = 1024**3
BOS = 0
def log(*a): print(*a, flush=True)

from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
from mtplx.models.deepseek_v41 import DecoderLayer
from mlx_lm.utils import load_tokenizer
from mlx_lm.models.cache import make_prompt_cache

tok = load_tokenizer(MODEL)
resident = load_deepseek_v41_streaming(
    MODEL, memory_limit_bytes=int(100*GIB), max_live_kv_tokens=4096, admit=True,
    admission_receipt=None, expert_cache_limit_bytes=int(15*GIB), apply_memory_cap=False,
    slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
model = resident.model; runtime = getattr(model, "_mtplx_expert_runtime")
prompt = [BOS] + list(tok.encode("def add(a, b):"))

_orig = DecoderLayer.__call__
_cap = {}
def cap_call(self, h, pre_mix, positions, lc, shared):
    out, nx = _orig(self, h, pre_mix, positions, lc, shared)
    _cap[self.layer_id] = np.array(out[0, -1].astype(mx.float32))  # last-pos hidden [hc,dim] flat
    return out, nx

def run_capture(fn):
    global _cap
    _cap = {}
    DecoderLayer.__call__ = cap_call
    try:
        tokn = fn()
    finally:
        DecoderLayer.__call__ = _orig
    return tokn, {k: _cap[k] for k in _cap}

# --- B: incremental KV path, capture at step K's forward ---
def path_b():
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)
    token = int(mx.argmax(logits[0, -1]).item())
    gen = [token]
    capK = None
    for s in range(1, K + 1):
        if s == K:
            def step():
                lg = model(mx.array([[gen[-1]]]), cache=cache)
                return int(mx.argmax(lg[0, -1]).item())
            token, cap = run_capture(step)
            gen.append(token)
            return gen, cap
        logits = model(mx.array([[gen[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        gen.append(token)
    return gen, {}

# --- C: full re-prefill of prompt + gen[:K], capture last position ---
def path_c(bgen):
    context = prompt + bgen[:K]   # predict token index K
    def full():
        lg = model(mx.array([context]), cache=make_prompt_cache(model))
        return int(mx.argmax(lg[0, -1]).item())
    return run_capture(full)

log(f"[bisect] K={K} prompt_len={len(prompt)}")
bgen, bcap = path_b()
ctoken, ccap = path_c(bgen)
log(f"[bisect] B step {K} token={bgen[K]} {tok.decode([bgen[K]])!r}  |  C token={ctoken} {tok.decode([ctoken])!r}")

first = None
rows = []
for lid in sorted(set(bcap) & set(ccap)):
    b, c = bcap[lid], ccap[lid]
    d = float(np.max(np.abs(b - c)))
    bmax = float(np.max(np.abs(b))); cmax = float(np.max(np.abs(c)))
    rows.append((lid, d, bmax, cmax))
    if d > 1e-2 and first is None:
        first = lid
for lid, d, bmax, cmax in rows:
    flag = "  <== FIRST DIVERGENCE" if lid == first else ""
    log(f"[bisect] layer {lid:>2}: maxabs_diff={d:.4e}  B_maxabs={bmax:.1f} C_maxabs={cmax:.1f}{flag}")
log(f"[bisect] first diverging layer (>1e-2): {first}")
(HERE / "bisect_out.json").write_text(json.dumps(
    {"K": K, "b_token": bgen[K], "c_token": ctoken,
     "first_layer": first, "rows": [[l, d] for l, d, *_ in rows]}, indent=2))
log("[bisect] DONE")
try: runtime.close()
except Exception: pass
