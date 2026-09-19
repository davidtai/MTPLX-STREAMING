"""W10 localization: finish W8's ablation and add a routed-expert swiglu-clamp arm.
Arms (teacher-forced 31-tok probe A, CPU, component-banks, streamed q2 experts + engram):
  baseline, engram_off, swa_only, engram_off+swa_only, clamp_routed (patch expert_mlx.swiglu).
Writes incrementally to ablate2_out.json / ablate2.log.
"""
import time, json
import mlx.core as mx
mx.set_default_device(mx.cpu)
mx.random.seed(0)
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "ablate2.log"; OUT = HERE / "ablate2_out.json"
MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2").expanduser()
GIB = 1024**3; BOS = 0
_logf = open(LOG, "a", buffering=1)
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _logf.write(s + "\n")

import mlx.nn as nn
from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
import mtplx.models.expert_mlx as expert_mlx
from mlx_lm.utils import load_tokenizer
from mlx_lm.models.cache import make_prompt_cache

tok = load_tokenizer(MODEL)
A_TEXT = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef mul(a, b):"
ids = [BOS] + list(tok.encode(A_TEXT)); S = len(ids)
log(f"[ablate2] loading ... prompt A = {S} tokens (incl BOS)")
t0=time.time()
resident = load_deepseek_v41_streaming(
    MODEL, memory_limit_bytes=int(100*GIB), max_live_kv_tokens=4096, admit=True,
    admission_receipt=None, expert_cache_limit_bytes=int(15*GIB), apply_memory_cap=False,
    slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
model = resident.model; runtime = getattr(model, "_mtplx_expert_runtime")
layers = model.model.layers
log(f"[ablate2] loaded {time.time()-t0:.1f}s")
out = {"prompt_ids": ids, "configs": {}}

def measure(tag):
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache); mx.eval(logits)
    match = 0; junk = []; preds = []
    for i in range(S - 1):
        p = int(mx.argmax(logits[0, i]).item()); preds.append(p)
        if p == ids[i + 1]: match += 1
        else: junk.append([i, p, tok.decode([p]), ids[i + 1]])
    log(f"[ablate2] {tag}: {match}/{S-1}; junk_pos={[j[0] for j in junk]}")
    out["configs"][tag] = {"match": match, "total": S-1, "junk": junk, "preds": preds}
    OUT.write_text(json.dumps(out, indent=2)); return match

# 1 baseline
measure("baseline")

# 5 clamp_routed: patch the streamed-expert swiglu to apply the reference limit=10 clamp
_orig_swiglu = expert_mlx.swiglu
LIM = 10.0
def clamped_swiglu(gate, x):
    return nn.silu(mx.minimum(gate, LIM)) * mx.clip(x, -LIM, LIM)
expert_mlx.swiglu = clamped_swiglu
measure("clamp_routed")
expert_mlx.swiglu = _orig_swiglu

# 2 engram OFF
saved_hooks = {}
for li, L in enumerate(layers):
    if getattr(L, "engram_hook", None) is not None:
        saved_hooks[li] = L.engram_hook; L.engram_hook = None
measure("engram_off")
for li, h in saved_hooks.items(): layers[li].engram_hook = h

# 3 swa_only
saved_cr = {}
for li, L in enumerate(layers):
    saved_cr[li] = L.attn.compress_ratio; L.attn.compress_ratio = 0
measure("swa_only")
for li, cr in saved_cr.items(): layers[li].attn.compress_ratio = cr

# 6 clamp + swa_only (clamp on, compressed path off, engram on)
expert_mlx.swiglu = clamped_swiglu
saved_cr = {}
for li, L in enumerate(layers):
    saved_cr[li] = L.attn.compress_ratio; L.attn.compress_ratio = 0
measure("clamp+swa_only")
for li, cr in saved_cr.items(): layers[li].attn.compress_ratio = cr
expert_mlx.swiglu = _orig_swiglu

log("[ablate2] DONE")
try: runtime.close()
except Exception: pass
