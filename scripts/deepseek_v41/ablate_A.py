"""W8 §4: ablation matrix on the 30-token teacher-forced probe A (with BOS).
Measures next-token argmax matches/30 vs ground truth under: baseline, engram-off,
SWA-only (force every attn.compress_ratio=0), and both.  Then dumps per-layer
last-hidden max-abs at the good pos 18 vs the bad pos 19 to localize where the bad
position goes abnormal.  CPU, component-banks, same loader.  Writes incrementally.
"""
import time, json
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
mx.random.seed(0)
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "ablate_A.log"
OUT = HERE / "ablate_A_out.json"
MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2").expanduser()
GIB = 1024**3
BOS = 0
_logf = open(LOG, "a", buffering=1)
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _logf.write(s + "\n")

from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
from mtplx.models.deepseek_v41 import DecoderLayer
from mlx_lm.utils import load_tokenizer
from mlx_lm.models.cache import make_prompt_cache

tok = load_tokenizer(MODEL)
A_TEXT = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef mul(a, b):"
ids = [BOS] + list(tok.encode(A_TEXT))
S = len(ids)
log(f"[ablate] loading ... prompt A = {S} tokens (incl BOS)")
resident = load_deepseek_v41_streaming(
    MODEL, memory_limit_bytes=int(100*GIB), max_live_kv_tokens=4096, admit=True,
    admission_receipt=None, expert_cache_limit_bytes=int(15*GIB), apply_memory_cap=False,
    slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
model = resident.model; runtime = getattr(model, "_mtplx_expert_runtime")
layers = model.model.layers
log("[ablate] loaded")
out = {"prompt_ids": ids, "configs": {}}

def measure(tag):
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache); mx.eval(logits)
    match = 0; junk = []
    preds = []
    for i in range(S - 1):
        p = int(mx.argmax(logits[0, i]).item()); preds.append(p)
        if p == ids[i + 1]:
            match += 1
        else:
            junk.append([i, p, tok.decode([p]), ids[i + 1]])
    log(f"[ablate] {tag}: {match}/{S-1} matches; junk positions={[j[0] for j in junk]}")
    out["configs"][tag] = {"match": match, "total": S - 1,
                            "junk": junk, "preds": preds}
    OUT.write_text(json.dumps(out, indent=2))
    return match

# 1. baseline
measure("baseline")

# 2. engram OFF
saved_hooks = {}
for li, L in enumerate(layers):
    if getattr(L, "engram_hook", None) is not None:
        saved_hooks[li] = L.engram_hook; L.engram_hook = None
measure("engram_off")
for li, h in saved_hooks.items():
    layers[li].engram_hook = h

# 3. SWA-only (force every layer's compressed path off)
saved_cr = {}
for li, L in enumerate(layers):
    saved_cr[li] = L.attn.compress_ratio
    L.attn.compress_ratio = 0
measure("swa_only")

# 4. engram OFF + SWA-only
for li, L in enumerate(layers):
    if getattr(L, "engram_hook", None) is not None:
        saved_hooks[li] = L.engram_hook; L.engram_hook = None
measure("engram_off+swa_only")
# restore
for li, h in saved_hooks.items():
    layers[li].engram_hook = h
for li, cr in saved_cr.items():
    layers[li].attn.compress_ratio = cr

# 5. per-layer last-hidden max-abs at pos 18 (good) vs pos 19 (bad), baseline
_orig = DecoderLayer.__call__
cap = {}
def cap_call(self, h, pre_mix, positions, lc, shared):
    o, nx = _orig(self, h, pre_mix, positions, lc, shared)
    hf = o.astype(mx.float32)
    cap[self.layer_id] = (float(mx.max(mx.abs(hf[0, 18])).item()),
                          float(mx.max(mx.abs(hf[0, 19])).item()),
                          float(mx.max(mx.abs(hf[0, 18])).item()))
    return o, nx
DecoderLayer.__call__ = cap_call
try:
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([ids]), cache=cache))
finally:
    DecoderLayer.__call__ = _orig
log("[ablate] per-layer max-abs  pos18(good) vs pos19(bad):")
rows = []
for lid in sorted(cap):
    p18, p19, _ = cap[lid]
    ratio = p19 / (p18 + 1e-9)
    rows.append([lid, p18, p19, ratio])
    log(f"[ablate]  layer {lid:>2}: pos18={p18:10.2f} pos19={p19:10.2f} ratio={ratio:.2f}")
out["pos18_vs_pos19"] = rows
OUT.write_text(json.dumps(out, indent=2))
log("[ablate] DONE")
try: runtime.close()
except Exception: pass
