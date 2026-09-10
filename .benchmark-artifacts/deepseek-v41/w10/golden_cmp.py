"""W10 deliverable 4: compare the ported forward's per-layer outputs against W9's
torch reference goldens (L0-L2) on the 31-token probe. Uses each golden's
per_pos_first64 [S][64] summary (first 64 features per position) for cosine +
relative-error, since the full-precision .npy buffers are git-ignored. Confirms
W10's territory (attention/HC/compressor/indexer/engram + residual stream) is
faithful to the reference at the q8-resident floor, and reports the MoE cos so the
q2-vs-bug question is quantified against the same probe.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
mx.random.seed(0)

HERE = Path(__file__).resolve().parent
GOLD = HERE / "goldens"
ROOT = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w10")
sys.path.insert(0, str(ROOT))
LOG = HERE / "golden_cmp.log"; OUT = HERE / "golden_cmp_out.json"
MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2").expanduser()
GIB = 1024**3; BOS = 0
_logf = open(LOG, "a", buffering=1)
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); _logf.write(s+"\n")

from mtplx.models import deepseek_v41 as M
from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
from mlx_lm.utils import load_tokenizer
from mlx_lm.models.cache import make_prompt_cache


def pp64(arr_mx):
    """per_pos_first64 [S,64] from an mlx tensor [1,S,...] flattened over trailing axes."""
    a = np.array(arr_mx.astype(mx.float32), dtype=np.float64)[0]   # [S, ...]
    a = a.reshape(a.shape[0], -1)
    return a[:, :64]


def cmp(mine, golden_block):
    g = np.array(golden_block["per_pos_first64"], dtype=np.float64)  # [S,64]
    m = mine[: g.shape[0]]
    a = m.reshape(-1); b = g.reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    relerr = float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))
    # per-position cos (min across positions - catches a single bad token)
    pc = [float(np.dot(m[p], g[p]) / (np.linalg.norm(m[p]) * np.linalg.norm(g[p]) + 1e-30))
          for p in range(g.shape[0])]
    return cos, min(pc), relerr


tok = load_tokenizer(MODEL)
ids = [0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291, 6328, 3465, 1241,
       6036, 14, 291, 3395, 361, 1354, 260, 565, 291, 6328, 3465, 21740, 6036, 14, 291, 2605]
log(f"[golden] loading ... probe {len(ids)} tokens")
t0=time.time()
resident = load_deepseek_v41_streaming(
    MODEL, memory_limit_bytes=int(100*GIB), max_live_kv_tokens=4096, admit=True,
    admission_receipt=None, expert_cache_limit_bytes=int(15*GIB), apply_memory_cap=False,
    slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
model = resident.model; runtime=getattr(model,"_mtplx_expert_runtime")
log(f"[golden] loaded {time.time()-t0:.1f}s")

cap = {}
_ol = M.DecoderLayer.__call__
_oa = M.Attention.__call__
def cap_attn(self, x, positions, lc, shared):
    d = cap.setdefault(self.layer_id, {}); d["attn_in"] = pp64(x)
    o = _oa(self, x, positions, lc, shared); d["attn_out"] = pp64(o); return o
def cap_layer(self, h, pre_mix, positions, lc, shared):
    d = cap.setdefault(self.layer_id, {})
    _om = self.mlp.__call__
    def cap_mlp(x):
        d["moe_in"] = pp64(x); y = _om(x); d["moe_out"] = pp64(y); return y
    self.mlp.__call__ = cap_mlp
    try:
        ho, fp = _ol(self, h, pre_mix, positions, lc, shared)
    finally:
        try: del self.mlp.__call__
        except Exception: pass
    d["layer_out"] = pp64(ho); return ho, fp
# engram L1 capture
eng = {}
for L in model.model.layers:
    if getattr(L, "engram_hook", None) is not None:
        hook = L.engram_hook; lid = L.layer_id
        def mk(hook, lid):
            def wrapped(h, tok_ids, cs):
                out = hook(h, tok_ids, cs); eng[lid] = pp64(out); return out
            return wrapped
        L.engram_hook = mk(hook, lid)

M.Attention.__call__ = cap_attn
M.DecoderLayer.__call__ = cap_layer
cache = make_prompt_cache(model)
logits = model(mx.array([ids]), cache=cache); mx.eval(logits)
M.Attention.__call__ = _oa
M.DecoderLayer.__call__ = _ol
log("[golden] MLX forward captured")

out = {"attn": {}, "moe": {}, "engram": {}, "layer_out": {}}
for L in (0, 1, 2):
    ga = json.load(open(GOLD / f"torchref_golden_attn_L{L}.json"))
    cos, mincos, re = cmp(cap[L]["attn_out"], ga["output"])
    icos, _, _ = cmp(cap[L]["attn_in"], ga["input_post_hc_norm"])
    out["attn"][L] = {"attn_in_cos": icos, "attn_out_cos": cos, "attn_out_mincos": mincos, "attn_out_relerr": re}
    log(f"[golden] attn L{L}: input_cos={icos:.5f} output_cos={cos:.5f} (min/pos {mincos:.5f}) relerr={re:.3f}")
    gm = json.load(open(GOLD / f"torchref_golden_moe_L{L}.json"))
    mcos, mmin, mre = cmp(cap[L]["moe_out"], gm["output"])
    mi, _, _ = cmp(cap[L]["moe_in"], gm["input"])
    out["moe"][L] = {"moe_in_cos": mi, "moe_out_cos": mcos, "moe_out_mincos": mmin, "moe_out_relerr": mre}
    log(f"[golden] moe  L{L}: input_cos={mi:.5f} output_cos={mcos:.5f} (min/pos {mmin:.5f}) relerr={mre:.3f}")

ge = json.load(open(GOLD / "torchref_golden_engram_L1.json"))
ecos, emin, ere = cmp(eng.get(1, np.zeros((31,64))), ge["output_post_engram"])
out["engram"][1] = {"cos": ecos, "mincos": emin, "relerr": ere}
log(f"[golden] engram L1: output_cos={ecos:.5f} (min/pos {emin:.5f}) relerr={ere:.3f}")

gl = json.load(open(GOLD / "torchref_layers012.json"))
for L in (0, 1, 2):
    key = f"layers.{L}" if f"layers.{L}" in gl else None
    block = None
    if "layers" in gl and isinstance(gl["layers"], list):
        block = gl["layers"][L]
    elif key:
        block = gl[key]
    if block and "first64_last_pos" in block:
        m = cap[L]["layer_out"][-1]  # my last-position first64
        g = np.array(block["first64_last_pos"], dtype=np.float64)
        lcos = float(np.dot(m, g) / (np.linalg.norm(m) * np.linalg.norm(g) + 1e-30))
        lre = float(np.max(np.abs(m - g)) / (np.max(np.abs(g)) + 1e-9))
        out["layer_out"][L] = {"cos": lcos, "relerr": lre}
        log(f"[golden] resid L{L} (last pos): output_cos={lcos:.5f} relerr={lre:.3f}")

OUT.write_text(json.dumps(out, indent=2))
log("[golden] DONE")
try: runtime.close()
except Exception: pass
