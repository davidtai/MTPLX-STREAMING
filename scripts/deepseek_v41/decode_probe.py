"""W8 §4 discriminating experiment: is the tail degeneration a DECODE-PATH bug or
quality?  A = teacher-forced prefill (prefill path only), B = incremental KV decode
(decode path), C = re-prefill-from-scratch each step (prefill path only).  If A & C
are healthy but B degenerates, the decode-step path is the defect.  CPU, BOS,
component-banks, same loader as the gate.  Writes incrementally.
"""
import time, json
import mlx.core as mx
mx.set_default_device(mx.cpu)
mx.random.seed(0)
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "decode_probe.log"
OUT = HERE / "decode_probe_out.json"
import os as _os
MODEL = Path(_os.environ.get("DSV41_MODEL", "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4")).expanduser()
GIB = 1024**3
BOS = 0

_logf = open(LOG, "a", buffering=1)
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); _logf.write(s + "\n")

from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
from mlx_lm.utils import load_tokenizer
from mlx_lm.models.cache import make_prompt_cache

tok = load_tokenizer(MODEL)
t0 = time.time(); log("[probe] loading ...")
resident = load_deepseek_v41_streaming(
    MODEL, memory_limit_bytes=int(100*GIB), max_live_kv_tokens=4096, admit=True,
    admission_receipt=None, expert_cache_limit_bytes=int(15*GIB), apply_memory_cap=False,
    slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
model = resident.model; runtime = getattr(model, "_mtplx_expert_runtime")
log(f"[probe] loaded {time.time()-t0:.1f}s")
out = {}

def dec(ids): return tok.decode(ids)

# ---- A: teacher-forced prefill, per-position argmax vs ground truth ----
A_TEXT = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef mul(a, b):"
a_ids = [BOS] + list(tok.encode(A_TEXT))
log(f"\n[A] teacher-forced prefill, {len(a_ids)} tokens (incl BOS)")
cache = make_prompt_cache(model)
logits = model(mx.array([a_ids]), cache=cache)  # [1, S, vocab]
mx.eval(logits)
a_rows = []
match = 0; total = 0
for i in range(len(a_ids) - 1):
    pred = int(mx.argmax(logits[0, i]).item())
    actual = a_ids[i + 1]
    ok = pred == actual
    match += ok; total += 1
    a_rows.append({"pos": i, "pred": pred, "pred_txt": dec([pred]), "actual": actual, "match": ok})
log(f"[A] next-token argmax matches ground truth at {match}/{total} positions "
    f"({100.0*match/total:.0f}%)")
# show the last ~12 positions (the "def mul(a, b):" region the model must continue)
for r in a_rows[-12:]:
    log(f"[A]  pos {r['pos']:>2}: pred {r['pred']}={r['pred_txt']!r}  actual {r['actual']}={dec([r['actual']])!r}  {'OK' if r['match'] else 'x'}")
out["A"] = {"text": A_TEXT, "ids": a_ids, "match": match, "total": total, "rows": a_rows}
OUT.write_text(json.dumps(out, indent=2))

# ---- B: incremental KV decode from "def add(a, b):" ----
STEPS = 24
b_prompt = [BOS] + list(tok.encode("def add(a, b):"))
log(f"\n[B] incremental KV decode, {STEPS} steps from {len(b_prompt)} tokens")
cache = make_prompt_cache(model)
logits = model(mx.array([b_prompt]), cache=cache)
token = int(mx.argmax(logits[0, -1]).item())
b_ids = [token]
log(f"[B]  step 0 (prefill last-pos) tok={token} {dec([token])!r}")
for s in range(1, STEPS):
    logits = model(mx.array([[token]]), cache=cache)
    token = int(mx.argmax(logits[0, -1]).item())
    b_ids.append(token)
    if s % 4 == 0 or s < 6:
        log(f"[B]  step {s} tok={token} {dec([token])!r}")
    out["B"] = {"ids": b_ids, "text": dec(b_ids), "used_kv_cache": True}
    OUT.write_text(json.dumps(out, indent=2))
log(f"[B] RESULT {b_ids}")
log(f"[B] TEXT {dec(b_ids)!r}")

# ---- C: re-prefill from scratch each step (prefill path only, no KV reuse) ----
log(f"\n[C] re-prefill-each-step (fresh cache every step), {STEPS} steps")
c_prompt = [BOS] + list(tok.encode("def add(a, b):"))
context = list(c_prompt)
c_ids = []
for s in range(STEPS):
    cache = make_prompt_cache(model)  # FRESH cache -> pure prefill of the full prefix
    logits = model(mx.array([context]), cache=cache)
    token = int(mx.argmax(logits[0, -1]).item())
    c_ids.append(token)
    context.append(token)
    if s % 4 == 0 or s < 6:
        log(f"[C]  step {s} tok={token} {dec([token])!r}")
    out["C"] = {"ids": c_ids, "text": dec(c_ids)}
    OUT.write_text(json.dumps(out, indent=2))
log(f"[C] RESULT {c_ids}")
log(f"[C] TEXT {dec(c_ids)!r}")

# ---- B vs C diff ----
first_div = None
for i in range(min(len(b_ids), len(c_ids))):
    if b_ids[i] != c_ids[i]:
        first_div = i; break
out["b_vs_c_first_divergence_step"] = first_div
OUT.write_text(json.dumps(out, indent=2))
log(f"\n[diff] B vs C first divergence at step: {first_div}")
if first_div is not None:
    log(f"[diff]  B[{first_div}]={b_ids[first_div]}={dec([b_ids[first_div]])!r}  "
        f"C[{first_div}]={c_ids[first_div]}={dec([c_ids[first_div]])!r}")
log("[probe] DONE")
try: runtime.close()
except Exception: pass
