"""W10 deliverable 3: greedy generation on David's standard 1,024-token
prefill_bench programming prompt (+ BOS), 16 steps, CPU, streamed q2 experts +
engram bank (component-banks). Writes one receipt per token so a restart never
loses progress. Health verdict + decoded text printed at the end.

Run:  PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \
        .worktrees/dsv41-w10/.benchmark-artifacts/deepseek-v41/w10/gen1024.py
"""
import json
import time
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)
mx.random.seed(0)

HERE = Path(__file__).resolve().parent
LOG = HERE / "gen1024.log"
OUT = HERE / "gen1024_receipt.json"
MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2").expanduser()
GIB = 1024 ** 3
BOS = 0
STEPS = 16

_logf = open(LOG, "a", buffering=1)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _logf.write(s + "\n")


from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
from mtplx.prefill_bench import _prompt_build_for_context
from mlx_lm.utils import load_tokenizer
from mlx_lm.models.cache import make_prompt_cache

tok = load_tokenizer(MODEL)
pb = _prompt_build_for_context(tok, 1024, prompt_format="raw")
prompt_ids = [BOS] + list(pb.token_ids)
meta = dict(pb.metadata)
meta["input_tokens"] = len(prompt_ids)
log(f"[gen1024] prompt built: {len(prompt_ids)} tokens (incl BOS); style={meta.get('prompt_style')}")

t0 = time.time()
log("[gen1024] loading model ...")
resident = load_deepseek_v41_streaming(
    MODEL, memory_limit_bytes=int(100 * GIB), max_live_kv_tokens=4096, admit=True,
    admission_receipt=None, expert_cache_limit_bytes=int(15 * GIB), apply_memory_cap=False,
    slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
model = resident.model
runtime = getattr(model, "_mtplx_expert_runtime", None)
log(f"[gen1024] loaded {time.time()-t0:.1f}s")

receipt = {"meta": meta, "steps": []}
OUT.write_text(json.dumps(receipt, indent=2))

cache = make_prompt_cache(model)
t_prefill = time.time()
logits = model(mx.array([prompt_ids]), cache=cache)
mx.eval(logits)
ttft = time.time() - t_prefill
token = int(mx.argmax(logits[0, -1]).item())
gen_ids = [token]
receipt["ttft_s"] = ttft
receipt["prefill_tokens"] = len(prompt_ids)
receipt["steps"].append({"step": 0, "token": token, "text": tok.decode([token]), "phase": "prefill_last"})
OUT.write_text(json.dumps(receipt, indent=2))
log(f"[gen1024] prefill {len(prompt_ids)} tok in {ttft:.1f}s; step 0 tok={token} {tok.decode([token])!r}")

for s in range(1, STEPS):
    ts = time.time()
    logits = model(mx.array([[token]]), cache=cache)
    token = int(mx.argmax(logits[0, -1]).item())
    dt = time.time() - ts
    gen_ids.append(token)
    receipt["steps"].append({"step": s, "token": token, "text": tok.decode([token]),
                             "phase": "decode", "step_s": dt})
    receipt["generated_ids"] = gen_ids
    receipt["generated_text"] = tok.decode(gen_ids)
    OUT.write_text(json.dumps(receipt, indent=2))
    log(f"[gen1024] step {s} tok={token} {tok.decode([token])!r}  ({dt:.1f}s)")

text = tok.decode(gen_ids)
# health verdict: distinct-token ratio + junk-lock check (13394/104113/36564)
junk = {13394, 104113, 36564}
distinct = len(set(gen_ids))
junk_hits = sum(1 for t in gen_ids if t in junk)
healthy = distinct >= max(6, STEPS // 2) and junk_hits <= 1
verdict = "HEALTHY" if healthy else "UNHEALTHY"
receipt["generated_text"] = text
receipt["distinct_tokens"] = distinct
receipt["junk_hits"] = junk_hits
receipt["verdict"] = verdict
OUT.write_text(json.dumps(receipt, indent=2))
log(f"\n[gen1024] TEXT: {text!r}")
log(f"[gen1024] distinct={distinct}/{STEPS} junk_hits={junk_hits} -> {verdict}")
log("[gen1024] DONE")
try:
    if runtime is not None:
        runtime.close()
except Exception:
    pass
