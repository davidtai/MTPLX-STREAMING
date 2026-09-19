"""Focused target M7 compile/eager and cache-state check; CPU MLX under guard."""

import importlib.util
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)
spec = importlib.util.spec_from_file_location(
    "attn_compile_fixture", Path("tests/models/test_deepseek_v41_attn_compile.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

all_ids = [7, 2, 41, 13, 5, 23, 29]
outcomes = []
for rows in (1, 4, 6, 7):
    result = {}
    for compiled in (False, True):
        model, args = fixture._new_model(seed=2)
        param_hash = hashlib.sha256()
        for key, array in tree_flatten(model.parameters()):
            param_hash.update(key.encode())
            param_hash.update(np.asarray(array).tobytes())
        # Match the production shape route: long prefill eager, <=7-row verify
        # eligible for the compiled tape. A cap of 32 would compile this tiny
        # 12-row prefill and contaminate the before/after cache comparison.
        with fixture._ac(compiled, max_rows=7):
            cache = fixture._prefill_one_shot(model, args, s=12, seed=1)
            logits = model(mx.array([all_ids[:rows]]), cache=cache)
            mx.eval(logits)
            state = fixture._cache_snapshot(cache)
        result[compiled] = (np.array(logits), state, cache.offset, param_hash.hexdigest())

    eager, tape = result[False], result[True]
    assert eager[3] == tape[3], "model weights differ between route probes"
    assert eager[0].shape == tape[0].shape == (1, rows, args.vocab_size)
    assert eager[1].keys() == tape[1].keys() and eager[1]
    assert eager[2] == tape[2] == 12 + rows
    diff = np.abs(eager[0] - tape[0])
    states_exact = all(np.array_equal(eager[1][key], tape[1][key]) for key in eager[1])
    outcomes.append({"rows": rows, "logit_unequal": int(np.count_nonzero(diff)),
                     "logit_max_abs_delta": float(diff.max()),
                     "argmax_rows_different": int(np.count_nonzero(
                         eager[0].argmax(axis=-1) != tape[0].argmax(axis=-1))),
                     "cache_entries": len(eager[1]), "cache_exact": states_exact})
print(json.dumps({"status": "diagnostic", "outcomes": outcomes}))
