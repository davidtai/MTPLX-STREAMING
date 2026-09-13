"""CPU-only allocation contracts for the benchmark's AR prefills and replay."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def ab():
    path = Path(__file__).resolve().parents[1] / "scripts/deepseek_v41/ab_decode_env_levers.py"
    spec = importlib.util.spec_from_file_location("ar_prefill_rows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model:
    def __init__(self):
        self.rows = []
        self.inputs = []
        self.cache = []

    def make_cache(self):
        return self.cache

    def __call__(self, ids, *, cache, logits_keep=None):
        assert cache is self.cache
        self.inputs.append(ids.tolist())
        cache.extend(ids[0].tolist())
        n = ids.shape[1] if logits_keep is None else min(ids.shape[1], logits_keep)
        self.rows.append(n)
        logits = np.zeros((1, n, 8), dtype=np.float32)
        logits[..., len(self.rows) + 1] = 10
        return logits


OPS = SimpleNamespace(input=np.array, sync=lambda x: None,
                      argmax_last=lambda x: int(np.argmax(x[0, -1])))
MEM = SimpleNamespace(reset_peak=lambda: None, peak_bytes=lambda: 0,
                      new_sampler=lambda: SimpleNamespace(start=lambda: None,
                                                          stop=lambda: None),
                      memory_block=lambda sampler: {})
PROMPT = [1] * 16384


def test_generation_heads_one_row_and_preserves_prompt_cache(ab):
    model = Model()
    result = ab._generate(model=model, ops=OPS, mem_probe=MEM, prompt_ids=PROMPT,
                          steps=5, stop_on_eos=True, eos_id=4)
    assert result["generated"] == [2, 3, 4]
    assert result["decode_steps_run"] == 2
    assert model.cache == PROMPT + [2, 3]
    assert model.rows == [1, 1, 1]


@pytest.mark.parametrize("index", [0, 2])
def test_replay_keeps_full_vocabulary_and_exact_ar_prefix(ab, index):
    model = Model()
    row = ab._ar_logits_row_at_index(model=model, ops=OPS,
                                     mx=SimpleNamespace(float32=np.float32),
                                     prompt_ids=PROMPT, ar_tokens=[2, 3], index=index)
    assert row.shape == (8,)
    assert int(np.argmax(row)) == index + 2
    assert model.cache == PROMPT + [2, 3][:index]
    assert model.rows == [1] * (index + 1)


@pytest.mark.parametrize("return_hidden", [False, True])
def test_model_captures_draft_hidden_only_when_requested(return_hidden):
    # Execute the real model entrypoint with an allocation-recording backbone;
    # no MLX import, parameters, or device execution is needed for this boundary.
    path = Path(__file__).resolve().parents[1] / "mtplx/models/deepseek_v41.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    noop = lambda *a, **kw: None
    namespace = {
        "_stime": SimpleNamespace(active=lambda: None,
                                  stage=lambda name: contextlib.nullcontext(SimpleNamespace(add=noop))),
        "_tl": SimpleNamespace(token_head_done=noop, forward_end=noop),
        "_resolve_logits_keep": lambda keep, rows: keep,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    h = np.arange(24).reshape(1, 3, 8)
    hidden = h + 100
    captures = []
    def backbone(ids, cache, **kw):
        captures.append(kw["return_main_hidden"])
        return (h, hidden) if kw["return_main_hidden"] else h
    model = SimpleNamespace(model=backbone, _apply_head=lambda x: x)
    result = namespace["__call__"](model, np.array([[1, 2, 3]]),
                                    return_hidden=return_hidden, logits_keep=1)
    if return_hidden:
        np.testing.assert_array_equal(result[0], h[:, -1:])
        np.testing.assert_array_equal(result[1], hidden)
    else:
        np.testing.assert_array_equal(result, h[:, -1:])
    assert captures == [return_hidden]
