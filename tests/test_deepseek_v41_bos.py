"""Regression tests for the W8 root cause: the DeepSeek-V4.1-Flash streaming
greedy path degenerates when the prompt is fed WITHOUT the leading
``<｜begin▁of▁sentence｜>`` (BOS, id 0) that the reference always prepends.

The reference ``inference/generate.py`` builds every prompt through
``encoding.encode_messages`` with ``add_default_bos_token=True`` (it prepends the
BOS when there is no prior context), while the artifact tokenizer sets
``add_bos_token: False`` -- so ``tokenizer.encode(prompt)`` (what the P1.7 gate,
the W6 CPU end-to-end proof and the first GPU gate run all used) omits it.  With
the BOS missing, greedy decoding produces degenerate/wrong first tokens
(``"def add(a, b):"`` -> ``" forward"``); prepending BOS restores the correct
continuation (``"def add(a, b):"`` -> ``" return a + b"``).  See W8_REPORT.md.

The W8 harness scripts (``scripts/deepseek_v41/{dump_hidden_states,
gate_stream_equals_resident}.py``) therefore build their input via
``mtplx.prefill_bench`` and prepend the BOS by default; the cheap tests below
lock that behavior, and the opt-in ``DSV41_RUN_BOS`` test proves BOS is
load-bearing on the real artifact.
"""

from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

_ARTIFACT = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2").expanduser()
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "deepseek_v41"
_BOS_ID = 0


def _load_script(name: str):
    """Import one of the W8 harness scripts as a module (top-level imports are
    stdlib-only, so this is cheap and needs no model/MLX heavy work)."""
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_w8_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    def encode(self, text):
        return [11, 22, 33]


# --------------------------------------------------------------------------
# cheap: the harness scripts prepend BOS and record build metadata
# --------------------------------------------------------------------------
@pytest.mark.parametrize("script", ["dump_hidden_states", "gate_stream_equals_resident"])
def test_build_prompt_prepends_bos_by_default(script):
    module = _load_script(script)
    args = types.SimpleNamespace(
        prompt="hello", context_tokens=1024, prompt_format="raw",
        bos=True, bos_id=_BOS_ID, model="M",
    )
    ids, meta = module.build_prompt(_FakeTokenizer(), args)
    assert ids[0] == _BOS_ID, "the reference BOS must be prepended by default"
    assert ids == [_BOS_ID, 11, 22, 33]
    assert meta["bos_prepended"] is True
    assert meta["bos_id"] == _BOS_ID
    assert meta["input_tokens"] == 4


@pytest.mark.parametrize("script", ["dump_hidden_states", "gate_stream_equals_resident"])
def test_build_prompt_no_bos_flag(script):
    module = _load_script(script)
    args = types.SimpleNamespace(
        prompt="hello", context_tokens=1024, prompt_format="raw",
        bos=False, bos_id=_BOS_ID, model="M",
    )
    ids, meta = module.build_prompt(_FakeTokenizer(), args)
    assert ids == [11, 22, 33]
    assert meta["bos_prepended"] is False
    assert meta["bos_id"] is None


# --------------------------------------------------------------------------
# artifact-gated (tokenizer only, fast): BOS id, and encode() omits it
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _ARTIFACT.is_dir(), reason="DeepSeek-V4.1 artifact not present")
def test_tokenizer_omits_bos_which_is_id_zero():
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(_ARTIFACT)
    assert tok.convert_tokens_to_ids("<｜begin▁of▁sentence｜>") == _BOS_ID
    ids = list(tok.encode("def add(a, b):"))
    # add_bos_token is False in tokenizer_config.json, so encode() must NOT
    # inject the BOS the reference relies on -- the whole point of the finding.
    assert ids and ids[0] != _BOS_ID


@pytest.mark.skipif(not _ARTIFACT.is_dir(), reason="DeepSeek-V4.1 artifact not present")
def test_prefill_bench_build_1024_and_16384_construct():
    from mlx_lm.utils import load_tokenizer
    from mtplx.prefill_bench import _prompt_build_for_context

    tok = load_tokenizer(_ARTIFACT)
    for ctx in (1024, 16384):
        pb = _prompt_build_for_context(tok, ctx, prompt_format="raw")
        assert len(pb.token_ids) == ctx
        assert pb.metadata["prompt_style"] == "coding-agent"
        assert pb.metadata["prompt_release_valid"] is True


# --------------------------------------------------------------------------
# opt-in real-artifact proof: BOS is load-bearing (DSV41_RUN_BOS=1)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("DSV41_RUN_BOS") != "1",
    reason="set DSV41_RUN_BOS=1 to run the real-artifact BOS proof (loads the model)",
)
def test_bos_is_load_bearing_on_real_artifact():
    from mlx_lm.utils import load_tokenizer

    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming

    gib = 1024**3
    resident = load_deepseek_v41_streaming(
        _ARTIFACT, memory_limit_bytes=int(100 * gib), max_live_kv_tokens=4096,
        admit=True, expert_cache_limit_bytes=int(20 * gib), apply_memory_cap=False,
        slot_layout="component-banks", cache_scope="layer", island_layers=(),
        verify_record_hashes=False,
    )
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime")
    try:
        tok = load_tokenizer(_ARTIFACT)
        prompt_ids = list(tok.encode("def add(a, b):"))

        def first_token(ids):
            logits = model(mx.array([ids]), cache=model.make_cache())
            return int(mx.argmax(logits[0, -1]).item())

        with_bos = first_token([_BOS_ID] + prompt_ids)
        without_bos = first_token(prompt_ids)
        # BOS changes the greedy prediction, and the BOS-prefixed continuation is
        # the correct code token " return" (id 1354); no-BOS gives " forward" (6058).
        assert with_bos != without_bos
        assert tok.decode([with_bos]).strip() == "return"
    finally:
        try:
            runtime.close()
        except Exception:
            pass
