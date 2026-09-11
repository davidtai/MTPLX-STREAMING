"""CPU-only tests for the DeepSeek-V4.1 chat render + BOS (W52).

Covers:
  * ``mtplx.chat_encoding``: the ``is_deepseek_v41_tokenizer`` detector and the
    ``render_deepseek_v41_prompt`` / ``encode_deepseek_v41_messages`` port, which
    reproduce the official reference encoder byte-for-byte;
  * the served path in ``mtplx.server.openai``: ``_encode_messages_uncached``
    (code fallback == artifact chat_template), and ``_encode_prompt`` prepending
    BOS id 0 idempotently on the /v1/completions path;
  * the installed artifact chat_template.jinja reproducing the reference.

Pure Python render tests use only the reference fixtures; tokenizer/served tests
skip if the DSV4.1 artifact (or the src encoding fixtures) is absent. No GPU, no
Metal, no model, no server, no network. MLX is pinned to the CPU per
memory/worker-tests-must-pin-mlx-cpu.md. Run under ``nice -n 19`` (no ``-n auto``).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

_WT = Path(__file__).resolve().parents[1]
_ARTIFACT = Path(os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"))
_SRC_TESTS = Path(os.path.expanduser("~/models/DeepSeek-V4.1-Flash-src/encoding/tests"))

# Reference default thinking mode per case (encode_case default is "chat";
# cases 1 and 5 carry thinking_mode="thinking" in their JSON). Case 5 is vision
# (out of scope for the text serving render).
_CASE_CFG = {1: (True, None), 2: (False, None), 3: (False, None), 4: (False, None)}


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _WT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# chat_encoding is deliberately MLX-free; load it directly off the file.
ce = _load_module("dsv41_chat_encoding", "mtplx/chat_encoding.py")


def _reference_case(cid: int):
    data = json.loads((_SRC_TESTS / f"test_input_{cid}.json").read_text())
    if isinstance(data, dict):
        msgs = copy.deepcopy(data["messages"])
        if "tools" in data:
            msgs[0]["tools"] = data["tools"]
        think = data.get("thinking_mode")
        effort = data.get("reasoning_effort")
    else:
        msgs = copy.deepcopy(data)
        think = None
        effort = None
    et = (think == "thinking") if think else _CASE_CFG[cid][0]
    eff = effort if effort is not None else _CASE_CFG[cid][1]
    expected = (_SRC_TESTS / f"test_output_{cid}.txt").read_text().rstrip("\n")
    return msgs, et, eff, expected


def _norm_tool_call_args(messages):
    """The server parses assistant tool_call arguments (str) to a dict before the
    template; mirror it so the template/port see the object shape."""
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                a = fn.get("arguments")
                if isinstance(a, str):
                    fn["arguments"] = json.loads(a)
    return out


# ---------------------------------------------------------------------------
# Pure-Python render (no tokenizer): byte-for-byte vs the reference vectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid", [1, 2, 3, 4])
def test_render_matches_reference_vectors(cid):
    if not _SRC_TESTS.exists():
        pytest.skip(f"reference encoding fixtures absent at {_SRC_TESTS}")
    msgs, et, eff, expected = _reference_case(cid)
    got = ce.render_deepseek_v41_prompt(msgs, enable_thinking=et, reasoning_effort=eff)
    assert got == expected


def test_render_single_turn_chat_and_thinking():
    chat = ce.render_deepseek_v41_prompt(
        [{"role": "user", "content": "What is 2+2?"}], enable_thinking=False
    )
    assert chat == "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜></think>"
    think = ce.render_deepseek_v41_prompt(
        [{"role": "user", "content": "What is 2+2?"}], enable_thinking=True
    )
    assert think == (
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the "
        "higher the value, the more thorough the reasoning)\n\n"
        "<｜User｜>What is 2+2?<｜Assistant｜><think>"
    )


def test_render_quickstart_thinking_with_system():
    qs = ce.render_deepseek_v41_prompt(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        enable_thinking=True,
        reasoning_effort=75,
    )
    assert qs == (
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher "
        "the value, the more thorough the reasoning)\n\nYou are a helpful assistant."
        "<｜User｜>What is 2+2?<｜Assistant｜><think>"
    )


def test_render_add_generation_prompt_false_omits_header():
    r = ce.render_deepseek_v41_prompt(
        [{"role": "user", "content": "hi"}],
        enable_thinking=False,
        add_generation_prompt=False,
    )
    assert r == "<｜begin▁of▁sentence｜><｜User｜>hi"


def test_render_multi_turn_thinking_drops_earlier_reasoning():
    r = ce.render_deepseek_v41_prompt(
        [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "reasoning_content": "early", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ],
        enable_thinking=True,
        drop_thinking=True,
    )
    assert "early" not in r  # dropped: before the last user turn
    assert r.endswith("<｜User｜>Q2<｜Assistant｜><think>")


# ---------------------------------------------------------------------------
# Tokenizer-backed: detector, BOS, template/fallback parity, completions BOS
# ---------------------------------------------------------------------------


def _tokenizer(with_template: bool):
    if not _ARTIFACT.exists():
        pytest.skip(f"DSV4.1 artifact absent at {_ARTIFACT}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(_ARTIFACT), trust_remote_code=False)
    if not with_template:
        tok.chat_template = None
        # bust the server's per-tokenizer detection cache attribute if present
        if hasattr(tok, "_mtplx_is_deepseek_v41"):
            delattr(tok, "_mtplx_is_deepseek_v41")
    return tok


def test_detector_true_for_dsv41():
    tok = _tokenizer(with_template=False)
    assert ce.is_deepseek_v41_tokenizer(tok) is True


def test_encode_emits_bos_first():
    tok = _tokenizer(with_template=False)
    chat = ce.encode_deepseek_v41_messages(
        tok, [{"role": "user", "content": "What is 2+2?"}], enable_thinking=False
    )
    assert chat == [0, 128803, 3085, 344, 223, 20, 13, 20, 33, 128804, 128822]
    think = ce.encode_deepseek_v41_messages(
        tok, [{"role": "user", "content": "hi"}], enable_thinking=True
    )
    assert think[0] == 0 and think[-1] == 128821  # BOS ... <think>


def test_served_code_fallback_matches_reference_and_not_plain():
    tok = _tokenizer(with_template=False)  # force the code fallback
    from mtplx.server.openai import _encode_messages_uncached, ChatMessage

    ids = _encode_messages_uncached(
        tok, [ChatMessage(role="user", content="What is 2+2?")], enable_thinking=False
    )
    assert ids == [0, 128803, 3085, 344, 223, 20, 13, 20, 33, 128804, 128822]
    assert ids[0] == 0  # BOS present
    assert ids[0] != 5265  # never the plain "user:" base render


def test_served_template_equals_code_fallback():
    tok_tpl = _tokenizer(with_template=True)
    if getattr(tok_tpl, "chat_template", None) is None:
        pytest.skip("artifact chat_template.jinja not installed")
    from mtplx.server.openai import _encode_messages_uncached, ChatMessage

    for et in (False, True):
        via_template = _encode_messages_uncached(
            tok_tpl, [ChatMessage(role="user", content="What is 2+2?")], enable_thinking=et
        )
        via_fallback = ce.encode_deepseek_v41_messages(
            tok_tpl,
            [{"role": "user", "content": "What is 2+2?"}],
            enable_thinking=et,
            preserve_thinking=False,
        )
        assert via_template == via_fallback
        assert via_template[0] == 0


def test_completions_prepends_bos_idempotently():
    tok = _tokenizer(with_template=False)
    from mtplx.server.openai import _encode_prompt

    ids_str = _encode_prompt(tok, "hello world")
    assert ids_str[0] == 0
    raw = [int(x) for x in tok.encode("hello world", add_special_tokens=True)]
    assert ids_str == [0, *raw]
    # client already sent BOS as list[int] -> unchanged (idempotent)
    assert _encode_prompt(tok, [0, 123, 456]) == [0, 123, 456]
    # list[int] without BOS -> prepended
    assert _encode_prompt(tok, [123, 456]) == [0, 123, 456]


def test_artifact_template_reproduces_reference_vectors():
    tok = _tokenizer(with_template=True)
    if getattr(tok, "chat_template", None) is None:
        pytest.skip("artifact chat_template.jinja not installed")
    if not _SRC_TESTS.exists():
        pytest.skip(f"reference encoding fixtures absent at {_SRC_TESTS}")
    for cid in (1, 2, 3, 4):
        msgs, et, eff, expected = _reference_case(cid)
        kwargs = {"enable_thinking": et}
        if eff is not None:
            kwargs["reasoning_effort"] = eff
        got = tok.apply_chat_template(
            _norm_tool_call_args(msgs),
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        assert got == expected, f"case {cid}"
