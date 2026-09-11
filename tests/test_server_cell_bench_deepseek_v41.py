"""CPU-only tests for the W49 DeepSeek-V4.1 arm of the FABLE server-cell harness.

Covers ``scripts/fable/server_cell_bench.py`` (the harness of record, whose
prompt construction the Qwen3.8 125B MTPLX PRs #475/#478/#482/#485/#488 use) and
its DSV4.1 model-family switch, plus the ``--prompt-ids-file`` hook added to
``scripts/deepseek_v41/{ab_decode_env_levers,bench_standard_shape}.py``:

  * the DSV4.1 builder sizes a templated prompt within +/-8 of 1,024 and of
    16,384 tokens for each production seed using the REAL DSV4.1 tokenizer
    (skipped if the artifact is absent);
  * the seed rotation offsets are 701/702/703 (seed % 1752 lines);
  * the instruction is byte-identical to the fixture line + the sweep suffix;
  * the DSV4.1 request body carries enable_thinking (default False, the official
    thinking-OFF) and omits reasoning_effort in chat mode (inert), matching the
    Qwen-PR body on everything else;
  * the DSV4.1 counter/ids template the real chat prompt WITH a leading BOS id 0
    (artifact chat_template.jinja), not the old plain "user:/assistant:" render;
  * an exported ids file round-trips into the A/B script's dry-run path;
  * the harness's streaming parse works on a canned SSE stream.

No GPU, no Metal, no model, no server, no network. MLX is pinned to the CPU per
memory/worker-tests-must-pin-mlx-cpu.md. Run under ``nice -n 19`` and without
``pytest -n auto``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

_WT = Path(__file__).resolve().parents[1]
_HARNESS = _WT / "scripts" / "fable" / "server_cell_bench.py"
_BENCH = _WT / "scripts" / "deepseek_v41" / "bench_standard_shape.py"
_DSV41_ARTIFACT = Path(
    os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4")
)
_SEEDS = (20260829, 20260830, 20260831)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scb = _load("w49_server_cell_bench", _HARNESS)
bench = _load("w49_bench_standard_shape", _BENCH)


def _dsv41_tokenizer():
    if not _DSV41_ARTIFACT.exists():
        pytest.skip(f"DSV4.1 artifact absent at {_DSV41_ARTIFACT}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(_DSV41_ARTIFACT), trust_remote_code=False)


# ---------------------------------------------------------------------------
# rotation offsets + instruction (pure, no tokenizer)
# ---------------------------------------------------------------------------


def test_rotation_offsets_are_701_702_703():
    context = scb.load_fixture_context()  # sha-pinned to c8ae2b17...
    lines = context.splitlines()
    assert len(lines) == 1752
    assert [seed % len(lines) for seed in _SEEDS] == [701, 702, 703]
    # rotate_context genuinely rotates by that offset.
    for seed, offset in zip(_SEEDS, (701, 702, 703)):
        rotated = scb.rotate_context(context, seed).splitlines()
        assert rotated == lines[offset:] + lines[:offset]


def test_instruction_is_fixture_line_plus_suffix():
    instruction = scb.load_fixture_instruction()
    # byte-identical to line 1 of the fixture jsonl
    first = json.loads(
        (scb.FIXTURES / "qwen38_naturalistic_generation_patch.jsonl")
        .read_text()
        .splitlines()[0]
    )["prompt"]
    assert instruction == first
    # build_sized_prompt appends instruction.strip() + SWEEP_INSTRUCTION_SUFFIX
    body = instruction.strip() + scb.SWEEP_INSTRUCTION_SUFFIX
    assert body.endswith(scb.SWEEP_INSTRUCTION_SUFFIX)
    assert body.startswith(instruction.strip())


# ---------------------------------------------------------------------------
# request body: DSV4.1 == Qwen-PR body minus the unsupported kwargs
# ---------------------------------------------------------------------------


def _args(model_family_resolved: str, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        no_seed=False,
        reasoning="xhigh",
        model_family_resolved=model_family_resolved,
        dsv41_enable_thinking=None,
        dsv41_reasoning_effort=None,
    )


def test_request_body_deepseek_sends_thinking_off_and_omits_effort(monkeypatch):
    # Deterministic official default regardless of the shell environment.
    monkeypatch.delenv("DSV41_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("DSV41_REASONING_EFFORT", raising=False)
    sweep_prompt = {"cell": "sweep", "target_tokens": 1024}
    seed = _SEEDS[0]

    qwen_sampling = scb.cell_sampling(_args(scb.QWEN38_FAMILY, seed), sweep_prompt, seed)
    dsv_sampling = scb.cell_sampling(
        _args(scb.DEEPSEEK_V41_FAMILY, seed), sweep_prompt, seed
    )
    qwen_body = scb.chat_body(model_id="qwen-id", **qwen_sampling)
    dsv_body = scb.chat_body(model_id="dsv41-id", **dsv_sampling)

    # The Qwen-PR body carries reasoning_effort=xhigh + enable_thinking=true.
    assert qwen_body["reasoning_effort"] == "xhigh"
    assert qwen_body["enable_thinking"] is True
    assert qwen_body["temperature"] == 1.0
    assert qwen_body["top_p"] == 0.95
    assert qwen_body["top_k"] == 20
    assert qwen_body["seed"] == seed
    assert qwen_body["stream"] is True
    assert qwen_body["stream_options"] == {"include_usage": True}

    # DSV4.1 SENDS enable_thinking=false (the official default) and OMITS
    # reasoning_effort (inert in chat mode); everything else matches Qwen.
    assert dsv_body["enable_thinking"] is False
    assert "reasoning_effort" not in dsv_body
    expected = {
        k: v
        for k, v in qwen_body.items()
        if k not in ("reasoning_effort", "enable_thinking", "model")
    }
    got = {k: v for k, v in dsv_body.items() if k not in ("enable_thinking", "model")}
    assert got == expected


def test_request_body_deepseek_thinking_override(monkeypatch):
    monkeypatch.delenv("DSV41_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("DSV41_REASONING_EFFORT", raising=False)
    args = _args(scb.DEEPSEEK_V41_FAMILY, _SEEDS[0])
    args.dsv41_enable_thinking = True
    args.dsv41_reasoning_effort = "max"
    sampling = scb.cell_sampling(args, {"cell": "sweep", "target_tokens": 1024}, _SEEDS[0])
    body = scb.chat_body(model_id="dsv41-id", **sampling)
    assert body["enable_thinking"] is True
    assert body["reasoning_effort"] == "max"
    # the vanity cell always runs thinking-off, even with the override on
    vanity = scb.cell_sampling(args, {"cell": "vanity", "target_tokens": 0}, _SEEDS[0])
    vbody = scb.chat_body(model_id="dsv41-id", **vanity)
    assert vbody["enable_thinking"] is False
    assert "reasoning_effort" not in vbody


def test_template_settings_records_thinking_off_and_bos():
    settings = scb.template_settings_for_family(scb.DEEPSEEK_V41_FAMILY)
    assert settings["model_family"] == "deepseek-v41"
    assert settings["thinking_mode"] is False
    assert settings["enable_thinking"] is False
    assert settings["reasoning_effort"] is None
    assert settings["bos_id_prepended"] is True
    assert settings["bos_token_id"] == 0
    assert settings["add_special_tokens"] is False
    # thinking-on override is reflected
    on = scb.template_settings_for_family(
        scb.DEEPSEEK_V41_FAMILY, dsv41_enable_thinking=True, dsv41_reasoning_effort="max"
    )
    assert on["thinking_mode"] is True
    assert on["enable_thinking"] is True
    assert on["reasoning_effort"] == "max"


# ---------------------------------------------------------------------------
# canned SSE parse (no network)
# ---------------------------------------------------------------------------


class _FakeSSEResponse:
    def __init__(self, lines):
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_stream_chat_parses_canned_sse(monkeypatch):
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        b'data: {"usage":{"prompt_tokens":1022,"completion_tokens":2}}\n',
        b"data: [DONE]\n",
    ]

    def fake_urlopen(request, timeout=None):
        # the harness pins its identity so the managed-client path never voids
        # the sampler; assert the header made it onto the wire.
        assert request.headers.get("User-agent") == scb.BENCH_USER_AGENT
        return _FakeSSEResponse(lines)

    monkeypatch.setattr(scb.urllib.request, "urlopen", fake_urlopen)
    out = scb.stream_chat(
        base_url="http://127.0.0.1:9",
        model_id="dsv41-id",
        prompt="hi",
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        seed=_SEEDS[0],
        reasoning_effort=None,
        enable_thinking=None,
        timeout_s=5.0,
    )
    assert out["ok"] is True
    assert out["text"] == "Hello world"
    assert out["completion_tokens"] == 2
    assert out["prompt_tokens"] == 1022
    assert out["finish_reason"] == "stop"
    assert out["ttft_s"] is not None
    # DSV4.1 omits the thinking kwargs from the wire body.
    assert "reasoning_effort" not in out["request_body"]
    assert "enable_thinking" not in out["request_body"]


def test_stream_chat_captures_http_400_status_and_body(monkeypatch):
    # W23: a 16K prompt + 1,024 output exceeds the served 16,384 context window,
    # so the server rejects with HTTP 400. urllib raises HTTPError at urlopen;
    # the harness must record the STATUS and the response BODY, never a silent
    # all-None row.
    import io
    import urllib.error

    body = (
        b'{"error":{"message":"requested context of 17408 tokens exceeds the '
        b"model's context window of 16384 tokens\",\"type\":\"invalid_request\"}}"
    )

    def raise_400(request, timeout=None):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:9/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(body),
        )

    monkeypatch.setattr(scb.urllib.request, "urlopen", raise_400)
    out = scb.stream_chat(
        base_url="http://127.0.0.1:9",
        model_id="dsv41-id",
        prompt="x" * 10,
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        seed=_SEEDS[0],
        reasoning_effort=None,
        enable_thinking=None,
        timeout_s=5.0,
    )
    assert out["ok"] is False
    assert out["http_status"] == 400
    assert "16384" in out["http_body"] and "17408" in out["http_body"]
    # the error string carries the status AND the body reason (not a bare
    # "HTTP Error 400: Bad Request")
    assert out["error"].startswith("HTTP 400")
    assert "context window" in out["error"]
    # the request body is still recorded so the receipt says what was asked
    assert out["request_body"]["max_tokens"] == 1024


def test_stream_chat_captures_urlerror(monkeypatch):
    import urllib.error

    def raise_conn(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(scb.urllib.request, "urlopen", raise_conn)
    out = scb.stream_chat(
        base_url="http://127.0.0.1:9",
        model_id="dsv41-id",
        prompt="hi",
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        seed=None,
        reasoning_effort=None,
        enable_thinking=None,
        timeout_s=1.0,
    )
    assert out["ok"] is False
    assert out["http_status"] is None
    assert "URLError" in out["error"]


def test_server_log_error_tail_prefers_error_lines(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text(
        "\n".join(
            [
                "[5/6] Context window: 16384 tokens",
                "MTPLX is ready.",
                "some benign line",
                "ERROR request rejected: context window 16384 exceeded",
                "another benign line",
            ]
        )
    )
    tail = scb._server_log_error_tail(str(log))
    assert tail is not None
    assert "context window 16384 exceeded" in tail
    # a missing file returns None, never raises
    assert scb._server_log_error_tail(str(tmp_path / "nope.log")) is None


# ---------------------------------------------------------------------------
# ids file round-trips into the A/B script's prompt resolution / dry-run
# ---------------------------------------------------------------------------


def test_prompt_ids_file_round_trips_into_ab_dry_run(tmp_path):
    ids = [5265, 28, 15361, 260, 535, 16251, 603, 624, 15059, 28]
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(
        json.dumps(
            {
                "schema": "mtplx-server-cell-prompt-ids-v1",
                "model_family": "deepseek-v41",
                "served_model_id": "deepseek-v41-flash-mxfp4",
                "prompts": [
                    {
                        "cell": "sweep",
                        "target_tokens": 1024,
                        "seed": 20260829,
                        "text_sha256": "deadbeef",
                        "templated_tokens": len(ids),
                        "input_tokens": len(ids),
                        "token_ids": ids,
                        "token_ids_sha256": "cafef00d",
                        "bos_id_prepended": False,
                    }
                ],
            }
        )
    )
    args = argparse.Namespace(prompt_ids_file=str(ids_file), prompt_seed=20260829)

    # The A/B resolver returns the exact ids + a source/sha metadata block,
    # bypassing the builder and the BOS prepend.
    prompt_ids, meta = bench._resolve_prompt(
        args, tokenizer=None, build_prompt=None, context_tokens=1024
    )
    assert prompt_ids == ids
    assert meta["prompt_source"] == "prompt-ids-file"
    assert meta["prompt_seed"] == 20260829
    assert meta["bos_prepended"] is False
    assert meta["token_ids_sha256"] == "cafef00d"
    assert meta["prompt_text_sha256"] == "deadbeef"


def test_prompt_ids_file_seed_selection(tmp_path):
    def entry(seed, ids):
        return {
            "cell": "sweep",
            "target_tokens": 1024,
            "seed": seed,
            "token_ids": ids,
            "token_ids_sha256": f"sha{seed}",
        }

    ids_file = tmp_path / "ids.json"
    ids_file.write_text(
        json.dumps(
            {
                "schema": "mtplx-server-cell-prompt-ids-v1",
                "prompts": [entry(20260829, [1, 2]), entry(20260830, [3, 4, 5])],
            }
        )
    )
    # ambiguous without --prompt-seed
    args = argparse.Namespace(prompt_ids_file=str(ids_file), prompt_seed=None)
    with pytest.raises(SystemExit):
        bench._resolve_prompt(args, None, None, 1024)
    # picks the requested seed
    args = argparse.Namespace(prompt_ids_file=str(ids_file), prompt_seed=20260830)
    prompt_ids, _ = bench._resolve_prompt(args, None, None, 1024)
    assert prompt_ids == [3, 4, 5]


# ---------------------------------------------------------------------------
# real-tokenizer sizing (skipped if the artifact is absent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [1024, 16384])
def test_dsv41_builder_sizes_within_tolerance(target):
    tokenizer = _dsv41_tokenizer()
    context = scb.load_fixture_context()
    instruction = scb.load_fixture_instruction()
    encode = lambda text: list(tokenizer.encode(text))  # noqa: E731
    decode = lambda ids: str(tokenizer.decode(list(ids)))  # noqa: E731
    count = scb.make_deepseek_v41_counter(tokenizer)
    for seed in _SEEDS:
        built = scb.build_sized_prompt(
            context=context,
            instruction=instruction,
            seed=seed,
            target_tokens=target,
            encode=encode,
            decode=decode,
            count_templated=count,
        )
        assert abs(built["templated_tokens"] - target) <= scb.PROMPT_TOKEN_TOLERANCE
        # the built prompt ends with the pinned instruction body
        body = instruction.strip() + scb.SWEEP_INSTRUCTION_SUFFIX
        assert built["text"].endswith(body)


def test_dsv41_exported_ids_carry_bos_and_match_counter():
    tokenizer = _dsv41_tokenizer()
    context = scb.load_fixture_context()
    instruction = scb.load_fixture_instruction()
    encode = lambda text: list(tokenizer.encode(text))  # noqa: E731
    decode = lambda ids: str(tokenizer.decode(list(ids)))  # noqa: E731
    count = scb.make_deepseek_v41_counter(tokenizer)
    built = scb.build_sized_prompt(
        context=context,
        instruction=instruction,
        seed=_SEEDS[0],
        target_tokens=1024,
        encode=encode,
        decode=decode,
        count_templated=count,
    )
    built["cell"] = "sweep"
    ids = scb.server_prompt_ids(
        tokenizer, scb.DEEPSEEK_V41_FAMILY, built, reasoning_effort="xhigh"
    )
    # the chat template render: BOS id 0 first, count == templated_tokens, and
    # the exported ids equal the named DSV4.1 helper (default thinking-off).
    assert ids[0] == 0
    assert len(ids) == built["templated_tokens"]
    assert ids == scb.deepseek_v41_prompt_ids(tokenizer, built["text"])
    # the render starts BOS + <｜User｜> (128803), never the plain "user:" (5265)
    assert ids[:2] == [0, 128803]
