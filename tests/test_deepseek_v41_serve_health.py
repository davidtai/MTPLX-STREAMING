"""CPU-only tests for scripts/deepseek_v41/serve_health.sh's response parsing.

The W11 failure: serve_health.sh parsed each HTTP body inline with
``printf '%s' "$BODY" | "$PY" - <<'HEREDOC' ... HEREDOC``. The here-doc claims
the child's fd 0 for the PROGRAM, so the piped body never reached ``sys.stdin``
and ``json.load(sys.stdin)`` always saw EOF -- both ``/health`` and the chat
completion logged "was not valid JSON" and the model-id parse silently fell back
to ``basename``, even though the server had generated successfully
(gpu-window-11-native.log). The fix moves the parsers into
``serve_health_parse.py``, invoked AS A FILE so fd 0 stays free for the body.

These tests:
  * feed canned /v1/models, /health and chat bodies through the SAME functions
    the script now calls, asserting model id / generation_mode / profile /
    completion head / tok/s are extracted, and that a non-JSON body yields the
    exact "was not valid JSON" lines;
  * drive the fixed shell pipe (``printf ... | python serve_health_parse.py
    <mode>``) end to end -- the body reaches stdin;
  * lock the regression: the old ``python3 - <<'HEREDOC'`` form does NOT parse
    the piped body.

Stdlib + subprocess only. No MLX/mtplx/GPU/server/network (serve_health_parse
never imports MLX). Run under ``nice -n 19``, no ``pytest -n auto``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_WT = Path(__file__).resolve().parents[1]
_SCRIPTS = _WT / "scripts" / "deepseek_v41"
_PARSE = _SCRIPTS / "serve_health_parse.py"

HEALTH_JSON = json.dumps(
    {
        "ok": True,
        "model": "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4",
        "generation_mode": "ar",
        "runtime_mode": "sustained AR",
        "profile": {
            "name": "sustained",
            "model_id": "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4",
        },
        "expert_profile": {"model_key": "deepseek-v41-flash-mxfp4"},
    }
)
MODELS_JSON = json.dumps(
    {"object": "list", "data": [{"id": "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"}]}
)
CHAT_JSON = json.dumps(
    {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "A transformer weighs token relationships via self-attention.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 21,
            "completion_tokens": 34,
            "mtplx_stats": {"decode_tok_s": 42.5, "tok_s": 42.5},
        },
    }
)


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dsv41_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parse():
    return _load("serve_health_parse")


# --------------------------------------------------------------------------
# the parsing functions on canned bodies
# --------------------------------------------------------------------------


def test_parse_model_id(parse):
    assert parse.parse_model_id(MODELS_JSON) == "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"
    assert parse.parse_model_id('{"data": []}') == ""
    assert parse.parse_model_id("not json") == ""


def test_format_health_extracts_mode_and_profile(parse):
    lines = parse.format_health(HEALTH_JSON, stamp="T")
    text = "\n".join(lines)
    assert "generation_mode = 'ar'" in text
    assert "profile         = 'sustained'" in text
    assert "runtime_mode='sustained AR'" in text
    assert "model_key       = 'deepseek-v41-flash-mxfp4'" in text
    # the served model id is also surfaced from /health.
    assert "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4" in text


def test_format_health_invalid_json(parse):
    lines = parse.format_health("<html>502</html>", stamp="T")
    assert lines == ["T [serve_health] /health was not valid JSON"]


def test_format_chat_extracts_tok_s_and_completion(parse):
    lines = parse.format_chat(CHAT_JSON, wall_s=0.80, stamp="T")
    text = "\n".join(lines)
    assert "prompt_tokens=21 completion_tokens=34" in text
    assert "tok/s (server-reported) = 42.50" in text
    # 34 tokens / 0.80 s = 42.50 tok/s
    assert "tok/s (wall 0.80s) = 42.50" in text
    assert "completion: 'A transformer weighs token relationships via self-attention.'" in text


def test_format_chat_invalid_json(parse):
    lines = parse.format_chat("event: ping\n\n", wall_s=1.0, stamp="T")
    assert lines == ["T [serve_health] chat response was not valid JSON"]


def test_format_chat_wall_fallback_when_no_usage(parse):
    body = json.dumps({"choices": [{"message": {"content": "hi"}}]})
    lines = parse.format_chat(body, wall_s=2.0, stamp="T")
    text = "\n".join(lines)
    assert "no completion_tokens in usage; wall 2.00s" in text
    assert "completion: 'hi'" in text


# --------------------------------------------------------------------------
# the shell integration: the body must reach the parser's stdin
# --------------------------------------------------------------------------


def _pipe_through_parser(body: str, mode: str, extra_env=None):
    """Run the FIXED invocation: printf '%s' body | python parser.py <mode>."""
    import os

    env = dict(os.environ)
    env["BODY"] = body
    env["PY"] = sys.executable
    env["PARSE"] = str(_PARSE)
    if extra_env:
        env.update(extra_env)
    cmd = 'printf "%s" "$BODY" | "$PY" "$PARSE" ' + mode
    return subprocess.run(
        ["bash", "-c", cmd], env=env, text=True, capture_output=True, check=False
    )


def test_shell_pipe_health_reaches_stdin(parse):
    r = _pipe_through_parser(HEALTH_JSON, "health")
    assert r.returncode == 0, r.stderr
    assert "generation_mode = 'ar'" in r.stdout
    assert "was not valid JSON" not in r.stdout


def test_shell_pipe_models_and_chat_reach_stdin(parse):
    r = _pipe_through_parser(MODELS_JSON, "models")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"

    r = _pipe_through_parser(CHAT_JSON, "chat", extra_env={"WALL_NS": "800000000"})
    assert r.returncode == 0, r.stderr
    assert "completion_tokens=34" in r.stdout
    assert "tok/s (wall 0.80s) = 42.50" in r.stdout
    assert "was not valid JSON" not in r.stdout


def test_old_heredoc_pattern_is_the_bug(parse):
    """Regression lock: the old ``python3 - <<'HEREDOC'`` form (script on stdin)
    cannot see the piped body -- json.load(sys.stdin) hits EOF. This is the
    exact W11 failure the fix removed."""
    import os

    env = dict(os.environ)
    env["BODY"] = HEALTH_JSON
    env["PY"] = sys.executable
    old = (
        'printf "%s" "$BODY" | "$PY" - <<\'PYEOF\' 2>/dev/null || true\n'
        "import json, sys\n"
        "try:\n"
        "    h = json.load(sys.stdin)\n"
        "    print('OK', h.get('generation_mode'))\n"
        "except Exception:\n"
        "    print('was not valid JSON')\n"
        "PYEOF\n"
    )
    r = subprocess.run(
        ["bash", "-c", old], env=env, text=True, capture_output=True, check=False
    )
    # The body never reached stdin: either the JSON collided with the program
    # (SyntaxError -> no OK line) or json.load saw EOF ("was not valid JSON").
    assert "OK 'ar'" not in r.stdout
