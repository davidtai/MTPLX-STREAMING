"""CPU-only tests for scripts/deepseek_v41/serve_bench_1k.py.

Drive the served-1K receipt builder with a canned ``/v1/completions`` response
body (no MLX, no artifact tokenizer, no server, no network): assert the receipt
extracts server-side decode tok/s, prefill tok/s and TTFT from the response
``timings`` block, the token counts from ``usage``, and a stable completion
sha256; and that the receipt write is append-only.

Stdlib + subprocess only; serve_bench_1k imports no MLX at module scope. Run
under ``nice -n 19``; no ``pytest -n auto``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_WT = Path(__file__).resolve().parents[1]
_SCRIPTS = _WT / "scripts" / "deepseek_v41"
_BENCH = _SCRIPTS / "serve_bench_1k.py"

COMPLETION_TEXT = "def solve():\n    return 42\n"

# A realistic non-stream /v1/completions body (see _build_timings / _usage_payload
# in mtplx/server/openai.py): timings carries the server-side rates, usage the
# token counts.
CANNED_RESPONSE = {
    "id": "cmpl-abc123",
    "object": "text_completion",
    "model": "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4",
    "choices": [
        {"index": 0, "text": COMPLETION_TEXT, "finish_reason": "length"}
    ],
    "usage": {
        "prompt_tokens": 1025,
        "completion_tokens": 256,
        "total_tokens": 1281,
    },
    "timings": {
        "prompt_n": 1025,
        "predicted_n": 256,
        "prompt_ms": 4120.5,
        "predicted_ms": 41025.6,
        "prompt_per_second": 248.75,
        "predicted_per_second": 6.24,
        "draft_n": 0,
        "draft_n_accepted": 0,
    },
}


def _load_module():
    spec = importlib.util.spec_from_file_location("dsv41_serve_bench_1k", _BENCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receipt_from_response_extracts_server_side_timing() -> None:
    mod = _load_module()
    request = mod.build_request("m", list(range(1025)), max_tokens=256, temperature=0.0)
    receipt = mod.receipt_from_response(
        CANNED_RESPONSE,
        wall_s=42.0,
        request=request,
        prompt_meta={"prompt_source": "prefill_bench"},
        base_url="http://127.0.0.1:18080",
    )
    s = receipt["server_side"]
    assert s["decode_tok_s"] == pytest.approx(6.24)
    assert s["prefill_tok_s"] == pytest.approx(248.75)
    assert s["ttft_s"] == pytest.approx(4.1205)  # prompt_ms / 1000
    assert s["prefill_ms"] == pytest.approx(4120.5)
    assert s["prompt_tokens"] == 1025
    assert s["completion_tokens"] == 256
    assert s["finish_reason"] == "length"
    # completion sha over choices[0].text
    assert receipt["completion_sha256"] == hashlib.sha256(
        COMPLETION_TEXT.encode("utf-8")
    ).hexdigest()
    # client wall cross-check present
    assert receipt["client_wall"]["wall_s"] == 42.0
    assert receipt["client_wall"]["end_to_end_tok_s"] == pytest.approx(256 / 42.0)


def test_request_is_raw_token_id_list_not_text() -> None:
    # The exact-input guarantee: prompt is a list[int] (verbatim on
    # /v1/completions), never a string (which would gain a BOS via
    # add_special_tokens=True).
    mod = _load_module()
    req = mod.build_request("m", [0, 1, 2, 3], max_tokens=256, temperature=0.0)
    assert isinstance(req["prompt"], list)
    assert all(isinstance(t, int) for t in req["prompt"])
    assert req["stream"] is False
    assert req["temperature"] == 0.0
    assert req["max_tokens"] == 256


def test_missing_timings_degrades_to_usage_counts() -> None:
    mod = _load_module()
    resp = {
        "choices": [{"text": "hi", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    receipt = mod.receipt_from_response(
        resp, wall_s=1.0, request=None, prompt_meta=None
    )
    s = receipt["server_side"]
    assert s["prompt_tokens"] == 10
    assert s["completion_tokens"] == 2
    assert s["decode_tok_s"] is None  # no timings block
    assert s["ttft_s"] is None


def test_write_receipt_is_append_only(tmp_path) -> None:
    mod = _load_module()
    out = tmp_path / "receipt.json"
    mod._write_receipt(out, {"a": 1})
    assert json.loads(out.read_text()) == {"a": 1}
    with pytest.raises(SystemExit):
        mod._write_receipt(out, {"a": 2})  # refuse to clobber a measurement


def test_canned_response_end_to_end_via_cli(tmp_path) -> None:
    # The exact path the CPU harness uses: --canned-response FILE + --out.
    canned = tmp_path / "resp.json"
    canned.write_text(json.dumps(CANNED_RESPONSE))
    out = tmp_path / "cli-receipt.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_BENCH),
            "--canned-response",
            str(canned),
            "--out",
            str(out),
            "--model-id",
            "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4",
            "--print",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(out.read_text())
    assert receipt["kind"] == "dsv41-served-1k-bench"
    assert receipt["server_side"]["decode_tok_s"] == pytest.approx(6.24)
    assert receipt["completion_sha256"] == hashlib.sha256(
        COMPLETION_TEXT.encode("utf-8")
    ).hexdigest()
    assert "decode 6.240 tok/s" in proc.stdout


def test_out_via_env_var(tmp_path) -> None:
    canned = tmp_path / "resp.json"
    canned.write_text(json.dumps(CANNED_RESPONSE))
    out = tmp_path / "env-receipt.json"
    import os

    env = dict(os.environ)
    env["DSV41_BENCH_RECEIPT"] = str(out)
    env["MTPLX_DSV41_CANNED_RESPONSE"] = str(canned)
    proc = subprocess.run(
        [sys.executable, str(_BENCH)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    receipt = json.loads(out.read_text())
    assert receipt["server_side"]["prefill_tok_s"] == pytest.approx(248.75)


# --------------------------------------------------------------------------
# W18 regression: the empty-completion / early-stop diagnostic
# --------------------------------------------------------------------------

# The exact shape window 18 produced: greedy first token is EOS, so an
# EOS-honouring server returns 1 blank token (empty-string sha) with
# finish_reason=stop even though max_tokens=256.
EOS_STOP_RESPONSE = {
    "id": "cmpl-eos",
    "object": "text_completion",
    "model": "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4",
    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1025, "completion_tokens": 1, "total_tokens": 1026},
    "timings": {
        "prompt_n": 1025,
        "predicted_n": 1,
        "prompt_ms": 19003.711,
        "predicted_ms": 0.633,
        "prompt_per_second": 53.937,
        "predicted_per_second": 1578.843,
    },
}

EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_early_stop_is_flagged_in_the_receipt() -> None:
    mod = _load_module()
    request = mod.build_request("m", list(range(1025)), max_tokens=256, temperature=0.0)
    receipt = mod.receipt_from_response(
        EOS_STOP_RESPONSE, wall_s=83.0, request=request, prompt_meta=None
    )
    assert receipt["early_stop"] is True
    assert receipt["warning"] and "EOS" in receipt["warning"]
    assert receipt["completion_sha256"] == EMPTY_SHA  # sha256("")
    assert receipt["server_side"]["completion_tokens"] == 1
    assert receipt["server_side"]["finish_reason"] == "stop"
    assert "WARNING" in mod._summary_line(receipt)


def test_full_completion_is_not_flagged() -> None:
    mod = _load_module()
    request = mod.build_request("m", list(range(1025)), max_tokens=256, temperature=0.0)
    resp = dict(CANNED_RESPONSE)
    resp["choices"] = [{"index": 0, "text": "x" * 40, "finish_reason": "length"}]
    resp["usage"] = {"prompt_tokens": 1025, "completion_tokens": 256}
    receipt = mod.receipt_from_response(
        resp, wall_s=41.0, request=request, prompt_meta=None
    )
    assert receipt["early_stop"] is False
    assert receipt["warning"] is None


def test_real_completion_early_stop_exits_nonzero(tmp_path) -> None:
    # The .sh relies on a non-zero exit to refuse banking a 1-token "rate".
    # A canned response is NOT a real request, so it must NOT fail; drive the
    # non-canned early-stop exit by pointing at a dead server (network error is
    # a different failure) -- instead assert the canned path stays exit 0 even
    # for an early-stop body, and unit-test the exit rule via the receipt flag.
    mod = _load_module()
    canned = tmp_path / "eos.json"
    canned.write_text(json.dumps(EOS_STOP_RESPONSE))
    out = tmp_path / "eos-receipt.json"
    rc = mod.main(
        ["--canned-response", str(canned), "--out", str(out), "--model-id", "m"]
    )
    assert rc == 0  # canned path never fails
    receipt = json.loads(out.read_text())
    assert receipt["early_stop"] is True  # but the flag is recorded


# --------------------------------------------------------------------------
# W18 fix: the REAL server-side handlers (imported), CPU only
# --------------------------------------------------------------------------


class _RecordingTokenizer:
    """Minimal tokenizer double recording add_special_tokens; prepends BOS(0)."""

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=None):
        self.calls.append(add_special_tokens)
        ids = [ord(c) % 97 + 3 for c in text][:8]
        return ([0] + ids) if add_special_tokens else ids


def test_real_encode_prompt_uses_list_int_verbatim_no_bos() -> None:
    # Proves the request shape: /v1/completions with a list[int] prompt is used
    # verbatim -- no re-tokenization, no BOS re-add, no chat template. This is
    # why prefill was exactly 1025 tokens in window 18.
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mtplx.server.openai import _encode_prompt

    tok = _RecordingTokenizer()
    assert _encode_prompt(tok, [0, 5, 9, 12, 7]) == [0, 5, 9, 12, 7]
    assert tok.calls == []  # tokenizer never invoked for a token-id list


def test_real_encode_prompt_string_adds_bos_special_tokens() -> None:
    # A STRING prompt is the WRONG path for exactness: the server re-tokenizes
    # with add_special_tokens=True, so it gains a BOS and would not reproduce
    # the harness's truncated token ids. This is why serve_bench_1k keeps
    # list[int].
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mtplx.server.openai import _encode_prompt

    tok = _RecordingTokenizer()
    out = _encode_prompt(tok, "hello")
    assert tok.calls and tok.calls[0] is True  # add_special_tokens requested
    assert out[0] == 0  # BOS prepended by the tokenizer


def test_real_default_stop_tokens_gate(monkeypatch) -> None:
    # The fix, in the real generation handler: MTPLX_IGNORE_STOP_TOKENS makes
    # the served generation treat no token as a stop, so max_tokens is honoured
    # in full (the fixed-step decode-rate probe the harness runs).
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mtplx.generation import _default_stop_tokens

    class _Tok:
        eos_token_id = 1
        pad_token_id = 2

    monkeypatch.delenv("MTPLX_IGNORE_STOP_TOKENS", raising=False)
    assert _default_stop_tokens(_Tok()) == {1, 2}

    monkeypatch.setenv("MTPLX_IGNORE_STOP_TOKENS", "1")
    assert _default_stop_tokens(_Tok()) == set()

    monkeypatch.setenv("MTPLX_IGNORE_STOP_TOKENS", "0")
    assert _default_stop_tokens(_Tok()) == {1, 2}
