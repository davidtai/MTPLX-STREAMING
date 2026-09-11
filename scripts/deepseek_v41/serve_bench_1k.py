#!/usr/bin/env python3
"""Served 1,024-token prefill_bench request + receipt for DeepSeek-V4.1-Flash.

The A/B decode-lever wins (W40/W41/W32/W45, windows 14-16) were measured OFF the
served path: the bench harness builds the model in-process and drives a
1,024-token prefill + 256 greedy decode. ``serve_health.sh`` proved the served
daemon works, but its 18-token prompt / 16-token completion is far too short to
show a decode-RATE gain (fixed serve overhead + a handful of decode steps
dominate). This sends the SAME standardized input through the running server so
the served decode/prefill rate is comparable to the bench receipts.

Exactness of the input
----------------------
The prefill_bench prompt is defined at the TOKEN level (the coding-agent text is
encoded then truncated to exactly ``--context-tokens`` ids, then the reference
BOS id 0 is prepended -- W8). ``/v1/completions`` accepts ``prompt`` as a
``list[int]`` and, per ``_encode_prompt`` in ``mtplx/server/openai.py``, uses a
token-id list VERBATIM (no chat template, no ``add_special_tokens`` BOS) -- a
string prompt would instead be re-tokenized with ``add_special_tokens=True`` and
gain a BOS. So we build the ids with the bench's own builder (real artifact
tokenizer) and POST them as a list[int] to the RAW completions endpoint: the
server prefills exactly the bench's token sequence. ``/v1/chat/completions`` is
NOT used (it would wrap the prompt in the chat template).

Server-side timing
-------------------
The non-stream ``/v1/completions`` response carries a ``timings`` block
(``_build_timings``): ``prompt_per_second`` (prefill tok/s), ``predicted_per_second``
(decode tok/s), and ``prompt_ms`` (prefill time == TTFT for a non-stream
request, since the first token is emitted only after prefill). ``usage`` carries
the token counts. We record those, plus a client wall-clock cross-check and the
completion sha256.

CPU-testable
------------
``--canned-response FILE`` (or ``MTPLX_DSV41_CANNED_RESPONSE``) skips the
tokenizer and the network entirely: the file's JSON is treated as the server
response body and turned into a receipt. That is the exact path
``tests/test_deepseek_v41_serve_bench_1k.py`` drives, so the receipt-building and
timing-extraction code is covered with no MLX / no artifact / no server.

Stdlib + (optionally, only for a real request) the bench builder, which loads the
artifact tokenizer (a metadata read; no MLX/Metal, no weights).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = Path.home() / "models" / "DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"
DEFAULT_CONTEXT_TOKENS = 1024
DEFAULT_MAX_TOKENS = 256
DEFAULT_BOS_ID = 0


def _load_build_prompt():
    """Import ``build_prompt`` from the sibling ``dump_hidden_states.py`` by path.

    ``scripts`` is not a package; ``dump_hidden_states`` imports only the stdlib
    at module scope (mlx/mtplx deferred into functions), so this stays CPU-safe
    until ``build_prompt`` actually touches the tokenizer.
    """
    module_path = Path(__file__).resolve().parent / "dump_hidden_states.py"
    spec = importlib.util.spec_from_file_location(
        "dsv41_dump_hidden_states_bench1k", module_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise ImportError(f"cannot load build_prompt from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_prompt


def _prompt_args(model: Path, context_tokens: int, bos: bool, bos_id: int):
    """The tiny namespace ``build_prompt(tokenizer, args)`` reads (raw format)."""
    return argparse.Namespace(
        prompt=None,
        context_tokens=int(context_tokens),
        prompt_format="raw",
        bos=bool(bos),
        bos_id=int(bos_id),
        model=model,
    )


def build_prompt_ids(
    model: Path,
    *,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    bos: bool = True,
    bos_id: int = DEFAULT_BOS_ID,
) -> tuple[list[int], dict[str, Any]]:
    """The exact bench input: prefill_bench @ context_tokens + reference BOS.

    Loads the real artifact tokenizer (metadata read, no weights/MLX).
    """
    from mlx_lm.utils import load_tokenizer

    build_prompt = _load_build_prompt()
    tokenizer = load_tokenizer(Path(model))
    ids, meta = build_prompt(
        tokenizer, _prompt_args(Path(model), context_tokens, bos, bos_id)
    )
    return [int(t) for t in ids], dict(meta)


def build_request(
    model_id: str,
    prompt_ids: list[int],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """A RAW /v1/completions body: token-id prompt (verbatim, no BOS re-add)."""
    return {
        "model": model_id,
        "prompt": [int(t) for t in prompt_ids],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }


def post_completion(
    base_url: str, request: dict[str, Any], *, timeout_s: float = 600.0
) -> tuple[dict[str, Any], float]:
    """POST the request to ``<base_url>/v1/completions``; return (json, wall_s)."""
    url = base_url.rstrip("/") + "/v1/completions"
    data = json.dumps(request).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
    wall_s = time.perf_counter() - start
    return json.loads(body), wall_s


def _completion_text(response: dict[str, Any]) -> str:
    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(choice, dict):
        # /v1/completions -> "text"; be forgiving of a chat-shaped echo too.
        if choice.get("text") is not None:
            return str(choice.get("text") or "")
        message = choice.get("message")
        if isinstance(message, dict):
            return str(message.get("content") or "")
    return ""


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def receipt_from_response(
    response: dict[str, Any],
    *,
    wall_s: float | None,
    request: dict[str, Any] | None,
    prompt_meta: dict[str, Any] | None,
    base_url: str | None = None,
    server_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON receipt from a /v1/completions response body.

    Server-side timing comes from the response ``timings`` block; ``usage`` gives
    the token counts; the completion sha256 is over ``choices[0].text``.
    """
    timings = response.get("timings") if isinstance(response, dict) else None
    timings = timings if isinstance(timings, dict) else {}
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}

    text = _completion_text(response)
    completion_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    prompt_ms = _num(timings.get("prompt_ms"))
    predicted_ms = _num(timings.get("predicted_ms"))
    prefill_tok_s = _num(timings.get("prompt_per_second"))
    decode_tok_s = _num(timings.get("predicted_per_second"))

    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        completion_tokens = int(timings.get("predicted_n") or 0)
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int):
        prompt_tokens = int(timings.get("prompt_n") or 0)

    receipt: dict[str, Any] = {
        "kind": "dsv41-served-1k-bench",
        "created": int(time.time()),
        "base_url": base_url,
        "request": {
            "model": (request or {}).get("model"),
            "max_tokens": (request or {}).get("max_tokens"),
            "temperature": (request or {}).get("temperature"),
            "prompt_kind": "list[int] (raw /v1/completions; verbatim, no BOS re-add)",
            "prompt_tokens_sent": (
                len(request["prompt"])
                if request and isinstance(request.get("prompt"), list)
                else None
            ),
        },
        "prompt_build": prompt_meta,
        "server_side": {
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s": decode_tok_s,
            "ttft_s": (prompt_ms / 1000.0) if prompt_ms is not None else None,
            "prefill_ms": prompt_ms,
            "decode_ms": predicted_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": (
                (response.get("choices") or [{}])[0].get("finish_reason")
                if isinstance(response, dict)
                else None
            ),
        },
        "client_wall": {
            "wall_s": wall_s,
            "end_to_end_tok_s": (
                (completion_tokens / wall_s)
                if wall_s and wall_s > 0 and completion_tokens
                else None
            ),
        },
        "completion_sha256": completion_sha256,
        "completion_len_chars": len(text),
        "completion_head": text[:200],
        "completion_tail": text[-200:],
        "usage": usage,
        "timings": timings,
    }
    if server_extra:
        receipt["server_extra"] = server_extra
    return receipt


def _resolve_receipt_path(args) -> Path:
    raw = args.out or os.environ.get("DSV41_BENCH_RECEIPT")
    if not raw:
        raise SystemExit(
            "no receipt path: pass --out PATH or set DSV41_BENCH_RECEIPT"
        )
    return Path(raw)


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append-only discipline (never-overwrite-a-measurement): refuse to clobber.
    if path.exists():
        raise SystemExit(
            f"receipt path already exists (append-only): {path}"
        )
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--model-id",
        default=None,
        help="served model id (from /v1/models). Default: the model basename.",
    )
    p.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--bos-id", type=int, default=DEFAULT_BOS_ID)
    p.add_argument("--no-bos", dest="bos", action="store_false", default=True)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument(
        "--out",
        default=None,
        help="receipt JSON path (default: $DSV41_BENCH_RECEIPT). Append-only.",
    )
    p.add_argument(
        "--canned-response",
        default=os.environ.get("MTPLX_DSV41_CANNED_RESPONSE"),
        help="CPU test path: JSON file to treat AS the server response body "
        "(skips the tokenizer + network entirely).",
    )
    p.add_argument(
        "--print",
        dest="print_summary",
        action="store_true",
        help="print a human summary line after writing the receipt.",
    )
    return p


def _summary_line(receipt: dict[str, Any]) -> str:
    s = receipt["server_side"]

    def _f(v):
        return "n/a" if v is None else f"{v:.3f}"

    return (
        f"[serve_bench_1k] decode {_f(s['decode_tok_s'])} tok/s | "
        f"prefill {_f(s['prefill_tok_s'])} tok/s | "
        f"ttft {_f(s['ttft_s'])} s | "
        f"prompt_tok={s['prompt_tokens']} completion_tok={s['completion_tokens']} | "
        f"sha={receipt['completion_sha256'][:16]}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_path = _resolve_receipt_path(args)

    if args.canned_response:
        # CPU path: the file IS the server response body.
        response = json.loads(Path(args.canned_response).read_text())
        request = build_request(
            args.model_id or Path(args.model).name,
            list(range(int(args.context_tokens) + (1 if args.bos else 0))),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        receipt = receipt_from_response(
            response,
            wall_s=None,
            request=request,
            prompt_meta={"prompt_source": "canned-response-test"},
            base_url=None,
            server_extra={"source": "canned-response"},
        )
    else:
        model_id = args.model_id or Path(args.model).name
        prompt_ids, prompt_meta = build_prompt_ids(
            args.model,
            context_tokens=args.context_tokens,
            bos=args.bos,
            bos_id=args.bos_id,
        )
        request = build_request(
            model_id,
            prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        response, wall_s = post_completion(
            args.base_url, request, timeout_s=args.timeout_s
        )
        receipt = receipt_from_response(
            response,
            wall_s=wall_s,
            request=request,
            prompt_meta=prompt_meta,
            base_url=args.base_url,
        )

    _write_receipt(out_path, receipt)
    print(f"[serve_bench_1k] receipt -> {out_path}")
    if args.print_summary or not args.canned_response:
        print(_summary_line(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
