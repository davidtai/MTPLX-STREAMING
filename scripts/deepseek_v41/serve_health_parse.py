#!/usr/bin/env python3
"""Response parsers for ``scripts/deepseek_v41/serve_health.sh``.

serve_health.sh pipes each HTTP response body into ONE of these subcommands,
invoked as a REAL script file (``python3 serve_health_parse.py <mode>``, never
``python3 -``), so the piped body actually reaches ``sys.stdin``.

WHY THIS FILE EXISTS (W11, gpu-window-11-native.log)
----------------------------------------------------
serve_health.sh used to parse each body inline with::

    printf '%s' "$BODY" | "$PY" - <<'PYEOF'
    import json, sys
    data = json.load(sys.stdin)   # <-- always saw EOF
    ...
    PYEOF

That form has TWO writers for the child's fd 0: the pipe (the body) and the
here-doc (the program). Bash applies the here-doc redirect after wiring the
pipe, so fd 0 ends up on the here-doc temp file: ``python3 -`` reads its PROGRAM
from there, and by the time the program runs ``sys.stdin`` is that same file at
EOF. ``json.load(sys.stdin)`` therefore always got an empty string and raised,
so BOTH ``/health`` and the chat completion logged "was not valid JSON" and the
model-id parse silently fell back to ``basename`` -- even though the server had
generated successfully (event ``mtplx_openai_generation``, 34 completion
tokens). The piped body never reached Python at all.

Reading the program from a real file frees fd 0 for the piped body, and keeping
the parsers here as importable functions lets a CPU test feed canned bodies
through the exact same code (tests/test_deepseek_v41_serve_health.py).

Stdlib only; no MLX/mtplx/GPU/network.
"""

from __future__ import annotations

import json
import os
import sys
import time


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _find_first(obj, key, *, numeric: bool = False):
    """Depth-first search for the first non-null value under ``key``.

    With ``numeric=True`` only int/float values match (skips string echoes of a
    numeric-looking key elsewhere in the payload)."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if key in cur:
                val = cur[key]
                if val is not None and (not numeric or isinstance(val, (int, float))):
                    return val
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def parse_model_id(text: str) -> str:
    """The served model id from a ``/v1/models`` body (``""`` if unavailable)."""
    try:
        data = json.loads(text)
    except Exception:
        return ""
    items = (data.get("data") if isinstance(data, dict) else None) or []
    if items and isinstance(items[0], dict):
        return str(items[0].get("id") or "")
    return ""


def format_health(text: str, *, stamp: str | None = None) -> list[str]:
    """Log lines for a ``/health`` body: generation_mode, profile, model_key."""
    stamp = stamp or _stamp()
    try:
        health = json.loads(text)
    except Exception:
        return [f"{stamp} [serve_health] /health was not valid JSON"]
    gen = _find_first(health, "generation_mode")
    runtime_mode = health.get("runtime_mode") if isinstance(health, dict) else None
    profile = health.get("profile") if isinstance(health, dict) else None
    # /health's "profile" is a dict ({"name": ..., "model_id": ...}); surface the
    # readable name, falling back to the raw value if the shape ever changes.
    profile_name = profile.get("name") if isinstance(profile, dict) else profile
    model_key = _find_first(health, "model_key")
    model = health.get("model") if isinstance(health, dict) else None
    return [
        f"{stamp} [serve_health] model (/health)  = {model!r}",
        f"{stamp} [serve_health] generation_mode = {gen!r}",
        f"{stamp} [serve_health] profile         = {profile_name!r} "
        f"(runtime_mode={runtime_mode!r})",
        f"{stamp} [serve_health] model_key       = {model_key!r}",
    ]


def format_chat(text: str, *, wall_s: float, stamp: str | None = None) -> list[str]:
    """Log lines for a chat-completion body: usage, tok/s, completion head."""
    stamp = stamp or _stamp()
    try:
        resp = json.loads(text)
    except Exception:
        return [f"{stamp} [serve_health] chat response was not valid JSON"]
    usage = resp.get("usage") or {}
    comp = usage.get("completion_tokens")
    prompt = usage.get("prompt_tokens")
    # Prefer a server-reported decode tok/s if the response carries one.
    server_tok_s = _find_first(resp, "decode_tok_s", numeric=True)
    if server_tok_s is None:
        server_tok_s = _find_first(resp, "tok_s", numeric=True)
    text_head = ""
    try:
        text_head = (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        pass
    lines = [
        f"{stamp} [serve_health] usage: prompt_tokens={prompt} "
        f"completion_tokens={comp}"
    ]
    if isinstance(server_tok_s, (int, float)) and server_tok_s > 0:
        lines.append(
            f"{stamp} [serve_health] tok/s (server-reported) = {server_tok_s:.2f}"
        )
    if isinstance(comp, int) and comp > 0 and wall_s > 0:
        lines.append(
            f"{stamp} [serve_health] tok/s (wall {wall_s:.2f}s) = {comp / wall_s:.2f}"
        )
    else:
        lines.append(
            f"{stamp} [serve_health] no completion_tokens in usage; wall {wall_s:.2f}s"
        )
    lines.append(f"{stamp} [serve_health] completion: {text_head[:200]!r}")
    return lines


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else ""
    body = sys.stdin.read()
    if mode == "models":
        # Just the id, so the shell can capture it in a variable.
        sys.stdout.write(parse_model_id(body))
        return 0
    if mode == "health":
        print("\n".join(format_health(body)))
        return 0
    if mode == "chat":
        wall_s = max(1e-9, int(os.environ.get("WALL_NS", "0")) / 1e9)
        print("\n".join(format_chat(body, wall_s=wall_s)))
        return 0
    sys.stderr.write(
        f"serve_health_parse: unknown mode {mode!r} (want models|health|chat)\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
