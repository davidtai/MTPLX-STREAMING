"""CPU-only regression test for W55: served_cell_bench.sh's DSV41_MEMORY_LIMIT_GIB
must compose a serve argv the `mtplx serve` parser ACCEPTS.

Window 21 died at server start because the script passed `--memory-budget
<N>GiB`, which `mtplx serve` (mtplx.cli's serve subparser) rejects as an
unrecognized argument. The real knob is `--expert-memory-limit <size>` (plus
`--expert-max-live-kv-tokens` / `--expert-runtime-reserve`). This pins:

  * the serve parser ACCEPTS `--expert-memory-limit 60GiB` (and stores it);
  * it still REJECTS `--memory-budget 60GiB` (the exact Window-21 failure);
  * the composed serve argv the shell builds (with and without an MTP extra
    arg) parses cleanly;
  * the script's own MEM_ARGS line uses `--expert-memory-limit`, not
    `--memory-budget`, so a regression in the shell is caught here.

No GPU, no Metal, no model, no server, no network: `build_parser()` is pure
argparse construction. MLX is pinned to CPU per
memory/worker-tests-must-pin-mlx-cpu.md. Run under `nice -n 19`, no `-n auto`.
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

_WT = Path(__file__).resolve().parents[1]
_SCRIPT = _WT / "scripts" / "deepseek_v41" / "served_cell_bench.sh"


def _serve_parser():
    from mtplx import cli

    return cli.build_parser()


def _composed_serve_argv(memory_gib=None, kv_tokens=17664, extra=()):
    """Mirror the argv scripts/deepseek_v41/served_cell_bench.sh execs."""

    argv = [
        "serve",
        "--model",
        "/tmp/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4",
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
    ]
    if kv_tokens is not None:
        argv += ["--expert-max-live-kv-tokens", str(kv_tokens)]
    argv += ["--no-auth"]
    if memory_gib is not None:
        argv += ["--expert-memory-limit", f"{memory_gib}GiB"]
    argv += list(extra)
    return argv


def test_serve_parser_accepts_expert_max_live_kv_tokens():
    # W23 fix: the 16K cell's prompt (16,384) + max_tokens (1,024) = 17,408 must
    # fit the served context window, so the script sizes
    # --expert-max-live-kv-tokens to max(contexts)+max_tokens+margin (17,664).
    parser = _serve_parser()
    ns = parser.parse_args(_composed_serve_argv(memory_gib=60, kv_tokens=17664))
    assert ns.command == "serve"
    assert ns.expert_max_live_kv_tokens == 17664
    assert ns.expert_memory_limit == "60GiB"


def test_script_serve_sizes_kv_window_and_cells_records_server_log():
    text = _SCRIPT.read_text()
    functional = [
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ]
    blob = "\n".join(functional)
    # the serve exec raises the KV window (else the 16K prompt + output 400s) ...
    assert "--expert-max-live-kv-tokens" in blob
    assert "REQUIRED_KV=$(( MAX_CTX + MAX_TOKENS + KV_MARGIN ))" in blob
    # ... and the cells client is handed the server log so a failed cell can
    # record its error tail.
    assert "--server-log" in blob


@pytest.mark.parametrize("extra", [(), ("--generation-mode", "mtp")])
def test_serve_parser_accepts_expert_memory_limit(extra):
    parser = _serve_parser()
    ns = parser.parse_args(_composed_serve_argv(memory_gib=60, extra=extra))
    assert ns.command == "serve"
    assert ns.expert_memory_limit == "60GiB"
    assert ns.no_auth is True
    if extra:
        assert ns.generation_mode == "mtp"


def test_serve_parser_accepts_argv_without_memory_override():
    parser = _serve_parser()
    ns = parser.parse_args(_composed_serve_argv(memory_gib=None))
    assert ns.command == "serve"
    # unset override => the planner/profile decides (flag stays None)
    assert ns.expert_memory_limit is None


def test_serve_parser_still_rejects_memory_budget():
    parser = _serve_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "serve",
                "--model",
                "/tmp/model",
                "--no-auth",
                "--memory-budget",
                "60GiB",
            ]
        )


def test_script_mem_args_uses_expert_memory_limit_not_memory_budget():
    text = _SCRIPT.read_text()
    functional = [
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ]
    # the populated MEM_ARGS assignment uses the accepted flag ...
    mem_populate = [line for line in functional if "MEM_ARGS=(--" in line.replace(" ", "")]
    assert mem_populate, "no functional MEM_ARGS=(...) assignment found"
    assert all("--expert-memory-limit" in line for line in mem_populate)
    # ... and no functional line anywhere passes the rejected --memory-budget.
    assert all("--memory-budget" not in line for line in functional)
