"""W46 — DeepSeek-V4.1 decode levers reach the daemon child + precedence.

Two questions this file pins, both CPU-only (no MLX/Metal/GPU, no artifact load;
MLX is pinned to CPU only because the served-log helpers live in
``mtplx.server.openai``, which imports MLX at module scope):

1. Does an exported ``MTPLX_DSV41_*`` lever survive into the daemon child, and do
   the profile's served defaults land when the operator sets nothing?  The serve
   path (``mtplx/commands/public.py``) builds the child env as
   ``os.environ.copy()`` + serve-flag stamps + ``apply_expert_profile_child_env``
   and hands it to the server child as its whole environment
   (``os.execvpe(..., child_env_base)`` / ``_run_server_child_with_app_parent_watchdog(env=...)``).
   We reproduce that composition and actually SPAWN a child with it, asserting the
   keys the child process sees.

2. Precedence.  The ``MTPLX_DSV41_*`` lever namespace is a SERVED DEFAULT: an
   explicit parent-shell export wins over the profile (an operator A/B-ing one
   lever for a single window must not have to edit the profile).  Every other
   child_env key stays FORCED (profile wins) so the W35 memory-safety cap
   ``MTPLX_SESSION_BANK_MAX_BYTES=2GiB`` cannot be defeated by a serve flag such
   as ``--ram-session-cache`` (which stamps that key before the profile applies).

Run under ``nice -n 19``; no ``pytest -n auto``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mtplx.expert_cli import apply_expert_profile_child_env
from mtplx.expert_profiles import load_expert_profiles

PROFILE_NAME = "deepseek-v41-mxfp4-75"

# The measured-positive, byte-identical levers promoted to served defaults (W46;
# windows 14-16 + W40/W41/W32/W45): head bf16, Metal Sinkhorn, attn-chain
# compile, and the sliding-window mask memo.
DEFAULT_LEVERS = {
    "MTPLX_DSV41_HEAD_MODE": "bf16",
    "MTPLX_DSV41_SINKHORN_METAL": "1",
    "MTPLX_DSV41_ATTN_COMPILE": "1",
    "MTPLX_DSV41_ATTN_WIN_MEMO": "1",
}
# Left OFF pending an A/B window (SWITCH_* measured +2.9% but unconfirmed on the
# served path; DEVICE_ROUTE has no reader yet; LAYER_MAJOR is a prefill lever).
LEVERS_LEFT_OFF = (
    "MTPLX_DSV41_SWITCH_FASTPATH",
    "MTPLX_DSV41_SWITCH_SUBMIT",
    "MTPLX_DSV41_DEVICE_ROUTE",
    "MTPLX_DSV41_PREFILL_LAYER_MAJOR",
)
# The forced memory-safety caps (never in the lever namespace).
FORCED_MEMORY_KEYS = {
    "MTPLX_ENGRAM_CACHE_LIMIT": "2GiB",
    "MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS": "0",
    "MTPLX_SESSION_BANK_MAX_BYTES": "2GiB",
}


def _args():
    class _Args:
        _resolved_expert_profile = load_expert_profiles()[PROFILE_NAME]

    return _Args()


def _spawn_child_env_view(child_env: dict[str, str], keys: list[str]) -> dict:
    """Spawn a subprocess with ``child_env`` and read back the keys IT sees.

    Faithful to the daemon spawn: the child is a fresh process whose entire
    environment is the composed ``child_env`` (as ``os.execvpe``/Popen(env=...)
    hand it), so this proves the levers cross the process boundary, not just the
    in-process dict.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,json,sys;"
            "print(json.dumps({k: os.environ.get(k) for k in sys.argv[1:]}))",
            *keys,
        ],
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------
# Part 2: the profile carries the promoted levers (and only those)
# --------------------------------------------------------------------------


def test_profile_child_env_carries_the_promoted_levers() -> None:
    child_env = dict(load_expert_profiles()[PROFILE_NAME].child_env)
    for key, value in DEFAULT_LEVERS.items():
        assert child_env.get(key) == value, key
    for key, value in FORCED_MEMORY_KEYS.items():
        assert child_env.get(key) == value, key


def test_profile_child_env_leaves_the_unconfirmed_levers_off() -> None:
    child_env = dict(load_expert_profiles()[PROFILE_NAME].child_env)
    for key in LEVERS_LEFT_OFF:
        assert key not in child_env, key


# --------------------------------------------------------------------------
# Part 1: env composition + it survives into a spawned child
# --------------------------------------------------------------------------


def test_default_levers_land_when_operator_sets_nothing(monkeypatch) -> None:
    for key in (*DEFAULT_LEVERS, *FORCED_MEMORY_KEYS):
        monkeypatch.delenv(key, raising=False)
    environ: dict[str, str] = {}
    apply_expert_profile_child_env(_args(), environ)
    for key, value in DEFAULT_LEVERS.items():
        assert environ[key] == value, key
    for key, value in FORCED_MEMORY_KEYS.items():
        assert environ[key] == value, key


def test_default_levers_survive_into_a_spawned_child() -> None:
    # Reproduce the serve path: parent env copy + profile child_env, spawned.
    child_env = os.environ.copy()
    for key in DEFAULT_LEVERS:
        child_env.pop(key, None)  # operator set nothing
    apply_expert_profile_child_env(_args(), child_env)
    seen = _spawn_child_env_view(child_env, list(DEFAULT_LEVERS))
    assert seen == DEFAULT_LEVERS


def test_explicit_parent_export_overrides_the_profile_default() -> None:
    # Operator flips two levers in the parent shell before serving.
    child_env = os.environ.copy()
    child_env["MTPLX_DSV41_HEAD_MODE"] = "mxfp8"
    child_env["MTPLX_DSV41_ATTN_COMPILE"] = "0"
    apply_expert_profile_child_env(_args(), child_env)
    seen = _spawn_child_env_view(
        child_env,
        ["MTPLX_DSV41_HEAD_MODE", "MTPLX_DSV41_ATTN_COMPILE",
         "MTPLX_DSV41_SINKHORN_METAL", "MTPLX_DSV41_ATTN_WIN_MEMO"],
    )
    # Parent wins for the two it set; profile default fills the other two.
    assert seen["MTPLX_DSV41_HEAD_MODE"] == "mxfp8"
    assert seen["MTPLX_DSV41_ATTN_COMPILE"] == "0"
    assert seen["MTPLX_DSV41_SINKHORN_METAL"] == "1"
    assert seen["MTPLX_DSV41_ATTN_WIN_MEMO"] == "1"


def test_memory_cap_stays_forced_over_a_serve_flag_value() -> None:
    # A serve flag (e.g. --ram-session-cache bounded) stamps the bank cap into
    # the child env BEFORE the profile applies; the profile must still win, or
    # the admitted memory plan is defeated (W35 / never-exceed-the-memory-knob).
    child_env = os.environ.copy()
    child_env["MTPLX_SESSION_BANK_MAX_BYTES"] = "8G"
    apply_expert_profile_child_env(_args(), child_env)
    assert child_env["MTPLX_SESSION_BANK_MAX_BYTES"] == "2GiB"


def test_double_apply_is_idempotent_like_the_two_site_serve_path() -> None:
    # public.py applies to child_env_base; openai.py re-applies to os.environ in
    # the child. A parent lever export must survive both.
    child_env = os.environ.copy()
    child_env["MTPLX_DSV41_HEAD_MODE"] = "q8"
    apply_expert_profile_child_env(_args(), child_env)  # public.py
    apply_expert_profile_child_env(_args(), child_env)  # openai.py (child)
    assert child_env["MTPLX_DSV41_HEAD_MODE"] == "q8"
    assert child_env["MTPLX_DSV41_SINKHORN_METAL"] == "1"
    assert child_env["MTPLX_SESSION_BANK_MAX_BYTES"] == "2GiB"


# --------------------------------------------------------------------------
# Part 1: the served startup log resolver
# --------------------------------------------------------------------------


def test_served_startup_log_resolver_reports_every_lever() -> None:
    from mtplx.server.openai import (
        _DSV41_LEVER_ENV_KEYS,
        _dsv41_resolved_lever_env,
        _format_dsv41_lever_env,
    )

    # Every key the task requires the window log to surface is present + ordered.
    for key in (
        "MTPLX_DSV41_HEAD_MODE",
        "MTPLX_DSV41_SINKHORN_METAL",
        "MTPLX_DSV41_ATTN_COMPILE",
        "MTPLX_DSV41_ATTN_WIN_MEMO",
        "MTPLX_DSV41_SWITCH_FASTPATH",
        "MTPLX_DSV41_SWITCH_SUBMIT",
        "MTPLX_DSV41_DEVICE_ROUTE",
        "MTPLX_DSV41_HC_COMPILE",
        "MTPLX_DSV41_SHARED_OVERLAP",
        "MTPLX_DSV41_PREFILL_LAYER_MAJOR",
    ):
        assert key in _DSV41_LEVER_ENV_KEYS, key

    env = dict(DEFAULT_LEVERS)  # only the served defaults set
    resolved = _dsv41_resolved_lever_env(env)
    assert list(resolved.keys()) == list(_DSV41_LEVER_ENV_KEYS)
    assert resolved["MTPLX_DSV41_HEAD_MODE"] == "bf16"
    assert resolved["MTPLX_DSV41_DEVICE_ROUTE"] is None

    line = _format_dsv41_lever_env(resolved)
    assert "HEAD_MODE=bf16" in line
    assert "SINKHORN_METAL=1" in line
    assert "ATTN_COMPILE=1" in line
    assert "ATTN_WIN_MEMO=1" in line
    assert "DEVICE_ROUTE=<unset>" in line
    assert "PREFILL_LAYER_MAJOR=<unset>" in line
