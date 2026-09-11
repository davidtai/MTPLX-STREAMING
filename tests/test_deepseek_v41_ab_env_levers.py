"""CPU-only tests for the W28/W30 DSV4.1 env-flag A/B harness.

Covers ``scripts/deepseek_v41/ab_decode_env_levers.py``:

  * the W11 crash fix -- the argparse now carries the prompt-build options
    ``bench_standard_shape.py``'s ``_prompt_args`` reads (``prompt``,
    ``prompt_format``, ``bos``, ``bos_id``), so ``bench._prompt_args(args, ctx)``
    no longer raises ``AttributeError: 'Namespace' object has no attribute
    'prompt'``;
  * the ``--dry-run`` CPU double (no model, no MLX/Metal op, no server);
  * per-arm env application for every preset
    (control / shared_overlap / layer_major / sinkhorn_metal / hc_compile /
    switch_fastpath / attn_compile / both / all_levers / stack_a / head_bf16 /
    head_mxfp8 / head_q8),
    including arm independence (each arm force-unsets the keys it does not set,
    and the W40 load-time head codec MTPLX_DSV41_HEAD_MODE and the six boolean
    per-forward levers never leak across each other);
  * prompt-build metadata parity with ``bench_standard_shape`` at 1024.

No GPU, no Metal, no model, no server, no network. The scripts are not a package
(``scripts/`` has no ``__init__.py``), so they load by file path. MLX is pinned
to the CPU device at import per memory/worker-tests-must-pin-mlx-cpu.md ("no GPU"
is not enough -- MLX defaults to Metal), though the dry-run path never runs an
MLX op. Run under ``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

_WT = Path(__file__).resolve().parents[1]
_SCRIPTS = _WT / "scripts" / "deepseek_v41"

_OV = "MTPLX_DSV41_SHARED_OVERLAP"
_LM = "MTPLX_DSV41_PREFILL_LAYER_MAJOR"
_SK = "MTPLX_DSV41_SINKHORN_METAL"
_HC = "MTPLX_DSV41_HC_COMPILE"
_FP = "MTPLX_DSV41_SWITCH_FASTPATH"    # W42 / K23: switch all-hit fast-path
_AC = "MTPLX_DSV41_ATTN_COMPILE"       # W41 / K22: attention-chain compile
_HM = "MTPLX_DSV41_HEAD_MODE"          # W40 / K21: load-time output-head codec
_ALL_KEYS = (_OV, _LM, _SK, _HC, _FP, _AC)  # the six boolean per-forward levers
_BOOL_AND_HEAD = _ALL_KEYS + (_HM,)    # + the load-time head codec = all seven keys

ALL_ARMS = [
    "control",
    "shared_overlap",
    "layer_major",
    "sinkhorn_metal",
    "hc_compile",
    "switch_fastpath",
    "attn_compile",
    "both",
    "all_levers",
    "stack_a",
    "head_bf16",
    "head_mxfp8",
    "head_q8",
]

# The boolean lever env keys each arm must leave set to "1" (every other unset).
EXPECTED_ON = {
    "control": set(),
    "shared_overlap": {_OV},
    "layer_major": {_LM},
    "sinkhorn_metal": {_SK},
    "hc_compile": {_HC},
    "switch_fastpath": {_FP},
    "attn_compile": {_AC},
    "both": {_OV, _LM},
    "all_levers": {_OV, _LM, _SK, _HC, _FP, _AC},
    "stack_a": {_SK, _FP, _AC},
    "head_bf16": set(),
    "head_mxfp8": set(),
    "head_q8": set(),
}

# The head-codec value each arm pins on MTPLX_DSV41_HEAD_MODE (None = force-unset).
EXPECTED_HEAD = {
    "control": None,
    "shared_overlap": None,
    "layer_major": None,
    "sinkhorn_metal": None,
    "hc_compile": None,
    "switch_fastpath": None,
    "attn_compile": None,
    "both": None,
    "all_levers": None,
    "stack_a": "bf16",
    "head_bf16": "bf16",
    "head_mxfp8": "mxfp8",
    "head_q8": "q8",
}


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dsv41_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def env_levers():
    return _load("ab_decode_env_levers")


@pytest.fixture(scope="module")
def bench():
    return _load("bench_standard_shape")


@pytest.fixture(autouse=True)
def _restore_lever_env():
    """Snapshot and restore the lever + probe env keys around every test so an
    arm applied in one test never leaks into the next."""
    watched = _BOOL_AND_HEAD + ("MTPLX_ROUTE_STAGE_PROBE",)
    saved = {k: os.environ.get(k) for k in watched}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------
# W11 crash fix: the parser now carries the prompt options _prompt_args reads
# --------------------------------------------------------------------------


def test_parser_has_prompt_options_with_bench_defaults(env_levers, bench):
    args = env_levers.build_parser().parse_args(["--out", "/dev/null"])
    bench_args = bench.build_parser().parse_args([])
    # every field bench._prompt_args reads is present, with bench's defaults.
    assert args.prompt is None and bench_args.prompt is None
    assert args.prompt_format == bench_args.prompt_format == "raw"
    assert args.bos is bench_args.bos is True
    assert args.bos_id == bench_args.bos_id == 0
    assert hasattr(args, "dry_run") and args.dry_run is False


def test_prompt_args_no_longer_raises_attributeerror(env_levers, bench):
    # This is the exact call that crashed in _run_arm at GPU window 11
    # (AttributeError: 'Namespace' object has no attribute 'prompt').
    args = env_levers.build_parser().parse_args(
        ["--out", "/dev/null", "--context-tokens", "1024"]
    )
    pa = bench._prompt_args(args, 1024)
    assert pa.prompt is None
    assert pa.context_tokens == 1024
    assert pa.prompt_format == "raw"
    assert pa.bos is True
    assert pa.bos_id == 0
    assert pa.model == args.model


# --------------------------------------------------------------------------
# per-arm env application + independence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_apply_arm_env_sets_and_clears(env_levers, arm):
    # Pre-pollute every lever key (booleans + head) so we prove the arm force-
    # unsets the ones it does not set (arms are independent), not merely sets the
    # ones it wants.
    for k in _BOOL_AND_HEAD:
        os.environ[k] = "bogus"
    env_levers._apply_arm_env(arm)
    for k in _ALL_KEYS:
        if k in EXPECTED_ON[arm]:
            assert os.environ.get(k) == "1", f"{arm}: {k} should be '1'"
        else:
            assert k not in os.environ, f"{arm}: {k} should be force-unset"
    # the head codec pins its value (a string), or is force-unset when None.
    if EXPECTED_HEAD[arm] is None:
        assert _HM not in os.environ, f"{arm}: {_HM} should be force-unset"
    else:
        assert os.environ.get(_HM) == EXPECTED_HEAD[arm], f"{arm}: {_HM}"


def test_head_arms_do_not_touch_boolean_levers(env_levers):
    # a head arm after all_levers clears the four booleans and sets only head.
    env_levers._apply_arm_env("all_levers")
    env_levers._apply_arm_env("head_mxfp8")
    assert os.environ.get(_HM) == "mxfp8"
    assert all(k not in os.environ for k in _ALL_KEYS)
    # a boolean arm after a head arm clears head.
    env_levers._apply_arm_env("both")
    assert _HM not in os.environ
    assert os.environ.get(_OV) == "1" and os.environ.get(_LM) == "1"


def test_apply_arm_env_independent_across_arms(env_levers):
    # all_levers on -> control must clear all keys (no leakage between arms).
    env_levers._apply_arm_env("all_levers")
    assert all(os.environ.get(k) == "1" for k in _ALL_KEYS)
    env_levers._apply_arm_env("control")
    assert all(k not in os.environ for k in _ALL_KEYS)
    # a single-lever arm after all_levers leaves exactly one key set.
    env_levers._apply_arm_env("all_levers")
    env_levers._apply_arm_env("sinkhorn_metal")
    assert os.environ.get(_SK) == "1"
    assert all(k not in os.environ for k in set(_ALL_KEYS) - {_SK})
    # the K22 attn_compile arm leaves exactly its own key set.
    env_levers._apply_arm_env("all_levers")
    env_levers._apply_arm_env("attn_compile")
    assert os.environ.get(_AC) == "1"
    assert all(k not in os.environ for k in set(_ALL_KEYS) - {_AC})


def test_apply_arm_env_rejects_unknown_arm(env_levers):
    with pytest.raises(ValueError):
        env_levers._apply_arm_env("does_not_exist")


# --------------------------------------------------------------------------
# --dry-run through main(): env per arm + prompt-metadata parity with bench
# --------------------------------------------------------------------------


def _run_dry_main(env_levers, out_path):
    rc = env_levers.main(
        [
            "--dry-run",
            "--context-tokens",
            "1024",
            "--arms",
            *ALL_ARMS,
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    receipts = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    assert [r["arm"] for r in receipts] == ALL_ARMS
    return receipts


def test_dry_run_main_records_env_per_arm(env_levers, tmp_path):
    # Pre-pollute so control's receipt proves the keys were cleared, not stale.
    os.environ[_OV] = "bogus"
    os.environ[_HC] = "bogus"
    os.environ[_HM] = "bogus"
    receipts = _run_dry_main(env_levers, tmp_path / "receipts.jsonl")
    for r in receipts:
        assert r["dry_run"] is True
        on = {k for k, v in r["arm_env"].items() if v == "1"}
        assert on == EXPECTED_ON[r["arm"]], r["arm"]
        # keys not in this arm must be recorded as unset (None), never "bogus".
        for k in _ALL_KEYS:
            if k not in EXPECTED_ON[r["arm"]]:
                assert r["arm_env"][k] is None, (r["arm"], k)
        # the head codec value is recorded verbatim (or None when unset).
        assert r["arm_env"].get(_HM) == EXPECTED_HEAD[r["arm"]], r["arm"]


def test_dry_run_prompt_metadata_matches_bench_1024(env_levers, bench, tmp_path):
    # bench_standard_shape's own dry-run build for 1024, in THIS process (so the
    # fake tokenizer's per-process hashing is identical for both harnesses).
    bench_args = bench.build_parser().parse_args([])
    build_prompt = bench._load_build_prompt()
    ref_ids, ref_meta = build_prompt(
        bench._FakeTokenizer(), bench._prompt_args(bench_args, 1024)
    )
    assert ref_meta["input_tokens"] == 1025  # 1024 prefill_bench + reference BOS

    receipts = _run_dry_main(env_levers, tmp_path / "receipts.jsonl")
    for r in receipts:
        assert r["context_tokens"] == 1024
        assert r["prompt_tokens"] == len(ref_ids) == 1025
        assert r["prompt_build"] == ref_meta, r["arm"]


# --------------------------------------------------------------------------
# W37: --stage-timing / --warm-repeat parser + dry-run flow
# --------------------------------------------------------------------------


def test_parser_has_stage_timing_and_warm_repeat(env_levers):
    args = env_levers.build_parser().parse_args(["--out", "/dev/null"])
    # defaults off; the flags exist and the step override defaults to None.
    assert args.stage_timing is False
    assert args.warm_repeat is False
    assert args.stage_timing_steps is None
    on = env_levers.build_parser().parse_args(
        ["--out", "/dev/null", "--stage-timing", "--warm-repeat",
         "--stage-timing-steps", "8"]
    )
    assert on.stage_timing is True
    assert on.warm_repeat is True
    assert on.stage_timing_steps == 8


def test_dry_run_records_stage_timing_and_warm_repeat_flags(env_levers, tmp_path):
    out = tmp_path / "r.jsonl"
    rc = env_levers.main(
        [
            "--dry-run", "--context-tokens", "1024", "--decode-tokens", "64",
            "--arms", "control", "--stage-timing", "--warm-repeat",
            "--out", str(out),
        ]
    )
    assert rc == 0
    r = json.loads(out.read_text().splitlines()[0])
    assert r["dry_run"] is True
    assert r["stage_timing"] is True
    assert r["warm_repeat"] is True
    # step override absent -> mirrors --decode-tokens.
    assert r["stage_timing_steps"] == 64


def test_dry_run_flags_default_off(env_levers, tmp_path):
    out = tmp_path / "r.jsonl"
    env_levers.main(
        ["--dry-run", "--context-tokens", "1024", "--arms", "control",
         "--out", str(out)]
    )
    r = json.loads(out.read_text().splitlines()[0])
    assert r["stage_timing"] is False
    assert r["warm_repeat"] is False


def test_stage_timing_arms_route_probe_env_marker(env_levers):
    # main() must arm MTPLX_ROUTE_STAGE_PROBE + MTPLX_DSV41_STAGE_TIMING before the
    # mtplx import when --stage-timing is requested (the route probe reads ENABLED
    # at import; the merged route_stage breakdown depends on it).  Assert on the
    # helper wiring, not a GPU run: the arming block keys off args.stage_timing.
    args = env_levers.build_parser().parse_args(
        ["--out", "/dev/null", "--stage-timing"]
    )
    assert args.stage_timing is True
    # the two module-level env names the arming block sets are stable constants.
    assert env_levers.PROBE_ENV == "MTPLX_ROUTE_STAGE_PROBE"
    assert env_levers.STAGE_TIMING_ENV == "MTPLX_DSV41_STAGE_TIMING"


def test_warm_and_stage_timing_pass_helpers_exist(env_levers):
    # The GPU-window pass builders are present with the documented signatures so a
    # window failure is never a missing-helper AttributeError (the W11 class of
    # crash the dry-run gates guard against).
    import inspect

    warm = inspect.signature(env_levers._warm_repeat_pass)
    assert set(warm.parameters) == {
        "model", "ops", "mem_probe", "prompt_ids", "steps", "cold_ids"
    }
    st = inspect.signature(env_levers._stage_timing_pass)
    assert set(st.parameters) == {"model", "ops", "prompt_ids", "steps"}
