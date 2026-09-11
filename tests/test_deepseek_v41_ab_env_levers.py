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
    both / all_levers), including arm independence (each arm force-unsets the
    keys it does not set);
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
_ALL_KEYS = (_OV, _LM, _SK, _HC)

ALL_ARMS = [
    "control",
    "shared_overlap",
    "layer_major",
    "sinkhorn_metal",
    "hc_compile",
    "both",
    "all_levers",
]

# The lever env keys each arm must leave set to "1" (every other key unset).
EXPECTED_ON = {
    "control": set(),
    "shared_overlap": {_OV},
    "layer_major": {_LM},
    "sinkhorn_metal": {_SK},
    "hc_compile": {_HC},
    "both": {_OV, _LM},
    "all_levers": {_OV, _LM, _SK, _HC},
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
    watched = _ALL_KEYS + ("MTPLX_ROUTE_STAGE_PROBE",)
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
    # Pre-pollute every lever key so we prove the arm force-unsets the ones it
    # does not set (arms are independent), not merely sets the ones it wants.
    for k in _ALL_KEYS:
        os.environ[k] = "bogus"
    env_levers._apply_arm_env(arm)
    for k in _ALL_KEYS:
        if k in EXPECTED_ON[arm]:
            assert os.environ.get(k) == "1", f"{arm}: {k} should be '1'"
        else:
            assert k not in os.environ, f"{arm}: {k} should be force-unset"


def test_apply_arm_env_independent_across_arms(env_levers):
    # all_levers on -> control must clear all four (no leakage between arms).
    env_levers._apply_arm_env("all_levers")
    assert all(os.environ.get(k) == "1" for k in _ALL_KEYS)
    env_levers._apply_arm_env("control")
    assert all(k not in os.environ for k in _ALL_KEYS)
    # a single-lever arm after all_levers leaves exactly one key set.
    env_levers._apply_arm_env("all_levers")
    env_levers._apply_arm_env("sinkhorn_metal")
    assert os.environ.get(_SK) == "1"
    assert all(k not in os.environ for k in (_OV, _LM, _HC))


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
    receipts = _run_dry_main(env_levers, tmp_path / "receipts.jsonl")
    for r in receipts:
        assert r["dry_run"] is True
        on = {k for k, v in r["arm_env"].items() if v == "1"}
        assert on == EXPECTED_ON[r["arm"]], r["arm"]
        # keys not in this arm must be recorded as unset (None), never "bogus".
        for k in _ALL_KEYS:
            if k not in EXPECTED_ON[r["arm"]]:
                assert r["arm_env"][k] is None, (r["arm"], k)


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
