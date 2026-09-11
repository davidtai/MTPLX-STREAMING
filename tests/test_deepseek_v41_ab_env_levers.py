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
    switch_fastpath / switch_fastpath_b / attn_compile / attn_win_memo /
    device_route / both / all_levers / stack_a / head_bf16 / head_mxfp8 /
    head_q8), including arm independence (each arm force-unsets the keys it does
    not set, and the W40 load-time head codec MTPLX_DSV41_HEAD_MODE, the W44
    device-route boolean, and the eight boolean per-forward levers never leak
    across each other);
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
_SB = "MTPLX_DSV41_SWITCH_SUBMIT"      # W42 / K23 var B: all-hit async submit
_AC = "MTPLX_DSV41_ATTN_COMPILE"       # W41 / K22: attention-chain compile
_WM = "MTPLX_DSV41_ATTN_WIN_MEMO"      # W45 / K24: sliding-window mask memo
_DR = "MTPLX_DSV41_DEVICE_ROUTE"       # W44 / K24: barrier-free all-hit device route
_PD = "MTPLX_DSV41_PREFILL_DENSE_EXPERTS"     # W51 / K26: prefill dense experts
_PDMR = "MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS"  # W51 / K26: per-expert row threshold
_PDB = "MTPLX_DSV41_PREFILL_DENSE_BATCH"      # W51 / K26: dequant batch size
_HM = "MTPLX_DSV41_HEAD_MODE"          # W40 / K21: load-time output-head codec
# The eight booleans all_levers turns on together. DEVICE_ROUTE (W44) and
# PREFILL_DENSE_EXPERTS (W51) are separate booleans tracked like the head codec:
# NOT part of all_levers, so they never join the "all-on" independence invariant.
# DEVICE_ROUTE is in stack_a; PREFILL_DENSE_EXPERTS is its own arm. The two dense
# value knobs (min_rows, batch) take an integer string and are left unset by every
# arm (the code default applies), so they are always recorded None.
_ALL_KEYS = (_OV, _LM, _SK, _HC, _FP, _SB, _AC, _WM)  # the eight booleans all_levers sets
_BOOL_AND_HEAD = _ALL_KEYS + (_DR, _PD, _PDMR, _PDB, _HM)  # every key a preset pins

ALL_ARMS = [
    "control",
    "shared_overlap",
    "layer_major",
    "sinkhorn_metal",
    "hc_compile",
    "switch_fastpath",
    "switch_fastpath_b",
    "attn_compile",
    "attn_win_memo",
    "device_route",
    "prefill_dense_experts",
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
    "switch_fastpath_b": {_FP, _SB},
    "attn_compile": {_AC},
    "attn_win_memo": {_WM},
    "device_route": set(),   # its only key is _DR, tracked in EXPECTED_DEVICE
    # W51: the dense-experts arm rides the layer-major schedule; its own boolean
    # (_PD) is tracked in EXPECTED_DENSE, so _LM is the only _ALL_KEYS member here.
    "prefill_dense_experts": {_LM},
    "both": {_OV, _LM},
    "all_levers": {_OV, _LM, _SK, _HC, _FP, _SB, _AC, _WM},
    # W42 window-14: pure fast path LEFT OUT (−13.4%).  W45: the window-mask memo
    # (byte-identical) joins the stack.  W44/window-19: device_route LEFT OUT (not
    # exact on the real model -- tracked in EXPECTED_DEVICE, off for stack_a).
    "stack_a": {_SK, _AC, _WM},
    "head_bf16": set(),
    "head_mxfp8": set(),
    "head_q8": set(),
}

# The device-route boolean each arm pins (W44 K24; separate from _ALL_KEYS because
# it is not part of all_levers). Only the standalone device_route arm sets it --
# W44/window-19 showed it is NOT exact on the real model (unpinned deferred gather
# vs mid-decode slot recycling), so it is OUT of stack_a until parity is clean.
EXPECTED_DEVICE = {arm: (arm == "device_route") for arm in ALL_ARMS}

# The prefill-dense-experts boolean each arm pins (W51 K26; separate from
# _ALL_KEYS, not part of all_levers). Only its own arm sets it.
EXPECTED_DENSE = {arm: (arm == "prefill_dense_experts") for arm in ALL_ARMS}

# The head-codec value each arm pins on MTPLX_DSV41_HEAD_MODE (None = force-unset).
EXPECTED_HEAD = {
    "control": None,
    "shared_overlap": None,
    "layer_major": None,
    "sinkhorn_metal": None,
    "hc_compile": None,
    "switch_fastpath": None,
    "switch_fastpath_b": None,
    "attn_compile": None,
    "attn_win_memo": None,
    "device_route": None,
    "prefill_dense_experts": None,
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
    # the device-route boolean is set for exactly its arms (device_route, stack_a).
    if EXPECTED_DEVICE[arm]:
        assert os.environ.get(_DR) == "1", f"{arm}: {_DR} should be '1'"
    else:
        assert _DR not in os.environ, f"{arm}: {_DR} should be force-unset"
    # the prefill-dense-experts boolean is set for exactly its own arm.
    if EXPECTED_DENSE[arm]:
        assert os.environ.get(_PD) == "1", f"{arm}: {_PD} should be '1'"
    else:
        assert _PD not in os.environ, f"{arm}: {_PD} should be force-unset"
    # the dense value knobs are never pinned by an arm (code default applies).
    assert _PDMR not in os.environ, f"{arm}: {_PDMR} should be force-unset"
    assert _PDB not in os.environ, f"{arm}: {_PDB} should be force-unset"
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
    # the K24 attn_win_memo arm leaves exactly its own key set.
    env_levers._apply_arm_env("all_levers")
    env_levers._apply_arm_env("attn_win_memo")
    assert os.environ.get(_WM) == "1"
    assert all(k not in os.environ for k in set(_ALL_KEYS) - {_WM})


def test_prefill_dense_arm_sets_only_its_keys_and_clears(env_levers):
    # The dense arm rides layer-major and sets its own boolean; nothing else.
    env_levers._apply_arm_env("all_levers")
    env_levers._apply_arm_env("prefill_dense_experts")
    assert os.environ.get(_PD) == "1"
    assert os.environ.get(_LM) == "1"
    assert all(k not in os.environ for k in set(_ALL_KEYS) - {_LM})
    assert _DR not in os.environ and _HM not in os.environ
    # the value knobs stay unset (code default), even for the dense arm.
    assert _PDMR not in os.environ and _PDB not in os.environ
    # a later arm clears the dense boolean (arms are independent).
    env_levers._apply_arm_env("control")
    assert _PD not in os.environ


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
    os.environ[_PD] = "bogus"
    os.environ[_PDMR] = "bogus"
    receipts = _run_dry_main(env_levers, tmp_path / "receipts.jsonl")
    for r in receipts:
        assert r["dry_run"] is True
        # _ALL_KEYS booleans only (device_route + head tracked separately).
        on = {k for k, v in r["arm_env"].items() if v == "1" and k in _ALL_KEYS}
        assert on == EXPECTED_ON[r["arm"]], r["arm"]
        # keys not in this arm must be recorded as unset (None), never "bogus".
        for k in _ALL_KEYS:
            if k not in EXPECTED_ON[r["arm"]]:
                assert r["arm_env"][k] is None, (r["arm"], k)
        # the device-route boolean and the head codec are recorded per arm.
        assert (r["arm_env"].get(_DR) == "1") == EXPECTED_DEVICE[r["arm"]], r["arm"]
        assert r["arm_env"].get(_HM) == EXPECTED_HEAD[r["arm"]], r["arm"]
        # the prefill-dense boolean is recorded per arm; its value knobs stay None.
        assert (r["arm_env"].get(_PD) == "1") == EXPECTED_DENSE[r["arm"]], r["arm"]
        assert r["arm_env"].get(_PDMR) is None, r["arm"]
        assert r["arm_env"].get(_PDB) is None, r["arm"]


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


# --------------------------------------------------------------------------
# W47: --prefill-stage-timing parser + dry-run flow + a small real forward
# --------------------------------------------------------------------------


def test_parser_has_prefill_stage_timing(env_levers):
    args = env_levers.build_parser().parse_args(["--out", "/dev/null"])
    assert args.prefill_stage_timing is False
    on = env_levers.build_parser().parse_args(
        ["--out", "/dev/null", "--prefill-stage-timing"]
    )
    assert on.prefill_stage_timing is True


def test_dry_run_records_prefill_stage_timing_flag(env_levers, tmp_path):
    out = tmp_path / "r.jsonl"
    rc = env_levers.main(
        ["--dry-run", "--context-tokens", "16384", "--arms", "control", "layer_major",
         "--prefill-stage-timing", "--out", str(out)]
    )
    assert rc == 0
    rows = [json.loads(x) for x in out.read_text().splitlines() if x]
    assert [r["arm"] for r in rows] == ["control", "layer_major"]
    for r in rows:
        assert r["prefill_stage_timing"] is True
        assert r["context_tokens"] == 16384
    # the layer_major arm records the schedule env for the prefill pass to read.
    assert rows[1]["prefill_layer_major"] == "1"
    assert rows[0]["prefill_layer_major"] is None


def test_prefill_pass_helper_signature(env_levers):
    import inspect

    sig = inspect.signature(env_levers._prefill_stage_timing_pass)
    assert set(sig.parameters) == {"model", "ops", "prompt_ids"}


class _Ops:
    """Minimal MLXOps shim (real forward, CPU) for the pass function."""

    def input(self, ids_2d):
        return mx.array(ids_2d)

    def sync(self, logits):
        mx.eval(logits)

    def argmax_last(self, logits):
        return int(mx.argmax(logits[0, -1]).item())


def _tiny_model():
    from mlx.utils import tree_flatten, tree_unflatten
    from mtplx.models.deepseek_v41 import Model, ModelArgs

    args = ModelArgs(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=12, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5,
        sliding_window=8, window_size=8, swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )
    model = Model(args)
    mx.random.seed(1)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = 0.1 * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())
    return model, args


def test_prefill_stage_timing_pass_small_real_forward(env_levers, monkeypatch):
    # A small real prefill forward through the harness function on the tiny double
    # (chunk 4 over 12 tokens -> 3 chunks), proving the pass builds a valid
    # per-chunk prefill report and always tears the session down.
    monkeypatch.setenv("MTPLX_DSV41_PREFILL_CHUNK", "4")
    model, args = _tiny_model()
    report = env_levers._prefill_stage_timing_pass(
        model=model, ops=_Ops(), prompt_ids=list(range(12)),
    )
    assert report["kind"] == "prefill"
    assert report["schedule"] == "chunk_major"
    assert report["chunks"] == 3
    assert any(k.endswith(".score") for k in report["stages"])
    assert "moe.routed_switch" in report["stages"]
    # the session is closed after the pass (no leaked global probe).
    import mtplx.models.deepseek_v41_stage_timing as stime
    assert stime.active() is None
