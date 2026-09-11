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
    device_route / prefill_dense_experts / dense_min32 / dense_batch16 /
    dense_f32 / both / all_levers / stack_a / stack_b / head_bf16 / head_mxfp8 /
    head_q8 /
    score_bf16 / score_chunked / score_bf16_chunked / score_lean / prefill_fast /
    prefill_lean / selected_keys / prefill_lean_sel / softmax_kernel /
    prefill_lean_k28 / prefill_best / prefill_best_nok28 / decode_attn_kernel),
    including arm independence (each arm force-unsets the keys it
    does not set, and the W40 load-time head codec MTPLX_DSV41_HEAD_MODE, the W44
    device-route boolean, the W51 dense-experts boolean + its three value knobs
    (min_rows / batch / matmul_dtype), the three W50 prefill score-path keys
    MTPLX_DSV41_PREFILL_SCORE_DTYPE / MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK /
    MTPLX_DSV41_PREFILL_SCORE_PATH, the W58 K28 fused-softmax-kernel boolean
    MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL, the K27 MTPLX_DSV41_LAYOUT_FIX boolean (set
    by the W58 prefill_best* full-stack arms), and the eight boolean per-forward
    levers never leak across each other);
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
_VSB = "MTPLX_DSV41_VERIFY_SINGLE_BARRIER"  # W61 / K31: small-M verify one barrier/layer (default ON)
_PD = "MTPLX_DSV41_PREFILL_DENSE_EXPERTS"     # W51 / K26: prefill dense experts
_PDMR = "MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS"  # W51 / K26: per-expert row threshold
_PDB = "MTPLX_DSV41_PREFILL_DENSE_BATCH"      # W51 / K26: dequant batch size
_PDD = "MTPLX_DSV41_PREFILL_DENSE_MATMUL_DTYPE"  # W51 / K26: dense matmul dtype
_HM = "MTPLX_DSV41_HEAD_MODE"          # W40 / K21: load-time output-head codec
_SD = "MTPLX_DSV41_PREFILL_SCORE_DTYPE"      # W50 / K25: prefill score matmul dtype (bf16, lossy)
_SC = "MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK"  # W50 / K25: split-K online-softmax chunk width
_SP = "MTPLX_DSV41_PREFILL_SCORE_PATH"       # W50: score impl (lean = f32 pass-cut one-shot)
_SEL = "MTPLX_DSV41_SELECTED_KEYS"           # W59 / K30: prefill selected-key gather
_SFK = "MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL"  # W58 / K28: fused mask+sink+softmax Metal kernel
_LFX = "MTPLX_DSV41_LAYOUT_FIX"              # W56 / K27 F1: sorted routed gather (set by prefill_best*)
_DAK = "MTPLX_DSV41_DECODE_ATTN_KERNEL"  # W60 / K29: fused decode/verify MLA attention Metal kernel
_PWS = "MTPLX_DSV41_PIN_WORKING_SET"     # W64 / R3-pin: post-prefill pinned working set
_DRP = "MTPLX_DSV41_DEVICE_ROUTE_PINNED"  # W71 / K24 revived: pinned device route
# The eight booleans all_levers turns on together. DEVICE_ROUTE (W44) and
# PREFILL_DENSE_EXPERTS (W51) are separate booleans tracked like the head codec:
# NOT part of all_levers, so they never join the "all-on" independence invariant.
# DEVICE_ROUTE is in stack_a; PREFILL_DENSE_EXPERTS is its own arm (+ prefill_fast).
# The two dense value knobs (min_rows, batch) take an integer string and are left
# unset by every arm (the code default applies), so they are always recorded None.
# The three W50 score-path keys (_SD "bf16", _SC an int width, _SP "lean") are
# value-taking like _HM, tracked separately and never in all_levers.
_ALL_KEYS = (_OV, _LM, _SK, _HC, _FP, _SB, _AC, _WM)  # the eight booleans all_levers sets
_BOOL_AND_HEAD = _ALL_KEYS + (_DR, _PD, _PDMR, _PDB, _PDD, _HM)  # every pre-W50 key a preset pins
# + the three W50 score-path keys + the W59 K30 selected-key gather boolean + the
# W58 K28 fused-softmax-kernel boolean + the K27 layout_fix boolean (the W58
# prefill_best* full-stack arms set it) + the W60 K29 decode-attention-kernel boolean.
_ALL_WATCHED = _BOOL_AND_HEAD + (_SD, _SC, _SP, _SEL, _SFK, _LFX, _DAK, _VSB, _PWS, _DRP)

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
    "pin_ws",
    "device_route_pinned",
    "verify_single_barrier",
    "prefill_dense_experts",
    "dense_min32",
    "dense_batch16",
    "dense_f32",
    "both",
    "all_levers",
    "stack_a",
    "stack_b",
    "head_bf16",
    "head_mxfp8",
    "head_q8",
    "score_bf16",
    "score_chunked",
    "score_bf16_chunked",
    "score_lean",
    "prefill_fast",
    "prefill_lean",
    "selected_keys",
    "prefill_lean_sel",
    "softmax_kernel",
    "prefill_lean_k28",
    "prefill_best",
    "prefill_best_nok28",
    "decode_attn_kernel",
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
    # W64 pin_ws: only _PWS (tracked in EXPECTED_PIN_WS). W71 device_route_pinned:
    # _PWS + _DR + _DRP (tracked in EXPECTED_PIN_WS / EXPECTED_DEVICE / EXPECTED_DEVICE_PINNED).
    "pin_ws": set(),
    "device_route_pinned": set(),
    "verify_single_barrier": set(),  # its only key is _VSB, tracked in EXPECTED_VERIFY
    # W51: the dense-experts arms all ride layer-major; the dense boolean (_PD) and
    # the value knobs are tracked separately, so _LM is the only _ALL_KEYS member.
    "prefill_dense_experts": {_LM},
    "dense_min32": {_LM},
    "dense_batch16": {_LM},
    "dense_f32": {_LM},
    "both": {_OV, _LM},
    "all_levers": {_OV, _LM, _SK, _HC, _FP, _SB, _AC, _WM},
    # W42 window-14: pure fast path LEFT OUT (−13.4%).  W45: the window-mask memo
    # (byte-identical) joins the stack.  W44/window-19: device_route LEFT OUT (not
    # exact on the real model -- tracked in EXPECTED_DEVICE, off for stack_a).
    "stack_a": {_SK, _AC, _WM},
    "stack_b": {_SK, _AC, _WM},  # stack_a + selected_keys (K30 tracked in EXPECTED_SELECTED)
    "head_bf16": set(),
    "head_mxfp8": set(),
    "head_q8": set(),
    "score_bf16": set(),
    "score_chunked": set(),
    "score_bf16_chunked": set(),
    "score_lean": set(),
    # W50+W51: the stacked prefill candidates ride the layer-major schedule; their
    # _PD (dense) is in EXPECTED_DENSE and their score keys in EXPECTED_SCORE_*, so
    # _LM is the only _ALL_KEYS member here.
    "prefill_fast": {_LM},
    "prefill_lean": {_LM},
    # W59 K30: selected_keys is a standalone boolean (tracked in EXPECTED_SELECTED,
    # not in _ALL_KEYS); prefill_lean_sel rides the prefill_lean stack (_LM here,
    # _PD in EXPECTED_DENSE, lean in EXPECTED_SCORE_PATH) plus the K30 boolean.
    "selected_keys": set(),
    "prefill_lean_sel": {_LM},
    # W58 K28: standalone kernel arm sets no boolean lever (its only key is _SFK,
    # tracked in EXPECTED_SOFTMAX_KERNEL); the stacked arm rides layer-major.
    "softmax_kernel": set(),
    "prefill_lean_k28": {_LM},
    # W58 full-stack arms: layer-major is the only _ALL_KEYS boolean; dense (_PD),
    # layout_fix (_LFX), score_path (_SP) and the K28 kernel (_SFK) are tracked
    # separately.
    "prefill_best": {_LM},
    "prefill_best_nok28": {_LM},
    # W60 K29: standalone decode-attention kernel arm sets no _ALL_KEYS boolean
    # (its only key is _DAK, tracked in EXPECTED_DECODE_ATTN_KERNEL).
    "decode_attn_kernel": set(),
}

# The device-route boolean each arm pins (W44 K24; separate from _ALL_KEYS because
# it is not part of all_levers). Only the standalone device_route arm sets it --
# W44/window-19 showed it is NOT exact on the real model (unpinned deferred gather
# vs mid-decode slot recycling), so it is OUT of stack_a until parity is clean.
# W71: device_route_pinned ALSO pins _DR ("1") -- it arms the backbone cold
# recovery -- alongside its own _DRP + _PWS, so _DR is set for both arms.
EXPECTED_DEVICE = {arm: (arm in ("device_route", "device_route_pinned")) for arm in ALL_ARMS}
# W71 K24-revived: the pinned-device-route boolean (_DRP). Only device_route_pinned.
EXPECTED_DEVICE_PINNED = {arm: (arm == "device_route_pinned") for arm in ALL_ARMS}
# W64/W71: the pinned-working-set value (_PWS). pin_ws + device_route_pinned pin "all".
EXPECTED_PIN_WS = {
    arm: ("all" if arm in ("pin_ws", "device_route_pinned") else None)
    for arm in ALL_ARMS
}
# W61 K31: verify single-barrier (default ON in code; the arm pins it explicitly).
EXPECTED_VERIFY = {arm: (arm == "verify_single_barrier") for arm in ALL_ARMS}

# The prefill-dense-experts boolean each arm pins (W51 K26; separate from
# _ALL_KEYS, not part of all_levers). Every dense arm sets it -- W51's sweeps plus
# the W50+W51 prefill stacks (prefill_fast / prefill_lean).
_DENSE_ARMS = (
    "prefill_dense_experts", "dense_min32", "dense_batch16", "dense_f32",
    "prefill_fast", "prefill_lean", "prefill_lean_sel", "prefill_lean_k28",
    "prefill_best", "prefill_best_nok28",
)
EXPECTED_DENSE = {arm: (arm in _DENSE_ARMS) for arm in ALL_ARMS}
# The dense value knobs each arm pins (None = force-unset / code default). Only the
# W51 sweep arms set them; the plain dense arm and the prefill stacks leave all
# three at the code default.
EXPECTED_MIN_ROWS = {arm: None for arm in ALL_ARMS}
EXPECTED_MIN_ROWS["dense_min32"] = "32"
EXPECTED_BATCH = {arm: None for arm in ALL_ARMS}
EXPECTED_BATCH["dense_batch16"] = "16"
EXPECTED_MATMUL_DTYPE = {arm: None for arm in ALL_ARMS}
EXPECTED_MATMUL_DTYPE["dense_f32"] = "f32"

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
    "pin_ws": None,
    "device_route_pinned": None,
    "verify_single_barrier": None,
    "prefill_dense_experts": None,
    "dense_min32": None,
    "dense_batch16": None,
    "dense_f32": None,
    "both": None,
    "all_levers": None,
    "stack_a": "bf16",
    "stack_b": "bf16",
    "head_bf16": "bf16",
    "head_mxfp8": "mxfp8",
    "head_q8": "q8",
    "score_bf16": None,
    "score_chunked": None,
    "score_bf16_chunked": None,
    "score_lean": None,
    "prefill_fast": None,
    "prefill_lean": None,
    "selected_keys": None,
    "prefill_lean_sel": None,
    "softmax_kernel": None,
    "prefill_lean_k28": None,
    "prefill_best": None,
    "prefill_best_nok28": None,
    "decode_attn_kernel": None,
}

# The W50 prefill score-path values each arm pins (None = force-unset). _SD is the
# QK^T/PV matmul dtype ("bf16"); _SC the split-K chunk width ("2048"); _SP the score
# impl ("lean").  Off on every pre-W50 arm; set on the matching score/stacked arms.
EXPECTED_SCORE_DTYPE = {arm: None for arm in ALL_ARMS}
EXPECTED_SCORE_DTYPE["score_bf16"] = "bf16"
EXPECTED_SCORE_DTYPE["score_bf16_chunked"] = "bf16"
EXPECTED_SCORE_DTYPE["prefill_fast"] = "bf16"

EXPECTED_SCORE_CHUNK = {arm: None for arm in ALL_ARMS}
EXPECTED_SCORE_CHUNK["score_chunked"] = "2048"
EXPECTED_SCORE_CHUNK["score_bf16_chunked"] = "2048"
EXPECTED_SCORE_CHUNK["prefill_fast"] = "2048"

EXPECTED_SCORE_PATH = {arm: None for arm in ALL_ARMS}
EXPECTED_SCORE_PATH["score_lean"] = "lean"
EXPECTED_SCORE_PATH["prefill_lean"] = "lean"
EXPECTED_SCORE_PATH["prefill_lean_sel"] = "lean"

# The W59 K30 selected-key gather boolean each arm pins (separate from _ALL_KEYS,
# not part of all_levers -- like the device-route / dense booleans).
EXPECTED_SELECTED = {arm: "1" if arm in ("selected_keys", "prefill_lean_sel", "stack_b") else None
                     for arm in ALL_ARMS}
EXPECTED_SCORE_PATH["prefill_lean_k28"] = "lean"
EXPECTED_SCORE_PATH["prefill_best"] = "lean"
EXPECTED_SCORE_PATH["prefill_best_nok28"] = "lean"

# The W58 K28 fused-softmax-kernel boolean each arm pins ("1" or None = force-unset).
# Set by the standalone kernel arm, the prefill_lean_k28 stack, and prefill_best
# (its no-kernel twin prefill_best_nok28 leaves it unset -- that is the A/B).
EXPECTED_SOFTMAX_KERNEL = {arm: None for arm in ALL_ARMS}
EXPECTED_SOFTMAX_KERNEL["softmax_kernel"] = "1"
EXPECTED_SOFTMAX_KERNEL["prefill_lean_k28"] = "1"
EXPECTED_SOFTMAX_KERNEL["prefill_best"] = "1"

# The K27 layout_fix boolean each arm pins ("1" or None). Only the W58 full-stack
# arms (prefill_best + its no-kernel twin) set it among the arms this test covers.
EXPECTED_LAYOUT = {arm: None for arm in ALL_ARMS}
EXPECTED_LAYOUT["prefill_best"] = "1"
EXPECTED_LAYOUT["prefill_best_nok28"] = "1"

# The W60 K29 decode-attention-kernel boolean each arm pins ("1" or None =
# force-unset).  Only the standalone decode_attn_kernel arm sets it (LEFT OUT of
# stack_a until the MTPLX_GPU_PARITY window is clean).
EXPECTED_DECODE_ATTN_KERNEL = {arm: None for arm in ALL_ARMS}
EXPECTED_DECODE_ATTN_KERNEL["decode_attn_kernel"] = "1"


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
    watched = _ALL_WATCHED + ("MTPLX_ROUTE_STAGE_PROBE",)
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
    # Pre-pollute every lever key (booleans + dense + head + the two W50 score
    # keys) so we prove the arm force-unsets the ones it does not set (arms are
    # independent), not merely sets the ones it wants.
    for k in _ALL_WATCHED:
        os.environ[k] = "bogus"
    env_levers._apply_arm_env(arm)
    for k in _ALL_KEYS:
        if k in EXPECTED_ON[arm]:
            assert os.environ.get(k) == "1", f"{arm}: {k} should be '1'"
        else:
            assert k not in os.environ, f"{arm}: {k} should be force-unset"
    # the device-route boolean is set for exactly its arms (device_route +
    # device_route_pinned, which arms the backbone recovery too).
    if EXPECTED_DEVICE[arm]:
        assert os.environ.get(_DR) == "1", f"{arm}: {_DR} should be '1'"
    else:
        assert _DR not in os.environ, f"{arm}: {_DR} should be force-unset"
    # W71: the pinned-device-route boolean (only device_route_pinned).
    if EXPECTED_DEVICE_PINNED[arm]:
        assert os.environ.get(_DRP) == "1", f"{arm}: {_DRP} should be '1'"
    else:
        assert _DRP not in os.environ, f"{arm}: {_DRP} should be force-unset"
    # W64/W71: the pinned-working-set value (pin_ws + device_route_pinned -> "all").
    assert os.environ.get(_PWS) == EXPECTED_PIN_WS[arm], f"{arm}: {_PWS}"
    # the verify single-barrier boolean is set for exactly its arm (default ON in
    # the code; the preset pins it explicitly so a "0" baseline can A/B the delta).
    if EXPECTED_VERIFY[arm]:
        assert os.environ.get(_VSB) == "1", f"{arm}: {_VSB} should be '1'"
    else:
        assert _VSB not in os.environ, f"{arm}: {_VSB} should be force-unset"
    # the prefill-dense-experts boolean is set for exactly the dense arms.
    if EXPECTED_DENSE[arm]:
        assert os.environ.get(_PD) == "1", f"{arm}: {_PD} should be '1'"
    else:
        assert _PD not in os.environ, f"{arm}: {_PD} should be force-unset"
    # the dense value knobs pin their sweep value or are force-unset (code default).
    assert os.environ.get(_PDMR) == EXPECTED_MIN_ROWS[arm], f"{arm}: {_PDMR}"
    assert os.environ.get(_PDB) == EXPECTED_BATCH[arm], f"{arm}: {_PDB}"
    assert os.environ.get(_PDD) == EXPECTED_MATMUL_DTYPE[arm], f"{arm}: {_PDD}"
    # the head codec pins its value (a string), or is force-unset when None.
    if EXPECTED_HEAD[arm] is None:
        assert _HM not in os.environ, f"{arm}: {_HM} should be force-unset"
    else:
        assert os.environ.get(_HM) == EXPECTED_HEAD[arm], f"{arm}: {_HM}"
    # the W50 score-path keys pin their value (a string), or are force-unset (None).
    for key, expected in (
        (_SD, EXPECTED_SCORE_DTYPE[arm]),
        (_SC, EXPECTED_SCORE_CHUNK[arm]),
        (_SP, EXPECTED_SCORE_PATH[arm]),
        (_SFK, EXPECTED_SOFTMAX_KERNEL[arm]),
        (_LFX, EXPECTED_LAYOUT[arm]),
        (_DAK, EXPECTED_DECODE_ATTN_KERNEL[arm]),
    ):
        if expected is None:
            assert key not in os.environ, f"{arm}: {key} should be force-unset"
        else:
            assert os.environ.get(key) == expected, f"{arm}: {key} should be {expected!r}"
    # the W59 K30 selected-key gather boolean is set for exactly its arms.
    if EXPECTED_SELECTED[arm] is None:
        assert _SEL not in os.environ, f"{arm}: {_SEL} should be force-unset"
    else:
        assert os.environ.get(_SEL) == EXPECTED_SELECTED[arm], f"{arm}: {_SEL}"


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
    # the plain dense arm leaves all value knobs unset (code default applies).
    assert _PDMR not in os.environ and _PDB not in os.environ and _PDD not in os.environ
    # the sweep arms each pin exactly one value knob.
    env_levers._apply_arm_env("dense_min32")
    assert os.environ.get(_PDMR) == "32" and _PDB not in os.environ and _PDD not in os.environ
    env_levers._apply_arm_env("dense_f32")
    assert os.environ.get(_PDD) == "f32" and _PDMR not in os.environ
    # a later arm clears the dense boolean + knobs (arms are independent).
    env_levers._apply_arm_env("control")
    assert _PD not in os.environ and _PDD not in os.environ


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
    os.environ[_PDB] = "bogus"
    os.environ[_PDD] = "bogus"
    os.environ[_SD] = "bogus"
    os.environ[_SC] = "bogus"
    os.environ[_SP] = "bogus"
    os.environ[_SEL] = "bogus"
    os.environ[_SFK] = "bogus"
    os.environ[_LFX] = "bogus"
    os.environ[_DAK] = "bogus"
    os.environ[_PWS] = "bogus"
    os.environ[_DRP] = "bogus"
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
        # W71: the pinned-device-route boolean + the pinned-working-set value.
        assert (r["arm_env"].get(_DRP) == "1") == EXPECTED_DEVICE_PINNED[r["arm"]], r["arm"]
        assert r["arm_env"].get(_PWS) == EXPECTED_PIN_WS[r["arm"]], r["arm"]
        assert (r["arm_env"].get(_VSB) == "1") == EXPECTED_VERIFY[r["arm"]], r["arm"]
        assert r["arm_env"].get(_HM) == EXPECTED_HEAD[r["arm"]], r["arm"]
        # the prefill-dense boolean + value knobs are recorded per arm.
        assert (r["arm_env"].get(_PD) == "1") == EXPECTED_DENSE[r["arm"]], r["arm"]
        assert r["arm_env"].get(_PDMR) == EXPECTED_MIN_ROWS[r["arm"]], r["arm"]
        assert r["arm_env"].get(_PDB) == EXPECTED_BATCH[r["arm"]], r["arm"]
        assert r["arm_env"].get(_PDD) == EXPECTED_MATMUL_DTYPE[r["arm"]], r["arm"]
        # the three W50 score-path keys + the W58 K28 kernel boolean per arm.
        assert r["arm_env"].get(_SD) == EXPECTED_SCORE_DTYPE[r["arm"]], r["arm"]
        assert r["arm_env"].get(_SC) == EXPECTED_SCORE_CHUNK[r["arm"]], r["arm"]
        assert r["arm_env"].get(_SP) == EXPECTED_SCORE_PATH[r["arm"]], r["arm"]
        # the W59 K30 selected-key gather boolean is recorded per arm.
        assert r["arm_env"].get(_SEL) == EXPECTED_SELECTED[r["arm"]], r["arm"]
        assert r["arm_env"].get(_SFK) == EXPECTED_SOFTMAX_KERNEL[r["arm"]], r["arm"]
        assert r["arm_env"].get(_LFX) == EXPECTED_LAYOUT[r["arm"]], r["arm"]
        assert r["arm_env"].get(_DAK) == EXPECTED_DECODE_ATTN_KERNEL[r["arm"]], r["arm"]


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
