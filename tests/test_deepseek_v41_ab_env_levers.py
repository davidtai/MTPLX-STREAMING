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
_DC = "MTPLX_DSV41_DRAFT_COMPILE"      # W65 / K33: DSpark draft-block tape collapse
_DHB = "MTPLX_DSV41_DRAFT_HEAD_BF16"   # W103: DSpark draft-head fp32-cast fix (wired W104)
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
_MLXBUF = "MLX_MAX_MB_PER_BUFFER"        # K14 / W63: MLX command-buffer MB cap passthrough
_PWS = "MTPLX_DSV41_PIN_WORKING_SET"     # W64 / R3-pin: post-prefill pinned working set
_DRP = "MTPLX_DSV41_DEVICE_ROUTE_PINNED"  # W71 / K24 revived: pinned device route
_SSP = "MTPLX_DSV41_SINGLE_SLOT_POOL"     # W87: merged scan-resistant slot pool
_WOAC = "MTPLX_DSV41_ATTN_WO_A_CACHE"     # W97: cached dequantized o-LoRA wo_a
_ACC = "MTPLX_DSV41_ATTN_CORE_COMPILE"    # W97: fixed-shape core compile (rounding-class)
_ALC = "MTPLX_DSV41_ATTN_LEAN_CASTS"      # W99: byte-identical cast lean
_AFP = "MTPLX_DSV41_ATTN_FUSED_PROJ"      # W101 / K36: fused projection-chain kernels (rounding-class)
_KCG = "MTPLX_DSV41_KV_CHUNK_GROW"       # W73 / K32: chunk-grown KV append backing
_SS = "MTPLX_DSV41_SMALL_STAGES_FUSED"   # W91 / K35: fused per-layer small stages
_HPK = "MTPLX_DSV41_HC_PREMIX_KERNEL"    # W91 / K35: GPU-only fused HC-premix kernel
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
_ALL_WATCHED = _BOOL_AND_HEAD + (_SD, _SC, _SP, _SEL, _SFK, _LFX, _DAK, _VSB, _MLXBUF, _DC, _DHB, _PWS, _DRP, _KCG, _SS, _HPK)

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
    "draft_compile",
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
    "prefill_best_sel",
    "mlx_buffer_500",
    "decode_attn_kernel",
    "kv_chunk_grow",
    "prefill_lean_sel_chunk",
    "cell16k",
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
    "draft_compile": set(),  # its only key is _DC, tracked in EXPECTED_DRAFT
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
    # W63: prefill_best_nok28 + selected_keys -- rides layer-major (dense in
    # EXPECTED_DENSE, lean in EXPECTED_SCORE_PATH, layout_fix in EXPECTED_LAYOUT,
    # K30 in EXPECTED_SELECTED).
    "prefill_best_sel": {_LM},
    # K14 (W63): the MLX buffer-cap passthrough arm sets no MTPLX lever (its only
    # key is _MLXBUF, tracked in EXPECTED_MLX_BUFFER).
    "mlx_buffer_500": set(),
    # W60 K29: standalone decode-attention kernel arm sets no _ALL_KEYS boolean
    # (its only key is _DAK, tracked in EXPECTED_DECODE_ATTN_KERNEL).
    "decode_attn_kernel": set(),
    # W73 K32: standalone chunk-grow arm sets no _ALL_KEYS boolean (its only key is
    # _KCG, tracked in EXPECTED_KV_CHUNK_GROW); prefill_lean_sel_chunk rides the
    # prefill_lean_sel stack (_LM here; _PD, lean, _SEL, _KCG tracked separately).
    "kv_chunk_grow": set(),
    "prefill_lean_sel_chunk": {_LM},
    # the standard 16K cell: prefill lane (layer-major; _PD/_SP/_SEL/_LFX/_KCG
    # tracked separately) + decode lane stack_a (Sinkhorn + attn compile + window
    # memo + head bf16 in EXPECTED_HEAD).
    "cell16k": {_LM, _SK, _AC, _WM},
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
# W65 K33: DSpark draft-block compile (separate boolean, not in _ALL_KEYS -- like
# device_route / verify_single_barrier; only the standalone draft_compile arm sets it).
EXPECTED_DRAFT = {arm: (arm == "draft_compile") for arm in ALL_ARMS}
# W104: DSpark draft-head bf16 fix (separate boolean, not in _ALL_KEYS -- like _DC).
# NO arm in ALL_ARMS sets it (the only arms that do are the ring composites
# cell16k_ring_draft / cell16k_ring_v2_draft, checked in test_cell16k_ring_composite_arms),
# so every ALL_ARMS arm must force-unset it (proves a parent-shell export can't survive).
EXPECTED_DRAFT_HEAD_BF16 = {arm: False for arm in ALL_ARMS}
# W91 K35: none of ALL_ARMS (up to cell16k) arms the fused-small-stages or the
# HC-premix-kernel booleans -- K35 is rounding-class on GPU (window-37, null win),
# so its ONLY arm is the isolation A/B small_stages_fused (verified in
# test_cell16k_ring_composite_arms); it is in NO composite arm. So both are
# force-unset for every arm here (proves an arm can't leave a stale K35 export set).
EXPECTED_SMALL_STAGES = {arm: False for arm in ALL_ARMS}
EXPECTED_PREMIX_KERNEL = {arm: False for arm in ALL_ARMS}

# The prefill-dense-experts boolean each arm pins (W51 K26; separate from
# _ALL_KEYS, not part of all_levers). Every dense arm sets it -- W51's sweeps plus
# the W50+W51 prefill stacks (prefill_fast / prefill_lean).
_DENSE_ARMS = (
    "prefill_dense_experts", "dense_min32", "dense_batch16", "dense_f32",
    "prefill_fast", "prefill_lean", "prefill_lean_sel", "prefill_lean_k28",
    "prefill_best", "prefill_best_nok28", "prefill_best_sel", "prefill_lean_sel_chunk",
    "cell16k",
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
    "draft_compile": None,
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
    "prefill_best_sel": None,
    "mlx_buffer_500": None,
    "decode_attn_kernel": None,
    "kv_chunk_grow": None,
    "prefill_lean_sel_chunk": None,
    "cell16k": "bf16",
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
EXPECTED_SCORE_PATH["prefill_lean_sel_chunk"] = "lean"

# The W59 K30 selected-key gather boolean each arm pins (separate from _ALL_KEYS,
# not part of all_levers -- like the device-route / dense booleans). W63 adds
# prefill_best_sel (prefill_best_nok28 + selected_keys); W73 adds
# prefill_lean_sel_chunk (prefill_lean_sel + kv_chunk_grow).
EXPECTED_SELECTED = {
    arm: "1"
    if arm in (
        "selected_keys", "prefill_lean_sel", "stack_b", "prefill_best_sel",
        "prefill_lean_sel_chunk", "cell16k",
    )
    else None
    for arm in ALL_ARMS
}
EXPECTED_SCORE_PATH["prefill_lean_k28"] = "lean"
EXPECTED_SCORE_PATH["prefill_best"] = "lean"
EXPECTED_SCORE_PATH["prefill_best_nok28"] = "lean"
EXPECTED_SCORE_PATH["prefill_best_sel"] = "lean"
EXPECTED_SCORE_PATH["cell16k"] = "lean"

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
EXPECTED_LAYOUT["prefill_best_sel"] = "1"
EXPECTED_LAYOUT["cell16k"] = "1"

# The W60 K29 decode-attention-kernel boolean each arm pins ("1" or None =
# force-unset).  Only the standalone decode_attn_kernel arm sets it (LEFT OUT of
# stack_a until the MTPLX_GPU_PARITY window is clean).
EXPECTED_DECODE_ATTN_KERNEL = {arm: None for arm in ALL_ARMS}
EXPECTED_DECODE_ATTN_KERNEL["decode_attn_kernel"] = "1"

# K14 (W63): the MLX command-buffer MB cap each arm pins (a positive-int string,
# None = MLX default). Only the standalone mlx_buffer_500 arm sets it (500 MB, the
# value the Qwen lane measured +1.6% at). It is an MLX passthrough, not an MTPLX
# lever, so no stacked arm carries it.
EXPECTED_MLX_BUFFER = {arm: None for arm in ALL_ARMS}
EXPECTED_MLX_BUFFER["mlx_buffer_500"] = "500"

# The W73 K32 chunk-grow boolean each arm pins ("1" or None = force-unset).  Set by
# the standalone kv_chunk_grow arm and the prefill_lean_sel_chunk stack (its A/B
# twin prefill_lean_sel leaves it unset -- that pair isolates the append fix).
EXPECTED_KV_CHUNK_GROW = {arm: None for arm in ALL_ARMS}
EXPECTED_KV_CHUNK_GROW["kv_chunk_grow"] = "1"
EXPECTED_KV_CHUNK_GROW["prefill_lean_sel_chunk"] = "1"
EXPECTED_KV_CHUNK_GROW["cell16k"] = "1"



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
# W79: the served profile's lever child_env MUST equal the cell16k preset
# --------------------------------------------------------------------------


def test_profile_child_env_equals_cell16k_preset(env_levers):
    """The deepseek-v41-mxfp4-75 profile ships the cell16k lever set as served
    defaults (W79).  Pin the profile child_env and the A/B preset to ONE source of
    truth -- the ab script's ``ARM_PRESETS['cell16k']`` -- so neither can drift
    from the other silently.  The preset pins EVERY lever key (most to None =
    force-unset); the profile carries only the armed ones plus its non-lever memory
    caps, so compare the profile's child_env RESTRICTED to the preset's lever key
    space against the preset's non-None (armed) entries.
    """
    from mtplx.expert_profiles import load_expert_profiles

    preset = env_levers.ARM_PRESETS["cell16k"]
    lever_keys = set(preset)  # every DSV4.1 lever env key the preset knows
    preset_armed = {k: v for k, v in preset.items() if v is not None}

    child_env = dict(load_expert_profiles()["deepseek-v41-mxfp4-75"].child_env)
    child_levers = {k: v for k, v in child_env.items() if k in lever_keys}

    # The drift guard: profile lever env == cell16k preset armed env, exactly.
    assert child_levers == preset_armed
    # Pin the preset itself to the known-good contract so it, too, cannot silently
    # change: the prefill lane (W30/W51/W50/W59/W73/W56) + the decode lane (W40/
    # W32/W41/W45).  Changing cell16k requires changing this and the profile JSON
    # together, on purpose.
    assert preset_armed == {
        "MTPLX_DSV41_PREFILL_LAYER_MAJOR": "1",
        "MTPLX_DSV41_PREFILL_DENSE_EXPERTS": "1",
        "MTPLX_DSV41_PREFILL_SCORE_PATH": "lean",
        "MTPLX_DSV41_SELECTED_KEYS": "1",
        "MTPLX_DSV41_KV_CHUNK_GROW": "1",
        "MTPLX_DSV41_LAYOUT_FIX": "1",
        "MTPLX_DSV41_HEAD_MODE": "bf16",
        "MTPLX_DSV41_SINKHORN_METAL": "1",
        "MTPLX_DSV41_ATTN_COMPILE": "1",
        "MTPLX_DSV41_ATTN_WIN_MEMO": "1",
    }


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
    # the K33 draft-block compile boolean is set for exactly its arm (draft_compile).
    if EXPECTED_DRAFT[arm]:
        assert os.environ.get(_DC) == "1", f"{arm}: {_DC} should be '1'"
    else:
        assert _DC not in os.environ, f"{arm}: {_DC} should be force-unset"
    # W104: the W103 draft-head-bf16 boolean is force-unset for every ALL_ARMS arm
    # (only the ring composites arm it; tested in test_cell16k_ring_composite_arms).
    if EXPECTED_DRAFT_HEAD_BF16[arm]:
        assert os.environ.get(_DHB) == "1", f"{arm}: {_DHB} should be '1'"
    else:
        assert _DHB not in os.environ, f"{arm}: {_DHB} should be force-unset"
    # W91 K35: the fused-small-stages + HC-premix-kernel booleans are force-unset for
    # every ALL_ARMS arm (the K35 arms are in the ring family, tested separately) --
    # proves a parent-shell K35 export cannot survive an arm application.
    if EXPECTED_SMALL_STAGES[arm]:
        assert os.environ.get(_SS) == "1", f"{arm}: {_SS} should be '1'"
    else:
        assert _SS not in os.environ, f"{arm}: {_SS} should be force-unset"
    if EXPECTED_PREMIX_KERNEL[arm]:
        assert os.environ.get(_HPK) == "1", f"{arm}: {_HPK} should be '1'"
    else:
        assert _HPK not in os.environ, f"{arm}: {_HPK} should be force-unset"
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
        (_KCG, EXPECTED_KV_CHUNK_GROW[arm]),
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
    # K14 (W63): the MLX buffer-cap passthrough pins its value or force-unsets it.
    if EXPECTED_MLX_BUFFER[arm] is None:
        assert _MLXBUF not in os.environ, f"{arm}: {_MLXBUF} should be force-unset"
    else:
        assert os.environ.get(_MLXBUF) == EXPECTED_MLX_BUFFER[arm], f"{arm}: {_MLXBUF}"


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
    # the K33 draft_compile arm sets only its own (non-_ALL_KEYS) boolean and
    # clears every _ALL_KEYS lever a prior arm left set.
    env_levers._apply_arm_env("all_levers")
    env_levers._apply_arm_env("draft_compile")
    assert os.environ.get(_DC) == "1"
    assert all(k not in os.environ for k in _ALL_KEYS)
    env_levers._apply_arm_env("control")
    assert _DC not in os.environ


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
    os.environ[_MLXBUF] = "bogus"
    os.environ[_DC] = "bogus"
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
        assert (r["arm_env"].get(_DC) == "1") == EXPECTED_DRAFT[r["arm"]], r["arm"]
        assert (r["arm_env"].get(_DHB) == "1") == EXPECTED_DRAFT_HEAD_BF16[r["arm"]], r["arm"]
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
        # K14 (W63): the MLX buffer-cap passthrough, in arm_env and as a
        # dedicated top-level receipt field.
        assert r["arm_env"].get(_MLXBUF) == EXPECTED_MLX_BUFFER[r["arm"]], r["arm"]
        assert r["mlx_max_mb_per_buffer"] == EXPECTED_MLX_BUFFER[r["arm"]], r["arm"]
        # W63 / K32: the device-sample flag is recorded (default off, no env set).
        assert r["device_sample"] is False, r["arm"]


def test_dry_run_mlx_buffer_arm_and_device_sample_flag(env_levers, tmp_path):
    # K14 (W63): the mlx_buffer_500 arm records the pinned MB cap; control clears
    # it. W63 / K32: --device-sample overrides the env default and is recorded on
    # every arm's receipt.
    out = tmp_path / "receipts.jsonl"
    rc = env_levers.main(
        [
            "--dry-run",
            "--context-tokens",
            "1024",
            "--arms",
            "control",
            "mlx_buffer_500",
            "--device-sample",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    receipts = {
        json.loads(line)["arm"]: json.loads(line)
        for line in out.read_text().splitlines()
        if line
    }
    assert receipts["control"]["mlx_max_mb_per_buffer"] is None
    assert receipts["control"]["arm_env"][_MLXBUF] is None
    assert receipts["mlx_buffer_500"]["mlx_max_mb_per_buffer"] == "500"
    assert receipts["mlx_buffer_500"]["arm_env"][_MLXBUF] == "500"
    # every non-buffer lever key stays unset on the buffer arm (arm independence).
    for k in _ALL_KEYS + (_DR, _PD, _HM, _SD, _SC, _SP, _SEL, _SFK, _LFX, _DAK):
        assert receipts["mlx_buffer_500"]["arm_env"][k] is None, k
    assert receipts["control"]["device_sample"] is True
    assert receipts["mlx_buffer_500"]["device_sample"] is True


def test_dry_run_device_sample_follows_env(env_levers, tmp_path, monkeypatch):
    # W63 / K32: with the --device-sample flag left at its None default, the
    # receipt follows MTPLX_DSV41_DEVICE_SAMPLE.
    monkeypatch.setenv("MTPLX_DSV41_DEVICE_SAMPLE", "1")
    out = tmp_path / "receipts.jsonl"
    rc = env_levers.main(["--dry-run", "--arms", "control", "--out", str(out)])
    assert rc == 0
    r = json.loads(out.read_text().splitlines()[0])
    assert r["device_sample"] is True


def test_parser_has_device_sample_default_none(env_levers):
    args = env_levers.build_parser().parse_args(["--out", "/dev/null"])
    assert args.device_sample is None
    on = env_levers.build_parser().parse_args(["--out", "/dev/null", "--device-sample"])
    assert on.device_sample is True
    off = env_levers.build_parser().parse_args(
        ["--out", "/dev/null", "--no-device-sample"]
    )
    assert off.device_sample is False


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
    assert {
        "model", "ops", "mem_probe", "prompt_ids", "steps", "cold_ids"
    } <= set(warm.parameters)
    # W113 MEDIUM-2: optional served-parity early-stop params, defaulting off so the
    # shipped call sites are unaffected.
    assert warm.parameters["stop_on_eos"].default is False
    assert warm.parameters["eos_id"].default is None
    st = inspect.signature(env_levers._stage_timing_pass)
    # W90 added optional keyword-only telemetry params (cooldown_s / util_sampler,
    # both defaulting to a no-op so the shipped call sites are unaffected).
    assert {"model", "ops", "prompt_ids", "steps"} <= set(st.parameters)
    assert st.parameters["cooldown_s"].default == 0.0
    assert st.parameters["util_sampler"].default is None


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


# --------------------------------------------------------------------------
# W94: the untimed decode pass (_generate) must clear the route probe IMMEDIATELY
# before the after_prefill ("before") snapshot, so the DECODE-scoped route_probe
# delta (end - after_prefill) is exact and never negative.  Regression: the clear
# used to run AFTER the before-snapshot, leaving prefill totals in "before" and
# decode-only in "after" -> prefill-dominated stages (hot.begin_split_route)
# deltaed NEGATIVE (window-37 ar-ring-ref sums_ns = -3.8e8).
# --------------------------------------------------------------------------
class _FakeRuntimeWithCache:
    """A minimal streamed-runtime double so _stream_counters_snapshot returns a real
    dict (not None): snapshot() yields the expert_cache block, and the route probe
    counts come from the module globals (independent of the runtime)."""

    def snapshot(self):
        return {"cache": {"expert_hits": 1, "expert_misses": 0, "bytes_read": 0,
                          "persistent_loads": 0, "transient_loads": 0}}


class _RouteBumpModel:
    """Fake model whose forward bumps a route-probe counter: a lot on the (first)
    prefill forward, a little on each decode forward -- so a delta that failed to
    exclude the prefill accumulation would go negative for the prefill-heavy stage."""

    _mtplx_expert_runtime = _FakeRuntimeWithCache()

    def __init__(self, rp, stage, barrier, prefill_bump, per_step_bump):
        self._rp = rp
        self._stage = stage
        self._barrier = barrier
        self._prefill_bump = prefill_bump
        self._per_step = per_step_bump
        self._did_prefill = False

    def make_cache(self):
        return {}

    def __call__(self, ids, cache=None):
        n = int(ids.shape[-1])
        bump = self._prefill_bump if not self._did_prefill else self._per_step
        # exercise both counter kinds: count()-only stage + a sums-bearing barrier.
        self._rp._COUNTS[self._stage] += bump
        self._rp._COUNTS[self._barrier] += bump
        self._rp._SUMS[self._barrier] += bump * 1000
        self._did_prefill = True
        return mx.zeros((1, n, 8))


class _ZeroSampler:
    # W106: the off-hot-path RSS/system-used sampler _generate starts/stops around
    # the whole generation.  A no-op stub for the CPU harness tests.
    def start(self):
        return None

    def stop(self):
        return None


class _ZeroMemProbe:
    def reset_peak(self):
        return None

    def peak_bytes(self):
        return 0

    def new_sampler(self):
        return _ZeroSampler()

    def memory_block(self, sampler):
        return {}


def test_generate_route_probe_delta_is_exact_and_non_negative(env_levers, monkeypatch):
    """The decode-scoped route_probe_counts / route_probe_sums_ns delta equals the
    direct DECODE count (prefill excluded) and is never negative -- the clear runs
    before the before-snapshot."""
    from mtplx import expert_route_probe as rp
    from mtplx.serve_stream_counters import stream_counters_delta

    STAGE = "hot.begin_split_route"   # the stage the buggy order deltaed negative
    BARRIER = "hot.eval_indices"
    PREFILL_BUMP = 500                # prefill accumulates a lot
    PER_STEP = 8
    STEPS = 6

    # Arm the probe as the launcher (MTPLX_ROUTE_STAGE_PROBE=1) does, so prefill
    # accumulates and BOTH snapshots include route_probe (the env-armed path).
    monkeypatch.setattr(rp, "ENABLED", True)
    rp._SUMS.clear()
    rp._COUNTS.clear()
    try:
        run = env_levers._generate(
            model=_RouteBumpModel(rp, STAGE, BARRIER, PREFILL_BUMP, PER_STEP),
            ops=_Ops(), mem_probe=_ZeroMemProbe(),
            prompt_ids=list(range(4)), steps=STEPS, stage_timing=True,
        )
        before = run["stream_after_prefill"]
        after = run["stream_end"]

        # The clear ran BEFORE the before-snapshot -> its route_probe baseline is
        # empty (the fix's signature; the buggy order left PREFILL_BUMP here).
        assert before.get("route_probe_counts", {}) == {}
        assert before.get("route_probe_sums_ns", {}) == {}

        d = stream_counters_delta(before, after, tokens=STEPS)
        counts = d["route_probe_counts"]
        sums = d["route_probe_sums_ns"]
        # delta == the direct DECODE count (prefill 500 excluded), NON-NEGATIVE.
        assert counts[STAGE] == STEPS * PER_STEP
        assert counts[BARRIER] == STEPS * PER_STEP
        assert sums[BARRIER] == STEPS * PER_STEP * 1000
        assert all(v >= 0 for v in counts.values()), counts
        assert all(v >= 0 for v in sums.values()), sums
    finally:
        rp._SUMS.clear()
        rp._COUNTS.clear()


# --------------------------------------------------------------------------
# W81: two composite stacking arms for window 34. Each must be EXACTLY the merged
# cell16k_ring key set plus one intended lever group, so a later edit to
# cell16k_ring propagates and no stray key drifts into the stack.
# --------------------------------------------------------------------------
def test_cell16k_ring_composite_arms(env_levers):
    presets = env_levers.ARM_PRESETS
    for name in ("cell16k_ring_draft", "cell16k_ring_v2_draft",
                 "cell16k_ring_pinned", "cell16k_ring_pool"):
        assert name in presets, f"{name} arm missing from ARM_PRESETS"
    ring = presets["cell16k_ring"]

    # W104: cell16k_ring_draft = cell16k_ring + BOTH DSpark draft-head levers
    # (K33 draft-block compile + the W103 draft-head bf16 fix).  Before W104 this arm
    # pinned MTPLX_DSV41_DRAFT_COMPILE only; DRAFT_COMPILE in isolation is still the
    # standalone `draft_compile` arm.
    expected_draft = dict(ring)
    expected_draft[_DC] = "1"
    expected_draft[_DHB] = "1"
    assert presets["cell16k_ring_draft"] == expected_draft, (
        "cell16k_ring_draft must equal cell16k_ring + MTPLX_DSV41_DRAFT_COMPILE=1 + "
        "MTPLX_DSV41_DRAFT_HEAD_BF16=1"
    )

    # W104: cell16k_ring_v2_draft = cell16k_ring_v2 + the two draft-head levers.  Built
    # off cell16k_ring_v2 (which is cell16k_ring + the v2 runner) so a later edit to
    # either propagates; no draft-MoE key exists (W104 traced the draft MoE to an
    # already barrier-free resident gather_qmm -- see W104_DRAFT_RESIDENT_MOE.md).
    expected_v2_draft = dict(presets["cell16k_ring_v2"])
    expected_v2_draft[_DC] = "1"
    expected_v2_draft[_DHB] = "1"
    assert presets["cell16k_ring_v2_draft"] == expected_v2_draft, (
        "cell16k_ring_v2_draft must equal cell16k_ring_v2 + MTPLX_DSV41_DRAFT_COMPILE=1 "
        "+ MTPLX_DSV41_DRAFT_HEAD_BF16=1"
    )

    # cell16k_ring_pinned = cell16k_ring + W64 pin + W71 pinned device route only.
    expected_pinned = dict(ring)
    expected_pinned[_PWS] = "all"
    expected_pinned[_DR] = "1"
    expected_pinned[_DRP] = "1"
    assert presets["cell16k_ring_pinned"] == expected_pinned, (
        "cell16k_ring_pinned must equal cell16k_ring + pin_working_set=all + "
        "device_route=1 + device_route_pinned=1"
    )

    # W91 K35 is ROUNDING-CLASS ON GPU (window-37: null perf, token-id sha differs),
    # so it is kept OUT of every composite arm.  The isolation A/B arm
    # small_stages_fused is the ONLY arm that carries it, and it sets ONLY
    # small_stages + sinkhorn (hermetic).
    assert "small_stages_fused" in presets, "small_stages_fused arm missing"
    # W101 reclaimed the name ``cell16k_ring_fused`` for the fused PROJECTION-CHAIN
    # stack (K36, NOT K35): it sets attn_fused_proj + K29 + wo_a + lean, never
    # small_stages (_SS).  The K35 "no composite carries small_stages" invariant is
    # still enforced by the general _SS loop below, which now also covers this arm.
    ssf = presets["small_stages_fused"]
    assert ssf[_SS] == "1" and ssf[_SK] == "1", "small_stages_fused missing its keys"
    assert all(v is None for k, v in ssf.items() if k not in (_SS, _SK)), (
        "small_stages_fused must set ONLY small_stages + sinkhorn (hermetic)"
    )
    # No composite (cell16k*) arm carries K35 small_stages; and the rounding-class
    # premix kernel is armed by NO arm (GPU-parity-gated).  Both K35 keys stay in the
    # master lever list (receipt arm_env + hermetic clear).
    assert _SS in env_levers.ALL_LEVER_ENVS and _HPK in env_levers.ALL_LEVER_ENVS
    for name, preset in presets.items():
        assert preset.get(_HPK) is None, f"{name} must not arm HC_PREMIX_KERNEL"
        if name != "small_stages_fused":
            assert preset.get(_SS) is None, (
                f"{name} must not arm K35 small_stages (rounding-class, no GPU win)"
            )

    # W87: cell16k_ring_pool = cell16k_ring + the single-slot pool only.  A pure
    # residency change (allocation identical); the direct A/B vs cell16k_ring
    # isolates the cold-start recovery.
    expected_pool = dict(ring)
    expected_pool[_SSP] = "1"
    assert presets["cell16k_ring_pool"] == expected_pool, (
        "cell16k_ring_pool must equal cell16k_ring + MTPLX_DSV41_SINGLE_SLOT_POOL=1"
    )
    # It differs from cell16k_ring ONLY by the single-slot-pool key.
    assert {
        k: v for k, v in presets["cell16k_ring_pool"].items() if v != ring.get(k)
    } == {_SSP: "1"}, "cell16k_ring_pool must touch only the single-slot-pool key"

    # W97: cell16k_ring_wo_a_cache = cell16k_ring + the wo_a-dequant cache ONLY.
    assert "cell16k_ring_wo_a_cache" in presets, "cell16k_ring_wo_a_cache arm missing"
    expected_woac = dict(ring)
    expected_woac[_WOAC] = "1"
    assert presets["cell16k_ring_wo_a_cache"] == expected_woac, (
        "cell16k_ring_wo_a_cache must equal cell16k_ring + MTPLX_DSV41_ATTN_WO_A_CACHE=1"
    )
    assert {
        k: v for k, v in presets["cell16k_ring_wo_a_cache"].items() if v != ring.get(k)
    } == {_WOAC: "1"}, "cell16k_ring_wo_a_cache must touch only the wo_a-cache key"
    # The standalone isolation arm sets ONLY selected_keys + wo_a_cache (hermetic).
    assert "wo_a_cache" in presets, "wo_a_cache isolation arm missing"
    woac = presets["wo_a_cache"]
    assert woac[_SEL] == "1" and woac[_WOAC] == "1", "wo_a_cache missing its keys"
    assert all(v is None for k, v in woac.items() if k not in (_SEL, _WOAC)), (
        "wo_a_cache must set ONLY selected_keys + wo_a_cache (hermetic)"
    )
    # W97 lever is in the master list (receipt arm_env + hermetic clear).
    assert _WOAC in env_levers.ALL_LEVER_ENVS

    # W97 core-compile arms (ROUNDING-CLASS -- not byte-identical; flagged elsewhere).
    assert _ACC in env_levers.ALL_LEVER_ENVS
    # isolation arm: ONLY selected_keys + attn_core_compile (hermetic).
    acc = presets["attn_core_compile"]
    assert acc[_SEL] == "1" and acc[_ACC] == "1", "attn_core_compile missing its keys"
    assert all(v is None for k, v in acc.items() if k not in (_SEL, _ACC)), (
        "attn_core_compile must set ONLY selected_keys + attn_core_compile (hermetic)"
    )
    # cell16k_ring_attn_core = cell16k_ring + attn_core_compile ONLY.
    expected_core = dict(ring)
    expected_core[_ACC] = "1"
    assert presets["cell16k_ring_attn_core"] == expected_core, (
        "cell16k_ring_attn_core must equal cell16k_ring + MTPLX_DSV41_ATTN_CORE_COMPILE=1"
    )
    assert {
        k: v for k, v in presets["cell16k_ring_attn_core"].items() if v != ring.get(k)
    } == {_ACC: "1"}, "cell16k_ring_attn_core must touch only the core-compile key"
    # cell16k_ring_wo_a_core = cell16k_ring + wo_a cache + core compile (the two W97
    # attention-dispatch levers stacked).
    expected_stack = dict(ring)
    expected_stack[_WOAC] = "1"
    expected_stack[_ACC] = "1"
    assert presets["cell16k_ring_wo_a_core"] == expected_stack, (
        "cell16k_ring_wo_a_core must equal cell16k_ring + wo_a_cache + attn_core_compile"
    )
    assert {
        k: v for k, v in presets["cell16k_ring_wo_a_core"].items() if v != ring.get(k)
    } == {_WOAC: "1", _ACC: "1"}, "cell16k_ring_wo_a_core must touch only the two W97 keys"

    # W97 follow-on: cell16k_ring_wo_a_k29 = cell16k_ring + wo_a cache + K29 fused core.
    assert {
        k: v for k, v in presets["cell16k_ring_wo_a_k29"].items() if v != ring.get(k)
    } == {_WOAC: "1", _DAK: "1"}, "cell16k_ring_wo_a_k29 must touch only wo_a + K29 keys"

    # W99 lean-casts arms (attn_lean_casts is BYTE-IDENTICAL; _ALC in the master list).
    assert _ALC in env_levers.ALL_LEVER_ENVS
    alc = presets["attn_lean_casts"]
    assert alc[_SEL] == "1" and alc[_ALC] == "1", "attn_lean_casts missing its keys"
    assert all(v is None for k, v in alc.items() if k not in (_SEL, _ALC)), (
        "attn_lean_casts must set ONLY selected_keys + attn_lean_casts (hermetic)"
    )
    # cell16k_ring_lean = cell16k_ring + wo_a cache + lean casts (byte-identical stack).
    assert {
        k: v for k, v in presets["cell16k_ring_lean"].items() if v != ring.get(k)
    } == {_WOAC: "1", _ALC: "1"}, "cell16k_ring_lean must touch only wo_a + lean-casts keys"
    # cell16k_ring_lean_k29 = the above + K29 (rounding-class via K29).
    assert {
        k: v for k, v in presets["cell16k_ring_lean_k29"].items() if v != ring.get(k)
    } == {_WOAC: "1", _ALC: "1", _DAK: "1"}, (
        "cell16k_ring_lean_k29 must touch only wo_a + lean-casts + K29 keys"
    )

    # W101 K36 fused-projection-chain arms (ROUNDING-CLASS -- fused rmsnorm
    # reassociates the fp32 sum; the o-LoRA einsum reads bf16 wo_a; flagged in the
    # byte-identity summary).  The lever is in the master list.
    assert _AFP in env_levers.ALL_LEVER_ENVS
    # isolation arm: ONLY selected_keys + attn_fused_proj (hermetic; core stays
    # eager -- fused-proj is INDEPENDENT of K29).
    afp = presets["attn_fused_proj"]
    assert afp[_SEL] == "1" and afp[_AFP] == "1", "attn_fused_proj missing its keys"
    assert all(v is None for k, v in afp.items() if k not in (_SEL, _AFP)), (
        "attn_fused_proj must set ONLY selected_keys + attn_fused_proj (hermetic)"
    )
    # cell16k_ring_fused = cell16k_ring + wo_a cache + lean casts + K29 + fused proj
    # (the FULL attention dispatch stack: qkv/out glue fused, core -> 1 via K29,
    # per-token dequant + redundant casts gone).
    assert "cell16k_ring_fused" in presets, "cell16k_ring_fused arm missing"
    assert {
        k: v for k, v in presets["cell16k_ring_fused"].items() if v != ring.get(k)
    } == {_WOAC: "1", _ALC: "1", _DAK: "1", _AFP: "1"}, (
        "cell16k_ring_fused must touch only wo_a + lean-casts + K29 + fused-proj keys"
    )


# --------------------------------------------------------------------------
# W122 gate-prefetch-width pair: cell16k_ring_v2_attn_pf0 / _pf8 must each be an
# EXACT copy of cell16k_ring_v2_attn plus a single pinned MTPLX_DSV41_GATE_PREFETCH,
# so an edit to the base arm propagates and no stray key drifts into the pair.  The
# width is a receipt-attributable A/B (off vs wide vs the k=6 v2 auto-arm), NOT a
# rounding-class lever (prefetch only warms the cache on the true route).
# --------------------------------------------------------------------------
_GP = "MTPLX_DSV41_GATE_PREFETCH"


def test_gate_prefetch_width_arms_w122(env_levers):
    presets = env_levers.ARM_PRESETS
    base = presets["cell16k_ring_v2_attn"]
    for name in ("cell16k_ring_v2_attn_pf0", "cell16k_ring_v2_attn_pf8"):
        assert name in presets, f"{name} arm missing from ARM_PRESETS"
    # pf0 = base + gate_prefetch="0" (explicit off, wins over the v2 auto-arm ->
    # _resolve_gate_prefetch_k == 0, byte-identical routing, no speculative traffic).
    assert {
        k: v for k, v in presets["cell16k_ring_v2_attn_pf0"].items()
        if v != base.get(k)
    } == {_GP: "0"}, "pf0 must be cell16k_ring_v2_attn + ONLY gate_prefetch=0"
    # pf8 = base + gate_prefetch="8" (explicit wider predict width).
    assert {
        k: v for k, v in presets["cell16k_ring_v2_attn_pf8"].items()
        if v != base.get(k)
    } == {_GP: "8"}, "pf8 must be cell16k_ring_v2_attn + ONLY gate_prefetch=8"
    # The base arm leaves the width UNPINNED (None) so the v2 runner auto-arms it;
    # the pair pins it explicitly.  gate_prefetch is not a rounding-class key, so the
    # pair inherits the base arm's rounding-class status UNCHANGED (both carry
    # attn_fused_proj -> rounding-class; adding the width does not alter the class).
    assert base.get(_GP) is None, "base arm must leave gate_prefetch unpinned (v2 auto-arm)"
    assert _GP not in env_levers.ROUNDING_CLASS_ENVS, "gate_prefetch must not be rounding-class"
    for name in ("cell16k_ring_v2_attn_pf0", "cell16k_ring_v2_attn_pf8"):
        assert env_levers._rounding_class_keys(name) == \
            env_levers._rounding_class_keys("cell16k_ring_v2_attn"), (
            f"{name} rounding-class keys must match cell16k_ring_v2_attn (prefetch adds none)"
        )


# --------------------------------------------------------------------------
# W123 routing-barrier pair: cell16k_ring_v2_attn_ovl / _dr / _ovl_dr must each be an
# EXACT copy of cell16k_ring_v2_attn plus only the shared-overlap and/or device-route
# key, so an edit to the base propagates and no stray key drifts in.  Both levers are
# byte-identical barrier attacks (a pure reorder / a barrier-free all-hit route with
# byte-identical cold recovery) and NEITHER is a rounding-class key, so the pair keeps
# the base arm's rounding-class status unchanged.
# --------------------------------------------------------------------------
def test_routing_barrier_arms_w123(env_levers):
    presets = env_levers.ARM_PRESETS
    base = presets["cell16k_ring_v2_attn"]
    for name in ("cell16k_ring_v2_attn_ovl", "cell16k_ring_v2_attn_dr",
                 "cell16k_ring_v2_attn_ovl_dr"):
        assert name in presets, f"{name} arm missing from ARM_PRESETS"
    # _ovl = base + ONLY shared_overlap=1
    assert {
        k: v for k, v in presets["cell16k_ring_v2_attn_ovl"].items() if v != base.get(k)
    } == {_OV: "1"}, "cell16k_ring_v2_attn_ovl must be base + ONLY shared_overlap=1"
    # _dr = base + ONLY device_route=1 (NOT the pinned variant)
    assert {
        k: v for k, v in presets["cell16k_ring_v2_attn_dr"].items() if v != base.get(k)
    } == {_DR: "1"}, "cell16k_ring_v2_attn_dr must be base + ONLY device_route=1"
    # _ovl_dr = base + BOTH
    assert {
        k: v for k, v in presets["cell16k_ring_v2_attn_ovl_dr"].items() if v != base.get(k)
    } == {_OV: "1", _DR: "1"}, "cell16k_ring_v2_attn_ovl_dr must be base + shared_overlap + device_route"
    # The base arm leaves both unpinned (None) so an ambient export is force-unset.
    assert base.get(_OV) is None and base.get(_DR) is None, "base must leave OVL/DR unpinned"
    # NEITHER lever is rounding-class -> pair inherits the base's rounding-class keys.
    assert _OV not in env_levers.ROUNDING_CLASS_ENVS, "shared_overlap must not be rounding-class"
    assert _DR not in env_levers.ROUNDING_CLASS_ENVS, "device_route must not be rounding-class"
    for name in ("cell16k_ring_v2_attn_ovl", "cell16k_ring_v2_attn_dr",
                 "cell16k_ring_v2_attn_ovl_dr"):
        assert env_levers._rounding_class_keys(name) == \
            env_levers._rounding_class_keys("cell16k_ring_v2_attn"), (
            f"{name} rounding-class keys must match cell16k_ring_v2_attn (barrier levers add none)"
        )
    # device_route (W44) is the barrier-free path, NOT the pinned variant (W71).
    assert presets["cell16k_ring_v2_attn_dr"].get("MTPLX_DSV41_DEVICE_ROUTE_PINNED") is None, (
        "cell16k_ring_v2_attn_dr must NOT arm the pinned device-route variant"
    )


# --------------------------------------------------------------------------
# W97 (review item 7): rounding-class arm labelling.  A rounding-class attention
# lever (the n=1 core compile / K29 fused decode kernel / K35 fused small stages)
# reassociates the fp32 attention core, so a greedy near-tie can flip -- a token-id
# sha mismatch on such an arm is EXPECTED, not a broken exact lever.  The label is
# DERIVED from the arm's env keys (ROUNDING_CLASS_ENVS), never hand-listed, and
# written into every receipt as ``rounding_class`` + the reason ``rounding_class_keys``
# so the byte-identity summary can tell a rounding tie from a genuine exact-lever bug.
# --------------------------------------------------------------------------
def test_rounding_class_envs_documented_keys(env_levers):
    envs = set(env_levers.ROUNDING_CLASS_ENVS)
    # The K29 fused decode kernel, the W97 fixed-shape core compile, and the two K35
    # compile levers (fused small stages + the GPU HC-premix kernel) are rounding-class.
    assert _DAK in envs, "K29 decode-attention kernel must be rounding-class"
    assert _ACC in envs, "W97 core compile must be rounding-class"
    assert _SS in envs, "K35 fused small stages must be rounding-class"
    assert _HPK in envs, "K35 HC-premix kernel must be rounding-class"
    # The bf16 DSpark draft head is named by key so a future arm classifies for free.
    assert "MTPLX_DSV41_DRAFT_HEAD_BF16" in envs
    # DELIBERATELY EXCLUDED so a genuine exact-lever divergence still FAILs: K4 HC
    # compile + K3 Sinkhorn (byte-identical execution reorders on this CPU A/B path,
    # and carried by exact composite arms), the lossy-by-design head codec, and the
    # W99 byte-identical cast dedupe.
    for k in (_HC, _SK, _HM, _ALC):
        assert k not in envs, f"{k} must NOT be rounding-class (kept FAIL)"


def test_rounding_class_derived_not_hand_listed(env_levers):
    presets = env_levers.ARM_PRESETS
    # ROUNDING_CLASS_ARMS is exactly the arms whose preset arms a rounding-class key.
    derived = {
        a for a, p in presets.items()
        if any(p.get(k) not in (None, "") for k in env_levers.ROUNDING_CLASS_ENVS)
    }
    assert set(env_levers.ROUNDING_CLASS_ARMS) == derived
    # _is_rounding_class / _rounding_class_keys agree with membership for EVERY arm.
    for arm in presets:
        keys = env_levers._rounding_class_keys(arm)
        assert env_levers._is_rounding_class(arm) == (arm in env_levers.ROUNDING_CLASS_ARMS)
        assert bool(keys) == (arm in env_levers.ROUNDING_CLASS_ARMS)
        # Every reason key an arm reports is one it actually arms (not None/"").
        for k in keys:
            assert presets[arm].get(k) not in (None, ""), (arm, k)


# Exact arms whose tokens must match control by construction -> rounding_class False,
# so a sha mismatch on them stays a LOUD FAIL (an exact lever that flips is a bug).
# Includes arms that carry head=bf16/sinkhorn (lossy/GPU-rounding, but a DIFFERENT
# class than the rounding-class attention reorders item 7 covers).
_ROUNDING_CLASS_FALSE_ARMS = (
    "control", "cell16k_ring", "cell16k_ring_wo_a_cache", "cell16k_ring_lean",
    "attn_lean_casts", "wo_a_cache", "stack_a", "stack_b", "hc_compile",
    "sinkhorn_metal", "head_bf16", "cell16k_ring_switch",
)
# Rounding-class arms (K29 fused kernel / W97 core compile / K35 fused small stages)
# -> rounding_class True with the reason key(s).
_ROUNDING_CLASS_TRUE_ARMS = {
    "attn_core_compile": _ACC,
    "cell16k_ring_attn_core": _ACC,
    "cell16k_ring_wo_a_core": _ACC,
    "cell16k_ring_wo_a_k29": _DAK,
    "cell16k_ring_lean_k29": _DAK,
    "decode_attn_kernel": _DAK,
    "small_stages_fused": _SS,
}


def test_rounding_class_flag_per_arm(env_levers):
    for arm in _ROUNDING_CLASS_FALSE_ARMS:
        assert env_levers._is_rounding_class(arm) is False, arm
        assert env_levers._rounding_class_keys(arm) == [], arm
        assert arm not in env_levers.ROUNDING_CLASS_ARMS, arm
    for arm, key in _ROUNDING_CLASS_TRUE_ARMS.items():
        assert env_levers._is_rounding_class(arm) is True, arm
        assert key in env_levers._rounding_class_keys(arm), (arm, key)
        assert arm in env_levers.ROUNDING_CLASS_ARMS, arm


def test_dry_run_receipt_carries_rounding_class_label(env_levers, tmp_path):
    # The label + reason keys must be written into every receipt (dry-run path), so
    # the byte-identity summary and downstream census can read them off the receipt.
    out = tmp_path / "rc.jsonl"
    arms = ["control", "cell16k_ring_wo_a_cache", "attn_core_compile",
            "cell16k_ring_wo_a_k29", "small_stages_fused"]
    rc = env_levers.main(
        ["--dry-run", "--context-tokens", "1024", "--arms", *arms, "--out", str(out)]
    )
    assert rc == 0
    receipts = {
        json.loads(line)["arm"]: json.loads(line)
        for line in out.read_text().splitlines() if line
    }
    for arm in arms:
        r = receipts[arm]
        assert "rounding_class" in r, f"{arm}: rounding_class label missing"
        assert "rounding_class_keys" in r, f"{arm}: rounding_class_keys missing"
        assert r["rounding_class"] == env_levers._is_rounding_class(arm), arm
        assert r["rounding_class_keys"] == env_levers._rounding_class_keys(arm), arm
    # Exact arms are False; the K29 / core-compile / K35 arms are True with a reason.
    assert receipts["control"]["rounding_class"] is False
    assert receipts["cell16k_ring_wo_a_cache"]["rounding_class"] is False
    assert receipts["attn_core_compile"]["rounding_class"] is True
    assert _ACC in receipts["attn_core_compile"]["rounding_class_keys"]
    assert receipts["cell16k_ring_wo_a_k29"]["rounding_class"] is True
    assert _DAK in receipts["cell16k_ring_wo_a_k29"]["rounding_class_keys"]
    assert receipts["small_stages_fused"]["rounding_class"] is True
    assert _SS in receipts["small_stages_fused"]["rounding_class_keys"]
