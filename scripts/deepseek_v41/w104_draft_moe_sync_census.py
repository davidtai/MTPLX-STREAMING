"""W104 audit: count blocking host syncs the DSpark DRAFT block's resident MoE
incurs per draft cycle, on the tiny real-structure model.

Unlike :mod:`w96_sync_census` (which prices the *streamed* backbone switch through
a fake ``ExpertStreamingRuntime``), the 3 DSpark MTP stages run a **resident** MoE:
``DSparkBlock.mlp`` is a plain :class:`mtplx.models.deepseek_v41_moe.MoE` whose
``switch_mlp`` is an mlx-lm ``SwitchGLU`` (its ``SwitchLinear`` leaves repacked to
``QuantizedSwitchLinear(mode="mxfp4", gs32)`` by ``Model._build_mtp_head``).  There
is no expert-streaming runtime, no ``mx.eval(indices)`` routing barrier, no layer
lock and no per-call dequant/repack: the whole ``self.mlp(moe_input)`` chain is
lazy MLX ops terminating in three ``mx.gather_qmm(mode="mxfp4")`` calls.

This census proves that by wrapping ``mx.eval`` / ``mx.async_eval`` /
``mx.array.tolist`` / ``mx.array.item`` with counters that tag each call with its
deepest project-frame ``file:line``, then running ONE ``draft_block`` (3 stages)
WITHOUT a terminal eval and counting how many host syncs originate anywhere in the
MoE files.  It also exercises the real mxfp4 ``gather_qmm`` path on an isolated,
gs32-aligned :class:`MoE` module to confirm that call issues no host sync either.

No model, no bank, no GPU: ``mx.set_default_device(mx.cpu)``.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

import numpy as np  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

_WT = Path(__file__).resolve().parents[2]
if str(_WT) not in sys.path:
    sys.path.insert(0, str(_WT))

from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.models import deepseek_v41_dspark as _dsp  # noqa: E402

# MoE-origin source files: a sync tagged with any of these came from the routed
# MoE path (the module, the mlx-lm switch, or the streamed-switch kernels).
_MOE_FILES = ("deepseek_v41_moe.py", "switch_layers.py", "expert_mlx.py")

_orig_eval = mx.eval
_orig_async = mx.async_eval
_orig_tolist = mx.array.tolist
_orig_item = mx.array.item

# each entry: (kind, "file:line" of the deepest project frame that triggered it)
SYNCS: list[tuple[str, str]] = []
_RECORD = False


def _tag() -> str:
    for fr in reversed(traceback.extract_stack()[:-2]):
        base = os.path.basename(fr.filename)
        if base == "w104_draft_moe_sync_census.py":
            continue
        if base.startswith("<") or "/mlx/" in fr.filename:
            continue
        return f"{base}:{fr.lineno}"
    return "<unknown>"


def _mk(kind, orig, method=False):
    def wrapper(*a, **k):
        if _RECORD:
            SYNCS.append((kind, _tag()))
        return orig(*a, **k)
    return wrapper


def _install():
    mx.eval = _mk("mx.eval", _orig_eval)
    mx.async_eval = _mk("mx.async_eval", _orig_async)
    mx.array.tolist = _mk("array.tolist", _orig_tolist)  # type: ignore[assignment]
    mx.array.item = _mk("array.item", _orig_item)  # type: ignore[assignment]


def _restore():
    mx.eval = _orig_eval
    mx.async_eval = _orig_async
    mx.array.tolist = _orig_tolist  # type: ignore[assignment]
    mx.array.item = _orig_item  # type: ignore[assignment]


def _moe_syncs():
    return [(kind, tag) for (kind, tag) in SYNCS if tag.split(":")[0] in _MOE_FILES]


# ---------------------------------------------------------------------------
# 1) full draft_block on the tiny real-structure DSpark model
# ---------------------------------------------------------------------------
def _dspark_args(block_size=5):
    return ModelArgs(
        vocab_size=64, hidden_size=32, num_hidden_layers=5, num_attention_heads=4,
        head_dim=16, qk_rope_head_dim=8, q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        sliding_window=8, window_size=8, hc_mult=4, hc_sinkhorn_iters=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0,
        n_mtp_layers=3, dspark_block_size=block_size, dspark_noise_token_id=63,
        dspark_target_layer_ids=[2, 3, 4], dspark_markov_rank=12,
        dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )


def _build_dspark(seed=1, block_size=5):
    mx.random.seed(seed)
    args = _dspark_args(block_size=block_size)
    model = Model(args, quantize=False, mtp=True)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if value.ndim == 1:
            centre = 1.0 if leaf.endswith("norm_weight") or leaf == "scale" else 0.0
            new = mx.random.normal(value.shape) * 0.1 + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    mx.eval(model.parameters())
    return model, args


def _run_full_draft(flag, seed=1):
    global SYNCS, _RECORD
    model, args = _build_dspark(seed=seed)
    ids = mx.array(np.random.RandomState(0).randint(0, args.vocab_size, size=(1, 17)))
    logits, main_hidden = model(ids, return_hidden=True)
    mx.eval(logits, main_hidden)
    caches = model.make_mtp_cache()
    model.mtp.seed_main(main_hidden, caches)
    mx.eval([c.window for c in caches if c.window is not None])
    primary = mx.array([int(mx.argmax(logits[0, -1]))])
    main_h = main_hidden[:, -1:, :]
    embed, head = model.model.embed_tokens, model.head

    prev = _dsp._DRAFT_COMPILE
    _dsp._DRAFT_COMPILE = flag
    _dsp._DRAFT_COMPILED.clear()
    SYNCS = []
    _install()
    _RECORD = True
    try:
        out_ids, dlogits, conf = model.mtp.draft_block(main_h, primary, caches, embed, head)
        # the caller's single terminal eval (dspark_decode.py) -- tagged, NOT MoE.
        _RECORD = False
        _restore()
        _orig_eval(out_ids, dlogits, conf)
    finally:
        _RECORD = False
        _restore()
        _dsp._DRAFT_COMPILE = prev
        _dsp._DRAFT_COMPILED.clear()
    return list(SYNCS)


# ---------------------------------------------------------------------------
# 2) isolated resident MoE, gs32-aligned, repacked to real mxfp4 gather_qmm
# ---------------------------------------------------------------------------
def _moe_only_args():
    return ModelArgs(
        vocab_size=64, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        head_dim=16, moe_intermediate_size=64, n_routed_experts=8,
        num_experts_per_tok=3, hc_mult=4, scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5, swiglu_limit=10.0, n_shared_experts=1,
        norm_topk_prob=False,
    )


def _run_isolated_mxfp4_moe(block_size=5, seed=3):
    import mlx.nn as nn
    from mtplx.models.deepseek_v41_moe import MoE
    from mtplx.models.deepseek_v41 import _make_mtp_expert_quant_predicate

    global SYNCS, _RECORD
    mx.random.seed(seed)
    args = _moe_only_args()
    moe = MoE(40, args)  # a DSpark-style resident stage MoE (layer_id 40)
    filled = []
    for name, value in tree_flatten(moe.parameters()):
        new = (mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
               if value.ndim >= 2 else mx.random.normal(value.shape) * 0.1)
        filled.append((name, new.astype(value.dtype)))
    moe.update(tree_unflatten(filled))
    # repack the routed experts to real mxfp4 gs32 (== Model._build_mtp_head).
    nn.quantize(moe, group_size=32, bits=4, mode="mxfp4",
                class_predicate=_make_mtp_expert_quant_predicate(32))
    mx.eval(moe.parameters())
    kind = type(moe.switch_mlp.gate_proj).__name__

    x = mx.random.normal((1, block_size, args.hidden_size)).astype(mx.bfloat16)
    mx.eval(x)
    SYNCS = []
    _install()
    _RECORD = True
    try:
        y = moe(x)
        _RECORD = False
        _restore()
        _orig_eval(y)
    finally:
        _RECORD = False
        _restore()
    return list(SYNCS), kind, y.shape


def _render(label, syncs):
    moe = [s for s in syncs if s[1].split(":")[0] in _MOE_FILES]
    print(f"\n### {label}")
    print(f"    host syncs DURING the call (before the terminal eval): {len(syncs)}")
    print(f"    of which originate in the MoE files {_MOE_FILES}: {len(moe)}")
    from collections import Counter
    for tag, n in Counter(t for _, t in syncs).most_common():
        flag = "  <-- MoE" if tag.split(":")[0] in _MOE_FILES else ""
        print(f"        {n:3d}  {tag}{flag}")


if __name__ == "__main__":
    for flag in (False, True):
        syncs = _run_full_draft(flag)
        _render(f"draft_block (3 stages), DRAFT_COMPILE={'ON' if flag else 'OFF'}", syncs)
        assert not _moe_syncs(), f"MoE issued a host sync: {_moe_syncs()}"

    syncs, kind, shape = _run_isolated_mxfp4_moe()
    _render(f"isolated resident MoE, switch leaf={kind}, out {shape}", syncs)
    assert not [s for s in syncs if s[1].split(':')[0] in _MOE_FILES], "mxfp4 MoE synced"
    print("\nOK: the DSpark draft MoE (resident mxfp4 gather_qmm) issues ZERO host "
          "syncs per draft cycle -- no routing barrier, no per-call dequant/repack.")
