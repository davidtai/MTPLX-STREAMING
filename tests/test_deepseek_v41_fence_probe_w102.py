"""W102 CPU unit test for the --fence-per-layer probe in
scripts/deepseek_v41/metal_decode_attn_bisect.py.

Proves, at tiny dims on the CPU: (a) the CLI flags parse; (b) an isolated run with
--fence-per-layer adds isolated_ms_per_layer_unfenced / isolated_ms_per_layer_fenced
/ fence_kind to the receipt (and leaves the census output otherwise present); (c)
the fenced loop produces BIT-IDENTICAL layer outputs to the unfenced (pipelined)
loop -- the fence changes only WHEN the host blocks, never any value, which is what
makes the pipelined-vs-fenced ms/layer numbers a valid H1/H2 discriminator; and (d)
with the flag OFF the receipt is byte-for-byte the pre-W102 census (no fence keys).

CPU-only plumbing/shape check: the absolute ms/layer magnitudes are the GPU window's
job (the CPU backend does not reproduce Metal host-encode / pipelining).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import mlx.core as mx
import pytest

# Pin the CPU stream at import: "no GPU" means CPU here, and MLX defaults to Metal.
mx.set_default_device(mx.cpu)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "scripts", "deepseek_v41", "metal_decode_attn_bisect.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("w102_metal_decode_attn_bisect", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module()

# Tiny, fast probe geometry (8 backbone layers, hidden 512, ctx 256).
_CTX = 256
_STEPS = 4
_WARMUP = 2
_FENCE_KEYS = (
    "isolated_ms_per_layer_unfenced",
    "isolated_ms_per_layer_fenced",
    "fence_kind",
)


def _assert_bit_identical(outs_u, outs_f):
    assert len(outs_u) == len(outs_f) and len(outs_u) > 0, (len(outs_u), len(outs_f))
    for i, (a, b) in enumerate(zip(outs_u, outs_f)):
        assert a.shape == b.shape, (i, a.shape, b.shape)
        assert bool(mx.array_equal(a, b).item()), f"layer {i} output differs fenced vs unfenced"


def test_pinned_to_cpu():
    assert mx.default_device() == mx.cpu


def test_cli_flags_parse():
    p = _MOD.build_parser()

    # defaults
    a = p.parse_args([])
    assert a.fence_per_layer is False
    assert a.fence_kind == "eval_tolist"
    assert a.fence_steps == _MOD._FENCE_DEFAULT_STEPS == 64
    assert a.fence_warmup == _MOD._FENCE_DEFAULT_WARMUP == 8

    # the real-geometry probe argv (context 16384, 64 steps, 8 warmup, utilization, out)
    a = p.parse_args([
        "--gpu", "--fence-per-layer", "--fence-kind", "eval",
        "--context-tokens", "16384", "--fence-steps", "64", "--fence-warmup", "8",
        "--utilization", "--out", "/tmp/w102.json",
    ])
    assert a.fence_per_layer is True
    assert a.fence_kind == "eval"
    assert a.context_tokens == 16384
    assert a.fence_steps == 64 and a.fence_warmup == 8
    assert a.utilization is True and a.out == "/tmp/w102.json"

    # only the two documented sync kinds are accepted
    with pytest.raises(SystemExit):
        p.parse_args(["--fence-kind", "bogus"])


@pytest.mark.parametrize("fence_kind", ["eval", "eval_tolist"])
def test_fence_probe_bit_identical_and_keys(fence_kind):
    args = _MOD.tiny_args()
    probe = _MOD.fence_probe(
        args, T=_CTX, steps=_STEPS, warmup=_WARMUP, fence_kind=fence_kind,
    )
    # both ms/layer numbers + fence_kind present and well formed
    for k in _FENCE_KEYS:
        assert k in probe, k
    assert probe["fence_kind"] == fence_kind
    for k in ("isolated_ms_per_layer_unfenced", "isolated_ms_per_layer_fenced"):
        v = probe[k]
        assert isinstance(v, float) and v >= 0.0, (k, v)
    assert probe["n_layers"] == args.num_hidden_layers == 8
    assert probe["context_tokens"] == _CTX
    assert probe["steps"] == _STEPS and probe["warmup_steps"] == _WARMUP

    # the fenced loop must reproduce the unfenced loop bit-for-bit
    assert probe["outputs_bit_identical"] is True
    _assert_bit_identical(probe["_outputs_unfenced"], probe["_outputs_fenced"])


def test_run_receipt_carries_fence_keys():
    receipt = _MOD.run({
        "tiny": True, "gpu": False, "Ts": [_CTX], "iters": 2, "warmup": 1,
        "use_selected": True,
        "fence_per_layer": True, "fence_kind": "eval_tolist",
        "fence_steps": _STEPS, "fence_warmup": _WARMUP, "context_tokens": _CTX,
    })
    assert receipt["device"] == "cpu"
    # the three required top-level receipt keys
    for k in _FENCE_KEYS:
        assert k in receipt, k
    assert receipt["fence_kind"] == "eval_tolist"
    # the census output is still present (the probe is additive)
    assert set(receipt["results"]) == {"swa_only", "full", "reindex", "reuse"}
    # detail block present, JSON-clean (no raw-array leak from the underscore keys)
    fp = receipt["fence_probe"]
    assert fp["outputs_bit_identical"] is True
    assert not any(key.startswith("_") for key in fp), list(fp)
    assert fp["readback_elems"] == _MOD._FENCE_READBACK_ELEMS  # eval_tolist -> tiny read


def test_flag_off_leaves_census_unchanged():
    receipt = _MOD.run({
        "tiny": True, "gpu": False, "Ts": [_CTX], "iters": 2, "warmup": 1,
        "use_selected": True,
    })
    # OFF -> no fence keys at all; the census receipt is byte-for-byte pre-W102.
    for k in _FENCE_KEYS:
        assert k not in receipt, k
    assert "fence_probe" not in receipt
    assert set(receipt["results"]) == {"swa_only", "full", "reindex", "reuse"}


def test_invalid_fence_kind_rejected():
    args = _MOD.tiny_args()
    with pytest.raises(ValueError):
        _MOD.fence_probe(args, T=_CTX, steps=_STEPS, warmup=_WARMUP, fence_kind="nope")
