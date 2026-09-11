"""W11 follow-up: the STREAMING path applies the reference SwiGLU clamp.

The reference clamps every routed expert's SwiGLU at +/-10 (inference/model.py
Expert.forward L846-847). W11's resident path already did (deepseek_v41_moe.
ClampedSwiGLU); this proves the streamed switch does too, via
``ExpertStreamingModelSpec.swiglu_limit`` threaded into every streamed execution
helper in ``mtplx/models/expert_mlx.py``.

Proven here (CPU only):
* ``spec.swiglu_limit`` is 10.0 for dsv41 and None for every other spec, and it
  flows spec -> runtime -> streamed switch (``HotExpertSwitchGLU.swiglu_limit``).
* the streamed affine helpers (``_gather_component_bank`` = the dsv41
  component-banks path, and ``_run_mapped_q4`` = the metal-mmap path) apply
  *bit-identically* the same clamp as the resident ``ClampedSwiGLU``, for
  limit None / 3.0 / 10.0, on a real layer-0 expert record.
* the clamp is load-bearing where activations exceed the limit: the streamed
  output at a binding limit differs from the unclamped (None) output.
* None keeps the plain ``swiglu`` byte-for-byte (hy3 / glm / deepseek_v4 spec
  paths are unchanged).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx_lm.models.activations import swiglu  # noqa: E402

from mtplx.expert_streaming_models import MODEL_SPECS, get_model_spec  # noqa: E402
from mtplx.models.deepseek_v41_moe import ClampedSwiGLU  # noqa: E402
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    _clamped_swiglu,
    _gather_component_bank,
    _run_mapped_q4,
)

ARTIFACT = Path(
    os.environ.get(
        "DSV41_ARTIFACT",
        os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2"),
    )
)
MANIFEST = ARTIFACT / "expert-manifest.json"
BANK = ARTIFACT / "experts.bin"
KEY = "deepseek-v41-flash-expert-q2"
Q2 = dict(group_size=64, bits=2)

_bank = pytest.mark.skipif(
    not BANK.is_file(), reason="experts.bin not present"
)


def _mx_from_raw(raw: bytes, dtype: str, shape: tuple) -> mx.array:
    if dtype == "BF16":
        return mx.array(np.frombuffer(raw, "<u2").reshape(shape).copy()).view(mx.bfloat16)
    if dtype in ("U32", "I32"):
        return mx.array(np.frombuffer(raw, "<u4").reshape(shape).copy().astype(np.uint32))
    raise ValueError(dtype)


def _layer0_record0():
    from mtplx.expert_manifest import load_expert_manifest

    manifest = load_expert_manifest(MANIFEST)
    return next(r for r in manifest.records if r.layer == 0 and r.expert == 0)


def _read_components(fd, record):
    """{full component name -> mx.array}, e.g. 'gate_proj.weight'."""
    return {
        seg.component: _mx_from_raw(os.pread(fd, seg.length, seg.offset), seg.dtype, tuple(seg.shape))
        for seg in record.segments
    }


# --------------------------------------------------------------------------
# spec wiring
# --------------------------------------------------------------------------
def test_spec_swiglu_limit_values():
    # Both DeepSeek-V4.1-Flash streaming specs carry the reference +/-10 clamp:
    # the affine-Q2 spec (KEY) and the native-mxfp4 spec derived from it via
    # ``replace`` (W15), which inherits ``swiglu_limit=10.0`` unchanged
    # (config text_config.swiglu_limit=10.0). Every other model (hy3 / glm)
    # leaves it None so the plain-swiglu path stays byte-for-byte identical.
    clamped = {"deepseek-v41-flash-expert-q2", "deepseek-v41-flash-expert-mxfp4"}
    assert KEY in clamped
    for key in clamped:
        assert get_model_spec(key).swiglu_limit == 10.0
    others = {k: v.swiglu_limit for k, v in MODEL_SPECS.items() if k not in clamped}
    assert set(others.values()) == {None}, others


def test_none_limit_keeps_plain_swiglu_byte_identical():
    rng = np.random.default_rng(0)
    gate = mx.array(rng.standard_normal((4, 16)).astype(np.float32) * 20)
    up = mx.array(rng.standard_normal((4, 16)).astype(np.float32) * 20)
    assert bool(mx.all(_clamped_swiglu(gate, up, None) == swiglu(gate, up)).item())
    assert bool(mx.all(_clamped_swiglu(gate, up, 0) == swiglu(gate, up)).item())
    # and the clamp really changes the result at a binding limit
    assert not bool(mx.all(_clamped_swiglu(gate, up, 3.0) == swiglu(gate, up)).item())


def test_hot_switch_reads_spec_swiglu_limit():
    # spec -> runtime.spec -> switch.swiglu_limit, with a minimal fake runtime
    # (affine codec never touches runtime.manifest; shadow lookup absent -> None).
    dsv41 = SimpleNamespace(spec=get_model_spec(KEY))
    switch = HotExpertSwitchGLU(dsv41, layer_index=0)
    assert switch.swiglu_limit == 10.0
    hy3 = SimpleNamespace(spec=get_model_spec("hy3-expert-q2"))
    assert HotExpertSwitchGLU(hy3, layer_index=0).swiglu_limit is None


# --------------------------------------------------------------------------
# streamed helpers apply exactly the resident ClampedSwiGLU clamp
# --------------------------------------------------------------------------
def _resident_component_bank(bank, slot, x, limit):
    """The resident ClampedSwiGLU over the same gather_qmm kernel as
    ``_gather_component_bank`` -- so any difference is the clamp alone."""
    rows = int(x.shape[0])
    sel = x.reshape((rows, 1, 1, int(x.shape[-1])))

    def qmm(v, proj):
        return mx.gather_qmm(
            v,
            bank.arrays[f"{proj}.weight"],
            bank.arrays[f"{proj}.scales"],
            bank.arrays[f"{proj}.biases"],
            rhs_indices=slot,
            transpose=True,
            group_size=64,
            bits=2,
            mode="affine",
        )

    gate = qmm(sel, "gate_proj")
    up = qmm(sel, "up_proj")
    act = ClampedSwiGLU(limit or 0.0)(up, gate)  # SwitchGLU order: (up, gate)
    out = qmm(act, "down_proj")
    return out.reshape((rows, int(x.shape[-1])))


def _resident_mapped(arrays, x, limit):
    def qmm(v, proj):
        return mx.quantized_matmul(
            v,
            arrays[f"{proj}.weight"],
            scales=arrays[f"{proj}.scales"],
            biases=arrays[f"{proj}.biases"],
            group_size=64,
            bits=2,
            mode="affine",
        )

    gate = qmm(x, "gate_proj")
    up = qmm(x, "up_proj")
    act = ClampedSwiGLU(limit or 0.0)(up, gate)
    return qmm(act, "down_proj")


@_bank
def test_component_bank_helper_applies_reference_clamp():
    fd = os.open(BANK, os.O_RDONLY)
    try:
        comps = _read_components(fd, _layer0_record0())
    finally:
        os.close(fd)
    bank = SimpleNamespace(arrays={c: v[None] for c, v in comps.items()})  # capacity 1
    slot = mx.zeros((3, 1), dtype=mx.int32)
    rng = np.random.default_rng(1)
    # scale large so gate/up overshoot +/-10 -> the clamp binds.
    x = mx.array((rng.standard_normal((3, 5120)) * 8.0).astype(np.float32))

    outs = {}
    for limit in (None, 3.0, 10.0):
        streamed = _gather_component_bank(x, bank, slot, group_size=64, bits=2, swiglu_limit=limit)
        resident = _resident_component_bank(bank, slot, x, limit)
        mx.eval(streamed, resident)
        assert bool(mx.all(streamed == resident).item()), f"limit={limit}"
        outs[limit] = np.array(streamed)
    # load-bearing: the clamp changes the streamed output at a binding limit.
    assert not np.array_equal(outs[3.0], outs[None])
    assert not np.array_equal(outs[10.0], outs[None])


@_bank
def test_mapped_helper_applies_reference_clamp():
    fd = os.open(BANK, os.O_RDONLY)
    try:
        arrays = _read_components(fd, _layer0_record0())
    finally:
        os.close(fd)
    mapped = SimpleNamespace(arrays=arrays)
    rng = np.random.default_rng(2)
    x = mx.array((rng.standard_normal((3, 5120)) * 8.0).astype(np.float32))
    for limit in (None, 3.0, 10.0):
        streamed = _run_mapped_q4(x, mapped, group_size=64, bits=2, swiglu_limit=limit)
        resident = _resident_mapped(arrays, x, limit)
        mx.eval(streamed, resident)
        assert bool(mx.all(streamed == resident).item()), f"limit={limit}"
    plain = np.array(_run_mapped_q4(x, mapped, group_size=64, bits=2, swiglu_limit=None))
    clamped = np.array(_run_mapped_q4(x, mapped, group_size=64, bits=2, swiglu_limit=3.0))
    assert not np.array_equal(plain, clamped)
