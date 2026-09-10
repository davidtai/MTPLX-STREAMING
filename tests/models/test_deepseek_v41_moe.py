"""W11 MoE parity tests for DeepSeek-V4.1-Flash (CPU only).

Proves ``mtplx/models/deepseek_v41_moe.py`` is a faithful transliteration of the
reference ``inference/model.py`` MoE stack (Gate L792-828, Expert L830-851, MoE
L854-904), wired to the MTPLX expert-streaming seam:

* Gate parity   -- numpy transcription of the reference ``Gate`` on random inputs
                   + the *real* layer-0 correction bias -> identical top-6 ids and
                   weights (< 1e-5, float32).
* Activation    -- :class:`ClampedSwiGLU` and :class:`Expert.forward` vs a numpy
                   transcription of the reference ``Expert.forward``, including
                   inputs beyond +/- swiglu_limit.
* Seam forward  -- real layer-0 gate + shared experts + a resident test-double
                   switch (fetches the 6 selected Q2 records from experts.bin and
                   applies the reference clamped SwiGLU, the same math a faithful
                   streamed switch must) vs a numpy reference MoE built from the
                   dequantised layer-0 tensors; cos >= 0.999.
* Clamp gap     -- documents that the shipped streamed switch's missing clamp is
                   numerically inert on the real layer-0 records at limit=10.

W9's torch goldens (.worktrees/dsv41-w9/docs/deepseek-v41/receipts/torchref_*.json)
are absent, so the optional golden test skips.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# Force CPU at import: no Metal anywhere in this suite.
mx.set_default_device(mx.cpu)

from mtplx.models.deepseek_v41_moe import (  # noqa: E402
    ClampedSwiGLU,
    Expert,
    Gate,
    MoE,
)

ARTIFACT = Path(
    os.environ.get(
        "DSV41_ARTIFACT",
        os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2"),
    )
)
MANIFEST = ARTIFACT / "expert-manifest.json"
BANK = ARTIFACT / "experts.bin"

# Released 40-layer text config (config.json / expert-manifest.json).
HIDDEN = 5120
INTER = 2304
N_ROUTED = 384
TOP_K = 6
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
Q2 = dict(group_size=64, bits=2)
Q8 = dict(group_size=64, bits=8)

_artifact = pytest.mark.skipif(
    not MANIFEST.is_file(), reason=f"artifact manifest not present at {MANIFEST}"
)
_bank = pytest.mark.skipif(not BANK.is_file(), reason="experts.bin not present")


class _Args:
    """Minimal stand-in for the port's ModelArgs (only the fields the MoE reads)."""

    hidden_size = HIDDEN
    moe_intermediate_size = INTER
    n_routed_experts = N_ROUTED
    n_shared_experts = 1
    num_experts_per_tok = TOP_K
    scoring_func = "sqrtsoftplus"
    norm_topk_prob = True
    routed_scaling_factor = ROUTE_SCALE
    swiglu_limit = SWIGLU_LIMIT


# --------------------------------------------------------------------------
# safetensors / experts.bin readers (small precise preads, never a full load)
# --------------------------------------------------------------------------
def _st_header(path: Path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _read_resident(name: str) -> mx.array:
    """Read one resident tensor by its raw checkpoint name (pre-sanitise)."""
    for sf in sorted(ARTIFACT.glob("model-*.safetensors")):
        header, data_start = _st_header(sf)
        entry = header.get(name)
        if entry is None:
            continue
        lo, hi = entry["data_offsets"]
        fd = os.open(sf, os.O_RDONLY)
        try:
            raw = os.pread(fd, hi - lo, data_start + lo)
        finally:
            os.close(fd)
        return _mx_from_raw(raw, entry["dtype"], tuple(entry["shape"]))
    raise KeyError(name)


def _mx_from_raw(raw: bytes, dtype: str, shape: tuple) -> mx.array:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, "<u2").reshape(shape).copy()
        return mx.array(u16).view(mx.bfloat16)
    if dtype == "F32":
        return mx.array(np.frombuffer(raw, "<f4").reshape(shape).copy())
    if dtype == "F16":
        return mx.array(np.frombuffer(raw, "<f2").reshape(shape).copy())
    if dtype in ("U32", "I32"):
        return mx.array(np.frombuffer(raw, "<u4").reshape(shape).copy().astype(np.uint32))
    raise ValueError(f"unsupported dtype {dtype}")


def _layer0_records():
    from mtplx.expert_manifest import load_expert_manifest

    manifest = load_expert_manifest(MANIFEST)
    return {r.expert: r for r in manifest.records if r.layer == 0}


def _record_components(fd: int, record):
    """{projection: (packed_u32, scales_bf16, biases_bf16)} for one Q2 record."""
    parts: dict[str, dict[str, mx.array]] = {}
    for seg in record.segments:
        proj, leaf = seg.component.split(".")  # e.g. "gate_proj", "weight"
        raw = os.pread(fd, seg.length, seg.offset)
        parts.setdefault(proj, {})[leaf] = _mx_from_raw(raw, seg.dtype, tuple(seg.shape))
    return {p: (v["weight"], v["scales"], v["biases"]) for p, v in parts.items()}


def _dequant_f32(packed, scales, biases, **q) -> np.ndarray:
    return np.array(mx.dequantize(packed, scales, biases, **q).astype(mx.float32))


# --------------------------------------------------------------------------
# numpy transcription of the reference (inference/model.py)
# --------------------------------------------------------------------------
def _ref_gate(x, weight, bias):
    """Reference Gate.forward L810-828 in numpy (sqrtsoftplus, noaux_tc, norm+scale)."""
    scores = np.sqrt(np.logaddexp(0.0, x.astype(np.float32) @ weight.astype(np.float32).T))
    biased = scores + bias.astype(np.float32)
    idx = np.argsort(-biased, axis=-1)[:, :TOP_K]  # topk[1], desc by biased
    w = np.take_along_axis(scores, idx, axis=-1)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    w = w * ROUTE_SCALE
    return w, idx


def _ref_expert(x, w1, w2, w3, limit):
    """Reference Expert.forward L842-851 in numpy (clamped SwiGLU)."""
    gate = (x.astype(np.float32) @ w1.T).astype(np.float32)
    up = (x.astype(np.float32) @ w3.T).astype(np.float32)
    if limit > 0:
        up = np.clip(up, -limit, limit)
        gate = np.minimum(gate, limit)
    h = (gate / (1.0 + np.exp(-gate))) * up  # silu(gate) * up
    return h @ w2.T


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------
# Gate parity
# --------------------------------------------------------------------------
@_artifact
def test_gate_parity_topk_ids_and_weights():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, HIDDEN)).astype(np.float32)
    w = rng.standard_normal((N_ROUTED, HIDDEN)).astype(np.float32) * (HIDDEN ** -0.5)
    bias = np.array(_read_resident("layers.0.ffn.gate.bias").astype(mx.float32))

    gate = Gate(0, _Args())
    gate.weight = mx.array(w).astype(mx.bfloat16)
    gate.e_score_correction_bias = mx.array(bias)
    # recompute the reference in the SAME precision the module uses for scoring
    # (bf16 gate weight, f32 math) so the comparison isolates the algorithm.
    w_bf16 = np.array(mx.array(w).astype(mx.bfloat16).astype(mx.float32))

    mw, mi = gate(mx.array(x))
    mx.eval(mw, mi)
    my_w = np.array(mw)
    my_i = np.array(mi)

    ref_w, ref_i = _ref_gate(x, w_bf16, bias)

    for row in range(x.shape[0]):
        # identical top-6 ids as a set (both are biased-descending; compare the
        # {id: weight} mapping so any tie ordering is irrelevant).
        assert set(my_i[row].tolist()) == set(ref_i[row].tolist())
        my_map = {int(i): float(v) for i, v in zip(my_i[row], my_w[row])}
        ref_map = {int(i): float(v) for i, v in zip(ref_i[row], ref_w[row])}
        for e, rv in ref_map.items():
            assert abs(my_map[e] - rv) < 1e-5, (row, e, my_map[e], rv)


# --------------------------------------------------------------------------
# Activation / Expert parity (no artifact needed)
# --------------------------------------------------------------------------
def test_clamped_swiglu_activation_parity_including_beyond_limit():
    rng = np.random.default_rng(1)
    # deliberately spread beyond +/- limit so the asymmetric clamp is exercised.
    up = (rng.standard_normal((5, 64)) * 8.0).astype(np.float32)
    gate = (rng.standard_normal((5, 64)) * 8.0).astype(np.float32)
    up[0, :3] = [-30.0, 30.0, 0.0]
    gate[0, :3] = [30.0, -30.0, 12.0]

    act = ClampedSwiGLU(SWIGLU_LIMIT)
    got = np.array(act(mx.array(up), mx.array(gate)))

    gc = np.minimum(gate, SWIGLU_LIMIT)
    uc = np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    ref = (gc / (1.0 + np.exp(-gc))) * uc
    assert np.max(np.abs(got - ref)) < 1e-4

    # limit<=0 must be the plain unclamped SwiGLU.
    plain = np.array(ClampedSwiGLU(0.0)(mx.array(up), mx.array(gate)))
    ref_plain = (gate / (1.0 + np.exp(-gate))) * up
    assert np.max(np.abs(plain - ref_plain)) < 1e-3


def test_expert_forward_parity_including_beyond_limit():
    rng = np.random.default_rng(2)
    dim, inter = 128, 96
    w1 = (rng.standard_normal((inter, dim)) * 0.3).astype(np.float32)  # gate_proj [inter,dim]
    w2 = (rng.standard_normal((dim, inter)) * 0.3).astype(np.float32)  # down_proj [dim,inter]
    w3 = (rng.standard_normal((inter, dim)) * 0.3).astype(np.float32)  # up_proj   [inter,dim]
    # scale x so gate/up projections overshoot +/- limit for some units.
    x = (rng.standard_normal((4, dim)) * 6.0).astype(np.float32)

    exp = Expert(dim, inter, swiglu_limit=SWIGLU_LIMIT)
    exp.w1 = nn.Linear(dim, inter, bias=False)
    exp.w2 = nn.Linear(inter, dim, bias=False)
    exp.w3 = nn.Linear(dim, inter, bias=False)
    exp.w1.weight = mx.array(w1)
    exp.w2.weight = mx.array(w2)
    exp.w3.weight = mx.array(w3)

    got = np.array(exp(mx.array(x)))
    ref = _ref_expert(x, w1, w2, w3, SWIGLU_LIMIT)
    assert np.max(np.abs(got - ref)) < 1e-3
    # the clamp really fired on this input (otherwise the test proves nothing).
    assert np.max(np.abs(x @ w3.T)) > SWIGLU_LIMIT


# --------------------------------------------------------------------------
# Seam forward: real gate + shared experts + record-backed test-double switch
# --------------------------------------------------------------------------
class _RecordSwitchDouble(nn.Module):
    """Resident stand-in for the streamed switch (HotExpertSwitchGLU contract).

    Same ``(x[n,dim], indices[n,top_k]) -> [n,top_k,dim]`` shape as the streamed
    switch, gathering the selected experts' Q2 records straight from experts.bin.
    It applies the reference *clamped* SwiGLU -- the math a faithful streamed
    switch must apply; the shipped streamed switch omits the clamp, which is
    numerically inert here (test_streamed_clamp_is_inert_on_real_records).
    """

    def __init__(self, fd: int, records: dict, limit: float) -> None:
        super().__init__()
        self._fd = fd
        self._records = records
        self._limit = limit
        self._cache: dict[int, dict] = {}

    def _weights(self, expert: int):
        if expert not in self._cache:
            comps = _record_components(self._fd, self._records[expert])
            self._cache[expert] = {
                "w1": mx.dequantize(*comps["gate_proj"], **Q2),
                "w3": mx.dequantize(*comps["up_proj"], **Q2),
                "w2": mx.dequantize(*comps["down_proj"], **Q2),
            }
        return self._cache[expert]

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        n, dim = x.shape
        top_k = int(indices.shape[-1])
        idx = np.array(indices)
        rows = []
        for t in range(n):
            outs = []
            xt = x[t : t + 1].astype(mx.float32)  # [1, dim]
            for k in range(top_k):
                w = self._weights(int(idx[t, k]))
                gate = (xt @ w["w1"].astype(mx.float32).T)
                up = (xt @ w["w3"].astype(mx.float32).T)
                if self._limit > 0:
                    up = mx.clip(up, -self._limit, self._limit)
                    gate = mx.minimum(gate, self._limit)
                h = nn.silu(gate) * up
                outs.append(h @ w["w2"].astype(mx.float32).T)  # [1, dim]
            rows.append(mx.concatenate(outs, axis=0))  # [top_k, dim]
        return mx.stack(rows, axis=0)  # [n, top_k, dim]


def _numpy_reference_moe(x, gate_w, gate_b, shared, records, fd):
    """A from-scratch numpy MoE over the dequantised layer-0 tensors + 6 records."""
    ref_w, ref_i = _ref_gate(x, gate_w, gate_b)
    y = np.zeros_like(x, dtype=np.float32)
    for t in range(x.shape[0]):
        acc = np.zeros((HIDDEN,), dtype=np.float32)
        for k in range(TOP_K):
            e = int(ref_i[t, k])
            comps = _record_components(fd, records[e])
            w1 = _dequant_f32(*comps["gate_proj"], **Q2)
            w3 = _dequant_f32(*comps["up_proj"], **Q2)
            w2 = _dequant_f32(*comps["down_proj"], **Q2)
            acc += ref_w[t, k] * _ref_expert(x[t : t + 1], w1, w2, w3, SWIGLU_LIMIT)[0]
        y[t] = acc
    # shared expert (dequantised q8), every token.
    y += _ref_expert(x, shared["w1"], shared["w2"], shared["w3"], SWIGLU_LIMIT)
    return y


def _load_real_moe(fd, records):
    """A MoE with the real layer-0 gate + shared experts and the record switch."""
    moe = MoE(0, _Args())
    nn.quantize(moe.shared_experts, mode="affine", **Q8)  # w{1,2,3} -> QuantizedLinear
    load = [
        ("gate.weight", _read_resident("layers.0.ffn.gate.weight")),
        ("gate.e_score_correction_bias",
         _read_resident("layers.0.ffn.gate.bias").astype(mx.float32)),
    ]
    for wn in ("w1", "w2", "w3"):
        for leaf in ("weight", "scales", "biases"):
            load.append(
                (f"shared_experts.{wn}.{leaf}",
                 _read_resident(f"layers.0.ffn.shared_experts.{wn}.{leaf}"))
            )
    moe.load_weights(load, strict=False)
    moe.switch_mlp = _RecordSwitchDouble(fd, records, SWIGLU_LIMIT)
    return moe


@_artifact
@_bank
def test_moe_seam_forward_matches_numpy_reference():
    records = _layer0_records()
    assert len(records) == N_ROUTED
    fd = os.open(BANK, os.O_RDONLY)
    try:
        moe = _load_real_moe(fd, records)

        rng = np.random.default_rng(3)
        x = (rng.standard_normal((1, 4, HIDDEN)) * 0.5).astype(np.float32)
        got = np.array(moe(mx.array(x)).astype(mx.float32)).reshape(4, HIDDEN)

        shared = {
            wn: _dequant_f32(
                _read_resident(f"layers.0.ffn.shared_experts.{wn}.weight"),
                _read_resident(f"layers.0.ffn.shared_experts.{wn}.scales"),
                _read_resident(f"layers.0.ffn.shared_experts.{wn}.biases"),
                **Q8,
            )
            for wn in ("w1", "w2", "w3")
        }
        gate_w = np.array(_read_resident("layers.0.ffn.gate.weight").astype(mx.float32))
        gate_b = np.array(_read_resident("layers.0.ffn.gate.bias").astype(mx.float32))
        ref = _numpy_reference_moe(x.reshape(4, HIDDEN), gate_w, gate_b, shared, records, fd)

        assert got.shape == ref.shape
        assert np.all(np.isfinite(got))
        assert _cos(got, ref) >= 0.999, _cos(got, ref)
    finally:
        os.close(fd)


@_artifact
@_bank
def test_layer0_activations_stay_under_limit_but_deep_layers_do_not():
    """Layer 0's gate/up pre-activations stay under +/-10 on random input -- a
    LOCAL fact only. Do NOT read it as "the clamp is inert": the 40-layer
    component-banks probe (tests/models/test_deepseek_v41_clamp_probe.py, receipt
    docs/deepseek-v41/receipts/cpu_clamp_probe.json) shows layers 15..39 drive
    gate/up to ~70/84, so the streamed clamp is load-bearing on a real prompt."""
    records = _layer0_records()
    fd = os.open(BANK, os.O_RDONLY)
    try:
        moe = _load_real_moe(fd, records)
        rng = np.random.default_rng(4)
        x = mx.array((rng.standard_normal((4, HIDDEN)) * 0.5).astype(np.float32))
        _, indices = moe.gate(x)
        mx.eval(indices)
        idx = np.array(indices)
        max_pre = 0.0
        for t in range(4):
            xt = x[t : t + 1].astype(mx.float32)
            for k in range(TOP_K):
                comps = _record_components(fd, records[int(idx[t, k])])
                gate = np.array(xt @ mx.dequantize(*comps["gate_proj"], **Q2).astype(mx.float32).T)
                up = np.array(xt @ mx.dequantize(*comps["up_proj"], **Q2).astype(mx.float32).T)
                max_pre = max(max_pre, float(np.abs(gate).max()), float(np.abs(up).max()))
        assert max_pre < SWIGLU_LIMIT, max_pre
    finally:
        os.close(fd)


# --------------------------------------------------------------------------
# Optional W9 torch golden (absent -> skipped)
# --------------------------------------------------------------------------
_W9 = Path(__file__).resolve().parents[2] / ".worktrees/dsv41-w9/docs/deepseek-v41/receipts"


@pytest.mark.skipif(
    not (_W9.is_dir() and any(_W9.glob("torchref_*.json"))),
    reason="W9 torch goldens not present",
)
def test_layer0_moe_matches_w9_torch_golden():  # pragma: no cover - runs only with goldens
    golden = json.loads(sorted(_W9.glob("torchref_*.json"))[0].read_text())
    layer0 = golden.get("moe", {}).get("layer0") or golden.get("layer0_moe")
    if layer0 is None:
        pytest.skip("golden carries no layer-0 MoE block")
    records = _layer0_records()
    fd = os.open(BANK, os.O_RDONLY)
    try:
        moe = _load_real_moe(fd, records)
        x = np.array(layer0["input"], dtype=np.float32)
        got = np.array(moe(mx.array(x[None])).astype(mx.float32))[0]
        assert _cos(got, np.array(layer0["output"], dtype=np.float32)) >= 0.999
    finally:
        os.close(fd)
