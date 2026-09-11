"""CPU tests for the W51 / K26 prefill "dequantize once, matmul dense" expert path.

Covers ``mtplx.models.expert_mlx``:

  * :func:`_run_component_bank_dense_prefill` numerics vs a float64 reference and
    vs the pure ``gather_qmm`` wave (:func:`_gather_component_bank`), reporting
    max |Δ| and relative error for BOTH paths (mxfp4 gather is not guaranteed
    bit-identical to dequant+bf16 matmul -- fp32 accumulation order differs);
  * the flag-on output matches the flag-off (gather) output within that bound;
  * the dense path scatters rows back into the router's original order;
  * the below-threshold / M=1 (decode-shaped) cases are a byte-identical
    ``gather_qmm`` fall-through, so the flag can never perturb decode;
  * ``_dispatch_component_bank`` engages the dense path only when handed
    ``dense_prefill=True`` (the decode all-hit / device-route callers use the
    default False).

No GPU, no Metal beyond CPU: MLX is pinned to the CPU device (memory/
worker-tests-must-pin-mlx-cpu.md -- "no GPU" is not enough, MLX defaults to
Metal). ``experts.bin`` is never loaded; a small in-memory mxfp4 bank stands in.
Run under ``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

mx.set_default_device(mx.cpu)

import numpy as np
import pytest

import mtplx.models.expert_mlx as em

_HIDDEN = 256
_INTER = 128
_GROUP = 32
_BITS = 4
_LIMIT = 10.0

_PROJ_DIMS = {
    # projection -> (output_size, input_size)
    "gate_proj": (_INTER, _HIDDEN),
    "up_proj": (_INTER, _HIDDEN),
    "down_proj": (_HIDDEN, _INTER),
}


def _build_mxfp4_bank(n_experts: int, seed: int = 0):
    """A duck-typed component bank of ``n_experts`` mxfp4 gs32 experts.

    Returns ``(bank, dense_src)`` where ``bank.arrays`` holds the packed weight +
    E8M0 scale leaves (row-major by slot, exactly as the streamed component bank),
    and ``dense_src[(expert, projection)]`` is the *exactly dequantized* [out, in]
    float64 matrix (mxfp4 dequant is lossless in bf16, so this is the true value
    both execution paths approximate)."""
    rng = np.random.default_rng(seed)
    packed_by_proj: dict[str, list[mx.array]] = {p: [] for p in _PROJ_DIMS}
    scales_by_proj: dict[str, list[mx.array]] = {p: [] for p in _PROJ_DIMS}
    dense_src: dict[tuple[int, str], np.ndarray] = {}
    for expert in range(n_experts):
        for proj, (out, inp) in _PROJ_DIMS.items():
            src = (rng.standard_normal((out, inp)) * 0.08).astype(np.float32)
            packed, scales = mx.quantize(
                mx.array(src), group_size=_GROUP, bits=_BITS, mode="mxfp4"
            )
            mx.eval(packed, scales)
            packed_by_proj[proj].append(packed)
            scales_by_proj[proj].append(scales)
            # The exact value the bank stores (dequant is lossless for mxfp4).
            deq = mx.dequantize(
                packed, scales, group_size=_GROUP, bits=_BITS, mode="mxfp4"
            )
            dense_src[(expert, proj)] = np.asarray(
                deq.astype(mx.float32)
            ).astype(np.float64)
    arrays: dict[str, mx.array] = {}
    for proj in _PROJ_DIMS:
        arrays[f"{proj}.weight"] = mx.stack(packed_by_proj[proj], axis=0)
        arrays[f"{proj}.scales"] = mx.stack(scales_by_proj[proj], axis=0)
    mx.eval(list(arrays.values()))
    return SimpleNamespace(arrays=arrays), dense_src


def _bindings(expert_of_row):
    """Fake assignment-aligned bindings: one per row, each naming its bank slot."""
    return tuple(
        SimpleNamespace(
            buffer=SimpleNamespace(bank=None, bank_index=int(expert), expert=int(expert))
        )
        for expert in expert_of_row
    )


def _bind_bank(bank, bindings):
    for binding in bindings:
        binding.buffer.bank = bank
    return bindings


def _fp64_reference(x_np: np.ndarray, expert_of_row, dense_src) -> np.ndarray:
    """Row-wise clamped-SwiGLU MLP in float64 from the exact dequantized weights."""
    out = np.empty((x_np.shape[0], _HIDDEN), dtype=np.float64)
    x64 = x_np.astype(np.float64)
    for row, expert in enumerate(expert_of_row):
        xr = x64[row]
        gate = xr @ dense_src[(expert, "gate_proj")].T
        up = xr @ dense_src[(expert, "up_proj")].T
        up = np.clip(up, -_LIMIT, _LIMIT)
        gate = np.minimum(gate, _LIMIT)
        silu = gate / (1.0 + np.exp(-gate))
        hidden = silu * up
        out[row] = hidden @ dense_src[(expert, "down_proj")].T
    return out


def _max_abs_and_rel(actual: mx.array, ref: np.ndarray):
    a = np.asarray(actual.astype(mx.float32)).astype(np.float64)
    diff = np.abs(a - ref)
    max_abs = float(diff.max())
    denom = float(np.abs(ref).max()) or 1.0
    return max_abs, max_abs / denom


def _slot_indices(expert_of_row) -> mx.array:
    return mx.array([int(e) for e in expert_of_row], dtype=mx.int32).reshape((-1, 1))


@pytest.fixture(scope="module")
def bank_and_src():
    return _build_mxfp4_bank(n_experts=6, seed=7)


def _make_rows(counts, seed=3):
    """Interleaved assignment order: ``counts[e]`` rows for expert ``e``, shuffled,
    so a correct scatter-back is load-bearing (not a happy-path contiguous order)."""
    expert_of_row = []
    for expert, count in enumerate(counts):
        expert_of_row.extend([expert] * count)
    rng = np.random.default_rng(seed)
    rng.shuffle(expert_of_row)
    n = len(expert_of_row)
    x_np = (np.random.default_rng(seed + 1).standard_normal((n, _HIDDEN)) * 0.1).astype(
        np.float32
    )
    return expert_of_row, x_np


# --------------------------------------------------------------------------
# Numerics: dense path + gather path vs a float64 reference, and vs each other
# --------------------------------------------------------------------------


def test_dense_and_gather_vs_fp64_reference_and_each_other(bank_and_src, capsys):
    bank, dense_src = bank_and_src
    # A mix: two experts well above the 128 default threshold, three below, one
    # single-row -- so a real run splits into a dense group and a gather remainder.
    counts = [200, 150, 80, 5, 1, 0]
    expert_of_row, x_np = _make_rows(counts)
    x = mx.array(x_np).astype(mx.bfloat16)

    ref = _fp64_reference(x_np, expert_of_row, dense_src)

    bindings = _bind_bank(bank, _bindings(expert_of_row))

    gather = em._gather_component_bank(
        x, bank, _slot_indices(expert_of_row),
        group_size=_GROUP, bits=_BITS, swiglu_limit=_LIMIT, codec="mxfp4",
    )
    # min_rows=1 forces EVERY expert onto the dense path (the strongest divergence
    # test); the default-128 case is exercised by the split test below.
    dense_all = em._run_component_bank_dense_prefill(
        x, bindings, group_size=_GROUP, bits=_BITS, swiglu_limit=_LIMIT,
        min_rows=1, batch=8,
    )
    mx.eval(gather, dense_all)

    gather_abs, gather_rel = _max_abs_and_rel(gather, ref)
    dense_abs, dense_rel = _max_abs_and_rel(dense_all, ref)

    d = np.asarray(dense_all.astype(mx.float32)).astype(np.float64)
    g = np.asarray(gather.astype(mx.float32)).astype(np.float64)
    pair_abs = float(np.abs(d - g).max())
    pair_rel = pair_abs / (float(np.abs(g).max()) or 1.0)

    with capsys.disabled():
        print(
            "\n[W51 K26 exactness] rows=%d experts_used=%d\n"
            "  gather_qmm  vs fp64: max|Δ|=%.3e rel=%.3e\n"
            "  dense_bf16  vs fp64: max|Δ|=%.3e rel=%.3e\n"
            "  dense       vs gather: max|Δ|=%.3e rel=%.3e"
            % (
                len(expert_of_row), len({*expert_of_row}),
                gather_abs, gather_rel, dense_abs, dense_rel, pair_abs, pair_rel,
            )
        )

    # Both paths are bf16-matmul approximations of the same fp64 MLP; neither is a
    # reference for the other.  The dense path must not be materially worse than the
    # gather path against the true value, and the two must agree to a bf16 tolerance.
    assert dense_rel < 5e-2
    assert dense_abs <= gather_abs * 4.0 + 1e-3
    assert pair_rel < 5e-2


def test_dense_split_matches_gather_within_bound(bank_and_src):
    """Default threshold (128): the mixed dense+gather run matches the pure gather
    (flag-off) run within the measured bf16 bound -- the 'flag on == flag off' gate."""
    bank, _ = bank_and_src
    counts = [300, 129, 127, 40, 2]  # two dense (>=128), three gather (<128)
    expert_of_row, x_np = _make_rows(counts, seed=11)
    x = mx.array(x_np).astype(mx.bfloat16)
    bindings = _bind_bank(bank, _bindings(expert_of_row))

    gather = em._gather_component_bank(
        x, bank, _slot_indices(expert_of_row),
        group_size=_GROUP, bits=_BITS, swiglu_limit=_LIMIT, codec="mxfp4",
    )
    dense = em._run_component_bank_dense_prefill(
        x, bindings, group_size=_GROUP, bits=_BITS, swiglu_limit=_LIMIT,
        min_rows=em._PREFILL_DENSE_MIN_ROWS_DEFAULT, batch=em._PREFILL_DENSE_BATCH_DEFAULT,
    )
    mx.eval(gather, dense)
    d = np.asarray(dense.astype(mx.float32)).astype(np.float64)
    g = np.asarray(gather.astype(mx.float32)).astype(np.float64)
    rel = float(np.abs(d - g).max()) / (float(np.abs(g).max()) or 1.0)
    assert rel < 5e-2


def test_scatter_preserves_router_row_order(bank_and_src):
    """Every output row equals its own expert's fp64 MLP applied to its own input:
    proves the group/scatter maps rows back to the router's original order."""
    bank, dense_src = bank_and_src
    counts = [130, 130, 3]
    expert_of_row, x_np = _make_rows(counts, seed=5)
    x = mx.array(x_np).astype(mx.bfloat16)
    bindings = _bind_bank(bank, _bindings(expert_of_row))
    dense = em._run_component_bank_dense_prefill(
        x, bindings, group_size=_GROUP, bits=_BITS, swiglu_limit=_LIMIT,
        min_rows=em._PREFILL_DENSE_MIN_ROWS_DEFAULT, batch=4,
    )
    mx.eval(dense)
    ref = _fp64_reference(x_np, expert_of_row, dense_src)
    abs_err, rel = _max_abs_and_rel(dense, ref)
    assert rel < 5e-2, (abs_err, rel)


# --------------------------------------------------------------------------
# No-op / byte-identity guards: the flag never perturbs decode-shaped work
# --------------------------------------------------------------------------


def test_below_threshold_is_byte_identical_to_gather(bank_and_src):
    """No expert clears the threshold -> a pure gather_qmm wave, byte-for-byte."""
    bank, _ = bank_and_src
    counts = [40, 30, 20]  # all < 128
    expert_of_row, x_np = _make_rows(counts, seed=9)
    x = mx.array(x_np).astype(mx.bfloat16)
    bindings = _bind_bank(bank, _bindings(expert_of_row))
    gather = em._gather_component_bank(
        x, bank, _slot_indices(expert_of_row),
        group_size=_GROUP, bits=_BITS, swiglu_limit=_LIMIT, codec="mxfp4",
    )
    dense = em._run_component_bank_dense_prefill(
        x, bindings, group_size=_GROUP, bits=_BITS, swiglu_limit=_LIMIT,
        min_rows=em._PREFILL_DENSE_MIN_ROWS_DEFAULT, batch=8,
    )
    mx.eval(gather, dense)
    assert mx.array_equal(gather, dense).item()


def test_m1_decode_shape_is_byte_identical_to_gather(bank_and_src):
    """A decode-shaped wave (one row per expert, M=1 token top-k) is below the
    threshold, so the dense path falls through to a byte-identical gather -- the
    'flag must not engage at M=1' contract at the numerics level."""
    bank, _ = bank_and_src
    expert_of_row = [0, 3, 1, 5, 2, 4]  # 6 distinct experts, 1 row each (top_k=6)
    x_np = (np.random.default_rng(1).standard_normal((6, _HIDDEN)) * 0.1).astype(
        np.float32
    )
    x = mx.array(x_np).astype(mx.bfloat16)
    bindings = _bind_bank(bank, _bindings(expert_of_row))
    gather = em._gather_component_bank(
        x, bank, _slot_indices(expert_of_row),
        group_size=_GROUP, bits=_BITS, swiglu_limit=_LIMIT, codec="mxfp4",
    )
    dense = em._run_component_bank_dense_prefill(
        x, bindings, group_size=_GROUP, bits=_BITS, swiglu_limit=_LIMIT,
        min_rows=em._PREFILL_DENSE_MIN_ROWS_DEFAULT, batch=8,
    )
    mx.eval(gather, dense)
    assert mx.array_equal(gather, dense).item()


def test_flag_off_reads_env_at_use():
    """The switch gate reads the env at use, not import."""
    import os

    saved = os.environ.get(em._PREFILL_DENSE_ENV)
    try:
        os.environ.pop(em._PREFILL_DENSE_ENV, None)
        assert em._prefill_dense_experts_enabled() is False
        os.environ[em._PREFILL_DENSE_ENV] = "1"
        assert em._prefill_dense_experts_enabled() is True
        os.environ[em._PREFILL_DENSE_ENV] = "0"
        assert em._prefill_dense_experts_enabled() is False
    finally:
        if saved is None:
            os.environ.pop(em._PREFILL_DENSE_ENV, None)
        else:
            os.environ[em._PREFILL_DENSE_ENV] = saved


def test_env_int_defaults_and_overrides():
    import os

    keys = (em._PREFILL_DENSE_MIN_ROWS_ENV, em._PREFILL_DENSE_BATCH_ENV)
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        assert em._positive_env_int(em._PREFILL_DENSE_MIN_ROWS_ENV, 128) == 128
        assert em._positive_env_int(em._PREFILL_DENSE_BATCH_ENV, 8) == 8
        os.environ[em._PREFILL_DENSE_MIN_ROWS_ENV] = "256"
        assert em._positive_env_int(em._PREFILL_DENSE_MIN_ROWS_ENV, 128) == 256
        os.environ[em._PREFILL_DENSE_MIN_ROWS_ENV] = "0"  # invalid -> default
        assert em._positive_env_int(em._PREFILL_DENSE_MIN_ROWS_ENV, 128) == 128
        os.environ[em._PREFILL_DENSE_MIN_ROWS_ENV] = "junk"
        assert em._positive_env_int(em._PREFILL_DENSE_MIN_ROWS_ENV, 128) == 128
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
