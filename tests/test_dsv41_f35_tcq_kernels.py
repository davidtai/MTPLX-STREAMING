"""F35 trellis projection kernels: the CPU-testable parts (tables, geometry gates, the reference decode path).

The Metal kernels themselves are checked on the GPU by scripts/deepseek_v41/trellis/tcq_kernel_check.py
(rel ~5e-7 vs the bit-exact decoder, windows 2026-09-20 194304/194527); this file never touches Metal.
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "deepseek_v41" / "trellis"))
sys.path.insert(0, str(ROOT))

import tcq_kernel_check as harness  # noqa: E402
import tcq_kernels as tk  # noqa: E402
from mtplx import eschamoe  # noqa: E402


def test_warp_tables_cover_the_tile_exactly_once():
    lane_a, lane_b, lane_p, pos_r, pos_c = harness.warp_tables()
    assert lane_a.shape == (32,) and lane_b.shape == (32,) and lane_p.shape == (32,)
    assert sorted(set(np.array(lane_p).tolist())) == [0, 8, 16, 24]      # the funnel-shift cases the tile kernel handles
    r, c = np.array(pos_r), np.array(pos_c)
    assert r.shape == (256,) and c.shape == (256,)
    assert sorted((int(a) * 16 + int(b)) for a, b in zip(r, c)) == list(range(256))
    assert int(np.array(lane_a).max() >> 1) + 1 < tk.NW and int(np.array(lane_b).max() >> 1) + 1 < tk.NW


def test_pos_tables_agree_with_the_reference_permutation():
    """pos_r/pos_c invert perm_lane/perm_m: decoding lane L's m-th window must land where the vendor decoder puts it."""
    z = np.load(ROOT / "mtplx" / "eschamoe_gather.npz")
    perm_lane, perm_m = z["perm_lane_K3"], z["perm_m_K3"]
    _, _, _, pos_r, pos_c = harness.warp_tables()
    r, c = np.array(pos_r), np.array(pos_c)
    for p in range(256):
        s = int(perm_lane[p]) * 8 + int(perm_m[p])
        assert (int(r[s]), int(c[s])) == (p // 16, p % 16)


def test_geometry_gate_and_kernel_construction():
    with pytest.raises(ValueError):
        tk.make_tcq_projection(2304, 2304)
    for variant in ("qmv", "tile"):
        k1 = tk.make_tcq_projection(2304, 5120, variant)
        assert tk.make_tcq_projection(2304, 5120, variant) is k1          # cached per (variant, geometry)
        tk.make_tcq_projection(5120, 2304, variant)
    with pytest.raises(ValueError):
        tk.make_tcq_projection(2304, 5120, "nope")


def test_reference_decode_matches_the_vendor_reference_decoder():
    """The harness reference (mtplx.eschamoe.decode_expert_weights) equals the clean reference decoder bit for bit."""
    ref_path = Path("/Users/davidtai/escha-extract/escha_decode_ref.py")
    if not ref_path.exists():
        pytest.skip("vendor reference decoder not on this box")
    import importlib.util

    spec = importlib.util.spec_from_file_location("escha_decode_ref", ref_path)
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    rng = np.random.default_rng(7)
    code = rng.integers(-32768, 32767, size=(2, 3, tk.NW), dtype=np.int16)    # IN=32, OUT=48
    w_ref = ref.decode_expert(code, tk.K_BITS).astype(np.float32)
    w_mlx = np.array(eschamoe.decode_expert_weights(mx.array(code), tk.K_BITS).astype(mx.float32))
    assert w_mlx.shape == (32, 48)
    np.testing.assert_array_equal(w_mlx, w_ref)


def test_random_bank_and_reference_shapes():
    code = harness.random_bank(2, 32, 48, seed=1)
    assert code.shape == (2, 2, 3, tk.NW) and code.dtype == mx.int16
    xh = mx.array(np.random.default_rng(0).standard_normal((3, 32)).astype(np.float32))
    ids = np.array([0, 1, 1], dtype=np.uint32)
    y = harness.reference(xh, ids, code)
    assert y.shape == (3, 48)
    # the same expert for rows 1 and 2 must give the same map applied to different inputs
    W1 = eschamoe.decode_expert_weights(code[1], tk.K_BITS).astype(mx.float32)
    np.testing.assert_allclose(np.array(y[1]), np.array(xh[1:2] @ W1)[0], rtol=1e-6, atol=1e-6)
