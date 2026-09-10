"""Regression test for the DeepSeek-V4.1-Flash torch-reference goldens (W9).

Encodes the float32 CPU reference values (docs/deepseek-v41/receipts/torchref_*.json,
produced by scripts/deepseek_v41/torchref/ref_forward.py on the 31-token probe) as
the golden.  The fast tests are pure-JSON invariants + pinned exact values (no MLX,
no artifact); the heavy end-to-end parity test is opt-in
(``MTPLX_RUN_HEAVY_DSV41=1``) and forces MLX onto the CPU.

Provenance: the goldens are the *ideal* full-precision oracle.  The MLX serve path
runs bf16 activations + q8/q2 weights, so a match is expected only within that
tolerance -- W9 proved (scripts/.../isolate_wkv.py) that the per-submodule cosine
deficits vs this fp32 golden are bf16 activation storage + q2 routed-expert quant,
NOT a code divergence (window-KV code+q8 are faithful to cos 0.99999 at equal
dtype; engram row ids are bit-exact).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

RECEIPTS = Path(__file__).resolve().parents[1] / "docs" / "deepseek-v41" / "receipts"


def _load(name):
    p = RECEIPTS / name
    if not p.is_file():
        pytest.skip(f"golden receipt missing: {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# per-layer residual stream
# ---------------------------------------------------------------------------
def test_layers_present_finite_and_pinned():
    d = _load("torchref_layers012.json")
    assert d["meta"]["source"] == "torchref"
    assert d["meta"]["prompt_ids"][0] == 0  # BOS
    assert len(d["meta"]["prompt_ids"]) == 31
    layers = {l["layer"]: l for l in d["layers"]}
    assert set(layers) == {0, 1, 2}
    for l in layers.values():
        assert l["finite"] is True
    # pinned float32 reference stats (tight tol: exact deterministic oracle)
    assert layers[0]["std"] == pytest.approx(0.23368, abs=2e-4)
    assert layers[0]["max_abs"] == pytest.approx(9.2841, abs=2e-3)
    assert layers[1]["std"] == pytest.approx(1.76343, abs=2e-3)
    assert layers[2]["std"] == pytest.approx(1.93631, abs=2e-3)


# ---------------------------------------------------------------------------
# engram (layer 1): row ids are bit-exact integers -> the strongest anchor
# ---------------------------------------------------------------------------
def test_engram_row_ids_pinned():
    d = _load("torchref_golden_engram_L1.json")
    rid = d["row_ids"]
    assert rid["shape"] == [1, 31, 24]
    vals = rid["values"][0]
    # pinned exact row ids (hash recipe is deterministic, dtype-independent)
    assert vals[1][:4] == [7364980, 26521422, 47188369, 52620735]
    assert vals[30][:4] == [117835, 28430911, 47036826, 52074921]
    # bounded by the layer-1 table row count (num_embeddings[0])
    assert max(max(row) for row in vals) < 384006168


def test_engram_gate_and_value_finite():
    d = _load("torchref_golden_engram_L1.json")
    assert d["gate"]["finite"] and d["value"]["finite"]
    assert d["gate"]["shape"] == [1, 31, 4]           # per (token, hc copy)
    assert 0.0 <= d["gate"]["max_abs"] <= 1.0          # sigmoid output


# ---------------------------------------------------------------------------
# MoE router: top-6 selection + normalized weights sum to route_scale
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layer,token0_ids", [
    (0, [49, 250, 275, 317, 355, 383]),
    (1, [47, 219, 220, 294, 297, 382]),
    (2, [101, 217, 222, 248, 292, 323]),
])
def test_router_topk_pinned(layer, token0_ids):
    d = _load(f"torchref_golden_moe_L{layer}.json")
    r = d["router"]
    assert len(r["topk_ids"]) == 31 and all(len(x) == 6 for x in r["topk_ids"])
    assert sorted(r["topk_ids"][0]) == token0_ids
    for w in r["topk_weights"]:
        assert sum(w) == pytest.approx(1.5, abs=1e-4)  # norm_topk_prob * route_scale(1.5)


# ---------------------------------------------------------------------------
# layer-2 compressor/indexer selection (the Full-layer intermediates)
# ---------------------------------------------------------------------------
def test_attn_l2_topk_idxs_pinned():
    d = _load("torchref_golden_attn_L2.json")
    assert d["mode"].startswith("Full")
    ti = d["topk_idxs"]["values"][0]
    # last query sees all 15 compressed rows (offset by window length 31)
    assert ti[-1] == list(range(31, 46))
    # query 5 (position 5) reaches (5+1)//2 = 3 compressed rows, rest -1
    assert ti[5] == [31, 32, 33] + [-1] * 12
    assert d["compressed_kv"]["shape"] == [1, 15, 512]
    assert d["index_k"]["shape"] == [1, 15, 128]


# ---------------------------------------------------------------------------
# committed compare (if present): engram hashing is bit-exact between ref & mlx
# ---------------------------------------------------------------------------
def test_compare_engram_row_ids_bit_exact():
    p = RECEIPTS / "compare_ref_vs_mlx.json"
    if not p.is_file():
        pytest.skip("compare receipt not generated in this checkout")
    c = json.loads(p.read_text())
    er = c.get("engram_row_ids", {})
    assert er.get("all_match") is True, "engram row-id hashing diverged between ref and MLX"
    assert er.get("exact") == er.get("total")


# ---------------------------------------------------------------------------
# heavy opt-in: run the MLX serve-path forward and re-assert the W9 verdicts
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("MTPLX_RUN_HEAVY_DSV41") != "1",
                    reason="set MTPLX_RUN_HEAVY_DSV41=1 to run the streaming-model parity check")
def test_mlx_matches_reference_within_tolerance():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)  # never touch the GPU in a worker test
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    sub = root / "scripts" / "deepseek_v41" / "torchref"
    env = dict(os.environ)
    subprocess.run([sys.executable, str(sub / "mlx_dump.py")], check=True, cwd=str(root), env=env)
    subprocess.run([sys.executable, str(sub / "compare_ref_vs_mlx.py")], check=True, cwd=str(root), env=env)
    c = json.loads((RECEIPTS / "compare_ref_vs_mlx.json").read_text())
    # engram hashing must be bit-exact; attention/hc/norm must agree at bf16 tolerance
    assert c["engram_row_ids"]["all_match"] is True
    rows = {r["name"]: r for r in c["rows"]}
    # the post-hc-norm attention input is dtype/quant-robust -> must stay high-cos
    assert rows["attn_L0_input"]["global_cos"] > 0.999
    # router top-6 overlap should stay high (structure preserved through quant)
    assert c["router"]["0"]["mean_overlap"] >= 5.0
