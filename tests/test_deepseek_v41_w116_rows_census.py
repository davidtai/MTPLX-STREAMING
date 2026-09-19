"""W116 CPU unit test for the verify-attention ROWS census in
scripts/deepseek_v41/verify_attn_rows_census.py.

Proves, at tiny dims on the CPU (T=256, k=32, hd=32; no Metal, no model load, no
model paths): (a) the CLI flags parse; (b) a ``--cpu-smoke`` run returns the census
schema -- per (mode, rows) sub-op entries each carrying BOTH a fenced_ms and a
pipelined_ms, an op_count, plus the k_union read-amplification accounting and the
derived per-rows scaling; (c) the eager gathered core runs on the CPU while the
Metal-only K29 kernel is recorded absent (with a reason) rather than crashing; and
(d) the script references no ``/Users/davidtai/models`` path and loads no model.

CPU-only plumbing/shape check: the absolute ms magnitudes and the rows-scaling
verdict are the GPU window's job (the CPU backend does not reproduce Metal
per-row dispatch / core cost).
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
_SCRIPT = os.path.join(_REPO, "scripts", "deepseek_v41", "verify_attn_rows_census.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("w116_verify_attn_rows_census", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module()

_SMOKE_CFG = {
    "cpu_smoke": True,
    "modes": ["full"],
    "rows": [1, 2],
    "repeats": 2,
    "chain": 2,
    "warmup": 1,
}


@pytest.fixture(scope="module")
def smoke_receipt():
    return _MOD.run(dict(_SMOKE_CFG))


def test_pinned_to_cpu():
    assert mx.default_device() == mx.cpu


def test_cli_flags_parse():
    p = _MOD.build_parser()
    a = p.parse_args([])
    assert a.gpu is False and a.cpu_smoke is False
    assert a.rows is None and a.modes is None

    a = p.parse_args([
        "--gpu", "--rows", "1", "2", "4", "6", "8",
        "--modes", "full,reindex", "--codec", "mxfp8", "--out", "/tmp/w116.json",
    ])
    assert a.gpu is True
    assert a.rows == [1, 2, 4, 6, 8]
    assert a.modes == "full,reindex"
    assert a.codec == "mxfp8"

    a = p.parse_args(["--cpu-smoke"])
    assert a.cpu_smoke is True


def test_smoke_shape_targets(smoke_receipt):
    d = smoke_receipt["dims"]
    assert smoke_receipt["device"] == "cpu"
    assert smoke_receipt["cpu_smoke"] is True
    assert d["head_dim"] == 32
    assert d["T"] == 256
    # k = sliding_window (16) + index_topk (16) = 32 (the task's plumbing target)
    r1 = smoke_receipt["results"]["full"]["1"]
    assert r1["k"] == 32


def test_rows_list_and_keys(smoke_receipt):
    assert smoke_receipt["rows"] == [1, 2]
    res = smoke_receipt["results"]["full"]
    assert set(res.keys()) == {"1", "2"}
    for rk in ("1", "2"):
        assert res[rk]["rows"] == int(rk)


def test_all_subop_keys_present(smoke_receipt):
    subops = smoke_receipt["results"]["full"]["1"]["subops"]
    for name in _MOD._SUBOPS:
        assert name in subops, name
        assert "present" in subops[name]


def test_fenced_and_pipelined_both_present(smoke_receipt):
    # The eager gathered core runs on the CPU: it must carry BOTH timings.
    for rk in ("1", "2"):
        for name in ("qkv_proj", "select", "gather", "core_eager", "out_proj"):
            e = smoke_receipt["results"]["full"][rk]["subops"][name]
            assert e["present"] is True, (name, e)
            assert isinstance(e["fenced_ms"], float)
            assert isinstance(e["pipelined_ms"], float)
            assert e["fenced_ms"] >= 0.0 and e["pipelined_ms"] >= 0.0
            # op_count is an int (graph inspection) or None (export unavailable)
            assert e["op_count"] is None or isinstance(e["op_count"], int)


def test_metal_only_core_recorded_absent_on_cpu(smoke_receipt):
    # K29 is a Metal kernel: on the CPU it must be recorded absent with a reason,
    # never crash the census.
    e = smoke_receipt["results"]["full"]["1"]["subops"]["core_k29"]
    assert e["present"] is False
    assert isinstance(e.get("reason"), str) and e["reason"]


def test_select_absent_on_non_index_source():
    # A reuse layer reads the source's published selection (~0), so it carries no
    # select sub-op -- present=False, reason recorded.
    rec = _MOD.run({**_SMOKE_CFG, "modes": ["reuse"]})
    e = rec["results"]["reuse"]["1"]["subops"]["select"]
    assert e["present"] is False
    assert "reason" in e


def test_k_union_accounting(smoke_receipt):
    ku = smoke_receipt["results"]["full"]["2"]["k_union"]
    for key in ("k_union", "window_union", "compress_union", "rows_times_k",
                "amplification", "per_row_k"):
        assert key in ku
    assert ku["k_union"] >= 1
    assert ku["rows_times_k"] >= ku["k_union"]  # per-row read >= shared-tile read
    assert isinstance(ku["amplification"], float) and ku["amplification"] >= 1.0


def test_derived_scaling_present(smoke_receipt):
    der = smoke_receipt["derived"]["full"]
    assert "scaling" in der and "implied_ms_per_cycle_rows6_x40" in der
    sc = der["scaling"]["core_eager"]["fenced_ratio_vs_rows1"]
    assert sc["1"] == 1.0  # rows=1 ratio vs itself is 1.0


def test_proj_accounting_bf16_for_smoke(smoke_receipt):
    acct = smoke_receipt["proj_accounting"]["full"]
    assert acct["codec"] == "bf16"
    assert acct["quantized"] is False
    assert acct["proj_bytes"] > 0


def test_no_model_paths_or_load():
    with open(_SCRIPT) as f:
        src = f.read()
    assert "/Users/davidtai/models" not in src
    assert ".safetensors" not in src
    assert "make_cache" not in src
    assert "_load_model" not in src
    assert "load_model" not in src
