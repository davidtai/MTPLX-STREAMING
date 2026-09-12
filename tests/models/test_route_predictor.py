"""W89 -- tests for the DSV4.1 route-trace collector + predictor trainer.

CPU-only (pins MLX to CPU); the fake model from
``tests/models/test_deepseek_v41_stage_timing.py`` (no artifact, no GPU, <1.5 GB).
Run one file per pytest process, no ``-n auto``, under ``nice -n 19``.

Covers both scripts:
  * ``collect_route_traces``: bf16<->uint16 round-trip, tiny end-to-end capture
    (shapes / manifest / phase split / top-k validity / row alignment), and the
    NON-INVASIVE guarantee (decode logits byte-identical with the hooks enabled).
  * ``train_route_predictor``: end-to-end train/eval on the tiny trace, the
    train/test phase split, the four predictors' layer coverage, metric ranges,
    the miss-reduction monotonicity in prefetch width, recovery of a genuinely
    linear router (the sanity-check mechanism), and the logistic/mlp GD paths.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import collect_route_traces as C  # noqa: E402
import train_route_predictor as T  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_linear_trace(dir_path, *, H, E, K, Ntr, Nte, seed=0):
    """Synthesize a trace whose route IS top-K of a linear map X @ Wtrue, with a
    phase split (first Ntr train, rest test).  router_in == layer_in == X (bf16)."""
    rng = np.random.RandomState(seed)
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    Wt = rng.randn(H, E).astype(np.float32)
    N = Ntr + Nte
    phase = np.zeros(N, np.uint8)
    phase[Ntr:] = C.PHASE_DECODE
    np.save(d / "phase.npy", phase)
    np.save(d / "tokens.npy", rng.randint(0, 50, size=N).astype(np.int32))
    np.save(d / "positions.npy", np.arange(N).astype(np.int32))
    lids = [0, 1, 2]
    for lid in lids:
        X = rng.randn(N, H).astype(np.float32)
        sc = X @ Wt
        t6 = np.argsort(-sc, 1)[:, :K].astype(np.int32)
        bits = C.bf16_bits(mx.array(X), mx)
        np.save(d / f"layer{lid:03d}_router_in.npy", bits)
        np.save(d / f"layer{lid:03d}_layer_in.npy", bits)
        np.save(d / f"layer{lid:03d}_top6.npy", t6)
        np.save(d / f"layer{lid:03d}_gate.npy",
                np.take_along_axis(sc, t6, 1).astype(np.float32))
    json.dump(
        {"schema": "w89-route-trace-v1", "layer_ids": lids, "n_layers": len(lids),
         "n_experts": E, "top_k": K, "stored_hidden": H, "hidden_size": H,
         "n_rows": N},
        open(d / "manifest.json", "w"),
    )
    return d


# ---------------------------------------------------------------------------
# bf16 round-trip
# ---------------------------------------------------------------------------
def test_bf16_roundtrip_is_exact_for_bf16_values():
    rng = np.random.RandomState(3)
    x = (rng.randn(64, 40).astype(np.float32) * 7.0)
    xm = mx.array(x)
    bits = C.bf16_bits(xm, mx)
    assert bits.dtype == np.uint16
    back = C.bf16_to_f32(bits)
    # the stored value is exactly the bf16 rounding of x
    ref = np.array(xm.astype(mx.bfloat16).astype(mx.float32))
    assert np.array_equal(back, ref), "bf16 uint16 round-trip is not exact"
    # and re-encoding an already-bf16 value is a fixed point
    assert np.array_equal(C.bf16_bits(mx.array(back), mx), bits)


# ---------------------------------------------------------------------------
# collector: tiny end-to-end
# ---------------------------------------------------------------------------
def test_collect_tiny_shapes_manifest_and_alignment(tmp_path):
    out = tmp_path / "trace"
    man = C.collect_tiny_trace(out, context_tokens=64, prefill_tail=16,
                               decode_tokens=8, hidden_stride=1)
    assert man["n_layers"] == 8
    assert man["n_experts"] == 8 and man["top_k"] == 2 and man["hidden_size"] == 32
    assert man["n_rows"] == 24
    assert man["n_prefill_tail"] == 16 and man["n_decode"] == 8
    assert man["overflow"] is False
    phase = np.load(out / "phase.npy")
    tokens = np.load(out / "tokens.npy")
    assert phase.tolist() == [C.PHASE_PREFILL_TAIL] * 16 + [C.PHASE_DECODE] * 8
    assert tokens.shape == (24,)
    # every layer: right shapes, valid top-k ids, per-row uniqueness, finite hiddens
    for lid in man["layer_ids"]:
        ri = np.load(out / f"layer{lid:03d}_router_in.npy")
        li = np.load(out / f"layer{lid:03d}_layer_in.npy")
        t6 = np.load(out / f"layer{lid:03d}_top6.npy")
        g = np.load(out / f"layer{lid:03d}_gate.npy")
        assert ri.shape == (24, 32) and ri.dtype == np.uint16
        assert li.shape == (24, 32) and li.dtype == np.uint16
        assert t6.shape == (24, 2) and t6.dtype == np.int32
        assert g.shape == (24, 2) and g.dtype == np.float32
        assert t6.min() >= 0 and t6.max() < 8
        assert all(len(set(r.tolist())) == 2 for r in t6), "duplicate expert in a row"
        assert np.isfinite(C.bf16_to_f32(ri)).all()


def test_collect_hidden_stride_subsamples(tmp_path):
    out = tmp_path / "trace"
    man = C.collect_tiny_trace(out, context_tokens=64, prefill_tail=12,
                               decode_tokens=6, hidden_stride=4)
    # hidden 32 with stride 4 -> 8 stored columns
    assert man["stored_hidden"] == 8
    ri = np.load(out / "layer000_router_in.npy")
    assert ri.shape == (18, 8)


# ---------------------------------------------------------------------------
# collector: NON-INVASIVE (byte-identical decode) + hook lifecycle
# ---------------------------------------------------------------------------
def _tiny_decode_logits(mx_, *, hooked, steps=5, seed=1):
    import mtplx.models.deepseek_v41 as dv41
    import mtplx.models.deepseek_v41_moe as moe

    model, args = C._tiny_model(mx_, seed=seed)
    cache = model.make_cache()
    prompt = list(np.random.RandomState(0).randint(0, args.vocab_size, size=40))
    logits = model(mx_.array([[int(t) for t in prompt]]), cache=cache)
    mx_.eval(logits)
    token = int(mx_.argmax(logits[0, -1]).item())
    rec = None
    saved = None
    if hooked:
        rec = C.RouteRecorder(hidden=args.hidden_size, top_k=args.num_experts_per_tok,
                              n_experts=args.n_routed_experts, hc_mult=args.hc_mult,
                              total_rows=steps, hidden_stride=1, mx=mx_)
        saved = C.install_hooks(rec, moe, dv41)
    out = []
    try:
        for i in range(steps):
            if hooked:
                rec.begin_forward([token], base_pos=len(prompt) + i,
                                  phase=C.PHASE_DECODE)
                rec.enabled = True
            lo = model(mx_.array([[token]]), cache=cache)
            mx_.eval(lo)
            if hooked:
                rec.enabled = False
            out.append(np.array(lo))
            token = int(mx_.argmax(lo[0, -1]).item())
    finally:
        if saved is not None:
            C.uninstall_hooks(saved)
    return out


def test_capture_is_byte_identical_to_unhooked_decode():
    off = _tiny_decode_logits(mx, hooked=False)
    on = _tiny_decode_logits(mx, hooked=True)
    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), (
            f"decode step {i} differs with hooks on -- capture is not "
            f"non-invasive (max abs {np.max(np.abs(a - b))})"
        )


def test_hooks_uninstall_restores_originals():
    import mtplx.models.deepseek_v41 as dv41
    import mtplx.models.deepseek_v41_moe as moe

    moe_before = moe.MoE.__call__
    layer_before = dv41.DecoderLayer.__call__
    rec = C.RouteRecorder(hidden=32, top_k=2, n_experts=8, hc_mult=2,
                          total_rows=1, hidden_stride=1, mx=mx)
    saved = C.install_hooks(rec, moe, dv41)
    assert moe.MoE.__call__ is not moe_before  # patched
    C.uninstall_hooks(saved)
    assert moe.MoE.__call__ is moe_before
    assert dv41.DecoderLayer.__call__ is layer_before


# ---------------------------------------------------------------------------
# trainer: phase split, coverage, metric ranges
# ---------------------------------------------------------------------------
def test_trace_phase_split_partitions_rows(tmp_path):
    C.collect_tiny_trace(tmp_path / "t", context_tokens=64,
                         prefill_tail=16, decode_tokens=8)
    tr = T.Trace(tmp_path / "t")
    assert tr.train_mask.sum() == 16 and tr.test_mask.sum() == 8
    assert not np.any(tr.train_mask & tr.test_mask)  # disjoint
    assert np.all(tr.train_mask | tr.test_mask)      # complete


def test_train_tiny_end_to_end_ridge(tmp_path):
    C.collect_tiny_trace(tmp_path / "t", context_tokens=96, prefill_tail=24,
                         decode_tokens=10)
    tr = T.Trace(tmp_path / "t")
    ks = [6, 8, 12]
    res = T.train_and_eval(tr, model="ridge", lam=1.0, steps=0, hidden=0, lr=0,
                           prefetch_ks=ks)
    # (a) covers every layer; (b)/(d) skip the first; (c) skips the first two
    assert len(res["a"]) == len(tr.layer_ids)
    assert len(res["b"]) == len(tr.layer_ids) - 1
    assert len(res["c"]) == len(tr.layer_ids) - 2
    assert len(res["d"]) == len(tr.layer_ids) - 1
    for p in ("a", "b", "c", "d"):
        for r in res[p]:
            assert 0.0 <= r["precision_at_k"] <= 1.0
            for K in ks:
                assert 0.0 <= r[f"miss_red_at_{K}"] <= 1.0
    summ = T.summarize(res, ks)
    assert summ["a"]["n_layers"] == len(tr.layer_ids)
    assert len(summ["a"]["worst5"]) <= 5


def test_miss_reduction_monotone_in_prefetch_width(tmp_path):
    d = _write_linear_trace(tmp_path / "lin", H=16, E=32, K=6, Ntr=800, Nte=200)
    tr = T.Trace(d)
    ks = [6, 8, 12]
    res = T.train_and_eval(tr, model="ridge", lam=1.0, steps=0, hidden=0, lr=0,
                           prefetch_ks=ks)
    for p in ("a", "b", "c"):
        for r in res[p]:
            assert r["miss_red_at_6"] <= r["miss_red_at_8"] + 1e-9
            assert r["miss_red_at_8"] <= r["miss_red_at_12"] + 1e-9


def test_ridge_recovers_a_recoverable_linear_router(tmp_path):
    # With a learnable linear router (low-dim, ample samples) predictor (a) --
    # router_in(L) -> top-k at L -- recovers most of the route: the sanity-check
    # mechanism works.  (On the real model this is the ceiling the useful
    # one-/two-ahead predictors are measured against.)
    d = _write_linear_trace(tmp_path / "lin", H=8, E=6, K=2, Ntr=3000, Nte=500)
    tr = T.Trace(d)
    res = T.train_and_eval(tr, model="ridge", lam=1.0, steps=0, hidden=0, lr=0,
                           prefetch_ks=[6, 8, 12])
    mean_prec = float(np.mean([r["precision_at_k"] for r in res["a"]]))
    assert mean_prec > 0.85, f"ridge did not recover the linear router: {mean_prec:.3f}"


def test_logistic_and_mlp_paths_run(tmp_path):
    C.collect_tiny_trace(tmp_path / "t", context_tokens=64, prefill_tail=16,
                         decode_tokens=8)
    tr = T.Trace(tmp_path / "t")
    ks = [6, 8, 12]
    for model in ("logistic", "mlp"):
        res = T.train_and_eval(tr, model=model, lam=1e-3, steps=30, hidden=16,
                               lr=0.5, prefetch_ks=ks)
        assert len(res["a"]) == len(tr.layer_ids)
        for r in res["a"]:
            assert 0.0 <= r["precision_at_k"] <= 1.0


def test_main_cli_tiny_smoke(tmp_path, capsys):
    # the trainer's --tiny path self-generates a trace and prints the report
    rc = T.main(["--tiny", "--model", "ridge"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "route-predictor feasibility" in out
    assert "one layer ahead" in out
