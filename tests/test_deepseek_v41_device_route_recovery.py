"""W44 (K24) -- end-to-end cold recovery for the barrier-free device route.

Drives the REAL DeepSeek-V4.1 backbone decode/verify forward (real attention /
Hyper-Connection / compressor+index / engram advance / MoE gate+combine / cache)
with each layer's expert switch replaced by a controllable fake streamed switch
over a fake bank whose per-layer residency this test sets. It proves that, with
the device route armed, a cold token (misses) is repaired per-layer so the final
token output, the per-layer KV/compress/index cache state, and the engram state
are ALL byte-identical to the fully fenced path -- across all-hit / single-miss /
multi-miss / all-miss layer patterns at M=1 (AR) and M=4 (verify rows) -- and that
the routing barriers paid equal the number of miss layers.

CPU-pinned, no GPU, no artifact, no real bank. Run under ``nice -n 19``.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

from mtplx.expert_streaming import RoutingPhase  # noqa: E402
from mtplx.models.expert_mlx import (  # noqa: E402
    current_expert_routing_phase,
    expert_routing_phase,
)
from mtplx.models.deepseek_v41 import Model  # noqa: E402
from tests.test_deepseek_v41_served_generation import (  # noqa: E402
    _csa_args,
    _ngram_state,
    _randomize,
)

DEVICE = "MTPLX_DSV41_DEVICE_ROUTE"


@pytest.fixture(autouse=True)
def _cpu_and_flag():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = os.environ.get(DEVICE)
    os.environ.pop(DEVICE, None)
    try:
        yield
    finally:
        mx.set_default_device(prev)
        if saved is None:
            os.environ.pop(DEVICE, None)
        else:
            os.environ[DEVICE] = saved


class _FakeRT:
    """The device-route surface the backbone + fake switch use, over a residency
    map this test controls. Only ``barrier()`` (a fenced ``mx.eval(indices)``)
    counts toward routing barriers -- the device path issues none."""

    def __init__(self, residency):
        self.residency = {int(k): set(v) for k, v in residency.items()}
        self._device_route_force_fenced = frozenset()
        self.probes: list = []
        self.barriers = 0

    def flush_device_route_probes(self):
        pr, self.probes = self.probes, []
        misses = []
        for lid, ids, snap in pr:
            missed = tuple(sorted({e for e in ids if e not in snap}))
            if missed:
                misses.append((lid, missed))
        return misses

    def set_device_route_force_fenced(self, layers):
        self._device_route_force_fenced = frozenset(int(x) for x in layers)

    def admit(self, layer, experts):
        self.residency.setdefault(int(layer), set()).update(int(e) for e in experts)

    def barrier(self, indices):
        self.barriers += 1
        mx.eval(indices)


class _FakeSwitch(nn.Module):
    """Deterministic, expert-id-dependent stand-in for the streamed switch.

    device path: gather with the resident expert id, but a MISS expert clamps to
    id 0 (a void value) -- and enqueues a probe (no barrier). fenced path: gather
    with the true expert id (correct), pay one barrier, and admit any miss experts
    (so the next device route all-hits). All-hit device == fenced byte-for-byte."""

    def __init__(self, layer_id, rt):
        super().__init__()
        self.layer_id = int(layer_id)
        self.runtime = rt

    def __call__(self, xf, indices):
        rt = self.runtime
        device = (
            os.environ.get(DEVICE) == "1"
            and current_expert_routing_phase(token_count=int(xf.shape[0]))
            is RoutingPhase.DECODE
            and self.layer_id not in rt._device_route_force_fenced
        )
        ids = [[int(e) for e in row] for row in indices.tolist()]
        resident = rt.residency.get(self.layer_id, set())
        if device:
            eff = [[e if e in resident else 0 for e in row] for row in ids]
            rt.probes.append(
                (self.layer_id, [e for row in ids for e in row], frozenset(resident))
            )
        else:
            eff = ids
            rt.barrier(indices)
            miss = {e for row in ids for e in row if e not in resident}
            if miss:
                rt.admit(self.layer_id, miss)
        eff_arr = mx.array(eff, dtype=mx.float32)[..., None]  # [n, top_k, 1]
        routed = xf[:, None, :].astype(mx.float32) * (1.0 + 0.03 * eff_arr)
        return routed.astype(xf.dtype)


def _install(model, rt):
    for layer in model.model.layers:
        layer.mlp.switch_mlp = _FakeSwitch(layer.layer_id, rt)


def _cache_fingerprint(cache):
    """Per-layer (offset, window, compress_kv, index_k) + engram (_buf, _len)."""
    fp = []
    for lc in cache.layers:
        entry = {"offset": int(lc.offset)}
        for name in ("window", "compress_kv", "index_k"):
            a = getattr(lc, name, None)
            entry[name] = None if a is None else np.array(mx.eval(a) or a)
        fp.append(entry)
    es = cache.engram_state
    engram = None
    if es is not None:
        buf = es._buf
        engram = (None if buf is None else np.array(buf), int(es._len))
    return fp, engram


def _assert_cache_equal(a, b):
    fa, ea = a
    fb, eb = b
    assert len(fa) == len(fb)
    for i, (la, lb) in enumerate(zip(fa, fb)):
        assert la["offset"] == lb["offset"], f"layer {i} offset {la['offset']} != {lb['offset']}"
        for name in ("window", "compress_kv", "index_k"):
            xa, xb = la[name], lb[name]
            assert (xa is None) == (xb is None), f"layer {i} {name} presence differs"
            if xa is not None:
                assert xa.shape == xb.shape, f"layer {i} {name} shape {xa.shape}!={xb.shape}"
                assert np.array_equal(xa, xb), f"layer {i} {name} content differs"
    assert (ea is None) == (eb is None)
    if ea is not None:
        ba, la = ea
        bb, lb = eb
        assert la == lb, f"engram len {la} != {lb}"
        assert (ba is None) == (bb is None)
        if ba is not None:
            assert np.array_equal(ba, bb), "engram buffer differs"


def _run(pattern, m, *, seed=3, engram=False):
    """Build a fresh seeded model, install fake switches with the pattern's
    residency, run one span of ``m`` tokens (M=1 AR / M=4 verify), return
    (logits, cache_fingerprint, barriers)."""
    args = _csa_args()
    model = Model(args)
    _randomize(model, seed=seed)
    if engram:
        model.model.engram_hash = _ngram_state(args.vocab_size)

    n_layers = len(model.model.layers)
    # every token routes to a fixed set of experts; residency per the pattern.
    routed_experts = set(range(args.n_routed_experts))
    if pattern == "fenced":
        residency = {lid: set(routed_experts) for lid in range(n_layers)}
    elif pattern == "all_hit":
        residency = {lid: set(routed_experts) for lid in range(n_layers)}
    elif pattern == "single_miss":
        residency = {lid: set(routed_experts) for lid in range(n_layers)}
        residency[3] = routed_experts - {0, 1, 2, 3, 4, 5, 6}  # layer 3 misses most
    elif pattern == "multi_miss":
        residency = {lid: set(routed_experts) for lid in range(n_layers)}
        for lid in (2, 5, 6):
            residency[lid] = set()  # these layers all-miss
    elif pattern == "all_miss":
        residency = {lid: set() for lid in range(n_layers)}
    else:
        raise ValueError(pattern)

    rt = _FakeRT(residency)
    _install(model, rt)

    cache = model.make_cache()
    rng = np.random.default_rng(11)
    ids = mx.array(rng.integers(0, args.vocab_size, size=(1, m)), dtype=mx.int32)

    if pattern == "fenced":
        os.environ.pop(DEVICE, None)
    else:
        os.environ[DEVICE] = "1"

    def forward():
        return model(ids, cache=cache)

    if m == 1:
        logits = forward()
    else:
        # verify rows: force DECODE phase so the device route engages for M>1.
        with expert_routing_phase(RoutingPhase.DECODE):
            logits = forward()
    mx.eval(logits)
    return logits, _cache_fingerprint(cache), rt.barriers


# ---------------------------------------------------------------------------
# byte-identity of output + cache state, and barrier arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize(
    "pattern,expected_barriers",
    [("all_hit", 0), ("single_miss", 1), ("multi_miss", 3), ("all_miss", 8)],
)
def test_device_route_cold_recovery_matches_fenced(pattern, expected_barriers, m):
    ref_logits, ref_cache, _ref_bar = _run("fenced", m, seed=3)
    dev_logits, dev_cache, dev_bar = _run(pattern, m, seed=3)

    assert dev_logits.shape == ref_logits.shape
    assert mx.array_equal(dev_logits, ref_logits), (
        f"{pattern} M={m}: device-route output != fenced"
    )
    _assert_cache_equal(dev_cache, ref_cache)
    # barriers/token == number of miss layers (all-hit pays none; the recovery
    # pass fences exactly the miss layers).
    assert dev_bar == expected_barriers, (
        f"{pattern} M={m}: {dev_bar} barriers, expected {expected_barriers}"
    )


# ---------------------------------------------------------------------------
# engram state survives recovery untouched (advanced once/token, never trimmed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 4])
def test_device_route_recovery_preserves_engram_state(m):
    ref_logits, ref_cache, _ = _run("fenced", m, seed=5, engram=True)
    dev_logits, dev_cache, dev_bar = _run("multi_miss", m, seed=5, engram=True)
    assert mx.array_equal(dev_logits, ref_logits), f"M={m}: output != fenced (engram)"
    _assert_cache_equal(dev_cache, ref_cache)  # includes engram _buf/_len
    assert dev_bar == 3
