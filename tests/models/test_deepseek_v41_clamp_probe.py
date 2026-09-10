"""W11 follow-up, task 2: is the SwiGLU clamp load-bearing on a real prompt?

Instrumentation (test-only, CPU, gated behind DSV41_RUN_CLAMP_PROBE=1 because it
runs full 40-layer streamed forwards on the real artifact). Prefills the 31-token
probe A ("def add(a, b): ...", the same input whose tail the old un-clamped forward
turned to junk) through the real component-banks loader, TWICE on one model load:
clamp ON (spec.swiglu_limit=10, the fix) and clamp OFF (swiglu_limit=None, the old
behavior). It reports:

  * teacher-forced next-token argmax vs ground truth, clamp ON vs OFF, and which
    positions the clamp changes / fixes -- the causal test for "did the missing
    clamp cause the tail junk";
  * per layer, on the OLD (un-clamped) trajectory: max |gate|, max |up|, the count
    of activations the reference clamp would cut (gate > +10, |up| > 10), and every
    (layer, position) exceeding +/-10, compared with probe A's junk positions
    6, 11, 12, 19, 20, 21, 24, 26.

Writes docs/deepseek-v41/receipts/cpu_clamp_probe.json.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

ARTIFACT = Path(
    os.environ.get(
        "DSV41_ARTIFACT",
        os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2"),
    )
)
MANIFEST = ARTIFACT / "expert-manifest.json"
BANK = ARTIFACT / "experts.bin"
GIB = 1024**3
LIMIT = 10.0

# Probe A (docs/deepseek-v41/receipts/cpu_decode_probe_ABC.json): BOS + the
# add/sub/mul source, the exact 31-token teacher-forced prefill the decode probe
# scored 16/30 on (un-clamped). Junk = positions the coordinator flagged.
PROBE_IDS = [
    0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291, 6328, 3465,
    1241, 6036, 14, 291, 3395, 361, 1354, 260, 565, 291, 6328, 3465, 21740,
    6036, 14, 291, 2605,
]
JUNK_POSITIONS = [6, 11, 12, 19, 20, 21, 24, 26]

RECEIPT = Path(__file__).resolve().parents[2] / "docs/deepseek-v41/receipts/cpu_clamp_probe.json"

pytestmark = pytest.mark.skipif(
    not os.environ.get("DSV41_RUN_CLAMP_PROBE"),
    reason="set DSV41_RUN_CLAMP_PROBE=1 to run the heavy CPU clamp probe",
)


def _mx_from_raw(raw, dtype, shape):
    if dtype == "BF16":
        return mx.array(np.frombuffer(raw, "<u2").reshape(shape).copy()).view(mx.bfloat16)
    if dtype in ("U32", "I32"):
        return mx.array(np.frombuffer(raw, "<u4").reshape(shape).copy().astype(np.uint32))
    raise ValueError(dtype)


def _dequant_proj(fd, record, proj):
    parts = {}
    for seg in record.segments:
        p, leaf = seg.component.split(".")
        if p == proj:
            parts[leaf] = _mx_from_raw(os.pread(fd, seg.length, seg.offset), seg.dtype, tuple(seg.shape))
    return mx.dequantize(parts["weight"], parts["scales"], parts["biases"], group_size=64, bits=2).astype(mx.float32)


class _SwitchRecorder(nn.Module):
    """Wrap a bound streamed switch to capture its (x, indices) into ``store``."""

    def __init__(self, inner, layer_index, store):
        super().__init__()
        self.inner = inner
        self._layer = int(layer_index)
        self._store = store

    def __call__(self, x, indices):
        mx.eval(x, indices)
        self._store[self._layer] = (
            np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32)),
            np.array(indices.reshape(-1, indices.shape[-1]).astype(mx.int32)),
        )
        return self.inner(x, indices)


def _preds(logits):
    return [int(mx.argmax(logits[0, i]).item()) for i in range(len(PROBE_IDS) - 1)]


def _matches(preds):
    return [p for p in range(len(preds)) if preds[p] == PROBE_IDS[p + 1]]


def _measure_preactivations(store, fd, rec_map, n_pos):
    """Per-layer max |gate|/|up|, count > +/-10, and positions over, on ``store``."""
    layers_report, positions_over = [], {}
    gmax_g = gmax_u = 0.0
    for layer in sorted(store):
        x_np, idx = store[layer]
        x = mx.array(x_np)
        pmax_g = np.zeros(n_pos)
        pmax_u = np.zeros(n_pos)
        gate_over = up_over = 0
        for e in sorted({int(v) for v in idx.reshape(-1)}):
            rows = [p for p in range(n_pos) if e in idx[p].tolist()]
            if not rows:
                continue
            wg = _dequant_proj(fd, rec_map[(layer, e)], "gate_proj")
            wu = _dequant_proj(fd, rec_map[(layer, e)], "up_proj")
            gate = np.array(x @ wg.T)
            up = np.array(x @ wu.T)
            del wg, wu
            for p in rows:
                pmax_g[p] = max(pmax_g[p], float(np.abs(gate[p]).max()))
                pmax_u[p] = max(pmax_u[p], float(np.abs(up[p]).max()))
                gate_over += int((gate[p] > LIMIT).sum())
                up_over += int((np.abs(up[p]) > LIMIT).sum())
            mx.clear_cache()
        over = [p for p in range(n_pos) if pmax_g[p] > LIMIT or pmax_u[p] > LIMIT]
        if over:
            positions_over[layer] = over
        gmax_g = max(gmax_g, float(pmax_g.max()))
        gmax_u = max(gmax_u, float(pmax_u.max()))
        layers_report.append(
            {
                "layer": layer,
                "max_abs_gate": round(float(pmax_g.max()), 4),
                "max_abs_up": round(float(pmax_u.max()), 4),
                "gate_gt_10": gate_over,
                "up_abs_gt_10": up_over,
                "positions_over_10": over,
            }
        )
    return layers_report, positions_over, gmax_g, gmax_u


def test_streaming_clamp_probe_measurement():
    from mtplx.expert_manifest import load_expert_manifest
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming

    os.environ.setdefault("MTPLX_ENGRAM_CACHE_LIMIT", "64MiB")
    t0 = time.time()
    resident = load_deepseek_v41_streaming(
        ARTIFACT,
        memory_limit_bytes=int(100 * GIB),
        max_live_kv_tokens=4096,
        admit=True,
        expert_cache_limit_bytes=int(15 * GIB),
        apply_memory_cap=False,
        slot_layout="component-banks",
        cache_scope="layer",
        island_layers=(),
        verify_record_hashes=False,
    )
    model = resident.model

    capture: dict[int, tuple] = {}
    recorders = []
    for i, layer in enumerate(model.model.layers):
        sw = getattr(layer.mlp, "switch_mlp", None)
        if sw is not None:
            assert getattr(sw, "swiglu_limit", "missing") == LIMIT, (i, getattr(sw, "swiglu_limit", "missing"))
            rec = _SwitchRecorder(sw, i, capture)
            layer.mlp.switch_mlp = rec
            recorders.append(rec)

    # clamp ON (the fix): spec.swiglu_limit=10 flows to every switch.
    logits_on = model(mx.array([PROBE_IDS]))
    mx.eval(logits_on)
    preds_on = _preds(logits_on)

    # clamp OFF (the old behavior): force every switch back to plain SwiGLU and
    # re-prefill on the same load; capture this (old) trajectory for the pre-act
    # measurement.
    for rec in recorders:
        rec.inner.swiglu_limit = None
    capture.clear()
    logits_off = model(mx.array([PROBE_IDS]))
    mx.eval(logits_off)
    store_off = dict(capture)
    preds_off = _preds(logits_off)
    assert set(store_off) == set(range(len(model.model.layers)))
    load_forward_s = time.time() - t0

    match_on, match_off = _matches(preds_on), _matches(preds_off)
    diff = [p for p in range(len(preds_on)) if preds_on[p] != preds_off[p]]
    fixed = [p for p in diff if p in match_on and p not in match_off]
    broke = [p for p in diff if p in match_off and p not in match_on]

    manifest = load_expert_manifest(MANIFEST)
    rec_map = {(r.layer, r.expert): r for r in manifest.records}
    n_pos = len(PROBE_IDS)
    fd = os.open(BANK, os.O_RDONLY)
    try:
        layers_report, positions_over, gmax_g, gmax_u = _measure_preactivations(
            store_off, fd, rec_map, n_pos
        )
    finally:
        os.close(fd)

    over_at_junk = sorted({p for over in positions_over.values() for p in over if p in JUNK_POSITIONS})
    receipt = {
        "schema": 1,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe": "A (def add/sub/mul, BOS)",
        "n_tokens": n_pos,
        "swiglu_limit": LIMIT,
        "slot_layout": "component-banks",
        "load_two_forward_seconds": round(load_forward_s, 1),
        "teacher_forced": {
            "targets": PROBE_IDS[1:],
            "preds_clamp_on": preds_on,
            "preds_clamp_off": preds_off,
            "match_on": len(match_on),
            "match_off": len(match_off),
            "total": len(preds_on),
            "positions_changed_by_clamp": diff,
            "positions_fixed_by_clamp": fixed,
            "positions_broken_by_clamp": broke,
            "junk_positions": JUNK_POSITIONS,
            "junk_fixed_by_clamp": [p for p in fixed if p in JUNK_POSITIONS],
        },
        "preactivation_old_trajectory": {
            "global_max_abs_gate": round(gmax_g, 4),
            "global_max_abs_up": round(gmax_u, 4),
            "any_over_limit": bool(positions_over),
            "positions_over_limit_by_layer": {str(k): v for k, v in sorted(positions_over.items())},
            "over_limit_at_junk_positions": over_at_junk,
            "layers": layers_report,
        },
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2))

    print(
        f"[clamp-probe] teacher-forced match: clamp OFF {len(match_off)}/{len(preds_off)} -> "
        f"ON {len(match_on)}/{len(preds_on)}; changed={diff} fixed={fixed} broke={broke}"
    )
    print(
        f"[clamp-probe] OLD-trajectory pre-act: global max|gate|={gmax_g:.2f} max|up|={gmax_u:.2f} "
        f"any>10={bool(positions_over)} over@junk={over_at_junk} ({load_forward_s:.0f}s)"
    )
    for row in layers_report:
        if row["positions_over_10"]:
            print(f"[clamp-probe]  L{row['layer']:02d} max|gate|={row['max_abs_gate']:.1f} "
                  f"max|up|={row['max_abs_up']:.1f} gate>10={row['gate_gt_10']} "
                  f"|up|>10={row['up_abs_gt_10']} over={row['positions_over_10']}")

    assert np.isfinite(gmax_g) and np.isfinite(gmax_u)
    assert RECEIPT.is_file()
