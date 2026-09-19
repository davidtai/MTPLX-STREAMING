"""F5 arm A2: the TimedPackedDecode stamp probe issues EXACTLY the retained lane's
MLX ops in the same order (CPU, no Metal, no import of the GPU lane).

The probe's ``run`` is derived from ``plane_lane.PackedDecode.run`` by pure
line-insertion (:func:`timed_plane_lane.stamp_run_source`).  These gates prove the
op sequence is unchanged WITHOUT running the lane:

  * the retained plane_lane.py the stamps are pinned to still hashes to the
    recorded sha256 (a drifted lane fails loudly);
  * stamp_run_source on the extracted ``PackedDecode.run`` source inserts exactly 6
    ``self._stamp`` calls + 1 ``self._note`` call, none of which contains an ``mx``
    op, and removing every inserted line recovers the original byte-for-byte (the
    "identical MLX ops in the same order" property, checked structurally);
  * the same transform composes on the projection-scheduled variant (the
    ``self.issue_next()`` insert projection_install.py:52-60 makes), so arm A2 keeps
    the retained next-layer projection expansion and still times it;
  * the summarizer's bucket math and the same-forward vs across-forward inter-layer
    gap split are correct on a synthetic stamp stream.

Extracts the run source via ``ast`` from the tracked receipt copy, so nothing here
imports paired_kernels / Metal.  MLX is pinned to CPU for parity with the suite.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "deepseek_v41" / "f5_compile"))
import timed_plane_lane as tpl  # noqa: E402

_PLANE_LANE = (
    _ROOT / "docs" / "deepseek-v41" / "receipts" / "extension-bank-20260919"
    / "full" / "sources" / "packed" / "plane_lane.py"
)


def _packed_run_source() -> str:
    text = _PLANE_LANE.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PackedDecode":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "run":
                    seg = ast.get_source_segment(text, item)
                    assert seg is not None
                    return seg
    raise AssertionError("PackedDecode.run not found in the retained plane_lane.py")


# ---------------------------------------------------------------------------
# 1. the retained lane the stamps target still hashes to the pinned sha256
# ---------------------------------------------------------------------------
def test_retained_plane_lane_pin():
    digest = hashlib.sha256(_PLANE_LANE.read_bytes()).hexdigest()
    assert digest == tpl.RETAINED_PLANE_LANE_SHA256, (
        f"plane_lane.py drifted: {digest} != pinned {tpl.RETAINED_PLANE_LANE_SHA256}")


# ---------------------------------------------------------------------------
# 2. stamp insertion is op-preserving: 6 stamps + 1 note, no mx op, round-trips
# ---------------------------------------------------------------------------
def test_stamp_run_source_op_preserving():
    src = _packed_run_source()
    stamped = tpl.stamp_run_source(src)
    stamp_lines = [ln for ln in stamped.splitlines() if ln.strip().startswith("self._stamp(")]
    note_lines = [ln for ln in stamped.splitlines() if ln.strip().startswith("self._note(")]
    assert len(stamp_lines) == 6, stamp_lines
    assert len(note_lines) == 1, note_lines
    # the six stamps are t0..t5 in order
    assert [ln.strip() for ln in stamp_lines] == [f"self._stamp({i})" for i in range(6)]
    # no inserted line issues an mx op
    for ln in stamp_lines + note_lines:
        assert "mx." not in ln, ln
    # removing every inserted line recovers the original run source byte-for-byte
    inserted = {ln.strip() for ln in stamp_lines + note_lines}
    recovered = "\n".join(ln for ln in stamped.splitlines() if ln.strip() not in inserted)
    assert recovered == "\n".join(src.splitlines())


# ---------------------------------------------------------------------------
# 3. composes on the projection-scheduled variant (issue_next kept + timed)
# ---------------------------------------------------------------------------
def test_stamp_composes_on_scheduled_variant():
    src = _packed_run_source()
    # Reproduce projection_install.py:52-60 scheduled insert (one line before the
    # shared_work anchor) without importing the GPU install module.
    anchor = "        if shared_work is not None:"
    assert src.count(anchor) == 1
    scheduled = src.replace(anchor, "        self.issue_next()\n" + anchor)
    stamped = tpl.stamp_run_source(scheduled)
    # strip stamps/notes -> recover the scheduled variant; strip issue_next -> src
    inserted = {f"self._stamp({i})" for i in range(6)}
    inserted |= {ln.strip() for ln in stamped.splitlines()
                 if ln.strip().startswith("self._note(")}
    back_to_scheduled = "\n".join(
        ln for ln in stamped.splitlines() if ln.strip() not in inserted)
    assert back_to_scheduled == "\n".join(scheduled.splitlines())
    assert "self.issue_next()" in back_to_scheduled  # projection expansion retained


# ---------------------------------------------------------------------------
# 4. summarizer bucket math + same-forward vs across-forward gap split
# ---------------------------------------------------------------------------
def test_summarize_buckets_and_inter_layer_gap():
    sink = tpl.ProbeSink()
    # 2 verify forwards x 3 layers = 6 calls. Each call: t0..t5 with per-bucket
    # deltas [10,20,30,40,50] ns and a 5 ns inter-call gap; across a forward
    # boundary (every 3 calls) the gap is 1000 ns (draft/accept/commit).
    clock = 0

    def emit_call(layer, rows, inter_gap):
        nonlocal clock
        # t0..t5: bucket widths 10,20,30,40,50
        for idx, width in zip(range(6), (0, 10, 20, 30, 40, 50)):
            clock += width
            sink.t_layer.append(layer)
            sink.t_idx.append(idx)
            sink.t_ns.append(clock)
        sink.m_layer.append(layer)
        sink.m_rows.append(rows)
        sink.m_parts.append(2)
        sink.m_hits.append(4)
        clock += inter_gap

    calls = [(0, 6), (1, 6), (2, 6), (0, 6), (1, 6), (2, 6)]
    for c, (layer, rows) in enumerate(calls):
        # gap AFTER a call: forward boundary after call 2 (index 2) and 5.
        gap = 1000 if (c + 1) % 3 == 0 else 5
        emit_call(layer, rows, gap)

    s = tpl.summarize(sink, verify_forward_layers=3)
    assert s["n_calls"] == 6
    b = s["buckets_seconds"]
    assert abs(b["barrier"]["mean"] - 10e-9) < 1e-15, b
    assert abs(b["host_pre_read"]["mean"] - 20e-9) < 1e-15, b
    assert abs(b["miss_wait"]["mean"] - 40e-9) < 1e-15, b
    assert abs(b["post"]["mean"] - 50e-9) < 1e-15, b
    # inter-layer (same forward): 4 gaps of 5 ns; across-forward: 1 gap of 1000 ns
    # (the last call has no successor).
    assert s["inter_layer_build_seconds"]["n"] == 4
    assert abs(s["inter_layer_build_seconds"]["mean"] - 5e-9) < 1e-15
    assert s["across_forward_gap_seconds"]["n"] == 1
    assert abs(s["across_forward_gap_seconds"]["mean"] - 1000e-9) < 1e-15
    # totals reconcile: per-call in-run buckets sum = 6*(150) ns.
    assert abs(s["totals_seconds"]["in_run_all_buckets"] - 6 * 150e-9) < 1e-15


# ---------------------------------------------------------------------------
# 5. the transform rejects a lane whose anchors are not unique (fails loudly)
# ---------------------------------------------------------------------------
def test_stamp_rejects_drifted_lane():
    src = _packed_run_source()
    # duplicate the routing-barrier line -> the mx.eval(indices) anchor is no longer
    # unique, so the transform must refuse rather than time an ambiguous lane.
    broken = src.replace("        mx.eval(indices)",
                         "        mx.eval(indices)\n        mx.eval(indices)")
    import pytest
    with pytest.raises(RuntimeError, match="not unique"):
        tpl.stamp_run_source(broken)
