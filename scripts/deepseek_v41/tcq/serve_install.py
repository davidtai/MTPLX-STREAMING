"""F39: install the tcq3 decode into the SERVED DeepSeek-V4.1 lane (`mtplx serve`), armed by MTPLX_DSV41_TCQ3=1.

The served decode path is `expert_mlx.py` (independent of the packed benchmark lane): the switch's
`_dispatch_component_bank` (`:2394`) runs one `gather_qmm(mode="mxfp4")` per projection over the layer's component
bank with `rhs_indices = the routed slots`. This module adds the tcq3 equivalent.

IMPORTANT layout finding (differs from the packed lane / the original task framing).  The server's component-bank
allocator (`make_mlx_component_bank_allocator` -> `MlxComponentBank(record)`) is **generic and record-driven**: it
allocates one array PER RECORD SEGMENT.  A tcq3 record has six segments, so the served bank holds **per-projection**
arrays ``{proj}.code`` (int16 [cap, IN/16, OUT/16, 48]) and ``{proj}.rout`` (f16 [cap, OUT]) — NOT one whole-record
slot.  So the fit is the F35 **contiguous** tile kernel (``tcq_kernels.make_tcq_projection(OUT, IN, "tile")``) over
``bank.arrays[f"{proj}.code"]`` with the routed slots as ``ids``, and the rout read per-slot from
``bank.arrays[f"{proj}.rout"]`` — the strided/whole-record kernel + a 299 MB resident rout table is the packed-lane
(TcqPackedOps) model, which the served allocator does not produce.  The decode MATH is identical either way:
``y = t128( t128(x) @ W_q ) * rout`` (matmul in the trellis domain), validated against ``tcq_encode.effective_weight``.

What this installs (all env-gated; stock mxfp4/affine untouched when MTPLX_DSV41_TCQ3 != 1):
  1. record validator — ``tcq3_expected_signature`` / ``validate_tcq3_record`` accept a tcq3 record (13,290,496 B,
     code+rout per projection, no scales).  (The one inline seam in ``make_mlx_component_bank_allocator`` that must
     also accept it is a one-line edit given in the F39 report — it cannot be monkeypatched without copying that
     250-line function; the validator FUNCTION here is the ready piece + the truth-table test.)
  2. decode dispatch — ``_dispatch_component_bank`` is rebound so a ``tcq3`` switch runs the contiguous tile kernel
     (t128 in, t128*rout out) instead of ``gather_qmm``.  NOT decode-to-bf16 per token (6 experts x 40 layers x ~35M
     weights/token would make the eval take days).
  3. prefill — the dense-prefill analog: ``decode_expert_to_bf16`` once per routed expert then a dense matmul (like
     the server's ``_dequantize_mxfp4_slot`` + ``_run_component_bank_dense_prefill``).

Not CPU-testable: the tile-kernel dispatch is Metal (validated by ``serve_parity.py`` in a guarded window).  CPU-
tested: the wrapper MATH (F34 sample) equals ``tcq_encode.effective_weight``; the monkeypatch round-trips and is a
no-op when the env is unset; the validator truth table.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRELLIS = os.path.join(os.path.dirname(_HERE), "trellis")
if _TRELLIS not in sys.path:
    sys.path.insert(0, _TRELLIS)

TCQ3_CODEC = "tcq3"
NW = 48
ENV_FLAG = "MTPLX_DSV41_TCQ3"
# eschamoe orientation W[in,out]: gate/up decode [in=5120, out=2304]; down [in=2304, out=5120].
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- record validator (item 1)

def tcq3_expected_signature(spec) -> tuple:
    """The six-segment (component, dtype, shape, length) signature of a tcq3 record for this model spec.

    Mirrors transcode_bank.record_layout on the spec dims: per projection an int16 code [IN/16, OUT/16, 48] and an
    f16 rout [OUT]; no scale leaf.  ``spec.hidden_size`` (5120) and ``spec.expert_hidden_size`` (2304)."""
    sig = []
    for proj in _PROJECTIONS:
        out_f = spec.expert_hidden_size if proj in ("gate_proj", "up_proj") else spec.hidden_size
        in_f = spec.hidden_size if proj in ("gate_proj", "up_proj") else spec.expert_hidden_size
        nI, nJ = in_f // 16, out_f // 16
        code_len = nI * nJ * NW * 2
        sig.append((f"{proj}.code", "I16", (nI, nJ, NW), code_len))
        sig.append((f"{proj}.rout", "F16", (out_f,), out_f * 2))
    return tuple(sig)


def validate_tcq3_record(record_signature, spec) -> None:
    """Raise ValueError unless ``record_signature`` (a tuple of (component, dtype, shape, length)) is a tcq3 record.

    ``record_signature`` is ``tuple((seg.component, seg.dtype, tuple(seg.shape), seg.length) for seg in
    record.segments)`` — exactly what ``make_mlx_component_bank_allocator``'s ``component_signature`` builds."""
    expected = tcq3_expected_signature(spec)
    got = tuple(tuple(x) for x in record_signature)
    if got != expected:
        raise ValueError(f"manifest record geometry is not tcq3: got {got} != expected {expected}")


# ---------------------------------------------------------------- decode math (item 2) — CPU-testable via `matmul`

def _tcq3_project(x, *, code_bank, rout_bank, slot_ids, out_dim, tables, matmul=None):
    """y = t128( t128(x) @ W_q ) * rout for one projection, over the server's per-projection component bank.

    ``code_bank`` = bank.arrays[f"{proj}.code"] (int16 [cap, IN/16, OUT/16, 48]); ``rout_bank`` = bank.arrays
    [f"{proj}.rout"] (f16 [cap, OUT]); ``slot_ids`` = the routed slots (rhs_indices).  ``matmul(xh)`` computes
    ``xh @ W_q``: the F35 contiguous tile kernel on GPU, or an injected CPU matmul for the math test."""
    import mlx.core as mx
    import tcq_runtime as R
    import tcq_kernels as tk
    xh = R._t128(mx, x.astype(mx.float32))
    if matmul is None:
        kern = tk.make_tcq_projection(out_dim, x.shape[-1], "tile")
        z = tk.run_tcq_projection(kern, "tile", xh, slot_ids, code_bank, out_dim, tables)
    else:
        z = matmul(xh)
    rout = mx.take(rout_bank, slot_ids, axis=0).astype(mx.float32)      # [rows, OUT]
    return R._t128(mx, z) * rout


def run_component_bank_tcq3(selected, bindings, *, swiglu_limit, tables):
    """Decode wave: one tile-kernel projection per gate/up/down over the layer's tcq3 component bank + t128/rout.

    Drop-in for ``_run_component_bank_q4`` on a tcq3 switch: ``selected`` [rows, hidden], ``bindings`` carry the
    routed slots (``binding.buffer.bank_index``) and the shared bank.  Returns [rows, hidden]."""
    import mlx.core as mx
    from mtplx.models.expert_mlx import _clamped_swiglu
    bank = bindings[0].buffer.bank
    slot_ids = mx.array([int(b.buffer.bank_index) for b in bindings], dtype=mx.uint32)
    rows = int(selected.shape[0])
    x = selected.reshape(rows, int(selected.shape[-1]))

    def proj(inp, name, out_dim):
        return _tcq3_project(inp, code_bank=bank.arrays[f"{name}.code"], rout_bank=bank.arrays[f"{name}.rout"],
                             slot_ids=slot_ids, out_dim=out_dim, tables=tables)

    gate = proj(x, "gate_proj", 2304)
    up = proj(x, "up_proj", 2304)
    h = _clamped_swiglu(gate, up, swiglu_limit)
    down = proj(h, "down_proj", 5120)
    return down.reshape(rows, int(down.shape[-1]))


# ---------------------------------------------------------------- prefill (item 3) — decode-to-bf16 once per expert

def run_component_bank_tcq3_prefill(selected, bindings, *, swiglu_limit):
    """Prefill wave: decode each routed expert's tcq3 record to a dense bf16 effective weight ONCE, then dense matmul
    (the tcq3 analog of ``_dequantize_mxfp4_slot`` + ``_run_component_bank_dense_prefill``)."""
    import mlx.core as mx
    import tcq_runtime as R
    from mtplx.models.expert_mlx import _clamped_swiglu
    bank = bindings[0].buffer.bank
    rows = int(selected.shape[0])
    x = selected.reshape(rows, int(selected.shape[-1]))
    slot_of_row = [int(b.buffer.bank_index) for b in bindings]
    groups: dict[int, list[int]] = {}
    for row, slot in enumerate(slot_of_row):
        groups.setdefault(slot, []).append(row)
    # E_eff[in,out] per slot per projection = decode_expert_to_bf16(code[slot], rout[slot]); one decode/expert.
    out = mx.zeros((rows, 5120), dtype=mx.float32)
    order, parts = [], []
    for slot, slot_rows in groups.items():
        xr = mx.take(x, mx.array(slot_rows, mx.int32), axis=0)
        def eff(name):
            return R.decode_expert_to_bf16(bank.arrays[f"{name}.code"][slot], bank.arrays[f"{name}.rout"][slot]).astype(mx.float32)
        g = xr @ eff("gate_proj")
        u = xr @ eff("up_proj")
        h = _clamped_swiglu(g, u, swiglu_limit)
        parts.append(h @ eff("down_proj"))
        order.extend(slot_rows)
    joined = mx.concatenate(parts, axis=0)
    inv = mx.argsort(mx.array(order, mx.int32))
    return mx.take(joined, inv, axis=0)


# ---------------------------------------------------------------- install / uninstall (monkeypatch the dispatch)

_INSTALLED = {"orig": {}}


def _switch_classes():
    from mtplx.models import expert_mlx
    return [getattr(expert_mlx, n) for n in ("HotExpertSwitchGLU", "MappedExpertSwitchGLU", "DenseIslandSwitchGLU")
            if hasattr(getattr(expert_mlx, n, None), "_dispatch_component_bank")]


def install(*, tables=None) -> dict:
    """Rebind ``_dispatch_component_bank`` on the streamed switch classes so a tcq3 switch decodes via the tile
    kernel (and prefills via decode-to-bf16).  Idempotent; a no-op for every non-tcq3 switch (checks ``self.codec``).
    Armed by MTPLX_DSV41_TCQ3=1 (the caller — eval_driver's serve step / a serve site hook — calls this)."""
    if _INSTALLED["orig"]:
        return {"already": True}
    if tables is None:
        from tcq_kernel_check import warp_tables
        tables = warp_tables()
    classes = _switch_classes()
    for cls in classes:
        orig = cls._dispatch_component_bank
        _INSTALLED["orig"][cls] = orig

        def _patched(self, selected, bindings, *, dense_prefill=False, _orig=orig):
            if getattr(self, "codec", None) == TCQ3_CODEC:
                if dense_prefill:
                    return run_component_bank_tcq3_prefill(selected, bindings, swiglu_limit=self.swiglu_limit)
                return run_component_bank_tcq3(selected, bindings, swiglu_limit=self.swiglu_limit, tables=tables)
            return _orig(self, selected, bindings, dense_prefill=dense_prefill)

        cls._dispatch_component_bank = _patched
    return {"installed_on": [c.__name__ for c in classes]}


def uninstall() -> None:
    """Restore the original ``_dispatch_component_bank`` (round-trip; used by the CPU test and for a clean re-arm)."""
    for cls, orig in _INSTALLED["orig"].items():
        cls._dispatch_component_bank = orig
    _INSTALLED["orig"] = {}


def install_from_env(*, tables=None):
    """Serve-construction entry: install only when MTPLX_DSV41_TCQ3=1; otherwise leave the stock decode untouched."""
    if os.environ.get(ENV_FLAG) != "1":
        return None
    return install(tables=tables)
