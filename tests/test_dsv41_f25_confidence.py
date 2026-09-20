"""CPU tests for the F25 confidence-gated draft-length stager (no MLX, no GPU).

Pure text edits against the REAL archived retained source (docs/deepseek-v41/receipts/
extension-bank-20260919/full/sources/packed/hybrid_install.py): the anchors are unique,
the two edits round-trip byte-for-byte, they refuse a second/mangled application, they
compose with the verify-chunks stagers in either order, and the edited install() still
compiles + still refuses a wrong requested_depth. hybrid_install.py imports no MLX, so a
fake ``lookup`` module is enough to load install(). Run under nice -n 19.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import f2.stage_f2_runner as st  # noqa: E402

_HYBRID = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed/hybrid_install.py"

_REMOVED = " or confidence_threshold is not None"           # the exact substring the edit removes
_ADDED = "'confidence_threshold':confidence_threshold,"      # the report-dict key the edit adds


# --------------------------------------------------------------------------- anchors
def test_confidence_anchors_unique_and_not_pre_staged():
    src = _HYBRID.read_text()
    assert src.count(st._CONF_REFUSAL_ANCHOR) == 1
    assert src.count(st._CONF_REPORT_ANCHOR) == 1
    assert _REMOVED in src                 # the clause the edit drops is present exactly as written
    assert src.count(_REMOVED) == 1
    assert _ADDED not in src               # not already staged


# --------------------------------------------------------------------- apply + round-trip
def test_confidence_applies_once_and_roundtrips():
    src = _HYBRID.read_text()
    out = st.stage_confidence(src)
    assert out != src
    assert _REMOVED not in out             # refusal clause gone
    assert _ADDED in out                   # report carries the live value
    # reverse both edits -> the archived source, byte-for-byte.
    step = out.replace(st._CONF_REPORT_NEW, st._CONF_REPORT_ANCHOR)
    recovered = step.replace(st._CONF_REFUSAL_NEW, st._CONF_REFUSAL_ANCHOR)
    assert recovered == src


def test_confidence_refuses_second_application():
    out = st.stage_confidence(_HYBRID.read_text())
    with pytest.raises(RuntimeError):
        st.stage_confidence(out)


# ---------------------------------------------------------- missing / duplicated anchors
def test_confidence_refuses_when_refusal_anchor_missing():
    src = _HYBRID.read_text().replace(st._CONF_REFUSAL_ANCHOR, "    if (requested_depth != 5")
    with pytest.raises(RuntimeError):
        st.stage_confidence(src)


def test_confidence_refuses_when_refusal_anchor_duplicated():
    src = _HYBRID.read_text()
    dup = src.replace(st._CONF_REFUSAL_ANCHOR, st._CONF_REFUSAL_ANCHOR + "\n" + st._CONF_REFUSAL_ANCHOR, 1)
    with pytest.raises(RuntimeError):
        st.stage_confidence(dup)


def test_confidence_refuses_when_report_anchor_missing():
    src = _HYBRID.read_text().replace(st._CONF_REPORT_ANCHOR, "    return {'native_head_depth':6,")
    with pytest.raises(RuntimeError):
        st.stage_confidence(src)


# ---------------------------------------------------- composition with the verify stagers
def test_confidence_composes_with_verify_balanced_in_either_order():
    src = _HYBRID.read_text()
    a = st.stage_confidence(st.stage_verify_balanced(src))
    b = st.stage_verify_balanced(st.stage_confidence(src))
    assert a == b                          # disjoint anchors -> order-independent
    assert _REMOVED not in a and _ADDED in a and "_BALANCED_CHUNKS" in a
    compile(a, "hybrid_install_conf_balanced", "exec")


def test_confidence_composes_with_verify_chunks_in_either_order():
    src = _HYBRID.read_text()
    a = st.stage_confidence(st.stage_verify_chunks(src, chunks=(4, 4)))
    b = st.stage_verify_chunks(st.stage_confidence(src), chunks=(4, 4))
    assert a == b
    assert _REMOVED not in a and _ADDED in a and "verify_chunks = (4, 4)" in a
    compile(a, "hybrid_install_conf_chunks", "exec")


# ------------------------------------------------------------- AST: only the clause dropped
def _refusal_or_terms(source: str):
    """The unparsed OR-terms of install()'s first ``if`` refusal condition."""
    fn = next(n for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.FunctionDef) and n.name == "install")
    if_node = next(n for n in fn.body if isinstance(n, ast.If))
    assert isinstance(if_node.test, ast.BoolOp) and isinstance(if_node.test.op, ast.Or)
    return [ast.dump(v) for v in if_node.test.values]


def test_edit_drops_only_the_confidence_clause_from_the_refusal():
    src = _HYBRID.read_text()
    before = _refusal_or_terms(src)
    after = _refusal_or_terms(st.stage_confidence(src))
    dropped = [t for t in before if "confidence_threshold" in t]
    assert len(dropped) == 1                          # exactly one confidence clause existed
    assert after == [t for t in before if "confidence_threshold" not in t]  # rest unchanged & in order


def test_edited_report_dict_maps_confidence_threshold_to_the_argument():
    out = st.stage_confidence(_HYBRID.read_text())
    fn = next(n for n in ast.walk(ast.parse(out))
              if isinstance(n, ast.FunctionDef) and n.name == "install")
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.Dict)
    vals = [v for k, v in zip(ret.value.keys, ret.value.values)
            if isinstance(k, ast.Constant) and k.value == "confidence_threshold"]
    assert len(vals) == 1 and isinstance(vals[0], ast.Name) and vals[0].id == "confidence_threshold"


# ------------------------------------- functional: compiles + refusal behaviour (fakes)
class _Reached(Exception):
    """Raised by the fake LookupExtension to prove install() passed the refusal gate."""


def _load_install(monkeypatch):
    fake = types.ModuleType("lookup")
    fake.LookupExtension = lambda *a, **k: (_ for _ in ()).throw(_Reached())
    monkeypatch.setitem(sys.modules, "lookup", fake)
    out = st.stage_confidence(_HYBRID.read_text())
    ns: dict = {}
    exec(compile(out, "hybrid_install_staged", "exec"), ns)   # edited source still compiles + loads
    return ns["install"]


def _valid_model():
    cfg = types.SimpleNamespace(max_live_kv_tokens=17664)
    return types.SimpleNamespace(mtp=types.SimpleNamespace(block_size=5),
                                 _mtplx_expert_runtime=types.SimpleNamespace(config=cfg))


def test_edited_install_still_refuses_wrong_depth(monkeypatch):
    install = _load_install(monkeypatch)
    # requested_depth != 5 short-circuits the OR -> the refusal fires before LookupExtension.
    with pytest.raises(RuntimeError, match="lookup extension requires"):
        install(object(), _valid_model(), [0] * 16384,
                requested_depth=7, verify_chunks=None, confidence_threshold=0.6)


def test_edited_install_accepts_a_live_confidence_threshold(monkeypatch):
    install = _load_install(monkeypatch)
    # depth/verify/model all valid + a non-None threshold: the OLD clause would have refused;
    # the edited condition is False, so install() proceeds and hits the (faked) LookupExtension.
    with pytest.raises(_Reached):
        install(object(), _valid_model(), [0] * 16384,
                requested_depth=5, verify_chunks=None, confidence_threshold=0.6)
