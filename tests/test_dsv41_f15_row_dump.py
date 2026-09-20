"""CPU tests for the F15 tie-classifier harness (row_dump + stager + classify_pair).

MLX is pinned to the CPU device at the very top of the module (MLX defaults to Metal;
"no GPU" is not the same as CPU). Nothing here touches Metal, the GPU lock, or the
production service. Run with: ``nice -n 19 pytest tests/test_dsv41_f15_row_dump.py``
(no -n auto).
"""
from __future__ import annotations

# --- CPU pin BEFORE importing anything that inits MLX -----------------------
import mlx.core as mx

mx.set_default_device(mx.cpu)

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Pinned runtime (provides mtplx + classify_divergence) and this worktree's f15 dir.
PINNED_RUNTIME = "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-run-d5f15e7a"
_WT = "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f15-tieclass"
F15_DIR = _WT + "/scripts/deepseek_v41/f15"
RUN_FULL = _WT + "/docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed/run_full.py"
F15_STAGER = F15_DIR + "/stage_f15_runner.py"
F5_STAGER = "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f5-compile/scripts/deepseek_v41/f5_compile/stage_f5_runner.py"
F6_STAGER = "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f6-engram/scripts/deepseek_v41/f6/stage_f6_runner.py"

sys.path.insert(0, PINNED_RUNTIME)  # `import mtplx` -> pinned runtime
sys.path.insert(0, F15_DIR)         # `import row_dump`, `import classify_pair`

import row_dump  # noqa: E402
import classify_pair  # noqa: E402

# The wrap must target the pinned runtime's DivergenceCapture, not an editable install.
assert row_dump._dec.__file__.startswith(PINNED_RUNTIME), row_dump._dec.__file__

# The true, unwrapped observe -- captured once, before any install, for hard resets.
_TRUE_OBSERVE = row_dump._dec.DivergenceCapture.observe


def _hard_reset() -> None:
    row_dump._dec.DivergenceCapture.observe = _TRUE_OBSERVE
    row_dump._ORIG_OBSERVE = None
    row_dump._STATE = None
    for key in (row_dump.ENV_DIR, row_dump.ENV_INDICES):
        os.environ.pop(key, None)


@pytest.fixture(autouse=True)
def _clean_rowdump():
    _hard_reset()
    yield
    _hard_reset()


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# parse_indices
# ===========================================================================
def test_parse_indices_list_and_ranges():
    assert row_dump.parse_indices("297,470-490,500") == {297, 500} | set(range(470, 491))
    assert row_dump.parse_indices(" 3 ; 5-7 , 9 ") == {3, 5, 6, 7, 9}
    assert row_dump.parse_indices("480-480") == {480}
    assert row_dump.parse_indices("") == set()
    assert row_dump.parse_indices(None) == set()
    assert row_dump.parse_indices("  ,  ") == set()


def test_parse_indices_rejects_malformed():
    for bad in ("abc", "4-", "-4", "1.5", "490-470", "1_0", "3,x"):
        with pytest.raises(ValueError):
            row_dump.parse_indices(bad)


# ===========================================================================
# install gating
# ===========================================================================
def test_no_files_when_env_unset(tmp_path):
    os.environ.pop(row_dump.ENV_DIR, None)
    assert row_dump.install_from_env() is None
    assert row_dump.state() is None
    assert row_dump._ORIG_OBSERVE is None
    # class method untouched -> no wrap installed
    assert row_dump._dec.DivergenceCapture.observe is _TRUE_OBSERVE
    # nothing was created on disk
    assert list(tmp_path.iterdir()) == []


def test_install_is_idempotent(tmp_path):
    os.environ[row_dump.ENV_DIR] = str(tmp_path / "d")
    os.environ[row_dump.ENV_INDICES] = "5"
    st1 = row_dump.install_from_env()
    st2 = row_dump.install_from_env()
    assert st1 is st2  # one state per process
    assert row_dump._dec.DivergenceCapture.observe is row_dump._wrapped_observe


# ===========================================================================
# wrapper: identical args + return, original side effects preserved
# ===========================================================================
def test_wrapper_calls_original_with_identical_args_and_return(tmp_path):
    dec = row_dump._dec
    ar_ref = [5, 5, 5, 5, 5, 5]
    vl = mx.array(np.arange(1 * 2 * 4, dtype=np.float32).reshape(1, 2, 4))

    # baseline: run the true original on a fresh instance (committed[0]=9 != ar_ref[1]=5)
    base = dec.DivergenceCapture(ar_ref)
    r0 = _TRUE_OBSERVE(base, base_len=0, committed=[9, 5], verify_logits=vl)

    os.environ[row_dump.ENV_DIR] = str(tmp_path / "d")
    os.environ[row_dump.ENV_INDICES] = ""  # nothing requested -> pure pass-through
    row_dump.install_from_env()

    calls = []
    true_orig = row_dump._ORIG_OBSERVE

    def spy(self, **kw):
        calls.append((self, kw))
        return true_orig(self, **kw)

    row_dump._ORIG_OBSERVE = spy

    inst = dec.DivergenceCapture(ar_ref)
    r1 = inst.observe(base_len=0, committed=[9, 5], verify_logits=vl)

    # identical args forwarded to the original
    assert len(calls) == 1
    self_seen, kw = calls[0]
    assert self_seen is inst
    assert kw["base_len"] == 0
    assert kw["committed"] == [9, 5]
    assert kw["verify_logits"] is vl               # exact object, not a copy
    assert kw["verify_logits_parts"] is None
    # identical return (both None)
    assert r1 == r0
    # original's own capture side effects preserved bit-for-bit
    assert inst.index == base.index == 1
    assert inst.ar_token == base.ar_token
    assert inst.dspark_token == base.dspark_token == 9
    np.testing.assert_array_equal(inst.dspark_logits_row, base.dspark_logits_row)


# ===========================================================================
# dumping: verify_logits form and verify_logits_parts (incl. second part)
# ===========================================================================
def test_dump_verify_logits_form(tmp_path):
    d = tmp_path / "ctrl"
    os.environ[row_dump.ENV_DIR] = str(d)
    os.environ[row_dump.ENV_INDICES] = "2-3"  # gpos 2 and 3 only
    st = row_dump.install_from_env()

    dec = row_dump._dec
    cap = dec.DivergenceCapture([5] * 10)
    vocab, ncommit = 4, 4
    arr = np.zeros((1, ncommit, vocab), dtype=np.float32)
    for m in range(ncommit):
        arr[0, m, :] = [m * 10 + v for v in range(vocab)]
    committed = [5, 5, 7, 5]  # mismatch at m=2 (gpos 3) sets the original's `found`

    cap.observe(base_len=0, committed=committed, verify_logits=mx.array(arr))

    assert (d / "row-2.npy").is_file() and (d / "row-3.npy").is_file()
    np.testing.assert_array_equal(np.load(d / "row-2.npy"), arr[0, 1, :])  # gpos2 -> m1
    np.testing.assert_array_equal(np.load(d / "row-3.npy"), arr[0, 2, :])  # gpos3 -> m2
    assert not (d / "row-1.npy").exists()  # gpos1 (m0) not requested
    assert not (d / "row-4.npy").exists()  # gpos4 (m3) not requested

    path = st.write_index()
    idx = json.loads(Path(path).read_text())
    assert idx["2"] == {"token": 5, "cycle_base_len": 0, "m": 1}
    assert idx["3"] == {"token": 7, "cycle_base_len": 0, "m": 2}


def test_dump_verify_logits_parts_second_part(tmp_path):
    d = tmp_path / "cand"
    os.environ[row_dump.ENV_DIR] = str(d)
    os.environ[row_dump.ENV_INDICES] = "5"  # gpos 5 -> m4 -> falls in the SECOND part
    row_dump.install_from_env()

    dec = row_dump._dec
    cap = dec.DivergenceCapture([5] * 20)
    vocab = 4
    p0 = np.zeros((1, 3, vocab), dtype=np.float32)
    p1 = np.zeros((1, 3, vocab), dtype=np.float32)
    for r in range(3):
        p0[0, r, :] = [100 + r * 10 + v for v in range(vocab)]
        p1[0, r, :] = [200 + r * 10 + v for v in range(vocab)]
    parts = [(0, mx.array(p0)), (3, mx.array(p1))]
    committed = [5, 5, 5, 5, 5, 5]  # len 6 -> gpos 1..6

    cap.observe(base_len=0, committed=committed, verify_logits_parts=parts)

    assert (d / "row-5.npy").is_file()
    # gpos5 = m4; part starts (0,3): 3 <= 4 < 6 -> p1[0, 4-3=1]
    np.testing.assert_array_equal(np.load(d / "row-5.npy"), p1[0, 1, :])
    assert not (d / "row-6.npy").exists()


def test_dump_works_after_capture_already_found(tmp_path):
    d = tmp_path / "aftr"
    os.environ[row_dump.ENV_DIR] = str(d)
    os.environ[row_dump.ENV_INDICES] = "7"
    row_dump.install_from_env()

    dec = row_dump._dec
    cap = dec.DivergenceCapture([5] * 20)
    vocab = 4

    # cycle 1: trigger the original's `found` at gpos 1 (token 9 != ar_ref[1]=5)
    a1 = np.zeros((1, 2, vocab), dtype=np.float32)
    a1[0, 0, :] = [1, 2, 3, 40]
    a1[0, 1, :] = [5, 6, 7, 8]
    cap.observe(base_len=0, committed=[9, 5], verify_logits=mx.array(a1))
    assert cap.found and cap.index == 1

    # cycle 2: base_len=2 -> gpos 3..8; request gpos 7 (m4). Original early-returns (found);
    # the wrapper must still dump the row.
    a2 = np.zeros((1, 6, vocab), dtype=np.float32)
    for m in range(6):
        a2[0, m, :] = [300 + m * 10 + v for v in range(vocab)]
    cap.observe(base_len=2, committed=[5, 5, 5, 5, 5, 5], verify_logits=mx.array(a2))

    assert (d / "row-7.npy").is_file()
    np.testing.assert_array_equal(np.load(d / "row-7.npy"), a2[0, 4, :])  # gpos7 = base2 + m4 + 1
    # the original's earlier capture is untouched by cycle 2
    assert cap.index == 1


def test_cycle_entirely_outside_band_dumps_nothing(tmp_path):
    d = tmp_path / "band"
    os.environ[row_dump.ENV_DIR] = str(d)
    os.environ[row_dump.ENV_INDICES] = "470-490"
    row_dump.install_from_env()

    dec = row_dump._dec
    cap = dec.DivergenceCapture([5] * 40)
    vocab = 4
    arr = np.zeros((1, 4, vocab), dtype=np.float32)
    # base_len 10 -> gpos 11..14, entirely below the requested band -> no rows
    cap.observe(base_len=10, committed=[5, 5, 5, 5], verify_logits=mx.array(arr))
    assert list(p for p in d.glob("row-*.npy")) == []


# ===========================================================================
# stager: round-trip, composition after F5/F6, double-apply refused
# ===========================================================================
_F15_BLOCK = (
    "                    growth_transition()\n"
    "                    import row_dump as _f15rd\n"
    "                    _f15rd.install_from_env()"
)
_ANCHOR = "                    growth_transition()"


def test_stager_roundtrip_on_real_runner():
    stage = _load_module(F15_STAGER, "f15_stager").stage
    src = Path(RUN_FULL).read_text()
    staged = stage(src)
    assert "_f15rd.install_from_env" in staged
    assert staged.count(_ANCHOR) == 1
    assert _F15_BLOCK in staged
    # reversing the single insert recovers the retained source byte-for-byte
    assert staged.replace(_F15_BLOCK, _ANCHOR) == src


def test_stager_composition_after_f5_and_f6():
    f5 = _load_module(F5_STAGER, "f5_stager").stage
    f6 = _load_module(F6_STAGER, "f6_stager").stage
    f15 = _load_module(F15_STAGER, "f15_stager_c").stage

    src = Path(RUN_FULL).read_text()
    f5f6 = f6(f5(src))                 # F5 then F6, as the F6 stager documents
    staged = f15(f5f6)                 # must not raise: growth_transition() anchor still unique

    for marker in ("_f5.enable_from_env", "_f6ep.install_from_env", "_f15rd.install_from_env"):
        assert marker in staged, marker
    # F15 added ONLY its own insert on top of the F5+F6 state (composition is clean)
    assert staged.replace(_F15_BLOCK, _ANCHOR) == f5f6


def test_stager_double_apply_refused():
    stage = _load_module(F15_STAGER, "f15_stager_d").stage
    src = Path(RUN_FULL).read_text()
    staged = stage(src)
    with pytest.raises(RuntimeError):
        stage(staged)


# ===========================================================================
# classify_pair: synthetic near-tie -> tie_flip, clear-margin -> not tie_flip
# ===========================================================================
def _make_arm(dir_path: Path, tokens, index: int, row):
    dir_path.mkdir(parents=True, exist_ok=True)
    np.save(dir_path / f"row-{index}.npy", np.asarray(row, dtype=np.float32))
    (dir_path / "rows.json").write_text(
        json.dumps({str(index): {"token": int(tokens[index]), "cycle_base_len": 0, "m": 0}})
    )


def _make_receipt(path: Path, tokens):
    path.write_text(json.dumps({"dspark": {"token_ids": list(tokens)}}))


def _run_classify(tmp_path, ctrl_tokens, cand_tokens, ctrl_row, cand_row, index=3):
    cd, nd = tmp_path / "ctrl", tmp_path / "cand"
    _make_arm(cd, ctrl_tokens, index, ctrl_row)
    _make_arm(nd, cand_tokens, index, cand_row)
    crec, nrec = tmp_path / "c.json", tmp_path / "n.json"
    _make_receipt(crec, ctrl_tokens)
    _make_receipt(nrec, cand_tokens)
    out = tmp_path / "res.json"
    rc = classify_pair.main([
        "--control-dir", str(cd), "--candidate-dir", str(nd),
        "--control-tokens", str(crec), "--candidate-tokens", str(nrec),
        "--runtime-path", PINNED_RUNTIME, "--out", str(out),
    ])
    assert rc == 0
    return json.loads(out.read_text())


def test_classify_pair_near_tie_is_tie_flip(tmp_path):
    # streams agree except at index 3: control emits token 0, candidate token 1.
    ctrl = [0, 0, 0, 0, 0]
    cand = [0, 0, 0, 1, 0]
    # control row argmax = token 0; candidate row argmax = token 1; contested margins and
    # cross-row deltas are all << tie_margin (0.03) -> a rounding-class near-tie.
    res = _run_classify(tmp_path, ctrl, cand,
                        ctrl_row=[0.020, 0.010, -1.0, -2.0],
                        cand_row=[0.012, 0.018, -1.0, -2.0])
    assert res["index"] == 3
    assert res["first_divergence_index"] == 3
    assert res["control_token"] == 0 and res["candidate_token"] == 1
    assert res["class"] == "tie_flip", res
    assert res["rows_consistent"] is True
    assert len(res["control_top5"]) == 4 and res["control_top5"][0][0] == 0
    assert res["candidate_top5"][0][0] == 1


def test_classify_pair_clear_margin_is_not_tie_flip(tmp_path):
    ctrl = [0, 0, 0, 0, 0]
    cand = [0, 0, 0, 1, 0]
    # a 10-logit gap at both contested tokens -> way past the bf16 rounding envelope.
    res = _run_classify(tmp_path, ctrl, cand,
                        ctrl_row=[10.0, 0.0, -1.0, -2.0],
                        cand_row=[0.0, 10.0, -1.0, -2.0])
    assert res["index"] == 3
    assert res["class"] == "divergent"
    assert res["class"] != "tie_flip"


def test_classify_pair_missing_row_errors_clearly(tmp_path):
    ctrl = [0, 0, 0, 0, 0]
    cand = [0, 0, 0, 1, 0]
    cd, nd = tmp_path / "ctrl", tmp_path / "cand"
    _make_arm(cd, ctrl, 3, [0.02, 0.01, -1.0, -2.0])
    nd.mkdir(parents=True, exist_ok=True)  # candidate dir has NO row-3.npy
    (nd / "rows.json").write_text(json.dumps({}))
    crec, nrec = tmp_path / "c.json", tmp_path / "n.json"
    _make_receipt(crec, ctrl)
    _make_receipt(nrec, cand)
    with pytest.raises(SystemExit):
        classify_pair.main([
            "--control-dir", str(cd), "--candidate-dir", str(nd),
            "--control-tokens", str(crec), "--candidate-tokens", str(nrec),
            "--runtime-path", PINNED_RUNTIME,
        ])


def test_classify_pair_reads_dspark_token_ids_not_ar_reference(tmp_path):
    # a receipt whose top-level token_ids DIFFERS from dspark.token_ids and whose
    # dspark.ar_reference.token_ids differs again: we must pick dspark.token_ids.
    ctrl = [0, 0, 0, 0, 0]
    cand = [0, 0, 0, 1, 0]
    cd, nd = tmp_path / "ctrl", tmp_path / "cand"
    _make_arm(cd, ctrl, 3, [0.020, 0.010, -1.0, -2.0])
    _make_arm(nd, cand, 3, [0.012, 0.018, -1.0, -2.0])
    crec = tmp_path / "c.json"
    nrec = tmp_path / "n.json"
    crec.write_text(json.dumps({
        "token_ids": [9, 9, 9, 9, 9],  # decoy top-level
        "dspark": {"token_ids": ctrl, "ar_reference": {"token_ids": [7, 7, 7, 7, 7]}},
    }))
    _make_receipt(nrec, cand)
    out = tmp_path / "r.json"
    classify_pair.main([
        "--control-dir", str(cd), "--candidate-dir", str(nd),
        "--control-tokens", str(crec), "--candidate-tokens", str(nrec),
        "--runtime-path", PINNED_RUNTIME, "--out", str(out),
    ])
    res = json.loads(out.read_text())
    assert res["control"]["tokens_source"] == "dspark.token_ids"
    assert res["control_token"] == 0  # from dspark.token_ids, not 9 (top) or 7 (ar_reference)


def test_local_rule_matches_pinned_ab_rule():
    fn, source = classify_pair._resolve_rule_fn(PINNED_RUNTIME)
    if source != "pinned":
        pytest.skip("pinned _dspark_divergence_rule not reusable here; local copy is the fallback")
    cases = [
        {"unavailable_logits": ["dspark"]},
        {"unavailable_logits": [], "rows_consistent": False},
        {"unavailable_logits": [], "rows_consistent": True, "deltas_within_tie_band": False},
        {"unavailable_logits": [], "rows_consistent": True, "deltas_within_tie_band": True,
         "tie_band_used": 0.03, "ar_contested_margin": 0.01, "dspark_contested_margin": 0.005,
         "rounding_class_by_delta": False},
        {"unavailable_logits": [], "rows_consistent": True, "deltas_within_tie_band": True,
         "tie_band_used": 0.03, "ar_contested_margin": 1.0, "dspark_contested_margin": 1.0,
         "rounding_class_by_delta": True},
        {"unavailable_logits": [], "rows_consistent": True, "deltas_within_tie_band": True,
         "tie_band_used": 0.03, "ar_contested_margin": 0.01, "dspark_contested_margin": 0.005,
         "rounding_class_by_delta": True},
        {"unavailable_logits": [], "rows_consistent": True, "deltas_within_tie_band": True,
         "tie_band_used": 0.03, "ar_contested_margin": 1.0, "dspark_contested_margin": 1.0,
         "rounding_class_by_delta": False},
        {"unavailable_logits": None, "rows_consistent": None, "deltas_within_tie_band": None},
    ]
    for d in cases:
        assert classify_pair._local_dspark_divergence_rule(d) == fn(d), d


def test_classify_pair_end_to_end_matches_direct_classify_divergence(tmp_path):
    # the tool's verdict equals classify_divergence called directly on the same rows.
    ctrl = [0, 0, 0, 0, 0]
    cand = [0, 0, 0, 1, 0]
    ctrl_row = [0.020, 0.010, -1.0, -2.0]
    cand_row = [0.012, 0.018, -1.0, -2.0]
    res = _run_classify(tmp_path, ctrl, cand, ctrl_row, cand_row)
    from mtplx.models.deepseek_v41_dspark_decode import classify_divergence
    direct = classify_divergence(
        index=3, ar_token=0, dspark_token=1,
        ar_logits_row=np.asarray(ctrl_row, dtype=np.float32),
        dspark_logits_row=np.asarray(cand_row, dtype=np.float32),
    )
    assert res["class"] == direct["class"]
    assert res["ar_contested_margin"] == direct["ar_contested_margin"]
    assert res["dspark_contested_margin"] == direct["dspark_contested_margin"]
