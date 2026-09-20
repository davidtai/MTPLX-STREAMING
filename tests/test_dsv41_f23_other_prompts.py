"""CPU tests for the F23 other-prompt harness (no MLX, no GPU).

Covers the builder (exact 16,384 ids, schema, determinism, tail/BOS, printed digest ==
file digest), the stager's new --other-prompt edits (round-trip on the archived
run_full.py, unique anchors, double-apply refused, compiles, env-missing refusal text,
default path byte-identical to today) and reference compatibility (a synthetic
generate-mode receipt passes the reuse validator extracted from the STAGED source; a
receipt carrying ar_reference_reuse is rejected). Run under nice -n 19, no -n auto,
explicit file path.
"""
from __future__ import annotations

import hashlib
import json
import py_compile
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import f2.stage_f2_runner as st  # noqa: E402

_ARCH = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
_RUN_FULL = _ARCH / "run_full.py"

# Pinned sha of TODAY's base-only staged run_full.py (stage_run_full without other_prompt);
# the F23 changes must not move it (default path byte-identical).
_TODAY_BASE_STAGED_SHA = "bb2efd675381426c9425cb9b5d8852da8235c5949edf7bf56502135aee4432c6"
# The pinned run worktree HEAD, used as the AR-reference commit for generate-mode receipts.
_RUN_WORKTREE_COMMIT = "d5f15e7a02d225115debf6ce317e26dc4a935b60"


# --------------------------------------------------------------------------- stager
def test_default_path_unchanged_without_other_prompt():
    out = st.stage_run_full(_RUN_FULL.read_text())
    assert hashlib.sha256(out.encode()).hexdigest() == _TODAY_BASE_STAGED_SHA
    # none of the F23 markers leak into the default (benchmark) staged tree.
    assert "DSV41_STAGE_AR_MODE" not in out
    assert "STAGE_PROMPT_IDS_SHA256" not in out


def test_other_prompt_anchors_unique_in_archived():
    src = _RUN_FULL.read_text()
    for anchor in (
        st._F23_REFCOMMIT_ANCHOR, st._F23_PROMPT_DIGEST_ANCHOR, st._F23_FIXTURE_ANCHOR,
        st._F23_FIXTURE_DIGEST_ANCHOR, st._F23_CONTROL_GATE_ANCHOR, st._F23_ARLOGITS_ANCHOR,
        st._F23_GENERATE_INSTALL_ANCHOR, st._F23_REFBLOCK_START, st._F23_REFBLOCK_END,
        st._F23_ARMENV_START, st._F23_ARMENV_END, st._F23_REUSE_START, st._F23_REUSE_END,
    ):
        assert src.count(anchor) == 1, (anchor, src.count(anchor))


@pytest.mark.parametrize("old,new", [
    ("_F23_REFCOMMIT_ANCHOR", "_F23_REFCOMMIT_INSERT"),
    ("_F23_PROMPT_DIGEST_ANCHOR", "_F23_PROMPT_DIGEST_NEW"),
    ("_F23_FIXTURE_ANCHOR", "_F23_FIXTURE_NEW"),
    ("_F23_FIXTURE_DIGEST_ANCHOR", "_F23_FIXTURE_DIGEST_NEW"),
    ("_F23_CONTROL_GATE_ANCHOR", "_F23_CONTROL_GATE_NEW"),
    ("_F23_ARLOGITS_ANCHOR", "_F23_ARLOGITS_NEW"),
    ("_F23_GENERATE_INSTALL_ANCHOR", "_F23_GENERATE_INSTALL_NEW"),
])
def test_single_line_edits_roundtrip_on_archived(old, new):
    src = _RUN_FULL.read_text()
    a, n = getattr(st, old), getattr(st, new)
    out = st._replace_once(src, a, n, old)
    assert out != src and out.replace(n, a) == src  # reverses byte-for-byte


@pytest.mark.parametrize("start,end,guard,else_body", [
    ("_F23_ARMENV_START", "_F23_ARMENV_END", "_F23_ARMENV_GUARD", "_F23_ARMENV_ELSE"),
    ("_F23_REUSE_START", "_F23_REUSE_END", "_F23_REUSE_GUARD", "_F23_REUSE_ELSE"),
    ("_F23_REFBLOCK_START", "_F23_REFBLOCK_END", "_F23_REFBLOCK_GUARD", "_F23_REFBLOCK_ELSE"),
])
def test_wrap_regions_roundtrip_on_archived(start, end, guard, else_body):
    src = _RUN_FULL.read_text()
    # _wrap_region raises if its internal dedent+rebuild round-trip fails.
    out = st._wrap_region(src, getattr(st, start), getattr(st, end), getattr(st, guard),
                          getattr(st, else_body), start)
    assert out != src and getattr(st, guard) in out


def test_other_prompt_compiles_and_carries_markers(tmp_path):
    out = st.stage_run_full(_RUN_FULL.read_text(), other_prompt=True)
    assert out != st.stage_run_full(_RUN_FULL.read_text())  # differs from base
    # env pins read at construction; generate lane installed; benchmark pins gone.
    assert st._F23_ENV_MISSING_MSG in out
    assert "REFERENCE_SOURCE_COMMIT = os.environ['DSV41_STAGE_AR_REFERENCE_COMMIT']" in out
    assert "admitted_ar if DSV41_STAGE_AR_MODE == 'generate'" in out
    assert "original_ar_logits_row if DSV41_STAGE_AR_MODE == 'generate'" in out
    assert "DSV41_STAGE_EXPECT_DSPARK_SHA" in out
    assert out.count("38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2") == 0
    p = tmp_path / "run_full.py"
    p.write_text(out)
    py_compile.compile(str(p), doraise=True)


def test_other_prompt_double_apply_refused():
    out = st.stage_run_full(_RUN_FULL.read_text(), other_prompt=True)
    with pytest.raises(RuntimeError):
        st.stage_other_prompt(out)


# ------------------------------------------------------- reference compatibility (3.2)
def _extract_reuse_validator(staged_source: str) -> str:
    """Slice the reuse-validator predicate (assignments + the `if (...) : raise`) out of
    the STAGED source and dedent it to top level, so a test can exec it directly."""
    lines = staged_source.split("\n")
    start = next(i for i, l in enumerate(lines)
                 if l.strip() == "reference_ids = reference.get('token_ids', [])")
    end = next(i for i, l in enumerate(lines)
               if l.strip() == "raise RuntimeError('AR reference is not a complete matched native-target run')")
    return textwrap.dedent("\n".join(lines[start:end + 1]))


_PROMPT_SHA = hashlib.sha256(b"f23-other-prompt-ids").hexdigest()
_DECODE_STEPS = 1023


def _good_generate_receipt():
    ids = list(range(1, _DECODE_STEPS + 2))  # 1024 valid ints
    return (
        {
            "arm": "cell16k_ring_v2_draft_attn_pf0",
            "prompt_ids_sha256": _PROMPT_SHA,
            "prompt_tokens": 16384,
            "token_ids": ids,
            "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "ttft_s": 141.8,
            "decode_wall_s": 149.1,
            "resident_load_report": {"head_mode": "bf16"},
            # NOTE: no ar_reference_reuse -> a real generate-mode receipt.
        },
        {
            "source_commit": _RUN_WORKTREE_COMMIT,
            "decode_steps": _DECODE_STEPS,
            "selected_mtp_experts_by_stage": [93, 58, 32],
        },
    )


def _run_validator(reference: dict, reference_bounds: dict) -> None:
    """Exec the extracted validator; raises RuntimeError iff the receipt is not a valid
    AR reference (the same predicate the retained runner applies at start-up)."""
    block = _extract_reuse_validator(st.stage_run_full(_RUN_FULL.read_text(), other_prompt=True))
    ns = {
        "reference": reference, "reference_bounds": reference_bounds,
        "REFERENCE_SOURCE_COMMIT": _RUN_WORKTREE_COMMIT,
        "STAGE_PROMPT_IDS_SHA256": _PROMPT_SHA,
        "decode_steps": _DECODE_STEPS, "hashlib": hashlib, "json": json,
    }
    exec(block, ns)


def test_generate_receipt_passes_the_unchanged_reuse_validator():
    ref, bounds = _good_generate_receipt()
    _run_validator(ref, bounds)  # must NOT raise


def test_reuse_receipt_is_rejected():
    ref, bounds = _good_generate_receipt()
    ref["ar_reference_reuse"] = {"receipt": "prior"}
    with pytest.raises(RuntimeError):
        _run_validator(ref, bounds)


def test_wrong_prompt_digest_is_rejected():
    # proves the env prompt pin is wired into the retained validator.
    ref, bounds = _good_generate_receipt()
    ref["prompt_ids_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        _run_validator(ref, bounds)


def test_nulled_ar_timings_are_rejected():
    # a reuse-mode receipt nulls decode_wall_s/ttft_s; such a receipt cannot be reused.
    ref, bounds = _good_generate_receipt()
    ref["decode_wall_s"] = 0
    with pytest.raises(RuntimeError):
        _run_validator(ref, bounds)


# ------------------------------------------------------------------------- builder
def _build_module():
    import importlib
    return importlib.import_module("f2.build_other_prompts")


def _assert_prompt_file(data: dict, printed_digests: dict):
    assert data["schema"] == "mtplx-server-cell-prompt-ids-v1"
    assert data["model_family"] == "deepseek-v41"
    for key in ("model", "context_sha256", "instruction", "template_settings",
                "tokenizer_sha256", "template_sha256", "prompts"):
        assert key in data
    assert data["template_settings"] == {
        "add_generation_prompt": True, "enable_thinking": False,
        "add_special_tokens": False, "bos_token_id": 0,
    }
    entry = next(e for e in data["prompts"] if e["target_tokens"] == 16384)
    ids = entry["token_ids"]
    assert len(ids) == 16384                    # EXACTLY 16,384 ids
    assert ids[0] == 0                          # BOS
    assert entry["bos_id_prepended"] is True
    # printed digest == digest recorded in the file == digest recomputed from the ids
    file_sha = entry["token_ids_sha256"]
    assert file_sha == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    assert printed_digests[16384] == file_sha


def test_builder_io_module_exact_deterministic(tmp_path):
    bmod = _build_module()
    r1 = bmod.build_one("io_module", out_root=tmp_path / "a")
    r2 = bmod.build_one("io_module", out_root=tmp_path / "b")
    f1 = (tmp_path / "a/io_module/python-prompt-ids.json").read_bytes()
    f2 = (tmp_path / "b/io_module/python-prompt-ids.json").read_bytes()
    assert f1 == f2                             # determinism: byte-identical
    data = json.loads(f1)
    _assert_prompt_file(data, r1["_digests"])
    # io_module is a unified-diff request; the instruction text must be present.
    text = (tmp_path / "a/io_module/python-16384.txt").read_text()
    assert "Return only a unified diff" in text
