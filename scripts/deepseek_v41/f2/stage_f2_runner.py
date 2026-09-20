"""Stage patched copies of the retained runner for the F2b prefetch window.

The window runs the EXACT retained runner from a fresh /private/tmp staging dir (never
Codex's pinned original). run_full imports helpers by module name (PYTHONPATH order
decides) while self-checking the pinned originals, so a staged copy runs while the hash
self-checks still pass. Every edit is anchored to a UNIQUE line and round-trip-checked
(the f5_compile/stage_f5_runner.py discipline). CPU-safe, no MLX.

Edits (all to STAGED copies):
  * ``stage_admission`` -- cap the decode capacity-search start (equal-capacity ladder,
    like F5). No ring charge: the F2b ring is HOST memory, not MLX, and is not admitted.
  * ``stage_run_full`` -- (1) install F2b as the LAST step of ``observe_seed_prefill``
    (after ``prime_model``, run_full.py:737), no-op unless MTPLX_DSV41_F2B=1; (2) surface
    the hidden transition/install error in ``observe_prefill_boundary``'s except clause,
    which otherwise raises a bare SystemExit with no traceback (run_full.py:783).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_CAP_ANCHOR = "    for capacity in range(112, old_capacity, -1):"

# Install F2b after prime_model, inside observe_seed_prefill (8-space indent).
_INSTALL_ANCHOR = "        projection_owner_report.update(prime_model(target))"
_INSTALL_INSERT = (
    "        import f2.install as _f2b_install  # F2b\n"
    "        _f2b_install.install_from_env(target)  # F2b (no-op unless MTPLX_DSV41_F2B=1)"
)

# Surface the hidden transition/install error (16-space indent) before the bare SystemExit.
_TRACE_ANCHOR = (
    "                raise SystemExit('critical prefill/cache transition failed; "
    "aborting generation') from error"
)
_TRACE_INSERT = "                import traceback; traceback.print_exc()  # F2b: surface the hidden error"


# F19: swap the reader's fanout pool right before plane_lane.install captures its submit (16-space indent).
_RO_ANCHOR = "                from plane_lane import install as install_plane_lane"
_RO_INSERT = ("                import f2.read_order as _f19_read_order; "
              "_f19_read_order.install_from_env(rt.reader)  # F19 (no-op unless MTPLX_DSV41_F19_FANOUT_WORKERS)")
# F21: same boundary -- wrap the live runtime's begin_split_route so the reader threads get the GIL at submit.
_SY_INSERT = ("                import f2.submit_yield as _f21_submit_yield; "
              "_f21_submit_yield.install_from_env(rt)  # F21 (no-op unless MTPLX_DSV41_F21_SUBMIT_YIELDS)")


def _assert_once(text: str, anchor: str, label: str) -> None:
    n = text.count(anchor)
    if n != 1:
        raise RuntimeError(f"{label} anchor is not unique ({n} occurrences)")


def stage_admission(source_text: str, *, max_rows: int) -> str:
    if not 85 <= int(max_rows) <= 112:
        raise RuntimeError("max_rows must be within 85..112")
    _assert_once(source_text, _CAP_ANCHOR, "capacity-search")
    new_cap = f"    for capacity in range({int(max_rows)}, old_capacity, -1):  # F2b equal-capacity cap"
    updated = source_text.replace(_CAP_ANCHOR, new_cap)
    if updated.replace(new_cap, _CAP_ANCHOR) != source_text:
        raise RuntimeError("F2b capacity cap changed more than the search start")
    return updated


# The F2b host ring is anonymous HOST memory the retained admission does not know about.
# Charge it to the physical (whole-machine) bound so a high live baseline lowers the admitted
# capacity instead of letting the measured machine peak cross the 110e9 ceiling (the guard
# killed an arm at a 13.32 GB baseline on 2026-09-19 before this edit existed).
_PHYS_ANCHOR = (
    "        physical = base + original['host_reserve_bytes'] + embedding_host + lookup_host"
    " + expansion_host + active + original['decode_cache_allowance_bytes']"
)


def stage_admission_ring(source_text: str, *, ring_bytes: int) -> str:
    ring_bytes = int(ring_bytes)
    if not 0 < ring_bytes <= 4 * 1024**3:
        raise RuntimeError("ring_bytes must be within (0, 4 GiB]")
    _assert_once(source_text, _PHYS_ANCHOR, "physical-bound")
    new_line = _PHYS_ANCHOR.replace(
        "physical = base + ", f"physical = base + {ring_bytes} + ", 1
    ) + "  # F2b host ring charged"
    updated = source_text.replace(_PHYS_ANCHOR, new_line)
    if updated.replace(new_line, _PHYS_ANCHOR) != source_text:
        raise RuntimeError("F2b ring charge changed more than the physical-bound line")
    return updated


# Row-split exactness probe: the retained hybrid install hardcodes ONE verify chunk of up to
# 8 rows. Staging a two-chunk schedule (e.g. 4+4) makes the target verify the same rows as
# two causal forwards (the second attends to the first's KV) with the unchanged greedy
# accept/commit path. Emitted tokens are identical IF AND ONLY IF the split-row arithmetic
# is bit-stable in practice, so the run's output digest decides whether a two-group verify
# pipeline would be an exact lever. (Sequential chunks are slower; this arm is not a
# throughput candidate.)
_VC_ANCHOR = "            '    verify_chunks = (8,)')"


def stage_verify_chunks(source_text: str, *, chunks) -> str:
    chunks = tuple(int(c) for c in chunks)
    if len(chunks) < 2 or any(c < 1 for c in chunks) or sum(chunks) != 8:
        raise RuntimeError("verify chunks must be >= 2 positive widths summing to 8")
    _assert_once(source_text, _VC_ANCHOR, "hybrid verify_chunks")
    new_line = _VC_ANCHOR.replace("(8,)", "(" + ", ".join(str(c) for c in chunks) + ")")
    updated = source_text.replace(_VC_ANCHOR, new_line)
    if updated.replace(new_line, _VC_ANCHOR) != source_text:
        raise RuntimeError("verify-chunks edit changed more than the one tuple")
    return updated


# Balanced row-split oracle: per cycle, verify ceil(n/2) rows then floor(n/2) rows (single chunk
# when n <= 4) -- the sequential reference for the F16 pipeline's "balanced" split. Two anchored
# insertions into the STAGED hybrid_install.py: one more symmetric replace() inside rewrite()
# (no mx call added, so its AST mx-call check still holds) and one namespace injection.
_VB_REWRITE_ANCHOR = "    restored = updated"
_VB_REWRITE_INSERT = (
    "    replace('        for configured_width in verify_chunks:',\n"
    "            '        for configured_width in _BALANCED_CHUNKS(len(block_ids)):')"
)
_VB_NS_ANCHOR = "    namespace['_LOOKUP_EXTENSION'] = lookup"
_VB_NS_INSERT = (
    "    namespace['_BALANCED_CHUNKS'] = (lambda n: (n,) if n <= 4 else ((n + 1) // 2, n // 2))"
)


def stage_verify_balanced(source_text: str) -> str:
    if "_BALANCED_CHUNKS" in source_text:
        raise RuntimeError("balanced verify schedule already staged")
    _assert_once(source_text, _VB_REWRITE_ANCHOR, "hybrid rewrite restore")
    _assert_once(source_text, _VB_NS_ANCHOR, "hybrid namespace")
    step = source_text.replace(_VB_REWRITE_ANCHOR, _VB_REWRITE_INSERT + "\n" + _VB_REWRITE_ANCHOR)
    if step.replace(_VB_REWRITE_INSERT + "\n" + _VB_REWRITE_ANCHOR, _VB_REWRITE_ANCHOR) != source_text:
        raise RuntimeError("balanced rewrite edit changed more than the one insertion")
    updated = step.replace(_VB_NS_ANCHOR, _VB_NS_ANCHOR + "\n" + _VB_NS_INSERT)
    if updated.replace(_VB_NS_ANCHOR + "\n" + _VB_NS_INSERT, _VB_NS_ANCHOR) != step:
        raise RuntimeError("balanced namespace edit changed more than the one insertion")
    return updated


# F25 confidence-gated draft length on the retained lane. The pinned decode loop already reads
# MTPLX_DSV41_DSPARK_CONF_THRESHOLD (deepseek_v41_dspark_decode.py: _effective_draft_len keeps the
# leading run of drafts whose sigmoid confidence >= threshold, at least one), but the retained hybrid
# install refuses a live confidence_threshold. Two anchored edits to the STAGED hybrid_install.py:
# drop exactly the one confidence refusal clause from install()'s condition, and record the live
# threshold in the report install() returns (anchored on the return-dict line) so every receipt
# carries the value that was live. Both anchors are disjoint from the verify_chunks anchors, so this
# composes with stage_verify_chunks / stage_verify_balanced in either order; it adds no mx call, so
# rewrite()'s AST mx-call check is untouched.
_CONF_REFUSAL_ANCHOR = (
    "    if (requested_depth != 5 or verify_chunks is not None or confidence_threshold is not None")
_CONF_REFUSAL_NEW = "    if (requested_depth != 5 or verify_chunks is not None"
_CONF_REPORT_ANCHOR = "    return {'native_head_depth':5,"
_CONF_REPORT_NEW = "    return {'native_head_depth':5,'confidence_threshold':confidence_threshold,"


def stage_confidence(source_text: str) -> str:
    if "'confidence_threshold':confidence_threshold," in source_text:
        raise RuntimeError("confidence threshold already staged")
    _assert_once(source_text, _CONF_REFUSAL_ANCHOR, "hybrid confidence refusal")
    _assert_once(source_text, _CONF_REPORT_ANCHOR, "hybrid report dict")
    step = source_text.replace(_CONF_REFUSAL_ANCHOR, _CONF_REFUSAL_NEW)
    if step.replace(_CONF_REFUSAL_NEW, _CONF_REFUSAL_ANCHOR) != source_text:
        raise RuntimeError("confidence refusal edit changed more than the one clause")
    updated = step.replace(_CONF_REPORT_ANCHOR, _CONF_REPORT_NEW)
    if updated.replace(_CONF_REPORT_NEW, _CONF_REPORT_ANCHOR) != step:
        raise RuntimeError("confidence report edit changed more than the one insertion")
    return updated


def stage_read_order(source_text: str) -> str:
    """packed_phase.py: install the F19 read-order pool immediately before the plane lane binds the reader."""
    if "_f19_read_order" in source_text:
        raise RuntimeError("F19 read order already staged")
    _assert_once(source_text, _RO_ANCHOR, "plane_lane install import")
    block = _RO_INSERT + "\n" + _SY_INSERT + "\n" + _RO_ANCHOR
    updated = source_text.replace(_RO_ANCHOR, block)
    if updated.replace(block, _RO_ANCHOR) != source_text:
        raise RuntimeError("F19/F21 I/O hook edit changed more than the two insertions")
    return updated


def stage_run_full(source_text: str, *, other_prompt: bool = False) -> str:
    _assert_once(source_text, _INSTALL_ANCHOR, "observe_seed_prefill prime_model")
    _assert_once(source_text, _TRACE_ANCHOR, "observe_prefill_boundary SystemExit")
    updated = source_text.replace(_INSTALL_ANCHOR, _INSTALL_ANCHOR + "\n" + _INSTALL_INSERT)
    if updated.replace(_INSTALL_ANCHOR + "\n" + _INSTALL_INSERT, _INSTALL_ANCHOR) != source_text:
        raise RuntimeError("F2b install edit changed more than the one insertion")
    step = updated
    updated = updated.replace(_TRACE_ANCHOR, _TRACE_INSERT + "\n" + _TRACE_ANCHOR)
    if updated.replace(_TRACE_INSERT + "\n" + _TRACE_ANCHOR, _TRACE_ANCHOR) != step:
        raise RuntimeError("F2b traceback edit changed more than the one insertion")
    # F23: with --other-prompt, ALSO env-pin the prompt/AR-reference commit and install
    # the generate mode. Absent the flag this returns the byte-identical base edits, so
    # the default (benchmark) staged tree is unchanged.
    if other_prompt:
        updated = stage_other_prompt(updated)
    return updated


# =========================================================================== F23
# Run the pinned runner on OTHER 16K prompts. All edits below are applied ONLY under
# --other-prompt (stage_run_full(other_prompt=True)); without the flag stage_run_full
# is byte-identical to today. Every edit is anchored to a UNIQUE line and round-trip
# checked, exactly like the edits above. CPU-safe, no MLX. The edits, in dependency
# order (P3's line lives inside G1's wrapped region, so P3 precedes G1):
#   P1/P2  prompt + AR-reference-commit pins read from the environment at start-up
#   P3     the reuse validator's prompt-ids digest becomes the env pin
#   P4     the fixture path + its digest assertion become the env pin
#   P5     the per-arm control-output gate becomes env-driven (no benchmark pin)
#   G3/G5  generate mode installs the real AR logits row + measures AR for real
#   G2/G4/G1  generate mode skips the reference read/validate, its arm_env comparison,
#             and the reuse post-processing (so the receipt carries real AR fields and
#             no ar_reference_reuse -> it satisfies the UNCHANGED reuse validator later)

# P1/P2: read the env pins once at construction (before any model load). The env-missing
# refusal is a clear RuntimeError. Replaces the hard-coded REFERENCE_SOURCE_COMMIT line.
_F23_REFCOMMIT_ANCHOR = "REFERENCE_SOURCE_COMMIT = 'e589c1e4b17856f506d90d9fb2bbb5ce45711648'"
_F23_ENV_MISSING_MSG = (
    "F23 other-prompt runner requires DSV41_STAGE_PROMPT_IDS_FILE and "
    "DSV41_STAGE_PROMPT_IDS_SHA256"
)
_F23_REFCOMMIT_INSERT = "\n".join((
    "REFERENCE_SOURCE_COMMIT = os.environ['DSV41_STAGE_AR_REFERENCE_COMMIT']  # F23: AR-reference provenance pin (env)",
    "DSV41_STAGE_AR_MODE = os.environ.get('DSV41_STAGE_AR_MODE', 'reuse')  # F23: reuse (default) | generate",
    "if DSV41_STAGE_AR_MODE not in ('reuse', 'generate'):",
    "    raise RuntimeError(\"DSV41_STAGE_AR_MODE must be 'reuse' or 'generate'\")",
    "_stage_prompt_ids_file = os.environ.get('DSV41_STAGE_PROMPT_IDS_FILE')",
    "_stage_prompt_ids_sha256 = os.environ.get('DSV41_STAGE_PROMPT_IDS_SHA256')",
    "if not _stage_prompt_ids_file or not _stage_prompt_ids_sha256:",
    "    raise RuntimeError('" + _F23_ENV_MISSING_MSG + "')",
    "STAGE_PROMPT_IDS_FILE = Path(_stage_prompt_ids_file)",
    "_stage_prompt_entries = [r for r in json.loads(STAGE_PROMPT_IDS_FILE.read_text())['prompts'] if r.get('target_tokens') == 16384]",
    "if len(_stage_prompt_entries) != 1:",
    "    raise RuntimeError('DSV41_STAGE_PROMPT_IDS_FILE must contain exactly one target_tokens==16384 prompt')",
    "STAGE_PROMPT_IDS_SHA256 = hashlib.sha256(json.dumps(_stage_prompt_entries[0]['token_ids']).encode()).hexdigest()",
    "if STAGE_PROMPT_IDS_SHA256 != _stage_prompt_ids_sha256:",
    "    raise RuntimeError('DSV41_STAGE_PROMPT_IDS_FILE token-id digest differs from DSV41_STAGE_PROMPT_IDS_SHA256')",
))

# P3: the reuse validator's benchmark prompt-ids digest -> the env pin.
_F23_PROMPT_DIGEST_ANCHOR = (
    "    or reference.get('prompt_ids_sha256') != "
    "'38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2'"
)
_F23_PROMPT_DIGEST_NEW = "    or reference.get('prompt_ids_sha256') != STAGE_PROMPT_IDS_SHA256"

# P4: the fixture path + its own digest assertion -> the env pin.
_F23_FIXTURE_ANCHOR = (
    "    fixture = Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json')"
)
_F23_FIXTURE_NEW = "    fixture = STAGE_PROMPT_IDS_FILE  # F23: env-pinned prompt ids file"
_F23_FIXTURE_DIGEST_ANCHOR = (
    "    assert hashlib.sha256(json.dumps(rows[0]['token_ids']).encode()).hexdigest() == "
    "'38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2'"
)
_F23_FIXTURE_DIGEST_NEW = (
    "    assert hashlib.sha256(json.dumps(rows[0]['token_ids']).encode()).hexdigest() == "
    "STAGE_PROMPT_IDS_SHA256"
)

# P5: the per-arm control-output gate. Self-compares (no-op) unless the launcher pins an
# expected DSpark digest for this window; the benchmark's CONTROL_OUTPUT_SHA256 cannot
# hold for a different prompt. The cross-arm digest gate lives in the launcher readout.
_F23_CONTROL_GATE_ANCHOR = (
    "        if receipt['dspark']['token_ids_sha256'] != CONTROL_OUTPUT_SHA256:"
)
_F23_CONTROL_GATE_NEW = (
    "        if receipt['dspark']['token_ids_sha256'] != "
    "os.environ.get('DSV41_STAGE_EXPECT_DSPARK_SHA', receipt['dspark']['token_ids_sha256']):"
)

# G3: in generate mode use the real AR logits row (the cached one needs a prior reference).
_F23_ARLOGITS_ANCHOR = "    ab._ar_logits_row_at_index = cached_ar_logits_row"
_F23_ARLOGITS_NEW = (
    "    ab._ar_logits_row_at_index = original_ar_logits_row "
    "if DSV41_STAGE_AR_MODE == 'generate' else cached_ar_logits_row  # F23"
)

# G5: in generate mode measure the AR pass for real (admitted_ar) instead of replaying.
_F23_GENERATE_INSTALL_ANCHOR = "    ab._generate = reused_ar_reference"
_F23_GENERATE_INSTALL_NEW = (
    "    ab._generate = admitted_ar if DSV41_STAGE_AR_MODE == 'generate' "
    "else reused_ar_reference  # F23"
)

# G1: the reference read/validate/provenance block (skipped in generate mode).
_F23_REFBLOCK_START = "reference_path = Path(os.environ['DSV41_STAGE_AR_REFERENCE']).resolve(strict=True)"
_F23_REFBLOCK_END = "bounds['ar_reference_reuse'] = reference_provenance"
_F23_REFBLOCK_GUARD = "if DSV41_STAGE_AR_MODE != 'generate':"
_F23_REFBLOCK_ELSE = [
    "else:",
    "    # F23 generate mode: no prior AR reference; the AR pass is measured for real",
    "    # (ab._generate = admitted_ar) and THIS receipt becomes the reference.",
    "    reference_path = None",
    "    reference = {}",
    "    reference_bounds = {}",
    "    reference_ids = []",
    "    reference_digest = None",
    "    reference_provenance = None",
]

# G2: the reference arm_env comparison (skipped in generate mode).
_F23_ARMENV_START = (
    "    if {k: v for k, v in reference.get('arm_env', {}).items() "
    "if k not in _bound_prefill_flags} != {"
)
_F23_ARMENV_END = "        raise RuntimeError('target environment differs outside explicit bound flags')"
_F23_ARMENV_GUARD = "    if DSV41_STAGE_AR_MODE != 'generate':"
_F23_ARMENV_ELSE: list[str] = []

# G4: the reuse post-processing (ar_reference_reuse + AR nulling), skipped in generate mode.
_F23_REUSE_START = "        receipt['ar_reference_reuse'] = reference_provenance"
_F23_REUSE_END = "        receipt['dspark']['ar_reference_reuse'] = reference_provenance"
_F23_REUSE_GUARD = "        if DSV41_STAGE_AR_MODE != 'generate':"
_F23_REUSE_ELSE = [
    "        else:",
    "            receipt['measurement_origin'] = {'ar': 'current_run', 'dspark': 'current_run'}  # F23 generate",
]

_F23_STAGED_MARKER = "DSV41_STAGE_AR_MODE"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    """Anchored single-occurrence replace with a byte-for-byte round-trip check."""
    _assert_once(text, old, label)
    updated = text.replace(old, new)
    if updated.replace(new, old) != text:
        raise RuntimeError(f"{label}: edit changed more than the one anchor")
    return updated


def _wrap_region(text: str, start: str, end: str, guard: str,
                 else_lines: list, label: str) -> str:
    """Wrap the inclusive line region [start..end] in ``guard`` (indent it 4 spaces) and
    append ``else_lines`` verbatim. Both anchors must be unique whole lines. Round-trips
    by dedenting the wrapped region and dropping the inserted wrapper -> original text."""
    lines = text.split("\n")
    starts = [i for i, l in enumerate(lines) if l == start]
    ends = [i for i, l in enumerate(lines) if l == end]
    if len(starts) != 1 or len(ends) != 1:
        raise RuntimeError(f"{label}: anchors not unique (start={len(starts)} end={len(ends)})")
    si, ei = starts[0], ends[0]
    if ei < si:
        raise RuntimeError(f"{label}: end anchor precedes start anchor")
    region = lines[si:ei + 1]
    indented = ["    " + l if l.strip() else l for l in region]
    new_block = [guard] + indented + list(else_lines)
    out_lines = lines[:si] + new_block + lines[ei + 1:]
    updated = "\n".join(out_lines)
    # round-trip: dedent the wrapped region and remove the wrapper, expect the original.
    chk = updated.split("\n")
    if chk[si] != guard:
        raise RuntimeError(f"{label}: guard not placed as expected")
    recovered = chk[si + 1:si + 1 + len(region)]
    dedented = [l[4:] if l.strip() else l for l in recovered]
    rebuilt = "\n".join(chk[:si] + dedented + chk[si + 1 + len(region) + len(else_lines):])
    if rebuilt != text:
        raise RuntimeError(f"{label}: round-trip failed")
    return updated


def stage_other_prompt(source_text: str) -> str:
    """Apply the F23 env-pin + generate-mode edits to a STAGED run_full.py copy."""
    if _F23_STAGED_MARKER in source_text:
        raise RuntimeError("F23 other-prompt edits already staged")
    out = _replace_once(source_text, _F23_REFCOMMIT_ANCHOR, _F23_REFCOMMIT_INSERT,
                        "F23 env pins")
    out = _replace_once(out, _F23_PROMPT_DIGEST_ANCHOR, _F23_PROMPT_DIGEST_NEW,
                        "F23 reuse-validator prompt digest")
    out = _replace_once(out, _F23_FIXTURE_ANCHOR, _F23_FIXTURE_NEW, "F23 fixture path")
    out = _replace_once(out, _F23_FIXTURE_DIGEST_ANCHOR, _F23_FIXTURE_DIGEST_NEW,
                        "F23 fixture digest")
    out = _replace_once(out, _F23_CONTROL_GATE_ANCHOR, _F23_CONTROL_GATE_NEW,
                        "F23 control-output gate")
    out = _replace_once(out, _F23_ARLOGITS_ANCHOR, _F23_ARLOGITS_NEW, "F23 AR-logits mode")
    out = _replace_once(out, _F23_GENERATE_INSTALL_ANCHOR, _F23_GENERATE_INSTALL_NEW,
                        "F23 generate install")
    out = _wrap_region(out, _F23_ARMENV_START, _F23_ARMENV_END, _F23_ARMENV_GUARD,
                       _F23_ARMENV_ELSE, "F23 skip arm_env in generate")
    out = _wrap_region(out, _F23_REUSE_START, _F23_REUSE_END, _F23_REUSE_GUARD,
                       _F23_REUSE_ELSE, "F23 skip reuse post-processing in generate")
    # G1 last: P3 already edited the digest line inside this region.
    out = _wrap_region(out, _F23_REFBLOCK_START, _F23_REFBLOCK_END, _F23_REFBLOCK_GUARD,
                       _F23_REFBLOCK_ELSE, "F23 guard reference block in generate")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the F2b-patched runner in place")
    ap.add_argument("--admission", default=None, help="STAGED packed_admission.py to cap in place")
    ap.add_argument("--max-rows", type=int, default=None, help="cap the decode capacity-search start")
    ap.add_argument("--run-full", default=None, help="STAGED run_full.py to patch in place (install + traceback)")
    ap.add_argument("--other-prompt", action="store_true",
                    help="with --run-full: ALSO apply the F23 env-pin + generate-mode edits")
    ap.add_argument("--hybrid-install", default=None, help="STAGED hybrid_install.py (with --verify-chunks and/or --confidence)")
    ap.add_argument("--verify-chunks", default=None, help="e.g. 4,4 : row-split exactness probe")
    ap.add_argument("--confidence", action="store_true",
                    help="F25: with --hybrid-install, accept a live confidence_threshold and record it in the report")
    ap.add_argument("--packed-phase", default=None, help="STAGED packed_phase.py: F19 read-order pool hook")
    ap.add_argument("--ring-bytes", type=int, default=None,
                    help="with --admission: charge the F2b host ring to the physical bound")
    args = ap.parse_args(argv)
    if (args.admission is None) != (args.max_rows is None):
        raise SystemExit("--admission and --max-rows go together")
    if args.admission is not None:
        p = Path(args.admission)
        out = stage_admission(p.read_text(), max_rows=args.max_rows)
        if args.ring_bytes is not None:
            out = stage_admission_ring(out, ring_bytes=args.ring_bytes)
            print("staged_admission_ring_bytes", args.ring_bytes)
        p.write_text(out)
        print("staged_admission", "max_rows", args.max_rows, "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    if (args.hybrid_install is None) != (args.verify_chunks is None and not args.confidence):
        raise SystemExit("--hybrid-install goes with --verify-chunks and/or --confidence")
    if args.hybrid_install is not None:
        p = Path(args.hybrid_install)
        out = p.read_text()
        if args.verify_chunks is not None:
            if args.verify_chunks == "balanced":
                out = stage_verify_balanced(out)
            else:
                out = stage_verify_chunks(out, chunks=[int(x) for x in args.verify_chunks.split(",")])
            print("staged_verify_chunks", args.verify_chunks, "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
        if args.confidence:
            out = stage_confidence(out)
            print("staged_confidence", "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
        p.write_text(out)
    if args.packed_phase is not None:
        p = Path(args.packed_phase)
        out = stage_read_order(p.read_text())
        p.write_text(out)
        print("staged_read_order", "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    if args.run_full is not None:
        p = Path(args.run_full)
        out = stage_run_full(p.read_text(), other_prompt=args.other_prompt)
        p.write_text(out)
        print("staged_run_full", "other_prompt", args.other_prompt,
              "sha", hashlib.sha256(out.encode()).hexdigest()[:16])
    elif args.other_prompt:
        raise SystemExit("--other-prompt requires --run-full")
    return 0


if __name__ == "__main__":
    sys.exit(main())
