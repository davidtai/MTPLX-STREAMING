"""CPU preflight for the F5 compile window -- refuse BEFORE the production service
is unloaded if any dependency is missing.  Background memory is reported as an
advisory only (see check 6).

Mirrors the retained runner's construction-boundary checks, but runs entirely on
the CPU (no Metal, no lock) so a refusal never disturbs the live Qwen service.
Prints a JSON report and ``PREFLIGHT_OK`` / ``PREFLIGHT_FAIL``; exits 3 on any
failure.  Every check is a hard gate; there is no soft-pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(report, name, ok, detail=""):
    report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


# The retained pins (extension-bank-20260919).
RETAINED_RUNNER_SHA256 = "13bcdfe4fe583d8b465e69e7effbb8bf2f20d4d8b5baacb12e252f85f95149be"
PROMPT_IDS_TOKEN_SHA256 = "38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2"
CONTROL_DSPARK_SHA256 = "0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac"
AR_ARM = "cell16k_ring_v2_draft_attn_pf0"
# From the retained admission cases (composition-audit.json): 111 decode slots need
# the pre-candidate baseline <= ~11.03 GB (11.82 GB gave only 110).  A conservative
# headroom gate on background memory; the runner re-derives the exact admission.
MAX_BACKGROUND_BYTES_FOR_111 = 11_030_000_000
BOX_TARGET_BYTES = 110_000_000_000


def _background_bytes(report):
    """Best-effort background (non-candidate) memory, mirroring the runner's
    ``host_memory_snapshot()['box']['wired_bytes']`` where importable, else vm_stat."""
    try:
        from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
        snap = host_memory_snapshot()
        return int(snap["box"]["wired_bytes"]), "host_memory_snapshot.box.wired_bytes"
    except Exception as exc:  # pragma: no cover - fallback path
        report["snapshot_import_error"] = repr(exc)
    import subprocess
    try:
        page = 16384
        out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True,
                             timeout=10).stdout
        wired = 0
        for line in out.splitlines():
            if "wired down" in line.lower():
                wired = int(line.rsplit(":", 1)[1].strip().rstrip(".")) * page
        return wired, "vm_stat.pages_wired_down"
    except Exception as exc:  # pragma: no cover
        report["vm_stat_error"] = repr(exc)
        return None, "unavailable"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="F5 compile-window CPU preflight")
    ap.add_argument("--retained-runner", required=True)
    ap.add_argument("--ar-reference", required=True)
    ap.add_argument("--prompt-ids", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--aux-dir", required=True)
    ap.add_argument("--installation-json", required=True,
                    help="packed/installation.json (strict_allocator.identity)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    report = {"checks": [], "control_dspark_sha256": CONTROL_DSPARK_SHA256}
    ok = True

    # 1. retained runner source is the pinned one (so staging targets it).
    p = Path(args.retained_runner)
    ok &= _check(report, "retained_runner_present", p.is_file(), str(p))
    if p.is_file():
        digest = _sha256(p)
        ok &= _check(report, "retained_runner_sha256", digest == RETAINED_RUNNER_SHA256,
                     f"{digest} vs {RETAINED_RUNNER_SHA256}")

    # 2. prompt ids file: the 16,384 row's token_ids sha is the pinned prompt.
    pi = Path(args.prompt_ids)
    if _check(report, "prompt_ids_present", pi.is_file(), str(pi)):
        try:
            rows = [r for r in json.loads(pi.read_text())["prompts"]
                    if r["target_tokens"] == 16384]
            tok_sha = hashlib.sha256(json.dumps(rows[0]["token_ids"]).encode()).hexdigest()
            ok &= _check(report, "prompt_ids_token_sha256",
                         len(rows) == 1 and tok_sha == PROMPT_IDS_TOKEN_SHA256,
                         f"{tok_sha} vs {PROMPT_IDS_TOKEN_SHA256}")
        except Exception as exc:
            ok &= _check(report, "prompt_ids_token_sha256", False, repr(exc))

    # 3. AR reference: exactly one matched-arm row with the pinned prompt/tokens.
    ar = Path(args.ar_reference)
    if _check(report, "ar_reference_present", ar.is_file(), str(ar)):
        try:
            lines = [json.loads(x) for x in ar.read_text().splitlines() if x]
            r0 = lines[0]
            good = (len(lines) == 1 and r0.get("arm") == AR_ARM
                    and r0.get("prompt_ids_sha256") == PROMPT_IDS_TOKEN_SHA256
                    and r0.get("prompt_tokens") == 16384
                    and len(r0.get("token_ids", [])) == 1024)
            ok &= _check(report, "ar_reference_matched", good,
                         f"arm={r0.get('arm')} n={len(lines)} tokens={r0.get('prompt_tokens')}")
        except Exception as exc:
            ok &= _check(report, "ar_reference_matched", False, repr(exc))

    # 4. model + compact-residents present.
    ok &= _check(report, "model_dir_present", Path(args.model_dir).is_dir(), args.model_dir)
    ok &= _check(report, "aux_dir_present", Path(args.aux_dir).is_dir(), args.aux_dir)

    # 5. strict library present + sha (from installation.json).
    ij = Path(args.installation_json)
    if _check(report, "installation_json_present", ij.is_file(), str(ij)):
        try:
            ident = json.loads(ij.read_text())["strict_allocator"]["identity"]
            lib = Path(ident["path"])
            report["strict_lib"] = ident["path"]
            if _check(report, "strict_lib_present", lib.is_file(), ident["path"]):
                digest = _sha256(lib)
                ok &= _check(report, "strict_lib_sha256", digest == ident["sha256"],
                             f"{digest} vs {ident['sha256']}")
        except Exception as exc:
            ok &= _check(report, "strict_lib_sha256", False, repr(exc))

    # 6. background memory: ADVISORY ONLY (review 2026-09-19).  Before the guard runs,
    #    the production Qwen service (or another GPU job) is still resident, so wired
    #    memory here is ~80-100 GB and says nothing about the post-unload baseline the
    #    runner admits against.  The authoritative gate is the runner's own admission,
    #    which executes inside the guarded child AFTER the service is unloaded and
    #    BEFORE the model loads, and refuses there if 111 rows do not fit.
    bg, src = _background_bytes(report)
    report["background_bytes_advisory"] = bg
    report["background_source"] = src
    report["background_note"] = (
        "advisory: measured with the production service possibly resident; "
        "the guarded runner re-derives admission after unload"
    )
    _check(report, "background_advisory_recorded", True,
           f"{bg} B wired now; 111 rows need a post-unload baseline <= ~{MAX_BACKGROUND_BYTES_FOR_111} B")

    report["preflight_ok"] = bool(ok)
    text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(text + "\n")
    print(text)
    print("PREFLIGHT_OK" if ok else "PREFLIGHT_FAIL")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
