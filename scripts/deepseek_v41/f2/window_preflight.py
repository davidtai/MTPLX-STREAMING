"""CPU preflight for the F2 prefetch GPU window -- runs BEFORE the service is unloaded.

Resolves every runtime / reader / slot / ring / gate attribute the F2 lane touches on
the REAL shipped classes (hasattr / dataclass field checks), imports every module the
candidate needs with MLX pinned to CPU, and -- when paths are given -- verifies the
staged-tree and prompt-id dependencies exist. Refuses with a clear, specific message
if any name or dependency is missing, so the window fails ONCE here rather than after
the production model is unloaded (AGENTS.md: fail once, clearly, before measured
generation). MLX is imported but pinned to the CPU device; no Metal, no model load.

Callable as ``python -m f2.window_preflight [--dep PATH ...] [--sha PATH=HEX ...]``
or via ``main(...)``; the window script drives it before any GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path


class PreflightError(RuntimeError):
    pass


def _source_pin(run_worktree, compat_installation):
    """The check that kills a window AFTER unload if run from the wrong tree: the run
    worktree's git HEAD must equal the pinned source_commit, be clean, and hash every
    pinned runtime source identically. Run BEFORE the service is unloaded."""
    problems = []
    root = Path(run_worktree)
    compat = json.loads(Path(compat_installation).read_text())
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return [f"cannot read run-worktree HEAD ({root}): {exc}"]
    if head != compat["source_commit"]:
        problems.append(f"run worktree HEAD {head} != pinned {compat['source_commit']}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        problems.append("run worktree has tracked modifications:\n" + dirty)
    for rel, digest in compat.get("runtime_source_sha256", {}).items():
        p = root / rel
        if not p.exists():
            problems.append(f"pinned runtime source missing: {rel}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            problems.append(f"pinned runtime source differs: {rel}")
    return problems


def _verify_helpers(archived_dir, packed_installation, compat_installation):
    """Every archived helper (the staging INPUT) must be byte-identical to the pinned
    original per the packed/compat installation.json ``helper_sha256`` maps."""
    problems = []
    arch = Path(archived_dir)
    for sub, inst in (("packed", packed_installation), ("compat", compat_installation)):
        if inst is None:
            continue
        table = json.loads(Path(inst).read_text()).get("helper_sha256", {})
        for rel, digest in table.items():
            p = arch / sub / rel
            if not p.exists():
                problems.append(f"archived helper missing: {sub}/{rel}")
            elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                problems.append(f"archived helper differs from pin: {sub}/{rel}")
    return problems


def _compile_files(paths):
    problems = []
    for path in paths or ():
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"staged file does not compile: {path}: {exc}")
    return problems


# Resolve every F2b lane seam against the real classes (hasattr / dataclass fields).
def _seam_checks():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)

    from mtplx.expert_runtime import ExpertStreamingConfig
    from mtplx.expert_io import PositionalExpertReader
    from mtplx.expert_streaming import LayerExpertSlotBank
    from mtplx.expert_manifest import ExpertManifest, ExpertRecord
    from mtplx.models import deepseek_v41_moe as moe

    # F2b package modules (import proves the lane loads with its deps).
    import f2.host_ring as host_ring
    import f2.reader_intercept as reader_intercept
    import f2.predictor as predictor
    import f2.speculative as speculative
    import f2.install as f2b_install
    import f2.stage_f2_runner as stage

    # Only CLASS-resolvable seams are hasattr-checked here (methods, dataclass fields);
    # runtime/reader/pool INSTANCE attributes (runtime.reader/slots/manifest, reader.metrics,
    # pool._persistent) are resolved at install on the real runtime, and the retained reader
    # API is exercised below via the build_bind_reader derivation.
    checks: list[tuple[str, object, tuple[str, ...]]] = [
        ("PositionalExpertReader", PositionalExpertReader, (
            "_readv_range_into", "read_record_into", "read_component_records_into",
        )),
        ("LayerExpertSlotBank", LayerExpertSlotBank, ("resident_experts",)),
        ("ExpertManifest", ExpertManifest, ("record",)),
        ("deepseek_v41_moe", moe, ("_gate_prefix", "_gate_prefix_impl", "_attn_compile_gate", "Gate")),
        ("host_ring", host_ring, ("HostRing", "F2bCounters", "PLANE_OFFSETS")),
        ("reader_intercept", reader_intercept, (
            "derive_bind_reader_source", "build_bind_reader", "install_intercept",
            "RETAINED_PLANE_LANE_SHA256",
        )),
        ("predictor", predictor, ("GatePredictor", "rank_targets", "select_prefetch_sources")),
        ("speculative", speculative, ("SpeculativePool",)),
        ("install", f2b_install, ("install", "install_from_env", "dump_counters")),
        ("stage_f2_runner", stage, ("stage_admission", "stage_run_full")),
    ]
    missing = []
    for owner_name, owner, names in checks:
        for name in names:
            if not hasattr(owner, name):
                missing.append(f"{owner_name}.{name}")

    # ExpertStreamingRuntime.reader is an instance attr; class hasattr misses it -- resolve
    # via the constructor's assignment set is fragile, so trust the reader/slots class
    # attrs above and the config field here. F2b requires prefetch_slots (must be 0).
    if "prefetch_slots" not in set(getattr(ExpertStreamingConfig, "__dataclass_fields__", {})):
        missing.append("ExpertStreamingConfig.prefetch_slots")
    if "sidecar_offset" not in set(getattr(ExpertRecord, "__dataclass_fields__", {})):
        missing.append("ExpertRecord.sidecar_offset")
    if not callable(getattr(moe.Gate, "__call__", None)):
        missing.append("Gate.__call__")

    # The retained reader intercept: verify the pinned lane hasn't drifted and the
    # derivation round-trips, and the install-point module resolves -- when the packed
    # helpers are on PYTHONPATH (the window preflight puts the staged packed dir there).
    try:
        import plane_lane  # noqa: F401
    except Exception:
        pass
    else:
        try:
            reader_intercept.build_bind_reader(lambda offset, view: False)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"reader_intercept.build_bind_reader failed: {exc}")
        try:
            import projection_install
        except Exception as exc:  # noqa: BLE001
            missing.append(f"projection_install not importable: {exc}")
        else:
            for name in ("prime_model", "scheduled_run_source", "install_model"):
                if not hasattr(projection_install, name):
                    missing.append(f"projection_install.{name}")

    return missing


def _check_deps(deps, shas):
    missing = []
    for dep in deps or ():
        if not Path(dep).exists():
            missing.append(f"missing dependency: {dep}")
    for spec in shas or ():
        path, _, expected = spec.partition("=")
        p = Path(path)
        if not p.exists():
            missing.append(f"missing sha-pinned file: {path}")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if expected and got != expected:
            missing.append(f"sha256 mismatch for {path}: got {got}, want {expected}")
    return missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="F2 prefetch window CPU preflight")
    parser.add_argument("--dep", action="append", default=[], help="path that must exist")
    parser.add_argument("--sha", action="append", default=[], help="PATH=HEX sha256 pin")
    parser.add_argument("--run-worktree", default=None, help="detached run worktree at the pinned commit")
    parser.add_argument("--compat-installation", default=None, help="compat/installation.json (source pin)")
    parser.add_argument("--packed-installation", default=None, help="packed/installation.json (helper hashes)")
    parser.add_argument("--archived-dir", default=None, help="receipt-archived sources dir (staging input)")
    parser.add_argument("--compile", action="append", default=[], help="staged file to py_compile")
    parser.add_argument("--no-seams", action="store_true", help="skip the MLX-importing seam checks")
    args = parser.parse_args(argv)
    problems = [] if args.no_seams else _seam_checks()
    problems += _check_deps(args.dep, args.sha)
    if args.run_worktree and args.compat_installation:
        problems += _source_pin(args.run_worktree, args.compat_installation)
    if args.archived_dir:
        problems += _verify_helpers(args.archived_dir, args.packed_installation, args.compat_installation)
    problems += _compile_files(args.compile)
    if problems:
        for p in problems:
            print(f"[f2-preflight] FAIL: {p}", file=sys.stderr)
        print(f"[f2-preflight] {len(problems)} problem(s); refusing before the service is "
              "unloaded.", file=sys.stderr)
        return 1
    print("[f2-preflight] OK: every lane seam resolves on the real classes; deps present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
