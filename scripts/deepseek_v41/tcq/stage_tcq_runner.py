"""F39 stager: apply the tcq3 lane's construction-time edits to the pinned runner sources.

Every edit is an anchored, single-occurrence replacement that (a) round-trips byte-for-byte (reversing it recovers
the original source) and (b) is env-gated so the STOCK mxfp4 path is byte-identical when ``MTPLX_DSV41_TCQ3`` is
unset — a construction-time route, not a runtime fallback (AGENTS.md "correct by design").

packed_phase.py (2 edits):
  * plane-lane bind site -> route_plane_lane (tcq3 XOR mxfp4 decode-verify lane);
  * install_growth codec gate -> also accept the tcq3 record (13,290,496 B) / codec, when the lane is armed.
run_full.py (3 edits):
  * checked_load -> install the tcq3 loader monkeypatch before the model loads (when armed);
  * checked_load -> stamp the loaded spec to the tcq3 codec (when armed);
  * growth_admission -> retarget the ladder to the tcq3 slot (when armed), so the extra rows are admitted.

Usage (CPU, no GPU):
    python stage_tcq_runner.py --packed-phase <packed_phase.py> [--run-full <run_full.py>] --out-packed-phase <..> [--out-run-full <..>]
(``--out-*`` may equal the input for in-place; if omitted, edits are applied in place.)
"""
from __future__ import annotations

import argparse
import ast

# --------------------------------------------------------------------- packed_phase.py: plane-lane decode route
_PL_ANCHOR = (
    "                from plane_lane import install as install_plane_lane\n"
    "                plane_runners.update(install_plane_lane(rt, dict(zip(layers, switches)), owners))"
)
_PL_ROUTED = (
    "                from plane_lane import install as install_plane_lane\n"
    "                import tcq.install as _tcq_route\n"
    "                plane_runners.update(_tcq_route.route_plane_lane(rt, layers, switches, owners, install_plane_lane))"
)

# --------------------------------------------------------------------- packed_phase.py: install_growth codec gate
_GG_ANCHOR = "        or rt.spec.expert_record_bytes != 18800640 or rt.spec.expert_codec != 'mxfp4'"
_GG_ROUTED = (
    "        or ((rt.spec.expert_record_bytes != 18800640 or rt.spec.expert_codec != 'mxfp4')\n"
    "            and not (os.environ.get('MTPLX_DSV41_TCQ3') == '1' and rt.spec.expert_codec == 'tcq3'\n"
    "                     and rt.spec.expert_record_bytes == 13290496))"
)

# --------------------------------------------------------------------- run_full.py: loader install (checked_load)
_LOAD_ANCHOR = "        resident = original_load(*a, **kw)"
_LOAD_ROUTED = (
    "        if os.environ.get('MTPLX_DSV41_TCQ3') == '1':\n"
    "            import tcq.loader_install as _tcq_loader\n"
    "            _tcq_loader.install_tcq_loader()\n"
    "        resident = original_load(*a, **kw)"
)

# --------------------------------------------------------------------- run_full.py: spec stamp (checked_load)
_STAMP_ANCHOR = "        rt = resident.model._mtplx_expert_runtime"
_STAMP_ROUTED = (
    "        rt = resident.model._mtplx_expert_runtime\n"
    "        if os.environ.get('MTPLX_DSV41_TCQ3') == '1':\n"
    "            import tcq.loader_install as _tcq_loader\n"
    "            _tcq_loader.stamp_spec_tcq3(rt.spec)"
)

# --------------------------------------------------------------------- run_full.py: admission retarget
_ADM_ANCHOR = (
    "growth_admission = resolve_admission(base, host_memory_snapshot()['box']['wired_bytes'],\n"
    "    grow=GROWTH_ENABLED, expected_receipt_hash=COMPATIBILITY['phase_memory_control_sha256'], strict_allocator=STRICT_ALLOCATOR)"
)
_ADM_ROUTED = (
    _ADM_ANCHOR + "\n"
    "if os.environ.get('MTPLX_DSV41_TCQ3') == '1':\n"
    "    import tcq.tcq_admission as _tcq_adm\n"
    "    growth_admission = _tcq_adm.retarget(growth_admission, base=base, wired=host_memory_snapshot()['box']['wired_bytes'])"
)

# --------------------------------------------------------------------- run_full.py: bounded-workload model-dir gate (Task 8)
# The bounded-MTP-workload check (run_full.py ~:515) hardcodes the mxfp4 model dir; the +tcq arm passes --model
# <tcq3 dir>, so the check fires and raises 'arguments differ from the bounded MTP workload'.  Env-gated: when the
# tcq3 lane is armed the admitted model IS the tcq3 artifact; otherwise the mxfp4 dir (mxfp4 arms byte-identical).
_MODEL_ANCHOR = (
    "    if (Path(args.model).resolve() != "
    "Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')"
)
_MODEL_ROUTED = (
    "    if (Path(args.model).resolve() != "
    "(Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-tcq3') "
    "if os.environ.get('MTPLX_DSV41_TCQ3') == '1' "
    "else Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4'))"
)

# --------------------------------------------------------------------- run_full.py: growth-transition route (Task 3)
_GROWTH_ANCHOR = (
    "            growth_transition, growth_report = install_growth(resident.model, "
    "initial_admission['decode_slots_per_layer'], mx=mx, admission=initial_admission)"
)
_GROWTH_ROUTED = (
    "            if os.environ.get('MTPLX_DSV41_TCQ3') == '1':\n"
    "                import tcq.growth as _tcq_growth\n"
    "                growth_transition, growth_report = _tcq_growth.install_growth_tcq3(resident.model, "
    "growth_admission['decode_slots_per_layer'], mx=mx, admission=growth_admission)\n"
    "            else:\n"
    "                growth_transition, growth_report = install_growth(resident.model, "
    "initial_admission['decode_slots_per_layer'], mx=mx, admission=initial_admission)"
)

# --------------------------------------------------------------------- run_full.py: grow_rows no-op for tcq3 (Task 3)
_GROWROWS_ANCHOR = (
    "        overflow_report = grow_rows(runtime, capacity=growth_admission['decode_slots_per_layer'], "
    "layout='extension', mx=mx)"
)
_GROWROWS_ROUTED = (
    "        if os.environ.get('MTPLX_DSV41_TCQ3') == '1':\n"
    "            overflow_report = {'tcq3': 'grown to decode capacity in install_growth_tcq3', "
    "'slots_per_layer': runtime.plan.slots_per_layer}\n"
    "        else:\n"
    "            overflow_report = grow_rows(runtime, capacity=growth_admission['decode_slots_per_layer'], "
    "layout='extension', mx=mx)"
)


def _mlx_calls(text: str):
    return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "mx"]


def _apply(source: str, edits, label: str) -> str:
    """Apply anchored (old, new) edits; assert single occurrence, byte-exact round trip, unchanged mx.* calls, parse."""
    updated = source
    for old, new in edits:
        if updated.count(old) != 1:
            raise RuntimeError(f"{label}: anchor not found exactly once ({updated.count(old)}x): {old[:60]!r}")
        updated = updated.replace(old, new)
    reversed_ = updated
    for old, new in edits:
        reversed_ = reversed_.replace(new, old)
    if reversed_ != source:
        raise RuntimeError(f"{label}: edits do not recover the original source")
    if _mlx_calls(source) != _mlx_calls(updated):
        raise RuntimeError(f"{label}: edits changed the module's direct mx.* operations")
    ast.parse(updated)
    return updated


def rewrite_packed_phase(source: str) -> str:
    return _apply(source, [(_PL_ANCHOR, _PL_ROUTED), (_GG_ANCHOR, _GG_ROUTED)], "packed_phase.py")


def rewrite_run_full(source: str) -> str:
    return _apply(source, [(_MODEL_ANCHOR, _MODEL_ROUTED), (_LOAD_ANCHOR, _LOAD_ROUTED), (_STAMP_ANCHOR, _STAMP_ROUTED),
                           (_ADM_ANCHOR, _ADM_ROUTED), (_GROWTH_ANCHOR, _GROWTH_ROUTED),
                           (_GROWROWS_ANCHOR, _GROWROWS_ROUTED)], "run_full.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-phase", required=True)
    ap.add_argument("--run-full")
    ap.add_argument("--out-packed-phase")
    ap.add_argument("--out-run-full")
    args = ap.parse_args()
    with open(args.packed_phase) as f:
        pp = rewrite_packed_phase(f.read())
    with open(args.out_packed_phase or args.packed_phase, "w") as f:
        f.write(pp)
    print(f"tcq3 packed_phase edits staged -> {args.out_packed_phase or args.packed_phase} "
          "(plane-lane route + growth codec gate; round-trip + mx-call checks passed)")
    if args.run_full:
        with open(args.run_full) as f:
            rf = rewrite_run_full(f.read())
        with open(args.out_run_full or args.run_full, "w") as f:
            f.write(rf)
        print(f"tcq3 run_full edits staged -> {args.out_run_full or args.run_full} "
              "(loader install + spec stamp + admission retarget; round-trip + mx-call checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
