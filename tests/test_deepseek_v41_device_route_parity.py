"""W44 device-route exactness: the slot-recycle race + a GPU parity harness.

GPU window 19 showed ``device_route`` is NOT byte-identical to the fenced path on
the real artifact (tokens collapsed to 0 from the first decode step) even though
the CPU fake-bank tests pass. Root cause (proven on CPU here): the barrier-free
device gather reads a component-bank slot WITHOUT pinning it and DEFERS execution
(async, forced only at the token-end flush), while ``gather_qmm`` reads the bank at
EVAL time -- so any admission that recycles that slot in place (the cold-recovery
pass, or the next token's LRU churn) between issue and eval corrupts the gather.
The fenced path is safe because it pins the route's slots and evaluates the gather
immediately (the wave fence) before the slot can be reused.

Two tests:
  1. ``test_deferred_device_gather_races_with_slot_recycle`` (CPU, always runs) --
     locks the mechanism: a deferred gather over a real component bank reflects an
     in-place slot mutation applied after issue, i.e. it is NOT isolated from slot
     recycling (a pinned+fenced gather would be).
  2. ``test_gpu_parity_device_route_vs_fenced`` (skipped unless
     ``MTPLX_GPU_PARITY=1``; run inside a GPU window) -- decodes N tokens on the
     real artifact with device_route vs fenced, reports the first mismatching
     token position with per-layer routing diffs, and writes a JSON receipt to
     ``MTPLX_PARITY_RECEIPT``.

CPU test: MLX pinned to CPU, tiny hy3 component-bank artifact, no GPU. Run under
``nice -n 19``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)


# ---------------------------------------------------------------------------
# 1. CPU: prove the deferred-gather slot-recycle race (the W44 root cause)
# ---------------------------------------------------------------------------
def test_deferred_device_gather_races_with_slot_recycle(tmp_path) -> None:
    """A device-route gather issued but not pinned/fenced (as the barrier-free
    path does) reads the bank at eval time, so an in-place slot recycle applied
    before the deferred eval corrupts it. This is why device_route is not exact on
    the real model: the cold-recovery admissions (and LRU churn across the decode)
    recycle slots that pending device gathers still read."""
    from mtplx.expert_manifest import load_expert_manifest
    from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
    from mtplx.models.expert_mlx import (
        _gather_component_bank,
        make_mlx_component_bank_allocator,
    )
    from mtplx.resident_loader import construct_resident_model
    from tests.test_streamed_models import _integrated_hy3_artifact

    root, config, spec, manifest_path = _integrated_hy3_artifact(
        tmp_path, expert_count=4, top_k=2
    )
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    sc = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(4),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        slot_layout="component-banks",
    )
    plan = sc.memory_plan(spec)
    rt = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        sc,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan, spec, load_expert_manifest(manifest_path)
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, rt, config=config)
        model = resident.model
        cache = model.make_cache()
        for t in (1, 2, 3, 1, 2, 3):  # warm the routed layer's bank
            mx.eval(model(mx.array([[t]], dtype=mx.int32), cache=cache))

        lid = spec.routed_layer_start
        bank = rt.component_bank_for_layer(lid)
        assert bank is not None, "bank not captured -- warm-up did not route"
        sw = model.model.layers[lid].mlp.switch_mlp
        gs, bits, codec = sw.group_size, sw.bits, sw.codec

        mx.random.seed(0)
        x = (0.3 * mx.random.normal((1, spec.hidden_size))).astype(mx.bfloat16)
        slot = mx.array([[0]], dtype=mx.int32)  # read bank row 0 (a warm slot)

        # Issue the gather the way the device path does: lazy, NOT evaluated.
        routed_deferred = _gather_component_bank(
            x, bank, slot, group_size=gs, bits=bits, codec=codec
        )
        # The correct value is the same gather evaluated NOW -- what the fenced
        # path guarantees (pin + immediate mx.eval before the slot can be reused).
        routed_now = _gather_component_bank(
            x, bank, slot, group_size=gs, bits=bits, codec=codec
        )
        mx.eval(routed_now)

        # Recycle slot 0 in place (an admission of a different expert writes new
        # bytes into exactly this storage; the device path never pinned it).
        mutated = False
        for name, arr in bank.arrays.items():
            if name.endswith(".weight"):
                view = memoryview(arr).cast("B")
                for i in range(min(len(view), 4096)):
                    view[i] = (view[i] + 137) & 0xFF
                mutated = True
        assert mutated, "no weight array to mutate"

        # Force the deferred gather AFTER the recycle (as the token-end flush does).
        mx.eval(routed_deferred)

        # A pinned+fenced gather would be isolated; the deferred one is not.
        assert not bool(mx.array_equal(routed_deferred, routed_now).item()), (
            "deferred device gather was unexpectedly isolated from the slot "
            "recycle -- the race this test documents did not reproduce"
        )
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# 2. GPU parity harness (window-only): device_route vs fenced on the real model
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("MTPLX_GPU_PARITY") != "1",
    reason="real-artifact GPU parity; set MTPLX_GPU_PARITY=1 inside a GPU window",
)
def test_gpu_parity_device_route_vs_fenced() -> None:
    """Decode N tokens on the real DSV4.1 artifact with device_route vs fenced,
    report the first mismatching token position with per-layer routing diffs, and
    write a JSON receipt to ``MTPLX_PARITY_RECEIPT``. Run inside the GPU flock."""
    from mtplx.models import expert_mlx as _expert_mlx
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming

    model_path = Path(
        os.environ.get(
            "MTPLX_PARITY_MODEL",
            os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"),
        )
    )
    n_tokens = int(os.environ.get("MTPLX_PARITY_TOKENS", "32"))
    n_prompt = int(os.environ.get("MTPLX_PARITY_PROMPT", "64"))
    mem_gib = float(os.environ.get("MTPLX_PARITY_MEM_GIB", "82"))
    receipt_path = os.environ.get("MTPLX_PARITY_RECEIPT")

    prompt = mx.array([[(i % 97) + 1 for i in range(n_prompt)]], dtype=mx.int32)

    def _decode(device_route: bool):
        if device_route:
            os.environ["MTPLX_DSV41_DEVICE_ROUTE"] = "1"
        else:
            os.environ.pop("MTPLX_DSV41_DEVICE_ROUTE", None)
        resident = load_deepseek_v41_streaming(
            model_path,
            memory_limit_bytes=int(mem_gib * (1024 ** 3)),
            max_live_kv_tokens=n_prompt + n_tokens + 8,
            slot_layout="component-banks",
            cache_scope="layer",
            island_layers=(),
        )
        model = resident.model
        # Record per-step, per-layer routed indices via the streamed switches.
        per_step_routes: list[dict[int, list[int]]] = []
        current: dict[int, list[int]] = {}
        originals = {}
        for layer in model.model.layers:
            sw = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
            if sw is None or not hasattr(sw, "layer_index"):
                continue
            lidx = int(sw.layer_index)
            orig = sw.__call__

            def wrapped(x, indices, _orig=orig, _lidx=lidx):
                try:
                    current[_lidx] = [int(v) for v in indices.reshape(-1).tolist()]
                except Exception:
                    current[_lidx] = None
                return _orig(x, indices)

            sw.__call__ = wrapped  # type: ignore[method-assign]
            originals[lidx] = (sw, orig)
        try:
            cache = model.make_cache()
            logits = model(prompt, cache=cache)
            mx.eval(logits)
            tokens: list[int] = []
            for _ in range(n_tokens):
                current = {}
                nxt = int(mx.argmax(logits[:, -1, :].astype(mx.float32), axis=-1)[0])
                tokens.append(nxt)
                per_step_routes.append(dict(current))
                logits = model(mx.array([[nxt]], dtype=mx.int32), cache=cache)
                mx.eval(logits)
        finally:
            for lidx, (sw, orig) in originals.items():
                sw.__call__ = orig  # type: ignore[method-assign]
        return tokens, per_step_routes

    fenced_tokens, fenced_routes = _decode(False)
    device_tokens, device_routes = _decode(True)

    first_mismatch = None
    for i, (a, b) in enumerate(zip(fenced_tokens, device_tokens)):
        if a != b:
            first_mismatch = i
            break

    route_diffs = None
    if first_mismatch is not None and first_mismatch < len(device_routes):
        fr = fenced_routes[first_mismatch]
        dr = device_routes[first_mismatch]
        route_diffs = {
            str(lid): {"fenced": fr.get(lid), "device": dr.get(lid)}
            for lid in sorted(set(fr) | set(dr))
            if fr.get(lid) != dr.get(lid)
        }

    receipt = {
        "n_tokens": n_tokens,
        "n_prompt": n_prompt,
        "byte_identical": fenced_tokens == device_tokens,
        "first_mismatch_pos": first_mismatch,
        "fenced_tokens": fenced_tokens,
        "device_tokens": device_tokens,
        "route_diffs_at_first_mismatch": route_diffs,
    }
    if receipt_path:
        Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt_path).write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("fenced_tokens", "device_tokens")}, indent=2))
    assert fenced_tokens == device_tokens, (
        f"device_route diverged from fenced at token {first_mismatch}; "
        f"route diffs: {route_diffs}"
    )
