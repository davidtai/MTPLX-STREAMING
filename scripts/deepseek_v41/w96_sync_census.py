"""W96 audit: count blocking host syncs per streamed-switch call on CPU.

Fake runtime modeled on tests/test_multi_wave_deferred_release.py (same lock
ownership), extended with miss parts so the split path's per-part fences are
exercised.  No model, no bank, no GPU: mx.set_default_device(mx.cpu).
"""
from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mtplx.expert_runtime import RouteWave  # noqa: E402
from mtplx.expert_streaming import RoutingPhase  # noqa: E402
from mtplx.models import expert_mlx  # noqa: E402
from mtplx.models.expert_mlx import HotExpertSwitchGLU, expert_routing_phase  # noqa: E402

EVENTS: list[str] = []
_orig_eval = mx.eval
_orig_async = mx.async_eval


def _eval(*a, **k):
    EVENTS.append("mx.eval")
    return _orig_eval(*a, **k)


def _async(*a, **k):
    EVENTS.append("mx.async_eval")
    return _orig_async(*a, **k)


mx.eval = _eval
mx.async_eval = _async

# stub the gather: identity over the selected rows (shape [rows, hidden])
expert_mlx._run_component_bank_q4 = lambda selected, *_a, **_k: selected

TOP_K = 6
HID = 2


def _ready(experts, bank, tag):
    return SimpleNamespace(
        plan=SimpleNamespace(experts=tuple(experts), hits=tuple(experts), loads=()),
        bindings=tuple(
            SimpleNamespace(expert=e, buffer=SimpleNamespace(bank=bank)) for e in experts
        ),
        release=lambda **kw: EVENTS.append(f"release:{tag}"),
    )


class _Pending:
    def __init__(self, lock, bank, experts, hits, miss_parts):
        self._lock = lock
        self.plan = SimpleNamespace(hits=tuple(hits), misses=tuple(m for part in miss_parts for m in part))
        self.hit_ready = _ready(hits, bank, "hits") if hits else None
        self._parts = [(_ready(part, bank, f"miss{i}")) for i, part in enumerate(miss_parts)]
        self._pending = len(self._parts)

    @property
    def misses_pending(self):
        return self._pending > 0

    def iter_ready_misses(self):
        for r in self._parts:
            EVENTS.append("ssd-wait(miss part ready)")
            self._pending -= 1
            yield r

    def release_hits(self):
        EVENTS.append("release_hits")

    def release_miss(self, r):
        EVENTS.append("release_miss")

    def close(self):
        EVENTS.append("close(lock released)")
        self._lock.release()

    def abort(self, exc):
        EVENTS.append("abort")


class Runtime:
    def __init__(self, *, hits, miss_parts, deferred_cfg):
        self.spec = SimpleNamespace(top_k=TOP_K, hidden_size=HID, quant_group_size=32,
                                    quant_bits=4, expert_codec="mxfp4", swiglu_limit=10.0,
                                    key="fake")
        self.config = SimpleNamespace(
            slot_layout="component-banks", resource_telemetry=False,
            deferred_pin_release=deferred_cfg, split_route_release=("deferred" if deferred_cfg else "fenced"),
            overlap_miss_reads=False, trace_routes=False,
        )
        self.manifest = SimpleNamespace(sidecar=None)
        self._pipeline_ledger = None
        self._bank = object()
        self.layer_lock = threading.Lock()
        self._hits = tuple(hits)
        self._miss_parts = tuple(tuple(p) for p in miss_parts)
        self.deferred = []

    def observe_route(self, *a, **k):
        EVENTS.append("observe_route")

    def prepare_prefill_seed(self, *a, **k):
        return ()

    def route_waves(self, expert_ids, **k):
        ex = tuple(expert_ids)
        EVENTS.append("route_waves")
        return (RouteWave(positions=tuple(range(len(ex))), experts=ex),)

    def _batch_admission_slots(self):
        return 48

    def shadow_bank_for_layer(self, layer):
        return None

    def try_all_hit_route(self, layer, experts, **k):
        with self.layer_lock:
            EVENTS.append("try_all_hit_route(lock)")
            if self._miss_parts:
                return None
            return _ready(experts, self._bank, "allhit")

    def begin_split_route(self, layer, experts, **k):
        self.layer_lock.acquire()
        EVENTS.append("begin_split_route(lock acquired; miss reads submitted)")
        return _Pending(self.layer_lock, self._bank, tuple(experts), self._hits, self._miss_parts)

    def defer_slot_release(self, ready, outputs):
        self.deferred.append(ready)
        EVENTS.append("defer_slot_release")

    def flush_deferred_slot_releases(self, *, evaluate=False):
        n = len(self.deferred)
        self.deferred.clear()
        EVENTS.append(f"flush_deferred({n})")


def run(label, *, hits, miss_parts, deferred_cfg=False, env=None):
    global EVENTS
    EVENTS = []
    saved = {k: os.environ.get(k) for k in ("MTPLX_DSV41_SWITCH_FASTPATH", "MTPLX_DSV41_SWITCH_SUBMIT")}
    for k in saved:
        os.environ.pop(k, None)
    if env:
        os.environ.update(env)
    try:
        rt = Runtime(hits=hits, miss_parts=miss_parts, deferred_cfg=deferred_cfg)
        sw = HotExpertSwitchGLU(rt, 1)
        x = mx.zeros((1, 1, HID), dtype=mx.bfloat16)
        idx = mx.array([[list(range(TOP_K))]], dtype=mx.int32)
        with expert_routing_phase(RoutingPhase.DECODE):
            out = sw(x, idx)
        _orig_eval(out)
        n_eval = EVENTS.count("mx.eval")
        n_async = EVENTS.count("mx.async_eval")
        print(f"\n### {label}: blocking mx.eval={n_eval}  async_eval={n_async}  lock_held_after={rt.layer_lock.locked()}")
        for e in EVENTS:
            print("   ", e)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    run("ALL-HIT layer, shipped (fenced) config", hits=range(6), miss_parts=[])
    run("SPLIT layer: 4 hits + 2 miss parts, shipped (fenced) config",
        hits=[0, 1, 2, 3], miss_parts=[[4], [5]])
    run("SPLIT layer: 4 hits + 2 miss parts, SWITCH_FASTPATH+SUBMIT (window-37 ring_switch arm)",
        hits=[0, 1, 2, 3], miss_parts=[[4], [5]],
        env={"MTPLX_DSV41_SWITCH_FASTPATH": "1", "MTPLX_DSV41_SWITCH_SUBMIT": "1"})
    run("ALL-HIT layer, SWITCH_FASTPATH+SUBMIT", hits=range(6), miss_parts=[],
        env={"MTPLX_DSV41_SWITCH_FASTPATH": "1", "MTPLX_DSV41_SWITCH_SUBMIT": "1"})
