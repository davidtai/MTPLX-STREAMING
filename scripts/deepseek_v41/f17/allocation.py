"""F17 per-layer extension-row allocation for the DeepSeek-V4.1 packed decode lane.

The expert cache gives every routed layer ``old`` (=84) persistent rows at prefill
plus a shared pool of *extension* rows added at the post-prefill growth boundary.
The retained ``extension.grow_rows`` gives every layer the SAME ``capacity-old``
extension rows.  A cache decision never changes model output, so ANY allocation of
the SAME total extension rows is byte-identical (F14 study, receipt
``docs/deepseek-v41/receipts/f14-per-layer-capacity-20260919``): a non-uniform split
removes expert-record reads at zero extra memory (oracle -4.9%, causal prefill rule
-1.8%).

``allocate`` returns ``{layer: added_L}`` and is selected ONCE (at the growth
boundary, off the decode hot path) by ``MTPLX_DSV41_F17_ALLOC``:

  unset / ``uniform``   every layer gets ``capacity-old`` -- the staged grow_rows then
                        behaves EXACTLY like the retained (uniform) path.
  ``prefill_rule``      extension rows proportional to the per-layer prefill
                        diffuseness ``1 - (mass of the top-`old` experts in the
                        layer's prefill route frequency)`` (F14's selected statistic
                        ``one_minus_top84_mass``, read from the live
                        ``LayerExpertSlotBank._prefill_route_freq``), quantised to
                        multiples of 4, each layer in ``[4, MAX_ADDED]``, remainder
                        fixed deterministically to the exact total.
  ``vector:<v0,...>``   explicit per-layer added rows (one per routed layer, each >=4,
                        exact sum) -- e.g. the F14 oracle allocation.

The TOTAL stays ``len(layers) * (capacity - old)`` in every mode, so the retained
admission / accounting math is untouched.  Per-layer cap: ``MTPLX_DSV41_F17_MAX_ADDED``
(default 96; floored to a multiple of 4, >=4).  No layer ever gets a zero-capacity
extension bank (minimum 4 rows).

CPU-only, pure Python (no MLX / no GPU / no I/O).  Prints ONE provenance line
``F17_ALLOC {json}`` at install; nothing on the hot path.
"""
from __future__ import annotations

import json
import os

ALLOC_ENV = "MTPLX_DSV41_F17_ALLOC"
MAX_ADDED_ENV = "MTPLX_DSV41_F17_MAX_ADDED"

MIN_ADDED = 4            # no zero-capacity extension banks
QUANTUM = 4              # prefill_rule rows are multiples of 4
DEFAULT_MAX_ADDED = 96
_VECTOR_PREFIX = "vector:"
_SHAPE_PREFIX = "shape:"   # 40 non-negative weights, apportioned to WHATEVER capacity is admitted


def allocate(runtime, *, capacity, old=84):
    """Return ``{layer: extension_rows}`` for the routed layers, mode from env.

    ``sum(result.values()) == len(layers) * (capacity - old)`` in every mode.
    """
    layers = tuple(runtime.spec.routed_layer_indices)
    n = len(layers)
    if capacity <= old:
        raise ValueError(
            f"F17 allocate requires capacity>old (extension rows); got "
            f"capacity={capacity}, old={old}"
        )
    total = n * (capacity - old)
    max_added = _resolve_max_added()

    raw = os.environ.get(ALLOC_ENV)
    mode = (raw or "uniform").strip()
    statistic = None

    if mode in ("", "uniform"):
        provenance = "uniform"
        added = {layer: capacity - old for layer in layers}
    elif mode == "prefill_rule":
        provenance = "prefill_rule"
        statistic = _prefill_statistic(runtime, layers, old)
        vec = _apportion(
            statistic, total, n=n,
            min_added=MIN_ADDED, max_added=max_added, quantum=QUANTUM,
        )
        added = {layer: vec[i] for i, layer in enumerate(layers)}
    elif mode.startswith(_SHAPE_PREFIX):
        # A fixed per-layer PROFILE (e.g. the F14 oracle shape) rescaled to the capacity the
        # live admission picked, so a high-baseline run that admits one row fewer cannot die on
        # an exact-sum mismatch after a full prefill (review 2026-09-19).
        provenance = "shape"
        weights = _parse_shape(mode[len(_SHAPE_PREFIX):], n=n)
        vec = _apportion(
            weights, total, n=n,
            min_added=MIN_ADDED, max_added=max_added, quantum=QUANTUM,
        )
        added = {layer: vec[i] for i, layer in enumerate(layers)}
    elif mode.startswith(_VECTOR_PREFIX):
        provenance = "vector"
        vec = _parse_vector(mode[len(_VECTOR_PREFIX):], n=n, total=total)
        added = {layer: vec[i] for i, layer in enumerate(layers)}
    else:
        raise ValueError(
            f"{ALLOC_ENV}={raw!r} is not a recognised F17 allocation mode "
            f"(expected unset/'uniform', 'prefill_rule', or 'vector:<{n} ints>')"
        )

    _validate(added, layers, total)
    print(
        "F17_ALLOC "
        + json.dumps(
            {
                "mode": provenance,
                "capacity": capacity,
                "old": old,
                "layers": n,
                "total_added": total,
                "max_added": max_added,
                "vector": [added[layer] for layer in layers],
                "statistic": (
                    None if statistic is None
                    else [round(s, 6) for s in statistic]
                ),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return added


def _resolve_max_added():
    raw = os.environ.get(MAX_ADDED_ENV)
    if raw is None or raw.strip() == "":
        value = DEFAULT_MAX_ADDED
    else:
        try:
            value = int(raw.strip())
        except ValueError:
            raise ValueError(f"{MAX_ADDED_ENV}={raw!r} must be an integer")
    # Floor to a multiple of the quantum so the multiple-of-4 invariant holds.
    value -= value % QUANTUM
    if value < MIN_ADDED:
        raise ValueError(
            f"{MAX_ADDED_ENV} must be >= {MIN_ADDED} after flooring to a "
            f"multiple of {QUANTUM}; got {raw!r} -> {value}"
        )
    return value


def _prefill_statistic(runtime, layers, old):
    """``1 - top-`old` mass`` of each layer's live ``_prefill_route_freq``.

    Read once at the growth boundary; the counter holds the prompt's prefill
    routing (it is populated during prefill seeding and is not cleared until a
    decode route is seen).  An empty counter means the statistic is unavailable
    -- fail here, before any measured generation, rather than install a lane that
    cannot do its job.
    """
    stats = []
    for layer in layers:
        freq = runtime._banks[layer]._prefill_route_freq
        counts = sorted((int(c) for c in freq.values()), reverse=True)
        total = sum(counts)
        if total <= 0:
            raise RuntimeError(
                f"F17 prefill_rule: layer {layer} has an empty _prefill_route_freq; "
                "the prefill routing statistic is unavailable at the growth boundary"
            )
        top_mass = sum(counts[:old])
        stats.append(1.0 - top_mass / total)
    return stats


def _apportion(weights, total, *, n, min_added, max_added, quantum):
    """Largest-remainder apportionment in quanta.

    Returns ``n`` values, each a multiple of ``quantum`` in ``[min_added,
    max_added]``, summing to exactly ``total``, proportional to ``weights``.
    Deterministic (ties broken by ascending index).
    """
    if total % quantum or min_added % quantum or max_added % quantum:
        raise ValueError("F17 apportion needs total/min/max as multiples of the quantum")
    umin, umax, units = min_added // quantum, max_added // quantum, total // quantum
    if not (n * umin <= units <= n * umax):
        raise ValueError(
            f"F17 apportion infeasible: total={total} is outside "
            f"[{n * min_added}, {n * max_added}] for {n} layers"
        )
    weights = [max(0.0, float(w)) for w in weights]
    add = [0] * n                    # extra units above umin
    headroom = [umax - umin] * n
    remaining = units - n * umin
    while remaining > 0:
        active = [i for i in range(n) if add[i] < headroom[i]]
        weight_sum = sum(weights[i] for i in active)
        if weight_sum <= 0.0:
            # No signal among the layers that still have headroom: fill by
            # ascending index so the result stays deterministic.
            for i in active:
                if remaining <= 0:
                    break
                add[i] += 1
                remaining -= 1
            continue
        ideal = {i: weights[i] / weight_sum * remaining for i in active}
        placed = 0
        for i in active:
            give = min(headroom[i] - add[i], int(ideal[i]))
            add[i] += give
            placed += give
        remaining -= placed
        if remaining <= 0:
            break
        for i in sorted(active, key=lambda i: (-(ideal[i] - int(ideal[i])), i)):
            if remaining <= 0:
                break
            if add[i] < headroom[i]:
                add[i] += 1
                remaining -= 1
    return [(umin + add[i]) * quantum for i in range(n)]


def _parse_shape(text, *, n):
    parts = text.split(",")
    if len(parts) != n:
        raise ValueError(f"F17 shape needs exactly {n} comma-separated weights; got {len(parts)}")
    try:
        weights = [float(part.strip()) for part in parts]
    except ValueError:
        raise ValueError("F17 shape weights must be numbers")
    if any(w < 0.0 for w in weights) or sum(weights) <= 0.0:
        raise ValueError("F17 shape weights must be non-negative with a positive sum")
    return weights


def _parse_vector(text, *, n, total):
    parts = text.split(",")
    if len(parts) != n:
        raise ValueError(
            f"F17 vector needs exactly {n} comma-separated values (one per routed "
            f"layer); got {len(parts)}"
        )
    vec = []
    for part in parts:
        token = part.strip()
        try:
            value = int(token)
        except ValueError:
            raise ValueError(f"F17 vector value {token!r} is not an integer")
        vec.append(value)
    if any(v < MIN_ADDED for v in vec):
        raise ValueError(
            f"F17 vector values must each be >= {MIN_ADDED} (no zero-capacity banks)"
        )
    if sum(vec) != total:
        raise ValueError(
            f"F17 vector must sum to {total} (= layers*(capacity-old)); got {sum(vec)}"
        )
    return vec


def _validate(added, layers, total):
    if set(added) != set(layers):
        raise AssertionError("F17 allocation keys must match the routed layers exactly")
    values = [added[layer] for layer in layers]
    if any(v < MIN_ADDED for v in values):
        raise AssertionError(f"F17 allocation has a layer below {MIN_ADDED} extension rows")
    if sum(values) != total:
        raise AssertionError(f"F17 allocation sum {sum(values)} != required total {total}")
