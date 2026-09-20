"""F33 fast transition-observe: a construction-time, state-identical replacement of
each per-layer transition-window bank's ``_observe_transition_window``.

The two-group verify pipeline calls ``LayerExpertSlotBank._observe_transition_window``
twice per layer per forward (once per group's ``plan()``); the F30 sampling profile
prices its ``counts[np.ix_(previous, current)] += 1.0`` 2-D fancy read-modify-write on
the 384x384 float32 table at ~1.7% of the generation thread (~1.04 s/run).  The fancy
outer-product update touches exactly ``|previous| x |current|`` cells; because ``current``
and ``previous`` each hold UNIQUE experts, no cell repeats, so the same set of cells with
the same ``+= 1.0`` is expressed as a 1-D fancy add on the flattened counts VIEW at the
row-major flat indices.  ``counts`` is C-contiguous, so ``counts.reshape(-1)`` is a view
(no copy): the add writes straight back into ``counts``.  Every other line -- denominators,
window append/pop, window frequency, ``_transition_previous`` -- is byte-for-byte the
pinned method, in the same order, so the bank's state is identical after every call and
routed outputs stay bit-exact (a cache decision only, never a routed output).

Installed ONCE at the F16 quiescent post-prime boundary (``install`` when armed and
``MTPLX_DSV41_F33_FAST_OBSERVE=1``): every bank is validated first, then every bank's
bound method is replaced, so a rejected bank leaves no bank half-patched (AGENTS.md
correct-by-design: validate once, fail once, clearly, before measured generation).  Pure
numpy; imports no MLX.
"""
from __future__ import annotations

from types import MethodType

import numpy as np

from mtplx.expert_streaming import TRANSITION_WINDOW_CACHE_POLICY

# Marks a bank whose observe has already been replaced, so a second install refuses
# rather than double-binding (correct-by-design: fail once, clearly, at construction).
_FAST_OBSERVE_ATTR = "_f33_fast_observe"


def _fast_observe_transition_window(self, current: tuple[int, ...]) -> None:
    """State-identical to ``LayerExpertSlotBank._observe_transition_window`` (pinned in
    ``mtplx/expert_streaming.py``): same updates to ``counts``, ``denominators``,
    ``window``, ``window_frequency`` and ``_transition_previous``, in the same order.  The
    ONLY change is the transition-counts update: the 2-D ``counts[np.ix_(previous, current)]
    += 1.0`` becomes a 1-D fancy add on ``counts.reshape(-1)`` (a view) at the row-major flat
    indices ``previous[:, None] * ncols + current[None, :]``.  ``current`` and ``previous``
    hold unique experts, so no cell repeats and the ``+= 1.0`` semantics are identical."""
    counts = self._transition_counts
    denominators = self._transition_denominators
    window_frequency = self._transition_window_frequency
    window = self._transition_window
    assert counts is not None
    assert denominators is not None
    assert window_frequency is not None
    assert window is not None

    current_array = np.fromiter(current, dtype=np.intp, count=len(current))
    previous = self._transition_previous
    if previous is not None:
        previous_array = np.fromiter(
            previous, dtype=np.intp, count=len(previous)
        )
        flat_counts = counts.reshape(-1)
        flat_counts[
            (previous_array[:, None] * counts.shape[1] + current_array[None, :]).reshape(-1)
        ] += 1.0
        denominators[previous_array] += float(len(current))

    window.append(current)
    window_frequency[current_array] += 1.0
    if len(window) > self._transition_window_limit:
        expired = window.popleft()
        expired_array = np.fromiter(
            expired, dtype=np.intp, count=len(expired)
        )
        window_frequency[expired_array] -= 1.0
    self._transition_previous = current


def _validate_bank(bank, layer) -> None:
    """Refuse unless ``bank`` is the plain transition-window policy with the exact table
    the flat add relies on: non-None float32 (384, 384) C-contiguous ``counts`` and a
    (384,) ``denominators``.  Raises before any bank is touched (validate-first)."""
    if getattr(bank, _FAST_OBSERVE_ATTR, False):
        raise RuntimeError(
            f"F33 fast_observe: layer {layer} bank already has the fast observe installed"
        )
    if bank.cache_policy != TRANSITION_WINDOW_CACHE_POLICY:
        raise RuntimeError(
            f"F33 fast_observe: layer {layer} bank.cache_policy is {bank.cache_policy!r}, "
            f"not the plain {TRANSITION_WINDOW_CACHE_POLICY!r}"
        )
    counts = bank._transition_counts
    if counts is None:
        raise RuntimeError(f"F33 fast_observe: layer {layer} bank._transition_counts is None")
    if counts.shape != (384, 384):
        raise RuntimeError(
            f"F33 fast_observe: layer {layer} _transition_counts shape is "
            f"{counts.shape!r}, not (384, 384)"
        )
    if counts.dtype != np.float32:
        raise RuntimeError(
            f"F33 fast_observe: layer {layer} _transition_counts dtype is "
            f"{counts.dtype!r}, not float32"
        )
    if not counts.flags["C_CONTIGUOUS"]:
        raise RuntimeError(
            f"F33 fast_observe: layer {layer} _transition_counts is not C-contiguous "
            "(counts.reshape(-1) must be a view for the flat add to write back)"
        )
    denominators = bank._transition_denominators
    if denominators is None or denominators.shape != (384,):
        raise RuntimeError(
            f"F33 fast_observe: layer {layer} _transition_denominators shape is "
            f"{None if denominators is None else denominators.shape!r}, not (384,)"
        )


def install_fast_observe(runtime) -> dict:
    """Replace ``_observe_transition_window`` with the flat-add variant on EVERY per-layer
    bank in ``runtime._banks`` (F33).  Validates every bank first, then rebinds each, so a
    rejected bank leaves no bank half-changed.  Returns the install-report fragment."""
    banks = runtime._banks
    if not banks:
        raise RuntimeError(
            "F33 fast_observe: runtime._banks is empty; the F16 pipeline requires "
            "per-layer transition-window banks (global bank scope is refused upstream)"
        )
    for layer, bank in banks.items():
        _validate_bank(bank, layer)
    for bank in banks.values():
        bank._observe_transition_window = MethodType(_fast_observe_transition_window, bank)
        setattr(bank, _FAST_OBSERVE_ATTR, True)
    return {"fast_observe_banks": len(banks)}
