#!/usr/bin/env python3
"""F7: score-aware expert-cache retention screen for DeepSeek-V4.1 M6 decode.

CPU ONLY (MLX imports are hard-blocked by a NoMLX meta-path finder), numpy only,
single process; run under ``nice -n 19``.  This is a retrospective cache-policy
screen, NOT a throughput result and NOT a production-code proposal.  Full method,
provenance and results: docs/deepseek-v41/receipts/f7-cache-policy-20260919/.

The shipped decode cache is the transition-window policy in
``mtplx/expert_streaming.py`` (LayerExpertSlotBank, single pool, 384 experts).
Its retention score is
    score(e) = w_pred * P(e | previous route)          (online 1st-order transition)
             + w_freq * windowed_frequency(e)/max       (last WINDOW routes)
             + w_rec  * 1/(1 + epoch - last_used(e))     (recency)
and at each route the top-scoring of {evictable residents + this route's misses}
are retained; a rejected miss is served through transient scratch.  All three
terms are HISTORY-only.  This module adds a fourth term ``w_gate * g(expert)``
derived from the router's own gate scores -- a causal signal history cannot see.

PROXY LIMITATION (stated everywhere): the 64-cycle capture stores, for target
layer T at cycle c, the gate applied to layer T-1's post-attention input -- a
1-layer-ahead PREDICTION of layer T's scores (top-6 recall ~72% vs the true
route), NOT layer T's true gate scores.  g here is built from that predicted
tensor, so every gain reported is a LOWER BOUND on a true-score signal.

The retention decision is a pure cache decision: it changes only which records
are read, never a routed expert's computed output, so any policy is bit-exact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import runpy
import sys
import time
from pathlib import Path


class _NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("score_aware_cache_sim is CPU-only; MLX is forbidden")


if not any(isinstance(f, _NoMLX) for f in sys.meta_path):
    sys.meta_path.insert(0, _NoMLX())
sys.path.insert(0, str(Path.cwd()))            # worktree mtplx ahead of any install
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_router_capture as rrc  # noqa: E402  (capture loader, ranked predictor)
import overlap_schedule_sim as osim  # noqa: E402  (trace loader, make_bank, timing)
import analyze_route_cache as arc  # noqa: E402  (clairvoyant optimal_batch_misses)
from mtplx.expert_streaming import LayerExpertSlotBank  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
FT, NT, NE, CYC = rrc.FIRST_TARGET, rrc.N_TARGET, rrc.N_EXPERTS, rrc.CYCLES  # 4,36,384,64
CAP_PERSIST, CAP_TRANSIENT = 105, 48           # captured decode_slots_per_layer + transient
HELDOUT, TRAIN = slice(32, 64), slice(0, 32)
REC_BYTES = osim.REC_BYTES                     # 17,694,720 B ("17.7 MB")
RATE = osim.DEFAULT_RATE                        # 12.9 GB/s decimal
REC_S = REC_BYTES / (RATE * 1e9)                # ~1.3717 ms per record read

# Shipped transition-window weights, read from LayerExpertSlotBank.__init__:
#   transition-window       (default, the CAPTURED run): 0.7 / 0.2 / 0.1, window 16
#   transition-window-tuned                             : 0.8 / 0.1 / 0.1, window 32
W_PRED, W_FREQ, W_REC, WINDOW = 0.7, 0.2, 0.1, 16
W_PRED_TUNED, W_FREQ_TUNED, W_REC_TUNED, WINDOW_TUNED = 0.8, 0.1, 0.1, 32
# w_gate log grid, relative to the existing weights (w_pred 0.7 .. w_rec 0.1),
# ~x2 spacing, extended past w_pred to bracket the peak (as w->inf a fourth-term
# ranking degenerates to the pure-score policy, which the sweep shows is far worse):
W_GATE_GRID = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8)
G_KINDS = ("max", "mean", "recip_rank", "ind12", "ind24", "ind48")
EMA_DECAYS = (None, 0.5, 0.7, 0.9)             # None = raw (as-is), else EMA over cycles


# --------------------------------------------------------------------------- #
# g-vector builders (feature 1 = post-attention router; the stronger proxy)
# --------------------------------------------------------------------------- #
def build_g(cap: dict, feature: int = 1) -> dict:
    """g[(kind, ema)][idx] -> (64, 384) f32 causal gate-derived retention signal.

    Every kind is normalized to [0,1] (like the shipped window_frequency/max term)
    so w_gate is comparable to w_freq/w_rec.  EMA over cycles is causal: cycle c
    uses only scores at cycles <= c.  All from the PREDICTED (1-ahead) tensor.
    """
    s = cap["scores"][feature]                  # (64,36,6,384)
    raw = {}
    mx = s.max(axis=2)                          # (64,36,384) max-over-rows
    mn = s.mean(axis=2)                         # (64,36,384) mean-over-rows
    for kind in G_KINDS:
        g = np.zeros((NT, CYC, NE), np.float32)
        for idx in range(NT):
            if kind == "max":
                a = mx[:, idx, :]
                g[idx] = a / np.maximum(a.max(1, keepdims=True), 1e-9)
            elif kind == "mean":
                a = mn[:, idx, :]
                g[idx] = a / np.maximum(a.max(1, keepdims=True), 1e-9)
            else:
                # rank of each expert by max-over-rows (0 = highest)
                order = np.argsort(-mx[:, idx, :], axis=1, kind="stable")
                rank = np.empty((CYC, NE), np.int32)
                np.put_along_axis(
                    rank, order, np.broadcast_to(np.arange(NE), (CYC, NE)), axis=1)
                if kind == "recip_rank":
                    g[idx] = (1.0 / (1.0 + rank)).astype(np.float32)
                else:
                    R = {"ind12": 12, "ind24": 24, "ind48": 48}[kind]
                    g[idx] = (rank < R).astype(np.float32)
        raw[kind] = g
    out = {}
    for kind in G_KINDS:
        out[(kind, None)] = raw[kind]
        for d in EMA_DECAYS[1:]:
            g = np.empty_like(raw[kind])
            acc = np.zeros((NT, NE), np.float32)
            for c in range(CYC):
                acc = d * acc + (1.0 - d) * raw[kind][:, c, :]
                g[:, c, :] = acc
            out[(kind, d)] = g
    return out


# --------------------------------------------------------------------------- #
# Score-aware bank: the shipped LayerExpertSlotBank with a configurable retention
# score.  Only the retention/eviction score is changed; plan(), admission slot
# assignment, transient overflow and every route output stay byte-identical.
# --------------------------------------------------------------------------- #
class ScoreAwareBank(LayerExpertSlotBank):
    _mode = "control"        # control | fourth_term | pure | victim_only
    _w_gate = 0.0
    _gate_g = None           # current-cycle [384] g vector

    def _transition_window_scores(self):
        base = super()._transition_window_scores()
        if self._gate_g is None:
            return base
        if self._mode == "fourth_term":
            return base + np.float32(self._w_gate) * self._gate_g
        if self._mode == "pure":
            return self._gate_g.astype(np.float32)   # ranking is scale-invariant
        return base                                  # victim_only handled below

    def _transition_window_admissions(self, misses, *, pinned):
        if self._mode != "victim_only" or self._gate_g is None:
            return super()._transition_window_admissions(misses, pinned=pinned)
        # History decides ADMISSION (base score, shipped rule); the score-EMA g
        # picks the VICTIM among the evictable residents.
        if not misses:
            return {}
        base = super()._transition_window_scores()
        admitted = self._admitted_by_base(misses, pinned, base)
        if not admitted:
            return {}
        blocked = pinned | self._pinned if self._pinned else pinned
        evictable = [(e, s) for s, e in enumerate(self._slot_to_expert)
                     if e is not None and e not in blocked]
        free = max(0, self._persistent_capacity - self.occupancy)
        empty = [s for s, e in enumerate(self._slot_to_expert) if e is None][:free]
        n_victim = max(0, len(admitted) - len(empty))
        # coldest g first (lowest gate score = best eviction victim)
        evictable.sort(key=lambda es: (float(self._gate_g[es[0]]),
                                        self._history[es[0]].last_used, -es[0]))
        victim = [s for _e, s in evictable[:n_victim]]
        slots = empty + victim
        return dict(zip(admitted, slots[:len(admitted)], strict=True))

    def _admitted_by_base(self, misses, pinned, scores):
        """Which misses the shipped base-score retention would admit (identity)."""
        blocked = pinned | self._pinned if self._pinned else pinned
        evict = [e for s, e in enumerate(self._slot_to_expert)
                 if e is not None and e not in blocked]
        free = max(0, self._persistent_capacity - self.occupancy)
        empty = [s for s, e in enumerate(self._slot_to_expert) if e is None][:free]
        adjustable = len(empty) + len(evict)
        if adjustable == 0:
            return []
        cand = evict + list(misses)
        keep_n = min(adjustable, len(cand))
        rank = sorted(cand, key=lambda e: self._transition_window_rank(e, scores),
                      reverse=True)[:keep_n]
        rset = set(rank)
        return [e for e in misses if e in rset]


def seed_bank(persist0: np.ndarray, *, mode="control", w_gate=0.0,
              policy="transition-window") -> ScoreAwareBank:
    """A ScoreAwareBank seeded with the captured cycle-0 persistent residents."""
    b = ScoreAwareBank(expert_count=NE, persistent_slots=CAP_PERSIST,
                       transient_slots=CAP_TRANSIENT, cache_policy=policy,
                       single_pool=True)
    b._mode, b._w_gate = mode, w_gate
    for i, e in enumerate(int(x) for x in np.nonzero(persist0)[0]):
        b._slot_to_expert[i] = e
        b._expert_to_slot[e] = i
        b._pool_clock += 1
        b._pool_recency[e] = b._pool_clock
        b._history[e].last_used = 0             # resident == used at the boundary
    b._decode_epoch = 0
    return b


def routes_from_capture(cap: dict) -> dict:
    """routes[idx][c] = unique routed experts for target layer idx+4 at cycle c."""
    actual = cap["actual"]
    return {idx: [np.unique(actual[c, idx + FT]).astype(int).tolist()
                  for c in range(CYC)] for idx in range(NT)}


def replay_capture(cap, routes, *, mode="control", g=None, w_gate=0.0,
                   policy="transition-window") -> np.ndarray:
    """Per-(cycle, target-layer) demand-miss count under the given retention mode."""
    persist0 = cap["persistent"][0]
    miss = np.zeros((CYC, NT), int)
    for idx in range(NT):
        b = seed_bank(persist0[idx], mode=mode, w_gate=w_gate, policy=policy)
        gi = g[idx] if g is not None else None
        for c in range(CYC):
            b._gate_g = gi[c] if gi is not None else None
            miss[c, idx] = len(b.plan(routes[idx][c], phase="decode").misses)
    return miss


# --------------------------------------------------------------------------- #
# Task C: Belady-style clairvoyant lower bounds at 105+48
# --------------------------------------------------------------------------- #
def optimal_mandatory(sequence, capacity, initial=()):
    """Belady MIN with MANDATORY admission: every current-batch expert is retained
    this step (cannot be bypassed); evict farthest-next-use among the OTHERS.
    (The 'runtime's older mandatory-admission oracle' of analyze_route_cache.)"""
    batches = [set(s) for s in sequence]
    never = len(batches) + 1
    nxt, after = {}, [{} for _ in batches]
    for i in range(len(batches) - 1, -1, -1):
        for e in batches[i]:
            after[i][e] = nxt.get(e, never)
            nxt[e] = i
    resident = {e: nxt.get(e, never) for e in set(initial)}
    misses = 0
    for i, needed in enumerate(batches):
        misses += len(needed - resident.keys())
        resident.update(after[i])
        if len(resident) > capacity:
            room = capacity - len(needed)
            evict = [e for e in resident if e not in needed]
            keep_ev = sorted(evict, key=lambda e: (resident[e], e))[:max(0, room)]
            resident = {e: resident[e]
                        for e in list(needed)[:capacity] + keep_ev}
    return misses


def task_c(cap, routes) -> dict:
    persist0 = cap["persistent"][0]
    bypass = mand = 0
    for idx in range(NT):
        init = [int(e) for e in np.nonzero(persist0[idx])[0]]
        bypass += arc.optimal_batch_misses(routes[idx], CAP_PERSIST, init)
        mand += optimal_mandatory(routes[idx], CAP_PERSIST, init)
    ho_bypass = ho_mand = 0
    for idx in range(NT):
        init = [int(e) for e in np.nonzero(cap["physical"][32, idx])[0]][:CAP_PERSIST]
        ho_bypass += arc.optimal_batch_misses(routes[idx][32:], CAP_PERSIST, init)
        ho_mand += optimal_mandatory(routes[idx][32:], CAP_PERSIST, init)
    return {"definition": "clairvoyant per-layer eviction, temporary service; "
            "bypass = analyze_route_cache.optimal_batch_misses (may drop a just-used "
            "one-use expert, == the 29,812 floor family at 73 slots); mandatory = "
            "force-keep every current-batch expert, evict farthest-next-use among "
            "the rest.  Seeded from persistent[0] (all-64) / physical[32] (held-out).",
            "capacity": f"{CAP_PERSIST}+{CAP_TRANSIENT}",
            "all64": {"bypass_reads": bypass, "mandatory_reads": mand},
            "heldout": {"bypass_reads": ho_bypass, "mandatory_reads": ho_mand}}


# --------------------------------------------------------------------------- #
# Task B: score-aware retention sweep + fidelity + significance
# --------------------------------------------------------------------------- #
def _sign_test(delta_layer: np.ndarray) -> dict:
    """Per-layer sign test: delta<0 == variant reads fewer (improved)."""
    imp = int((delta_layer < 0).sum())
    wor = int((delta_layer > 0).sum())
    tie = int((delta_layer == 0).sum())
    n = imp + wor
    # two-sided binomial tail p against p=0.5 (exact, no scipy)
    from math import comb
    if n == 0:
        p = 1.0
    else:
        k = min(imp, wor)
        tail = sum(comb(n, j) for j in range(0, k + 1)) / (2.0 ** n)
        p = min(1.0, 2.0 * tail)
    return {"layers_improved": imp, "layers_worse": wor, "layers_tied": tie,
            "sign_test_p_two_sided": p}


def _bootstrap(delta_layer: np.ndarray, iters=10000, seed=20260919) -> dict:
    rng = np.random.default_rng(seed)
    n = len(delta_layer)
    means = delta_layer[rng.integers(0, n, size=(iters, n))].mean(1)
    return {"mean_delta_reads_per_layer": float(delta_layer.mean()),
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5))}


def task_b(cap, routes, g_all) -> dict:
    capt = np.array([[int(cap["reads"][c, idx + FT].sum()) for idx in range(NT)]
                     for c in range(CYC)])
    ctl = replay_capture(cap, routes, mode="control")
    ctl_tuned = replay_capture(cap, routes, mode="control",
                               policy="transition-window-tuned")

    def totals(m):
        return {"all": int(m.sum()), "heldout": int(m[HELDOUT].sum()),
                "train": int(m[TRAIN].sum())}

    fidelity = {
        "captured": totals(capt), "control_replay": totals(ctl),
        "control_vs_captured_pct": {
            "all": 100 * (ctl.sum() - capt.sum()) / capt.sum(),
            "heldout": 100 * (ctl[HELDOUT].sum() - capt[HELDOUT].sum())
            / capt[HELDOUT].sum()},
        "heldout_cell_exact_match": int((ctl[HELDOUT] == capt[HELDOUT]).sum()),
        "heldout_cells": int(ctl[HELDOUT].size),
        "control_tuned_replay": totals(ctl_tuned),
        "note": "control = shipped transition-window replayed from persistent[0]; "
                "tuned row shows transition-window-tuned matches captured reads "
                "worse, confirming the captured run used the default weights."}

    ctl_ho_layer = ctl[HELDOUT].sum(0)          # (36,)
    ctl_ho = int(ctl[HELDOUT].sum())
    ctl_tr = int(ctl[TRAIN].sum())
    ctl_all = int(ctl.sum())
    variants = []

    def evaluate(name, m):
        d_layer = m[HELDOUT].sum(0) - ctl_ho_layer
        rec = {"name": name,
               "reads_all": int(m.sum()), "reads_heldout": int(m[HELDOUT].sum()),
               "pct_all": 100 * (m.sum() - ctl_all) / ctl_all,
               "pct_train": 100 * (m[TRAIN].sum() - ctl_tr) / ctl_tr,
               "pct_heldout": 100 * (m[HELDOUT].sum() - ctl_ho) / ctl_ho}
        rec.update(_sign_test(d_layer))
        rec.update(_bootstrap(d_layer.astype(float)))
        return rec

    # fourth-term grid over g-kind x EMA x w_gate; victim-only and pure need no
    # w_gate (victim ranks the eviction candidate by g alone; pure ranking is
    # scale-invariant), so each runs once per g-config.
    for (kind, ema), g in g_all.items():
        for wg in W_GATE_GRID:
            variants.append(evaluate(
                f"fourth|{kind}|ema={ema}|w={wg}",
                replay_capture(cap, routes, mode="fourth_term", g=g, w_gate=wg)))
        variants.append(evaluate(
            f"victim|{kind}|ema={ema}",
            replay_capture(cap, routes, mode="victim_only", g=g)))
        variants.append(evaluate(
            f"pure|{kind}|ema={ema}",
            replay_capture(cap, routes, mode="pure", g=g)))

    # HONEST protocol: fit the config on TRAIN (0-31), report ITS held-out result.
    # best_by_heldout selects on the test set and is an optimistic upper reference.
    best_train = min(variants, key=lambda v: v["pct_train"])
    best_ho = min(variants, key=lambda v: v["pct_heldout"])
    return {"shipped_weights": {"transition-window": [W_PRED, W_FREQ, W_REC, WINDOW],
                                "transition-window-tuned":
                                [W_PRED_TUNED, W_FREQ_TUNED, W_REC_TUNED, WINDOW_TUNED],
                                "captured_run_policy": "transition-window (default)"},
            "w_gate_grid": list(W_GATE_GRID), "g_kinds": list(G_KINDS),
            "ema_decays": list(EMA_DECAYS),
            "fidelity": fidelity, "control_heldout_reads": ctl_ho,
            "control_train_reads": ctl_tr, "control_all_reads": ctl_all,
            "variants": variants,
            "best_by_train_config": best_train,     # fit on train, held-out is the test
            "best_by_heldout": best_ho}             # selected on test = optimistic


# --------------------------------------------------------------------------- #
# Task A: transient + warm start on the 206-cycle trace @ 111+48 transition-window
# --------------------------------------------------------------------------- #
def _install_residents(bank, experts):
    bank._slot_to_expert = [None] * bank.persistent_slots
    bank._expert_to_slot = {}
    bank._pool_recency = {}
    bank._protected = set()
    for i, e in enumerate(int(x) for x in experts[:bank.persistent_slots]):
        bank._slot_to_expert[i] = e
        bank._expert_to_slot[e] = i
        bank._pool_clock += 1
        bank._pool_recency[e] = bank._pool_clock
        bank._history[e].last_used = 0
    return bank


def _replay_trace(banks, routes, cyc):
    mpc = np.zeros(cyc, int)
    for c in range(cyc):
        for L, b in banks.items():
            mpc[c] += len(set(b.plan(routes[L][c], phase="decode").misses))
    return mpc


def task_a(persist=111) -> dict:
    restore = runpy.run_path(str(osim.HELPER))["restore"]
    trace = osim.load_trace()
    cyc = trace["cycles"]
    layers = sorted(int(l) for l in trace["target_routes_by_layer"])
    routes = {L: trace["target_routes_by_layer"][str(L)] for L in layers}
    snap = {L: trace["initial_banks"][str(L)] for L in layers}

    def fresh_banks():
        return {L: osim.make_bank(restore, snap[L], persist, 48, "transition-window")
                for L in layers}

    base_mpc = _replay_trace(fresh_banks(), routes, cyc)
    steady = float(base_mpc[100:].mean())       # cycles 100..205 steady state
    decade = {f"{lo}-{lo+9}": float(base_mpc[lo:lo + 10].mean())
              for lo in range(0, 100, 10)}
    decade["steady(100-205)"] = steady
    excess_rec = float(np.maximum(base_mpc - steady, 0).sum())

    # oracle warm start: initial residents = 111 most-used experts in cycles 0..N
    def most_used(L, upto):
        from collections import Counter
        cnt = Counter()
        for c in range(min(upto, cyc)):
            cnt.update(set(routes[L][c]))
        return [e for e, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]

    oracle = {}
    for N in (10, 30, cyc):
        bks = fresh_banks()
        for L in layers:
            _install_residents(bks[L], most_used(L, N))
        tot = int(_replay_trace(bks, routes, cyc).sum())
        oracle[f"N={'all' if N == cyc else N}"] = tot

    # causal warm start: initial residents = top-111 by prefill route frequency
    bks = fresh_banks()
    pf_available = all(bool(snap[L].get("_prefill_route_freq")) for L in layers)
    for L in layers:
        pf = {int(k): v for k, v in snap[L]["_prefill_route_freq"].items()}
        ranked = [e for e, _ in sorted(pf.items(), key=lambda kv: (-kv[1], kv[0]))]
        _install_residents(bks[L], ranked)
    causal_tot = int(_replay_trace(bks, routes, cyc).sum())

    base_tot = int(base_mpc.sum())
    return {"policy": "transition-window", "capacity": f"{persist}+48",
            "cycles": cyc, "record_seconds": REC_S,
            "baseline_total_records": base_tot,
            "f1_control_anchor": "f1-overlap-sim @111+48 = 31,636 records",
            "misses_per_cycle_decade_mean": decade,
            "steady_state_mean_per_cycle": steady,
            "transient_excess_over_steady": {
                "records": excess_rec, "seconds": excess_rec * REC_S},
            "oracle_warm_start_totals": oracle,
            "oracle_warm_start_removed": {k: base_tot - v for k, v in oracle.items()},
            "causal_warm_start": {
                "prefill_route_freq_available": pf_available,
                "total_records": causal_tot,
                "removed_vs_baseline": base_tot - causal_tot,
                "note": "initial residents = top-111 by the policy's own "
                        "_prefill_route_freq (boundary-available; causal). The "
                        "baseline snapshot holds only 73 residents (grown to 111 "
                        "with 38 empty), so this pre-fills the extra slots with "
                        "prompt-frequent experts."}}


# --------------------------------------------------------------------------- #
# Task D: host microseconds per layer call -- shipped scoring vs best variant
# --------------------------------------------------------------------------- #
def _warm_bank(policy="transition-window", mode="control", g=None, w_gate=0.0):
    rng = np.random.default_rng(7)
    b = ScoreAwareBank(expert_count=NE, persistent_slots=CAP_PERSIST,
                       transient_slots=CAP_TRANSIENT, cache_policy=policy,
                       single_pool=True)
    b._mode, b._w_gate = mode, w_gate
    resid = rng.choice(NE, CAP_PERSIST, replace=False)
    for i, e in enumerate(int(x) for x in resid):
        b._slot_to_expert[i] = e
        b._expert_to_slot[e] = i
        b._pool_clock += 1
        b._pool_recency[e] = b._pool_clock
        b._history[e].last_used = 0
    b._decode_epoch = 0
    for _ in range(WINDOW + 4):                 # warm the transition window
        b.plan([int(x) for x in rng.choice(NE, 28, replace=False)], phase="decode")
    if g is not None:
        b._gate_g = g
    return b, rng


def task_d(best_mode, best_g_vec, best_w) -> dict:
    iters = 4000
    b0, _ = _warm_bank()
    b1, _ = _warm_bank(mode=best_mode, g=best_g_vec, w_gate=best_w)
    rng = np.random.default_rng(11)
    prevs = [tuple(int(x) for x in rng.choice(NE, 24, replace=False))
             for _ in range(iters)]

    def bench(bank, with_g):
        # retention scoring path only: score vector + admission cut. misses must be
        # genuinely non-resident and disjoint from pinned (route hits are residents).
        misses = [e for e in range(NE) if e not in bank._expert_to_slot][:4]
        pinned = set(list(bank._expert_to_slot)[:6])
        t0 = time.perf_counter()
        for i in range(iters):
            bank._transition_previous = prevs[i]
            if with_g:                          # gate hands a [384] vector; host adds it
                bank._gate_g = best_g_vec
            bank._transition_window_admissions(misses, pinned=pinned)
        return (time.perf_counter() - t0) / iters * 1e6

    shipped_us = bench(b0, False)
    variant_us = bench(b1, best_mode != "control")
    # g-vector maintenance cost alone: max-over-rows of [6,384] + normalize (+EMA)
    rows = rng.standard_normal((6, NE)).astype(np.float32)
    prev = np.zeros(NE, np.float32)
    t0 = time.perf_counter()
    for _ in range(iters):
        a = rows.max(0)
        gg = a / max(float(a.max()), 1e-9)
        prev = np.float32(0.7) * prev + np.float32(0.3) * gg
    g_build_us = (time.perf_counter() - t0) / iters * 1e6
    return {"iters": iters, "best_variant_mode": best_mode, "best_w_gate": best_w,
            "shipped_scoring_us_per_call": shipped_us,
            "variant_scoring_us_per_call": variant_us,
            "g_build_us_per_call": g_build_us,
            "note": "shipped scoring = _transition_window_scores (iterates 384 "
                    "_ExpertHistory objects for last_used) + _transition_window_"
                    "admissions (sorts ~115 candidates with 4-tuple keys). Variant "
                    "adds one [384] multiply-add; g_build is the max-over-rows + "
                    "normalize (+EMA) the device would hand over pre-computed."}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(
        "docs/deepseek-v41/receipts/f7-cache-policy-20260919/results.json"))
    ap.add_argument("--skip-a", action="store_true", help="skip the 206-cycle trace arm")
    args = ap.parse_args()
    if any(m == "mlx" or m.startswith("mlx.") for m in sys.modules):
        raise SystemExit("MLX leaked into the process")

    t0 = time.perf_counter()
    cap = rrc.load_capture()
    routes = routes_from_capture(cap)
    g_all = build_g(cap, feature=1)

    res_b = task_b(cap, routes, g_all)
    res_c = task_c(cap, routes)

    # Task D costs the honestly-selected best variant (fit on train 0-31).
    pick = res_b["best_by_train_config"]
    mode, kind, ema, w = "control", "max", None, 0.0
    if pick and pick["name"] != "control":
        parts = pick["name"].split("|")
        mode = {"fourth": "fourth_term", "victim": "victim_only",
                "pure": "pure"}[parts[0]]
        kind = parts[1]
        ema = None if parts[2] == "ema=None" else float(parts[2].split("=")[1])
        w = float(parts[3].split("=")[1]) if len(parts) > 3 else 0.0
    best_g_vec = g_all[(kind, ema)][10][40]      # a representative [384] g for cost
    res_d = task_d(mode, best_g_vec, w)

    res_a = None if args.skip_a else task_a()

    result = {
        "purpose": "F7 score-aware cache-policy screen; CPU-only, numpy-only, no GPU, "
                   "no service, no production code changed. Cache policy never alters "
                   "a routed expert's output, so any policy is bit-exact.",
        "source_commit": osim._git_head(),
        "elapsed_s": time.perf_counter() - t0,
        "mlx_imported": any(m == "mlx" or m.startswith("mlx.") for m in sys.modules),
        "proxy_limitation": "g is built from the PREDICTED 1-layer-ahead gate tensor "
                            "(layer T's gate on layer T-1's input; top-6 recall ~72%), "
                            "a NOISY PROXY for layer T's true scores. A true-score "
                            "signal can only do better; every gain here is a lower bound.",
        "inputs": {
            "capture_npz": cap["path"], "capture_sha256": cap["sha256"],
            "trace": str(osim.TRACE), "trace_sha256": osim.TRACE_SHA256,
            "expert_streaming_sha256":
                hashlib.sha256(Path("mtplx/expert_streaming.py").read_bytes()).hexdigest(),
            "record_bytes": REC_BYTES, "rate_gbps": RATE, "record_seconds": REC_S,
            "capacity": f"{CAP_PERSIST}+{CAP_TRANSIENT}"},
        "task_b_score_aware_retention": res_b,
        "task_c_clairvoyant_headroom": res_c,
        "task_d_host_cost": res_d,
        "task_a_transient_warm_start": res_a,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"wrote {args.out} in {result['elapsed_s']:.1f}s "
          f"(mlx_imported={result['mlx_imported']})")
    b = res_b
    print(f"[B] control heldout reads={b['control_heldout_reads']} "
          f"(captured 4878; replay {b['fidelity']['control_vs_captured_pct']['heldout']:+.2f}%)")
    bt = b["best_by_train_config"]
    print(f"[B] best-by-TRAIN (honest): {bt['name']} -> heldout "
          f"{bt['pct_heldout']:+.2f}% ({bt['layers_improved']}/36 improved, "
          f"p={bt['sign_test_p_two_sided']:.3g}); train {bt['pct_train']:+.2f}%")
    bh = b["best_by_heldout"]
    print(f"[B] best-by-heldout (optimistic): {bh['name']} "
          f"{bh['pct_heldout']:+.2f}% heldout ({bh['layers_improved']}/36)")
    print(f"[C] clairvoyant heldout: bypass={res_c['heldout']['bypass_reads']} "
          f"mandatory={res_c['heldout']['mandatory_reads']} (control {b['control_heldout_reads']})")
    print(f"[D] shipped={res_d['shipped_scoring_us_per_call']:.2f}us "
          f"variant={res_d['variant_scoring_us_per_call']:.2f}us "
          f"g_build={res_d['g_build_us_per_call']:.2f}us")
    if res_a:
        print(f"[A] baseline={res_a['baseline_total_records']} records; "
              f"transient excess={res_a['transient_excess_over_steady']['records']:.0f} "
              f"({res_a['transient_excess_over_steady']['seconds']:.2f}s); "
              f"causal warm start removes {res_a['causal_warm_start']['removed_vs_baseline']}")


if __name__ == "__main__":
    main()
