#!/usr/bin/env python3
"""W24 routing-locality census for DeepSeek-V4.1-Flash streaming (CPU).

Runs the standardized ``mtplx.prefill_bench`` programming prompt (default 1,024
context tokens, BOS prepended) through the real streaming loader path on the MLX
CPU device, then greedily decodes ``--decode-tokens`` (default 64) tokens.  A
non-invasive class-level wrapper on the MoE ``Gate`` records, for every
(phase, step, layer), the routed top-6 expert ids the gate selects (exactly the
ids that drive the ``switch_mlp`` record reads).  From that ordered trace it
computes the decode-locality census that decides which cache/IO levers are worth
building:

  * unique routed experts per layer over the decode window;
  * reuse-distance (LRU stack-distance) distribution per layer;
  * per-layer LRU + Belady(optimal-eviction) hit rate at each slot budget
    (default 115 / 205 / 384), cold (decode-only) and warm (prefill-primed);
  * the infinite-cache oracle hit-rate ceiling;
  * theoretical SSD bytes/token at each budget (record size from the manifest);
  * a held-out trained-quota gate -- does per-layer FREQUENCY slot allocation
    beat uniform on an unseen eval window?  Reports cross-layer coverage stdev
    (hy3-q4's was .0274 => the lever was DEAD; only claim it for DSV4.1 if the
    variance and the held-out eval both show real cross-layer skew);
  * an MTP verify-window dedup projection: union of the routed sets across W
    consecutive decode tokens (W in --mtp-widths) -> the bytes MTP amortizes.

MEMORY / CONCURRENCY CONTRACT (David 2026-09-11; six workers share a 100 GB box;
a guard kills any worker python above 14 GB):

  * CPU only (``mx.set_default_device(mx.cpu)`` before any forward).
  * The single real-artifact load runs ONLY while holding
    ``/tmp/dsv41-cpu-model-load.lock`` (fcntl LOCK_EX); this module takes the
    lock itself (pass ``--no-self-lock`` when an outer wrapper already holds it).
  * ``memory_limit`` <= 12 GiB, expert cache <= 2 GiB (default 1.5), text-only
    residents (MTP head + vision are never touched on the AR path so their mmap
    pages stay unresident), ``apply_memory_cap`` on; an in-process RSS watchdog
    thread aborts the run at ``--rss-abort-gib`` (default 12).
  * The census JSON is written to ``--out`` at the end (append-only receipt); a
    partial ``<out>.trace-progress.json`` is refreshed during the forward so a
    kill leaves evidence of how far it got.

No GPU work at import; ``--help`` is CPU-safe.  The pure analysis functions
(``simulate_lru``, ``simulate_belady``, ``reuse_distances``,
``allocate_frequency_quota``, ``held_out_alloc_gate``, ``mtp_union_projection``)
take plain int sequences and are unit-tested in
``tests/test_deepseek_v41_decode_levers.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DEFAULT_MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()
GIB = 1024 ** 3
DEFAULT_SLOT_BUDGETS = (115, 205, 384)
# mxfp4 gs32 routed-expert record size (dsv41-native-mxfp4-bank.md: 18.80 MB /
# record, 269 GiB / 40x384).  Overridable with --record-bytes; the census also
# tries to read the true size off the loaded manifest.
FALLBACK_RECORD_BYTES = 18_800_000


# --------------------------------------------------------------------------- #
# Pure analysis (no MLX, no I/O) -- unit-tested on synthetic sequences.
# --------------------------------------------------------------------------- #
def simulate_lru(measure: Sequence[int], capacity: int,
                 warm: Sequence[int] = ()) -> Tuple[int, int]:
    """LRU cache of ``capacity`` slots.  Run ``warm`` (uncounted) then ``measure``
    (counted).  Returns (hits, misses) over ``measure`` only."""
    if capacity <= 0:
        return 0, len(measure)
    cache: "OrderedDict[int, bool]" = OrderedDict()
    for e in warm:
        if e in cache:
            cache.move_to_end(e)
        else:
            cache[e] = True
            if len(cache) > capacity:
                cache.popitem(last=False)
    hits = misses = 0
    for e in measure:
        if e in cache:
            hits += 1
            cache.move_to_end(e)
        else:
            misses += 1
            cache[e] = True
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits, misses


def simulate_belady(measure: Sequence[int], capacity: int,
                    warm: Sequence[int] = ()) -> Tuple[int, int]:
    """Belady MIN (optimal eviction): evict the resident expert whose next use is
    farthest in the future.  ``warm`` primes the cache (uncounted), ``measure`` is
    counted.  Deterministic; O(len * capacity)."""
    if capacity <= 0:
        return 0, len(measure)
    seq = list(warm) + list(measure)
    start = len(warm)
    # next-occurrence linked list: nxt[i] = index of the next access to the same
    # expert after i, or a large sentinel.
    INF = len(seq) + 1
    last_seen: Dict[int, int] = {}
    nxt = [INF] * len(seq)
    for i in range(len(seq) - 1, -1, -1):
        e = seq[i]
        nxt[i] = last_seen.get(e, INF)
        last_seen[e] = i
    cache: Dict[int, int] = {}  # expert -> its current position's next-use index
    hits = misses = 0
    for i, e in enumerate(seq):
        counted = i >= start
        if e in cache:
            if counted:
                hits += 1
            cache[e] = nxt[i]
        else:
            if counted:
                misses += 1
            if len(cache) >= capacity:
                # evict the resident whose next use is farthest.
                victim = max(cache, key=lambda k: cache[k])
                del cache[victim]
            cache[e] = nxt[i]
    return hits, misses


def reuse_distances(seq: Sequence[int]) -> List[int]:
    """LRU stack distance per access: number of DISTINCT experts seen since this
    expert's previous access (-1 == first access / cold).  A distance d means an
    LRU cache of > d slots would hit."""
    last_index: Dict[int, int] = {}
    out: List[int] = []
    for i, e in enumerate(seq):
        if e not in last_index:
            out.append(-1)
        else:
            window = seq[last_index[e] + 1:i]
            out.append(len(set(window)))
        last_index[e] = i
    return out


def allocate_frequency_quota(train_counts: Dict[int, Dict[int, int]],
                             total_slots: int, n_layers: int) -> Dict[int, int]:
    """Greedy water-filling allocation of ``total_slots`` across layers using the
    train-window per-(layer,expert) counts.  Each next slot goes to the layer
    where caching one more (next-most-popular) expert removes the most predicted
    future accesses (a stationary-frequency proxy for miss reduction).  Every
    layer is guaranteed >= 1 slot.  Returns {layer: slots}."""
    layers = sorted(train_counts)
    if not layers:
        return {}
    # popularity-sorted counts per layer
    ranked = {
        L: sorted(train_counts[L].values(), reverse=True) for L in layers
    }
    alloc = {L: 0 for L in layers}
    remaining = total_slots
    # guarantee one slot each first
    for L in layers:
        if remaining <= 0:
            break
        alloc[L] = 1
        remaining -= 1
    # marginal gain of the NEXT slot for layer L = count of the (alloc[L])-th
    # most popular expert (0 if none left / already fully allocated).
    import heapq
    heap = []
    for L in layers:
        r = ranked[L]
        gain = r[alloc[L]] if alloc[L] < len(r) else 0
        heapq.heappush(heap, (-gain, L))
    while remaining > 0 and heap:
        neg_gain, L = heapq.heappop(heap)
        if -neg_gain <= 0:
            break  # no layer has any further useful slot
        alloc[L] += 1
        remaining -= 1
        r = ranked[L]
        gain = r[alloc[L]] if alloc[L] < len(r) else 0
        heapq.heappush(heap, (-gain, L))
    return alloc


def held_out_alloc_gate(train_seq_by_layer: Dict[int, List[int]],
                        eval_seq_by_layer: Dict[int, List[int]],
                        total_slots: int,
                        warm_by_layer: Dict[int, List[int]] | None = None
                        ) -> dict:
    """Does per-layer FREQUENCY slot allocation beat UNIFORM on an unseen eval
    window?  Train the quota on ``train_seq_by_layer`` frequencies, then measure
    LRU miss/token on ``eval_seq_by_layer`` under (a) uniform B=total/n and
    (b) the frequency quota.  This is the hy3-q4 kill-test ported to DSV4.1."""
    layers = sorted(eval_seq_by_layer)
    n = len(layers)
    warm_by_layer = warm_by_layer or {}
    train_counts = {
        L: {e: seq.count(e) for e in set(seq)}
        for L, seq in train_seq_by_layer.items()
    }
    uniform_b = max(1, total_slots // n)
    freq_alloc = allocate_frequency_quota(train_counts, total_slots, n)

    def total_misses(alloc_fn) -> int:
        m = 0
        for L in layers:
            b = alloc_fn(L)
            _, miss = simulate_lru(eval_seq_by_layer[L], b,
                                   warm=warm_by_layer.get(L, ()))
            m += miss
        return m

    uni_miss = total_misses(lambda L: uniform_b)
    freq_miss = total_misses(lambda L: max(1, freq_alloc.get(L, uniform_b)))
    eval_accesses = sum(len(eval_seq_by_layer[L]) for L in layers)
    return {
        "total_slots": total_slots,
        "n_layers": n,
        "uniform_slots_per_layer": uniform_b,
        "uniform_miss": uni_miss,
        "frequency_miss": freq_miss,
        "eval_accesses": eval_accesses,
        "uniform_miss_rate": uni_miss / eval_accesses if eval_accesses else None,
        "frequency_miss_rate": freq_miss / eval_accesses if eval_accesses else None,
        "frequency_relative_miss_reduction": (
            (uni_miss - freq_miss) / uni_miss if uni_miss else 0.0
        ),
        "frequency_alloc_min": min(freq_alloc.values()) if freq_alloc else None,
        "frequency_alloc_max": max(freq_alloc.values()) if freq_alloc else None,
    }


def mtp_union_projection(decode_steps_by_layer: Dict[int, List[List[int]]],
                         widths: Sequence[int]) -> dict:
    """For each verify-window width W, group the decode steps into consecutive
    non-overlapping windows and, per layer, take the union of the routed-expert
    sets across the W tokens.  Returns, per W: mean union records per window per
    layer, vs the naive W*top_k, and the resulting UNIQUE bytes-per-accepted-token
    dedup factor (assuming every drafted token in the window is accepted -- an
    upper bound on the amortization)."""
    layers = sorted(decode_steps_by_layer)
    out: Dict[int, dict] = {}
    for W in widths:
        union_sizes: List[int] = []
        naive_sizes: List[int] = []
        for L in layers:
            steps = decode_steps_by_layer[L]
            for i in range(0, len(steps) - (len(steps) % W), W):
                window = steps[i:i + W]
                u = set()
                naive = 0
                for s in window:
                    u.update(s)
                    naive += len(s)
                union_sizes.append(len(u))
                naive_sizes.append(naive)
        if not union_sizes:
            continue
        mean_union = statistics.mean(union_sizes)
        mean_naive = statistics.mean(naive_sizes)
        out[W] = {
            "windows": len(union_sizes),
            "mean_union_records_per_layer": mean_union,
            "mean_naive_records_per_layer": mean_naive,
            "dedup_factor_vs_naive": mean_naive / mean_union if mean_union else None,
            # per ACCEPTED token (W tokens per window): union bytes / W.
            "unique_records_per_accepted_token": mean_union / W,
            "read_amortization_vs_ar": (mean_naive / W) / (mean_union / W)
            if mean_union else None,
        }
    return out


# --------------------------------------------------------------------------- #
# Trace collection helpers.
# --------------------------------------------------------------------------- #
def _flatten_layer_seq(decode_steps_by_layer: Dict[int, List[List[int]]],
                       ) -> Dict[int, List[int]]:
    """Flatten per-step [top_k] sets into one access sequence per layer."""
    return {
        L: [e for step in steps for e in step]
        for L, steps in decode_steps_by_layer.items()
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--context-tokens", type=int, default=1024, choices=(1024, 16384))
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--prompt-format", default="raw", choices=("raw", "chat"))
    p.add_argument("--bos-id", type=int, default=0)
    p.add_argument("--no-bos", dest="bos", action="store_false", default=True)
    p.add_argument("--slot-budgets", type=int, nargs="+", default=list(DEFAULT_SLOT_BUDGETS))
    p.add_argument("--mtp-widths", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--record-bytes", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--slot-layout", default="component-banks")
    p.add_argument("--memory-limit-gib", type=float, default=12.0)
    p.add_argument("--expert-cache-limit-gib", type=float, default=1.5)
    # The runtime's fixed-footprint PLAN counts the full resident manifest
    # (incl. the 15.3 GiB MTP experts) against memory_limit, so it rejects at 12
    # GiB even though the AR path only touches ~9 GiB of text residents via lazy
    # mmap.  On CPU the plan cap is a GPU-wired-budget guard, not RSS -- the CPU
    # probes (decode_probe.py / dump_hidden_states.py --cpu) turn it off and rely
    # on lazy mmap + a capped cache.  Real RSS safety here is the watchdog below.
    p.add_argument("--apply-memory-cap", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--max-kv", type=int, default=4096)
    p.add_argument("--rss-abort-gib", type=float, default=12.0)
    p.add_argument("--self-lock", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lock-path", type=Path, default=Path("/tmp/dsv41-cpu-model-load.lock"))
    p.add_argument("--seed", type=int, default=0)
    return p


def _rss_gib() -> float:
    """Best-effort resident set size of this process, in GiB."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KiB.
        if sys.platform == "darwin":
            return peak / GIB
        return peak * 1024 / GIB
    except Exception:
        return 0.0


def _start_rss_watchdog(limit_gib: float, log) -> threading.Event:
    stop = threading.Event()

    def _poll():
        while not stop.wait(2.0):
            rss = _rss_gib()
            if rss >= limit_gib:
                log(f"[watchdog] RSS {rss:.2f} GiB >= {limit_gib} GiB -- aborting")
                os._exit(3)
    t = threading.Thread(target=_poll, name="rss-watchdog", daemon=True)
    t.start()
    return stop


def _record_bytes(runtime, override: int | None, log) -> int:
    if override:
        log(f"[census] record_bytes = {override} (cli override)")
        return override
    manifest = getattr(runtime, "manifest", None)
    for attr in ("record_bytes", "record_size", "record_stride", "bytes_per_record"):
        v = getattr(manifest, attr, None)
        if isinstance(v, int) and v > 0:
            log(f"[census] record_bytes = {v} (manifest.{attr})")
            return v
    log(f"[census] record_bytes = {FALLBACK_RECORD_BYTES} (fallback mxfp4 gs32)")
    return FALLBACK_RECORD_BYTES


def run_census(args, log) -> dict:
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    mx.random.seed(int(args.seed))

    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
    from mtplx.models.deepseek_v41_moe import Gate
    from mlx_lm.utils import load_tokenizer

    t0 = time.time()
    log("[census] loading model (CPU, capped) ...")
    resident = load_deepseek_v41_streaming(
        args.model,
        memory_limit_bytes=int(args.memory_limit_gib * GIB),
        max_live_kv_tokens=int(args.max_kv),
        admit=True,
        admission_receipt=None,
        expert_cache_limit_bytes=int(args.expert_cache_limit_gib * GIB),
        apply_memory_cap=True,
        slot_layout=args.slot_layout,
        cache_scope="layer",
        island_layers=(),
        verify_record_hashes=False,
    )
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime")
    log(f"[census] loaded in {time.time() - t0:.1f}s  RSS={_rss_gib():.2f} GiB")

    record_bytes = _record_bytes(runtime, args.record_bytes, log)
    n_layers = len(model.model.layers)

    tokenizer = load_tokenizer(Path(args.model))
    from mtplx.prefill_bench import _prompt_build_for_context
    pb = _prompt_build_for_context(tokenizer, int(args.context_tokens),
                                   prompt_format=args.prompt_format)
    prompt_ids = list(pb.token_ids)
    prompt_meta = dict(pb.metadata)
    if args.bos:
        prompt_ids = [int(args.bos_id)] + prompt_ids
    prompt_meta["bos_prepended"] = bool(args.bos)
    prompt_meta["input_tokens"] = len(prompt_ids)
    log(f"[census] prompt {len(prompt_ids)} tokens (context={args.context_tokens}, "
        f"bos={args.bos})")

    # -- switch wrapper: capture the gate's routed ids per (phase, step, layer) --
    ctx = {"phase": "prefill", "step": 0}
    # prefill_counts[layer][expert] -> count ; decode_steps[layer] -> [ [ids]... ]
    prefill_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    decode_steps: Dict[int, List[List[int]]] = defaultdict(list)

    orig_gate_call = Gate.__call__

    def gate_call(self, x, image_mask=None):
        weights, indices = orig_gate_call(self, x, image_mask)
        try:
            ids = indices.tolist()  # [n_tokens, top_k]
            L = int(self.layer_id)
            if ctx["phase"] == "prefill":
                lc = prefill_counts[L]
                for row in ids:
                    for e in row:
                        lc[int(e)] += 1
            else:
                # decode: one token per forward -> ids is [1, top_k]
                decode_steps[L].append([int(e) for e in ids[-1]])
        except Exception as exc:  # never break the forward
            log(f"[census] capture error L={getattr(self,'layer_id','?')}: {exc!r}")
        return weights, indices

    progress_path = Path(str(args.out) + ".trace-progress.json")

    def _refresh_progress():
        try:
            progress_path.write_text(json.dumps({
                "phase": ctx["phase"], "step": ctx["step"],
                "decode_tokens_done": len(next(iter(decode_steps.values()), [])),
                "rss_gib": _rss_gib(), "elapsed_s": time.time() - t0,
            }))
        except Exception:
            pass

    Gate.__call__ = gate_call
    prefill_s = decode_s = 0.0
    ttft_s = None
    decoded_ids: List[int] = []
    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
        # -- prefill --
        ctx["phase"] = "prefill"; ctx["step"] = 0
        tp = time.time()
        logits = model(mx.array([prompt_ids]), cache=cache)
        mx.eval(logits)
        prefill_s = time.time() - tp
        ttft_s = prefill_s
        token = int(mx.argmax(logits[0, -1]).item())
        decoded_ids.append(token)
        log(f"[census] prefill {len(prompt_ids)} tok in {prefill_s:.1f}s "
            f"({len(prompt_ids)/prefill_s:.1f} tok/s)  RSS={_rss_gib():.2f} GiB")
        _refresh_progress()
        # -- decode --
        ctx["phase"] = "decode"
        td = time.time()
        for s in range(int(args.decode_tokens)):
            ctx["step"] = s
            logits = model(mx.array([[token]]), cache=cache)
            token = int(mx.argmax(logits[0, -1]).item())
            decoded_ids.append(token)
            if s % 8 == 0 or s == int(args.decode_tokens) - 1:
                el = time.time() - td
                log(f"[census] decode step {s+1}/{args.decode_tokens} "
                    f"{(s+1)/el:.2f} tok/s  RSS={_rss_gib():.2f} GiB")
                _refresh_progress()
        decode_s = time.time() - td
        log(f"[census] decode {args.decode_tokens} tok in {decode_s:.1f}s "
            f"({args.decode_tokens/decode_s:.2f} tok/s)")
    finally:
        Gate.__call__ = orig_gate_call

    # -- real runtime telemetry (shipped-path ACTUAL io + cache behaviour) -----
    # Decisive for lever choice: snapshot()["slots"]["io"]["read_bytes"] is the
    # true bytes pulled off SSD, ["cache"]["hit_rate"] the real hit rate, and
    # ["slots"]["metrics"] whether reads batched/deduped.  cache_by_phase isolates
    # the decode phase.
    rt_tel: dict = {}
    try:
        snap = runtime.snapshot()
        io = (snap.get("slots") or {}).get("io") or {}
        cache = snap.get("cache") or {}
        rt_tel = {
            "cache_aggregate": {
                "hit_rate": cache.get("hit_rate"),
                "expert_hits": cache.get("expert_hits"),
                "expert_misses": cache.get("expert_misses"),
                "evictions": cache.get("evictions"),
                "bytes_read_logical": cache.get("bytes_read"),
            },
            "cache_by_phase": snap.get("cache_by_phase"),
            "io": {
                "read_bytes_ssd": io.get("read_bytes"),
                "requested_bytes": io.get("requested_bytes"),
                "read_mib_per_second": io.get("read_mib_per_second"),
                "python_preadv_invocations": io.get("python_preadv_invocations"),
                "native_positional_calls": io.get("native_positional_calls"),
                "preadv_bytes_returned": io.get("preadv_bytes_returned"),
                "bytes_read_saved": io.get("bytes_read_saved"),
                "short_reads": io.get("short_reads"),
            },
            "slot_metrics": (snap.get("slots") or {}).get("metrics"),
            "config_echo": {
                "cache_policy": getattr(getattr(runtime, "config", None), "cache_policy", None),
                "cache_scope": getattr(getattr(runtime, "config", None), "cache_scope", None),
                "frequency_decay": getattr(getattr(runtime, "config", None), "frequency_decay", None),
                "overlap_miss_reads": getattr(getattr(runtime, "config", None), "overlap_miss_reads", None),
                "prefetch_slots": getattr(getattr(runtime, "config", None), "prefetch_slots", None),
                "max_inflight_io_bytes": getattr(getattr(runtime, "config", None), "max_inflight_io_bytes", None),
                "max_read_chunk_bytes": getattr(getattr(runtime, "config", None), "max_read_chunk_bytes", None),
                "slots_per_layer": getattr(getattr(runtime, "plan", None), "slots_per_layer", None),
                "transient_slots": getattr(getattr(runtime, "plan", None), "transient_slots", None),
            },
        }
    except Exception as exc:
        rt_tel = {"error": repr(exc)}

    # -- analysis -------------------------------------------------------------
    layers = sorted(decode_steps)
    decode_flat = _flatten_layer_seq(decode_steps)
    prefill_flat: Dict[int, List[int]] = {}
    # reconstruct a prefill access order proxy (frequency-expanded, popularity
    # order) for warm-priming -- exact prefill order is lost to counts, but LRU
    # warm-priming only needs the *resident set* the prefill leaves, which for
    # capacity>=distinct is the whole set and for capacity<distinct we prime with
    # the most-popular (best case) then measure decode.
    for L in layers:
        pc = prefill_counts.get(L, {})
        prefill_flat[L] = [e for e, _ in sorted(pc.items(), key=lambda kv: -kv[1])]

    per_layer = {}
    budgets = list(args.slot_budgets)
    for L in layers:
        seq = decode_flat[L]
        distinct = sorted(set(seq))
        rd = reuse_distances(seq)
        cold = {}
        warm = {}
        for b in budgets:
            h_lru, m_lru = simulate_lru(seq, b)
            h_bel, m_bel = simulate_belady(seq, b)
            cold[str(b)] = {
                "lru_hit_rate": h_lru / len(seq) if seq else None,
                "belady_hit_rate": h_bel / len(seq) if seq else None,
                "lru_miss": m_lru, "belady_miss": m_bel,
            }
            hw_lru, mw_lru = simulate_lru(seq, b, warm=prefill_flat[L])
            warm[str(b)] = {
                "lru_hit_rate": hw_lru / len(seq) if seq else None,
                "lru_miss": mw_lru,
            }
        first_touch = sum(1 for d in rd if d == -1)
        per_layer[L] = {
            "accesses": len(seq),
            "distinct_experts": len(distinct),
            "prefill_distinct_experts": len(prefill_counts.get(L, {})),
            "oracle_infinite_hit_rate": (len(seq) - first_touch) / len(seq) if seq else None,
            "reuse_distance_median": (statistics.median([d for d in rd if d >= 0])
                                      if any(d >= 0 for d in rd) else None),
            "cold": cold,
            "warm_prefill_primed": warm,
        }

    # aggregate across layers (per-token = per decode step, all layers summed)
    n_dec = int(args.decode_tokens)
    top_k = len(decode_steps[layers[0]][0]) if layers and decode_steps[layers[0]] else 6
    agg = {}
    for b in budgets:
        cold_miss = sum(per_layer[L]["cold"][str(b)]["lru_miss"] for L in layers)
        cold_bel = sum(per_layer[L]["cold"][str(b)]["belady_miss"] for L in layers)
        warm_miss = sum(per_layer[L]["warm_prefill_primed"][str(b)]["lru_miss"] for L in layers)
        total_acc = sum(per_layer[L]["accesses"] for L in layers)
        agg[str(b)] = {
            "slots_per_layer": b,
            "resident_gib_if_all_layers": b * len(layers) * record_bytes / GIB,
            "cold_lru_hit_rate": 1 - cold_miss / total_acc if total_acc else None,
            "cold_belady_hit_rate": 1 - cold_bel / total_acc if total_acc else None,
            "warm_lru_hit_rate": 1 - warm_miss / total_acc if total_acc else None,
            "cold_lru_miss_bytes_per_token": cold_miss / n_dec * record_bytes,
            "warm_lru_miss_bytes_per_token": warm_miss / n_dec * record_bytes,
            "warm_lru_decode_tok_s_ceiling_ssd": (
                1.0 / ((warm_miss / n_dec * record_bytes) / (12.47 * GIB))
                if warm_miss else None
            ),
        }

    # held-out frequency-allocation gate: chronological split of the decode
    # window (train = first half, eval = second half); prime eval with prefill.
    half = n_dec // 2
    train_by_layer = {L: [e for step in decode_steps[L][:half] for e in step] for L in layers}
    eval_by_layer = {L: [e for step in decode_steps[L][half:] for e in step] for L in layers}
    warmp = {L: prefill_flat[L] for L in layers}
    alloc_gate = {}
    for b in budgets:
        total_slots = b * len(layers)
        alloc_gate[str(b)] = held_out_alloc_gate(
            train_by_layer, eval_by_layer, total_slots, warm_by_layer=warmp)

    coverage_stdev = None
    try:
        # cross-layer coverage stdev at the smallest budget (hy3 diagnostic).
        b0 = min(budgets)
        cov = []
        for L in layers:
            pc = prefill_counts.get(L, {})
            tot = sum(pc.values())
            if tot:
                topk = sorted(pc.values(), reverse=True)[:b0]
                cov.append(sum(topk) / tot)
        coverage_stdev = statistics.pstdev(cov) if len(cov) > 1 else None
    except Exception:
        pass

    mtp = mtp_union_projection(decode_steps, args.mtp_widths)

    census = {
        "meta": {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "worker": "W24",
            "device": "cpu",
            "model": str(args.model),
            "spec_key": getattr(getattr(runtime, "spec", None), "key", None),
            "n_layers": n_layers,
            "n_moe_layers": len(layers),
            "top_k": top_k,
            "record_bytes": record_bytes,
            "context_tokens": args.context_tokens,
            "decode_tokens": n_dec,
            "slot_budgets": budgets,
            "prompt_build": prompt_meta,
            "prompt_ids_head": prompt_ids[:12],
            "decoded_ids": decoded_ids,
            "decoded_text": tokenizer.decode(decoded_ids),
            "prefill_tok_s": len(prompt_ids) / prefill_s if prefill_s else None,
            "decode_tok_s_cpu": n_dec / decode_s if decode_s else None,
            "ttft_s_cpu": ttft_s,
            "peak_rss_gib": _rss_gib(),
        },
        "runtime_telemetry": rt_tel,
        "per_token_geometry": {
            "routed_records_per_token": top_k * len(layers),
            "bytes_per_token_all_miss": top_k * len(layers) * record_bytes,
            "gib_per_token_all_miss": top_k * len(layers) * record_bytes / GIB,
        },
        "aggregate_by_budget": agg,
        "held_out_frequency_allocation_gate": alloc_gate,
        "cross_layer_coverage_stdev": coverage_stdev,
        "mtp_verify_window_dedup": {str(k): v for k, v in mtp.items()},
        "per_layer": {str(L): per_layer[L] for L in layers},
    }
    try:
        runtime.close()
    except Exception:
        pass
    try:
        progress_path.unlink()
    except Exception:
        pass
    return census


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logf = open(str(args.out) + ".log", "a", buffering=1)

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")

    lock_fd = None
    if args.self_lock:
        import fcntl
        lock_fd = os.open(str(args.lock_path), os.O_CREAT | os.O_RDWR)
        log(f"[census] acquiring load lock {args.lock_path} ...")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        log("[census] lock acquired")

    _start_rss_watchdog(args.rss_abort_gib, log)
    try:
        census = run_census(args, log)
    finally:
        if lock_fd is not None:
            import fcntl
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            log("[census] lock released")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(census, indent=2))
    log(f"[census] wrote {args.out}")
    log(f"[census] peak RSS {census['meta']['peak_rss_gib']:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
