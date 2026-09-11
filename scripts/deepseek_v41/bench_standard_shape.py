#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash q2 streaming — David's standard-shape decode benchmark.

Runs the standardized single-prompt greedy benchmark shape
(memory/dsv41-standard-benchmark-shape.md, memory/follow-the-specific-setup.md)
through the real serve-path loader
(:func:`mtplx.models.deepseek_v41_loader.load_deepseek_v41_streaming`, streamed
experts + engram attached), for each requested prefill cell:

  * builds the deterministic ``mtplx.prefill_bench`` coding-agent programming
    prompt at ``--context-tokens`` with the reference BOS prepended, using the
    SAME importable ``build_prompt`` the gate and the hidden-state dump use
    (``scripts/deepseek_v41/dump_hidden_states.build_prompt``), so every receipt
    carries identical prompt-build metadata;
  * one greedy (argmax) generation of ``--steps`` decode tokens (default 256);
  * reports per cell: prefill tok/s, TTFT, decode tok/s, peak GB
    (``mx.get_peak_memory`` + process RSS), wall time, expert records gathered,
    engram rows gathered, and the first 200 chars of the decoded text.

The default cells are 1,024 (David's THE input) and 16,384 (the prefill cell);
both are run in one invocation. A greedy run is deterministic, so three seeds are
NOT needed (memory/humaneval-one-seed.md, memory/report-fastest-of-seeds.md);
``--repeats`` is provided for later speed windows and, when >1, records every
repeat plus the fastest-of summary (max decode tok/s / min TTFT, the range).

WHY THE COUNTERS ARE READ OFF THE HOT PATH
------------------------------------------
Expert-record and engram-row counts come from the runtime's own cheap counters
read *between* cells — ``runtime.snapshot()["cache"]`` (a ``CacheCounters``
dict) and each engram hook's ``row_cache.stats`` — never by wrapping
``switch_mlp`` on the decode path (that per-call ``indices.tolist()`` would
inflate the decode tok/s this bench exists to measure; the P1.7 gate wraps them
only because it needs the exact per-step selection, not a rate).

Receipts are APPEND-ONLY (memory/never-overwrite-a-measurement.md): one fresh
UTC-stamped directory per invocation under
``<out-dir>/bench_standard_shape/<utc-stamp>/`` (never reused), and the receipt
filename is suffixed with the context set / steps / seed. The writer refuses to
overwrite an existing receipt file.

RUN CONTEXT (orchestrator, on the GPU, INSIDE ``gpu_window.sh`` which holds the
exclusive lock — NOT the harness author): worker W1's
``mtplx/models/deepseek_v41.py`` must exist and the faithful port must pass its
CPU probe (the streamed prefill still had a defect at W8, see W8_REPORT.md); the
local manifest must carry the HF identity so admission passes (W3), or
``--no-admit --admission-receipt`` injects it. This module does NO GPU work at
import time — ``--help``, import, and ``--dry-run`` are CPU-safe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()
DEFAULT_OUT_DIR = Path(".benchmark-artifacts/deepseek-v41")
STEP = "bench_standard_shape"
DEFAULT_CONTEXT_CELLS = (1024, 16384)
CONTEXT_CHOICES = (1024, 16384)
DEFAULT_STEPS = 256
DEFAULT_BOS_ID = 0
GIB = 1024**3
_TEXT_PREVIEW_CHARS = 200


# --------------------------------------------------------------------------
# importable prompt helper (shared with the gate + the hidden-state dump)
# --------------------------------------------------------------------------


def _load_build_prompt():
    """Import ``build_prompt`` from the sibling ``dump_hidden_states.py``.

    ``scripts`` is not a package (no ``__init__.py``), so we load the module by
    file path. ``dump_hidden_states`` only imports the standard library at module
    scope (mlx/mtplx are deferred into functions), so this stays CPU-safe.
    """

    module_path = Path(__file__).resolve().parent / "dump_hidden_states.py"
    spec = importlib.util.spec_from_file_location(
        "dsv41_dump_hidden_states", module_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise ImportError(f"cannot load build_prompt from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_prompt


def _prompt_ids_override(path, *, context_tokens: int, seed=None, cell: str = "sweep"):
    """Load exact server token ids exported by server_cell_bench --prompt-ids-out.

    Selects the entry for (cell, target_tokens==context_tokens, seed) and returns
    ``(prompt_ids, prompt_meta)`` byte-identically to the served cell -- no
    builder, no BOS prepend (the exported ids already ARE what the server saw).
    Lets an in-process arm run on the SAME token ids as the served sweep cell.
    """

    data = json.loads(Path(path).read_text())
    prompts = data.get("prompts", [])
    matches = [
        e
        for e in prompts
        if str(e.get("cell")) == str(cell)
        and int(e.get("target_tokens") or 0) == int(context_tokens)
        and (seed is None or e.get("seed") == int(seed))
    ]
    if not matches:
        raise SystemExit(
            f"--prompt-ids-file {path} has no entry for cell={cell} "
            f"target_tokens={context_tokens} seed={seed}"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"--prompt-ids-file {path} is ambiguous for cell={cell} "
            f"target_tokens={context_tokens}: pass --prompt-seed "
            f"(seeds present: {sorted(e.get('seed') for e in matches)})"
        )
    entry = matches[0]
    ids = [int(t) for t in entry["token_ids"]]
    ids_sha = entry.get("token_ids_sha256") or hashlib.sha256(
        json.dumps(ids).encode("utf-8")
    ).hexdigest()
    meta = {
        "prompt_source": "prompt-ids-file",
        "prompt_ids_file": str(path),
        "prompt_ids_schema": data.get("schema"),
        "prompt_family": data.get("model_family"),
        "served_model_id": data.get("served_model_id"),
        "prompt_cell": entry.get("cell"),
        "prompt_target_tokens": entry.get("target_tokens"),
        "prompt_seed": entry.get("seed"),
        "prompt_text_sha256": entry.get("text_sha256"),
        "token_ids_sha256": ids_sha,
        "templated_tokens": entry.get("templated_tokens"),
        "input_tokens": len(ids),
        "bos_prepended": ids[:1] == [0] if ids else False,
        "bos_id": ids[0] if (ids and ids[:1] == [0]) else None,
        "prompt_release_valid": True,
    }
    return ids, meta


def _resolve_prompt(args, tokenizer, build_prompt, context_tokens: int):
    """Either the exported served ids (``--prompt-ids-file``) or the default
    prefill_bench builder. Default keeps old receipts comparable."""

    ids_file = getattr(args, "prompt_ids_file", None)
    if ids_file:
        return _prompt_ids_override(
            ids_file,
            context_tokens=int(context_tokens),
            seed=getattr(args, "prompt_seed", None),
        )
    return build_prompt(tokenizer, _prompt_args(args, int(context_tokens)))


def _prompt_args(args, context_tokens: int) -> argparse.Namespace:
    """The tiny namespace ``build_prompt(tokenizer, args)`` reads, per cell."""

    return argparse.Namespace(
        prompt=args.prompt,
        context_tokens=int(context_tokens),
        prompt_format=args.prompt_format,
        bos=args.bos,
        bos_id=args.bos_id,
        model=args.model,
    )


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--context-tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_CELLS),
        choices=CONTEXT_CHOICES,
        metavar="N",
        help="prefill cell size(s), one greedy generation each. Default: both "
        "1024 (David's THE input) and 16384 (the prefill cell).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="decode tokens generated per cell (default 256). The prefill "
        "produces the first token (TTFT); the decode loop then runs --steps "
        "forwards and decode tok/s = steps / decode wall.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=("ar", "dspark"),
        default="ar",
        help=(
            "Decode lane for the cell. 'ar' is the standard greedy target-only "
            "autoregression. 'dspark' additionally runs the DSpark-DIRECT "
            "speculative loop (W57), ASSERTS its greedy ids are byte-identical to "
            "the AR cell, and records tokens/cycle + accept-by-depth under "
            "the cell's 'dspark' key (the AR metrics stay the reported shape)."
        ),
    )
    parser.add_argument(
        "--dspark-depth",
        type=int,
        default=3,
        help="draft block width K per DSpark-DIRECT cycle (--decode-mode dspark)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeats per cell for later speed windows (default 1). >1 records "
        "every repeat plus the fastest-of summary.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="literal prompt text; overrides the prefill_bench builder "
        "(default None = build the deterministic prefill_bench prompt).",
    )
    parser.add_argument("--prompt-format", default="raw", choices=("raw", "chat"))
    parser.add_argument(
        "--bos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prepend the reference BOS (id --bos-id, default 0). The artifact "
        "tokenizer does not add it; the reference always does (W8_REPORT.md).",
    )
    parser.add_argument("--bos-id", type=int, default=DEFAULT_BOS_ID)
    parser.add_argument(
        "--prompt-ids-file",
        default=None,
        help="run on the EXACT server token ids exported by "
        "scripts/fable/server_cell_bench.py --prompt-ids-out (the Qwen-PR sized "
        "cells). Overrides the prefill_bench builder AND --bos (the exported ids "
        "already are what the server saw; DSV4.1 served ids carry NO BOS). "
        "Selects the entry for (cell=sweep, target_tokens==--context-tokens, "
        "seed==--prompt-seed). Default None keeps the built prompt so old "
        "receipts stay comparable.",
    )
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=None,
        help="which seed's ids to take from --prompt-ids-file (e.g. 20260829).",
    )
    parser.add_argument("--slot-layout", default="component-banks")
    parser.add_argument(
        "--max-kv",
        type=int,
        default=None,
        help="max_live_kv_tokens for the loader (default: auto = max cell "
        "context + steps + 64). Must cover the largest cell.",
    )
    parser.add_argument("--memory-limit-gib", type=float, default=100.0)
    parser.add_argument("--expert-cache-limit-gib", type=float, default=None)
    parser.add_argument(
        "--verify-record-hashes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="verify each gathered expert record's SHA-256 (slow; off for a "
        "speed bench).",
    )
    parser.add_argument("--no-admit", dest="admit", action="store_false", default=True)
    parser.add_argument("--admission-receipt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default=None, help="optional receipt label.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--apply-memory-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the reconciled MLX memory cap before allocation (on for the "
        "GPU window; pass --no-apply-memory-cap for a CPU reproduction).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="force the MLX default device to the CPU stream (no Metal).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="CPU-only test double: no model, no MLX/Metal. Proves argument "
        "parsing, prompt build, receipt paths and the append-only guard.",
    )
    return parser


def resolve_max_kv(context_cells, steps: int, max_kv) -> int:
    """The KV budget the loader needs to cover the largest prefill + decode."""

    needed = max(int(c) for c in context_cells) + int(steps) + 64
    if max_kv is None:
        return needed
    max_kv = int(max_kv)
    if max_kv < needed:
        raise ValueError(
            f"--max-kv {max_kv} is below the {needed} tokens needed for the "
            f"largest cell ({max(context_cells)}) + {steps} decode + margin"
        )
    return max_kv


# --------------------------------------------------------------------------
# provenance + append-only receipt writing
# --------------------------------------------------------------------------


def _git_rev(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def fresh_out_dir(base: Path, step: str = STEP) -> Path:
    """A never-reused UTC-stamped receipt directory ``<base>/<step>/<stamp>/``."""

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    parent = Path(base) / step
    candidate = parent / stamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = parent / f"{stamp}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def receipt_filename(receipt: dict) -> str:
    """cap/seed/window-suffixed receipt name (never-overwrite-a-measurement)."""

    cells = "-".join(str(c) for c in receipt.get("context_cells", []))
    steps = receipt.get("steps", "NA")
    seed = receipt.get("seed", "NA")
    stamp = receipt.get("utc_compact", "NA")
    return f"{STEP}__ctx{cells}__steps{steps}__seed{seed}__{stamp}.json"


def write_receipt(out_dir: Path, receipt: dict) -> Path:
    """Write the receipt, refusing to overwrite an existing file (append-only)."""

    path = Path(out_dir) / receipt_filename(receipt)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing measurement receipt: {path}"
        )
    path.write_text(json.dumps(receipt, indent=2))
    return path


# --------------------------------------------------------------------------
# memory + gather probes (real MLX / dry-run doubles share the call sites)
# --------------------------------------------------------------------------


def _process_rss_bytes() -> int:
    """Resident set size of this process. macOS ru_maxrss is bytes, Linux KiB."""

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(maxrss)
    return int(maxrss) * 1024  # Linux reports KiB


class _MLXMemProbe:
    def __init__(self, mx):
        self._mx = mx

    def reset_peak(self) -> None:
        for fn in ("reset_peak_memory",):
            call = getattr(self._mx, fn, None)
            if callable(call):
                try:
                    call()
                    return
                except Exception:
                    pass
        metal = getattr(self._mx, "metal", None)
        call = getattr(metal, "reset_peak_memory", None)
        if callable(call):
            try:
                call()
            except Exception:
                pass

    def peak_bytes(self) -> int:
        for owner in (self._mx, getattr(self._mx, "metal", None)):
            call = getattr(owner, "get_peak_memory", None)
            if callable(call):
                try:
                    return int(call())
                except Exception:
                    pass
        return 0

    def rss_bytes(self) -> int:
        return _process_rss_bytes()


class _GatherProbe:
    """Reads expert-record and engram-row counters off the runtime (cheap)."""

    def __init__(self, model, mx=None):
        self._runtime = getattr(model, "_mtplx_expert_runtime", None)
        self._engram_hooks = []
        inner = getattr(getattr(model, "model", None), "layers", None) or []
        for layer in inner:
            hook = getattr(layer, "engram_hook", None)
            if hook is not None and getattr(hook, "row_cache", None) is not None:
                self._engram_hooks.append(hook)
        self._mx = mx

    def _expert_requests(self) -> int | None:
        if self._runtime is None:
            return None
        try:
            snap = self._runtime.snapshot(mx_module=self._mx)
            return int(snap.get("cache", {}).get("expert_requests", 0))
        except Exception:
            return None

    def _engram_rows(self) -> int | None:
        if not self._engram_hooks:
            return None
        total = 0
        seen = False
        for hook in self._engram_hooks:
            stats = getattr(hook.row_cache, "stats", None)
            if not isinstance(stats, dict):
                continue
            seen = True
            total += int(stats.get("hits", 0)) + int(stats.get("misses", 0))
        return total if seen else None

    def baseline(self) -> dict:
        return {
            "expert_requests": self._expert_requests(),
            "engram_rows": self._engram_rows(),
        }

    def delta(self, baseline: dict) -> dict:
        def _diff(now, before):
            if now is None or before is None:
                return now
            return now - before

        return {
            "expert_records_gathered": _diff(
                self._expert_requests(), baseline.get("expert_requests")
            ),
            "engram_rows_gathered": _diff(
                self._engram_rows(), baseline.get("engram_rows")
            ),
        }


# --------------------------------------------------------------------------
# device-agnostic ops (real MLX vs. dry-run pure-python)
# --------------------------------------------------------------------------


class _MLXOps:
    def __init__(self, mx):
        self._mx = mx

    def input(self, ids_2d):
        return self._mx.array(ids_2d)

    def argmax_last(self, logits) -> int:
        return int(self._mx.argmax(logits[0, -1]).item())

    def sync(self, logits) -> None:
        self._mx.eval(logits)


class _FakeOps:
    def input(self, ids_2d):
        return ids_2d

    def argmax_last(self, logits) -> int:
        row = logits[0][-1]
        return max(range(len(row)), key=lambda i: row[i])

    def sync(self, logits) -> None:
        return None


# --------------------------------------------------------------------------
# one benchmark cell (shared by the real path and the dry-run double)
# --------------------------------------------------------------------------


def bench_one_cell(
    *,
    model,
    tokenizer,
    ops,
    mem_probe,
    gather_probe,
    prompt_ids,
    steps: int,
    decode_mode: str = "ar",
    dspark_depth: int = 3,
) -> dict:
    """Greedy prefill + decode of one cell; returns the measured metrics.

    ``decode_mode == "dspark"`` additionally runs the DSpark-DIRECT speculative
    loop (W57) after the AR decode, asserts the greedy ids are byte-identical, and
    attaches its tokens/cycle + accept-by-depth under the ``dspark`` key."""

    prompt_len = len(prompt_ids)
    mem_probe.reset_peak()
    baseline = gather_probe.baseline()

    cell_start = time.perf_counter()

    # -- prefill (produces the first / TTFT token) -----------------------------
    t0 = time.perf_counter()
    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    ttft_s = time.perf_counter() - t0
    token = ops.argmax_last(logits)
    generated = [token]

    # -- decode (steps autoregressive forwards) --------------------------------
    decode_start = time.perf_counter()
    for _ in range(int(steps)):
        logits = model(ops.input([[token]]), cache=cache)
        ops.sync(logits)
        token = ops.argmax_last(logits)
        generated.append(token)
    decode_wall_s = time.perf_counter() - decode_start

    wall_s = time.perf_counter() - cell_start

    gathered = gather_probe.delta(baseline)
    try:
        text = tokenizer.decode(generated)
    except Exception:  # pragma: no cover - detok guard
        text = ""

    decode_tokens = int(steps)
    dspark_metrics = None
    if decode_mode == "dspark" and getattr(model, "mtp", None) is not None:
        # DSpark-DIRECT lane: greedy speculative decode MUST reproduce the AR ids
        # (verify is authoritative); assert byte-identity and record the accept
        # structure.  Skipped for the dry-run double (a _FakeModel has no mtp).
        from mtplx.models.deepseek_v41_dspark_decode import (
            DSparkDecodeStats,
            dspark_generate,
        )
        from mtplx.sampling import SamplerConfig

        mem_probe.reset_peak()
        st = DSparkDecodeStats()
        dsp_start = time.perf_counter()
        dsp_ids = dspark_generate(
            model,
            [int(t) for t in prompt_ids],
            max_tokens=decode_tokens + 1,
            sampler=SamplerConfig(temperature=0.0),
            seed=0,
            speculative_depth=int(dspark_depth),
            stats=st,
        )
        dsp_wall = time.perf_counter() - dsp_start
        byte_identical = list(dsp_ids) == list(generated)
        if not byte_identical:
            first = next(
                (i for i, (a, b) in enumerate(zip(dsp_ids, generated)) if a != b),
                min(len(dsp_ids), len(generated)),
            )
            raise AssertionError(
                "DSpark-DIRECT greedy decode diverged from AR at index "
                f"{first}; speculative lane is not lossless"
            )
        sd = st.to_dict()
        dspark_metrics = {
            "depth": int(dspark_depth),
            "byte_identical_vs_ar": byte_identical,
            "decode_wall_s": dsp_wall,
            "decode_tok_s": (len(dsp_ids) / dsp_wall) if dsp_wall > 0 else None,
            "peak_mlx_gb": mem_probe.peak_bytes() / GIB,
            "tokens_per_cycle": sd["tokens_per_cycle"],
            "accept_rate": sd["accept_rate"],
            "accept_rate_by_depth": sd["accept_rate_by_depth"],
            "drafted_by_depth": sd["drafted_by_depth"],
            "accepted_by_depth": sd["accepted_by_depth"],
            "cycles": sd["cycles"],
            "verify_calls": sd["verify_calls"],
        }
    return {
        "prompt_tokens": prompt_len,
        "ttft_s": ttft_s,
        "dspark": dspark_metrics,
        "prefill_tok_s": (prompt_len / ttft_s) if ttft_s > 0 else None,
        "decode_tokens": decode_tokens,
        "decode_wall_s": decode_wall_s,
        "decode_tok_s": (decode_tokens / decode_wall_s)
        if decode_wall_s > 0
        else None,
        "wall_s": wall_s,
        "peak_mlx_bytes": mem_probe.peak_bytes(),
        "peak_mlx_gb": mem_probe.peak_bytes() / GIB,
        "process_rss_bytes": mem_probe.rss_bytes(),
        "process_rss_gb": mem_probe.rss_bytes() / GIB,
        "expert_records_gathered": gathered.get("expert_records_gathered"),
        "engram_rows_gathered": gathered.get("engram_rows_gathered"),
        "generated_token_count": len(generated),
        "text_preview": text[:_TEXT_PREVIEW_CHARS],
    }


def fastest_of(repeats: list[dict]) -> dict | None:
    """Fastest-of summary (report-fastest-of-seeds.md): max decode tok/s / min
    TTFT with the min-max range."""

    if not repeats:
        return None

    def _vals(key):
        return [r[key] for r in repeats if isinstance(r.get(key), (int, float))]

    decode = _vals("decode_tok_s")
    prefill = _vals("prefill_tok_s")
    ttft = _vals("ttft_s")
    peak = _vals("peak_mlx_gb")
    return {
        "decode_tok_s_fastest": max(decode) if decode else None,
        "decode_tok_s_range": [min(decode), max(decode)] if decode else None,
        "prefill_tok_s_fastest": max(prefill) if prefill else None,
        "prefill_tok_s_range": [min(prefill), max(prefill)] if prefill else None,
        "ttft_s_fastest": min(ttft) if ttft else None,
        "ttft_s_range": [min(ttft), max(ttft)] if ttft else None,
        "peak_gb_highest": max(peak) if peak else None,
    }


# --------------------------------------------------------------------------
# dry-run CPU-only test double (no model, no MLX/Metal)
# --------------------------------------------------------------------------


class _FakeTokenizer:
    """Deterministic tokenizer: 1 id per whitespace token; readable decode."""

    def encode(self, text):
        words = text.split() or ["x"]
        return [(abs(hash(w)) % 30000) + 1 for w in words]

    def decode(self, ids):
        return " ".join(f"t{int(i)}" for i in ids)


class _FakeRuntimeSpec:
    key = "dry-run/deepseek-v41-flash-q2"


class _FakeRuntimeManifest:
    manifest_sha256 = "0" * 64


class _FakeRuntimeConfig:
    memory_limit_bytes = 100 * GIB
    expert_cache_limit_bytes = None


class _FakeRuntime:
    """Mimics the counters the real runtime exposes to the gather probe."""

    def __init__(self):
        self.spec = _FakeRuntimeSpec()
        self.manifest = _FakeRuntimeManifest()
        self.config = _FakeRuntimeConfig()
        self._expert_requests = 0

    def bump(self, n: int) -> None:
        self._expert_requests += int(n)

    def snapshot(self, *, mx_module=None):
        return {"cache": {"expert_requests": self._expert_requests}}

    def close(self):
        return None


class _FakeRowCache:
    def __init__(self):
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}


class _FakeEngramHook:
    def __init__(self):
        self.row_cache = _FakeRowCache()


class _FakeLayer:
    def __init__(self, layer_id: int, engram: bool):
        self.layer_id = layer_id
        self.engram_hook = _FakeEngramHook() if engram else None


class _FakeInner:
    def __init__(self, layers):
        self.layers = layers


class _FakeModel:
    """Serve-path-shaped double: make_cache + __call__ + runtime/engram counters.

    Its ``__call__`` bumps the fake runtime and engram counters exactly as a
    routed forward would, so the SAME gather-probe read path is exercised on CPU
    with no model. Logits are a tiny deterministic vocab.
    """

    VOCAB = 8

    def __init__(self, n_layers: int = 40, engram_layers=(1, 14)):
        layers = [
            _FakeLayer(i, engram=i in set(engram_layers)) for i in range(n_layers)
        ]
        self.model = _FakeInner(layers)
        self._mtplx_expert_runtime = _FakeRuntime()
        self._mtplx_engram_layer_ids = tuple(engram_layers)
        self._step = 0

    def make_cache(self):
        return {"offset": 0}

    def __call__(self, input_ids, cache=None):
        rows = input_ids  # list of lists
        seq = len(rows[0])
        # a routed forward touches top_k experts/token across routed layers,
        # and each engram layer gathers rows/token — model the counters cheaply.
        self._mtplx_expert_runtime.bump(seq * len(self.model.layers) * 8)
        for layer in self.model.layers:
            if layer.engram_hook is not None:
                layer.engram_hook.row_cache.stats["hits"] += seq * 24
        self._step += 1
        vocab = self.VOCAB
        pick = self._step % vocab
        row = [0.0] * vocab
        row[pick] = 1.0
        return [[row]]  # [b=1, last-pos-only, vocab]


def _dry_run_cell(args, build_prompt, context_tokens: int, steps: int) -> tuple:
    tokenizer = _FakeTokenizer()
    prompt_ids, prompt_meta = _resolve_prompt(
        args, tokenizer, build_prompt, context_tokens
    )
    model = _FakeModel()
    ops = _FakeOps()
    mem_probe = _DryMemProbe()
    gather_probe = _GatherProbe(model, mx=None)
    metrics = bench_one_cell(
        model=model,
        tokenizer=tokenizer,
        ops=ops,
        mem_probe=mem_probe,
        gather_probe=gather_probe,
        prompt_ids=prompt_ids,
        steps=steps,
    )
    return metrics, prompt_meta, model


class _DryMemProbe:
    def reset_peak(self):
        return None

    def peak_bytes(self):
        return 0

    def rss_bytes(self):
        return _process_rss_bytes()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _base_receipt(args, *, dry_run: bool, worktree: Path) -> dict:
    now = time.gmtime()
    return {
        "step": STEP,
        "benchmark_shape": "dsv41-standard-benchmark-shape",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
        "utc_compact": time.strftime("%Y%m%dT%H%M%SZ", now),
        "git_rev": _git_rev(worktree),
        "dry_run": bool(dry_run),
        "label": args.label,
        "model_path": str(args.model),
        "slot_layout": args.slot_layout,
        "context_cells": [int(c) for c in args.context_tokens],
        "steps": int(args.steps),
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "prompt_format": args.prompt_format,
        "bos_prepended": bool(args.bos),
        "bos_id": int(args.bos_id) if args.bos else None,
        "prompt_ids_file": getattr(args, "prompt_ids_file", None),
        "prompt_seed": getattr(args, "prompt_seed", None),
        "memory_limit_gib": float(args.memory_limit_gib),
        "verify_record_hashes": bool(args.verify_record_hashes),
        "greedy": True,
        "cells": [],
    }


def run_dry(args) -> int:
    build_prompt = _load_build_prompt()
    worktree = Path(__file__).resolve().parents[2]
    receipt = _base_receipt(args, dry_run=True, worktree=worktree)
    receipt["max_live_kv_tokens"] = resolve_max_kv(
        args.context_tokens, args.steps, args.max_kv
    )
    receipt["spec_key"] = _FakeRuntimeSpec.key
    receipt["manifest_sha256"] = _FakeRuntimeManifest.manifest_sha256
    receipt["expert_cache_limit_bytes"] = _FakeRuntimeConfig.expert_cache_limit_bytes
    engram_ids = None
    for context_tokens in args.context_tokens:
        repeats = []
        prompt_meta = None
        for _ in range(int(args.repeats)):
            metrics, prompt_meta, model = _dry_run_cell(
                args, build_prompt, context_tokens, args.steps
            )
            repeats.append(metrics)
            engram_ids = list(getattr(model, "_mtplx_engram_layer_ids", ()) or ())
        receipt["cells"].append(
            {
                "context_tokens": int(context_tokens),
                "prompt_build": prompt_meta,
                "repeats": repeats,
                "fastest_of": fastest_of(repeats),
            }
        )
    receipt["engram_layer_ids"] = engram_ids
    out_dir = fresh_out_dir(args.out_dir)
    path = write_receipt(out_dir, receipt)
    print(f"[bench] DRY-RUN receipt -> {path}", flush=True)
    for cell in receipt["cells"]:
        fast = cell["fastest_of"] or {}
        print(
            f"[bench] dry cell ctx={cell['context_tokens']} "
            f"decode_tok_s(fastest)={fast.get('decode_tok_s_fastest')} "
            f"prompt_tokens={cell['prompt_build'].get('input_tokens')}",
            flush=True,
        )
    return 0


def run_real(args) -> int:
    import mlx.core as mx

    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(int(args.seed))

    build_prompt = _load_build_prompt()
    worktree = Path(__file__).resolve().parents[2]

    from mlx_lm.utils import load_tokenizer
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming

    admission_receipt = None
    if args.admission_receipt is not None:
        admission_receipt = json.loads(Path(args.admission_receipt).read_text())

    max_kv = resolve_max_kv(args.context_tokens, args.steps, args.max_kv)
    cache_limit = (
        None
        if args.expert_cache_limit_gib is None
        else int(args.expert_cache_limit_gib * GIB)
    )
    # --decode-mode dspark loads with the DSpark head (with_mtp=True) and reprices
    # the MTP residents (~7.4 GiB) against the expert cache so the plan still fits.
    from mtplx.models.deepseek_v41_dspark_decode import dspark_bench_loader_overrides

    with_mtp, memory_limit_bytes, cache_limit = dspark_bench_loader_overrides(
        want_dspark=(getattr(args, "decode_mode", "ar") == "dspark"),
        memory_limit_bytes=int(args.memory_limit_gib * GIB),
        expert_cache_limit_bytes=cache_limit,
    )

    resident = load_deepseek_v41_streaming(
        args.model,
        memory_limit_bytes=memory_limit_bytes,
        max_live_kv_tokens=int(max_kv),
        admit=args.admit,
        admission_receipt=admission_receipt,
        expert_cache_limit_bytes=cache_limit,
        apply_memory_cap=args.apply_memory_cap,
        slot_layout=args.slot_layout,
        cache_scope="layer",
        island_layers=(),
        verify_record_hashes=args.verify_record_hashes,
        with_mtp=with_mtp,
    )
    model = resident.model
    if want_dspark and getattr(model, "mtp", None) is None:
        raise RuntimeError(
            "--decode-mode dspark needs the DSpark MTP head, but the loaded model "
            "has none (with_mtp did not build it -- the artifact ships no mtp.* "
            "residents, or the config declares no MTP stages). Load a DSpark "
            "artifact or drop --decode-mode dspark."
        )
    runtime = getattr(model, "_mtplx_expert_runtime")

    receipt = _base_receipt(args, dry_run=False, worktree=worktree)
    receipt["max_live_kv_tokens"] = int(max_kv)
    receipt["device"] = "cpu" if args.cpu else "default"
    receipt["spec_key"] = runtime.spec.key
    receipt["manifest_sha256"] = getattr(runtime.manifest, "manifest_sha256", None)
    receipt["memory_limit_bytes"] = runtime.config.memory_limit_bytes
    receipt["expert_cache_limit_bytes"] = runtime.config.expert_cache_limit_bytes
    receipt["engram_layer_ids"] = list(
        getattr(model, "_mtplx_engram_layer_ids", ()) or ()
    )

    try:
        tokenizer = load_tokenizer(Path(args.model))
        ops = _MLXOps(mx)
        mem_probe = _MLXMemProbe(mx)
        gather_probe = _GatherProbe(model, mx=mx)
        for context_tokens in args.context_tokens:
            prompt_ids, prompt_meta = _resolve_prompt(
                args, tokenizer, build_prompt, context_tokens
            )
            repeats = []
            for repeat_idx in range(int(args.repeats)):
                print(
                    f"[bench] cell ctx={context_tokens} repeat "
                    f"{repeat_idx + 1}/{args.repeats} "
                    f"(prompt_tokens={len(prompt_ids)}, steps={args.steps})",
                    flush=True,
                )
                metrics = bench_one_cell(
                    model=model,
                    tokenizer=tokenizer,
                    ops=ops,
                    mem_probe=mem_probe,
                    gather_probe=gather_probe,
                    prompt_ids=prompt_ids,
                    steps=args.steps,
                    decode_mode=getattr(args, "decode_mode", "ar"),
                    dspark_depth=getattr(args, "dspark_depth", 3),
                )
                repeats.append(metrics)
                print(
                    f"[bench]   prefill_tok_s={metrics['prefill_tok_s']:.2f} "
                    f"ttft_s={metrics['ttft_s']:.3f} "
                    f"decode_tok_s={metrics['decode_tok_s']:.2f} "
                    f"peak_gb={metrics['peak_mlx_gb']:.2f} "
                    f"rss_gb={metrics['process_rss_gb']:.2f}",
                    flush=True,
                )
            receipt["cells"].append(
                {
                    "context_tokens": int(context_tokens),
                    "prompt_build": prompt_meta,
                    "repeats": repeats,
                    "fastest_of": fastest_of(repeats),
                }
            )
    finally:
        try:
            runtime.close()
        except Exception:
            pass

    out_dir = fresh_out_dir(args.out_dir)
    path = write_receipt(out_dir, receipt)
    print(f"[bench] receipt -> {path}", flush=True)
    for cell in receipt["cells"]:
        fast = cell["fastest_of"] or {}
        print(
            f"[bench] cell ctx={cell['context_tokens']} "
            f"decode_tok_s(fastest)={fast.get('decode_tok_s_fastest')} "
            f"prefill_tok_s(fastest)={fast.get('prefill_tok_s_fastest')} "
            f"peak_gb={fast.get('peak_gb_highest')}",
            flush=True,
        )
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.steps) < 1:
        print("bench_standard_shape: --steps must be >= 1", file=sys.stderr)
        return 2
    if int(args.repeats) < 1:
        print("bench_standard_shape: --repeats must be >= 1", file=sys.stderr)
        return 2
    try:
        resolve_max_kv(args.context_tokens, args.steps, args.max_kv)
    except ValueError as exc:
        print(f"bench_standard_shape: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return run_dry(args)

    try:
        return run_real(args)
    except Exception as exc:  # loader/admission/model not ready, OOM, etc.
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(
            "bench_standard_shape: could not complete a run "
            f"({type(exc).__name__}: {exc}). If this is a ResidentLoadError, "
            "worker W1's mtplx/models/deepseek_v41.py may not be available yet, "
            "or admission failed on the manifest identity (see W3_REPORT.md).",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
