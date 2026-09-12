#!/usr/bin/env python3
"""W89 -- DeepSeek-V4.1-Flash per-layer route-trace collector (feasibility only).

Records, for every routed layer and every token of the prefill's last
``--prefill-tail`` tokens plus ``--decode-tokens`` decode tokens, the two hidden
states a route predictor could read plus the router's decision:

  * ``router_in``  -- the pre-MoE, post-attention hidden that feeds the gate
                      (``MoE.__call__``'s ``xf`` = the ffn-HC ``moe_input``,
                      reshaped ``[n, hidden]``).  This is what predictor (a)
                      re-runs the router from (a ~100 % sanity check).
  * ``layer_in``   -- the residual stream ENTERING the layer, mean over the
                      hc copies (``mean(h, axis=2)`` -- exactly the pre-layer
                      hidden the DSpark head reads, model L2415).  This is what a
                      one-/two-layer-ahead predictor reads: hidden-in(L-1) or
                      hidden-in(L-2) predicting layer L's top-6.
  * ``top6``       -- the router's top-``num_experts_per_tok`` expert ids
                      (``(scores+bias)``-descending, the shipped route).
  * ``gate``       -- the routed weights (unbiased, norm_topk_prob'd) at those ids.
  * per token: the token id, the absolute position, and the phase
                      (0 = prefill-tail, 1 = decode).

The capture is NON-INVASIVE: the hooks call the unchanged ``MoE.__call__`` /
``DecoderLayer.__call__`` and only READ intermediates (an extra pure ``gate``
matvec for the ids), so decode logits are byte-identical with the hooks off
(gated by ``tests/models/test_route_predictor.py``).  Nothing here changes a
weight, a format, or a model output; a route predictor built on these traces may
only waste a prefetch read, never change a result.

Hiddens are stored as bf16 (the runtime dtype) -- numpy 2.x cannot buffer a bf16
array and this box has no ``ml_dtypes``, so each hidden is the *exact*
bf16-rounded value kept as its uint16 bit pattern (``bf16_bits`` / ``bf16_to_f32``
round-trip, exact).  Estimate at the standard cell: 2,304 tok x 40 layers x 2
hiddens x 5,120 x 2 B ~= 1.9 GB; ``--hidden-stride`` subsamples the hidden axis
if a run needs to stay under ``--max-bytes``.

GPU-window run (real model; NOT run by this worker -- a window is running).  The
exact command is in ``docs/deepseek-v41/W89_ROUTE_PREDICTOR.md`` and echoed by
``--print-window-command``.  Validated on CPU with ``--tiny`` against the fake
model from ``tests/models/test_deepseek_v41_stage_timing.py`` (no artifact, no
GPU, <1.5 GB, ``nice -n 19``).

  # CPU validation (safe now, no GPU, no model load):
  PYTHONPATH=<worktree> nice -n 19 python3 scripts/deepseek_v41/collect_route_traces.py \
      --tiny --out /tmp/w89-tiny-trace

Author: Opus 4.8 worker (w89/route-predictor).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_SCRIPT_DIR = _THIS.parent

# The standard-cell window command (real model); documented + printable so the
# window operator copies it verbatim.  This worker never runs it (no GPU now).
STANDARD_WINDOW_COMMAND = (
    "scripts/deepseek_v41/gpu_window.sh \\\n"
    "  env PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \\\n"
    "  scripts/deepseek_v41/collect_route_traces.py \\\n"
    "    --arm cell16k --memory-limit-gib 60 --max-kv 17408 \\\n"
    "    --context-tokens 16384 --prefill-tail 2048 --decode-tokens 256 \\\n"
    "    --prompt-ids-file "
    "docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/"
    "prompt-ids-deepseek-v41.json \\\n"
    "    --prompt-seed 20260829 \\\n"
    "    --out docs/deepseek-v41/receipts/gpu-windows/window-89/route-traces"
)


# ---------------------------------------------------------------------------
# bf16 <-> float32 as uint16 bit patterns (no ml_dtypes; numpy 2.x can't buffer
# a bf16 array).  We cast through mlx bf16 so the stored value is EXACTLY the
# bf16 the runtime would feed a predictor; the low 16 bits are then zero, so the
# ">> 16" truncation is exact (never a second rounding).
# ---------------------------------------------------------------------------
def bf16_bits(x_mx, mx) -> np.ndarray:
    """mlx array -> uint16 numpy of its bf16-rounded value (exact bit pattern)."""
    f32 = np.ascontiguousarray(
        np.array(x_mx.astype(mx.bfloat16).astype(mx.float32)), dtype=np.float32
    )
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit pattern -> float32 (exact inverse of :func:`bf16_bits`)."""
    u16 = np.ascontiguousarray(u16, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


PHASE_PREFILL_TAIL = 0
PHASE_DECODE = 1


# ---------------------------------------------------------------------------
# Recorder + class-level hooks
# ---------------------------------------------------------------------------
class RouteRecorder:
    """Per-layer, per-token capture buffers filled by the two hooks.

    Preallocated to ``[total_rows, stored_hidden]`` per layer so the working set
    is exactly the final shard bytes (uint16), not a float32 shadow.  Row r of
    every layer is the same token (all routed layers see the same token stream in
    the same order), which :func:`write_shards` asserts before collapsing the
    per-token columns to one shared array.
    """

    def __init__(self, *, hidden, top_k, n_experts, hc_mult, total_rows,
                 hidden_stride, mx):
        self.mx = mx
        self.hidden = int(hidden)
        self.top_k = int(top_k)
        self.n_experts = int(n_experts)
        self.hc_mult = int(hc_mult)
        self.total_rows = int(total_rows)
        self.hidden_stride = max(1, int(hidden_stride))
        self.stored_hidden = len(range(0, self.hidden, self.hidden_stride))
        self.enabled = False
        # per-forward context (set by the driver before each recorded forward)
        self.cur_token_ids = None  # np.int64 [s]
        self.cur_base = 0
        self.cur_phase = PHASE_PREFILL_TAIL
        # per-layer buffers (lazily allocated on first sight of a layer id)
        self.router_in: dict[int, np.ndarray] = {}
        self.layer_in: dict[int, np.ndarray] = {}
        self.top6: dict[int, np.ndarray] = {}
        self.gate: dict[int, np.ndarray] = {}
        self.token: dict[int, np.ndarray] = {}
        self.pos: dict[int, np.ndarray] = {}
        self.phase: dict[int, np.ndarray] = {}
        self.row_ptr: dict[int, int] = {}
        self._stash: dict[int, tuple] = {}
        self.overflow = False

    def _alloc(self, lid: int) -> None:
        T, H = self.total_rows, self.stored_hidden
        self.router_in[lid] = np.zeros((T, H), np.uint16)
        self.layer_in[lid] = np.zeros((T, H), np.uint16)
        self.top6[lid] = np.zeros((T, self.top_k), np.int32)
        self.gate[lid] = np.zeros((T, self.top_k), np.float32)
        self.token[lid] = np.zeros((T,), np.int32)
        self.pos[lid] = np.zeros((T,), np.int32)
        self.phase[lid] = np.zeros((T,), np.uint8)
        self.row_ptr[lid] = 0

    def begin_forward(self, token_ids, base_pos: int, phase: int) -> None:
        self.cur_token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        self.cur_base = int(base_pos)
        self.cur_phase = int(phase)

    # -- hook side --
    def stash_moe(self, lid: int, xf, idx, w) -> None:
        self._stash[int(lid)] = (xf, idx, w)

    def record_layer(self, lid: int, h_layer_in, positions) -> None:
        lid = int(lid)
        st = self._stash.pop(lid, None)
        if st is None:
            return  # a non-MoE / dense layer never stashed; skip it
        xf, idx, w = st
        mx = self.mx
        if lid not in self.router_in:
            self._alloc(lid)
        stride = self.hidden_stride
        # layer-in: mean over the hc copies -> [n, hidden] (the pre-layer hidden
        # a one-layer-ahead predictor reads; matches the DSpark main-hidden read).
        li = mx.mean(h_layer_in.astype(mx.float32), axis=2).reshape(-1, self.hidden)
        ri = xf.reshape(-1, self.hidden)
        if stride > 1:
            li = li[:, ::stride]
            ri = ri[:, ::stride]
        ri_u = bf16_bits(ri, mx)
        li_u = bf16_bits(li, mx)
        idx_np = np.array(idx).astype(np.int32).reshape(ri_u.shape[0], self.top_k)
        w_np = np.array(w.astype(mx.float32)).astype(np.float32).reshape(
            ri_u.shape[0], self.top_k
        )
        pos_np = np.array(positions).astype(np.int64).reshape(-1)
        n = ri_u.shape[0]
        p0 = self.row_ptr[lid]
        if p0 + n > self.total_rows:
            # never write past the preallocation (a longer-than-budgeted forward);
            # clamp and flag so the run fails loudly rather than silently corrupts.
            self.overflow = True
            n = max(0, self.total_rows - p0)
            if n == 0:
                return
            ri_u, li_u, idx_np, w_np, pos_np = (
                ri_u[:n], li_u[:n], idx_np[:n], w_np[:n], pos_np[:n]
            )
        sl = slice(p0, p0 + n)
        self.router_in[lid][sl] = ri_u
        self.layer_in[lid][sl] = li_u
        self.top6[lid][sl] = idx_np
        self.gate[lid][sl] = w_np
        within = pos_np - self.cur_base
        # map absolute position -> token id via the current forward's id vector
        within = np.clip(within, 0, len(self.cur_token_ids) - 1)
        self.token[lid][sl] = self.cur_token_ids[within].astype(np.int32)
        self.pos[lid][sl] = pos_np.astype(np.int32)
        self.phase[lid][sl] = self.cur_phase
        self.row_ptr[lid] = p0 + n

    def layer_ids(self) -> list[int]:
        return sorted(self.router_in)


def install_hooks(rec: RouteRecorder, moe_mod, dv41_mod):
    """Monkeypatch ``MoE.__call__`` + ``DecoderLayer.__call__`` at class level so
    every instance (resident SwitchGLU or streamed switch) is captured.  Returns
    the saved originals for :func:`uninstall_hooks`."""
    MoE = moe_mod.MoE
    DecoderLayer = dv41_mod.DecoderLayer
    orig_moe = MoE.__call__
    orig_layer = DecoderLayer.__call__

    def moe_call(self, x, image_mask=None):
        if rec.enabled:
            xf = x.reshape(-1, self.dim)
            # the gate is a pure linear + top-k (no streaming side effect); this
            # extra call reproduces the shipped route ids/weights for the trace.
            w, idx = self.gate(xf)
            rec.stash_moe(self.layer_id, xf, idx, w)
        return orig_moe(self, x, image_mask)

    def layer_call(self, h, pre_mix, positions, layer_cache, shared):
        out = orig_layer(self, h, pre_mix, positions, layer_cache, shared)
        if rec.enabled:
            rec.record_layer(self.layer_id, h, positions)
        return out

    MoE.__call__ = moe_call
    DecoderLayer.__call__ = layer_call
    return (MoE, orig_moe, DecoderLayer, orig_layer)


def uninstall_hooks(saved) -> None:
    MoE, orig_moe, DecoderLayer, orig_layer = saved
    MoE.__call__ = orig_moe
    DecoderLayer.__call__ = orig_layer


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------
def write_shards(rec: RouteRecorder, out_dir: Path, extra_manifest: dict) -> dict:
    """Write per-layer .npy shards + a shared per-token block + manifest.json.

    Asserts every routed layer captured the same token stream (row alignment),
    then writes one shared ``tokens.npy`` / ``positions.npy`` / ``phase.npy``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lids = rec.layer_ids()
    if not lids:
        raise SystemExit("no routed layers were captured (recorder saw nothing)")
    # Row-count + alignment check across layers.
    n_rows = rec.row_ptr[lids[0]]
    base_tok = rec.token[lids[0]][:n_rows]
    base_phase = rec.phase[lids[0]][:n_rows]
    for lid in lids:
        if rec.row_ptr[lid] != n_rows:
            raise SystemExit(
                f"layer {lid} captured {rec.row_ptr[lid]} rows != {n_rows} "
                f"(layer 0) -- row alignment broken"
            )
        if not np.array_equal(rec.token[lid][:n_rows], base_tok):
            raise SystemExit(f"layer {lid} token stream != layer {lids[0]}")
    total_bytes = 0
    for lid in lids:
        tag = f"layer{lid:03d}"
        for name, arr in (
            ("router_in", rec.router_in[lid][:n_rows]),
            ("layer_in", rec.layer_in[lid][:n_rows]),
            ("top6", rec.top6[lid][:n_rows]),
            ("gate", rec.gate[lid][:n_rows]),
        ):
            p = out_dir / f"{tag}_{name}.npy"
            np.save(p, np.ascontiguousarray(arr))
            total_bytes += p.stat().st_size
    for name, arr in (
        ("tokens", base_tok.astype(np.int32)),
        ("positions", rec.pos[lids[0]][:n_rows].astype(np.int32)),
        ("phase", base_phase.astype(np.uint8)),
    ):
        p = out_dir / f"{name}.npy"
        np.save(p, np.ascontiguousarray(arr))
        total_bytes += p.stat().st_size
    n_tail = int((base_phase == PHASE_PREFILL_TAIL).sum())
    n_decode = int((base_phase == PHASE_DECODE).sum())
    manifest = {
        "schema": "w89-route-trace-v1",
        "layer_ids": lids,
        "n_layers": len(lids),
        "n_experts": rec.n_experts,
        "top_k": rec.top_k,
        "hidden_size": rec.hidden,
        "hidden_stride": rec.hidden_stride,
        "stored_hidden": rec.stored_hidden,
        "hc_mult": rec.hc_mult,
        "n_rows": int(n_rows),
        "n_prefill_tail": n_tail,
        "n_decode": n_decode,
        "hidden_dtype": "bf16-as-uint16-bits",
        "hidden_files": ["router_in", "layer_in"],
        "per_token_files": ["tokens", "positions", "phase"],
        "phase_codes": {"prefill_tail": PHASE_PREFILL_TAIL, "decode": PHASE_DECODE},
        "total_bytes": int(total_bytes),
        "overflow": bool(rec.overflow),
        "window_command": STANDARD_WINDOW_COMMAND,
    }
    manifest.update(extra_manifest or {})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# Tiny fake model (mirrors tests/models/test_deepseek_v41_stage_timing.py)
# ---------------------------------------------------------------------------
def _tiny_args(mx, **over):
    from mtplx.models.deepseek_v41 import ModelArgs

    base = dict(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=12, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5,
        sliding_window=8, window_size=8, swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )
    base.update(over)
    return ModelArgs(**base)


def _tiny_model(mx, seed=1):
    from mlx.utils import tree_flatten, tree_unflatten
    from mtplx.models.deepseek_v41 import Model

    args = _tiny_args(mx)
    model = Model(args)
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = 0.1 * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())
    return model, args


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def _drive_capture(*, model, mx, prompt_ids, prefill_tail, decode_tokens,
                   hidden, top_k, n_experts, hc_mult, hidden_stride,
                   force_chunk_major, out_dir, extra_manifest):
    """Prefill (head unrecorded, tail recorded), then greedy-AR decode recorded.

    ``force_chunk_major`` passes ``prefill_layer_major=False`` on the recorded
    tail so it flows through ``_forward_span`` -> ``DecoderLayer.__call__``
    (the path the hooks tile) and keeps absolute positions clean.
    """
    s = len(prompt_ids)
    tail = min(int(prefill_tail), s)
    head = s - tail
    total_rows = tail + int(decode_tokens)
    rec = RouteRecorder(
        hidden=hidden, top_k=top_k, n_experts=n_experts, hc_mult=hc_mult,
        total_rows=total_rows, hidden_stride=hidden_stride, mx=mx,
    )
    import mtplx.models.deepseek_v41 as dv41
    import mtplx.models.deepseek_v41_moe as dv41_moe

    saved = install_hooks(rec, dv41_moe, dv41)
    t0 = time.time()
    try:
        cache = model.make_cache()
        # 1) head prefill -- NOT recorded (warms the KV cache); uses the arm env
        #    schedule for speed.
        if head > 0:
            head_ids = mx.array([list(prompt_ids[:head])])
            logits = model(head_ids, cache=cache)
            mx.eval(logits)
        # 2) tail prefill -- recorded, forced chunk-major for clean positions.
        rec.begin_forward(prompt_ids[head:], base_pos=head, phase=PHASE_PREFILL_TAIL)
        rec.enabled = True
        tail_ids = mx.array([list(prompt_ids[head:])])
        kw = {"prefill_layer_major": False} if force_chunk_major else {}
        logits = model(tail_ids, cache=cache, **kw)
        mx.eval(logits)
        rec.enabled = False
        token = int(mx.argmax(logits[0, -1]).item())
        # 3) greedy-AR decode -- recorded, one token per forward (one-shot path).
        for i in range(int(decode_tokens)):
            rec.begin_forward([token], base_pos=s + i, phase=PHASE_DECODE)
            rec.enabled = True
            logits = model(mx.array([[token]]), cache=cache)
            mx.eval(logits)
            rec.enabled = False
            token = int(mx.argmax(logits[0, -1]).item())
    finally:
        uninstall_hooks(saved)
    wall = time.time() - t0
    manifest = dict(extra_manifest or {})
    manifest.update({
        "prefill_tail_requested": int(prefill_tail),
        "prefill_tail_captured": int(tail),
        "prompt_tokens": int(s),
        "decode_tokens": int(decode_tokens),
        "capture_wall_s": round(wall, 3),
    })
    if rec.overflow:
        raise SystemExit(
            "recorder overflow: a forward produced more rows than budgeted "
            "(total_rows). Increase --prefill-tail budget or check chunking."
        )
    return write_shards(rec, out_dir, manifest)


def collect_tiny_trace(out_dir, *, context_tokens=64, prefill_tail=16,
                       decode_tokens=8, hidden_stride=1, seed=1):
    """End-to-end capture on the fake model (CPU, no artifact).  Reused by the
    trainer's ``--tiny`` and by the tests."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    model, args = _tiny_model(mx, seed=seed)
    prompt_ids = list(
        np.random.RandomState(0).randint(0, args.vocab_size, size=context_tokens)
    )
    return _drive_capture(
        model=model, mx=mx, prompt_ids=[int(t) for t in prompt_ids],
        prefill_tail=prefill_tail, decode_tokens=decode_tokens,
        hidden=args.hidden_size, top_k=args.num_experts_per_tok,
        n_experts=args.n_routed_experts, hc_mult=args.hc_mult,
        hidden_stride=hidden_stride, force_chunk_major=False,
        out_dir=Path(out_dir),
        extra_manifest={"source": "tiny-fake-model", "seed": seed},
    )


def _load_ab_module():
    path = _SCRIPT_DIR / "ab_decode_env_levers.py"
    spec = importlib.util.spec_from_file_location("_w89_ab_env_levers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_real_trace(args, out_dir) -> dict:
    """GPU-window capture on the real streamed artifact (reuses the ab harness
    plumbing: arm env, prompt-ids resolution, streaming loader)."""
    import mlx.core as mx

    ab = _load_ab_module()
    bench = ab._load_bench_module()
    # Build a fully-defaulted ab args namespace so ab._load_model gets every field.
    scratch_out = out_dir / "_ab_scratch_receipt.jsonl"
    ab_argv = [
        "--arms", args.arm,
        "--context-tokens", str(args.context_tokens),
        "--decode-tokens", str(args.decode_tokens),
        "--memory-limit-gib", str(args.memory_limit_gib),
        "--max-kv", str(args.max_kv),
        "--prompt-ids-file", str(args.prompt_ids_file),
        "--prompt-seed", str(args.prompt_seed),
        "--out", str(scratch_out),
    ]
    ab_args = ab.build_parser().parse_args(ab_argv)
    ab._apply_arm_env(args.arm)
    build_prompt = bench._load_build_prompt()
    prompt_ids, prompt_meta = bench._resolve_prompt(
        ab_args, None, build_prompt, args.context_tokens
    )
    prompt_ids = [int(t) for t in prompt_ids]
    print(f"[w89] prompt_tokens={len(prompt_ids)} "
          f"sha={prompt_meta.get('token_ids_sha256', '?')[:12]}", flush=True)
    resident = ab._load_model(ab_args, bench, mx)
    model = resident.model
    # Read the real dims off the loaded layer stack (robust to Model/Backbone
    # wrapping and to the streamed-switch rebind, which keeps the MoE class).
    layer0 = model.layers[0]
    hidden = int(layer0.mlp.dim)
    top_k = int(layer0.mlp.n_activated_experts)
    n_experts = int(layer0.mlp.n_routed_experts)
    hc_mult = int(layer0.hc_mult)
    n_layers_total = len(model.layers)
    # Size guard (uint16 hiddens): abort before capture if over --max-bytes.
    stored_h = len(range(0, hidden, max(1, args.hidden_stride)))
    est = (int(args.prefill_tail) + int(args.decode_tokens)) * stored_h * 2 * 2 * n_layers_total
    print(f"[w89] estimated trace bytes ~= {est/1e9:.2f} GB "
          f"(stride={args.hidden_stride}, cap={args.max_bytes/1e9:.2f} GB)", flush=True)
    if est > args.max_bytes:
        raise SystemExit(
            f"estimated {est/1e9:.2f} GB > --max-bytes {args.max_bytes/1e9:.2f} GB; "
            f"raise --hidden-stride (e.g. {int(np.ceil(est/args.max_bytes))}) or --max-bytes"
        )
    return _drive_capture(
        model=model, mx=mx, prompt_ids=prompt_ids,
        prefill_tail=args.prefill_tail, decode_tokens=args.decode_tokens,
        hidden=hidden, top_k=top_k, n_experts=n_experts, hc_mult=hc_mult,
        hidden_stride=args.hidden_stride, force_chunk_major=True,
        out_dir=out_dir,
        extra_manifest={
            "source": "real-model", "arm": args.arm,
            "memory_limit_gib": args.memory_limit_gib, "max_kv": args.max_kv,
            "context_tokens": args.context_tokens,
            "prompt_ids_file": str(args.prompt_ids_file),
            "prompt_seed": args.prompt_seed,
            "prompt_token_ids_sha256": prompt_meta.get("token_ids_sha256"),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, help="output directory for shards + manifest")
    p.add_argument("--tiny", action="store_true",
                   help="CPU validation on the fake model (no artifact, no GPU)")
    p.add_argument("--arm", default="cell16k",
                   help="ab_decode_env_levers arm (standard cell = cell16k)")
    p.add_argument("--context-tokens", type=int, default=16384)
    p.add_argument("--prefill-tail", type=int, default=2048,
                   help="record the last N prefill positions")
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--memory-limit-gib", type=float, default=60.0)
    p.add_argument("--max-kv", type=int, default=17408)
    p.add_argument("--prompt-ids-file", type=Path, default=None)
    p.add_argument("--prompt-seed", type=int, default=20260829)
    p.add_argument("--hidden-stride", type=int, default=1,
                   help="subsample the hidden axis (store every Nth element)")
    p.add_argument("--max-bytes", type=float, default=3.0e9,
                   help="abort before capture if the estimate exceeds this")
    p.add_argument("--print-window-command", action="store_true",
                   help="print the exact GPU-window command and exit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_window_command:
        print(STANDARD_WINDOW_COMMAND)
        return 0
    if args.out is None:
        print("error: --out is required (or use --print-window-command)",
              file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    if args.tiny:
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        # tiny fixed dims (fake model); the standard-cell flags are ignored here
        # except an explicitly-small override.
        ctx = args.context_tokens if 0 < args.context_tokens <= 512 else 64
        manifest = collect_tiny_trace(
            out_dir, context_tokens=ctx,
            prefill_tail=min(args.prefill_tail, 16), decode_tokens=min(args.decode_tokens, 8),
            hidden_stride=args.hidden_stride,
        )
    else:
        if args.prompt_ids_file is None:
            print("error: --prompt-ids-file is required for a real run",
                  file=sys.stderr)
            return 2
        manifest = collect_real_trace(args, out_dir)
    print(f"[w89] wrote {manifest['n_layers']} layers x {manifest['n_rows']} rows "
          f"({manifest['total_bytes']/1e6:.1f} MB) to {out_dir}", flush=True)
    print(f"[w89] tail={manifest['n_prefill_tail']} decode={manifest['n_decode']} "
          f"n_experts={manifest['n_experts']} top_k={manifest['top_k']} "
          f"hidden={manifest['hidden_size']} stride={manifest['hidden_stride']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
