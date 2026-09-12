"""W119: eager K+1-row verify vs 1-row AR reference parity (CPU, fp32).

Window 45 found that with the K29 decode-attention kernel OFF for the whole DSpark
arm (``cell16k_ring_v2_draft_attn_eager``: MTPLX_DSV41_DECODE_ATTN_KERNEL=0 +
DSPARK_VERIFY_K29=0, so the eager gathered core ``_sparse_attend_selected`` serves
BOTH the K+1-row verify and the 1-row AR/draft), the DSpark greedy stream diverged
from the AR reference at index 111 with ar_top2_margin 0.125, dspark_top2_margin
0.0, max|Δlogit| 1.125 -- classified "divergent" -- whereas with K29 ON the only
divergence was a tie flip (index 27, ar_top2_margin 0.0).

Under greedy decoding the K+1-row verify forward must reproduce, per row, the
logits the 1-row AR forward produces for the identical prefix, up to accumulation-
order rounding.  This test proves whether the EAGER multi-row selected-key core
computes anything numerically different (beyond fp32 reduction order) from the
1-row eager core for the same row -- i.e. whether the GPU 1.125-logit delta is a
runtime bug or bf16 accumulation-order between the s=K+1 and s=1 einsum tiles.

Method: a tiny synthetic DeepSeek-V4.1 model (ratio-2 CSA layers, small index_topk,
small sliding_window) on CPU in fp32.  Prefill a >= 64-token prompt into two fresh
caches (deterministic model -> identical cache state), then:
  * VERIFY: one (K+1)-row forward over the block [primary, d1..dK]  -> per-row logits.
  * AR REF: the same block one token at a time (1-row forwards)      -> per-row logits.
Compare per-row logits; fp32 on CPU should be ~0 (assert <= 1e-4).  Also compares the
per-row argmax (the DSpark greedy acceptance decision) and the eager core against the
K29 CPU reference (``decode_attention_reference``) and the compiled core.

CPU-pinned ([[worker-tests-must-pin-mlx-cpu]]); no artifact, no GPU, no model load.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mtplx.models import deepseek_v41 as dsv41
from mtplx.models import deepseek_v41_attn_kernels as k29


# ---------------------------------------------------------------------------
# Tiny synthetic model (mirrors scripts/deepseek_v41/metal_decode_attn_bisect.py
# ``_tiny_full_args`` / ``_build_tiny_full_model`` -- the proven full-Model CPU
# config that derives all four CSA modes: ratio-2 kv/index-source layers, an
# index_topk small enough that a 64-token prompt saturates it, sliding_window 8).
# ---------------------------------------------------------------------------
def _tiny_full_args() -> "dsv41.ModelArgs":
    return dsv41.ModelArgs(
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


def _build_tiny_full_model(seed: int = 1):
    """Full tiny ``Model`` with random fp32 weights (no artifact, dense, CPU)."""
    args = _tiny_full_args()
    model = dsv41.Model(args, quantize=False)
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


_VOCAB = 48


def _prompt_ids(n: int) -> list:
    # deterministic non-trivial ids in [1, vocab)
    return [((i * 7 + 3) % (_VOCAB - 1)) + 1 for i in range(n)]


def _block_ids(k1: int) -> list:
    # the K+1-row verify block [primary, d1..dK]; deterministic ids in [1, vocab)
    return [((i * 5 + 11) % (_VOCAB - 1)) + 1 for i in range(k1)]


def _forward_logits(model, ids_2d):
    logits = model(mx.array(ids_2d))
    mx.eval(logits)
    return logits


def _verify_rows(model, prompt_ids, block_ids):
    """One (K+1)-row verify forward; returns per-row logits (numpy [K+1, vocab])."""
    cache = model.make_cache()
    model(mx.array([list(prompt_ids)]), cache=cache)  # prefill, discard logits
    logits = model(mx.array([list(int(t) for t in block_ids)]), cache=cache)
    mx.eval(logits)
    return np.asarray(logits[0].astype(mx.float32))  # [K+1, vocab]


def _ar_reference_rows(model, prompt_ids, block_ids):
    """The block one token at a time (1-row forwards) from an independently
    prefilled cache -- the faithful M=1 AR replay.  Row i sees the identical
    prefix (prompt + block[:i]) as verify row i."""
    cache = model.make_cache()
    model(mx.array([list(prompt_ids)]), cache=cache)  # prefill, discard logits
    rows = []
    for tok in block_ids:
        logits = model(mx.array([[int(tok)]]), cache=cache)  # [1,1,vocab]
        mx.eval(logits)
        rows.append(np.asarray(logits[0, 0].astype(mx.float32)))
    return np.stack(rows, axis=0)  # [K+1, vocab]


def _row_max_abs(a, b):
    return [float(np.max(np.abs(a[i].astype(np.float64) - b[i].astype(np.float64))))
            for i in range(a.shape[0])]


# ---------------------------------------------------------------------------
# Env fixture: the eager selected-key regime (K30 ON, K29 OFF, core-compile OFF,
# attention-chain compile OFF) -- exactly cell16k_ring_v2_draft_attn_eager.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dsv41._ATTN_CORE_COMPILE_ENV, raising=False)
    monkeypatch.delenv("MTPLX_DSV41_DECODE_ATTN_KERNEL", raising=False)
    monkeypatch.delenv("MTPLX_DSV41_SELECTED_KEYS", raising=False)
    dsv41._ATTN_CORE_COMPILED.clear()
    yield
    dsv41._ATTN_CORE_COMPILED.clear()


def _arm_eager(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")   # K30 selected-key gather
    monkeypatch.setenv("MTPLX_DSV41_DECODE_ATTN_KERNEL", "0")  # K29 off
    monkeypatch.setenv(dsv41._ATTN_CORE_COMPILE_ENV, "0")  # eager softmax core
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", False)     # eager qkv/out prep


# ---------------------------------------------------------------------------
# 1) full-model per-row parity: eager K+1-row verify == 1-row AR, fp32 CPU.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k1", [2, 4, 6])
def test_eager_verify_matches_ar_reference_per_row(monkeypatch, k1, capsys):
    _arm_eager(monkeypatch)
    model, _args = _build_tiny_full_model(seed=3)
    prompt = _prompt_ids(64)
    block = _block_ids(k1)

    verify = _verify_rows(model, prompt, block)
    arref = _ar_reference_rows(model, prompt, block)
    assert verify.shape == arref.shape == (k1, _VOCAB)

    per_row = _row_max_abs(verify, arref)
    max_abs = max(per_row)
    with capsys.disabled():
        print(f"\n[W119] K+1={k1} eager verify vs 1-row AR: per-row max|Δlogit|="
              f"{['%.2e' % d for d in per_row]}  overall={max_abs:.3e}")
    # fp32 on CPU: the s=K+1 and s=1 einsum compute the identical per-row quantity;
    # any residual is CPU-BLAS reduction order, orders of magnitude below 1e-4.
    assert max_abs <= 1e-4, (
        f"eager K+1-row verify diverged from the 1-row AR reference by "
        f"max|Δlogit|={max_abs:.3e} on CPU fp32 -- a REAL multi-row core bug, not "
        f"accumulation order (per-row {per_row})"
    )


# ---------------------------------------------------------------------------
# 2) acceptance-decision parity: per-row argmax (the DSpark greedy accept test)
#    agrees between verify and the AR reference, so accept/reject is identical
#    for ANY draft block.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k1", [2, 4, 6])
def test_eager_verify_argmax_matches_ar_reference(monkeypatch, k1, capsys):
    _arm_eager(monkeypatch)
    model, _args = _build_tiny_full_model(seed=3)
    prompt = _prompt_ids(64)
    block = _block_ids(k1)

    verify = _verify_rows(model, prompt, block)
    arref = _ar_reference_rows(model, prompt, block)
    v_arg = [int(np.argmax(verify[i])) for i in range(k1)]
    a_arg = [int(np.argmax(arref[i])) for i in range(k1)]
    with capsys.disabled():
        print(f"\n[W119] K+1={k1} argmax verify={v_arg} ar={a_arg}")
    assert v_arg == a_arg, (
        f"per-row argmax diverged verify={v_arg} ar={a_arg}; the greedy accept "
        f"decision would differ between the K+1 verify and the M=1 AR reference"
    )


# ---------------------------------------------------------------------------
# 3) compiled-core variant: the same parity holds with ATTN_CORE_COMPILE=1
#    (rounding-class n=1 compile of the core), so the compiled tape also computes
#    the per-row quantity batch-invariantly on CPU.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k1", [2, 4, 6])
def test_compiled_core_verify_matches_ar_reference_per_row(monkeypatch, k1, capsys):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_DECODE_ATTN_KERNEL", "0")
    monkeypatch.setenv(dsv41._ATTN_CORE_COMPILE_ENV, "1")  # compiled core ON
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", False)
    dsv41._ATTN_CORE_COMPILED.clear()

    model, _args = _build_tiny_full_model(seed=3)
    prompt = _prompt_ids(64)
    block = _block_ids(k1)

    verify = _verify_rows(model, prompt, block)
    arref = _ar_reference_rows(model, prompt, block)
    per_row = _row_max_abs(verify, arref)
    max_abs = max(per_row)
    with capsys.disabled():
        print(f"\n[W119] K+1={k1} COMPILED-core verify vs 1-row AR: "
              f"overall max|Δlogit|={max_abs:.3e}; core tapes built="
              f"{len(dsv41._ATTN_CORE_COMPILED)}")
    assert max_abs <= 1e-4, (
        f"compiled-core K+1-row verify diverged from the 1-row AR reference by "
        f"max|Δlogit|={max_abs:.3e} (per-row {per_row})"
    )


# ---------------------------------------------------------------------------
# 4) core-level isolation: the eager selected-key core ``_attn_core_impl`` is
#    BATCH-INVARIANT (row i of the s=K+1 call == the s=1 call on row i) and equals
#    the K29 CPU reference ``decode_attention_reference`` per row.  This isolates
#    the attention math from the rest of the network (candidates #1/#3/#4/#5).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s", [2, 4, 6])
def test_selected_core_is_batch_invariant_and_matches_k29_reference(s, capsys):
    b, H, hd, k = 1, 4, 16, 12
    mx.random.seed(7)
    q = (mx.random.normal((b, s, H, hd)) * 0.2).astype(mx.float32)
    KVg = (mx.random.normal((b, s, k, hd)) * 0.2).astype(mx.float32)
    # per-row valid masks with different reachable counts (mimics per-row window +
    # selected-compress edges), at least one True per row.
    counts = [(1 + (i % k)) for i in range(s)]
    vmask = np.zeros((b, s, k), dtype=bool)
    for i in range(s):
        vmask[0, i, : counts[i]] = True
    valid = mx.array(vmask)
    sink = (mx.random.normal((H,)) * 0.5).astype(mx.float32)
    scale = hd ** -0.5
    mx.eval(q, KVg, valid, sink)

    o_batched = dsv41._attn_core_impl(q, KVg, valid, sink, scale)  # [b,s,H,hd]
    mx.eval(o_batched)

    max_batch = 0.0
    max_k29 = 0.0
    for i in range(s):
        qi = q[:, i : i + 1]
        kvi = KVg[:, i : i + 1]
        vi = valid[:, i : i + 1]
        o_i = dsv41._attn_core_impl(qi, kvi, vi, sink, scale)  # [b,1,H,hd]
        mx.eval(o_i)
        d = float(mx.max(mx.abs(o_batched[:, i : i + 1] - o_i)).item())
        max_batch = max(max_batch, d)
        # K29 CPU reference at this row's own single-query batch (the exact shape the
        # K29 GPU path feeds: q [b,1,H,hd], KV [b,k,hd], attend [b,1,k]).
        o_ref = k29.decode_attention_reference(
            qi, kvi.reshape(b, k, hd), kvi.reshape(b, k, hd),
            attend=vi, attn_sink=sink, scale=scale, T=k,
        )
        mx.eval(o_ref)
        d2 = float(mx.max(mx.abs(o_i - o_ref)).item())
        max_k29 = max(max_k29, d2)

    with capsys.disabled():
        print(f"\n[W119] core s={s}: batch-invariance max|Δ|={max_batch:.3e}  "
              f"eager-vs-K29-reference max|Δ|={max_k29:.3e}")
    assert max_batch <= 1e-6, (
        f"selected-key core is NOT batch-invariant: row of the s={s} call differs "
        f"from the s=1 call by max|Δ|={max_batch:.3e} -- a real multi-row bug"
    )
    assert max_k29 <= 1e-5, (
        f"eager core diverged from the K29 CPU reference by max|Δ|={max_k29:.3e}"
    )
