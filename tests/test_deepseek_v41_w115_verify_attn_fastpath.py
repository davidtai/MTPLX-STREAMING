"""W115: the DSpark verify-attention fast path
(``MTPLX_DSV41_VERIFY_ATTN_FASTPATH`` / ``MTPLX_DSV41_VERIFY_ATTN_MAX_ROWS``,
``mtplx/models/deepseek_v41.py``).

The DSpark verify is a small-M (``1 < rows = K+1 <= 8``) target forward.  A bench arm
can arm the W101 fused PROJECTIONS while deliberately keeping the eager SDPA CORE
(``cell16k_ring_v2_draft_attn``), so the K+1 verify runs the eager per-row selected
core at ~M x the M=1 cost.  This lever AUTO-ARMS the fused decode SDPA core (the K29
kernel on GPU, the W97 core-compile tape on CPU/GPU) + fused projections for the
verify rows ONLY, scoped to the ``decode_verify`` attention phase.

CPU tests (the default here) prove, WITHOUT dispatching any Metal:

  * the flag / row-cap resolvers (default off/8, truthy parse, read-at-use, fail-fast);
  * the phase- and row-scoped gate ``_verify_attn_fastpath_use`` (only the verify
    batch in the ``decode_verify`` phase, never rows == 1, never a prefill chunk);
  * the gate widening: with the M=1 core levers OFF, the verify auto-arms the K29
    kernel (spied GPU) + fused projections + the core-compile tape, while rows == 1
    and the prefill phase never do;
  * greedy DSpark-with-fastpath == greedy AR over the tiny model at depths 1..5 (the
    verify is authoritative; the fused/compiled core is rounding-class, so greedy
    identity -- not byte identity -- is the bar), with the per-row verify-logit
    max|Δ| vs the eager core reported (rounding-class, argmax-exact);
  * the engagement counters (calls/rows/fallbacks{rows_gt_max, selected_keys_off}).

MLX pinned to CPU (memory/worker-tests-must-pin-mlx-cpu.md); no GPU/Metal, no model
download, no checkpoint, <1.5 GB RSS.  Run under ``nice -n 19``, pytest one file per
process (no ``-n auto``).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mtplx.attention_context import attention_phase  # noqa: E402
from mtplx.models import deepseek_v41 as V41  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.models.deepseek_v41_dspark_decode import (  # noqa: E402
    DSparkDecodeStats,
    dspark_generate,
)
from mtplx.sampling import SamplerConfig  # noqa: E402

_FP = V41._VERIFY_ATTN_FASTPATH_ENV
_MR = V41._VERIFY_ATTN_MAX_ROWS_ENV
GREEDY = SamplerConfig(temperature=0.0)


@pytest.fixture(autouse=True)
def _cpu_and_clean_env(monkeypatch):
    monkeypatch.setattr(mx, "set_default_device", mx.set_default_device)
    mx.set_default_device(mx.cpu)
    # A clean lever env every test (read-at-use): unset unless the test sets it.
    for k in (_FP, _MR, "MTPLX_DSV41_SELECTED_KEYS", "MTPLX_DSV41_ATTN_LEAN_CASTS",
              "MTPLX_DSV41_DECODE_ATTN_KERNEL", "MTPLX_DSV41_ATTN_FUSED_PROJ",
              "MTPLX_DSV41_ATTN_CORE_COMPILE"):
        monkeypatch.delenv(k, raising=False)
    V41._reset_verify_attn_fastpath_calls()
    V41._reset_attn_core_compile_calls()
    yield


# ---------------------------------------------------------------------------
# tiny-model harness (self-contained, CPU, fp32 bit-exact, no checkpoint)
# ---------------------------------------------------------------------------
DIM = 32
N_LAYERS = 5


def _args(vocab: int = 64, **over):
    kwargs = dict(
        vocab_size=vocab, hidden_size=DIM, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=8, q_lora_rank=16,
        o_lora_rank=8, o_groups=2, moe_intermediate_size=16, n_routed_experts=8,
        num_experts_per_tok=2, sliding_window=8, window_size=8, hc_mult=4,
        hc_sinkhorn_iters=2, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
        swiglu_limit=0.0, n_mtp_layers=3, dspark_block_size=8,
        dspark_noise_token_id=vocab - 1, dspark_target_layer_ids=[2, 3, 4],
        dspark_markov_rank=12, dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )
    kwargs.update(over)
    return ModelArgs(**kwargs)


def _seeded_model(seed=0, vocab=64, **over):
    mx.random.seed(seed)
    args = _args(vocab=vocab, **over)
    model = Model(args, quantize=False, mtp=True)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if value.ndim == 1:
            noise = mx.random.normal(value.shape) * 0.1
            centre = 1.0 if leaf.endswith("norm_weight") or leaf == "scale" else 0.0
            new = noise + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    mx.eval(model.parameters())
    return args, model


def _prompt(n, vocab=64, seed=7):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(1, vocab, size=n)]


def _ar_reference(model, prompt, n):
    """Greedy AR = a DSpark run at depth 0 (no verify batch, M=1 forwards)."""
    return dspark_generate(model, prompt, max_tokens=n, sampler=GREEDY, seed=0,
                           speculative_depth=0)


# ---------------------------------------------------------------------------
# 1. resolvers
# ---------------------------------------------------------------------------
def test_fastpath_resolver_default_off_and_parse(monkeypatch):
    assert V41._resolve_verify_attn_fastpath() is False
    for v in ("1", "true", "on", "yes", "TRUE"):
        monkeypatch.setenv(_FP, v)
        assert V41._resolve_verify_attn_fastpath() is True
    for v in ("", "0", "false", "off", "no", "none", "default"):
        monkeypatch.setenv(_FP, v)
        assert V41._resolve_verify_attn_fastpath() is False
    monkeypatch.setenv(_FP, "maybe")
    with pytest.raises(ValueError):
        V41._resolve_verify_attn_fastpath()


def test_max_rows_resolver_default_and_parse(monkeypatch):
    assert V41._resolve_verify_attn_max_rows() == 8
    monkeypatch.setenv(_MR, "4")
    assert V41._resolve_verify_attn_max_rows() == 4
    monkeypatch.setenv(_MR, "16")
    assert V41._resolve_verify_attn_max_rows() == 16
    monkeypatch.setenv(_MR, "1")  # a verify batch is 1 < rows
    with pytest.raises(ValueError):
        V41._resolve_verify_attn_max_rows()
    monkeypatch.setenv(_MR, "nope")
    with pytest.raises(ValueError):
        V41._resolve_verify_attn_max_rows()


# ---------------------------------------------------------------------------
# 2. phase- and row-scoped gate
# ---------------------------------------------------------------------------
def test_use_gate_requires_phase_rows_and_flag(monkeypatch):
    # flag off -> never, even in the verify phase
    with attention_phase("decode_verify"):
        assert V41._verify_attn_fastpath_use(6) is False
    monkeypatch.setenv(_FP, "1")
    # armed but wrong phase (a prefill chunk / AR decode) -> never
    assert V41._verify_attn_fastpath_use(6) is False
    with attention_phase("prefill"):
        assert V41._verify_attn_fastpath_use(6) is False
    with attention_phase("decode_verify"):
        assert V41._verify_attn_fastpath_use(1) is False   # rows == 1 is the M=1 lane
        assert V41._verify_attn_fastpath_use(2) is True
        assert V41._verify_attn_fastpath_use(8) is True
        assert V41._verify_attn_fastpath_use(9) is False   # above the kernel cap (8)
        monkeypatch.setenv(_MR, "4")
        assert V41._verify_attn_fastpath_use(4) is True
        assert V41._verify_attn_fastpath_use(5) is False   # above MAX_ROWS


# ---------------------------------------------------------------------------
# 3. gate widening: the verify auto-arms K29 / fused proj / core-compile
# ---------------------------------------------------------------------------
def _spy_gpu(monkeypatch):
    """Make the CPU host look like a Metal GPU to the pure gate functions (no dispatch
    happens -- the gates only read is_available()/default_device())."""
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)


def test_verify_autoarms_k29_and_fused_proj_when_m1_levers_off(monkeypatch):
    _spy_gpu(monkeypatch)
    q6 = mx.zeros((1, 6, 8, 512))
    q1 = mx.zeros((1, 1, 8, 512))
    # M=1 decode-core levers OFF: nothing auto-arms without the fast path...
    assert V41._decode_attn_kernel_use(q6) is False
    assert V41._fused_proj_use(6) is False
    # ...but with the lever + the verify phase, the verify batch auto-arms both.
    monkeypatch.setenv(_FP, "1")
    with attention_phase("decode_verify"):
        assert V41._decode_attn_kernel_use(q6) is True
        assert V41._fused_proj_use(6) is True
        # rows == 1 (M=1 AR decode) is NOT auto-armed by the verify lever.
        assert V41._decode_attn_kernel_use(q1) is False
        assert V41._fused_proj_use(1) is False
        # above the cap -> declined.
        assert V41._decode_attn_kernel_use(mx.zeros((1, 9, 8, 512))) is False
    # Outside the verify phase (a prefill chunk of the same width) -> not auto-armed.
    with attention_phase("prefill"):
        assert V41._decode_attn_kernel_use(q6) is False
        assert V41._fused_proj_use(6) is False


def test_m1_kernel_lever_still_governs_rows1(monkeypatch):
    """The widening is additive: arming the M=1 decode-kernel lever behaves exactly as
    before (rows 1..8 route on GPU), independent of the verify lever."""
    _spy_gpu(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_DECODE_ATTN_KERNEL", "1")
    assert V41._decode_attn_kernel_use(mx.zeros((1, 1, 8, 512))) is True
    assert V41._decode_attn_kernel_use(mx.zeros((1, 4, 8, 512))) is True
    assert V41._decode_attn_kernel_use(mx.zeros((1, 9, 8, 512))) is False


# ---------------------------------------------------------------------------
# 4. greedy DSpark-with-fastpath == greedy AR (depths 1..5) + engagement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_greedy_dspark_fastpath_equals_ar(monkeypatch, depth):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_ATTN_LEAN_CASTS", "1")
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(17)
    ref = _ar_reference(model, prompt, 64)   # fast path OFF (env unset)
    assert len(set(ref)) > 1, "premise: AR output must not be degenerate"

    monkeypatch.setenv(_FP, "1")
    V41._reset_verify_attn_fastpath_calls()
    V41._reset_attn_core_compile_calls()
    stats = DSparkDecodeStats()
    out = dspark_generate(model, prompt, max_tokens=64, sampler=GREEDY, seed=0,
                          speculative_depth=depth, stats=stats)
    diverge = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)
    assert out == ref, f"depth {depth}: DSpark-fastpath diverged from AR at {diverge}"

    eng = V41._verify_attn_fastpath_engagement()
    cc = V41._attn_core_compile_calls()
    assert eng["calls"] > 0, "the verify fast path never engaged"
    assert eng["rows"] == eng["calls"] * (depth + 1), "rows = calls * (K+1)"
    assert eng["fallbacks"] == {}, "no verify call should have fallen back"
    # On CPU the verify auto-arms the core-compile tape (K29 is GPU-only), so the
    # verify's selected-key core calls ran the compiled tape, not eager.
    assert cc["compiled"] >= eng["calls"], (cc, eng)


# ---------------------------------------------------------------------------
# 5. per-row verify-logit max|Δ| (fast core vs eager core) -- rounding-class
# ---------------------------------------------------------------------------
def test_verify_logits_rounding_class_and_argmax_exact(monkeypatch, capsys):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_ATTN_LEAN_CASTS", "1")
    _args_, model = _seeded_model(seed=1)
    prompt = _prompt(19, seed=11)
    block = [prompt[-1]] + _prompt(5, seed=12)  # a K+1 = 6-row verify block

    def _verify_logits(fastpath: bool):
        if fastpath:
            monkeypatch.setenv(_FP, "1")
        else:
            monkeypatch.delenv(_FP, raising=False)
        cache = model.make_cache()
        model(mx.array([prompt]), cache=cache, return_hidden=True)  # prime
        with attention_phase("decode_verify"):
            logits, _ = model(mx.array([block]), cache=cache, return_hidden=True)
        mx.eval(logits)
        return np.asarray(logits[0], dtype=np.float32)  # [rows, vocab]

    off = _verify_logits(False)
    on = _verify_logits(True)
    assert on.shape == off.shape
    per_row = np.max(np.abs(on - off), axis=-1)          # [rows]
    max_abs = float(per_row.max())
    argmax_on = on.argmax(axis=-1)
    argmax_off = off.argmax(axis=-1)
    print(f"[W115] verify logits per-row max|delta| = {per_row.tolist()} "
          f"(max {max_abs:.3e}); argmax_on={argmax_on.tolist()} "
          f"argmax_off={argmax_off.tolist()}")
    # rounding-class: small but (typically) nonzero; greedy argmax preserved per row.
    assert max_abs < 1e-2, f"delta {max_abs} too large for a rounding-class core"
    assert np.array_equal(argmax_on, argmax_off), "greedy argmax must be preserved"


# ---------------------------------------------------------------------------
# 6. engagement fallback reasons
# ---------------------------------------------------------------------------
def test_fallback_rows_gt_max(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv(_FP, "1")
    monkeypatch.setenv(_MR, "2")   # cap below the K+1 = 4..6 verify batches
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(17)
    V41._reset_verify_attn_fastpath_calls()
    dspark_generate(model, prompt, max_tokens=32, sampler=GREEDY, seed=0,
                    speculative_depth=4, stats=DSparkDecodeStats())
    eng = V41._verify_attn_fastpath_engagement()
    # depth-4 verify blocks are wider than MAX_ROWS=2, so they decline as rows_gt_max
    # (a 1-accepted cycle can still emit a 2-row verify that engages).
    assert eng["fallbacks"].get("rows_gt_max", 0) > 0, eng


def test_fallback_selected_keys_off(monkeypatch):
    monkeypatch.delenv("MTPLX_DSV41_SELECTED_KEYS", raising=False)  # masked core path
    monkeypatch.setenv(_FP, "1")
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(17)
    V41._reset_verify_attn_fastpath_calls()
    dspark_generate(model, prompt, max_tokens=32, sampler=GREEDY, seed=0,
                    speculative_depth=3, stats=DSparkDecodeStats())
    eng = V41._verify_attn_fastpath_engagement()
    assert eng["calls"] == 0, "the fused/compiled core lives on the selected path"
    assert eng["fallbacks"].get("selected_keys_off", 0) > 0, eng
