"""W37 -- DeepSeek-V4.1 decode stage-timing probe (CPU, synthetic).

Window 12 measured 246 ms/token but could not attribute it.  ``MTPLX_DSV41_STAGE_
TIMING`` (``mtplx.models.deepseek_v41_stage_timing``) fences every decode stage so
``model.stage_timing_report()`` gives a per-token census.  These gates prove, on a
tiny CPU config (no artifact, resident SwitchGLU):

  * OFF is free and byte-identical: no session armed -> ``stage_timing_report()``
    is ``None``, ``recording()`` is ``False``, and decode logits are ``mx.array_
    equal`` to a fenced (probe-ON) decode -- the probe only times, it never changes
    a value (proven with ``MTPLX_DSV41_HC_COMPILE`` both off AND on, since ON
    forces the eager Hyper-Connection path);
  * ON partitions the forward: every expected stage is present with the exact
    per-token count (embed 1, attention == n_layers split by CSA mode,
    hc.premix_sinkhorn / hc.combine 2*n_layers, the four moe.* == n_layers, head /
    final_norm / sample 1), and the per-stage sum ~= the frame wall;
  * the engram hook splits into hash / row_fetch / apply (tested in isolation with
    a fake row cache, no 104 GiB bank);
  * the report schema is the receipt schema the A/B harness embeds.

Pins MLX to CPU; tiny random config; no artifact load.  Run under ``nice -n 19``,
without ``-n auto``.
"""
from __future__ import annotations

import contextlib

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
import mtplx.models.deepseek_v41_stage_timing as stime  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.engram_v41 import EngramV41  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures (mirror tests/models/test_deepseek_v41_hc_compile.py)
# ---------------------------------------------------------------------------
def _csa_args(**over) -> ModelArgs:
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


def _randomize(model, seed=0, scale=0.1):
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = scale * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


def _new_model(seed=1, **over):
    args = _csa_args(**over)
    model = Model(args)
    _randomize(model, seed=seed)
    return model, args


def _prefill(model, args, s, seed):
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    cache = model.make_cache()
    logits = model(ids, cache=cache, prefill_chunk=0)
    mx.eval(logits)
    token = int(mx.argmax(logits[0, -1]).item())
    return cache, token


@contextlib.contextmanager
def _hc(flag: bool, max_rows: int = 7):
    old_f, old_r = dv41._HC_COMPILE, dv41._HC_COMPILE_MAX_ROWS
    dv41._HC_COMPILE = flag
    dv41._HC_COMPILE_MAX_ROWS = max_rows
    dv41._HC_COMPILED.clear()
    try:
        yield
    finally:
        dv41._HC_COMPILE, dv41._HC_COMPILE_MAX_ROWS = old_f, old_r
        dv41._HC_COMPILED.clear()


@contextlib.contextmanager
def _session():
    """Arm a stage-timing session and always tear it down (even on failure), so a
    leaked ``_ACTIVE`` never pollutes a later test's byte-identity gate."""
    stime.begin()
    try:
        yield stime.active()
    finally:
        stime.end()


def _decode_record(model, cache, tokens):
    """Fenced decode: one frame per step, argmax under a ``sample`` stage -- the
    exact shape ``ab_decode_env_levers.py --stage-timing`` drives."""
    out = []
    token = tokens[0]
    for _ in tokens:
        with stime.frame():
            logits = model(mx.array([[token]]), cache=cache)
            with stime.stage("sample"):
                token = int(mx.argmax(logits[0, -1]).item())
            out.append(np.array(logits))
    return out


def _decode_plain(model, cache, tokens):
    out = []
    token = tokens[0]
    for _ in tokens:
        logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
        token = int(mx.argmax(logits[0, -1]).item())
        out.append(np.array(logits))
    return out


def _expected_counts(args, tokens: int, *, engram_layers: int = 0) -> dict:
    L = args.num_hidden_layers
    exp = {
        "embed": tokens,
        "final_norm": tokens,
        "head": tokens,
        "sample": tokens,
        "hc.premix_sinkhorn": 2 * L * tokens,
        "hc.combine": 2 * L * tokens,
        "moe.gate_topk": L * tokens,
        "moe.routed_switch": L * tokens,
        "moe.shared_expert": L * tokens,
        "moe.combine": L * tokens,
    }
    for mode in set(args.layer_modes):
        exp["attn." + mode] = args.layer_modes.count(mode) * tokens
    if engram_layers:
        exp["engram.advance"] = tokens
        exp["engram.hash"] = engram_layers * tokens
        exp["engram.row_fetch"] = engram_layers * tokens
        exp["engram.apply"] = engram_layers * tokens
    return exp


# ---------------------------------------------------------------------------
# 1. OFF is free: no session -> report None, recording() False, no fences
# ---------------------------------------------------------------------------
def test_report_none_and_not_recording_when_off():
    model, args = _new_model(seed=1)
    assert model.stage_timing_report() is None
    assert stime.report() is None
    assert stime.recording() is False
    # _hc_use_compile is unaffected by the probe when no session is armed.
    x1 = mx.zeros((1, 1, 4, 32))
    with _hc(True, max_rows=7):
        assert dv41._hc_use_compile(x1) is True     # compiled, probe not recording
    with _hc(False):
        assert dv41._hc_use_compile(x1) is False


# ---------------------------------------------------------------------------
# 2. probe ON vs OFF is byte-identical (HC compile off -> both eager)
# ---------------------------------------------------------------------------
def test_probe_on_off_logits_identical_hc_off():
    stream = [3, 17, 5, 29]

    def run(record):
        model, args = _new_model(seed=1)
        with _hc(False):
            cache, _ = _prefill(model, args, s=12, seed=0)
            if record:
                with _session():
                    stime.active().enter_forward(1)  # armed before the first frame
                    return _decode_record(model, cache, stream)
            return _decode_plain(model, cache, stream)

    off = run(False)
    on = run(True)
    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), f"decode step {i} differs (max {np.max(np.abs(a-b))})"


# ---------------------------------------------------------------------------
# 3. probe ON forces the eager HC path regardless of MTPLX_DSV41_HC_COMPILE.
#    _hc_use_compile() must report False while recording (so the fences always
#    tile the eager body, never an opaque compiled tape), and the forced-eager
#    decode with the flag ON must be bit-exact with the plain eager decode with
#    the flag OFF -- proving the probe's forced path is the shipped control path.
#
#    (This deliberately never runs the compiled tape: at the pinned base commit
#    bfd361424 the compiled HC body has a pre-existing NameError, fixed later on
#    feat/deepseek-v41-streaming; the probe forces eager, so it is unaffected.)
# ---------------------------------------------------------------------------
def test_recording_forces_eager_hc_use_compile_false():
    x1 = mx.zeros((1, 1, 4, 32))  # 1 row, decode
    with _hc(True, max_rows=7):
        assert dv41._hc_use_compile(x1) is True         # off the probe: compiled
        with _session():
            stime.active().enter_forward(1)             # decode forward armed
            assert stime.recording() is True
            assert dv41._hc_use_compile(x1) is False    # recording -> eager forced


def test_probe_on_hc_on_matches_plain_eager_hc_off():
    stream = [7, 2, 41, 13]

    # plain eager (HC OFF, probe off).
    model_off, args = _new_model(seed=2)
    with _hc(False):
        cache, _ = _prefill(model_off, args, s=12, seed=1)
        off = _decode_plain(model_off, cache, stream)

    # probe ON with HC ON: recording forces the eager body -> same values, no tape.
    model_on, _ = _new_model(seed=2)
    with _hc(True, max_rows=7):
        cache, _ = _prefill(model_on, args, s=12, seed=1)
        with _session():
            stime.active().enter_forward(1)
            on = _decode_record(model_on, cache, stream)

    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), f"decode step {i} differs (max {np.max(np.abs(a-b))})"


# ---------------------------------------------------------------------------
# 4. ON: every stage present with the exact per-token count + sum ~= frame wall
# ---------------------------------------------------------------------------
def test_stage_counts_and_partition_and_schema():
    model, args = _new_model(seed=3)
    stream = [3, 17, 5, 29, 11, 4]
    with _hc(False):
        cache, _ = _prefill(model, args, s=10, seed=2)
        with _session():
            stime.active().enter_forward(1)
            _decode_record(model, cache, stream)
            report = model.stage_timing_report()

    tokens = len(stream)
    assert report is not None
    # -- schema (the receipt schema the A/B harness embeds) --
    assert report["enabled"] is True
    assert report["tokens"] == tokens
    for key in ("stage_sum_ms", "stage_sum_ms_per_token", "frame_wall_ms",
                "frame_wall_ms_per_token", "stages"):
        assert key in report, key
    for name, s in report["stages"].items():
        assert set(s) == {"total_ms", "count", "mean_ms", "mean_ms_per_token"}, name
        assert s["count"] > 0 and s["total_ms"] >= 0.0

    # -- exact per-token counts (no engram on this backbone) --
    expected = _expected_counts(args, tokens, engram_layers=0)
    got = {name: report["stages"][name]["count"] for name in report["stages"]}
    assert got == expected, (got, expected)
    # attention split by CSA mode sums to one call per layer per token.
    attn_total = sum(v for k, v in got.items() if k.startswith("attn."))
    assert attn_total == args.num_hidden_layers * tokens

    # -- partition: the per-stage sum tiles the frame wall (fences serialise every
    #    stage, so the only slack is Python glue between brackets) --
    ratio = report["stage_sum_ms"] / report["frame_wall_ms"]
    assert 0.6 <= ratio <= 1.05, f"stage sum / frame wall = {ratio:.3f} (a stage is un-bracketed?)"


# ---------------------------------------------------------------------------
# 5. engram hook splits into hash / row_fetch / apply (isolated; no 104 GiB bank)
# ---------------------------------------------------------------------------
class _Geo:
    values_per_row = 4


class _FakeRowCache:
    geometry = _Geo()

    def dequantize(self, row_ids):
        b, l, cols = row_ids.shape
        return mx.zeros((b, l, cols, self.geometry.values_per_row)) + 0.01


class _FakeState:
    token_mask = None

    def __init__(self, b, l, cols):
        self._r = np.zeros((b, l, cols), dtype=np.int64)

    def current_row_ids(self, _idx):
        return self._r


def test_engram_hook_stage_split():
    hd, cols, hc, dim = 4, 6, 2, 8
    hook = EngramV41(
        layer_id=1, layer_hash_index=0, row_cache=_FakeRowCache(),
        wkv=EngramV41.dense_wkv(mx.random.normal((dim * (hc + 1), cols * hd)) * 0.1),
        q_weight=mx.random.normal((hc, dim)) * 0.1,
        k_weight=mx.random.normal((hc, dim)) * 0.1,
        dim=dim, hc_mult=hc, norm_eps=1e-6,
    )
    h = mx.random.normal((1, 1, hc, dim))
    state = _FakeState(1, 1, cols)

    off = np.array(hook(h, mx.array([[5]]), state))
    assert stime.report() is None  # nothing recorded off the session

    with _session():
        stime.active().enter_forward(1)
        outs = [np.array(hook(h, mx.array([[5]]), state)) for _ in range(3)]
        report = stime.report()

    counts = {k: v["count"] for k, v in report["stages"].items()}
    assert counts == {"engram.hash": 3, "engram.row_fetch": 3, "engram.apply": 3}
    for o in outs:
        assert np.array_equal(o, off), "engram fences changed the residual write"


# ---------------------------------------------------------------------------
# 6. prefill (s > 1) never records, even inside an armed session
# ---------------------------------------------------------------------------
def test_prefill_forward_not_recorded():
    model, args = _new_model(seed=4)
    with _hc(False), _session():
        cache = model.make_cache()
        ids = mx.array(np.random.RandomState(3).randint(0, args.vocab_size, size=(1, 9)))
        logits = model(ids, cache=cache, prefill_chunk=0)  # enter_forward(9) -> not recording
        mx.eval(logits)
        report = stime.report()
    # s > 1 forward booked nothing: no stages, zero tokens.
    assert report["tokens"] == 0
    assert report["stages"] == {}
