"""Real-artifact 31-token teacher-forced probe for the DeepSeek-V4.1 text forward.

Opt-in (heavy: loads the streaming q2 artifact + engram bank on CPU). Enable with
``DSV41_RUN_PROBE=1``. Asserts the ported forward predicts the correct next token
at >= 27/30 positions on the deterministic 3-function code probe (with BOS), and
prints the exact per-position table. This is the W10 correctness bar for the port
(the old, reused-V4 forward scored 16/30 and locked onto junk ids 13394/104113).
"""

from __future__ import annotations

import os

import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)  # CPU only; the port is device-independent

# The reference probe (decode_probe.py [A]): a 3-function file whose continuation
# is near-deterministic. BOS (id 0) is prepended exactly as the reference serves.
A_TEXT = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef mul(a, b):"
BOS = 0
# The ids the task pins (artifact tokenizer, BOS-prefixed); asserted as a guard.
EXPECTED_IDS = [0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291,
                6328, 3465, 1241, 6036, 14, 291, 3395, 361, 1354, 260, 565, 291,
                6328, 3465, 21740, 6036, 14, 291, 2605]
MIN_MATCHES = 27
TOTAL = 30

MODEL_DIR = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2")
GIB = 1024 ** 3


@pytest.mark.skipif(
    os.environ.get("DSV41_RUN_PROBE") != "1",
    reason="opt-in: set DSV41_RUN_PROBE=1 to run the real-artifact 31-token probe",
)
def test_deepseek_v41_probe31_teacher_forced():
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
    from mlx_lm.utils import load_tokenizer
    from mlx_lm.models.cache import make_prompt_cache

    tok = load_tokenizer(MODEL_DIR)
    ids = [BOS] + list(tok.encode(A_TEXT))
    # guard: the artifact tokenizer must reproduce the pinned ids (BOS + encode)
    assert ids == EXPECTED_IDS, f"probe ids drifted: {ids}"

    resident = load_deepseek_v41_streaming(
        MODEL_DIR,
        memory_limit_bytes=int(100 * GIB),
        max_live_kv_tokens=4096,
        admit=True,
        admission_receipt=None,
        expert_cache_limit_bytes=int(15 * GIB),
        apply_memory_cap=False,
        slot_layout="component-banks",   # routed experts gathered from experts.bin
        cache_scope="layer",
        island_layers=(),
        verify_record_hashes=False,
    )
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime", None)
    try:
        cache = make_prompt_cache(model)
        logits = model(mx.array([ids]), cache=cache)  # [1, S, vocab]
        mx.eval(logits)

        match = 0
        rows = []
        for i in range(TOTAL):
            pred = int(mx.argmax(logits[0, i]).item())
            actual = ids[i + 1]
            ok = pred == actual
            match += ok
            rows.append((i, pred, tok.decode([pred]), actual, tok.decode([actual]), ok))

        print(f"\n[probe31] {match}/{TOTAL} next-token argmax matches (BOS-prefixed)")
        print(f"{'pos':>3}  {'pred':>7}  {'pred_txt':<14}  {'actual':>7}  {'actual_txt':<14}  ok")
        for i, pred, pt, actual, at, ok in rows:
            print(f"{i:>3}  {pred:>7}  {pt!r:<14}  {actual:>7}  {at!r:<14}  {'OK' if ok else 'x'}")
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass

    assert match >= MIN_MATCHES, (
        f"31-token probe scored {match}/{TOTAL} (need >= {MIN_MATCHES}); "
        f"junk positions {[r[0] for r in rows if not r[5]]}"
    )
