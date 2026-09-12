"""CPU-pinned unit tests for the W106 full-output persistence (David: "store the
output so we can audit it").

Covers ``scripts/deepseek_v41/ab_decode_env_levers.py``:

  * ``_decode_ids`` / ``_text_output_fields`` -- guarded decode with the loaded
    bench tokenizer; token_ids + decoded_text + head/tail (first/last 600 chars);
  * ``_divergence_context`` -- decoded text ~200 chars either side of a divergence
    TOKEN index;
  * ``_nonclobber_write`` -- atomic (tmp + rename) write that never overwrites an
    existing sidecar (suffix -2, -3);
  * ``_write_output_sidecars`` -- ``<stem>.output.txt`` for the measured stream and
    ``<stem>.ar-reference.output.txt`` for the DSpark comparison stream.

Uses a FAKE tokenizer (no model, no MLX op, no server). MLX is imported ONLY to
pin the default device to CPU (memory/worker-tests-must-pin-mlx-cpu.md). Run under
``nice -n 19`` and WITHOUT ``pytest -n auto``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mlx.core as mx

# HARD rule: pin MLX to CPU before anything can touch Metal.
mx.set_default_device(mx.cpu)

_WT = Path(__file__).resolve().parents[1]
_SCRIPTS = _WT / "scripts" / "deepseek_v41"


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dsv41_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mod():
    return _load("ab_decode_env_levers")


class _FakeTok:
    """Deterministic id->char decoder (each id maps to a letter), so the decoded
    text length is predictable for head/tail/context assertions."""

    def decode(self, ids):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


class _BoomTok:
    def decode(self, ids):
        raise RuntimeError("tokenizer exploded")


# --------------------------------------------------------------------------
# decode + text fields
# --------------------------------------------------------------------------


def test_decode_ids_guards_no_tokenizer_and_failure():
    mod = _mod()
    assert mod._decode_ids(None, [1, 2, 3]) is None      # no tokenizer
    assert mod._decode_ids(_FakeTok(), None) is None      # no ids
    assert mod._decode_ids(object(), [1]) is None         # tok w/o .decode
    assert mod._decode_ids(_BoomTok(), [1, 2]) is None    # decode raises -> None


def test_text_output_fields_full_ids_and_head_tail():
    mod = _mod()
    ids = list(range(700))  # 700 chars once decoded
    fields = mod._text_output_fields(_FakeTok(), ids)
    assert fields["token_ids"] == ids            # FULL id list, not truncated
    assert len(fields["decoded_text"]) == 700
    assert fields["decoded_text_head"] == fields["decoded_text"][:600]
    assert fields["decoded_text_tail"] == fields["decoded_text"][-600:]
    assert len(fields["decoded_text_head"]) == 600
    assert len(fields["decoded_text_tail"]) == 600


def test_text_output_fields_none_text_on_decode_failure():
    mod = _mod()
    fields = mod._text_output_fields(_BoomTok(), [1, 2, 3])
    assert fields["token_ids"] == [1, 2, 3]      # ids still recorded
    assert fields["decoded_text"] is None
    assert fields["decoded_text_head"] is None
    assert fields["decoded_text_tail"] is None


def test_divergence_context_span_either_side():
    mod = _mod()
    ids = list(range(1000))
    ctx = mod._divergence_context(_FakeTok(), ids, 500, span=200)
    # offset for token 500 == 500 chars; context = full[300:700] -> 400 chars.
    assert len(ctx) == 400
    full = _FakeTok().decode(ids)
    assert ctx == full[300:700]


def test_divergence_context_clamps_at_start():
    mod = _mod()
    ids = list(range(1000))
    ctx = mod._divergence_context(_FakeTok(), ids, 10, span=200)
    # offset 10 -> full[0:210] (clamped left).
    assert ctx == _FakeTok().decode(ids)[0:210]


def test_divergence_context_none_without_tokenizer():
    mod = _mod()
    assert mod._divergence_context(None, [1, 2, 3], 1) is None


# --------------------------------------------------------------------------
# sidecar writing: atomic + never-overwrite
# --------------------------------------------------------------------------


def test_nonclobber_write_creates_then_suffixes(tmp_path):
    mod = _mod()
    desired = tmp_path / "receipt.output.txt"
    p1 = mod._nonclobber_write(desired, "first")
    assert p1 == desired and desired.read_text() == "first"
    # a second write must NOT overwrite; it lands on -2.
    p2 = mod._nonclobber_write(desired, "second")
    assert p2 == tmp_path / "receipt.output-2.txt"
    assert p2.read_text() == "second"
    assert desired.read_text() == "first"  # untouched
    # and a third -> -3.
    p3 = mod._nonclobber_write(desired, "third")
    assert p3 == tmp_path / "receipt.output-3.txt"
    # no leftover .tmp files
    assert not list(tmp_path.glob("*.tmp"))


def test_write_output_sidecars_ar_only(tmp_path):
    mod = _mod()
    out = tmp_path / "cell.jsonl"
    receipt = {
        "arm": "control",
        "token_ids_sha256": "abc123",
        "decode_tok_s": 11.5,
        "decoded_text": "HELLO WORLD",
    }
    mod._write_output_sidecars(out, receipt)
    side = tmp_path / "cell.output.txt"
    assert side.exists()
    text = side.read_text()
    assert "# arm: control" in text
    assert "# stream: ar" in text
    assert "# token_ids_sha256: abc123" in text
    assert "# decode_tok_s: 11.5" in text
    assert "# divergence: none" in text
    assert "HELLO WORLD" in text
    # no AR-reference sidecar for a non-dspark run
    assert not (tmp_path / "cell.ar-reference.output.txt").exists()


def test_write_output_sidecars_dspark_writes_both(tmp_path):
    mod = _mod()
    out = tmp_path / "cell.jsonl"
    receipt = {
        "arm": "dspark",
        "token_ids_sha256": "ar_sha",
        "decode_tok_s": 2.2,
        "decoded_text": "AR REFERENCE TEXT",
        "dspark": {
            "token_ids_sha256": "dsp_sha",
            "decode_tok_s": 3.9,
            "decoded_text": "DSPARK STREAM TEXT",
            "divergence": {"index": 5, "kind": "tie_flip"},
        },
    }
    mod._write_output_sidecars(out, receipt)
    primary = (tmp_path / "cell.output.txt").read_text()
    # the primary sidecar is the measured (DSpark) stream
    assert "# stream: dspark" in primary
    assert "# token_ids_sha256: dsp_sha" in primary
    assert "DSPARK STREAM TEXT" in primary
    assert "tie_flip" in primary  # the divergence lands in the header
    ref = (tmp_path / "cell.ar-reference.output.txt").read_text()
    assert "# stream: ar-reference" in ref
    assert "# token_ids_sha256: ar_sha" in ref
    assert "AR REFERENCE TEXT" in ref


def test_write_output_sidecars_never_overwrites_across_arms(tmp_path):
    mod = _mod()
    out = tmp_path / "cell.jsonl"
    r1 = {"arm": "control", "token_ids_sha256": "s1", "decode_tok_s": 1.0,
          "decoded_text": "ARM ONE"}
    r2 = {"arm": "overlap", "token_ids_sha256": "s2", "decode_tok_s": 2.0,
          "decoded_text": "ARM TWO"}
    mod._write_output_sidecars(out, r1)
    mod._write_output_sidecars(out, r2)
    assert (tmp_path / "cell.output.txt").read_text().find("ARM ONE") != -1
    assert (tmp_path / "cell.output-2.txt").read_text().find("ARM TWO") != -1


def test_write_output_sidecars_decode_unavailable_still_writes(tmp_path):
    mod = _mod()
    out = tmp_path / "cell.jsonl"
    receipt = {"arm": "control", "token_ids_sha256": "s", "decode_tok_s": 1.0,
               "decoded_text": None}  # tokenizer was unavailable
    mod._write_output_sidecars(out, receipt)
    text = (tmp_path / "cell.output.txt").read_text()
    assert "decode unavailable" in text
    assert "# arm: control" in text


def test_receipt_stem_strips_jsonl(tmp_path):
    mod = _mod()
    assert mod._receipt_stem(tmp_path / "x.jsonl") == tmp_path / "x"
    assert mod._receipt_stem(tmp_path / "x.json") == tmp_path / "x"
    assert mod._receipt_stem(tmp_path / "x.log") == tmp_path / "x.log"
