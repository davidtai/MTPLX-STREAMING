"""CPU-pinned unit tests for the W106 full-output persistence (David: "store the
output so we can audit it").

Covers ``scripts/deepseek_v41/ab_decode_env_levers.py``:

  * ``_decode_ids`` / ``_text_output_fields`` -- guarded decode with the loaded
    bench tokenizer; token_ids + decoded_text + head/tail (first/last 600 chars);
  * ``_divergence_context`` -- decoded text ~200 chars either side of a divergence
    TOKEN index;
  * ``_reserve_paired`` -- atomic (tmp + rename) reservation that never overwrites
    an existing sidecar and applies the SAME ``-n`` suffix to a paired set;
  * ``_write_output_sidecars`` -- ``<stem>.<sha12>.output.txt`` for the measured
    stream and ``<stem>.<sha12>.ar-reference.output.txt`` for the DSpark comparison
    stream (sha[:12] in both names, paired suffix).

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


class _EmptyTok:
    """A tokenizer whose decode returns an EMPTY string (window-43-class failure:
    a decode that silently yields '' for a non-empty id list)."""

    def decode(self, ids):
        return ""


def test_decode_ids_returns_text_and_error_tuple():
    mod = _mod()
    # (text, error) tuple: never a silent ''.
    assert mod._decode_ids(None, [1, 2, 3]) == (None, "no tokenizer available for output decode")
    assert mod._decode_ids(_FakeTok(), None) == (None, "no ids")
    text, err = mod._decode_ids(object(), [1])   # tok w/o .decode
    assert text is None and "no usable decode method" in err
    text, err = mod._decode_ids(_BoomTok(), [1, 2])  # decode raises -> recorded
    assert text is None and "raised" in err
    # EMPTY result for a non-empty id list is an ERROR, not a silent '' (window 43).
    text, err = mod._decode_ids(_EmptyTok(), [1, 2, 3])
    assert text is None and "returned empty for 3 ids" in err
    # success path
    text, err = mod._decode_ids(_FakeTok(), [1, 2, 3])
    assert text == "BCD" and err is None


def test_text_output_fields_full_ids_and_head_tail():
    mod = _mod()
    ids = list(range(700))  # 700 chars once decoded
    fields = mod._text_output_fields(_FakeTok(), ids)
    assert fields["token_ids"] == ids            # FULL id list, not truncated
    assert len(fields["decoded_text"]) == 700
    assert fields["decoded_text_error"] is None
    assert fields["decoded_text_head"] == fields["decoded_text"][:600]
    assert fields["decoded_text_tail"] == fields["decoded_text"][-600:]
    assert len(fields["decoded_text_head"]) == 600
    assert len(fields["decoded_text_tail"]) == 600


def test_text_output_fields_records_error_on_decode_failure():
    mod = _mod()
    fields = mod._text_output_fields(_BoomTok(), [1, 2, 3])
    assert fields["token_ids"] == [1, 2, 3]      # ids still recorded
    assert fields["decoded_text"] is None
    assert fields["decoded_text_head"] is None
    assert fields["decoded_text_tail"] is None
    assert "raised" in fields["decoded_text_error"]  # LOUD, not silent


def test_text_output_fields_records_error_on_empty_decode():
    mod = _mod()
    # window-43-class: decode returns '' -> recorded as an error, decoded_text None
    # (never a silent empty string masquerading as real output).
    fields = mod._text_output_fields(_EmptyTok(), [1, 2, 3])
    assert fields["decoded_text"] is None
    assert "returned empty for 3 ids" in fields["decoded_text_error"]


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


def test_suffixed_and_reserve_paired(tmp_path):
    mod = _mod()
    a = tmp_path / "r.abc.output.txt"
    b = tmp_path / "r.abc.ar-reference.output.txt"
    assert mod._suffixed(a, 1) == a
    assert mod._suffixed(a, 2) == tmp_path / "r.abc.output-2.txt"
    # first reservation: both at n=1
    got = mod._reserve_paired([a, b])
    assert got == [a, b]
    assert a.exists() and b.exists()  # reserved (empty)
    # second reservation of the SAME pair: the SAME -2 applied to BOTH (paired)
    got2 = mod._reserve_paired([a, b])
    assert got2 == [tmp_path / "r.abc.output-2.txt",
                    tmp_path / "r.abc.ar-reference.output-2.txt"]


def test_reserve_paired_bumps_both_when_only_one_taken(tmp_path):
    mod = _mod()
    a = tmp_path / "r.abc.output.txt"
    b = tmp_path / "r.abc.ar-reference.output.txt"
    a.write_text("pre-existing primary only")  # only ONE of the pair is taken
    got = mod._reserve_paired([a, b])
    # n=1 is refused because `a` exists, so BOTH move to -2 (pair never splits)
    assert got == [tmp_path / "r.abc.output-2.txt",
                   tmp_path / "r.abc.ar-reference.output-2.txt"]
    assert b.exists() is False  # the un-suffixed b was NOT created (kept paired)


def test_write_output_sidecars_ar_only_sha_in_name(tmp_path):
    mod = _mod()
    out = tmp_path / "cell.jsonl"
    receipt = {
        "arm": "control",
        "token_ids_sha256": "abc123def456ghi",  # sha[:12] = abc123def456
        "decode_tok_s": 11.5,
        "decoded_text": "HELLO WORLD",
    }
    mod._write_output_sidecars(out, receipt)
    side = tmp_path / "cell.abc123def456.output.txt"
    assert side.exists()
    text = side.read_text()
    assert "# arm: control" in text
    assert "# stream: ar" in text
    assert "# token_ids_sha256: abc123def456ghi" in text
    assert "# decode_tok_s: 11.5" in text
    assert "# divergence: none" in text
    assert "HELLO WORLD" in text
    # no AR-reference sidecar for a non-dspark run
    assert not (tmp_path / "cell.abc123def456.ar-reference.output.txt").exists()


def test_write_output_sidecars_dspark_writes_both_paired(tmp_path):
    mod = _mod()
    out = tmp_path / "cell.jsonl"
    receipt = {
        "arm": "dspark",
        "token_ids_sha256": "arsha0000000",  # sha[:12] shared by BOTH names
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
    primary = (tmp_path / "cell.arsha0000000.output.txt").read_text()
    # the primary sidecar is the measured (DSpark) stream
    assert "# stream: dspark" in primary
    assert "# token_ids_sha256: dsp_sha" in primary
    assert "DSPARK STREAM TEXT" in primary
    assert "tie_flip" in primary  # the divergence lands in the header
    ref = (tmp_path / "cell.arsha0000000.ar-reference.output.txt").read_text()
    assert "# stream: ar-reference" in ref
    assert "# token_ids_sha256: arsha0000000" in ref
    assert "AR REFERENCE TEXT" in ref


def test_write_output_sidecars_same_sha_pairs_suffix(tmp_path):
    """Two DSpark runs with the SAME sha get the SAME -2 on BOTH sidecars."""
    mod = _mod()
    out = tmp_path / "cell.jsonl"
    receipt = {
        "arm": "dspark", "token_ids_sha256": "samesha00000", "decode_tok_s": 2.0,
        "decoded_text": "AR ONE",
        "dspark": {"token_ids_sha256": "d", "decode_tok_s": 3.0,
                   "decoded_text": "DSPARK ONE", "divergence": None},
    }
    mod._write_output_sidecars(out, receipt)
    mod._write_output_sidecars(out, receipt)  # same sha -> -2 pair
    assert (tmp_path / "cell.samesha00000.output.txt").exists()
    assert (tmp_path / "cell.samesha00000.ar-reference.output.txt").exists()
    assert (tmp_path / "cell.samesha00000.output-2.txt").exists()
    assert (tmp_path / "cell.samesha00000.ar-reference.output-2.txt").exists()
    # no leftover .tmp files
    assert not list(tmp_path.glob("*.tmp"))


def test_write_output_sidecars_decode_unavailable_still_writes(tmp_path):
    mod = _mod()
    out = tmp_path / "cell.jsonl"
    receipt = {"arm": "control", "token_ids_sha256": "sfoobarbaz00", "decode_tok_s": 1.0,
               "decoded_text": None}  # tokenizer was unavailable
    mod._write_output_sidecars(out, receipt)
    text = (tmp_path / "cell.sfoobarbaz00.output.txt").read_text()
    assert "decode unavailable" in text
    assert "# arm: control" in text


def test_receipt_stem_strips_jsonl(tmp_path):
    mod = _mod()
    assert mod._receipt_stem(tmp_path / "x.jsonl") == tmp_path / "x"
    assert mod._receipt_stem(tmp_path / "x.json") == tmp_path / "x"
    assert mod._receipt_stem(tmp_path / "x.log") == tmp_path / "x.log"


class _SpecialsOnlyTok:
    """decode(ids) -> '' (specials skipped), decode(ids, skip_special_tokens=False)
    -> the special-token rendering. Models an all-EOS/pad stream (LOW round 4)."""

    def decode(self, ids, skip_special_tokens=True):
        return "" if skip_special_tokens else "<|eos|>" * len(ids)


def test_decode_ids_special_tokens_only_is_not_an_error():
    mod = _mod()
    text, err = mod._decode_ids(_SpecialsOnlyTok(), [7, 7, 7])
    # legit empty under skip_special_tokens -> surface the with-specials rendering,
    # NOT an error.
    assert err is None
    assert text == "<|eos|>" * 3


def test_decode_ids_truly_empty_still_errors():
    mod = _mod()
    # _EmptyTok.decode() takes no skip_special_tokens kwarg -> the retry raises ->
    # both empty -> genuine error.
    text, err = mod._decode_ids(_EmptyTok(), [1, 2, 3])
    assert text is None
    assert "returned empty for 3 ids" in err
