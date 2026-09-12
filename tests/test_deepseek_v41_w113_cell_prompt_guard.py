"""CPU-only tests for the W113 cell-prompt guard + prompt provenance + EOS
surfacing in ``scripts/deepseek_v41/ab_decode_env_levers.py``.

Windows 39-42 MEASURED the RAW prefill_bench builder prompt (BOS + a filler
ladder + DEFAULT_FINAL_REQUEST = 16,385 tokens, no chat template, no generation
prompt) whose greedy first token is EOS (id 1); the decode loops had no EOS
check, so 256 forced post-EOS filler tokens were timed and a served path would
have returned an EMPTY answer.  The standard 16K cell is the chat-templated ids
file ``docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/
prompt-ids-deepseek-v41.json`` (schema mtplx-server-cell-prompt-ids-v1,
cell=sweep, target_tokens=16384, seed=20260829), reached only via
--prompt-ids-file.

These tests cover, WITHOUT a model / MLX / Metal / server / network:
  * the guard: refuses a cell16k_* arm / --context-tokens 16384 run without
    --prompt-ids-file, auto-defaults --prompt-ids-file to the standard file at
    ctx 16384, and --allow-raw-prompt escapes both;
  * the special-token-id resolver (tokenizer files only, no model);
  * chat-templated detection (assistant marker + generation-prompt tail);
  * the provenance stamps (prompt_source / prompt_ids_file / prompt_ids_sha256
    [PROMPT ids] / prompt_seed / prompt_tokens / prompt_chat_templated);
  * EOS surfacing (first_token_eos / eos_index / tokens_before_eos /
    answer_valid) for EOS at position 0 / mid / never -- incl. the real
    window-43 receipts as fixtures (ref eos None, v2 eos 238);
  * --stop-on-eos decode accounting via a fake model (no MLX).

The scripts are not a package (``scripts/`` has no ``__init__.py``), so they load
by file path.  The helpers under test never import MLX; the one test that drives
``_generate`` uses a fake model/ops/mem_probe, so MLX is never imported.  Run
under ``nice -n 19`` and without ``pytest -n auto``
(memory/worker-tests-must-pin-mlx-cpu.md, memory/Delegation).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_WT = Path(__file__).resolve().parents[1]
_SCRIPTS = _WT / "scripts" / "deepseek_v41"
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "deepseek_v41_w113"
_STANDARD_REL = (
    "docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/"
    "prompt-ids-deepseek-v41.json"
)
_EOS_ID = 1  # DeepSeek-V4.1 <｜end▁of▁sentence｜> (id 1); bos is id 0.
_ASSISTANT_ID = 128804
_THINK_ID = 128821
_END_THINK_ID = 128822


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dsv41_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def env_levers():
    return _load("ab_decode_env_levers")


def _args(env_levers, argv):
    return env_levers.build_parser().parse_args(argv)


def _synthetic_model_dir(tmp_path):
    """A tiny model dir with just the two tokenizer files the resolver reads."""
    d = tmp_path / "fake-model"
    d.mkdir()
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "add_bos_token": False,
                "add_eos_token": False,
                "bos_token": {"__type": "AddedToken",
                              "content": "<｜begin▁of▁sentence｜>"},
                "eos_token": {"__type": "AddedToken",
                              "content": "<｜end▁of▁sentence｜>"},
                "pad_token": {"__type": "AddedToken",
                              "content": "<｜end▁of▁sentence｜>"},
            }
        )
    )
    (d / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 0, "content": "<｜begin▁of▁sentence｜>"},
                    {"id": 1, "content": "<｜end▁of▁sentence｜>"},
                    {"id": 128799, "content": "<｜System｜>"},
                    {"id": 128803, "content": "<｜User｜>"},
                    {"id": 128804, "content": "<｜Assistant｜>"},
                    {"id": 128821, "content": "<think>"},
                    {"id": 128822, "content": "</think>"},
                ]
            }
        )
    )
    return d


# ---------------------------------------------------------------------------
# 1. cell-prompt guard: refusal, default resolution, escape hatch
# ---------------------------------------------------------------------------


def test_guard_refuses_cell16k_arm_without_file(env_levers):
    # A cell16k_* arm at ctx 1024 (the std file is target 16384 only, so no auto-
    # default applies) must REFUSE without --prompt-ids-file.
    a = _args(env_levers,
              ["--out", "/dev/null", "--context-tokens", "1024",
               "--arms", "cell16k_ring"])
    with pytest.raises(SystemExit) as exc:
        env_levers._apply_cell_prompt_guard(a)
    msg = str(exc.value)
    assert "REFUSED" in msg and "cell16k" in msg
    # the error names the standard file so the operator knows the fix.
    assert _STANDARD_REL in msg
    assert "--prompt-seed 20260829" in msg


def test_guard_refuses_ctx16384_when_standard_file_missing(env_levers, monkeypatch):
    monkeypatch.setattr(env_levers, "_standard_cell16k_prompt_path",
                        lambda: Path("/nonexistent/prompt-ids.json"))
    a = _args(env_levers,
              ["--out", "/dev/null", "--context-tokens", "16384",
               "--arms", "control"])
    with pytest.raises(SystemExit) as exc:
        env_levers._apply_cell_prompt_guard(a)
    assert "REFUSED" in str(exc.value) and "16384" in str(exc.value)


def test_guard_defaults_prompt_ids_file_at_ctx16384(env_levers):
    # The standard file exists in the worktree; ctx 16384 without a file auto-
    # defaults to it (repo-root relative) so launchers get the cell by default.
    a = _args(env_levers,
              ["--out", "/dev/null", "--context-tokens", "16384",
               "--arms", "control"])
    assert a.prompt_ids_file is None
    assert a.prompt_seed is None
    env_levers._apply_cell_prompt_guard(a)
    assert a.prompt_ids_file is not None
    assert a.prompt_ids_file.endswith(_STANDARD_REL)
    assert Path(a.prompt_ids_file).exists()
    assert Path(a.prompt_ids_file) == env_levers._standard_cell16k_prompt_path()
    # LOW-a: the auto-default also stamps the seed (receipt prompt_seed not null).
    assert a.prompt_seed == env_levers.STANDARD_CELL16K_PROMPT_SEED == 20260829


def test_guard_default_bare_cell16k_arm_is_matched(env_levers):
    # LOW-c: the bare `cell16k` preset (no underscore) is a 16K-cell arm too.
    assert env_levers._cell16k_arm("cell16k") is True
    assert env_levers._cell16k_arm("cell16k_ring") is True
    assert env_levers._cell16k_arm("control") is False
    # a bare cell16k arm at ctx 1024 without a file must refuse (no auto-default
    # at 1024, since the standard file is target 16384).
    a = _args(env_levers,
              ["--out", "/dev/null", "--context-tokens", "1024",
               "--arms", "cell16k"])
    with pytest.raises(SystemExit):
        env_levers._apply_cell_prompt_guard(a)


def test_guard_refuses_on_pinned_sha_mismatch(env_levers, tmp_path, monkeypatch):
    # LOW-a: a swapped/edited standard file (wrong ids -> wrong sha) is refused by
    # the auto-default's pin, not silently measured.
    bad = tmp_path / "prompt-ids-deepseek-v41.json"
    bad.write_text(json.dumps({
        "schema": "mtplx-server-cell-prompt-ids-v1",
        "prompts": [{"cell": "sweep", "target_tokens": 16384,
                     "seed": 20260829, "token_ids": [0, 1, 2, 3]}],  # wrong ids
    }))
    monkeypatch.setattr(env_levers, "_standard_cell16k_prompt_path", lambda: bad)
    a = _args(env_levers,
              ["--out", "/dev/null", "--context-tokens", "16384", "--arms", "control"])
    with pytest.raises(SystemExit) as exc:
        env_levers._apply_cell_prompt_guard(a)
    assert "does NOT match the pinned" in str(exc.value)


def test_verify_standard_cell_prompt_passes_on_real_file(env_levers):
    # the real standard file matches the pinned sha (no raise).
    env_levers._verify_standard_cell_prompt(env_levers._standard_cell16k_prompt_path())


def test_guard_explicit_file_wins(env_levers):
    a = _args(env_levers,
              ["--out", "/dev/null", "--context-tokens", "16384",
               "--arms", "cell16k_ring", "--prompt-ids-file", "/x/f.json",
               "--prompt-seed", "20260829"])
    env_levers._apply_cell_prompt_guard(a)
    assert a.prompt_ids_file == "/x/f.json"  # untouched


def test_guard_allow_raw_prompt_escape_hatch(env_levers, capsys):
    a = _args(env_levers,
              ["--out", "/dev/null", "--context-tokens", "16384",
               "--arms", "cell16k_ring", "--allow-raw-prompt"])
    env_levers._apply_cell_prompt_guard(a)
    # escape hatch: no refusal, no auto-default (stays on the raw builder).
    assert a.prompt_ids_file is None
    out = capsys.readouterr().out
    assert "allow-raw-prompt" in out and "EMPTY" in out  # loud warning


def test_guard_noop_for_non_cell_arm_at_1024(env_levers):
    a = _args(env_levers,
              ["--out", "/dev/null", "--context-tokens", "1024",
               "--arms", "control"])
    env_levers._apply_cell_prompt_guard(a)
    assert a.prompt_ids_file is None


def test_guard_wired_into_main_refuses_before_mlx(env_levers, tmp_path):
    # A real (non --dry-run) main() call for a cell16k_* arm without a file must
    # raise SystemExit from the guard BEFORE importing mlx / loading a model.
    out = tmp_path / "r.jsonl"
    with pytest.raises(SystemExit):
        env_levers.main(
            ["--context-tokens", "1024", "--arms", "cell16k_ring",
             "--out", str(out)]
        )


# ---------------------------------------------------------------------------
# 2. special-token-id resolver (tokenizer files only, no model)
# ---------------------------------------------------------------------------


def test_special_token_ids_from_tokenizer_files(env_levers, tmp_path):
    d = _synthetic_model_dir(tmp_path)
    sp = env_levers._special_token_ids(d)
    assert sp["bos"] == 0
    assert sp["eos"] == _EOS_ID
    assert sp["assistant"] == _ASSISTANT_ID
    assert sp["think"] == _THINK_ID
    assert sp["end_think"] == _END_THINK_ID


def test_special_token_ids_missing_dir_is_empty(env_levers, tmp_path):
    assert env_levers._special_token_ids(tmp_path / "does-not-exist") == {}


def test_resolve_eos_id_override_and_from_files(env_levers, tmp_path):
    d = _synthetic_model_dir(tmp_path)
    a = _args(env_levers, ["--out", "/dev/null", "--model", str(d)])
    assert env_levers._resolve_eos_id(a) == _EOS_ID
    a2 = _args(env_levers, ["--out", "/dev/null", "--model", str(d), "--eos-id", "7"])
    assert env_levers._resolve_eos_id(a2) == 7


def test_stop_on_eos_refuses_when_eos_unresolvable(env_levers, tmp_path):
    # MEDIUM-3: an empty model dir -> no EOS id -> --stop-on-eos must refuse, not
    # silently no-op while stamping stop_on_eos:true.
    empty = tmp_path / "empty-model"
    empty.mkdir()
    a = _args(env_levers,
              ["--out", "/dev/null", "--model", str(empty), "--stop-on-eos"])
    eos_id = env_levers._resolve_eos_id(a)  # None (empty dir)
    assert eos_id is None
    with pytest.raises(SystemExit) as exc:
        env_levers._require_eos_id_for_stop(bool(a.stop_on_eos), eos_id)
    assert "--eos-id" in str(exc.value)
    # with --eos-id given it does not refuse; and off is always fine.
    env_levers._require_eos_id_for_stop(True, 1)
    env_levers._require_eos_id_for_stop(False, None)


# ---------------------------------------------------------------------------
# 3. chat-templated detection
# ---------------------------------------------------------------------------


def test_chat_templated_true_with_generation_prompt(env_levers):
    special = {"assistant": _ASSISTANT_ID, "think": _THINK_ID,
               "end_think": _END_THINK_ID}
    # ... <｜Assistant｜></think>  (opener present, no content after)
    ids = [5, 6, 7, _ASSISTANT_ID, _END_THINK_ID]
    assert env_levers._prompt_chat_templated(ids, special) is True
    # bare assistant opener (empty tail) also counts.
    assert env_levers._prompt_chat_templated([5, _ASSISTANT_ID], special) is True


def test_chat_templated_false_for_raw_builder(env_levers):
    special = {"assistant": _ASSISTANT_ID, "think": _THINK_ID,
               "end_think": _END_THINK_ID}
    # raw builder: no assistant marker at all.
    assert env_levers._prompt_chat_templated([5, 6, 7, 8], special) is False
    # content after the assistant marker => completed turn, not a generation prompt.
    assert env_levers._prompt_chat_templated(
        [_ASSISTANT_ID, _END_THINK_ID, 999], special) is False


def test_chat_templated_none_without_assistant_special(env_levers):
    assert env_levers._prompt_chat_templated([1, 2, 3], {}) is None


def test_chat_templated_on_real_standard_cell_prompt(env_levers):
    # The real standard cell (prompt[1]) is chat-templated with a generation prompt.
    # LOW-d: skip when the model tokenizer files are absent (no special ids to
    # detect the <｜Assistant｜> marker against).
    std = env_levers._standard_cell16k_prompt_path()
    if not std.exists():
        pytest.skip("standard cell ids file not present")
    special = env_levers._special_token_ids(env_levers.DEFAULT_MODEL)
    if "assistant" not in special:
        pytest.skip("model tokenizer files absent -- no special ids to detect")
    data = json.loads(std.read_text())
    entry = next(e for e in data["prompts"]
                 if e.get("cell") == "sweep" and e.get("target_tokens") == 16384)
    assert env_levers._prompt_chat_templated(entry["token_ids"], special) is True


# ---------------------------------------------------------------------------
# 4. provenance stamps
# ---------------------------------------------------------------------------


def test_provenance_prompt_ids_file(env_levers, tmp_path):
    d = _synthetic_model_dir(tmp_path)
    ids = [0, 5, 6, _ASSISTANT_ID, _END_THINK_ID]
    a = _args(env_levers,
              ["--out", "/dev/null", "--model", str(d),
               "--prompt-ids-file", "/x/f.json", "--prompt-seed", "20260829"])
    prov = env_levers._prompt_provenance(a, ids, {"schema": "x"})
    assert prov["prompt_source"] == "prompt-ids-file"
    assert prov["prompt_ids_file"] == "/x/f.json"
    assert prov["prompt_seed"] == 20260829
    assert prov["prompt_tokens"] == len(ids)
    assert prov["prompt_chat_templated"] is True
    assert prov["allow_raw_prompt"] is False
    # prompt_ids_sha256 is the sha of the PROMPT ids, not the generated ids.
    expect = hashlib.sha256(json.dumps([int(t) for t in ids]).encode()).hexdigest()
    assert prov["prompt_ids_sha256"] == expect
    assert prov["prompt_build"] == {"schema": "x"}


def test_provenance_raw_builder(env_levers, tmp_path):
    d = _synthetic_model_dir(tmp_path)
    ids = [0, 5, 6, 7]  # no assistant marker => raw builder shape
    a = _args(env_levers, ["--out", "/dev/null", "--model", str(d)])
    prov = env_levers._prompt_provenance(a, ids, {"prompt_source": "prefill_bench"})
    assert prov["prompt_source"] == "raw-builder"
    assert prov["prompt_ids_file"] is None
    assert prov["prompt_seed"] is None
    assert prov["prompt_chat_templated"] is False


# ---------------------------------------------------------------------------
# 5. EOS surfacing -- synthetic (eos at 0 / mid / never) + rule
# ---------------------------------------------------------------------------


def test_eos_surfacing_first_token_eos(env_levers):
    # eos at position 0 -> empty answer; the 256 forced post-eos tokens are timed.
    surf = env_levers._eos_surfacing([_EOS_ID] + [9] * 256, _EOS_ID)
    assert surf["first_token_eos"] is True
    assert surf["eos_index"] == 0
    assert surf["tokens_before_eos"] == 0
    assert surf["answer_valid"] is False  # empty answer (first token EOS)
    assert surf["answer_truncated"] is False  # EOS present, not cap-truncated
    assert surf["post_eos_tokens_timed"] == 256  # the wasted filler (windows 39-42)
    assert surf["eos_id"] == _EOS_ID
    assert surf["n_generated"] == 257


def test_eos_surfacing_mid_stream(env_levers):
    # eos mid-stream: valid regardless of position (answer is non-empty).
    ids = [9] * 100 + [_EOS_ID] + [9] * 156
    surf = env_levers._eos_surfacing(ids, _EOS_ID)
    assert surf["first_token_eos"] is False
    assert surf["eos_index"] == 100
    assert surf["tokens_before_eos"] == 100
    assert surf["answer_valid"] is True
    assert surf["answer_truncated"] is False
    assert surf["post_eos_tokens_timed"] == 257 - 100 - 1  # 156 forced tokens


def test_eos_surfacing_never(env_levers):
    surf = env_levers._eos_surfacing([9] * 257, _EOS_ID)
    assert surf["first_token_eos"] is False
    assert surf["eos_index"] is None
    assert surf["tokens_before_eos"] == 257
    assert surf["answer_valid"] is True
    assert surf["answer_truncated"] is True  # hit the cap without EOS
    assert surf["post_eos_tokens_timed"] == 0


def test_eos_surfacing_unknown_eos_id_all_none(env_levers):
    surf = env_levers._eos_surfacing([9, _EOS_ID], None)
    assert surf["first_token_eos"] is None
    assert surf["eos_index"] is None
    assert surf["answer_valid"] is None
    assert surf["answer_truncated"] is None
    assert surf["post_eos_tokens_timed"] is None
    assert surf["eos_id"] is None
    assert surf["n_generated"] == 2


def test_eos_surfacing_answer_valid_is_cap_independent(env_levers):
    # MEDIUM-1: the SAME answer (eos at index 60) must get the SAME verdict whether
    # --stop-on-eos truncated the stream (n=61) or the full fixed-step decode ran
    # (n=257 with forced post-eos filler).  The withdrawn 0.5*N rule flipped this.
    stopped = env_levers._eos_surfacing([9] * 60 + [_EOS_ID], _EOS_ID)  # n=61
    full = env_levers._eos_surfacing([9] * 60 + [_EOS_ID] + [7] * 196, _EOS_ID)  # n=257
    assert stopped["answer_valid"] == full["answer_valid"] is True
    assert stopped["eos_index"] == full["eos_index"] == 60
    assert stopped["answer_truncated"] == full["answer_truncated"] is False
    # only the timed-filler count differs (that is the point of the field).
    assert stopped["post_eos_tokens_timed"] == 0
    assert full["post_eos_tokens_timed"] == 196


# ---------------------------------------------------------------------------
# 6. EOS surfacing on the real window-43 receipts (fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture,expect_eos_index,expect_truncated,expect_post_eos",
    [
        ("window43_ar_ring_ref.json", None, True, 0),
        ("window43_ar_ring_v2.json", 238, False, 257 - 238 - 1),
    ],
)
def test_eos_surfacing_window43_fixtures(env_levers, fixture, expect_eos_index,
                                         expect_truncated, expect_post_eos):
    receipt = json.loads((_FIXTURES / fixture).read_text())
    # window-43 ran the CORRECT chat-templated cell: 16,384 prompt tokens (not the
    # 16,385 raw builder), and the greedy first token is NOT EOS (id 666, not 1).
    assert receipt["prompt_tokens"] == 16384
    ids = receipt["token_ids"]
    assert ids[0] != _EOS_ID
    surf = env_levers._eos_surfacing(ids, _EOS_ID)
    assert surf["first_token_eos"] is False
    assert surf["eos_index"] == expect_eos_index
    assert surf["n_generated"] == 257  # decode_tokens (256) + prefill token
    assert surf["answer_valid"] is True  # cap-independent: first token not EOS
    assert surf["answer_truncated"] is expect_truncated
    assert surf["post_eos_tokens_timed"] == expect_post_eos


# ---------------------------------------------------------------------------
# 7. --stop-on-eos decode accounting (fake model, no MLX)
# ---------------------------------------------------------------------------


class _FakeOps:
    """The tiny ``ops`` surface ``_generate`` uses.  ``argmax_last`` returns the
    scripted token the fake model produced (no MLX)."""

    def input(self, x):
        return x

    def sync(self, x):
        return None

    def argmax_last(self, logits):
        return logits  # the fake model returns the next token id directly


class _FakeSampler:
    def start(self):
        return None

    def stop(self):
        return None


class _FakeMemProbe:
    def reset_peak(self):
        return None

    def new_sampler(self):
        return _FakeSampler()

    def peak_bytes(self):
        return 0

    def memory_block(self, sampler):
        return {}


class _FakeModel:
    """Greedy argmax scripted by ``script``: call 0 is the prefill (yields the
    first/prefill token), calls 1.. are the decode steps.  No cache, no MLX."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def make_cache(self):
        return object()

    def __call__(self, x, cache=None):
        tok = self._script[self._i]
        self._i += 1
        return tok


def _run_gen(env_levers, script, steps, **kw):
    return env_levers._generate(
        model=_FakeModel(script), ops=_FakeOps(), mem_probe=_FakeMemProbe(),
        prompt_ids=[0, 1, 2], steps=steps, **kw,
    )


def test_stop_on_eos_off_runs_full_and_counts_all(env_levers):
    # default: full fixed-step decode, decode_steps_run == steps (numbers unchanged).
    run = _run_gen(env_levers, [10, 11, 12, 13, 14, 15], steps=5)
    assert run["decode_steps_run"] == 5
    assert run["generated"] == [10, 11, 12, 13, 14, 15]  # prefill token + 5 decode


def test_stop_on_eos_off_ignores_eos_in_stream(env_levers):
    # eos appears but stop_on_eos is off -> keep going (post-EOS forced tokens).
    run = _run_gen(env_levers, [10, _EOS_ID, 12, 13, 14, 15], steps=5)
    assert run["decode_steps_run"] == 5
    assert run["generated"] == [10, _EOS_ID, 12, 13, 14, 15]


def test_stop_on_eos_stops_mid_stream(env_levers):
    # eos at decode step 2 -> break; decode_steps_run counts only tokens generated.
    run = _run_gen(env_levers, [10, 11, _EOS_ID, 99, 99, 99], steps=5,
                   stop_on_eos=True, eos_id=_EOS_ID)
    assert run["decode_steps_run"] == 2
    assert run["generated"] == [10, 11, _EOS_ID]  # eos is included, then stop


def test_stop_on_eos_first_token_eos_skips_decode(env_levers):
    # the prefill's own first token is EOS -> a served path emits nothing.
    run = _run_gen(env_levers, [_EOS_ID, 99, 99], steps=5,
                   stop_on_eos=True, eos_id=_EOS_ID)
    assert run["decode_steps_run"] == 0
    assert run["generated"] == [_EOS_ID]


def test_stop_on_eos_never_hit_runs_full(env_levers):
    run = _run_gen(env_levers, [10, 11, 12, 13, 14, 15], steps=5,
                   stop_on_eos=True, eos_id=_EOS_ID)
    assert run["decode_steps_run"] == 5
    assert run["generated"] == [10, 11, 12, 13, 14, 15]


# ---------------------------------------------------------------------------
# 8. MEDIUM-2: --stop-on-eos threaded into the warm-repeat pass
# ---------------------------------------------------------------------------


def test_warm_repeat_pass_honours_stop_on_eos(env_levers):
    # cold pass stops at EOS (step 2); the warm pass must stop at the SAME point so
    # the denominator matches and token_ids_match holds (not run the full 5 steps).
    script = [10, 11, _EOS_ID, 99, 99, 99]
    cold = env_levers._generate(
        model=_FakeModel(script), ops=_FakeOps(), mem_probe=_FakeMemProbe(),
        prompt_ids=[0, 1, 2], steps=5, stop_on_eos=True, eos_id=_EOS_ID,
    )
    assert cold["decode_steps_run"] == 2
    warm = env_levers._warm_repeat_pass(
        model=_FakeModel(script), ops=_FakeOps(), mem_probe=_FakeMemProbe(),
        prompt_ids=[0, 1, 2], steps=5, cold_ids=cold["generated"],
        stop_on_eos=True, eos_id=_EOS_ID,
    )
    assert warm["warm_decode_tokens_generated"] == 2  # not the full 5
    assert warm["token_ids_match"] is True


def test_warm_repeat_pass_default_runs_full(env_levers):
    script = [10, 11, 12, 13, 14, 15]
    cold = env_levers._generate(
        model=_FakeModel(script), ops=_FakeOps(), mem_probe=_FakeMemProbe(),
        prompt_ids=[0, 1, 2], steps=5,
    )
    warm = env_levers._warm_repeat_pass(
        model=_FakeModel(script), ops=_FakeOps(), mem_probe=_FakeMemProbe(),
        prompt_ids=[0, 1, 2], steps=5, cold_ids=cold["generated"],
    )
    assert warm["warm_decode_tokens_generated"] == 5
    assert warm["token_ids_match"] is True
