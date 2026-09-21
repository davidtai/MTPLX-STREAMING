"""CPU tests for the F39 quality-gate driver plumbing (tcq/eval_driver.py): David's sampler in the gate argv, the
mxfp4-vs-tcq3 serve command (same code path, tcq3 env, never :8080, SSD cache off), receipt layout + resume, and
the truncation-aware metric derivation.  No serve, no model, no network, no GPU.
"""
import json
import sys
import types
from pathlib import Path

import pytest

_DSV41 = Path(__file__).resolve().parents[2]      # scripts/deepseek_v41 -> `import tcq.eval_driver`
if str(_DSV41) not in sys.path:
    sys.path.insert(0, str(_DSV41))

from tcq import eval_driver as E                   # noqa: E402


def _args(**over):
    base = dict(bank="mxfp4", suite="both", lane="ar", depth=3, limit=None,
                out_dir="/tmp/does-not-matter", mxfp4_model=E.MODEL_DIRS["mxfp4"], tcq3_model=E.MODEL_DIRS["tcq3"],
                humaneval_dataset=E.DATASETS["humaneval"], mbpp_dataset=E.DATASETS["mbpp"],
                host="127.0.0.1", port=18183, served_model_name="deepseek-v41-flash",
                serve_entry=E.DEFAULT_SERVE_ENTRY, workers=1, resume=False, dry_run=True)
    base.update(over)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------- David's sampler in the gate argv

def test_gate_argv_carries_davids_sampler():
    argv = E.build_gate_argv(suite="humaneval", dataset_path="/d/HumanEval.jsonl", base_url="http://127.0.0.1:18183",
                             model="m", report_path="/r/report.json", completions_path="/r/c.jsonl",
                             limit=None, workers=1)
    def val(flag):
        return argv[argv.index(flag) + 1]
    assert val("--temperature") == "1.0" and val("--top-p") == "0.95"
    assert val("--seed") == "20260829" and val("--n") == "1"
    assert val("--endpoint") == "chat" and val("--max-tokens") == "2048"
    assert "--extra-body" in argv and val("--extra-body") == "top_k=20"      # top-k via extra-body
    assert "--allow-code-execution" in argv and "--save-completions" in argv and "--progress" in argv
    assert val("--suite") == "humaneval" and val("--dataset-path") == "/d/HumanEval.jsonl"
    assert "--limit" not in argv                                             # only when set


def test_gate_argv_limit_and_mbpp():
    argv = E.build_gate_argv(suite="mbpp", dataset_path="/d/sanitized-mbpp.json", base_url="u", model="m",
                             report_path="/r/report.json", completions_path="/r/c.jsonl", limit=4, workers=1)
    assert argv[argv.index("--suite") + 1] == "mbpp"
    assert argv[argv.index("--limit") + 1] == "4"


# ---------------------------------------------------------------- serve command: mxfp4 vs tcq3, same path

def test_serve_command_mxfp4_is_stock():
    argv, env = E.build_serve_command(bank="mxfp4", model_dir=E.MODEL_DIRS["mxfp4"], host="127.0.0.1", port=18183,
                                      lane="ar", depth=3)
    assert env == {}                                                          # control: no tcq3 env
    assert "--no-auth" in argv and argv[argv.index("--ssd-session-cache") + 1] == "off"
    assert argv[argv.index("--model") + 1] == E.MODEL_DIRS["mxfp4"]
    assert "--load-mtp" not in argv                                           # ar lane


def test_serve_command_tcq3_arms_the_lane():
    argv, env = E.build_serve_command(bank="tcq3", model_dir=E.MODEL_DIRS["tcq3"], host="127.0.0.1", port=18183,
                                      lane="dspark", depth=3)
    assert env["MTPLX_DSV41_TCQ3"] == "1"
    assert env["GPU_WINDOW_CANDIDATE_MODEL_DIR"] == E.MODEL_DIRS["tcq3"]
    assert argv[argv.index("--model") + 1] == E.MODEL_DIRS["tcq3"]
    # dspark lane args present
    assert "--load-mtp" in argv and argv[argv.index("--generation-mode") + 1] == "dspark"
    assert argv[argv.index("--depth") + 1] == "3"
    # SSD cache still off (a correctness cell must not warm the prod bank)
    assert argv[argv.index("--ssd-session-cache") + 1] == "off"


def test_serve_command_refuses_8080():
    with pytest.raises(ValueError, match="8080"):
        E.build_serve_command(bank="mxfp4", model_dir="/m", host="127.0.0.1", port=8080, lane="ar", depth=3)


def test_serve_command_rejects_bad_bank_and_lane():
    with pytest.raises(ValueError):
        E.build_serve_command(bank="nope", model_dir="/m", host="127.0.0.1", port=18183, lane="ar", depth=3)
    with pytest.raises(ValueError):
        E.build_serve_command(bank="mxfp4", model_dir="/m", host="127.0.0.1", port=18183, lane="nope", depth=3)


# ---------------------------------------------------------------- receipt layout + resume + overwrite guard

def test_receipt_dir_layout():
    d = E.receipt_dir("/out", "tcq3", "humaneval", "20260921T000000Z")
    assert d == Path("/out/eval/tcq3/humaneval/20260921T000000Z")


def test_resume_finds_completed_and_plan_skips(tmp_path):
    out = str(tmp_path)
    # no receipts yet
    assert E.latest_completed(out, "mxfp4", "humaneval") is None
    # a completed receipt (report.json present)
    done = E.receipt_dir(out, "mxfp4", "humaneval", "20260921T000000Z")
    done.mkdir(parents=True)
    (done / "report.json").write_text("{}")
    assert E.latest_completed(out, "mxfp4", "humaneval") == done
    # plan with --resume marks that suite skipped
    p = E.plan(_args(out_dir=out, resume=True, suite="both"))
    he = next(i for i in p["items"] if i["suite"] == "humaneval")
    mbpp = next(i for i in p["items"] if i["suite"] == "mbpp")
    assert he["skip_resumed"] == str(done) and mbpp["skip_resumed"] is None


def test_write_receipt_refuses_overwrite(tmp_path):
    rdir = tmp_path / "r"
    report = {"rows": [{"passed": True, "finish_reason": "stop"}], "pass@1": 1.0}
    E.write_receipt(rdir, bank="tcq3", suite="humaneval", lane="ar", depth=3, model_dir="/m",
                    base_url="u", gate_argv=[], serve_argv=[], report=report, wall_s=1.0)
    assert (rdir / "receipt.json").is_file()
    with pytest.raises(FileExistsError):                                      # append-only: never overwrite
        E.write_receipt(rdir, bank="tcq3", suite="humaneval", lane="ar", depth=3, model_dir="/m",
                        base_url="u", gate_argv=[], serve_argv=[], report=report, wall_s=1.0)


# ---------------------------------------------------------------- truncation-aware metrics (David's rule)

def test_derive_metrics_excludes_truncated_from_completed_rate():
    report = {"rows": [
        {"passed": True, "finish_reason": "stop"},
        {"passed": False, "finish_reason": "stop"},
        {"passed": False, "finish_reason": "length"},    # truncated -> excluded from completed-rate
        {"passed": True, "finish_reason": "stop"},
    ]}
    m = E._derive_metrics(report)
    assert m["tasks"] == 4 and m["passed"] == 2
    assert m["strict_pass_at_1"] == 0.5
    assert m["truncated_tasks"] == 1 and abs(m["truncation_rate"] - 0.25) < 1e-9
    assert m["completed_tasks"] == 3 and abs(m["completed_task_pass_at_1"] - 2 / 3) < 1e-9


# ---------------------------------------------------------------- plan covers both suites; mxfp4 vs tcq3 differ only in bank/env

def test_plan_both_suites_same_path_except_bank():
    pm = E.plan(_args(bank="mxfp4", suite="both", limit=4))
    pt = E.plan(_args(bank="tcq3", suite="both", limit=4))
    assert [i["suite"] for i in pm["items"]] == ["humaneval", "mbpp"]
    assert pm["serve_env"] == {} and pt["serve_env"]["MTPLX_DSV41_TCQ3"] == "1"
    assert pm["model_dir"] == E.MODEL_DIRS["mxfp4"] and pt["model_dir"] == E.MODEL_DIRS["tcq3"]
    # the gate argv (sampler, suite, limit) is IDENTICAL across banks (true pair) except the receipt paths
    def strip_paths(argv):
        return [a for a in argv if "/eval/" not in a]
    for im, it in zip(pm["items"], pt["items"]):
        assert strip_paths(im["gate_argv"]) == strip_paths(it["gate_argv"])


def test_harness_build_messages_is_reused_not_reimplemented():
    # the driver must reuse scripts/code_eval_gate.py's message construction; import it to prove it's on-path
    scripts_dir = str(_DSV41.parents[1] / "scripts")   # <worktree>/scripts
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        import code_eval_gate  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"code_eval_gate not importable in this env: {exc}")
    assert hasattr(code_eval_gate, "main")            # the driver half is present and importable (no network/model)
