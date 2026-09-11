"""CPU-only tests for the W68 DeepSeek-V4.1 HumanEval lane cells.

Covers the two W68 deliverables that the pure-metric tests in
``test_deepseek_v41_bench_scripts.py`` do not:

  1. ``scripts/deepseek_v41/humaneval_cell.py`` end to end against a **stub HTTP
     server** (a real ``http.server`` in this process, not a monkeypatched
     transport): the module drives ``code_eval_gate`` over real HTTP, the
     completions are **actually executed** by ``mtplx.benchmarks.code_eval``'s
     sandbox (the scoring path), and the derived receipt carries the
     truncation-aware metrics, the recorded lane, the parsed decode-lever env
     and the self-contained per-task pass map.
  2. ``scripts/deepseek_v41/humaneval_lane_compare.py``: the AR-vs-DSpark paired
     comparison (pass@1 delta + the per-task tie-flip diff list).

Plus the pure decode-lever parsing helpers. No GPU, no Metal, no model; importing
``code_eval_gate`` does NOT import MLX (verified: ~29 MB, mlx.core absent), so
these stay under the worker RSS cap. Run under ``nice -n 19``, no ``pytest -n
auto``.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_WT = Path(__file__).resolve().parents[1]
_SCRIPTS = _WT / "scripts" / "deepseek_v41"


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dsv41_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def humaneval():
    return _load("humaneval_cell")


@pytest.fixture(scope="module")
def lane_compare():
    return _load("humaneval_lane_compare")


# ==========================================================================
# decode-lever env parsing (pure)
# ==========================================================================

_LEVERS_LINE = (
    "[4/6] DeepSeek-V4.1 decode levers (resolved env): HEAD_MODE=bf16 "
    "SINKHORN_METAL=1 ATTN_COMPILE=1 ATTN_WIN_MEMO=1 SWITCH_FASTPATH=1 "
    "SWITCH_SUBMIT=1 DEVICE_ROUTE=1 HC_COMPILE=1 SHARED_OVERLAP=<unset> "
    "PREFILL_LAYER_MAJOR=<unset>"
)


def test_parse_decode_levers_maps_keys_and_unset(humaneval):
    parsed = humaneval.parse_decode_levers(_LEVERS_LINE)
    assert parsed is not None
    resolved = parsed["resolved"]
    assert resolved["HEAD_MODE"] == "bf16"
    assert resolved["SINKHORN_METAL"] == "1"
    # "<unset>" becomes None, matching the daemon's own rendering.
    assert resolved["SHARED_OVERLAP"] is None
    assert resolved["PREFILL_LAYER_MAJOR"] is None
    assert "resolved env" not in parsed["raw"] or parsed["raw"] == _LEVERS_LINE.strip()


def test_parse_decode_levers_handles_a_bare_prefix_free_render(humaneval):
    # Any leading text (log timestamp, stage tag) before the marker is ignored.
    line = "2026-09-11T00:00:00Z something decode levers (resolved env): HEAD_MODE=bf16"
    parsed = humaneval.parse_decode_levers(line)
    assert parsed["resolved"] == {"HEAD_MODE": "bf16"}


def test_parse_decode_levers_empty_or_marker_only_is_none(humaneval):
    assert humaneval.parse_decode_levers(None) is None
    assert humaneval.parse_decode_levers("") is None
    assert humaneval.parse_decode_levers("no levers here") is None
    assert humaneval.parse_decode_levers("decode levers (resolved env): ") is None


def test_scrape_levers_from_log_takes_the_last_matching_line(humaneval, tmp_path):
    log = tmp_path / "serve.log"
    log.write_text(
        "boot\n"
        "[4/6] DeepSeek-V4.1 decode levers (resolved env): HEAD_MODE=fp32\n"
        "reloading\n"
        "[4/6] DeepSeek-V4.1 decode levers (resolved env): HEAD_MODE=bf16 SINKHORN_METAL=1\n"
        "serving\n"
    )
    line = humaneval._scrape_levers_from_log(log)
    parsed = humaneval.parse_decode_levers(line)
    assert parsed["resolved"] == {"HEAD_MODE": "bf16", "SINKHORN_METAL": "1"}


def test_scrape_levers_from_missing_log_is_none(humaneval, tmp_path):
    assert humaneval._scrape_levers_from_log(tmp_path / "nope.log") is None


def test_resolve_decode_levers_prefers_explicit_line(humaneval, tmp_path):
    import argparse

    log = tmp_path / "serve.log"
    log.write_text("decode levers (resolved env): HEAD_MODE=fromlog\n")
    args = argparse.Namespace(
        decode_levers_line="decode levers (resolved env): HEAD_MODE=fromflag",
        server_log=log,
    )
    assert humaneval.resolve_decode_levers(args)["resolved"] == {"HEAD_MODE": "fromflag"}
    # Fall back to the log when no explicit line is given.
    args2 = argparse.Namespace(decode_levers_line=None, server_log=log)
    assert humaneval.resolve_decode_levers(args2)["resolved"] == {"HEAD_MODE": "fromlog"}


def test_per_task_rows_are_self_contained(humaneval):
    report = {
        "rows": [
            {"task_id": "HumanEval/0", "sample": 0, "passed": True,
             "finish_reason": "stop", "status": "passed"},
            {"task_id": "HumanEval/1", "sample": 0, "passed": False,
             "finish_reason": "length", "status": "failed"},
        ]
    }
    rows = humaneval.per_task_rows(report)
    assert rows == [
        {"task_id": "HumanEval/0", "sample": 0, "passed": True,
         "finish_reason": "stop", "status": "passed"},
        {"task_id": "HumanEval/1", "sample": 0, "passed": False,
         "finish_reason": "length", "status": "failed"},
    ]


# ==========================================================================
# end-to-end: humaneval_cell.py -> code_eval_gate -> stub server -> scoring
# ==========================================================================

_GOOD = "```python\ndef add(a, b):\n    return a + b\n```"


def _humaneval_dataset(tmp_path: Path, n: int = 3) -> Path:
    path = tmp_path / "HumanEval.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": f"HumanEval/{i}",
                    "prompt": "def add(a, b):\n",
                    "canonical_solution": "    return a + b\n",
                    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                    "entry_point": "add",
                }
            )
            for i in range(n)
        )
        + "\n"
    )
    return path


class _StubHandler(BaseHTTPRequestHandler):
    """Returns a canned chat completion for /v1/chat/completions."""

    # Set by the fixture: (completion_text, finish_reason).
    canned = (_GOOD, "stop")

    def log_message(self, *_args):  # silence the default stderr spam
        return

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        text, finish = type(self).canned
        body = json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                "usage": {"completion_tokens": 8, "prompt_tokens": 20},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _StubServer:
    def __init__(self, completion: str, finish: str = "stop"):
        handler = type(
            "_H", (_StubHandler,), {"canned": (completion, finish)}
        )
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _one_receipt(out_dir: Path):
    receipts = list(out_dir.glob("humaneval_cell/**/humaneval_cell__*.json"))
    assert len(receipts) == 1, receipts
    return json.loads(receipts[0].read_text()), receipts[0]


def test_served_cell_end_to_end_scores_correct_solutions(humaneval, tmp_path):
    """A real HTTP round-trip + the real sandbox scoring path -> pass@1 = 1.0."""

    dataset = _humaneval_dataset(tmp_path, n=3)
    out_dir = tmp_path / "receipts"
    with _StubServer(_GOOD, "stop") as server:
        rc = humaneval.main(
            [
                "--base-url", server.base_url,
                "--model", "dsv41-stub",
                "--dataset-path", str(dataset),
                "--out-dir", str(out_dir),
                "--lane", "ar",
                "--serve-flags", "(served AR default)",
                "--decode-levers-line", _LEVERS_LINE,
                "--workers", "2",
                "--score-workers", "2",
            ]
        )
    assert rc == 0
    receipt, path = _one_receipt(out_dir)
    assert "__ar__" in path.name
    assert receipt["dry_run"] is False
    assert receipt["lane"] == "ar"
    m = receipt["metrics"]
    assert m["tasks"] == 3
    assert m["passed"] == 3
    assert m["strict_pass_at_1"] == pytest.approx(1.0)
    assert m["completed_task_pass_at_1"] == pytest.approx(1.0)
    assert m["truncation_rate"] == pytest.approx(0.0)
    # lane + lever env recorded from the (explicit) daemon line
    assert receipt["decode_levers"]["resolved"]["HEAD_MODE"] == "bf16"
    assert receipt["decode_levers"]["resolved"]["SHARED_OVERLAP"] is None
    # self-contained per-task map + the underlying gate artifacts on disk
    assert {r["task_id"] for r in receipt["per_task"]} == {
        "HumanEval/0", "HumanEval/1", "HumanEval/2"
    }
    assert all(r["passed"] for r in receipt["per_task"])
    assert Path(receipt["gate_report_json"]).is_file()
    assert Path(receipt["completions_sidecar"]).is_file()


def test_served_cell_end_to_end_truncation_is_not_a_fail(humaneval, tmp_path):
    """finish_reason=length -> counted as truncated, excluded from completed rate."""

    dataset = _humaneval_dataset(tmp_path, n=2)
    out_dir = tmp_path / "receipts"
    with _StubServer(_GOOD, "length") as server:
        rc = humaneval.main(
            [
                "--base-url", server.base_url,
                "--model", "dsv41-stub",
                "--dataset-path", str(dataset),
                "--out-dir", str(out_dir),
                "--lane", "dspark",
                "--depth", "3",
                "--workers", "2",
                "--score-workers", "2",
            ]
        )
    assert rc == 0
    receipt, path = _one_receipt(out_dir)
    assert "__dspark__" in path.name
    assert receipt["lane"] == "dspark"
    assert receipt["depth"] == 3
    m = receipt["metrics"]
    # The code is correct, so strict pass@1 still counts it...
    assert m["strict_pass_at_1"] == pytest.approx(1.0)
    # ...but every row was truncated, so the completed-task rate has no completed
    # rows and the truncation rate is 1.0 (eval-truncation-is-not-failure.md).
    assert m["truncation_rate"] == pytest.approx(1.0)
    assert m["completed_tasks"] == 0
    assert m["completed_task_pass_at_1"] == pytest.approx(0.0)


# ==========================================================================
# paired-lane comparison
# ==========================================================================


def _receipt(lane: str, per_task: list[dict], strict: float, completed: float = 1.0,
             truncated: int = 0) -> dict:
    passed = sum(1 for r in per_task if r["passed"])
    return {
        "lane": lane,
        "dry_run": False,
        "serve_flags": f"({lane})",
        "git_rev": "deadbeef",
        "seed": 20260829,
        "max_tokens": 2048,
        "decode_levers": {"resolved": {"HEAD_MODE": "bf16"}},
        "metrics": {
            "tasks": len(per_task),
            "passed": passed,
            "strict_pass_at_1": strict,
            "completed_task_pass_at_1": completed,
            "truncated_tasks": truncated,
        },
        "per_task": per_task,
    }


def _row(task_id: str, passed: bool, finish: str = "stop") -> dict:
    return {"task_id": task_id, "sample": 0, "passed": passed,
            "finish_reason": finish, "status": "passed" if passed else "failed"}


def test_compare_lanes_lists_the_tie_flip_divergences(lane_compare):
    ar = _receipt(
        "ar",
        [_row("HumanEval/0", True), _row("HumanEval/1", True),
         _row("HumanEval/2", False), _row("HumanEval/3", False)],
        strict=0.5,
    )
    dspark = _receipt(
        "dspark",
        [_row("HumanEval/0", True),                       # both pass
         _row("HumanEval/1", False),                      # AR only
         _row("HumanEval/2", True, finish="stop"),        # DSpark only
         _row("HumanEval/3", False)],                     # both fail
        strict=0.5,
    )
    summary = lane_compare.compare_lanes(ar, dspark)
    comp = summary["comparison"]
    assert comp["shared_tasks"] == 4
    assert comp["both_pass"] == 1
    assert comp["both_fail"] == 1
    assert comp["disagreements"] == 2
    assert comp["agreement_rate"] == pytest.approx(0.5)
    assert [e["task_id"] for e in comp["ar_only_pass"]] == ["HumanEval/1"]
    assert [e["task_id"] for e in comp["dspark_only_pass"]] == ["HumanEval/2"]
    # each diff entry carries BOTH lanes' finish_reason for inspection
    assert comp["ar_only_pass"][0]["ar"]["passed"] is True
    assert comp["ar_only_pass"][0]["dspark"]["passed"] is False
    assert summary["delta_strict_pass_at_1"] == pytest.approx(0.0)
    assert summary["ar"]["lane"] == "ar" and summary["dspark"]["lane"] == "dspark"


def test_compare_lanes_flags_unmatched_task_sets(lane_compare):
    ar = _receipt("ar", [_row("HumanEval/0", True), _row("HumanEval/1", True)], strict=1.0)
    dspark = _receipt("dspark", [_row("HumanEval/0", True)], strict=1.0)
    comp = lane_compare.compare_lanes(ar, dspark)["comparison"]
    assert comp["shared_tasks"] == 1
    assert comp["ar_only_tasks"] == ["HumanEval/1"]
    assert comp["dspark_only_tasks"] == []


def test_compare_lanes_reports_the_pass_at_1_delta(lane_compare):
    ar = _receipt("ar", [_row("HumanEval/0", True), _row("HumanEval/1", False)], strict=0.5)
    dspark = _receipt("dspark", [_row("HumanEval/0", True), _row("HumanEval/1", True)], strict=1.0)
    summary = lane_compare.compare_lanes(ar, dspark)
    assert summary["delta_strict_pass_at_1"] == pytest.approx(0.5)


def test_lane_compare_cli_writes_summary_and_is_append_only(lane_compare, tmp_path):
    ar_path = tmp_path / "ar.json"
    ds_path = tmp_path / "dspark.json"
    ar_path.write_text(json.dumps(_receipt(
        "ar", [_row("HumanEval/0", True), _row("HumanEval/1", False)], strict=0.5)))
    ds_path.write_text(json.dumps(_receipt(
        "dspark", [_row("HumanEval/0", True), _row("HumanEval/1", True)], strict=1.0)))
    out = tmp_path / "compare.json"
    rc = lane_compare.main(["--ar", str(ar_path), "--dspark", str(ds_path), "--out", str(out)])
    assert rc == 0
    written = json.loads(out.read_text())
    assert written["delta_strict_pass_at_1"] == pytest.approx(0.5)
    assert written["inputs"]["ar"] == str(ar_path)
    # append-only: a second write to the same path is refused.
    rc2 = lane_compare.main(["--ar", str(ar_path), "--dspark", str(ds_path), "--out", str(out)])
    assert rc2 == 2


def test_lane_compare_cli_missing_receipt_exits_2(lane_compare, tmp_path):
    ar_path = tmp_path / "ar.json"
    ar_path.write_text(json.dumps(_receipt("ar", [_row("HumanEval/0", True)], strict=1.0)))
    rc = lane_compare.main(["--ar", str(ar_path), "--dspark", str(tmp_path / "nope.json")])
    assert rc == 2
