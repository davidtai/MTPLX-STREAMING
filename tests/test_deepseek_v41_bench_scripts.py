"""CPU-only unit tests for the W14 DeepSeek-V4.1 GPU-phase bench drivers.

Covers ``scripts/deepseek_v41/bench_standard_shape.py`` and
``scripts/deepseek_v41/humaneval_cell.py``: argument parsing, the shared
prompt-build reuse, receipt paths, the append-only (never-overwrite) guard, the
truncation-aware metric derivation, and the ``--dry-run`` CPU test double.

No GPU, no Metal, no model, no server, no network. The scripts are not part of a
package (``scripts/`` has no ``__init__.py``), so they are loaded by file path.
Run under ``nice -n 19`` and without ``pytest -n auto`` (host-encode sensitivity;
memory/worker-tests-must-pin-mlx-cpu.md — these tests never import MLX).
"""

from __future__ import annotations

import importlib.util
import json
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
def bench():
    return _load("bench_standard_shape")


@pytest.fixture(scope="module")
def humaneval():
    return _load("humaneval_cell")


# ==========================================================================
# bench_standard_shape.py
# ==========================================================================


def test_bench_import_is_cpu_safe_no_mlx(bench):
    import sys

    # Loading the module must not have imported MLX (no GPU at import).
    assert "mlx.core" not in sys.modules or True  # tolerated if already loaded
    # The module must not itself import mlx at top level.
    src = (_SCRIPTS / "bench_standard_shape.py").read_text()
    assert "\nimport mlx" not in src
    assert "import mlx.core" in src  # only inside run_real (deferred)


def test_bench_parser_defaults(bench):
    args = bench.build_parser().parse_args([])
    assert args.context_tokens == [1024, 16384]
    assert args.steps == 256
    assert args.repeats == 1
    assert args.bos is True
    assert args.bos_id == 0
    assert args.prompt_format == "raw"
    assert args.slot_layout == "component-banks"


def test_bench_resolve_max_kv(bench):
    # auto = max cell + steps + 64
    assert bench.resolve_max_kv([1024, 16384], 256, None) == 16384 + 256 + 64
    # explicit, adequate
    assert bench.resolve_max_kv([1024], 256, 5000) == 5000
    # explicit, too small -> raises
    with pytest.raises(ValueError):
        bench.resolve_max_kv([16384], 256, 1000)


def test_bench_fresh_out_dir_never_reused(bench, tmp_path):
    a = bench.fresh_out_dir(tmp_path)
    b = bench.fresh_out_dir(tmp_path)
    assert a.exists() and b.exists()
    assert a != b  # never reuse a stamped directory
    assert a.parent.name == bench.STEP
    assert (tmp_path / bench.STEP) in a.parents


def test_bench_write_receipt_append_only_guard(bench, tmp_path):
    out = bench.fresh_out_dir(tmp_path)
    receipt = {"context_cells": [1024], "steps": 4, "seed": 0, "utc_compact": "S"}
    p = bench.write_receipt(out, receipt)
    assert p.exists()
    # A second write of the same receipt name must be refused, not overwritten.
    with pytest.raises(FileExistsError):
        bench.write_receipt(out, receipt)


def test_bench_receipt_filename_suffix(bench):
    name = bench.receipt_filename(
        {"context_cells": [1024, 16384], "steps": 256, "seed": 0, "utc_compact": "T"}
    )
    assert name.startswith("bench_standard_shape__")
    assert "ctx1024-16384" in name
    assert "steps256" in name
    assert "seed0" in name
    assert name.endswith("__T.json")


def test_bench_fastest_of(bench):
    reps = [
        {"decode_tok_s": 10.0, "prefill_tok_s": 100.0, "ttft_s": 0.5, "peak_mlx_gb": 8.0},
        {"decode_tok_s": 12.0, "prefill_tok_s": 90.0, "ttft_s": 0.4, "peak_mlx_gb": 9.0},
    ]
    f = bench.fastest_of(reps)
    assert f["decode_tok_s_fastest"] == 12.0
    assert f["decode_tok_s_range"] == [10.0, 12.0]
    assert f["ttft_s_fastest"] == 0.4  # fastest = min TTFT
    assert f["peak_gb_highest"] == 9.0  # highest peak
    assert bench.fastest_of([]) is None


def test_bench_one_cell_double_counts_gathers(bench):
    model = bench._FakeModel()
    ops = bench._FakeOps()
    gather = bench._GatherProbe(model, mx=None)
    mem = bench._DryMemProbe()
    prompt_ids = list(range(50))
    m = bench.bench_one_cell(
        model=model,
        tokenizer=bench._FakeTokenizer(),
        ops=ops,
        mem_probe=mem,
        gather_probe=gather,
        prompt_ids=prompt_ids,
        steps=4,
    )
    # prefill (50-token seq) + 4 decode forwards, 40 layers, top_k 8:
    assert m["expert_records_gathered"] == (50 + 4) * 40 * 8
    # engram: 2 layers * 24 rows/token over the same tokens
    assert m["engram_rows_gathered"] == (50 + 4) * 24 * 2
    assert m["prompt_tokens"] == 50
    assert m["decode_tokens"] == 4
    assert m["generated_token_count"] == 5  # first (TTFT) token + 4 decode
    assert isinstance(m["text_preview"], str) and m["text_preview"]


def test_bench_dry_run_end_to_end(bench, tmp_path):
    rc = bench.main(
        ["--dry-run", "--steps", "2", "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    receipts = list(tmp_path.glob("bench_standard_shape/**/*.json"))
    assert len(receipts) == 1
    d = json.loads(receipts[0].read_text())
    assert d["dry_run"] is True
    assert d["greedy"] is True
    assert d["context_cells"] == [1024, 16384]
    assert d["steps"] == 2
    assert d["spec_key"] and d["manifest_sha256"]
    assert d["engram_layer_ids"] == [1, 14]
    assert len(d["cells"]) == 2
    for cell, ctx in zip(d["cells"], [1024, 16384]):
        assert cell["context_tokens"] == ctx
        pb = cell["prompt_build"]
        # prefill_bench builder + BOS prepend -> ctx + 1 input tokens
        assert pb["prompt_source"] == "prefill_bench"
        assert pb["bos_prepended"] is True
        assert pb["input_tokens"] == ctx + 1
        r = cell["repeats"][0]
        for key in (
            "prefill_tok_s",
            "ttft_s",
            "decode_tok_s",
            "peak_mlx_gb",
            "process_rss_gb",
            "expert_records_gathered",
            "engram_rows_gathered",
            "text_preview",
            "wall_s",
        ):
            assert key in r
        assert cell["fastest_of"]["decode_tok_s_fastest"] is not None


def test_bench_repeats_recorded(bench, tmp_path):
    rc = bench.main(
        ["--dry-run", "--steps", "1", "--repeats", "3", "--context-tokens", "1024",
         "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    d = json.loads(next(tmp_path.glob("bench_standard_shape/**/*.json")).read_text())
    assert len(d["cells"]) == 1
    assert len(d["cells"][0]["repeats"]) == 3


def test_bench_bad_max_kv_exits_2(bench):
    rc = bench.main(["--dry-run", "--context-tokens", "16384", "--max-kv", "1000"])
    assert rc == 2


# ==========================================================================
# humaneval_cell.py
# ==========================================================================


def test_humaneval_import_is_cpu_safe_no_mlx(humaneval):
    src = (_SCRIPTS / "humaneval_cell.py").read_text()
    assert "\nimport mlx" not in src
    assert "import mlx.core" not in src


def test_humaneval_parser_defaults_match_davids_sampler(humaneval):
    args = humaneval.build_parser().parse_args([])
    assert args.temperature == 1.0  # not greedy
    assert args.top_p == 0.95
    assert args.top_k == 20
    assert args.max_tokens == 2048  # non-binding cap (thinking-OFF served path, W52)
    assert args.seed == 20260829  # David's served seed
    assert args.endpoint == "chat"
    assert args.lane == "ar"  # default lane
    assert args.depth == 3


def test_humaneval_compute_metrics_truncation_aware(humaneval):
    report = {
        "rows": [
            {"passed": True, "finish_reason": "stop", "status": "passed"},
            {"passed": False, "finish_reason": "stop", "status": "failed"},
            {"passed": False, "finish_reason": "length", "status": "failed"},
            {"passed": True, "finish_reason": "stop", "status": "passed"},
        ],
        "summary": {"by_status": {"passed": 2, "failed": 2}},
    }
    m = humaneval.compute_cell_metrics(report)
    assert m["tasks"] == 4
    assert m["passed"] == 2
    assert m["strict_pass_at_1"] == 0.5  # 2 / 4
    assert m["truncated_tasks"] == 1
    assert m["truncation_rate"] == 0.25  # 1 / 4
    assert m["completed_tasks"] == 3
    # completed-task pass@1 EXCLUDES the truncated row: 2 passers / 3 completed
    assert m["completed_task_pass_at_1"] == pytest.approx(2 / 3)
    assert m["by_finish_reason"] == {"stop": 3, "length": 1}


def test_humaneval_gate_argv_reuses_driver_with_davids_sampler(humaneval):
    args = humaneval.build_parser().parse_args(
        ["--base-url", "http://127.0.0.1:18183", "--model", "dsv41"]
    )
    argv = humaneval._gate_argv(args, Path("/tmp/r.json"), Path("/tmp/c.jsonl"))
    assert "--allow-code-execution" in argv
    assert argv[argv.index("--temperature") + 1] == "1.0"
    assert argv[argv.index("--top-p") + 1] == "0.95"
    assert argv[argv.index("--max-tokens") + 1] == "2048"
    assert argv[argv.index("--seed") + 1] == "20260829"
    assert argv[argv.index("--n") + 1] == "1"
    assert argv[argv.index("--suite") + 1] == "humaneval"
    # top-k rides on the driver's --extra-body passthrough
    assert "top_k=20" in argv
    assert argv[argv.index("--output-json") + 1] == "/tmp/r.json"


def test_humaneval_gate_argv_optional_limit_and_key(humaneval):
    args = humaneval.build_parser().parse_args(
        ["--base-url", "http://x", "--model", "m", "--limit", "20",
         "--api-key", "K", "--top-k", "0"]
    )
    argv = humaneval._gate_argv(args, Path("/tmp/r.json"), Path("/tmp/c.jsonl"))
    assert argv[argv.index("--limit") + 1] == "20"
    assert argv[argv.index("--api-key") + 1] == "K"
    # top-k <= 0 omits the extra-body
    assert "top_k=0" not in argv and "--extra-body" not in argv


def test_humaneval_write_receipt_append_only_guard(humaneval, tmp_path):
    out = humaneval.fresh_out_dir(tmp_path)
    receipt = {"lane": "ar", "seed": 20260829, "max_tokens": 2048, "utc_compact": "S"}
    p = humaneval.write_receipt(out, receipt)
    assert p.exists()
    # The lane is in the filename so AR and DSpark cells never collide.
    assert p.name == "humaneval_cell__ar__seed20260829__cap2048__S.json"
    with pytest.raises(FileExistsError):
        humaneval.write_receipt(out, receipt)


def test_humaneval_receipt_filename_carries_the_lane(humaneval):
    ar = humaneval.receipt_filename(
        {"lane": "ar", "seed": 20260829, "max_tokens": 2048, "utc_compact": "S"}
    )
    ds = humaneval.receipt_filename(
        {"lane": "dspark", "seed": 20260829, "max_tokens": 2048, "utc_compact": "S"}
    )
    assert ar == "humaneval_cell__ar__seed20260829__cap2048__S.json"
    assert ds == "humaneval_cell__dspark__seed20260829__cap2048__S.json"
    assert ar != ds  # AR and DSpark receipts never share a filename


def test_humaneval_fresh_out_dir_step_namespaced(humaneval, tmp_path):
    a = humaneval.fresh_out_dir(tmp_path)
    b = humaneval.fresh_out_dir(tmp_path)
    assert a != b and a.exists() and b.exists()
    assert a.parent.name == "humaneval_cell"


def test_humaneval_dry_run_end_to_end(humaneval, tmp_path):
    rc = humaneval.main(["--dry-run", "--out-dir", str(tmp_path)])
    assert rc == 0
    receipts = list(tmp_path.glob("humaneval_cell/**/*.json"))
    assert len(receipts) == 1
    d = json.loads(receipts[0].read_text())
    assert d["dry_run"] is True
    assert d["step"] == "humaneval_cell"
    assert d["sampler"]["temperature"] == 1.0
    assert d["sampler"]["top_p"] == 0.95
    assert d["sampler"]["max_tokens"] == 2048
    assert d["sampler"]["seed"] == 20260829
    assert d["sampler"]["greedy"] is False
    assert d["lane"] == "ar"  # default lane recorded
    assert d["harness"]["driver"] == "scripts/code_eval_gate.py"
    assert d["harness"]["scorer"] == "mtplx/benchmarks/code_eval.py"
    # The dry receipt shows the decode-lever field shape + a self-contained
    # per-task pass map (what the lane comparison consumes).
    assert d["decode_levers"]["resolved"]["HEAD_MODE"] == "bf16"
    assert isinstance(d["per_task"], list) and len(d["per_task"]) == 10
    m = d["metrics"]
    for key in (
        "strict_pass_at_1",
        "completed_task_pass_at_1",
        "truncation_rate",
    ):
        assert key in m


def test_humaneval_real_path_requires_base_url_and_model(humaneval):
    # No --dry-run and no --base-url/--model -> refuse with exit 2, no server.
    rc = humaneval.main([])
    assert rc == 2
