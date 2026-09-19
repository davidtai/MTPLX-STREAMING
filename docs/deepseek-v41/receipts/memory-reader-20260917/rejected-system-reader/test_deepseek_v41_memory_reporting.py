"""Memory accounting regressions; stdlib/fake allocator only, no MLX import."""
from __future__ import annotations

import importlib.util
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import deepseek_v41_memory_profile as profile

GIB = 1024**3
ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "deepseek_v41" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 9.
Pages active: 10.
Pages inactive: 20.
Pages speculative: 3.
Pages wired down: 4.
Anonymous pages: 12.
File-backed pages: 21.
Pages occupied by compressor: 5.
Pages stored in compressor: 17.
Swapins: 6.
Swapouts: 7.
"""


def test_system_total_includes_file_pages_and_physical_compressor():
    snap = profile._parse_vm_stat(VM_STAT)
    assert snap["used_bytes"] == (10 + 20 + 4 + 5) * 16384
    assert snap["non_file_used_bytes"] == (12 + 4 + 5) * 16384
    assert snap["file_backed_bytes"] == 21 * 16384
    assert snap["compressor_bytes"] == 5 * 16384
    assert snap["compressed_bytes"] == 17 * 16384


def test_broken_system_reader_is_unknown(monkeypatch):
    def unavailable():
        raise OSError("host statistics unavailable")

    monkeypatch.setattr(profile, "_mach_host_vm_reader", unavailable)
    monkeypatch.setattr(profile.sys, "platform", "darwin")
    assert profile.box_memory_snapshot()["ok"] is False


@pytest.fixture
def native_host_reader(monkeypatch):
    """Exercise the ctypes call boundary without requiring a macOS kernel."""
    import ctypes

    state = {"result": 0, "words": 38, "release_result": 0,
             "acquired": [], "released": []}

    def host_self():
        port = 100 + len(state["acquired"])
        state["acquired"].append(port)
        return port

    def task_self():
        return 42

    def statistics(host, flavor, info, count):
        assert host == state["acquired"][-1] and flavor == 4
        assert ctypes.sizeof(info._obj) == 152 and count._obj.value == 38
        for name, value in {
            "free_count": 12, "active_count": 10, "inactive_count": 20,
            "wire_count": 4, "speculative_count": 3,
            "internal_page_count": 12, "external_page_count": 21,
            "compressor_page_count": 5,
            "total_uncompressed_pages_in_compressor": 17,
            "swapins": 6, "swapouts": 7,
        }.items():
            setattr(info._obj, name, value)
        count._obj.value = state["words"]
        return state["result"]

    def deallocate(task, host):
        assert task == 42
        state["released"].append(host)
        return state["release_result"]

    library = SimpleNamespace(mach_host_self=host_self, mach_task_self=task_self,
                              host_statistics64=statistics,
                              mach_port_deallocate=deallocate)
    profile._mach_host_vm_reader.cache_clear()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(profile.os, "sysconf", lambda name: 16384)
    monkeypatch.setattr(profile.sys, "platform", "darwin")
    yield state
    profile._mach_host_vm_reader.cache_clear()


def test_native_system_counters_match_vm_stat_and_balance_host_ports(native_host_reader):
    expected = profile._parse_vm_stat(VM_STAT)
    for _ in range(2):
        snapshot = profile.box_memory_snapshot()
        assert snapshot == {**expected, "ok": True,
                            "source": "host_statistics64",
                            "used_includes_file_cache": True}
    assert native_host_reader["acquired"] == [100, 101]
    assert native_host_reader["released"] == [100, 101]


@pytest.mark.parametrize("field,value", [
    ("result", 5), ("words", 24), ("words", 39), ("release_result", 5),
])
def test_native_system_read_failures_stay_unknown_and_release_ports(
    native_host_reader, field, value,
):
    native_host_reader[field] = value
    snapshot = profile.box_memory_snapshot()
    assert snapshot["ok"] is False
    assert "used_bytes" not in snapshot
    assert native_host_reader["acquired"] == native_host_reader["released"] == [100]


def test_footprint_reader_never_substitutes_rss(monkeypatch):
    bench = load_script("bench_standard_shape")
    monkeypatch.setattr(profile, "process_rss_snapshot", lambda: {
        "phys_footprint_bytes": None, "resident_bytes": 50 * GIB,
        "peak_maxrss_bytes": 80 * GIB,
    })
    assert bench._phys_footprint_bytes() is None


def test_receipt_units_are_explicit_and_no_sample_stays_unknown():
    bench = load_script("bench_standard_shape")
    probe = bench._MLXMemProbe(SimpleNamespace(get_peak_memory=lambda: 2 * GIB))
    block = probe.memory_block()
    assert block["mlx_peak_bytes"] == 2 * GIB
    assert block["mlx_peak_gib"] == 2.0
    assert block["mlx_peak_gb"] == 2 * GIB / 1e9
    assert block["process_footprint_peak_bytes"] is None
    assert block["system_used_peak_bytes"] is None

    def failed_counter():
        raise RuntimeError("allocator counter unavailable")

    for mx in (SimpleNamespace(), SimpleNamespace(get_peak_memory=failed_counter),
               SimpleNamespace(get_peak_memory=lambda: -1)):
        missing = bench._MLXMemProbe(mx).memory_block()
        assert missing["mlx_peak_bytes"] is None
        assert missing["mlx_peak_gb"] is None
        assert missing["mlx_peak_gib"] is None


def test_ab_preserves_measured_system_peak_and_never_calls_rss_footprint():
    ab = load_script("ab_decode_env_levers")
    block = {"process_peak_rss_gb": 90.0, "process_footprint_peak_bytes": None,
             "system_used_peak_bytes": 112_000_000_000,
             "mlx_peak_bytes": 2 * GIB, "schema_version": 2}
    mem = ab._ab_memory_block(block, {})
    assert mem["process_footprint_peak_gb"] is None
    assert mem["system_used_peak_gb"] == 112.0
    assert mem["mlx_peak_gb"] == 2 * GIB / 1e9


def test_box_usage_is_measured_and_baseline_sum_is_only_an_estimate(monkeypatch):
    ab = load_script("ab_decode_env_levers")
    monkeypatch.delenv("MTPLX_DSV41_BOX_BASELINE_GB", raising=False)
    mem = {"process_footprint_peak_bytes": 80_000_000_000,
           "process_footprint_peak_gb": 80.0, "system_used_peak_gb": 112.0}
    ab._inject_box_used(mem, SimpleNamespace(box_baseline_gb=10.0))
    assert mem["box_used_gb"] == 112.0
    assert mem["baseline_plus_process_peak_estimate_gb"] == 90.0


def test_headline_does_not_turn_missing_measurements_into_zero():
    ab = load_script("ab_decode_env_levers")
    line = ab._memory_headline({})
    assert "process_footprint_peak_gb=n/a" in line
    assert "box_used_gb=n/a" in line


@pytest.mark.parametrize("mode,candidate_sha,expected_status", [
    ("dspark", "target-changed", 1),
    ("dspark", "target-control", 0),
    ("ar", "target-changed", 0),
])
def test_ab_summary_uses_measured_pass_with_reused_ar_reference(
    monkeypatch, tmp_path, capsys, mode, candidate_sha, expected_status,
):
    from types import ModuleType

    ab = load_script("ab_decode_env_levers")
    core = ModuleType("mlx.core")
    core.random = SimpleNamespace(seed=lambda value: None)
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(ab, "_load_bench_module", lambda: None)
    monkeypatch.setattr(ab, "_apply_cell_prompt_guard", lambda args: None)
    monkeypatch.setattr(ab, "_write_output_sidecars", lambda *args: None)
    rows = {}
    for arm, sha, tps, budget in (
        ("control", "target-control", 11.5, 93_000_000_000),
        ("all_levers", candidate_sha, 12.5, 94_000_000_000),
    ):
        rows[arm] = {
            "arm": arm, "token_ids_sha256": "reused-ar-reference",
            "decode_tok_s": None, "memory": None,
            "resolved_plan": {"memory_limit_bytes": 90_000_000_000},
            "dspark": {
                "token_ids_sha256": sha, "decode_tok_s": tps,
                "resolved_plan": {"memory_limit_bytes": 90_000_000_000},
                "serve_stream_counters": {"slot_plan": {"memory_limit_bytes": budget}},
                "memory": {"mlx_peak_gb": 87.0, "process_footprint_peak_gb": 90.0,
                           "box_used_gb": 100.0},
            },
        }
    monkeypatch.setattr(ab, "_run_arm", lambda args, arm, bench, mx: rows[arm])
    status = ab.main(["--arms", "control", "all_levers", "--decode-mode", mode,
                      "--out", str(tmp_path / "receipts.jsonl")])
    output = capsys.readouterr().out
    assert status == expected_status
    if mode == "dspark":
        assert "dspark decode_tok_s=11.5 mlx_peak_gb=87.00" in output
        assert "process_footprint_peak_gb=90.00 box_used_gb=100.00" in output
        assert "decode_tok_s 11.500 -> 12.500" in output
        assert "DIFFERENT plan_limit_bytes values ([93000000000, 94000000000])" in output
        assert ("FAIL: all_levers changed the decoded tokens" in output) == (expected_status == 1)
    else:
        assert "ar decode_tok_s=None mlx_peak_gb=n/a" in output
        assert "all arms ran plan_limit_bytes=90000000000" in output
        assert "FAIL" not in output


def test_model_levers_follow_each_arm_after_module_import(monkeypatch):
    from types import ModuleType

    ab = load_script("ab_decode_env_levers")
    bench = load_script("bench_standard_shape")
    monkeypatch.setattr(ab.os, "environ", {})
    model = ModuleType("benchmark_model")
    model.__dict__.update(
        _HC_COMPILE=False, _ATTN_COMPILE=False, _ATTN_WIN_MEMO=False,
        _HC_COMPILE_MAX_ROWS=7, _ATTN_COMPILE_MAX_ROWS=32,
        _stime=SimpleNamespace(recording=lambda: False, is_prefill=lambda: False),
    )
    source = (ROOT / "mtplx/models/deepseek_v41.py").read_text()
    routes = [node for node in ast.parse(source).body
              if isinstance(node, ast.FunctionDef)
              and node.name in ("_hc_use_compile", "_attn_use_compile")]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *routes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "native-route-selection", "exec"), model.__dict__)
    for arm, expected in (("all_levers", (True, True, True)),
                          ("control", (False, False, False)),
                          ("hc_compile", (True, False, False))):
        ab._apply_arm_env(arm)
        bound = bench.bind_model_levers(model)
        assert tuple(bound.values()) == expected
        assert model._hc_use_compile(SimpleNamespace(shape=(1, 6, 4, 5120))) == expected[0]
        assert model._attn_use_compile(6) == expected[1]
        assert not model._hc_use_compile(SimpleNamespace(shape=(1, 512, 4, 5120)))
        assert not model._attn_use_compile(512)


def test_sampler_records_short_run_end_peak_and_same_sample_counters(monkeypatch):
    bench = load_script("bench_standard_shape")
    snapshots = iter([
        {"process": {"phys_footprint_bytes": GIB}, "box": {"ok": True, "used_bytes": 3 * GIB}},
        {"process": {"phys_footprint_bytes": 2 * GIB}, "box": {"ok": True, "used_bytes": 4 * GIB}},
    ])
    monkeypatch.setattr(profile, "host_memory_snapshot", lambda: next(snapshots))
    # Disable periodic reads to isolate the two mandatory boundary observations.
    sampler = bench._MemorySampler(interval_s=60)
    monkeypatch.setattr(sampler, "_loop", lambda: None)
    sampler.start().stop()
    block = bench._MLXMemProbe(SimpleNamespace(get_peak_memory=lambda: GIB)).memory_block(sampler)
    assert block["process_footprint_peak_bytes"] == 2 * GIB
    assert block["system_used_peak_bytes"] == 4 * GIB
    assert block["samples"][1]["process"]["phys_footprint_bytes"] == 2 * GIB
    assert block["samples"][1]["box"]["used_bytes"] == 4 * GIB
    assert block["sample_count"] == 2


def test_host_snapshot_has_one_monotonic_interval():
    snap = profile.host_memory_snapshot()
    assert snap["sample_start_monotonic_ns"] <= snap["sample_end_monotonic_ns"]
    if sys.platform == "darwin":
        assert snap["process"]["phys_footprint_bytes"] > 0
        assert snap["box"]["used_bytes"] > 0
    else:
        assert snap["box"]["ok"] is False


def test_serving_health_reports_os_memory_without_loading_a_model(monkeypatch):
    source = (ROOT / "mtplx/server/openai.py").read_text()
    tree = ast.parse(source)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_deepseek_v41_memory_health")
    scope = {"_served_model_type_is_deepseek_v41": lambda args: args.deepseek}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "health", "exec"), scope)
    monkeypatch.setattr(profile, "host_memory_snapshot", lambda: {"observed": 123})
    health = scope["_deepseek_v41_memory_health"]
    assert health(SimpleNamespace(deepseek=True)) == {"observed": 123}
    assert health(SimpleNamespace(deepseek=False)) is None
    endpoint = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == "health")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "_deepseek_v41_memory_health" for n in ast.walk(endpoint))


def test_standard_cell_reports_memory_and_decimal_units():
    bench = load_script("bench_standard_shape")
    model = bench._FakeModel()
    probe = bench._DryMemProbe()
    metrics = bench.bench_one_cell(
        model=model, tokenizer=bench._FakeTokenizer(), ops=bench._FakeOps(),
        mem_probe=probe, gather_probe=bench._GatherProbe(model),
        prompt_ids=[0, 2], steps=2,
    )
    assert metrics["memory"]["schema_version"] == 2
    assert metrics["memory"]["sample_count"] >= 2
    assert metrics["peak_mlx_gb"] == metrics["peak_mlx_bytes"] / 1e9
    assert metrics["process_rss_gb"] == metrics["process_rss_bytes"] / 1e9


def test_standard_runner_uses_shared_110gb_plan(monkeypatch, tmp_path):
    bench = load_script("bench_standard_shape")
    monkeypatch.setattr(bench.os, "environ", {"MTPLX_DSV41_BOX_BASELINE_GB": "12"})
    args = bench.build_parser().parse_args(["--out", str(tmp_path / "receipt.json")])
    derivation = bench._resolve_memory_derivation(args)
    assert args._dsv41_target_plan["box_target_gb"] == 110
    assert derivation.memory_limit_bytes == args._dsv41_target_plan["engine_budget_bytes"]


def test_pinned_plan_restores_python_capacity_and_rejects_larger_live_baseline(monkeypatch, tmp_path):
    import json
    ab = load_script("ab_decode_env_levers")
    monkeypatch.setattr(ab.os, "environ", {"MTPLX_DSV41_BOX_BASELINE_GB": "12"})
    args = SimpleNamespace(out=tmp_path / "arm.json")
    ab.os.environ["MTPLX_DSV41_SESSION_BANK_GIB"] = "2"
    pinned = ab._resolve_target_plan(args)
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(pinned))
    args.memory_plan_from = path
    ab.os.environ["MTPLX_ENGRAM_CACHE_LIMIT"] = "2GiB"
    ab.os.environ["MTPLX_DSV41_SESSION_BANK_GIB"] = "4"
    replay = ab._resolve_target_plan(args)
    assert replay["engine_budget_bytes"] == pinned["engine_budget_bytes"]
    assert ab.os.environ["MTPLX_ENGRAM_CACHE_LIMIT"] == str(256 * 1024**2)
    assert float(ab.os.environ["MTPLX_DSV41_SESSION_BANK_GIB"]) == 2
    ab.os.environ["MTPLX_DSV41_BOX_BASELINE_GB"] = "15"
    with pytest.raises(ValueError, match="baseline"):
        ab._resolve_target_plan(args)


@pytest.mark.parametrize("peak_available", [True, False])
def test_standard_cell_keeps_ar_and_dspark_memory_windows_separate(monkeypatch, peak_available):
    bench = load_script("bench_standard_shape")
    class Probe(bench._DryMemProbe):
        resets = 0
        def reset_peak(self):
            self.resets += 1
        def peak_bytes(self):
            return self.resets * GIB if peak_available else None
    stats = {key: 0 for key in ("tokens_per_cycle", "accept_rate", "accept_rate_by_depth",
             "drafted_by_depth", "accepted_by_depth", "cycles", "verify_calls",
             "verify_decode_phase", "per_cycle", "phase_time_s")}
    stats["verify_chunks"] = [4]
    stats["speculative_depth"] = 3
    monkeypatch.setitem(sys.modules, "mtplx.models.deepseek_v41_dspark_decode", SimpleNamespace(
        DSparkDecodeStats=lambda: SimpleNamespace(to_dict=lambda: stats),
        dspark_generate=lambda *a, **k: (k["completion_callback"](), [1, 2, 3])[1],
    ))
    monkeypatch.setitem(sys.modules, "mtplx.sampling", SimpleNamespace(SamplerConfig=lambda **k: None))
    model = bench._FakeModel()
    model.mtp = True
    result = bench.bench_one_cell(model=model, tokenizer=bench._FakeTokenizer(),
        ops=bench._FakeOps(), mem_probe=Probe(), gather_probe=bench._GatherProbe(model),
        prompt_ids=[0, 2], steps=2, decode_mode="dspark")
    assert result["peak_mlx_bytes"] == (GIB if peak_available else None)
    assert result["memory"]["mlx_peak_bytes"] == (GIB if peak_available else None)
    assert result["dspark"]["memory"]["mlx_peak_bytes"] == (2 * GIB if peak_available else None)
    assert result["peak_mlx_gb"] == result["memory"]["mlx_peak_gb"]
    assert result["dspark"]["peak_mlx_gb"] == result["dspark"]["memory"]["mlx_peak_gb"]
    assert (result["memory"]["samples"][-1]["sample_end_monotonic_ns"] <=
            result["dspark"]["memory"]["samples"][0]["sample_start_monotonic_ns"])


@pytest.mark.parametrize("peak_bytes", [9 * GIB, None])
def test_dspark_decode_memory_boundaries_exclude_prefill_and_cache_release(monkeypatch, peak_bytes):
    ab = load_script("ab_decode_env_levers")
    active = {"bytes": GIB}
    mx = SimpleNamespace(get_active_memory=lambda: active["bytes"],
                         get_cache_memory=lambda: 0)
    sampler = SimpleNamespace(start=lambda: None, stop=lambda: None)
    probe = SimpleNamespace(reset_peak=lambda: None, new_sampler=lambda: sampler,
                            peak_bytes=lambda: peak_bytes,
                            memory_block=lambda _: {"mlx_peak_bytes": peak_bytes})

    def generate(*args, **kwargs):
        active["bytes"] = 7 * GIB
        kwargs["prefill_callback"]({"prompt_tokens": 2})
        active["bytes"] = 9 * GIB
        kwargs["completion_callback"]()
        active["bytes"] = GIB
        return [3, 4]

    monkeypatch.setitem(sys.modules, "mtplx.models.deepseek_v41_dspark_decode", SimpleNamespace(
        DSparkDecodeStats=lambda: SimpleNamespace(to_dict=lambda: {}),
        DivergenceCapture=lambda _: None, dspark_generate=generate,
    ))
    monkeypatch.setitem(sys.modules, "mtplx.sampling", SimpleNamespace(SamplerConfig=lambda **k: None))
    monkeypatch.setattr(ab, "_stream_counters_snapshot", lambda _: {})
    monkeypatch.setattr(ab, "_reset_dspark_engagement_counters", lambda: {})
    monkeypatch.setattr(ab, "_capture_dspark_engagement", lambda _: {})
    monkeypatch.setattr(ab, "_runner_receipt_blocks", lambda _: {})
    result = ab._generate_dspark(model=object(), mx=mx, mem_probe=probe,
                                 prompt_ids=[1, 2], steps=1, depth=3)
    assert result["memory"]["mlx_active_bytes_at_decode_start"] == 7 * GIB
    assert result["memory"]["mlx_active_bytes_at_decode_end"] == 9 * GIB
    assert result["memory"]["mlx_peak_bytes"] == peak_bytes
    assert result["peak_gb"] == result["memory"]["mlx_peak_gb"]


@pytest.mark.parametrize("max_tokens", [1, 3])
def test_dspark_completion_observes_cache_before_release(monkeypatch, max_tokens):
    path = ROOT / "mtplx/models/deepseek_v41_dspark_decode.py"
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef)
             and n.name in ("dspark_generate", "_normalize_verify_chunks")]
    assert len(nodes) == 2
    alive = []
    class Cache:
        def __init__(self):
            alive.append(1)
        def __del__(self):
            alive.pop()
    class Tensor:
        def __getitem__(self, index):
            return self
    tensor = Tensor()
    scope = {
        "mx": SimpleNamespace(array=lambda x: tensor, eval=lambda *x: None),
        "np": SimpleNamespace(random=SimpleNamespace(default_rng=lambda seed: None)),
        "_confidence_threshold_from_env": lambda value: value,
        "_target_forward": lambda model: lambda *x: (tensor, tensor),
        "_seed_prefill_state": lambda model, hidden, caches: hidden,
        "_fire_prefill_callback": lambda *x: None, "_is_stop": lambda *x: False,
        "_verify_decode_phase_enabled": lambda: False,
        "_decode_cycles": lambda **k: ([2, 3], "length"),
    }
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    monkeypatch.setitem(sys.modules, "mtplx.generation", SimpleNamespace(_sample_from_logits=lambda *a: (1, None)))
    seen = []
    class Model(SimpleNamespace):
        def __call__(self, *args, **kwargs):
            return tensor, tensor
    model = Model(mtp=SimpleNamespace(block_size=3, seed_main=lambda *a: None),
                  make_cache=Cache, make_mtp_cache=Cache)
    result = scope["dspark_generate"](model, [0], max_tokens=max_tokens, sampler=None,
        stats=SimpleNamespace(), completion_callback=lambda: seen.append(len(alive)))
    assert seen == [2]
    assert alive == []
    assert len(result) == max_tokens


def test_standard_dspark_releases_ar_state_and_reports_decode_only(monkeypatch):
    import weakref
    bench = load_script("bench_standard_shape")
    clock = [0.0]
    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])
    refs = []
    class Cache(dict):
        pass
    class Logits(list):
        pass
    class Model(bench._FakeModel):
        mtp = True
        def make_cache(self):
            cache = Cache(offset=0)
            refs.append(weakref.ref(cache))
            return cache
        def __call__(self, *args, **kwargs):
            clock[0] += 1
            logits = Logits(super().__call__(*args, **kwargs))
            refs.append(weakref.ref(logits))
            return logits
    def dspark(*args, **kwargs):
        assert all(ref() is None for ref in refs), "AR state is still resident"
        clock[0] += 10
        kwargs["prefill_callback"]({"prompt_eval_time_s": 10.0})
        clock[0] += 2
        kwargs["completion_callback"]()
        return [1, 2, 3]
    stats = {key: 0 for key in ("tokens_per_cycle", "accept_rate", "accept_rate_by_depth",
        "drafted_by_depth", "accepted_by_depth", "cycles", "verify_calls",
        "verify_decode_phase", "per_cycle", "phase_time_s")}
    stats["verify_chunks"] = [6]
    stats["speculative_depth"] = 5
    monkeypatch.setitem(sys.modules, "mtplx.models.deepseek_v41_dspark_decode", SimpleNamespace(
        DSparkDecodeStats=lambda: SimpleNamespace(to_dict=lambda: stats), dspark_generate=dspark))
    monkeypatch.setitem(sys.modules, "mtplx.sampling", SimpleNamespace(SamplerConfig=lambda **k: None))
    model = Model()
    result = bench.bench_one_cell(model=model, tokenizer=bench._FakeTokenizer(),
        ops=bench._FakeOps(), mem_probe=bench._DryMemProbe(), gather_probe=bench._GatherProbe(model),
        prompt_ids=[0, 2], steps=2, decode_mode="dspark", dspark_depth=6)
    dsp = result["dspark"]
    assert dsp["depth"] == 5
    assert dsp["requested_depth"] == 6
    assert dsp["verify_chunks"] == [6]
    assert dsp["pass_wall_s"] == 12
    assert dsp["decode_wall_s"] == 2
    assert dsp["decode_tokens"] == 2
    assert dsp["decode_tok_s"] == 1
