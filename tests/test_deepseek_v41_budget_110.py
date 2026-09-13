"""Construction-time budget arithmetic, using fake allocators; no MLX import."""
from types import SimpleNamespace

import pytest

from mtplx import deepseek_v41_memory_profile as profile
from mtplx import expert_runtime as runtime

GIB = 1024**3
GB = 1_000_000_000


def test_default_target_and_host_caches_are_inside_110gb():
    budget = runtime.resolve_box_target_mlx_limit_bytes(
        {runtime.BOX_TARGET_ENV: "default", runtime.BOX_BASELINE_ENV: "12"})
    assert budget["box_target_gb"] == 110.0
    assert budget["python_cache_budget"]["engram_layer_count"] == 2
    assert budget["python_cache_budget"]["engram_payload_bytes"] == 512 * 1024**2
    assert budget["host_overhead_bytes"] >= budget["python_cache_budget"]["required_host_bytes"]
    assert (budget["mlx_limit_bytes"] + budget["box_baseline_bytes"] +
            budget["host_overhead_bytes"]) == 110 * GB
    assert budget["engine_budget_bytes"] + budget["engine_reserve_bytes"] == budget["mlx_limit_bytes"]


def test_operator_python_cache_is_priced_at_full_capacity():
    budget = runtime.resolve_box_target_mlx_limit_bytes({runtime.BOX_TARGET_ENV: "110",
        runtime.BOX_BASELINE_ENV: "12", "MTPLX_ENGRAM_CACHE_LIMIT": "2GiB"})
    assert budget["python_cache_budget"]["engram_payload_bytes"] == 4 * GIB
    assert budget["host_overhead_bytes"] > 8 * GIB


def test_small_host_override_cannot_hide_python_cache_capacity():
    with pytest.raises(runtime.ExpertStreamingConfigurationError, match="Python"):
        runtime.resolve_box_target_mlx_limit_bytes({runtime.BOX_TARGET_ENV: "110",
            runtime.BOX_BASELINE_ENV: "12", runtime.BOX_HOST_OVERHEAD_ENV: "0.5"})


def test_target_rejects_impossible_physical_and_wired_capacity():
    with pytest.raises(runtime.ExpertStreamingConfigurationError, match="physical"):
        runtime.resolve_box_target_mlx_limit_bytes({runtime.BOX_TARGET_ENV: "110",
            runtime.BOX_BASELINE_ENV: "12"}, memsize_bytes=110 * GB)
    with pytest.raises(runtime.ExpertStreamingConfigurationError, match="wired"):
        runtime.resolve_box_target_mlx_limit_bytes({runtime.BOX_TARGET_ENV: "130",
            runtime.BOX_BASELINE_ENV: "1"}, memsize_bytes=256 * GIB)


def test_serving_config_uses_same_derived_engine_budget(monkeypatch):
    monkeypatch.setattr(profile, "box_memory_snapshot", lambda: {
        "ok": True, "used_bytes": 12 * GB, "non_file_used_bytes": 6 * GB})
    cfg = runtime.ExpertStreamingConfig(model_key="deepseek-v41-flash-expert-mxfp4",
        memory_limit_bytes=60 * GIB, max_live_kv_tokens=16384)
    env = {}
    resolved, budget = runtime.prepare_deepseek_v41_memory_config(cfg, env=env)
    assert resolved.memory_limit_bytes == budget["engine_budget_bytes"]
    assert env[runtime.BOX_TARGET_ENV] == "110"
    assert env["MTPLX_ENGRAM_CACHE_LIMIT"] == str(256 * 1024**2)
    assert cfg.memory_limit_bytes == 60 * GIB
    assert env[runtime.BOX_BASELINE_ENV] == "12"


def test_non_deepseek_serving_keeps_config_and_environment():
    cfg = SimpleNamespace(model_key="another-model")
    env = {}
    assert runtime.prepare_deepseek_v41_memory_config(cfg, env=env) == (cfg, None)
    assert env == {}


@pytest.mark.parametrize("api", ["set_wired_limit", "set_cache_limit"])
@pytest.mark.parametrize("failure", ["missing", "refused"])
def test_target_requires_successful_wired_and_cache_caps(api, failure):
    def refused(_):
        raise RuntimeError("driver refused")
    mx = SimpleNamespace(set_memory_limit=lambda _: 0, set_wired_limit=lambda _: 0,
                         set_cache_limit=lambda _: 0)
    setattr(mx, api, None if failure == "missing" else refused)
    plan = SimpleNamespace(total_limit_bytes=10 * GIB, runtime_reserve_bytes=GIB,
                           io_staging_bytes=0, mmap_islands_wired=True, mmap_island_bytes=0,
                           fixed_bytes=2 * GIB, persistent_cache_bytes=8 * GIB)
    with pytest.raises(runtime.ExpertStreamingConfigurationError, match="required.*limit"):
        runtime.apply_mlx_memory_cap(plan, mx_module=mx,
            env={runtime.BOX_TARGET_ENV: "110", runtime.BOX_BASELINE_ENV: "12"})


def test_serving_preserves_explicit_smaller_engine_budget():
    cfg = runtime.ExpertStreamingConfig(model_key="deepseek-v41-flash-expert-mxfp4",
        memory_limit_bytes=60 * GIB, max_live_kv_tokens=16384)
    resolved, budget = runtime.prepare_deepseek_v41_memory_config(cfg,
        env={runtime.BOX_BASELINE_ENV: "12"}, preserve_memory_limit=True)
    assert resolved.memory_limit_bytes == 60 * GIB
    assert budget["configured_engine_budget_bytes"] == 60 * GIB
    with pytest.raises(runtime.ExpertStreamingConfigurationError, match="explicit.*budget"):
        runtime.prepare_deepseek_v41_memory_config(cfg,
            env={runtime.BOX_BASELINE_ENV: "65"}, preserve_memory_limit=True)


def test_python_index_reserve_covers_full_and_empty_cache():
    from collections import OrderedDict
    import sys
    for slots in (1024, 4096, 65536):
        free = list(range(slots))
        free_bytes = sys.getsizeof(free) + sum(map(sys.getsizeof, free))
        lru = OrderedDict((n + 2**32, n) for n in range(slots))
        full_bytes = sys.getsizeof(lru) + sum(
            sys.getsizeof(k) + sys.getsizeof(v) for k, v in lru.items())
        assert max(free_bytes, full_bytes) < slots * 256


def test_serving_disables_unbounded_ssd_cache_and_honors_allocator_cache_override():
    cfg = runtime.ExpertStreamingConfig(model_key="deepseek-v41-flash-expert-mxfp4",
        memory_limit_bytes=60 * GIB, max_live_kv_tokens=16384)
    args = SimpleNamespace(ssd_session_cache="on", mlx_cache_limit="2GiB", _cli_flags=set())
    _, budget = runtime.prepare_deepseek_v41_memory_config(cfg,
        env={runtime.BOX_BASELINE_ENV: "12"}, args=args)
    assert args.ssd_session_cache == "off"
    assert budget["allocator_cache_limit_bytes"] == 2 * GIB
    assert budget["session_bank_capacity_bytes"] == 2 * GIB
    assert budget["engine_reserve_bytes"] == max(
        budget["transient_band_bytes"], budget["active_overshoot_bytes"] + 2 * GIB) + 6 * GIB
    args.ssd_session_cache = "on"
    args._cli_flags = {"ssd-session-cache"}
    with pytest.raises(runtime.ExpertStreamingConfigurationError, match="SSD"):
        runtime.prepare_deepseek_v41_memory_config(cfg,
            env={runtime.BOX_BASELINE_ENV: "12"}, args=args)


@pytest.mark.parametrize("key", ["MTPLX_MEMORY_LIMIT_BYTES", "MTPLX_WIRED_LIMIT_BYTES",
                                 "MTPLX_MEMORY_BUDGET"])
def test_serving_does_not_silently_ignore_generic_caps(key):
    cfg = runtime.ExpertStreamingConfig(model_key="deepseek-v41-flash-expert-mxfp4",
        memory_limit_bytes=60 * GIB, max_live_kv_tokens=16384)
    with pytest.raises(runtime.ExpertStreamingConfigurationError, match="conflicts"):
        runtime.prepare_deepseek_v41_memory_config(cfg,
            env={runtime.BOX_BASELINE_ENV: "12", key: str(40 * GIB)},
            args=SimpleNamespace(ssd_session_cache="off", mlx_cache_limit=None))


@pytest.mark.parametrize("explicit", [False, True, "json_default"])
def test_server_constructor_passes_resolved_plan_to_loader_and_health(monkeypatch, explicit):
    """Execute the real constructor through pre-load planning; no MLX import."""
    import ast
    import os
    import sys
    from pathlib import Path
    import __future__
    source = Path(__file__).resolve().parents[1] / "mtplx/server/openai.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ServerState")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    late_import = next(i for i, n in enumerate(init.body)
                       if isinstance(n, ast.ImportFrom) and n.module == "mtplx.expert_cli"
                       and any(a.name == "apply_expert_profile_child_env" for a in n.names))
    late_apply = init.body[late_import + 1]
    # The following try starts unrelated paged-KV initialization.
    init.body = init.body[:next(i for i, n in enumerate(init.body) if isinstance(n, ast.Try))]
    cfg = runtime.ExpertStreamingConfig(model_key="deepseek-v41-flash-expert-mxfp4",
        memory_limit_bytes=60 * GIB, max_live_kv_tokens=16384)
    args = SimpleNamespace(model="fake", context_window=0, ssd_session_cache="on",
        _resolved_expert_profile_customized_fields=("memory_limit_bytes",) if explicit is True else (),
        _expert_memory_limit_explicit=(explicit == "json_default"),
        _resolved_expert_effective_config={"memory_limit_bytes": 60 * GIB})
    cli = SimpleNamespace(expert_streaming_load_kwargs=lambda *a: {
        "expert_streaming_config": cfg, "mtp": False},
        apply_expert_profile_child_env=lambda a, env: env.update(
            {"MTPLX_SESSION_BANK_MAX_BYTES": "2G"}))
    monkeypatch.setitem(sys.modules, "mtplx.expert_cli", cli)
    monkeypatch.setattr(os, "environ", {runtime.BOX_BASELINE_ENV: "12"})
    scope = {"os": os, **{name: lambda *a: None for name in (
        "_coerce_family_verify_strategy", "_validate_mtp_batch_settings", "_validate_hyper_settings")}}
    exec(compile(ast.Module(body=[init], type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    state = SimpleNamespace()
    scope["__init__"](state, args)
    loaded = state.expert_streaming_load_kwargs["expert_streaming_config"].memory_limit_bytes
    assert loaded == (60 * GIB if explicit else state.dsv41_memory_budget["engine_budget_bytes"])
    assert args._resolved_expert_effective_config["memory_limit_bytes"] == loaded
    assert args._resolved_expert_profile_customized is True
    assert args.ssd_session_cache == "off"
    scope.update(self=state, args=args, apply_expert_profile_child_env=cli.apply_expert_profile_child_env)
    exec(compile(ast.Module(body=[late_apply], type_ignores=[]), str(source), "exec"), scope)
    assert os.environ["MTPLX_SESSION_BANK_MAX_BYTES"] == "2000000000"


@pytest.mark.parametrize("mode,flags,env_mode,forwarded,expected", [
    ("on", set(), None, False, "on"),
    ("on", {"ssd-session-cache"}, None, True, "on"),
    ("off", set(), None, True, "off"),
    ("on", set(), "off", True, "off"),
    ("on", set(), "on", True, "on"),
    ("on", set(), "write-only", True, "write-only"),
    ("on", {"ssd-session-cache"}, "off", True, "on")])
def test_public_launch_preserves_ssd_default_provenance(monkeypatch, mode, flags, env_mode, forwarded, expected):
    import ast
    import os
    import shlex
    from pathlib import Path
    import __future__
    path = Path(__file__).resolve().parents[1] / "mtplx/commands/public.py"
    tree = ast.parse(path.read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    helper = funcs.get("_forward_ssd_session_cache_mode")
    scope = {"os": os, "shlex": shlex}
    monkeypatch.delenv("MTPLX_SSD_SESSION_CACHE", raising=False)
    if env_mode is not None:
        monkeypatch.setenv("MTPLX_SSD_SESSION_CACHE", env_mode)
    nodes = [funcs["_batching_command_suffix"]] + ([helper] if helper else [])
    if "_resolved_ssd_session_cache_mode" in funcs:
        nodes.append(funcs["_resolved_ssd_session_cache_mode"])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    args = SimpleNamespace(ssd_session_cache=mode, _cli_flags=flags)
    suffix = scope["_batching_command_suffix"](args)
    assert ("--ssd-session-cache " in suffix) == forwarded
    if forwarded:
        assert f"--ssd-session-cache {expected}" in suffix
    # Execute the real launcher's mode-forwarding statements as well.
    launch = funcs["cmd_serve_public"].body
    index = next(i for i, n in enumerate(launch) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "ssd_session_cache" for t in n.targets))
    scope.update(args=args, cmd=[])
    exec(compile(ast.Module(body=launch[index:index+2], type_ignores=[]), str(path), "exec"), scope)
    assert ("--ssd-session-cache" in scope["cmd"]) == forwarded
    if forwarded:
        assert scope["cmd"][1] == expected


def test_serving_bank_budget_normalizes_units_for_actual_bank_parser():
    cfg = runtime.ExpertStreamingConfig(model_key="deepseek-v41-flash-expert-mxfp4",
        memory_limit_bytes=60 * GIB, max_live_kv_tokens=16384)
    env = {runtime.BOX_BASELINE_ENV: "12", "MTPLX_SESSION_BANK_MAX_BYTES": "2G"}
    _, budget = runtime.prepare_deepseek_v41_memory_config(cfg, env=env,
        args=SimpleNamespace(ssd_session_cache="off"))
    assert env["MTPLX_SESSION_BANK_MAX_BYTES"] == "2000000000"
    assert budget["session_bank_capacity_bytes"] == int(env["MTPLX_SESSION_BANK_MAX_BYTES"])
    assert budget["session_bank_reserve_bytes"] == 3 * int(env["MTPLX_SESSION_BANK_MAX_BYTES"])
