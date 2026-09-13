import os, subprocess, sys
cmd=[sys.executable, '-m', 'pytest', '-o', 'addopts=', '-q',
 'tests/test_deepseek_v41_w121_load_path.py',
 'tests/test_deepseek_v41_default_peak_budget.py',
 'tests/test_deepseek_v41_budget_110.py',
 'tests/test_deepseek_v41_w121_target_limit.py',
 'tests/test_deepseek_v41_w121_wired_limit.py',
 'tests/test_deepseek_v41_memory_reporting.py',
 'tests/test_deepseek_v41_io_policy.py',
 'tests/test_deepseek_v41_resident_io.py',
 'tests/test_ngram_file_cache_policy.py',
 'tests/test_dsv41_prefetch_timeout_ownership.py',
 'tests/test_expert_slots_runtime.py',
 'tests/test_expert_streaming.py']
result=subprocess.run(cmd, timeout=180)
if result.returncode: raise SystemExit(result.returncode)
os.execv(sys.executable,[sys.executable,'/tmp/dsv41-110-preflight/run_python_16k_1024_defaults.py',*sys.argv[1:]])
