"""One bounded D5/M6 diagnostic using CPU-validated explicit timing scopes."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('the parent GPU guard must hold the lock before MLX import')

ROOT = Path('/tmp/dsv41-explicit-profile-20260917')
GROWTH_ROOT = Path('/tmp/dsv41-explicit-profile-20260917/growth84')
RESERVE = 128 * 1024**2
proof = json.loads((ROOT/'cpu-proof.json').read_text())
if proof['timer_sha256'] != hashlib.sha256((ROOT/'boundary_timer.py').read_bytes()).hexdigest():
    raise RuntimeError('CPU timing proof no longer matches')
if not (proof['foreign_thread_excluded'] and proof['iterator_consumer_time_excluded']
        and proof['root_wall_ns'] == proof['exclusive_sum_ns']
        and proof['overhead_ns_per_flat_call'] < 10000):
    raise RuntimeError('explicit boundary profiler failed CPU admission')
if (ROOT/'decode-boundaries.json').exists():
    raise RuntimeError('refusing to overwrite existing profile evidence')

sys.path.insert(0, str(GROWTH_ROOT))
import admission
original_admission = admission.resolve_admission


def profile_admission(base, wired, **kwargs):
    result = original_admission(base+RESERVE, wired+RESERVE, **kwargs)
    result['admission_baseline_with_profiler_reserve_bytes'] = result['baseline_bytes']
    result['baseline_bytes'] = base
    result['wired_before_bytes'] = wired
    result['profiler_host_reserve_bytes'] = RESERVE
    result['host_reserve_bytes'] += RESERVE
    # The runner's allocator policy is derived from the actual baseline and
    # 2GiB host reserve. Extra diagnostic storage is separately priced by the
    # stricter static physical bound and smaller selected expert capacity.
    result['allocator_limit_bytes'] += RESERVE
    result['allocator_policy_host_reserve_bytes'] = 2 * 1024**3
    return result


admission.resolve_admission = profile_admission

import mlx.core as mx
from boundary_timer import BoundaryTimer
from mtplx import expert_runtime as runtime
from mtplx.models import expert_mlx as adapter
from mtplx.models import deepseek_v41 as model
from mtplx.models import deepseek_v41_moe as moe
from mtplx.models import deepseek_v41_dspark as dspark
from mtplx.models import deepseek_v41_dspark_decode as decode

original_cycles = decode._decode_cycles
original_cycles_sha256 = hashlib.sha256(inspect.getsource(original_cycles).encode()).hexdigest()


def timed_cycles(**kwargs):
    timer = BoundaryTimer()
    patches = []

    def bind(owner, attribute, label, iterator=False):
        original = getattr(owner, attribute)
        wrap = timer.wrap_iterator if iterator else timer.wrap
        patches.append((owner, attribute, original))
        setattr(owner, attribute, wrap(label, original))

    bind(mx, 'eval', 'mlx_eval_wait_and_encode')
    bind(mx, 'async_eval', 'mlx_async_submit')
    bind(adapter.HotExpertSwitchGLU, '_run', 'streamed_switch')
    bind(adapter.HotExpertSwitchGLU, '_dispatch_component_bank', 'expert_graph_build')
    bind(adapter.HotExpertSwitchGLU, '_submit_verify_shared_overlap', 'shared_graph_and_submit')
    bind(runtime.ExpertStreamingRuntime, 'begin_split_route', 'route_plan_and_read_submit')
    bind(runtime.ExpertStreamingRuntime, 'try_all_hit_route', 'all_hit_probe')
    bind(runtime.ExpertStreamingRuntime, '_plan_route_transaction', 'cache_policy_transaction')
    bind(runtime.ExpertStreamingRuntime, 'flush_deferred_slot_releases', 'retire_prior_gathers')
    bind(runtime.PendingSplitRoute, 'iter_ready_misses', 'miss_wait_and_completion', iterator=True)
    bind(model.Attention, '__call__', 'attention_graph_build')
    bind(moe.Gate, '__call__', 'gate_graph_build')
    bind(moe.MoE, '__call__', 'moe_graph_build')
    bind(dspark.DSparkHead, 'draft_block', 'draft_graph_build')
    kwargs['forward'] = timer.wrap('target_forward', kwargs['forward'])
    print('BOUNDARY_PROFILE_START', flush=True)
    try:
        return timer.call('decode_cycles', original_cycles, **kwargs)
    finally:
        for owner, attribute, original in reversed(patches):
            setattr(owner, attribute, original)
        report = timer.snapshot()
        report.update(
            scope='Diagnostic main-thread wall time at explicit call boundaries; GPU encode and wait stay combined; miss-next includes completion bookkeeping; no inference speedup claim',
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            timer_sha256=proof['timer_sha256'], cpu_proof=proof,
            decode_cycles_source_sha256=original_cycles_sha256,
            growth_wrapper_sha256=hashlib.sha256((GROWTH_ROOT/'run_full.py').read_bytes()).hexdigest(),
            profiler_host_reserve_bytes=RESERVE,
        )
        (ROOT/'decode-boundaries.json').write_text(json.dumps(report,indent=2)+'\n')
        print('BOUNDARY_PROFILE_FINISHED', json.dumps({k:report[k] for k in (
            'root_wall_ns','exclusive_sum_ns')}), flush=True)


decode._decode_cycles = timed_cycles
runpy.run_path(str(GROWTH_ROOT/'run_full.py'), run_name='__main__')
