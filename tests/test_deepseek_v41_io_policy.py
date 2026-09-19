"""No-MLX regressions for bounded expert-bank I/O in both benchmark runners."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mtplx.models import deepseek_v41_loader as loader
from mtplx.expert_streaming_models import DEEPSEEK_V41_FLASH_EXPERT_MXFP4

ROOT = Path(__file__).resolve().parents[1]


def ab_module(name='ab_decode_env_levers'):
    spec = importlib.util.spec_from_file_location(
        'dsv41_io_ab', ROOT / f'scripts/deepseek_v41/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExpertBankIOPolicyTests(unittest.TestCase):
    def config(self, **overrides):
        # Ring resolution is independent of disk caching and imports the model.
        with patch.object(loader, 'resolve_gate_prefetch_ring_slots', return_value=0):
            return loader.build_streaming_config(
                DEEPSEEK_V41_FLASH_EXPERT_MXFP4,
                memory_limit_bytes=80 * 1024**3,
                max_live_kv_tokens=17408, **overrides)

    def test_ab_copies_served_profile_cache_bypass(self):
        ab = ab_module()
        args = ab.build_parser().parse_args(['--out', '/tmp/not-written.jsonl'])
        self.assertIs(ab._resolve_plan_overrides(args).get('bypass_page_cache'), True)

    def test_both_runners_preserve_explicit_buffered_profile(self):
        from mtplx import expert_profiles
        profile = SimpleNamespace(config={'bypass_page_cache':False})
        for name in ('ab_decode_env_levers', 'bench_standard_shape'):
            with self.subTest(runner=name), patch.object(
                expert_profiles, 'load_expert_profiles', return_value={'control':profile}
            ):
                args = SimpleNamespace(expert_profile='control', transient_slots=None)
                self.assertIs(ab_module(name)._resolve_plan_overrides(args).get(
                    'bypass_page_cache'), False)

    def test_shared_transient_pool_is_not_multiplied_by_layer_count(self):
        runtime = SimpleNamespace(
            config=SimpleNamespace(), plan=SimpleNamespace(transient_slots=48),
            spec=SimpleNamespace(expert_record_bytes=18800640, routed_layer_count=40))
        with patch.dict('os.environ', {'MTPLX_DSV41_GATE_PREFETCH':'0'}):
            report = ab_module()._resolved_plan(runtime, SimpleNamespace())
        self.assertEqual(report['transient_bytes_total'], 48 * 18800640)

    def test_macos_loader_bypasses_cache_without_profile(self):
        with patch.object(sys, 'platform', 'darwin'):
            self.assertTrue(self.config().bypass_page_cache)

    def test_other_platform_loader_keeps_supported_default(self):
        with patch.object(sys, 'platform', 'linux'):
            self.assertFalse(self.config().bypass_page_cache)

    def test_explicit_buffered_diagnostic_remains_explicit(self):
        with patch.object(sys, 'platform', 'darwin'):
            self.assertFalse(self.config(bypass_page_cache=False).bypass_page_cache)

    def test_receipt_uses_actual_reader_mode(self):
        runtime = SimpleNamespace(
            config=SimpleNamespace(bypass_page_cache=True),
            reader=SimpleNamespace(cache_mode='f-nocache'),
            plan=SimpleNamespace(), spec=SimpleNamespace())
        args = SimpleNamespace(expert_profile='none')
        for name in ('ab_decode_env_levers', 'bench_standard_shape'):
            with self.subTest(runner=name), patch.dict(
                'os.environ', {'MTPLX_DSV41_GATE_PREFETCH':'0'}
            ):
                report = ab_module(name)._resolved_plan(runtime, args)
                self.assertEqual(report.get('io_cache_mode'), 'f-nocache')


class DecodeCounterScopeTests(unittest.TestCase):
    def test_early_eos_uses_actual_ar_and_dspark_output(self):
        ab = ab_module()
        for generated in ([99], [1, 2, 99], [1, 2, 3, 4, 99]):
            with self.subTest(generated=generated):
                steps = len(generated) - 1
                run = {
                    'generated': generated,
                    'stream_after_prefill': {'expert_cache': {'bytes_read': 500}},
                    'stream_end': {'expert_cache': {'bytes_read': 500 + 120 * steps}},
                }
                report = ab._stream_counters_block(run, 1024, None)
                self.assertEqual(report['tokens'], steps)
                self.assertEqual(report['expert_cache']['bytes_read_per_token'],
                                 120 if steps else 0)

    def test_legacy_snapshot_without_generated_ids_keeps_explicit_count(self):
        run = {
            'stream_after_prefill': {'expert_cache': {'bytes_read': 500}},
            'stream_end': {'expert_cache': {'bytes_read': 860}},
        }
        report = ab_module()._stream_counters_block(run, 3, None)
        self.assertEqual(report['tokens'], 3)
        self.assertEqual(report['expert_cache']['bytes_read_per_token'], 120)


if __name__ == '__main__':
    unittest.main()
