# Copyright © 2023-2024 Apple Inc.

import gc
import unittest

import mlx.core as mx
import mlx_tests


class TestMemory(mlx_tests.MLXTestCase):
    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_cache_limit_releases_oversized_free(self):
        limit = 8 * 1024**2
        previous = mx.set_cache_limit(limit)
        self.addCleanup(mx.set_cache_limit, previous)
        self.addCleanup(mx.clear_cache)
        mx.clear_cache()
        value = mx.zeros((2 * limit,), dtype=mx.uint8)
        mx.eval(value)
        del value
        gc.collect()
        mx.synchronize()
        self.assertLessEqual(mx.get_cache_memory(), limit)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_cache_limit_bounds_multiple_frees(self):
        limit = 8 * 1024**2
        previous = mx.set_cache_limit(limit)
        self.addCleanup(mx.set_cache_limit, previous)
        self.addCleanup(mx.clear_cache)
        mx.clear_cache()
        first = mx.zeros((6 * 1024**2,), dtype=mx.uint8)
        second = mx.zeros((4 * 1024**2,), dtype=mx.uint8)
        mx.eval(first, second)
        mx.synchronize()
        mx.clear_cache()
        del first
        gc.collect()
        mx.synchronize()
        self.assertGreaterEqual(mx.get_cache_memory(), 6 * 1024**2)
        del second
        gc.collect()
        mx.synchronize()
        self.assertLessEqual(mx.get_cache_memory(), limit)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_lower_cache_limit_trims_immediately(self):
        previous = mx.set_cache_limit(32 * 1024**2)
        self.addCleanup(mx.set_cache_limit, previous)
        self.addCleanup(mx.clear_cache)
        mx.clear_cache()
        value = mx.zeros((16 * 1024**2,), dtype=mx.uint8)
        mx.eval(value)
        del value
        gc.collect()
        mx.synchronize()
        self.assertGreaterEqual(mx.get_cache_memory(), 16 * 1024**2)
        self.assertEqual(mx.set_cache_limit(8 * 1024**2), 32 * 1024**2)
        self.assertLessEqual(mx.get_cache_memory(), 8 * 1024**2)

    def test_memory_info(self):
        old_limit = mx.set_cache_limit(0)

        a = mx.zeros((4096,))
        mx.eval(a)
        del a
        self.assertEqual(mx.get_cache_memory(), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), old_limit)

        old_limit = mx.set_memory_limit(10)
        self.assertEqual(mx.set_memory_limit(old_limit), 10)
        self.assertEqual(mx.set_memory_limit(old_limit), old_limit)

        # Query active and peak memory
        a = mx.zeros((4096,))
        mx.eval(a)
        mx.synchronize()
        active_mem = mx.get_active_memory()
        self.assertTrue(active_mem >= 4096 * 4)

        b = mx.zeros((4096,))
        mx.eval(b)
        del b
        mx.synchronize()

        new_active_mem = mx.get_active_memory()
        self.assertEqual(new_active_mem, active_mem)
        peak_mem = mx.get_peak_memory()
        self.assertTrue(peak_mem >= 4096 * 8)

        if mx.metal.is_available():
            cache_mem = mx.get_cache_memory()
            self.assertTrue(cache_mem >= 4096 * 4)

        mx.clear_cache()
        self.assertEqual(mx.get_cache_memory(), 0)

        mx.reset_peak_memory()
        self.assertEqual(mx.get_peak_memory(), 0)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_wired_memory(self):
        old_limit = mx.set_wired_limit(1000)
        old_limit = mx.set_wired_limit(0)
        self.assertEqual(old_limit, 1000)

        max_size = mx.device_info(mx.gpu)["max_recommended_working_set_size"]
        with self.assertRaises(ValueError):
            mx.set_wired_limit(max_size + 10)

    def test_active_memory_count(self):
        mx.synchronize()
        mx.clear_cache()
        init_mem = mx.get_active_memory()
        a = mx.zeros((128, 128))
        mx.eval(a)
        mx.synchronize()
        del a
        a = mx.zeros((90, 128))
        mx.eval(a)
        mx.synchronize()
        del a
        self.assertEqual(init_mem, mx.get_active_memory())


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
