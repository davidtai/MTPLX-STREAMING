"""Two host-resource regressions added after the successful exact full run.

Real MLX imports are blocked. Only the final array conversion is stubbed;
actual BF16/Metal equality is covered by the guarded operator and full receipt.
"""
import fcntl
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest

import numpy as np


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('real MLX is forbidden in host regressions')


sys.meta_path.insert(0, NoMLX())
core = types.ModuleType('mlx.core')
nn = types.ModuleType('mlx.nn')
mlx = types.ModuleType('mlx')
nn.Module = type('Module', (), {})
core.bfloat16 = object()


class Result:
    def __init__(self, values):
        self.values = values

    def view(self, dtype):
        assert dtype is core.bfloat16
        return self

    def reshape(self, shape):
        return Result(self.values.reshape(shape))


core.array = lambda values: Result(values)
mlx.core, mlx.nn = core, nn
sys.modules.update({'mlx':mlx, 'mlx.core':core, 'mlx.nn':nn})
spec = importlib.util.spec_from_file_location('tested_file_embedding', Path(__file__).with_name('file_embedding.py'))
embedding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedding)
# Force frequent eviction without changing the row geometry or source format.
embedding.ARENA_BYTES = 2 * 5120 * 2


class HostRowRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'rows.safetensors'
        tensor = {'dtype':'BF16', 'shape':[129280,5120], 'data_offsets':[0,1323827200]}
        header = json.dumps({'embed.weight':tensor}).encode()
        self.offset = 8 + len(header)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.write(fd, struct.pack('<Q', len(header)) + header)
            os.ftruncate(fd, self.offset + 1323827200)
            self.rows = {token: (np.arange(5120,dtype=np.uint16) ^ (token & 65535))
                         for token in (0, 7, 77, 128799, 129279)}
            for token, row in self.rows.items():
                os.pwrite(fd, row.tobytes(), self.offset + token * 10240)
        finally:
            os.close(fd)
        st = self.path.stat()
        self.source = {'path':str(self.path), 'tensor':'embed.weight', 'shape':[129280,5120],
                       'tensor_header':tensor, 'offset':self.offset, 'nbytes':1323827200,
                       'identity':{key:getattr(st,key) for key in ('st_dev','st_ino','st_size','st_mtime_ns')}}
        self.cache = None

    def tearDown(self):
        if self.cache is not None:
            self.cache.close()
        self.tmp.cleanup()

    def test_duplicates_eviction_and_returned_owner(self):
        self.cache = embedding.FileRowEmbedding(self.source)
        ids = np.array([[129279, 7, 129279, 0, 7, 128799]], dtype=np.int32)
        held = self.cache(ids)
        expected = np.stack([self.rows[int(i)] for i in ids.flat]).reshape(1,6,5120)
        np.testing.assert_array_equal(held.values, expected)
        for token in (77, 0, 129279, 7, 128799, 77):
            fresh = self.cache(np.array([[token]], dtype=np.int32))
            np.testing.assert_array_equal(fresh.values, self.rows[token].reshape(1,1,5120))
            self.assertLessEqual(len(self.cache._lru), 2)
            self.assertEqual(len(self.cache._arena), 20480)
        np.testing.assert_array_equal(held.values, expected)
        fd = self.cache._fd
        self.cache.close()
        self.cache.close()
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertIsNone(self.cache._arena)
        self.assertEqual(len(self.cache._lru), 0)

    def test_short_read_cannot_publish_corrupt_row(self):
        self.cache = embedding.FileRowEmbedding(self.source)
        self.cache(np.array([[0, 7]], dtype=np.int32))
        # Model sources are immutable in normal use; an unexpected truncate
        # must fail without publishing partial bytes as a successful cache hit.
        os.truncate(self.path, self.offset + 8 * 10240)
        with self.assertRaisesRegex(RuntimeError, 'short native embedding row'):
            self.cache(np.array([[129279]], dtype=np.int32))
        self.assertNotIn(129279, self.cache._lru)
        np.testing.assert_array_equal(self.cache(np.array([[7]], dtype=np.int32)).values,
                                      self.rows[7].reshape(1,1,5120))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HostRowRegressions))
    Path(__file__).with_name('regressions-cpu.json').write_text(json.dumps({
        'tests_run':result.testsRun, 'failures':len(result.failures), 'errors':len(result.errors),
        'successful':result.wasSuccessful(), 'real_mlx_imports_blocked':True,
        'scope':'Host arena, duplicate ordering, eviction owners, close and short-read failure. Added after exact full candidate.'}, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
