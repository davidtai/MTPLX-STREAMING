"""CPU-only shutdown-cache regressions; no MLX or real service commands."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/deepseek_v41/reclaim_file_cache.py'

class FileCacheTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('cache_reclaim',SCRIPT)
        self.mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.mod)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'config.json').write_text('{}')
        (self.root/'model.safetensors.index.json').write_text('{}')
        self.file=self.root/'model-00001.safetensors'
        self.file.write_bytes(os.urandom(128*1024+7))

    @unittest.skipUnless(sys.platform=='darwin','Darwin cache API')
    def test_readonly_invalidation_releases_cache_without_changing_file(self):
        before=hashlib.sha256(self.file.read_bytes()).hexdigest()
        identity=self.file.stat()
        result=self.mod.reclaim_file(self.file,chunk_bytes=64*1024)
        self.assertGreater(result['cached_page_bytes_before'],0)
        self.assertEqual(result['cached_page_bytes_after'],0)
        self.assertEqual(hashlib.sha256(self.file.read_bytes()).hexdigest(),before)
        self.assertEqual(identity.st_mtime_ns,self.file.stat().st_mtime_ns)
        self.assertEqual(identity.st_ino,self.file.stat().st_ino)

    def test_rejects_all_symlinks_before_any_reclamation(self):
        (self.root/'z.safetensors').symlink_to(self.file)
        with mock.patch.object(self.mod,'reclaim_file') as reclaim:
            with self.assertRaises(ValueError):self.mod.reclaim_model(self.root)
            reclaim.assert_not_called()

    def test_rejects_non_model_directory(self):
        (self.root/'config.json').unlink()
        with self.assertRaises(ValueError):self.mod.model_files(self.root)

    def test_rejects_excessive_file_count(self):
        with self.assertRaises(ValueError):self.mod.model_files(self.root,max_files=0)

    def test_selects_only_flat_safetensors(self):
        (self.root/'README.md').write_text('leave alone')
        nested=self.root/'other';nested.mkdir();(nested/'ignored.safetensors').write_bytes(b'x')
        self.assertEqual(self.mod.model_files(self.root),[self.file.resolve()])

    @unittest.skipUnless(sys.platform=='darwin','Darwin cache API')
    def test_failed_invalidation_closes_descriptor_and_mapping(self):
        original=self.mod.os.open;fds=[]
        def opened(*a,**kw):
            fd=original(*a,**kw);fds.append(fd);return fd
        libc=self.mod._libc()
        with mock.patch.object(self.mod.os,'open',side_effect=opened), mock.patch.object(self.mod,'_libc',return_value=libc), mock.patch.object(libc,'msync',return_value=-1), mock.patch.object(libc,'munmap',wraps=libc.munmap) as unmap:
            with self.assertRaises(OSError):self.mod.reclaim_file(self.file)
        unmap.assert_called_once()
        self.assertEqual(len(fds),1)
        with self.assertRaises(OSError):os.fstat(fds[0])

if __name__=='__main__':unittest.main()
