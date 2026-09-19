"""Two CPU regressions added after the complete optimization win."""
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("MLX forbidden in admission regressions")

sys.meta_path.insert(0, NoMLX())
sys.path.insert(0, "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41")
os.environ["MTPLX_ENGRAM_CACHE_LIMIT"] = "67108864"
root = Path("/tmp/dsv41-extension-bank-20260919/full-v1")
spec = importlib.util.spec_from_file_location("extension_admission", root / "packed/packed_admission.py")
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)
installation = json.loads((root / "packed/installation.json").read_text())
compat = json.loads((root / "compat/installation.json").read_text())


def resolve(baseline, wired=3501899776):
    return admission.resolve_admission(baseline, wired, grow=True,
        expected_receipt_hash=compat["phase_memory_control_sha256"],
        strict_allocator=installation["strict_allocator"]["identity"])


class AdmissionRegressions(unittest.TestCase):
    def test_exact_decimal_ceiling_selects_capacity_at_one_byte_boundary(self):
        reference = resolve(10983129088)
        self.assertEqual(reference["decode_slots_per_layer"], 111)
        limit_base = reference["baseline_bytes"] + 110000000000 - reference["physical_bound_bytes"]
        exact = resolve(limit_base)
        next_byte = resolve(limit_base + 1)
        self.assertEqual(exact["physical_bound_bytes"], 110000000000)
        self.assertEqual(exact["decode_slots_per_layer"], 111)
        self.assertEqual(next_byte["decode_slots_per_layer"], 110)
        self.assertLessEqual(next_byte["physical_bound_bytes"], 110000000000)
        for row in (exact, next_byte):
            self.assertEqual(row["initial_decode_slots_per_layer"], 84)
            self.assertEqual(row["resize_active_bound_bytes"], 82058610908)
            self.assertGreaterEqual(row["active_bound_bytes"], row["overflow_append_active_bound_bytes"])
            self.assertLessEqual(row["active_bound_bytes"] + row["decode_cache_allowance_bytes"], row["allocator_limit_bytes"])

    def test_wired_pressure_refuses_even_when_machine_baseline_fits(self):
        with self.assertRaisesRegex(RuntimeError, "cap84 prefill cannot fit"):
            resolve(10983129088, wired=40 * 1024**3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
