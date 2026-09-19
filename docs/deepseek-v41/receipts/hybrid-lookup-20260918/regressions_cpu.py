"""Focused causal-lookup regressions, added after the exact full candidate."""
import copy
import importlib.abc
import json
from pathlib import Path
import sys
import unittest


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('real MLX imports forbidden in host regressions')


sys.meta_path.insert(0, NoMLX())
from lookup import LookupExtension


class CausalLookupRegressions(unittest.TestCase):
    def test_complete_prefix_context_order_and_readonly_query(self):
        draft = [10,11,12,13,14]
        prompt = ([101,102,7,8]+draft+[15,16]
                  +[201,202,7,8]+draft+[21,22]+[301,302,7])
        lookup = LookupExtension(prompt)
        lookup.append_committed([8])
        before = copy.deepcopy(lookup.__dict__)
        self.assertEqual(lookup.extend(draft), draft+[15,16])
        self.assertEqual(lookup.extend(draft[:-1]+[999]),draft[:-1]+[999])
        self.assertEqual(lookup.__dict__,before)
        # A longer real context takes precedence over an earlier shorter one.
        lookup = LookupExtension([9,9,7,8]+draft+[15,16]+[3,4,7,8]+draft+[21,22]+[1,2,3,4,7])
        lookup.append_committed([8])
        self.assertEqual(lookup.extend(draft),draft+[21,22])

    def test_only_committed_continuations_become_available(self):
        draft = [10,11,12,13,14]
        lookup = LookupExtension([91,92]+draft,minimum_context=0)
        lookup.append_committed([])
        self.assertEqual(lookup.extend(draft),draft)
        lookup.append_committed([15])
        self.assertEqual(lookup.extend(draft),draft+[15])
        lookup.append_committed([16,17,18])
        self.assertEqual(lookup.extend(draft),draft+[15,16])
        joined = LookupExtension([91,92]+draft,minimum_context=0)
        joined.append_committed([15,16,17,18])
        self.assertEqual(lookup.history,joined.history)
        self.assertEqual(lookup.ends,joined.ends)


if __name__ == '__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CausalLookupRegressions))
    Path(__file__).with_name('regressions-cpu.json').write_text(json.dumps({
        'tests_run':result.testsRun,'successful':result.wasSuccessful(),
        'failures':len(result.failures),'errors':len(result.errors),
        'real_mlx_imports_blocked':True,'added_after_full_candidate':True},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
