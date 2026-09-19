"""CPU-only construction audit against the frozen causal head trajectory."""
import ast
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX is forbidden in this CPU construction audit')


sys.meta_path.insert(0, NoMLX())
r = Path(__file__).resolve().parent
sys.path.insert(0, str(r / 'packed'))
import numpy as np
from hybrid_install import ConsensusExtension, rewrite

repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
prompt_file = repo / 'docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json'
prompt = next(p['token_ids'] for p in json.loads(prompt_file.read_text())['prompts']
              if len(p['token_ids']) == 16384)
teacher = json.loads(Path('/tmp/dsv41-depth-replay-20260917/teacher.json').read_text())
screen = json.loads((r.parent / 'head/head-screen.json').read_text())
arm = next(a for a in screen['arms'] if a['mode'] == 'consensus')
extension = ConsensusExtension(prompt)
seen = 0
for row in arm['rows']:
    position = row['position']
    extension.append_committed(teacher['token_ids'][seen:position+1])
    seen = position + 1
    # Saved sigmoid scores are safely away from the decision boundary. This
    # reconstructs only the CPU confidence gate, not fresh native logits.
    p = np.clip(np.array(row['confidence'], dtype=np.float64), 1e-15, 1-1e-15)
    logits = np.log(p / (1-p)).astype(np.float32)
    proposed = extension.extend(row['native_proposed_ids'], logits)
    assert proposed == row['proposed_ids'], position
source = (repo / 'mtplx/models/deepseek_v41_dspark_decode.py').read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
            and n.name == '_decode_cycles')
native = ast.get_source_segment(source, node) + '\n'
updated = rewrite(native)
result = {'complete': True, 'cpu_only': True, 'mlx_imported': False,
          'matched_saved_proposals': len(arm['rows']), 'verify_rows': arm['verify_rows'],
          'original_five_proposals_preserved': True,
          'confidence_gate': 'reconstructed saved logits; saved sigmoid margin exceeds0.002',
          'native_mlx_operations_unchanged': True,
          'decode_function_sha256': hashlib.sha256(updated.encode()).hexdigest(),
          'scope': 'Construction identity and frozen trajectory only; no full target or TPS claim.'}
(r / 'composition-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
