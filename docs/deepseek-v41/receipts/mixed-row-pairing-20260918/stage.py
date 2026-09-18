"""Preserve native geometry; stage one mixed-row hit dispatch at C109."""
from pathlib import Path
import hashlib,json,shutil,subprocess

r=Path(__file__).resolve().parent
b=Path('/tmp/dsv41-packed-row-order-20260918')
old=Path('/tmp/dsv41-row-pairing-20260917')
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
for name in ('packed_storage.py','paired_kernels.py','kernels.py','restore_bank.py','library_identity.py','run_screen.py','routes.json','plane_lane.py'):
    shutil.copyfile(b/name,r/name)
(r/'artifact').symlink_to(b/'artifact',target_is_directory=True)
s=(old/'row_pair_kernels.py').read_text()
s=s.replace('Two input rows share FP4 conversion; each row keeps native dot order.',
            'Pairs and singleton rows share one grid; native dot order is unchanged.')
s=s.replace('pairs[2*assignment]','pairs[4*assignment]').replace('pairs[2*assignment+1]','pairs[4*assignment+1]')
s=s.replace('const device T* xp=x+(2*assignment)*K+lane*V;', '''const uint row0=pairs[4*assignment+2];
const int row1=pairs[4*assignment+3];
const device T* xp=x+row0*K+lane*V;
const device T* xp1=x+uint(max(row1,0))*K+lane*V;''')
s=s.replace('x1[i]=float(xp[K+block*STEP+i]);','x1[i]=row1>=0 ? float(xp1[block*STEP+i]) : 0.0f;')
s=s.replace('out[(2*assignment)*N+outrow+r]=T(a);','out[row0*N+outrow+r]=T(a);')
s=s.replace('out[(2*assignment+1)*N+outrow+r]=T(b);','if(row1>=0) out[uint(row1)*N+outrow+r]=T(b);')
s=s.replace("name=f'dsv41_row_pair_{n}_{k}'","name=f'dsv41_mixed_row_pair_{n}_{k}'")
assert '2*assignment' not in s
(r/'mixed_row_kernels.py').write_text(s)
s=(r/'plane_lane.py').read_text()
s=s.replace('        self.completions = self.early_completions if early else self.complete_completions',
'''        self.completions = self.early_completions if early else self.complete_completions
        from mixed_hit_ops import MixedHitOps
        self.hit_ops_by_m = {m: ops for m in range(1,9)}
        self.hit_ops_by_m[6] = MixedHitOps(ops)''')
s=s.replace('        def finish(groups):\n            wave = [ops.down(g) for g in groups]',
            '        def finish(groups, down=ops.down):\n            wave = [down(g) for g in groups]')
s=s.replace('                finish(ops.gate_up(tokens,experts,buffers))',
'''                hit_ops = self.hit_ops_by_m[int(tokens.shape[0])]
                finish(hit_ops.gate_up(tokens,experts,buffers),hit_ops.down)''')
(r/'plane_lane_mixed.py').write_text(s)
s=(b/'probe.py').read_text().replace('plane_lane_ordered','plane_lane_mixed').replace("'ordered'","'mixed'")
(r/'probe.py').write_text(s)
p=json.loads((b/'installation.json').read_text());p['helper_sha256']={}
p['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
p['scope']='Layer34, all206 native M6 routes, actual73 warm residents expanded to109,48 shared transients. Mixed pair/single cache-hit kernel preserves native V16/V8,R4,two-SIMD geometry and float2 per-row dot order. One dispatch per bank; miss parts use unchanged native kernels.'
p['arms']=['native','mixed','native','mixed','native']
p['row_pair_predecessor_sha256']={name:hashlib.sha256((old/name).read_bytes()).hexdigest() for name in ('row_pair_kernels.py','probe.json','integration/integration-probe.json')}
for name,digest in p['runtime_source_sha256'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
(r/'installation.json').write_text(json.dumps(p,indent=2)+'\n')
(r/'command.sh').write_text((b/'command.sh').read_text().replace(str(b.resolve()),str(r)).replace(str(b),str(r)))
for path in r.glob('*.py'):compile(path.read_text(),str(path),'exec')
p['helper_sha256']={name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in
    ('probe.py','mixed_hit_ops.py','mixed_row_kernels.py','plane_lane.py','plane_lane_mixed.py','packed_storage.py','paired_kernels.py','kernels.py','restore_bank.py','library_identity.py','run_screen.py','routes.json')}
(r/'installation.json').write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'root':str(r),'source':p['source_commit'],'incremental_bound_bytes':p['static_incremental_bound_bytes'],'bank_capacity':p['bank_capacity']}))
