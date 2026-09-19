"""Stage earlier prefetch with demand-first I/O; no MLX imports."""
from pathlib import Path
import ast, hashlib, json, shutil, subprocess

r=Path(__file__).resolve().parent
b=Path('/tmp/dsv41-lookahead-adjacent-20260918')
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
for name in ('probe.py','predictor_cost.py','paired_config.py','packed_storage.py','paired_kernels.py','kernels.py','restore_bank.py','library_identity.py','run_screen.py','routes.json','preflight.py'):
    shutil.copyfile(b/name,r/name)
(r/'artifact').symlink_to(b/'artifact',target_is_directory=True)
s=(b/'plane_lane.py').read_text()
s=s.replace('from paired_kernels import make_projection','from paired_kernels import make_projection\nfrom priority_reads import PriorityReads')
s=s.replace('class PlanePart:\n','''class PlanePart:
    read_priority = 0

    @staticmethod
    def read_first(read,job,submit):
        read(job)

''')
s=s.replace('class IgnoreGUPublication:\n','''class IgnoreGUPublication:
    read_priority = 1

    @staticmethod
    def read_first(read,job,submit):
        submit(read,job,priority=1).result()

''')
begin=s.index('def bind_reader(');end=s.index('\n\n@dataclass',begin)
original=s[begin:end]
priority=original.replace('def bind_reader(','def bind_priority_reader(')
priority=priority.replace('submit(read, job)','submit(read, job, priority=part.read_priority)')
priority=priority.replace('                read(jobs[0])','                part.read_first(read,jobs[0],submit)')
s=s[:end]+'\n\n'+priority+s[end:]
s=s.replace('        remaining = len(parts)','        issue_pending = True')
s=s.replace('            nonlocal remaining','            nonlocal issue_pending')
s=s.replace('''            remaining -= 1
            if remaining == 0:
                # Every current demand part has enqueued its down jobs before
                # publishing GU readiness. Speculative jobs follow those jobs.
                self.issue()''','''            if issue_pending:
                issue_pending = False
                # The first GU witness creates earlier lead. All later demand
                # jobs outrank queued speculation; active reads still finish.
                self.issue()''')
old='    bind_reader(runtime.reader,local)'
new='''    if prefetch_sources is None:
        bind_reader(runtime.reader,local)
    else:
        runtime.reader._fanout_executor.shutdown(wait=True)
        runtime.reader._fanout_executor = PriorityReads()
        bind_priority_reader(runtime.reader,local)'''
assert s.count(old)==1 and 'remaining' not in s
s=s.replace(old,new)
(r/'plane_lane.py').write_text(s)
p=json.loads((b/'installation.json').read_text())
p['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
p['scope']+=' Candidate issues after first GU and gives demand jobs queue priority; unchanged native controls keep the original executor and issue no speculation.'
p['helper_sha256']={}
(r/'installation.json').write_text(json.dumps(p,indent=2)+'\n')
pre=(r/'preflight.py').read_text().replace("'probe.py','predictor_cost.py'","'probe.py','priority_reads.py','predictor_cost.py'")
(r/'preflight.py').write_text(pre)
(r/'command.sh').write_text((b/'command.sh').read_text().replace(str(b.resolve()),str(r)))
for path in r.glob('*.py'):compile(path.read_text(),str(path),'exec')
print(json.dumps({'root':str(r),'source':p['source_commit'],'incremental_bound_bytes':p['static_incremental_bound_bytes']}))
