"""Stage a CPU-only alignment comparison from the native reader screen."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parent
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base=repo/'docs/deepseek-v41/receipts/gu-combined-read-20260917/screen'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((base/'installation.json').read_text())
for name in ('native_reader.py','probe.py','run_screen.py'):
    assert sha(base/name)==proof['helper_sha256'][name],name
    shutil.copyfile(base/name,root/name)
p=root/'probe.py';s=p.read_text()
s=s.replace('compare the installed plane reader with GU scatter','compare source-matched and page-aligned destination offsets')
s=s.replace('import gc','import ctypes\nimport mmap\nimport gc',1)
s=s.replace("combined = load_helper('combined_reader')",'PAGE = os.sysconf("SC_PAGESIZE")\nif PAGE != 16384: raise RuntimeError("expected native16KiB page geometry")')
a=s.index('class Slot:');b=s.index('\n\nslots = ',a)
s=s[:a]+'''class Slot:
    def __init__(self):
        self.arrays = {name: mmap.mmap(-1,WEIGHT+PAGE) for name in NAMES}
        self.offsets = {name:0 for name in NAMES}
        self.addresses = {name:ctypes.addressof(ctypes.c_ubyte.from_buffer(array))
                          for name,array in self.arrays.items()}
        if any(address%PAGE for address in self.addresses.values()):
            raise RuntimeError('CPU control buffer is not page-aligned')

    def configure(self,record,matched):
        for name,offset in zip(NAMES,(0,6266880,12533760)):
            self.offsets[name]=(record.sidecar_offset+offset)%PAGE if matched else 0

    def component_view(self,name):
        offset=self.offsets[name]
        return memoryview(self.arrays[name])[offset:offset+WEIGHT]

    def digest(self):
        h=hashlib.sha256()
        for name in NAMES:
            with self.component_view(name) as view:h.update(view)
        return h.hexdigest()

    def close(self):
        for array in self.arrays.values():array.close()
'''+s[b:]
s=s.replace('range(6))','range(3))',1)
s=s.replace("'scope': 'CPU read-batch screen with native reader/fanout and gate/up readiness. No Metal owners, target model, cache pressure or decode TPS claim.'",
 "'scope': 'CPU native plane reader: page-aligned mmap destinations versus offsets matching source modulo16KiB. Same three plane reads and fanout. No MLX imports, Metal owners, model or full TPS claim; actual Metal base alignment is not measured by this screen.'")
s=s.replace("'weight_buffer_bytes': 6 * 3 * WEIGHT, 'scratch_buffer_bytes': 6 * GAP,", "'weight_buffer_bytes': 3 * 3 * WEIGHT, 'alignment_padding_bytes': 3 * 3 * PAGE,")
s=s.replace("'candidate_extra_read_bytes_per_record': GAP", "'candidate_extra_read_bytes_per_record': 0")
s=s.replace('for size in (1, 3, 6):','for size in (3,):')
s=s.replace("for arm, helper in (('control_before', native), ('combined', combined), ('control_after', native)):",
 "for arm, helper in (('page_before',native),('source_matched1',native),('page_middle',native),('source_matched2',native),('page_after',native)):")
s=s.replace('                def read_batch(batch):\n',"                def read_batch(batch):\n                    for slot,record in zip(slots,batch):slot.configure(record,arm.startswith('source_matched'))\n")
s=s.replace("per_record = 3 * WEIGHT + (GAP if arm == 'combined' else 0)","per_record = 3 * WEIGHT")
s=s.replace("calls = 2 if arm == 'combined' else 3",'calls = 3')
s=s.replace("'fanout': 4, 'worker_capacity': reader._fanout_pool_workers,", "'fanout': 4, 'worker_capacity': reader._fanout_pool_workers,\n                       'last_buffer_offsets': [slot.offsets.copy() for slot in slots[:size]],\n                       'source_mod_page_counts': {str(mod):sum((record.sidecar_offset+offset)%PAGE==mod for batch in batches for record in batch for offset in (0,6266880,12533760)) for mod in (0,8192)},")
s=s.replace("finally:\n    del slots", "finally:\n    for slot in slots:slot.close()\n    del slots")
p.write_text(s)
proof['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
proof['scope']='CPU-only matched positional read alignment, native3-plane format, fanout4,128batches of3 records, five interleaved arms.2GiB complete envelope; 53,231,616 mmap destination bytes and18,800,640 native validation bytes; no model tensor load.'
old=proof['runtime_source_sha256']
proof['runtime_source_sha256']={name:sha(repo/name) for name in old}
proof['runtime_refresh']={name:{'old':old[name],'current':h} for name,h in proof['runtime_source_sha256'].items() if old[name]!=h}
proof['helper_sha256']={p.name:sha(p) for p in root.glob('*.py') if p.name!='stage.py'}
(root/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
cmd=(base/'command.sh').read_text().replace('/tmp/dsv41-gu-scatter-20260917/v2',str(root)).replace('GPU_WINDOW_MIN_AVAIL_GB=100','GPU_WINDOW_MIN_AVAIL_GB=10')
(root/'command.sh').write_text(cmd)
for p in root.glob('*.py'):ast.parse(p.read_text())
print(json.dumps({'source':proof['source_commit'],'bound_bytes':proof['incremental_allowance_bytes'],'runtime_refresh':proof['runtime_refresh']},indent=2))
