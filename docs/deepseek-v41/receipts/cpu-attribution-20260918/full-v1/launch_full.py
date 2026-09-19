"""Select an isolated host library only inside the already-held GPU window."""
import hashlib,json,os,sys
from pathlib import Path
if os.environ.get('_GPU_WINDOW_LOCKED')!='1':raise RuntimeError('parent GPU guard required')
root=Path(__file__).resolve().parent
proof=json.loads((root/'packed/installation.json').read_text())
identity=proof['strict_allocator']['identity']
if hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest()!=identity['sha256']:raise RuntimeError('strict library changed')
env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(Path(identity['path']).parent)
os.execve(sys.executable,[sys.executable,str(root/'packed/run_full.py'),*sys.argv[1:]],env)
