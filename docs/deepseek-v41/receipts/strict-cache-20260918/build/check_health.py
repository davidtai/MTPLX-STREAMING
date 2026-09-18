"""Read-only restoration check after the owning guard is terminal."""
import fcntl,json,sys,time,urllib.request
from pathlib import Path
with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=8) as f: h=json.load(f)
with urllib.request.urlopen('http://127.0.0.1:8080/v1/models',timeout=8) as f: m=json.load(f)
with open('/tmp/mtplx-gpu-exclusive.lock','a') as f:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(f,fcntl.LOCK_UN)
assert h['ok'] and h['scheduler']['active_requests']==0
assert h['startup']['warmup']['background']['state']=='done'
assert [d['id'] for d in m['data']]==['mtplx-flash-next-optimized-speed']
result={'checked_at':time.time(),'health':h,'models':m,'lock_free':True,'guard_session':int(sys.argv[2]),'guard_exit_code':0}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'ok':True,'active_requests':0,'warmup':'done','lock_free':True,'checked_at':result['checked_at']}))
