"""Diagnostic-only capture of exact target hidden rows and initialized MTP KV."""
import hashlib
import json
from pathlib import Path
import numpy as np
import mlx.core as mx

def host_bits(value):
    if value.dtype == mx.bfloat16:
        return np.asarray(value.view(mx.uint16)), 'bfloat16'
    if value.dtype == mx.float32:
        return np.asarray(value), 'float32'
    raise RuntimeError(f'unsupported native teacher dtype: {value.dtype}')

def validate_storage_bridge():
    for dtype in (mx.bfloat16, mx.float32):
        original=mx.array([1.0,-2.5],dtype=dtype)
        mx.eval(original)
        bits,name=host_bits(original)
        stored=np.array(bits,copy=True)
        restored=mx.array(stored)
        if name=='bfloat16': restored=restored.view(mx.bfloat16)
        mx.eval(restored)
        if restored.dtype!=original.dtype or restored.astype(mx.float32).tolist()!=[1.0,-2.5]:
            raise RuntimeError('native teacher storage bridge is not exact')
    print('TEACHER_STORAGE_BRIDGE exact BF16 bits and FP32 values',flush=True)

class TeacherCapture:
    def __init__(self, model, root, *, steps):
        self.root=Path(root)
        self.steps=int(steps)
        self.width=15360
        self.rows=None
        self.row_mlx_dtype=None
        self.count=0
        self.commit_lengths=[]
        self.seed_calls=0
        self.initial_main=None
        self.initial_windows=[]
        self.initial_offsets=[]
        self.initial_main_dtype=None
        self.initial_window_dtypes=[]
        args=model.args
        if (args.hidden_size!=5120 or tuple(args.dspark_target_layer_ids)!=(37,38,39)
            or model.mtp.block_size!=5 or len(model.mtp.layers)!=3):
            raise RuntimeError('teacher geometry differs from native D5/M6 control')
        original=model.mtp.seed_main
        def capture_seed(main_hidden,caches):
            result=original(main_hidden,caches)
            if self.seed_calls:
                values,native_dtype=host_bits(main_hidden)
                n=int(values.shape[1])
                if values.shape!=(1,n,self.width) or not 1<=n<=6:
                    raise RuntimeError('committed hidden capture geometry differs')
                if self.rows is None:
                    self.rows=np.empty((self.steps+6,self.width),dtype=values.dtype)
                    self.row_mlx_dtype=native_dtype
                if native_dtype!=self.row_mlx_dtype or values.dtype!=self.rows.dtype:
                    raise RuntimeError('committed teacher dtype changed between cycles')
                if self.count+n>len(self.rows):
                    raise RuntimeError('teacher capture exceeded its fixed allocation')
                np.copyto(self.rows[self.count:self.count+n],values[0],casting='no')
                self.count+=n
                self.commit_lengths.append(n)
            self.seed_calls+=1
            return result
        object.__setattr__(model.mtp,'seed_main',capture_seed)

    def capture_prefill(self, main_h, caches):
        if self.initial_main is not None or self.seed_calls!=1:
            raise RuntimeError('teacher prefill boundary repeated or missing')
        values,self.initial_main_dtype=host_bits(main_h)
        self.initial_main=np.array(values,copy=True)
        for cache in caches:
            values,native_dtype=host_bits(cache.window)
            self.initial_windows.append(np.array(values,copy=True))
            self.initial_window_dtypes.append(native_dtype)
        self.initial_offsets=[int(c.offset) for c in caches]
        if (self.initial_main.shape!=(1,1,self.width)
            or self.initial_offsets!=[16384]*3
            or any(w.shape!=(1,128,512) for w in self.initial_windows)):
            raise RuntimeError('materialized native prefill seed differs')
        print('TEACHER_PREFILL_CAPTURE',json.dumps({'main_dtype':self.initial_main_dtype,'window_dtypes':self.initial_window_dtypes,'offsets':self.initial_offsets}),flush=True)

    def finalize(self, token_ids, *, cycles, prompt_sha256, source_commit):
        if (len(token_ids)!=self.steps+1 or not self.steps<=self.count<=self.steps+5
            or len(self.commit_lengths)!=cycles or self.seed_calls!=cycles+1
            or self.initial_main is None):
            raise RuntimeError('teacher trace is not a complete native D5/M6 run')
        files={}
        def save(name,arr,native_dtype):
            path=self.root/name
            if path.exists(): raise RuntimeError('refusing to overwrite teacher data')
            np.save(path,arr,allow_pickle=False)
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024**2),b''): digest.update(chunk)
            files[name]={'sha256':digest.hexdigest(),'bytes':path.stat().st_size,
                         'shape':list(arr.shape),'dtype':str(arr.dtype),'mlx_dtype':native_dtype}
        save('initial-main.npy',self.initial_main,self.initial_main_dtype)
        for stage,window in enumerate(self.initial_windows): save(f'initial-window-{stage}.npy',window,self.initial_window_dtypes[stage])
        save('committed-hidden.npy',self.rows[:self.count],self.row_mlx_dtype)
        metadata={'schema':'dsv41-exact-target-teacher-v2','source_commit':source_commit,
            'diagnostic_only':True,'performance_eligible':False,
            'prompt_tokens':16384,'prompt_sha256':prompt_sha256,
            'token_ids':list(token_ids),'token_ids_sha256':hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            'target_decode_width':6,'draft_block_size':5,'cycles':cycles,
            'commit_lengths':self.commit_lengths,'captured_hidden_rows':self.count,
            'initial_mtp_offsets':self.initial_offsets,'files':files,
            'source_teacher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'replay_limit':'Target hidden states come from the retained M6 trajectory; draft replay estimates acceptance and cannot prove another target batch shape or full decode throughput.'}
        path=self.root/'teacher.json'
        if path.exists(): raise RuntimeError('refusing to overwrite teacher metadata')
        path.write_text(json.dumps(metadata,indent=2)+'\n')
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'captured_hidden_rows':self.count,'bytes':sum(f['bytes'] for f in files.values()),
                'diagnostic_only':True,'performance_eligible':False}
