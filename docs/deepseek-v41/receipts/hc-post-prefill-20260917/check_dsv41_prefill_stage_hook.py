import ast
from pathlib import Path
from types import SimpleNamespace
source=Path('mtplx/models/deepseek_v41.py').read_text()
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV41Backbone')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_layer_major')
loops=[n for n in ast.walk(method) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and len(n.iter.args)==1 and isinstance(n.iter.args[0],ast.Name) and n.iter.args[0].id=='n_chunks']
loop=next(n for n in loops if any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='_PREFILL_HC_POST' for c in ast.walk(n)))
events=[]
class Stage:
 def __enter__(self):events.append('enter');return self
 def add(self,value):events.append(('add',value))
 def __exit__(self,*args):events.append('exit')
def stage(name):assert name=='hc.combine';return Stage()
def post(output,*carry):return (output,*carry)
ns={'n_chunks':2,'moe_outputs':[10,20],'carries':[(1,2,3),(4,5,6)],'hs':[None,None],'_stime':SimpleNamespace(stage=stage),'_PREFILL_HC_POST':post}
exec(compile(ast.Module(body=[loop],type_ignores=[]),'<actual-prefill-combine-loop>','exec'),ns)
assert ns['hs']==[(10,1,2,3),(20,4,5,6)]
assert events==['enter',('add',ns['hs'][0]),'exit','enter',('add',ns['hs'][1]),'exit']
print('Actual combine loop preserves both timing brackets and output registration; no MLX imported.')
