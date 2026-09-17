import ast,hashlib,json,subprocess
from pathlib import Path
base='ca207c3d7f923678f5355ca63a0b89a483120dff'
p=Path('mtplx/models/deepseek_v41.py')
def method(s):
 c=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV41Backbone')
 return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_layer_major')
old=method(subprocess.check_output(['git','show',f'{base}:{p}'],text=True))
class Measured(ast.NodeTransformer):
 def visit_Call(self,n):
  self.generic_visit(n)
  if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='layer' and n.func.attr=='moe_combine':
   n.func=ast.Name(id='_PREFILL_HC_POST',ctx=ast.Load())
   n.args[1]=ast.Starred(value=n.args[1],ctx=ast.Load())
  return n
class InactiveHook(ast.NodeTransformer):
 def visit_With(self,n):
  self.generic_visit(n)
  if len(n.items)==1:
   c=n.items[0].context_expr
   if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='_stime' and c.func.attr=='stage' and len(c.args)==1 and isinstance(c.args[0],ast.Constant) and c.args[0].value=='hc.combine':
    assert len(n.body)==2 and isinstance(n.body[1],ast.Expr)
    x=n.body[1].value
    assert isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='_st' and x.func.attr=='add'
    return n.body[:1]
  return n
expected=ast.dump(Measured().visit(old),include_attributes=False)
actual=ast.dump(InactiveHook().visit(method(p.read_text())),include_attributes=False)
assert expected==actual
# Execute the real inactive timing classes/function extracted from their source.
st=Path('mtplx/models/deepseek_v41_stage_timing.py')
tree=ast.parse(st.read_text());names={'_NullFence','_NoopCM','stage'}
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
ns={'_ACTIVE':None};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<real-inactive-timing>','exec'),ns)
ns['_NULL_FENCE']=ns['_NullFence']();ns['_NOOP_CM']=ns['_NoopCM']()
with ns['stage']('hc.combine') as fence:
 assert fence is ns['_NULL_FENCE']
 assert fence.add(object()) is None
assert not hasattr(fence,'__dict__')
report={'measured_source_commit':base,'promoted_model_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'timing_source_sha256':hashlib.sha256(st.read_bytes()).hexdigest(),'inactive_hook_normalized_method_ast_matches_measured':True,'inactive_hook_uses_real_stateless_noop':True,'mlx_imported':False,'numeric_check':'existing strict layer-major/chunk-major test passed under guard before retaining the original timing bracket; final off-path arithmetic is unchanged by the no-op hook','active_hook_check':'actual final combine loop checked by check_dsv41_prefill_stage_hook.py'}
out=Path('docs/deepseek-v41/receipts/hc-post-prefill-20260917/promotion-verification.json')
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
